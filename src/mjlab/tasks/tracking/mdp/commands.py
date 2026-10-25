from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.rl.supermimic.return_sampler import ReturnSampler
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  normalize,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from . import kernels

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))
# MotionLoader storage per npz key.
_STORAGE_DTYPES = {
  "joint_pos": np.float16,
  "joint_vel": np.float16,
  "body_pos_w": np.float32,
  "body_quat_w": np.float16,
  "body_lin_vel_w": np.float16,
  "body_ang_vel_w": np.float16,
}


def expand_motion_files(motion_file: str | Sequence[str]) -> list[str]:
  """The npz paths of ``motion_file``: npz paths, where an entry ending in ``.txt``
  stands for the npz paths it lists, one per line (libraries too large for argv)."""
  entries = [motion_file] if isinstance(motion_file, str) else list(motion_file)
  files = []
  for entry in entries:
    if entry.endswith(".txt"):
      files += [line for line in Path(entry).read_text().splitlines() if line]
    else:
      files.append(entry)
  return files


class MotionLoader:
  """One or several 50 Hz reference clips, concatenated along time.

  Frame indices are global over the concatenation. Clip ``c`` covers frames
  ``[clip_start[c], clip_end[c])``; ``clip_id_of_frame`` (F,) maps a frame to its clip.

  Only the tracked bodies (``body_indexes``) are kept. To fit large libraries on the
  GPU, everything but the body positions is stored in float16 (positions stay float32:
  float16 spacing is 8 mm at 8-16 m from the origin); readers cast the rows they
  gather to float32 (``.float()``).
  """

  def __init__(
    self,
    motion_file: str | Sequence[str],
    body_indexes: torch.Tensor,
    device: str = "cpu",
  ) -> None:
    files = expand_motion_files(motion_file)
    if not files:
      raise ValueError("MotionLoader needs at least one motion file.")
    bodies = body_indexes.cpu().numpy()
    parts: dict[str, list[np.ndarray]] = {key: [] for key in _STORAGE_DTYPES}
    lengths = []
    for file in files:
      with np.load(file) as clip:
        # One reference frame per 50 Hz env step; fps is otherwise not used.
        if "fps" in clip and float(clip["fps"].reshape(-1)[0]) != 50.0:
          raise ValueError(f"{file}: motions must be 50 Hz, got fps={clip['fps']}.")
        lengths.append(clip["joint_pos"].shape[0])
        for key, dtype in _STORAGE_DTYPES.items():
          data = clip[key][:, bodies] if key.startswith("body_") else clip[key]
          parts[key].append(data.astype(dtype, copy=False))
    if min(lengths) < 2:
      raise ValueError("Every motion clip needs at least 2 frames.")

    def load(key: str) -> torch.Tensor:
      return torch.from_numpy(np.concatenate(parts.pop(key))).to(device)

    self.joint_pos = load("joint_pos")
    self.joint_vel = load("joint_vel")
    self.body_pos_w = load("body_pos_w")
    self.body_quat_w = load("body_quat_w")
    self.body_lin_vel_w = load("body_lin_vel_w")
    self.body_ang_vel_w = load("body_ang_vel_w")
    self.time_step_total = self.joint_pos.shape[0]

    self.clip_names = tuple(Path(file).stem for file in files)
    clip_length = torch.tensor(lengths, dtype=torch.long, device=device)
    self.clip_end = torch.cumsum(clip_length, dim=0)
    self.clip_start = self.clip_end - clip_length
    self.clip_id_of_frame = torch.repeat_interleave(
      torch.arange(len(files), device=device), clip_length
    )


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # sampling_mode "start": env i plays clip start_clip[i] (evaluation sets it).
    self.start_clip = torch.arange(self.num_envs, device=self.device) % len(
      self.motion.clip_names
    )
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Bins of at most 1 s, per clip: clip c gets n_c = len_c // fps + 1 equal-width
    # bins (the single-clip rule mjlab always used), so no bin spans two clips.
    clip_length = self.motion.clip_end - self.motion.clip_start
    self._clip_bin_count = torch.tensor(
      [int(n // (1 / env.step_dt)) + 1 for n in clip_length.tolist()],
      device=self.device,
    )
    self._clip_first_bin = torch.cumsum(self._clip_bin_count, 0) - self._clip_bin_count
    self.bin_count = int(self._clip_bin_count.sum())
    clip_id = self.motion.clip_id_of_frame
    local_frame = (
      torch.arange(self.motion.time_step_total, device=self.device)
      - self.motion.clip_start[clip_id]
    )
    self.bin_of_frame = (
      self._clip_first_bin[clip_id]
      + local_frame * self._clip_bin_count[clip_id] // clip_length[clip_id]
    )
    bin_size = torch.bincount(self.bin_of_frame, minlength=self.bin_count)
    self.bin_end = torch.cumsum(bin_size, 0)
    self.bin_start = self.bin_end - bin_size
    # sampling_mode "return"; the SuperMimic runner feeds it the episode returns.
    self.return_sampler = ReturnSampler(
      self.bin_of_frame,
      self.bin_start,
      self.bin_end,
      self.clip_end_of(self.bin_start),
      g_max=cfg.g_max,
      return_alpha=cfg.bin_return_alpha,
      past_weight_alpha=cfg.past_weight_alpha,
      softmax_cap=cfg.softmax_cap,
    )

    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps].float()

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps].float()

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps].float()

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps].float()

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps].float()

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[
      self.time_steps, self.motion_anchor_body_index
    ].float()

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[
      self.time_steps, self.motion_anchor_body_index
    ].float()

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[
      self.time_steps, self.motion_anchor_body_index
    ].float()

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def clip_end_of(self, frames: torch.Tensor) -> torch.Tensor:
    """Exclusive end frame of the clip each frame belongs to: (B,) -> (B,)."""
    return self.motion.clip_end[self.motion.clip_id_of_frame[frames]]

  def bin_of(self, frames: torch.Tensor) -> torch.Tensor:
    """Sampling bin of each frame: (B,) -> (B,)."""
    return self.bin_of_frame[frames]

  def future_frames(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Frames ``frames .. frames + num_frames - 1``, clamped to the clip's last frame.

    (B,) -> (B, num_frames).
    """
    offsets = torch.arange(num_frames, device=frames.device)
    last = self.clip_end_of(frames) - 1
    return torch.minimum(frames.unsqueeze(-1) + offsets, last.unsqueeze(-1))

  def current_frames(self, frames: torch.Tensor) -> torch.Tensor:
    """Frame the state shown with ``frames`` is aligned with: ``frames - 1``, not
    before the clip start (a reset writes the state of frame k and shows k + 1; the
    next reward is scored against k + 1). (B,) -> (B,)."""
    start = self.motion.clip_start[self.motion.clip_id_of_frame[frames]]
    return torch.maximum(frames - 1, start)

  def reference_goal(self, frames: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Goal of ``kernels.reference_heading_goal`` at ``frames`` (B,): the anchor of
    frames ``frames .. frames + num_frames - 1`` (``future_frames``) in the heading
    frame of ``current_frames(frames)``. (B, num_frames * (15 + 2 J))."""
    current = self.current_frames(frames)
    future = self.future_frames(frames, num_frames)
    anchor = self.motion_anchor_body_index
    motion = self.motion
    # The library's float16 quaternions are unit only to ~1e-3; the heading (yaw_quat)
    # and the body-frame velocities (quat_apply_inverse) need unit quaternions.
    return kernels.reference_heading_goal(
      motion.body_pos_w[current, anchor],
      normalize(motion.body_quat_w[current, anchor].float()),
      motion.body_pos_w[future, anchor],
      normalize(motion.body_quat_w[future, anchor].float()),
      motion.body_lin_vel_w[future, anchor].float(),
      motion.body_ang_vel_w[future, anchor].float(),
      motion.joint_pos[future].float(),
      motion.joint_vel[future].float(),
    )

  def assign_clips_from_start(self, env_ids: torch.Tensor) -> None:
    """Point env ``i`` at the first frame of clip ``start_clip[i]`` (default ``i mod
    C``; play, evaluation)."""
    self.time_steps[env_ids] = self.motion.clip_start[self.start_clip[env_ids]]

  def _return_sampling(self, env_ids: torch.Tensor):
    frames, probabilities = self.return_sampler.sample_frames(len(env_ids))
    self.time_steps[env_ids] = frames
    self._update_sampling_metrics(probabilities)

  def _update_sampling_metrics(self, sampling_probabilities: torch.Tensor) -> None:
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      fail_bins = self.bin_of_frame[self.time_steps[env_ids][episode_failed]]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    clip_id = self.motion.clip_id_of_frame[self.bin_start[sampled_bins]]
    clip_start = self.motion.clip_start[clip_id]
    clip_length = self.motion.clip_end[clip_id] - clip_start
    self.time_steps[env_ids] = (
      clip_start
      + (
        (
          sampled_bins
          - self._clip_first_bin[clip_id]
          + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        )
        / self._clip_bin_count[clip_id]
        * (clip_length - 1)
      ).long()
    )

    self._update_sampling_metrics(sampling_probabilities)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # The library's float16 quaternions are unit only to ~1e-3; qpos needs unit norm.
    root_ori = root_ori / torch.linalg.norm(root_ori, dim=-1, keepdim=True)
    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.assign_clips_from_start(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    elif self.cfg.sampling_mode == "return":
      self._return_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w = self.anchor_pos_w
    pos_delta, quat_delta = kernels.anchor_frame(
      self.robot_anchor_pos_w,
      self.robot_anchor_quat_w,
      anchor_pos_w,
      self.anchor_quat_w,
    )
    self.body_pos_relative_w, self.body_quat_relative_w = kernels.relative_body_targets(
      self.body_pos_w, self.body_quat_w, anchor_pos_w, pos_delta, quat_delta
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (wraparound or timer expiry) need our refresh.
    self._pending_forward = False
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None):
    clip_end = self.clip_end_of(self.time_steps)
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    wrap_ids = torch.where(self.time_steps >= clip_end)[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    # _resample_command writes qpos/qvel but does not refresh derived
    # quantities; forward() so update_relative_body_poses reads the
    # post-teleport robot anchor instead of the stale pre-resample pose.
    # Covers both the wraparound above and a timer-expiry resample that ran
    # in compute before this call.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    # Fold failure counts into the EMA only on the per-step update so
    # manual resets do not decay it faster.
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        self.time_steps[idx] = int(scrubber.value)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    # reset_to_frame writes qpos/qvel; forward so update_relative_body_poses
    # reads the scrubbed pose instead of the stale pre-scrub kinematics.
    self._env.sim.forward()
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    self.time_steps[env_ids] = frame
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  # tuple first: with mjlab's tyro flags a single path still parses as a str and
  # "('a.npz', 'b.npz')" as a tuple (the other order parses both as a str).
  motion_file: tuple[str, ...] | str
  """One npz path, or several concatenated into a motion library; a ``.txt`` path
  lists npz paths, one per line (see ``expand_motion_files``)."""
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start", "return"] = "adaptive"
  g_max: float = 10.0
  """Return sampling: largest attainable discounted return, 0.1 / (1 - 0.99)."""
  bin_return_alpha: float = 0.1
  """Return sampling: EMA factor of the per-bin return estimates."""
  past_weight_alpha: float = 1e-4
  """Return sampling: EMA factor of the past softmax weights, per iteration."""
  softmax_cap: float = 50.0
  """Return sampling: largest softmax weight of a bin, as a multiple of the uniform
  share 1 / Nb (allmimic's value); the excess goes to the other bins."""
  termination_z_temperature: float = 0.02
  """SuperMimic imagination: sigmoid temperature (m) of the soft anchor-height
  termination (``TrackingModelInterface.termination_soft``; supermimic_vibed's
  value, mdp.py:155-162)."""
  termination_tilt_temperature: float = 0.04
  """SuperMimic imagination: sigmoid temperature of the soft anchor-tilt
  termination (supermimic_vibed's value)."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
