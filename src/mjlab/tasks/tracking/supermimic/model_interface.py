"""TrackingModelInterface: the motion tracking task as seen by the SuperMimic learner.

Implements :class:`mjlab.rl.supermimic.ModelInterface` for the SuperMimic tracking env
(``unitree_g1_flat_tracking_supermimic_env_cfg``). Every constant (reward weights and
stds, termination thresholds, observation noise, default joint positions, soft limits,
domain-randomization ranges, bodies, anchor) is read from the live env and its
managers; the heads call the same kernels as the env terms (``mdp.kernels``) on the
differentiable forward kinematics of the state (``mdp.kinematics``).

Timing differences to the env (documented, measured in
``tests/test_supermimic_interface.py``): the env scores a step on body kinematics one
physics substep stale and builds the relative body targets from the anchor of the
previous state; ``reward`` and ``termination_hard`` use the reached state s_next only.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import torch

from mjlab.rl.supermimic.interface import FloatingBaseState
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import kernels
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.kinematics import TorchKinematics
from mjlab.tasks.tracking.supermimic.recorder import (
  TerminalStateRecorder,
  floating_base_state,
)
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise.noise_cfg import NoiseCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_COMMAND = "motion"
_ACTOR = "actor"
_GOAL = "goal"
# Actor terms the observation head reproduces, in env order (Table A.1).
_ACTOR_FUNCS = (
  mdp.joint_pos_rel,
  mdp.joint_vel_rel,
  mdp.builtin_sensor,
  mdp.projected_gravity,
  mdp.last_action,
)
_TRACKING_REWARDS = (
  mdp.motion_global_anchor_position_error_exp,
  mdp.motion_global_anchor_orientation_error_exp,
  mdp.motion_relative_body_position_error_exp,
  mdp.motion_relative_body_orientation_error_exp,
  mdp.motion_global_body_linear_velocity_error_exp,
  mdp.motion_global_body_angular_velocity_error_exp,
)


class TrackingModelInterface:
  """SuperMimic model interface of the motion tracking task.

  Shapes: N envs, B any flat batch, nq = 29 joints on the G1. States follow
  :class:`FloatingBaseState` (root position env-local). Frames are global indices of
  the env's motion library (``MotionCommand``).
  """

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self._env = env
    command = cast(MotionCommand, env.command_manager.get_term(_COMMAND))
    self._command = command
    # The learner's CUDA graphs read the motion library. Static, they read it in place;
    # otherwise every replay copies the whole library into the graph's inputs.
    for value in vars(command.motion).values():
      if isinstance(value, torch.Tensor):
        torch._dynamo.mark_static_address(value)
    robot: Entity = env.scene[command.cfg.entity_name]
    self._robot = robot
    self.device = env.device

    self.nq = robot.num_joints
    self._state = FloatingBaseState(self.nq)
    self.state_dim = self._state.state_dim
    self.feature_dim = self._state.feature_dim
    self.action_dim = env.action_manager.total_action_dim
    self.context_dim = 1 + self.nq + 3

    prefix = f"{command.cfg.entity_name}/"
    self._kinematics = TorchKinematics(
      env.sim.mj_model,
      [prefix + name for name in command.cfg.body_names],
      device=self.device,
    )
    kinematic_joints = tuple(
      n.removeprefix(prefix) for n in self._kinematics.joint_names
    )
    if kinematic_joints != robot.joint_names:
      raise ValueError("Kinematics joint order differs from the robot's joint order.")
    self._anchor = command.motion_anchor_body_index

    self._init_observation()
    self._init_goal()
    self._init_reward()
    self._init_termination()
    self._context = self._read_context()

    recorders = [
      env.recorder_manager.get_term(name) for name in env.recorder_manager.active_terms
    ]
    found = [r for r in recorders if isinstance(r, TerminalStateRecorder)]
    if len(found) != 1:
      raise ValueError("The env cfg needs exactly one TerminalStateRecorder.")
    self._recorder = found[0]

  # Setup (everything read from the env's managers).

  def _init_observation(self) -> None:
    obs = self._env.observation_manager
    names = obs.active_terms[_ACTOR]
    cfgs = [obs.get_term_cfg(_ACTOR, name) for name in names]
    if tuple(cfg.func for cfg in cfgs) != _ACTOR_FUNCS:
      raise ValueError(
        f"Actor terms {names} are not [joint_pos_rel, joint_vel_rel, builtin_sensor "
        "(gyro), projected_gravity, last_action]."
      )
    if not cfgs[0].params.get("biased", False):
      raise ValueError("The joint_pos actor term must use biased=True.")
    # Noise is None when the group disables corruption (play), as in the env.
    self._noise: list[NoiseCfg | None] = []
    for name, cfg in zip(names, cfgs, strict=True):
      if cfg.clip is not None or cfg.scale is not None or cfg.delay_max_lag > 0:
        raise ValueError(f"Actor term {name!r}: clip, scale and delay are unsupported.")
      noise = cfg.noise
      if noise is None or (isinstance(noise, NoiseCfg) and noise.operation == "add"):
        self._noise.append(noise)
      else:
        raise ValueError(f"Actor term {name!r}: only additive NoiseCfg is supported.")
    self._check_gyro(cfgs[2].params["sensor_name"])
    term_dims = obs.group_obs_term_dim[_ACTOR]
    self.obs_term_dims = tuple(
      math.prod(dims) // max(cfg.history_length, 1)
      for dims, cfg in zip(term_dims, cfgs, strict=True)
    )
    self.obs_dim = sum(self.obs_term_dims)
    expected = (self.nq, self.nq, 3, 3, self.action_dim)
    if self.obs_term_dims != expected:
      raise ValueError(f"Actor term dims {self.obs_term_dims} != {expected}.")
    if torch.any(self._robot.data.default_joint_vel != 0.0):
      raise ValueError("joint_vel_rel with nonzero default joint velocities.")

  def _check_gyro(self, sensor_name: str) -> None:
    """The gyro must sit on the root body with the root's orientation, so that it
    reads the free joint's angular velocity (qvel[3:6])."""
    model = self._env.sim.mj_model
    sensor = model.sensor(sensor_name)
    site = int(sensor.objid[0])
    if (
      int(sensor.type[0]) != mujoco.mjtSensor.mjSENS_GYRO
      or int(sensor.objtype[0]) != mujoco.mjtObj.mjOBJ_SITE
      or int(model.site_bodyid[site]) != self._robot.indexing.root_body_id
      or not np.allclose(model.site_quat[site], (1.0, 0.0, 0.0, 0.0))
    ):
      raise ValueError(f"{sensor_name!r} is not a gyro aligned with the root body.")

  def _init_goal(self) -> None:
    obs = self._env.observation_manager
    names = obs.active_terms[_GOAL]
    cfgs = [obs.get_term_cfg(_GOAL, name) for name in names]
    if [cfg.func for cfg in cfgs] != [mdp.motion_future_frames_b]:
      raise ValueError("The goal group must hold only motion_future_frames_b.")
    self._future_frames: int = cfgs[0].params["future_frames"]
    self.goal_dim = self._future_frames * (9 + 2 * self.nq)
    if obs.group_obs_dim[_GOAL] != (self.goal_dim,):
      raise ValueError(f"Goal dim {obs.group_obs_dim[_GOAL]} != ({self.goal_dim},).")

  def _init_reward(self) -> None:
    env = self._env
    manager = env.reward_manager
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    self._reward_terms: list[tuple[str, object, float, dict]] = []
    for name in manager.active_terms:
      cfg = manager.get_term_cfg(name)
      known = (*_TRACKING_REWARDS, mdp.action_rate_l2, mdp.joint_pos_limits)
      if cfg.func not in known:
        raise ValueError(f"Reward term {name!r} has no analytic head.")
      if cfg.func in _TRACKING_REWARDS and cfg.params.get("body_names") is not None:
        raise ValueError(f"Reward term {name!r}: body_names subsets unsupported.")
      if cfg.func is mdp.action_rate_l2 and cfg.params.get("action_name") is not None:
        raise ValueError(f"Reward term {name!r}: action_name unsupported.")
      self._reward_terms.append((name, cfg.func, cfg.weight * scale, cfg.params))
    self.max_step_reward = sum(w for _, _, w, _ in self._reward_terms if w > 0)

    limits = self._robot.data.soft_joint_pos_limits
    if not torch.equal(limits, limits[:1].expand_as(limits)):
      raise ValueError("Soft joint limits differ across envs.")
    self._soft_limits = limits[0]  # (nq, 2)

  def _init_termination(self) -> None:
    manager = self._env.termination_manager
    thresholds: dict[object, float] = {}
    for name in manager.active_terms:
      cfg = manager.get_term_cfg(name)
      if cfg.time_out:
        continue
      if cfg.func not in (mdp.bad_anchor_pos_z_only, mdp.bad_anchor_ori):
        raise ValueError(f"Termination term {name!r} has no analytic rule.")
      thresholds[cfg.func] = cfg.params["threshold"]
    if len(thresholds) != 2:
      raise ValueError("Need both bad_anchor_pos_z_only and bad_anchor_ori.")
    self._z_threshold = thresholds[mdp.bad_anchor_pos_z_only]
    self._tilt_threshold = thresholds[mdp.bad_anchor_ori]

  def _read_context(self) -> torch.Tensor | None:
    """Normalised startup-DR context (N, 33), or None if the cfg has no such DR.

    xi = [(mu - c) / h, encoder bias / h (nq), torso CoM offset / h (3)], with c, h
    the centre and half-width of each DR range of the env cfg (paper: (mu - 0.75) /
    0.45, bias / 0.01, offset / (0.025, 0.05, 0.05)). The play cfg has no
    randomization (fixed friction, no bias, no CoM offset): no context then.
    """
    env = self._env
    events = env.event_manager
    startup = events.active_terms.get("startup", [])
    if not {"foot_friction", "encoder_bias", "base_com"} <= set(startup):
      return None
    indexing = self._robot.indexing

    friction_cfg = events.get_term_cfg("foot_friction")
    if friction_cfg.params["operation"] != "abs":
      raise ValueError("foot_friction must set an absolute friction.")
    low, high = friction_cfg.params["ranges"]
    if high <= low:
      return None
    geom = int(indexing.geom_ids[friction_cfg.params["asset_cfg"].geom_ids][0])
    # shared_random: every foot geom carries the same sampled friction.
    friction = env.sim.model.geom_friction[:, geom, 0]
    friction = (friction - 0.5 * (low + high)) / (0.5 * (high - low))

    bias_low, bias_high = events.get_term_cfg("encoder_bias").params["bias_range"]
    bias = self._robot.data.encoder_bias
    bias = (bias - 0.5 * (bias_low + bias_high)) / (0.5 * (bias_high - bias_low))

    com_cfg = events.get_term_cfg("base_com")
    if com_cfg.params["operation"] != "add":
      raise ValueError("base_com must add an offset to the default body_ipos.")
    bodies = indexing.body_ids[com_cfg.params["asset_cfg"].body_ids]
    if bodies.numel() != 1:
      raise ValueError("base_com must randomize exactly one body (the torso).")
    body = int(bodies[0])
    default_ipos = env.sim.get_default_field("body_ipos")[body]
    offset = env.sim.model.body_ipos[:, body] - default_ipos
    ranges = torch.tensor(
      [com_cfg.params["ranges"][axis] for axis in range(3)], device=self.device
    )
    offset = (offset - ranges.mean(-1)) / (0.5 * (ranges[:, 1] - ranges[:, 0]))
    return torch.cat([friction.unsqueeze(-1), bias, offset], dim=-1).float()

  # Sim access.

  def state_from_sim(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Current state (N, 71) (or (len(env_ids), 71)), read from qpos / qvel."""
    state = floating_base_state(self._env, self._command.cfg.entity_name)
    return state if env_ids is None else state[env_ids]

  def terminal_state_snapshot(self) -> torch.Tensor | None:
    """Pre-reset state (N, 71); rows valid for envs done in the last step."""
    return self._recorder.terminal_state_snapshot()

  def frame_from_sim(self) -> torch.Tensor:
    """Frame k_t (N,) int64 the current observation targets (a copy)."""
    return self._command.time_steps.clone()

  def context_normalized(self) -> torch.Tensor:
    """Normalised sim context (N, 33), fixed per env (see ``_read_context``)."""
    if self._context is None:
      raise RuntimeError(
        "No sim context: the env cfg lacks the startup DR events foot_friction, "
        "encoder_bias and base_com (e.g. the play cfg)."
      )
    return self._context

  # Heads on (imagined) states.

  def invariant_features(self, s: torch.Tensor) -> torch.Tensor:
    """(grad) Features (B, 68): root height, gravity, root-frame velocities, q, qd."""
    return self._state.invariant_features(s)

  def observation(
    self, s: torch.Tensor, a_prev: torch.Tensor, env_idx: torch.Tensor
  ) -> torch.Tensor:
    """(grad) Noise-free actor frame (B, 93) of state s (B, 71).

    [q + encoder bias - default q, qd, gyro (root-frame angular velocity), projected
    gravity, a_prev]; ``env_idx`` (B,) selects the encoder bias and default q.
    """
    _, quat, _, ang_vel_w, q, qd = self._state.split(s)
    data = self._robot.data
    return kernels.proprio_observation(
      q,
      qd,
      quat,
      quat_apply_inverse(quat, ang_vel_w),
      data.encoder_bias[env_idx],
      data.default_joint_pos[env_idx],
      a_prev,
    )

  def add_observation_noise(self, o: torch.Tensor) -> torch.Tensor:
    """Observation (B, 93) plus fresh additive noise of each actor term's NoiseCfg
    (none when the env disables corruption); the gradient of ``o`` is kept."""
    parts = torch.split(o, list(self.obs_term_dims), dim=-1)
    noisy = [
      part if noise is None else noise.apply(part)
      for part, noise in zip(parts, self._noise, strict=True)
    ]
    return torch.cat(noisy, dim=-1)

  def _anchor_pose(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat, _, _, q, _ = self._state.split(s)
    body_pos, body_quat = self._kinematics.forward(pos, quat, q)
    return body_pos[:, self._anchor], body_quat[:, self._anchor]

  def goal(self, s: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """(grad) Goal (B, 670) of state s (B, 71) at frame (B,): the reference frames
    frame .. frame + K - 1 (clamped to the clip) seen from the robot anchor, as
    ``mdp.motion_future_frames_b``."""
    anchor_pos, anchor_quat = self._anchor_pose(s)
    frames = self._command.future_frames(frame, self._future_frames)
    motion = self._command.motion
    return kernels.future_frames_goal(
      motion.body_pos_w[frames, self._anchor],
      motion.body_quat_w[frames, self._anchor].float(),
      motion.joint_pos[frames].float(),
      motion.joint_vel[frames].float(),
      anchor_pos,
      anchor_quat,
    )

  def reward_terms(
    self,
    s_next: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    frame: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """(grad) Per-term rewards (B,), keyed by the env's reward term names, weighted
    and multiplied by dt (they sum to :meth:`reward`).

    Tracking terms compare FK of s_next (B, 71) with reference frame ``frame`` (B,);
    the relative body targets are re-anchored on s_next's own anchor.
    """
    pos, quat, lin_vel, ang_vel, q, qd = self._state.split(s_next)
    body_pos, body_quat, body_lin_vel, body_ang_vel = (
      self._kinematics.forward_with_velocity(pos, quat, lin_vel, ang_vel, q, qd)
    )
    motion = self._command.motion
    ref_pos = motion.body_pos_w[frame]
    ref_quat = motion.body_quat_w[frame].float()
    i = self._anchor
    pos_delta, quat_delta = kernels.anchor_frame(
      body_pos[:, i], body_quat[:, i], ref_pos[:, i], ref_quat[:, i]
    )
    target_pos, target_quat = kernels.relative_body_targets(
      ref_pos, ref_quat, ref_pos[:, i], pos_delta, quat_delta
    )
    terms = {}
    for name, func, weight, params in self._reward_terms:
      if func is mdp.motion_global_anchor_position_error_exp:
        value = kernels.root_pos_reward(body_pos[:, i], ref_pos[:, i], params["std"])
      elif func is mdp.motion_global_anchor_orientation_error_exp:
        value = kernels.root_ori_reward(body_quat[:, i], ref_quat[:, i], params["std"])
      elif func is mdp.motion_relative_body_position_error_exp:
        value = kernels.body_pos_reward(body_pos, target_pos, params["std"])
      elif func is mdp.motion_relative_body_orientation_error_exp:
        value = kernels.body_ori_reward(body_quat, target_quat, params["std"])
      elif func is mdp.motion_global_body_linear_velocity_error_exp:
        value = kernels.body_lin_vel_reward(
          body_lin_vel, motion.body_lin_vel_w[frame].float(), params["std"]
        )
      elif func is mdp.motion_global_body_angular_velocity_error_exp:
        value = kernels.body_ang_vel_reward(
          body_ang_vel, motion.body_ang_vel_w[frame].float(), params["std"]
        )
      elif func is mdp.action_rate_l2:
        value = kernels.action_rate_l2(a, a_prev)
      else:
        joints = params["asset_cfg"].joint_ids
        value = kernels.joint_limit_penalty(q[:, joints], self._soft_limits[joints])
      terms[name] = weight * value
    return terms

  def reward(
    self,
    s_next: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    frame: torch.Tensor,
  ) -> torch.Tensor:
    """(grad) Reward (B,) of reaching s_next (B, 71) with action a (B, 29) after
    a_prev (B, 29), scored against frame (B,); times dt like the env reward."""
    return torch.stack(list(self.reward_terms(s_next, a, a_prev, frame).values())).sum(
      0
    )

  def termination_hard(self, s_next: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Anchor height / tilt failure (B,) bool of s_next (B, 71) against frame (B,),
    with the env cfg's thresholds."""
    anchor_pos, anchor_quat = self._anchor_pose(s_next)
    motion = self._command.motion
    bad_z, bad_tilt, _, _ = kernels.anchor_termination(
      anchor_pos,
      anchor_quat,
      motion.body_pos_w[frame, self._anchor],
      motion.body_quat_w[frame, self._anchor].float(),
      z_thresh=self._z_threshold,
      tilt_thresh=self._tilt_threshold,
    )
    return bad_z | bad_tilt

  def termination_soft(
    self,
    s_next: torch.Tensor,
    frame: torch.Tensor,
    z_temp: float = 0.02,
    tilt_temp: float = 0.04,
  ) -> torch.Tensor:
    """(grad) 1 - (1 - sigmoid((e_z - z_th) / z_temp)) (1 - sigmoid((e_tilt - tilt_th)
    / tilt_temp)), (B,): the anchor errors of :meth:`termination_hard` for s_next
    (B, 71) against frame (B,), thresholds from the env cfg (supermimic_vibed's
    temperatures, mdp.py:155-162)."""
    anchor_pos, anchor_quat = self._anchor_pose(s_next)
    motion = self._command.motion
    _, _, e_z, e_tilt = kernels.anchor_termination(
      anchor_pos,
      anchor_quat,
      motion.body_pos_w[frame, self._anchor],
      motion.body_quat_w[frame, self._anchor].float(),
      z_thresh=self._z_threshold,
      tilt_thresh=self._tilt_threshold,
    )
    alive = (1 - torch.sigmoid((e_z - self._z_threshold) / z_temp)) * (
      1 - torch.sigmoid((e_tilt - self._tilt_threshold) / tilt_temp)
    )
    return 1 - alive

  # Reset sampler (forwarded to the MotionCommand).

  def clip_end(self, frame: torch.Tensor) -> torch.Tensor:
    """Exclusive end frame (B,) of the clip of each frame (B,)."""
    return self._command.clip_end_of(frame)

  def bin_of(self, frame: torch.Tensor) -> torch.Tensor:
    """Reset-sampler bin (B,) of each frame (B,)."""
    return self._command.bin_of(frame)

  def update_bin_returns(self, bins: torch.Tensor, returns: torch.Tensor) -> None:
    """Feed discounted returns (M,) observed from bins (M,) to the sampler."""
    self._command.update_bin_returns(bins, returns)

  def update_past_weights(self) -> None:
    """Advance the sampler's past-weight EMA (once per iteration)."""
    self._command.update_past_weights()

  def sampler_state_dict(self) -> dict[str, torch.Tensor]:
    """Bin returns G_b and past weights (Nb,) each."""
    return {
      "bin_returns": self._command.bin_returns.clone(),
      "bin_past_weights": self._command.bin_past_weights.clone(),
    }

  def load_sampler_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Restore the sampler state; the library must have the same bins."""
    self._command.bin_returns.copy_(state["bin_returns"])
    self._command.bin_past_weights.copy_(state["bin_past_weights"])

  def sampler_log(self) -> dict[str, float]:
    """Normalised entropy of the reset distribution and mean bin return."""
    probs = self._command.bin_sampling_probabilities()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    count = probs.numel()
    return {
      "reset_entropy": float(entropy) / math.log(count) if count > 1 else 1.0,
      "mean_bin_return": float(self._command.bin_returns.mean()),
    }
