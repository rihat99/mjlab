#!/usr/bin/env python3
"""Render "action ghost" videos of trained mjlab trackers: the simulated G1 plus a ghost
G1 at the policy's joint targets.

Per clip, the clip's newest tracker run (final checkpoint) plays the clip once from
frame 0 in mjlab's play config (1 env, deterministic mean action, as
``train_tracker.py --play``). After every policy step the frame shows

- the robot, opaque, at the end of the step (after the 4 physics substeps);
- an orange ghost G1 whose free joint is the robot's root and whose 29 joints are the
  position targets the actuators tracked during that step, read back from ``data.ctrl``
  (= raw action * scale + default joint pos - encoder bias; MuJoCo <position> actuators
  with ctrllimited=false, so targets beyond the joint range are not clipped; only the
  torque is clamped to the effort limit).

The gap is not a tracking error: for these PD actuators
target - q = torque / kp + (kv / kp) * qdot, so the ghost leads a moving joint by
kv / kp seconds (0.064 s for the G1) and shows torque as a large angle on low-kp joints
(waist, wrists).

The step that wraps the motion back to frame 0 is not played, so a clip of N frames
gives N - 2 frames (t = 0.02 s .. (N - 2) * 0.02 s). Outputs per clip in
``<out-dir>/<clip>/native_action_ghost/``: ``realtime.mp4`` (50 fps), ``slow025.mp4``
(12.5 fps), ``gap.json`` (per-joint |target - actual| statistics) and a 12-frame contact
sheet ``sheet.png``. Without ``--clip`` it ranks all clips by mean reference joint speed
(``<out-dir>/action_ghost_ranking.json``) and renders the ``--top`` fastest:

    PY=.venv/bin/python  # from the repo root
    $PY tools/render_action_ghost.py --gpu 0 [--top 5]
    $PY tools/render_action_ghost.py --gpu 0 --clip amass_cmu_88_05 amass_cmu_88_06
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from paths import MOTIONS_DIR, RUNS_DIR, VIDEOS_DIR
from PIL import Image, ImageDraw, ImageFont
from train_tracker import TASK, ParamsLoader, resolve_checkpoint, run_motion_file

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from rsl_rl.models import MLPModel

  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.rl import RslRlVecEnvWrapper

GHOST_RGBA = (1.0, 0.45, 0.0, 0.6)
SLOW_FACTOR = 0.25
SHEET_LARGEST, SHEET_EVEN = 4, 8  # contact sheet: largest-gap moments + evenly spaced


def rank_clips(motions_dir: Path) -> list[dict]:
  """All clips, fastest first, by mean over frames of the RMS of the reference
  ``joint_vel`` over the 29 joints."""
  rows = []
  for path in sorted(motions_dir.glob("*.npz")):
    motion = np.load(path)
    rows.append(
      {
        "clip": path.stem,
        "frames": len(motion["joint_vel"]),
        "mean_joint_speed_rms": float(
          np.sqrt((motion["joint_vel"] ** 2).mean(axis=1)).mean()
        ),
        # Body 0 of the motion npz is the robot's first body, the pelvis.
        "mean_pelvis_speed": float(
          np.linalg.norm(motion["body_lin_vel_w"][:, 0], axis=1).mean()
        ),
      }
    )
  return sorted(rows, key=lambda row: -row["mean_joint_speed_rms"])


def make_play_env(
  checkpoint: Path,
  motion: Path,
  edit_cfg: Callable[[ManagerBasedRlEnvCfg], None] | None = None,
  task: str = TASK,
) -> tuple[RslRlVecEnvWrapper, MLPModel]:
  """mjlab play env (1 env, starts at clip frame 0) of ``task`` and the checkpoint's
  deterministic policy, built as play does; ``edit_cfg`` may change the play env config
  before the env is built."""
  import mjlab.tasks  # noqa: F401 -- registers the tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp import MotionCommandCfg
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  motion_cmd = env_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion)
  env_cfg.scene.num_envs = 1
  if edit_cfg is not None:
    edit_cfg(env_cfg)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0"),
    clip_actions=agent_cfg.clip_actions,
  )
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    env, asdict(agent_cfg), device="cuda:0"
  )
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  return env, runner.get_inference_policy(device="cuda:0")


class ActionGhostRenderer:
  """Offscreen frames of env 0: the robot opaque plus a ghost robot at a given qpos
  (mjlab's reference-ghost path)."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    width: int,
    height: int,
    rgba: tuple[float, ...] = GHOST_RGBA,
  ) -> None:
    import mujoco

    from mjlab.viewer.offscreen_renderer import OffscreenRenderer

    # The play videos' tracking camera, at a larger frame size.
    viewer_cfg = copy.deepcopy(env.cfg.viewer)
    viewer_cfg.width, viewer_cfg.height = width, height
    self._env = env
    self._renderer = OffscreenRenderer(
      model=env.sim.mj_model,
      cfg=viewer_cfg,
      scene=env.scene,
      sim_model=env.sim.model,
      expanded_fields=env.sim.expanded_fields,
    )
    self._renderer.initialize()
    # Same ghost model as MotionCommand's reference ghost: collision geoms hidden
    # (alpha 0), visual geoms recolored.
    self._ghost_model = copy.deepcopy(env.sim.mj_model)
    collision = (self._ghost_model.geom_contype != 0) | (
      self._ghost_model.geom_conaffinity != 0
    )
    self._ghost_model.geom_rgba[collision, 3] = 0.0
    self._ghost_model.geom_rgba[~collision] = rgba

    robot = env.scene["robot"]
    model = env.sim.mj_model
    self.free_q_adr = robot.indexing.free_joint_q_adr.cpu().numpy()  # (7,)
    # (29,), robot joint order.
    self.joint_q_adr = robot.indexing.joint_q_adr.cpu().numpy()
    # qpos address of the joint each actuator (ctrl index) drives.
    self.ctrl_q_adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]  # (nu,)
    if sorted(self.ctrl_q_adr.tolist()) != sorted(self.joint_q_adr.tolist()):
      raise RuntimeError("expected exactly one position actuator per robot joint")
    # data.ctrl is a joint angle only for unclamped MuJoCo <position> actuators; a
    # torque actuator would pass the one-per-joint check above and silently pose the
    # ghost at torques.
    position = (
      (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
      & (model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
      & (model.actuator_gear[:, 0] == 1.0)
      & (model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
      & (model.actuator_biasprm[:, 1] == -model.actuator_gainprm[:, 0])
      & (model.actuator_ctrllimited == 0)
    )
    if not position.all():
      raise RuntimeError(
        "expected unclamped MuJoCo position actuators (data.ctrl = joint target)"
      )

  def ghost_qpos(self) -> np.ndarray:
    """Ghost pose (nq,): free joint = env 0's current root, joints = env 0's current
    actuator targets."""
    data = self._env.sim.data
    qpos = data.qpos[0].cpu().numpy().astype(np.float64)
    qpos[self.ctrl_q_adr] = data.ctrl[0].cpu().numpy()
    return qpos

  def render(
    self,
    ghost_qpos: np.ndarray,
    draw_extra: Callable[[object], None] | None = None,
  ) -> np.ndarray:
    """RGB frame (H, W, 3) uint8 of env 0 with the ghost at ``ghost_qpos`` (nq,);
    ``draw_extra(visualizer)`` may add more debug geometry (e.g. arrows)."""

    def draw_ghost(visualizer) -> None:
      visualizer.add_ghost_mesh(
        ghost_qpos, model=self._ghost_model, label="action_ghost"
      )
      if draw_extra is not None:
        draw_extra(visualizer)

    self._renderer.update(self._env.sim.data, debug_vis_callback=draw_ghost)
    return self._renderer.render()

  def close(self) -> None:
    self._renderer.close()


def draw_text(
  frame: np.ndarray, lines: Sequence[str], bottom_lines: Sequence[str] = ()
) -> np.ndarray:
  """``frame`` (H, W, 3) uint8 with white outlined text lines at the top-left and
  bottom-left."""
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  size = max(12, frame.shape[0] // 30)
  font = ImageFont.load_default(size=size)
  outline = {"fill": "white", "stroke_width": 2, "stroke_fill": "black"}
  for row, line in enumerate(lines):
    draw.text((8, 6 + row * (size + 4)), line, font=font, **outline)
  for row, line in enumerate(reversed(bottom_lines)):
    y = frame.shape[0] - (row + 1) * (size + 4) - 4
    draw.text((8, y), line, font=font, **outline)
  return np.asarray(image)


def short_name(joint: str) -> str:
  """Joint name without the ``_joint`` suffix, for on-screen text."""
  return joint.removesuffix("_joint")


def gap_summary(
  gap_abs: np.ndarray, outside: np.ndarray, saturated: np.ndarray
) -> dict:
  """Statistics of ``gap_abs`` (T, ...) in rad and deg; ``outside``/``saturated`` are
  boolean like ``gap_abs``."""
  stats = {
    "mean_rad": gap_abs.mean(),
    "p95_rad": np.percentile(gap_abs, 95),
    "max_rad": gap_abs.max(),
  }
  stats.update(
    {key.replace("rad", "deg"): np.degrees(value) for key, value in list(stats.items())}
  )
  stats["frac_target_outside_range"] = outside.mean()
  stats["frac_torque_saturated_end_of_step"] = saturated.mean()
  return {key: float(value) for key, value in stats.items()}


def pick_sheet_frames(
  gap_mean: np.ndarray, step_dt: float
) -> tuple[list[int], list[int]]:
  """Largest-mean-gap frame indices (>= 0.3 s apart) and evenly spaced frame indices
  of a (T,) gap curve."""
  largest: list[int] = []
  for index in np.argsort(-gap_mean):
    if all(abs(int(index) - other) * step_dt >= 0.3 for other in largest):
      largest.append(int(index))
    if len(largest) == SHEET_LARGEST:
      break
  rest = [index for index in range(len(gap_mean)) if index not in largest]
  even = [rest[int(k)] for k in np.linspace(0, len(rest) - 1, SHEET_EVEN).round()]
  return largest, even


def contact_sheet(
  frames: list[np.ndarray], largest: list[int], even: list[int], title: str
) -> Image.Image:
  """4-column grid of the chosen frames in time order; largest-gap frames framed in red
  and labelled."""
  tile_h, tile_w = (int(s * 0.75) for s in frames[0].shape[:2])
  indices = sorted(largest + even)
  columns, rows = 4, -(-len(indices) // 4)
  title_h = 34
  sheet = Image.new("RGB", (columns * tile_w, title_h + rows * tile_h), "white")
  draw = ImageDraw.Draw(sheet)
  draw.text((8, 6), title, font=ImageFont.load_default(size=18), fill="black")
  for slot, index in enumerate(indices):
    tile = Image.fromarray(frames[index]).resize(
      (tile_w, tile_h), Image.Resampling.LANCZOS
    )
    x, y = (slot % columns) * tile_w, title_h + (slot // columns) * tile_h
    sheet.paste(tile, (x, y))
    if index in largest:
      draw.rectangle(
        (x + 1, y + 1, x + tile_w - 2, y + tile_h - 2), outline="red", width=4
      )
      label = f"largest gap #{largest.index(index) + 1}"
      font = ImageFont.load_default(size=16)
      draw.text((x + tile_w - 150, y + 8), label, font=font, fill="red")
  return sheet


def render_clip(clip: str, args: Namespace) -> dict:
  """Play ``clip``'s tracker once with the action ghost; write both videos, the gap JSON
  and the contact sheet."""
  import mediapy
  import torch

  from mjlab.envs.mdp.actions import JointPositionAction

  checkpoint = resolve_checkpoint(
    Namespace(checkpoint=None, run_dir=None, clip=clip, log_root=args.log_root)
  )
  if not checkpoint.parent.name.endswith("_full"):
    raise RuntimeError(f"{clip}: newest run {checkpoint.parent} is not a _full run")
  env_yaml = yaml.load(
    (checkpoint.parent / "params" / "env.yaml").read_text(), Loader=ParamsLoader
  )
  motion = run_motion_file(env_yaml, checkpoint.parent)
  num_frames = len(np.load(motion)["joint_pos"])
  print(f"[ghost] {clip}: {checkpoint} on {motion} ({num_frames} frames)", flush=True)

  vec_env, policy = make_play_env(checkpoint, motion)
  env = vec_env.unwrapped
  robot = env.scene["robot"]
  action_term = env.action_manager.get_term("joint_pos")
  assert isinstance(action_term, JointPositionAction)
  if action_term.cfg.clip is not None or vec_env.clip_actions is not None:
    raise RuntimeError(
      "action clipping is configured; the target formula below would be incomplete"
    )
  action_joint_ids = action_term.target_ids
  renderer = ActionGhostRenderer(env, args.width, args.height)
  joint_names = list(robot.joint_names)
  joint_q_adr = renderer.joint_q_adr
  action_q_adr = joint_q_adr[action_joint_ids.cpu().numpy()]
  # (29, 2) hard range, robot joint order.
  limits = robot.data.joint_pos_limits[0].cpu().numpy()
  # Joint index of each ctrl.
  ctrl_joint = np.searchsorted(joint_q_adr, renderer.ctrl_q_adr)
  if not np.array_equal(joint_q_adr[ctrl_joint], renderer.ctrl_q_adr):
    raise RuntimeError("robot joint qpos addresses are not sorted")
  force_limit = env.sim.mj_model.actuator_forcerange[:, 1]  # (nu,)
  step_dt = env.step_dt

  targets, actuals, saturated, frames, terminated_steps = [], [], [], [], []
  formula_err = root_err = 0.0
  obs = vec_env.get_observations()
  # Step num_frames - 1 would wrap the motion and teleport to frame 0.
  for step in range(1, num_frames - 1):
    with torch.no_grad():
      obs, _, dones, _ = vec_env.step(policy(obs))
    if bool(dones[0]):
      terminated_steps.append(step)
    ghost_qpos = renderer.ghost_qpos()
    qpos = env.sim.data.qpos[0].cpu().numpy()
    # The target the actuators received (data.ctrl) vs the action term's formula, per
    # action dimension.
    formula = action_term.raw_action * action_term.scale + action_term.offset
    formula = (formula - robot.data.encoder_bias[:, action_joint_ids])[0].cpu().numpy()
    formula_err = max(
      formula_err, float(np.abs(ghost_qpos[action_q_adr] - formula).max())
    )
    free = renderer.free_q_adr
    root_err = max(root_err, float(np.abs(ghost_qpos[free] - qpos[free]).max()))
    target, actual = ghost_qpos[joint_q_adr], qpos[joint_q_adr]
    force = env.sim.data.actuator_force[0].cpu().numpy()
    saturated_joint = np.zeros(len(joint_names), dtype=bool)
    saturated_joint[ctrl_joint] = np.abs(force) >= force_limit - 1e-6
    gap_deg = np.degrees(np.abs(target - actual))
    worst = int(gap_deg.argmax())
    frame = renderer.render(ghost_qpos)
    frames.append(
      draw_text(
        frame,
        [
          f"{clip}   t = {step * step_dt:5.2f} s",
          f"mean |target - actual| = {gap_deg.mean():4.1f} deg",
          f"largest: {short_name(joint_names[worst])} {gap_deg[worst]:5.1f} deg",
        ],
        ["orange ghost = policy joint targets (root = robot root)"],
      )
    )
    targets.append(target)
    actuals.append(actual)
    saturated.append(saturated_joint)
  renderer.close()
  vec_env.close()
  if formula_err > 1e-6 or root_err > 0.0:
    raise RuntimeError(
      f"{clip}: ghost is not the actuator target (ctrl-formula {formula_err:.2e}, "
      f"root {root_err:.2e})"
    )

  # (T, 29) each.
  targets, actuals, saturated = (
    np.array(targets),
    np.array(actuals),
    np.array(saturated),
  )
  gap_abs = np.abs(targets - actuals)
  outside = (targets < limits[:, 0]) | (targets > limits[:, 1])
  gap_mean_deg = np.degrees(gap_abs.mean(axis=1))  # (T,)

  out_dir = args.out_dir / clip / "native_action_ghost"
  out_dir.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(out_dir / "realtime.mp4", frames, fps=1.0 / step_dt)
  mediapy.write_video(out_dir / "slow025.mp4", frames, fps=SLOW_FACTOR / step_dt)
  largest, even = pick_sheet_frames(gap_mean_deg, step_dt)
  title = (
    f"{clip}: robot opaque (end of step), orange ghost = joint targets of that step; "
    "red = largest mean gap"
  )
  contact_sheet(frames, largest, even, title).save(out_dir / "sheet.png")

  report = {
    "clip": clip,
    "checkpoint": str(checkpoint),
    "motion_file": str(motion),
    "motion_frames": num_frames,
    "frames_rendered": len(frames),
    "step_dt": step_dt,
    "pairing": "target set at policy step k (held over its 4 substeps) vs robot joint "
    "pos at the end of step k",
    "target": "data.ctrl = raw_action * scale + default_joint_pos - encoder_bias; "
    "position actuators, ctrllimited=false (targets beyond the joint range are kept), "
    "torque clamped to +-effort_limit",
    "terminated_steps": terminated_steps,
    "check_max_abs_ctrl_minus_action_formula_rad": formula_err,
    "check_max_abs_ghost_root_minus_robot_root": root_err,
    "all_joints": gap_summary(gap_abs, outside, saturated),
    "joints": {
      name: {
        **gap_summary(gap_abs[:, j], outside[:, j], saturated[:, j]),
        "range_rad": limits[j].tolist(),
      }
      for j, name in enumerate(joint_names)
    },
    "largest_gap_moments": [
      {
        "t": (index + 1) * step_dt,
        "mean_gap_deg": float(gap_mean_deg[index]),
        "worst_joint": joint_names[int(gap_abs[index].argmax())],
        "worst_gap_deg": float(np.degrees(gap_abs[index].max())),
      }
      for index in largest
    ],
  }
  (out_dir / "gap.json").write_text(json.dumps(report, indent=2))
  summary = report["all_joints"]
  print(
    f"[ghost] {clip}: mean {summary['mean_deg']:.2f} deg, "
    f"max {summary['max_deg']:.1f} deg, "
    f"outside range {summary['frac_target_outside_range']:.3f}, "
    f"terminations {terminated_steps}, "
    f"ctrl-formula {formula_err:.2e}, root {root_err:.2e}",
    flush=True,
  )
  return report


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument(
    "--clip", nargs="+", help="clips to render (default: the --top fastest)"
  )
  parser.add_argument("--top", type=int, default=5)
  parser.add_argument(
    "--motions-dir",
    type=Path,
    default=MOTIONS_DIR,
    help="reference motions used for ranking",
  )
  parser.add_argument("--log-root", type=Path, default=RUNS_DIR)
  parser.add_argument("--out-dir", type=Path, default=VIDEOS_DIR)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  args = parser.parse_args()
  # Set before mjlab imports torch / mujoco. EGL device ids are physical GPU ids (EGL
  # ignores CUDA_VISIBLE_DEVICES), so rendering would otherwise land on GPU 0.
  os.environ.update(
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_EGL_DEVICE_ID=str(args.gpu),
    MUJOCO_GL="egl",
    WANDB_MODE="disabled",
  )

  clips = args.clip
  if clips is None:
    ranking = rank_clips(args.motions_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "action_ghost_ranking.json").write_text(
      json.dumps(ranking, indent=2)
    )
    for place, row in enumerate(ranking, 1):
      print(
        f"{place:2d} {row['clip']:32s} "
        f"joint speed rms {row['mean_joint_speed_rms']:.3f} rad/s, "
        f"pelvis {row['mean_pelvis_speed']:.3f} m/s"
      )
    clips = [row["clip"] for row in ranking[: args.top]]
  for clip in clips:
    render_clip(clip, args)


if __name__ == "__main__":
  main()
