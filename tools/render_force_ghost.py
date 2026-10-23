#!/usr/bin/env python3
"""Action-ghost videos with contact-force arrows (the retargeter's planned forces vs the
simulation) and a per-substep record of the simulated actuator torques and ground
contact forces.

Per clip, the newest ``_full`` run (final checkpoint) plays the clip once from frame 0
in mjlab's play config (1 env, deterministic action), as ``render_action_ghost.py``.
Unless ``--randomized``, the play config is made nominal and reproducible: ``seed`` =
42; the startup events ``foot_friction`` (feet keep the model's mu 0.6), ``base_com``
and ``encoder_bias`` are removed; the reset joint noise
``commands.motion.joint_position_range`` is (0, 0). Play already drops ``push_robot``,
observation noise and the reset pose / velocity noise. ``--randomized`` keeps the
startup randomization (seed 42 only).

A wrapper around ``env.sim.step`` records after every physics substep (200 Hz) what
acted during it, at the state of the substep start: ``actuator_force``
(= ``qfrc_actuator``, gear 1), joint-limit constraint torques, the robot centre of mass
and, per robot body, the force of the ground on the body in world axes
(``mujoco_warp.contact_force`` in the world frame; contact forces act from geom1 on
geom2) summed over its contacts, with its centre of pressure.

Video frames: the robot opaque and the orange ghost at the joint targets
(``render_action_ghost.py``), plus arrows at each body's centre of pressure (moved 0.2 m
towards the camera along its view ray, so the limb does not hide them), 1 body weight =
0.5 m: the planned force (wide, blue; linearly interpolated from the plan's frame rate
to the frame's motion time; its centre of pressure is carried in the body's frame onto
the simulated body, so both arrows sit on the robot where it drifts from the reference)
and the simulated force averaged over the step's 4 substeps (thin, green). Sim time is
motion time: a reset (start or termination) puts the robot at frame 0 with sim time 0.

Input contract: ``--br-dir`` holds ``<clip>_br.npz`` per clip, the retargeter's plan
(written by BetterRetarget's ``scripts/dump_mjlab_plan.py``; BR below). Keys read here, T plan
frames, B contacting bodies:

- ``time`` (T,) s, motion time of each plan frame; ``valid`` (T,) bool, frames with a
  plan (invalid frames draw no arrow);
- ``dof_names`` (29,) str, joint names of mjlab's G1 without the ``robot/`` prefix;
  ``root_pos`` (T, 3) m, ``root_quat_xyzw`` (T, 4), ``dof_pos`` (T, 29) rad: the
  retargeted pose (MuJoCo FK of it carries the centre of pressure into body frames);
- ``body_joints`` (B,) str, per contacting body the joint that moves it (mjlab joint
  name without prefix; ``root_joint`` = the pelvis);
- ``body_force`` (T, B, 3) N, force of the ground on the body, world axes;
  ``body_cop`` (T, B, 3) m, its centre of pressure, world.

Outputs: ``<out-dir>/<clip>/native[_randomized]/{realtime,slow025}.mp4`` and
``sheet.png`` (red = largest force gap), and the record
``<rollouts-dir>/<clip>/native[_randomized].npz``:

    PY=.venv/bin/python  # from the repo root
    $PY tools/render_force_ghost.py --gpu 0 --br-dir DIR --clip amass_cmu_88_05 \\
        [--randomized]
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import yaml
from numpy.typing import ArrayLike
from paths import ROLLOUTS_DIR, RUNS_DIR, VIDEOS_DIR
from render_action_ghost import (
  SLOW_FACTOR,
  ActionGhostRenderer,
  contact_sheet,
  draw_text,
  make_play_env,
  pick_sheet_frames,
)
from train_tracker import ParamsLoader, resolve_checkpoint, run_motion_file

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

SEED = 42
ARROW_M_PER_BW = 0.5
MIN_ARROW_BW = 0.02  # arrows shorter than 1 cm are not drawn
# Arrows are drawn this far towards the camera, along its view ray: on screen they stay
# where they are, but the opaque limb they start at (a vertical arrow from a foot or
# hand runs up through the shin or forearm) no longer hides them.
TOWARD_CAMERA_M = 0.2
BR_RGBA, BR_WIDTH = (0.1, 0.35, 1.0, 0.55), 0.025
SIM_RGBA, SIM_WIDTH = (0.0, 0.85, 0.2, 1.0), 0.015
PREFIX = "robot/"  # mjlab names the robot's bodies and joints robot/<urdf name>


def nominal_cfg(cfg: ManagerBasedRlEnvCfg) -> None:
  """Seeded play config without startup randomization or reset joint noise (the
  nominal robot)."""
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  cfg.seed = SEED
  for name in ("foot_friction", "base_com", "encoder_bias"):
    del cfg.events[name]
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.joint_position_range = (0.0, 0.0)


def seeded_cfg(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.seed = SEED


class SubstepRecorder:
  """Wraps ``env.sim.step`` (the env's decimation loop calls it once per substep) and
  keeps, per substep, what acted during it: torques, ground contact forces per robot
  body, and the state at the substep start."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    import mujoco
    import warp as wp

    self._sim = env.sim
    self._step = env.sim.step
    env.sim.step = self.step
    model = env.sim.mj_model
    robot = env.scene["robot"]
    self.body_ids = (
      robot.indexing.body_ids.cpu().numpy()
    )  # (B,) robot bodies, model ids
    self.body_index = np.full(model.nbody, -1)
    self.body_index[self.body_ids] = np.arange(len(self.body_ids))
    self.root_body = robot.indexing.root_body_id
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    self.ground_geom = model.geom_bodyid == terrain  # (ngeom,)
    if not self.ground_geom.any():
      raise RuntimeError("no terrain geom")
    self.geom_body = model.geom_bodyid
    self.jnt_range = model.jnt_range
    self.jnt_qposadr, self.jnt_dofadr = model.jnt_qposadr, model.jnt_dofadr
    wp_data = env.sim.wp_data
    device = env.sim.wp_device
    self._contact_ids = wp.array(
      np.arange(wp_data.naconmax, dtype=np.int32), dtype=int, device=device
    )
    self._contact_out = wp.zeros(
      wp_data.naconmax, dtype=wp.spatial_vectorf, device=device
    )
    self._limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    self.rows: dict[str, list[np.ndarray]] = {}

  def _keep(self, **values: ArrayLike) -> None:
    for key, value in values.items():
      self.rows.setdefault(key, []).append(np.asarray(value))

  def step(self) -> None:
    import mujoco_warp as mjw

    wp_data = self._sim.wp_data
    time = wp_data.time.numpy()[0]
    qpos, qvel = wp_data.qpos.numpy()[0].copy(), wp_data.qvel.numpy()[0].copy()
    self._step()
    # Everything below was computed by the step's forward pass at the substep-start
    # state (qpos, qvel).
    nacon = int(wp_data.nacon.numpy()[0])
    if nacon >= wp_data.naconmax:
      raise RuntimeError(f"contact buffer full ({nacon}); forces would be missing")
    mjw.contact_force(
      self._sim.wp_model, wp_data, self._contact_ids, True, self._contact_out
    )
    geom = wp_data.contact.geom.numpy()[:nacon]  # (n, 2)
    pos = wp_data.contact.pos.numpy()[:nacon]  # (n, 3)
    force = self._contact_out.numpy()[:nacon, :3]  # (n, 3) world, from geom1 on geom2
    ground = self.ground_geom[geom]  # (n, 2)
    keep = ground[:, 0] ^ ground[:, 1]
    robot_geom = np.where(ground[:, 0], geom[:, 1], geom[:, 0])[keep]
    on_robot = np.where(ground[:, 0], 1.0, -1.0)[keep, None] * force[keep]
    body = self.body_index[self.geom_body[robot_geom]]
    if (body < 0).any():
      raise RuntimeError("a ground contact on a geom outside the robot")
    num_bodies = len(self.body_ids)
    body_force = np.zeros((num_bodies, 3))
    body_fz = np.zeros(num_bodies)
    body_pz = np.zeros((num_bodies, 3))
    np.add.at(body_force, body, on_robot)
    np.add.at(body_fz, body, on_robot[:, 2])
    np.add.at(body_pz, body, on_robot[:, 2:] * pos[keep])
    # Joint-limit rows: force >= 0 along +1 (lower limit) or -1 (upper limit) on the
    # joint's dof.
    nefc = int(wp_data.nefc.numpy()[0])
    efc_type = wp_data.efc.type.numpy()[0, :nefc]
    efc_id = wp_data.efc.id.numpy()[0, :nefc]
    efc_force = wp_data.efc.force.numpy()[0, :nefc]
    qfrc_limit = np.zeros(len(qvel))
    is_limit = efc_type == self._limit_type
    for joint, value in zip(efc_id[is_limit], efc_force[is_limit], strict=True):
      q = qpos[self.jnt_qposadr[joint]]
      low, high = self.jnt_range[joint]
      qfrc_limit[self.jnt_dofadr[joint]] += value if q - low < high - q else -value
    self._keep(
      time=time,
      qpos=qpos,
      qvel=qvel,
      actuator_force=wp_data.actuator_force.numpy()[0],
      ctrl=wp_data.ctrl.numpy()[0],
      qfrc_actuator=wp_data.qfrc_actuator.numpy()[0],
      qfrc_constraint=wp_data.qfrc_constraint.numpy()[0],
      qfrc_limit=qfrc_limit,
      com=wp_data.subtree_com.numpy()[0, self.root_body],
      nacon=nacon,
      body_force=body_force,
      body_fz=body_fz,
      body_pz=body_pz,
    )

  def arrays(self) -> dict[str, np.ndarray]:
    return {f"sub_{key}": np.stack(values) for key, values in self.rows.items()}


class PlannedForces:
  """The plan's per-body forces at any motion time, with the centre of pressure kept in
  the body's frame."""

  def __init__(self, br: dict[str, np.ndarray], model, free_q_adr: np.ndarray) -> None:
    import mujoco

    def body_id(name: str) -> int:
      return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PREFIX + name)

    def joint_id(name: str) -> int:
      return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PREFIX + name)

    self.time, self.valid = br["time"], br["valid"]
    names = [str(name) for name in br["body_joints"]]
    self.bodies = np.array(
      [
        body_id("pelvis") if name == "root_joint" else model.jnt_bodyid[joint_id(name)]
        for name in names
      ]
    )
    if (self.bodies < 0).any():
      raise RuntimeError(f"unmapped BR bodies {names}")
    # The reference motion's body poses and centre of mass from MuJoCo FK of the plan's
    # frames.
    data = mujoco.MjData(model)
    dof_adr = [model.jnt_qposadr[joint_id(str(name))] for name in br["dof_names"]]
    root_body = body_id("pelvis")
    local_cop, com = [], []
    for i in range(len(self.time)):
      data.qpos[:] = model.qpos0
      data.qpos[free_q_adr[:3]] = br["root_pos"][i]
      data.qpos[free_q_adr[3:]] = br["root_quat_xyzw"][i][[3, 0, 1, 2]]
      data.qpos[dof_adr] = br["dof_pos"][i]
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      rotation = data.xmat[self.bodies].reshape(-1, 3, 3)
      local_cop.append(
        np.einsum("bji,bj->bi", rotation, br["body_cop"][i] - data.xpos[self.bodies])
      )
      com.append(data.subtree_com[root_body].copy())
    # (T, B, 3), (T, 3).
    self.local_cop, self.com = np.array(local_cop), np.array(com)
    self.force = br["body_force"] * self.valid[:, None, None]  # (T, B, 3)

  def at(self, t: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(force (B, 3), body-frame cop (B, 3)) at motion time ``t`` between two valid
    frames, else None."""
    i = int(np.searchsorted(self.time, t, side="right")) - 1
    if i < 0 or i + 1 >= len(self.time) or not (self.valid[i] and self.valid[i + 1]):
      return None
    w = (t - self.time[i]) / (self.time[i + 1] - self.time[i])
    force = (1 - w) * self.force[i] + w * self.force[i + 1]
    return force, (1 - w) * self.local_cop[i] + w * self.local_cop[i + 1]


def arrow(
  visualizer, start: np.ndarray, force: np.ndarray, weight: float, rgba, width: float
) -> None:
  if np.linalg.norm(force) / weight >= MIN_ARROW_BW:
    visualizer.add_arrow(
      start, start + force / weight * ARROW_M_PER_BW, rgba, width=width
    )


def render_clip(clip: str, args: Namespace) -> None:
  """Play ``clip``'s tracker once, record every substep, write both videos, the contact
  sheet and the record."""
  import mediapy
  import torch

  from mjlab.tasks.tracking.mdp.commands import MotionCommand

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
  tag = "_randomized" if args.randomized else ""
  print(
    f"[force-ghost] {clip}{tag}: {checkpoint} on {motion} ({num_frames} frames)",
    flush=True,
  )

  vec_env, policy = make_play_env(
    checkpoint, motion, seeded_cfg if args.randomized else nominal_cfg
  )
  env = vec_env.unwrapped
  model = env.sim.mj_model
  renderer = ActionGhostRenderer(env, args.width, args.height)
  recorder = SubstepRecorder(env)
  with np.load(args.br_dir / f"{clip}_br.npz") as data:
    br = {key: data[key] for key in data.files}
  planned = PlannedForces(br, model, renderer.free_q_adr)
  weight = float(model.body_subtreemass[recorder.root_body] * -model.opt.gravity[2])
  # BR bodies as indices into the recorded robot bodies.
  sim_body = recorder.body_index[planned.bodies]
  step_dt, decimation = env.step_dt, env.cfg.decimation
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  azimuth = np.radians(env.cfg.viewer.azimuth)
  elevation = np.radians(env.cfg.viewer.elevation)
  # MuJoCo's free camera sits at lookat - distance * forward.
  forward = np.array(
    [
      np.cos(elevation) * np.cos(azimuth),
      np.cos(elevation) * np.sin(azimuth),
      np.sin(elevation),
    ]
  )
  shift = -TOWARD_CAMERA_M * forward

  frames, gaps, terminated, step_qpos, step_ctrl, motion_time = [], [], [], [], [], []
  obs = vec_env.get_observations()
  # Step num_frames - 1 would wrap the motion and teleport to frame 0.
  for step in range(1, num_frames - 1):
    with torch.no_grad():
      obs, _, dones, _ = vec_env.step(policy(obs))
    if bool(dones[0]):
      terminated.append(step)
    # After a step the robot is at motion frame time_steps - 1 (it restarts at frame 0
    # after a termination).
    t = (int(command.time_steps[0]) - 1) * step_dt
    ghost_qpos = renderer.ghost_qpos()
    # (B_robot, 3), mean over the step.
    sim_force = np.mean(recorder.rows["body_force"][-decimation:], axis=0)
    fz = np.sum(recorder.rows["body_fz"][-decimation:], axis=0)
    sim_cop = (
      np.sum(recorder.rows["body_pz"][-decimation:], axis=0)
      / np.maximum(fz, 1e-9)[:, None]
    )
    plan = planned.at(t)
    xpos = env.sim.data.xpos[0].cpu().numpy()
    xmat = env.sim.data.xmat[0].cpu().numpy().reshape(-1, 3, 3)
    br_cop = (
      None
      if plan is None
      else xpos[planned.bodies] + np.einsum("bij,bj->bi", xmat[planned.bodies], plan[1])
    )

    def draw_arrows(
      visualizer, sim_force=sim_force, sim_cop=sim_cop, plan=plan, br_cop=br_cop
    ) -> None:
      if plan is not None and br_cop is not None:
        for force, start in zip(plan[0], br_cop, strict=True):
          arrow(visualizer, start + shift, force, weight, BR_RGBA, BR_WIDTH)
      for body in np.flatnonzero(np.abs(sim_force).sum(1) > 0):
        arrow(
          visualizer,
          sim_cop[body] + shift,
          sim_force[body],
          weight,
          SIM_RGBA,
          SIM_WIDTH,
        )

    frame = renderer.render(ghost_qpos, draw_arrows)
    br_fz = "  -  " if plan is None else f"{plan[0][:, 2].sum() / weight:4.2f}"
    sim_fz = sim_force[:, 2].sum() / weight
    frames.append(
      draw_text(
        frame,
        [
          f"{clip}{tag}   t = {t:5.2f} s",
          f"total vertical force: sim {sim_fz:4.2f} BW, BR plan {br_fz} BW",
        ],
        [
          "orange ghost = joint targets; arrows at centre of pressure, 1 BW = 0.5 m",
          "wide blue = BR planned force, thin green = simulated",
        ],
      )
    )
    planned_on_sim = np.zeros_like(sim_force)
    if plan is not None:
      planned_on_sim[sim_body] = plan[0]
    gaps.append(np.linalg.norm(sim_force - planned_on_sim, axis=1).sum() / weight)
    step_qpos.append(env.sim.data.qpos[0].cpu().numpy())
    step_ctrl.append(env.sim.data.ctrl[0].cpu().numpy())
    motion_time.append(t)
  renderer.close()
  vec_env.close()

  stem = f"{clip}{tag}"
  video_dir = args.out_dir / clip / f"native{tag}"
  video_dir.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(video_dir / "realtime.mp4", frames, fps=1.0 / step_dt)
  mediapy.write_video(video_dir / "slow025.mp4", frames, fps=SLOW_FACTOR / step_dt)
  largest, even = pick_sheet_frames(np.array(gaps), step_dt)
  title = (
    f"{stem}: blue = BR planned, green = simulated contact force (1 BW = 0.5 m); "
    "red = largest force gap"
  )
  contact_sheet(frames, largest, even, title).save(video_dir / "sheet.png")

  actuated_joint = model.actuator_trnid[:, 0]
  (args.rollouts_dir / clip).mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.rollouts_dir / clip / f"native{tag}.npz",
    # The recorded keys all start with "sub_"; pyright still matches the unpacked
    # dict against savez_compressed's allow_pickle.
    **recorder.arrays(),  # pyright: ignore[reportArgumentType]
    step_motion_time=np.array(motion_time),
    step_qpos=np.array(step_qpos),
    step_ctrl=np.array(step_ctrl),
    step_force_gap_bw=np.array(gaps),
    terminated_steps=np.array(terminated, dtype=int),
    actuator_joint=np.array(
      [model.joint(j).name.removeprefix(PREFIX) for j in actuated_joint]
    ),
    actuator_dof_adr=model.jnt_dofadr[actuated_joint],
    actuator_qpos_adr=model.jnt_qposadr[actuated_joint],
    actuator_forcerange=model.actuator_forcerange,
    actuator_joint_range=model.jnt_range[actuated_joint],
    robot_body=np.array(
      [model.body(b).name.removeprefix(PREFIX) for b in recorder.body_ids]
    ),
    br_body=np.array([model.body(b).name.removeprefix(PREFIX) for b in planned.bodies]),
    br_time=planned.time,
    br_com=planned.com,
    mass=model.body_subtreemass[recorder.root_body],
    gravity=-model.opt.gravity[2],
    physics_dt=env.physics_dt,
    step_dt=step_dt,
    seed=env.cfg.seed,
    randomized=args.randomized,
    checkpoint=str(checkpoint),
    motion_file=str(motion),
  )
  print(
    f"[force-ghost] {stem}: {len(frames)} frames, {len(recorder.rows['time'])} "
    f"substeps, terminations {terminated}, max contacts {max(recorder.rows['nacon'])}",
    flush=True,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--clip", nargs="+", required=True)
  parser.add_argument(
    "--br-dir", type=Path, required=True, help="<clip>_br.npz files (the plan)"
  )
  parser.add_argument(
    "--randomized",
    action="store_true",
    help="keep play's startup randomization (seeded)",
  )
  parser.add_argument("--log-root", type=Path, default=RUNS_DIR)
  parser.add_argument(
    "--out-dir", type=Path, default=VIDEOS_DIR, help="videos and contact sheet"
  )
  parser.add_argument(
    "--rollouts-dir",
    type=Path,
    default=ROLLOUTS_DIR,
    help="the per-substep record (npz)",
  )
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  args = parser.parse_args()
  # Set before mjlab imports torch / mujoco; EGL device ids are physical GPU ids (EGL
  # ignores CUDA_VISIBLE_DEVICES).
  os.environ.update(
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_EGL_DEVICE_ID=str(args.gpu),
    MUJOCO_GL="egl",
    WANDB_MODE="disabled",
  )
  for clip in args.clip:
    render_clip(clip, args)


if __name__ == "__main__":
  main()
