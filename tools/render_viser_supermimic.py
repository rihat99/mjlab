"""Render a SuperMimic tracker on one clip of its library through viser.

Two videos of the same rollout in ``render_viser.py``'s look: one with the reference
motion as a blue ghost, one with the policy's joint targets as the orange ghost.

The rollout is ``evaluate_library.py``'s protocol on this one clip: the task's play
config (no observation noise, no randomization, foot friction 1.0, no reset noise,
clips start at frame 0) with the clip as a one-clip library, 1 env, the checkpoint
loaded with the task's runner (actor only), its deterministic policy, ``env.reset()``,
then steps until the clip's ``motion_end`` time-out or a fall. Kept after every
physics substep (200 Hz): the robot pose and the actuator targets ``data.ctrl``; per
policy step: the reward and the reference frame the step is scored against
(``MotionCommand.time_steps`` before the step; step k scores frame k), whose root pose
and joint angles are the reference ghost. Both ghosts hold over the 4 substeps of a
policy step.

Rendered: a window of WINDOW_S seconds, or the whole clip if it is shorter, starting
at the ``--window-start`` scored frame (default: the window with the largest summed
reference root speed), one frame per policy step (50 fps, the state at the end of the
step), each frame two browser renders blended (without the ghost and with an opaque
ghost, at GHOST_OPACITY). Outputs in ``<out-dir>/<clip>/``:

- ``reference.mp4``: ghost = the reference pose (REFERENCE_RGB);
- ``targets.mp4``: ghost = the robot's root with its 29 joints at ``data.ctrl``;
- ``reference_t<T>.png``, ``targets_t<T>.png``: stills at the ``--stills`` clip times
  (default: none); ``--only-stills`` renders only these;
- ``rollout.npz``: the whole clip's rollout (per substep ``qpos``, ``ctrl``; per step
  ``frame``, ``reward``, ``ref_qpos``) and the window.

    PY=<worktree>/.venv/bin/python
    $PY tools/render_viser_supermimic.py --gpu 0 --chrome CHROME \\
        --checkpoint RUN_DIR/model_14000.pt --clip dance1_subject3 \\
        [--stills 50 60 --only-stills]

``CHROME`` is a Chromium headless shell or a wrapper that execs one (see
``render_viser.py``).
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from argparse import Namespace
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from paths import LOG_ROOT, MOTIONS_DIR, VISER_DIR
from PIL import Image
from render_viser import (
  BACKGROUND_RGB,
  CAPTION_GHOST_RGB,
  GHOST_OPACITY,
  GHOST_RGB,
  HEIGHT,
  WIDTH,
  HeadlessBrowser,
  build_scene,
  camera_path,
  caption,
  free_port,
  ghost_model_of,
)
from train_library import TASK, resolve_checkpoint, run_library

if TYPE_CHECKING:
  import viser

  from mjlab.viewer.viser.scene import MjlabViserScene

WINDOW_S = 20.0
# Blue-grey, deep enough to read at GHOST_OPACITY.
REFERENCE_RGB, CAPTION_REFERENCE_RGB = (0.12, 0.34, 0.72), (30, 90, 190)
GHOSTS = {  # ghost: (ghost colour, legend colour, legend)
  "reference": (REFERENCE_RGB, CAPTION_REFERENCE_RGB, "blue = reference motion"),
  "targets": (GHOST_RGB, CAPTION_GHOST_RGB, "orange = commanded joint targets"),
}


def rollout(checkpoint: Path, motion: Path) -> dict:
  """Play ``motion`` once from frame 0 as ``evaluate_library.py`` does.

  Keeps qpos and ctrl after every substep and, per policy step, the reward and the
  reference pose the step is scored against.
  """
  import torch

  import mjlab.tasks  # noqa: F401 -- registers the tasks
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from mjlab.tasks.tracking.scripts.evaluate_library import make_library_env
  from mjlab.tasks.tracking.supermimic.model_interface import max_step_reward
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  agent_cfg = load_rl_cfg(TASK)
  vec_env = make_library_env(TASK, [str(motion)], "cuda:0", agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK)
  assert runner_cls is not None, f"{TASK} registers no runner"
  runner = runner_cls(vec_env, asdict(agent_cfg), device="cuda:0")
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = runner.get_inference_policy(device="cuda:0")
  env = vec_env.unwrapped
  sim, robot = env.sim, env.scene["robot"]
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  if command.cfg.body_names[0] != "pelvis" or robot.body_names[0] != "pelvis":
    raise RuntimeError("expected the pelvis (the free-joint root) as first body")
  # (7,) root position, quaternion wxyz
  free_q_adr = robot.indexing.free_joint_q_adr.cpu().numpy()
  # (29,), robot joint order (the command's order)
  joint_q_adr = robot.indexing.joint_q_adr.cpu().numpy()
  max_reward = max_step_reward(env)

  obs, _ = vec_env.reset()
  sub_qpos, sub_ctrl = [], []
  sim_step = sim.step  # the env's decimation loop calls sim.step once per substep

  def step_and_keep() -> None:
    sim_step()
    sub_qpos.append(sim.data.qpos[0].cpu().numpy().astype(np.float64))
    sub_ctrl.append(sim.data.ctrl[0].cpu().numpy().astype(np.float64))

  sim.step = step_and_keep
  step_frame, step_reward, step_ref_qpos = [], [], []
  while True:
    # The frame this step's reward is scored against.
    step_frame.append(int(command.time_steps[0]))
    ref_qpos = sim.data.qpos[0].cpu().numpy().astype(np.float64)
    # Includes the env origin.
    ref_qpos[free_q_adr[:3]] = command.body_pos_w[0, 0].cpu().numpy()
    ref_qpos[free_q_adr[3:]] = command.body_quat_w[0, 0].cpu().numpy()
    ref_qpos[joint_q_adr] = command.joint_pos[0].cpu().numpy()
    step_ref_qpos.append(ref_qpos)
    with torch.no_grad():
      obs, reward, dones, _ = vec_env.step(policy(obs))
    step_reward.append(float(reward[0]))
    if bool(dones[0]):
      failed = bool(env.termination_manager.terminated[0])
      break
  num_frames = int(command.motion.clip_end[0] - command.motion.clip_start[0])
  states = {
    "model": copy.deepcopy(sim.mj_model),
    "free_q_adr": free_q_adr,
    "sub_qpos": np.array(sub_qpos),
    "sub_ctrl": np.array(sub_ctrl),
    "step_frame": np.array(step_frame),
    "step_reward": np.array(step_reward),
    "step_ref_qpos": np.array(step_ref_qpos),
    "failed": failed,
    "num_frames": num_frames,
    "max_step_reward": max_reward,
    "normalized_return": sum(step_reward) / (max_reward * (num_frames - 1)),
    "physics_dt": env.physics_dt,
    "step_dt": env.step_dt,
    "decimation": env.cfg.decimation,
  }
  vec_env.close()
  return states


def best_window(motion: Path, num_steps: int) -> int:
  """First scored frame of the ``num_steps`` frames (of 1 .. last) that have the
  largest summed reference root speed."""
  speed = np.linalg.norm(np.load(motion)["body_lin_vel_w"][:, 0], axis=1)  # (F,)
  return 1 + int(np.argmax(np.convolve(speed[1:], np.ones(num_steps), mode="valid")))


def window_states(states: dict, first_frame: int, num_steps: int) -> dict:
  """``states`` cut to the policy steps scoring frames ``first_frame`` ..
  ``first_frame + num_steps - 1``.

  Adds the per-substep reference pose ``sub_ref_qpos`` and the window's first
  substep ``substep_offset``.
  """
  first_step = int(np.searchsorted(states["step_frame"], first_frame))
  steps = slice(first_step, first_step + num_steps)
  expected = np.arange(first_frame, first_frame + num_steps)
  if not np.array_equal(states["step_frame"][steps], expected):
    raise RuntimeError(
      f"the rollout does not score frames {first_frame} .. "
      f"{first_frame + num_steps - 1}"
    )
  decimation = states["decimation"]
  substeps = slice(first_step * decimation, (first_step + num_steps) * decimation)
  return {
    **states,
    "sub_qpos": states["sub_qpos"][substeps],
    "sub_ctrl": states["sub_ctrl"][substeps],
    "sub_ref_qpos": np.repeat(states["step_ref_qpos"][steps], decimation, axis=0),
    "substep_offset": first_step * decimation,
  }


def render_video(
  clip: str, states: dict, server: viser.ViserServer, args: Namespace, ghost: str
) -> None:
  """Render the window's policy-step frames with ``ghost`` (a GHOSTS key).

  Writes ``<ghost>.mp4`` and the stills.
  """
  import imageio_ffmpeg
  import mediapy
  import mujoco

  # The ffmpeg shipped with imageio (there may be none on PATH).
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  model, decimation = states["model"], states["decimation"]
  physics_dt = states["physics_dt"]
  ghost_rgb, legend_rgb, legend = GHOSTS[ghost]
  ghost_model = ghost_model_of(model, ghost_rgb)
  if ghost == "reference":
    ghost_qpos = states["sub_ref_qpos"]
  else:
    ghost_qpos = states["sub_qpos"].copy()
    ghost_qpos[:, model.jnt_qposadr[model.actuator_trnid[:, 0]]] = states["sub_ctrl"]
  data = mujoco.MjData(model)
  positions, targets = camera_path(states)
  background = Image.new("RGBA", (WIDTH, HEIGHT), (*BACKGROUND_RGB, 255))
  offset = states["substep_offset"]
  # Stills at the end of the policy step nearest to each requested clip time.
  stills = {
    round(t / states["step_dt"]) * decimation - 1 - offset: t for t in args.stills
  }
  if not all(0 <= i < len(states["sub_qpos"]) for i in stills):
    raise ValueError(f"--stills {args.stills} not all inside the window")
  if args.only_stills:
    frames = sorted(stills)
  else:
    frames = range(decimation - 1, len(states["sub_qpos"]), decimation)

  def show(scene: MjlabViserScene, i: int, ghost_opacity: float | None) -> None:
    data.qpos[:] = states["sub_qpos"][i]
    mujoco.mj_kinematics(scene.mj_model, data)
    scene.clear()
    if ghost_opacity is not None:
      scene.add_ghost_mesh(ghost_qpos[i], ghost_model, alpha=ghost_opacity, label=ghost)
    scene.update_from_mjdata(data)

  def frame_at(browser: HeadlessBrowser, scene: MjlabViserScene, i: int) -> np.ndarray:
    show(scene, i, None)
    plain = browser.render(positions[i], targets[i], background).astype(np.float32)
    show(scene, i, 1.0)
    opaque = browser.render(positions[i], targets[i], background).astype(np.float32)
    return (plain + GHOST_OPACITY * (opaque - plain)).round().astype(np.uint8)

  start = time.monotonic()
  with ExitStack() as stack:
    browser = stack.enter_context(
      HeadlessBrowser(server, args.chrome, None if args.software else args.gpu)
    )
    assert browser.client is not None  # set by __enter__
    scene = build_scene(browser.client, model)
    previous = frame_at(browser, scene, frames[0])
    # Meshes load asynchronously in the browser: wait until two renders agree.
    for _ in range(40):
      time.sleep(0.25)
      current = frame_at(browser, scene, frames[0])
      if np.array_equal(current, previous):
        break
      previous = current
    else:
      raise RuntimeError("the browser's render did not settle")
    print(
      f"[viser] {ghost}: browser ready after {time.monotonic() - start:.1f} s",
      flush=True,
    )

    out_dir = args.out_dir / clip
    out_dir.mkdir(parents=True, exist_ok=True)
    video = None
    if not args.only_stills:
      video = stack.enter_context(
        mediapy.VideoWriter(
          out_dir / f"{ghost}.mp4",
          shape=(HEIGHT, WIDTH),
          fps=1.0 / states["step_dt"],
          crf=18,
        )
      )
    start = time.monotonic()
    for i in frames:
      t = (offset + i + 1) * physics_dt
      frame = caption(
        frame_at(browser, scene, i),
        f"{clip}   SuperMimic   t = {t:6.2f} s",
        legend,
        legend_rgb,
      )
      if i in stills:
        Image.fromarray(frame).save(out_dir / f"{ghost}_t{stills[i]:.2f}.png")
      if video is not None:
        video.add_image(frame)
  elapsed = time.monotonic() - start
  print(
    f"[viser] {ghost}: {len(frames)} frames in {elapsed:.0f} s "
    f"({elapsed / len(frames):.2f} s per frame)",
    flush=True,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--gpu", type=int, default=0, help="simulation and WebGL GPU")
  parser.add_argument(
    "--chrome",
    required=True,
    help="Chromium headless shell, or a wrapper that execs one",
  )
  parser.add_argument(
    "--clip", default="dance1_subject3", help="clip id of the run's library"
  )
  parser.add_argument(
    "--run-dir", type=Path, help=f"run dir (default: newest run under {LOG_ROOT})"
  )
  parser.add_argument(
    "--checkpoint", type=Path, help="checkpoint file (default: last in the run dir)"
  )
  parser.add_argument(
    "--window-start",
    type=int,
    help="first scored frame of the window (default: most motion)",
  )
  parser.add_argument(
    "--ghosts", nargs="+", choices=tuple(GHOSTS), default=list(GHOSTS)
  )
  parser.add_argument(
    "--software",
    action="store_true",
    help="render with SwiftShader instead of the GPU",
  )
  parser.add_argument(
    "--stills",
    type=float,
    nargs="*",
    default=[],
    help="clip times (s) of full-size PNG stills",
  )
  parser.add_argument(
    "--only-stills",
    action="store_true",
    help="render only the stills (to check the look)",
  )
  parser.add_argument("--out-dir", type=Path, default=VISER_DIR)
  args = parser.parse_args()
  if args.only_stills and not args.stills:
    parser.error("--only-stills needs --stills")
  # Set before mjlab imports torch / mujoco.
  os.environ.update(
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_EGL_DEVICE_ID=str(args.gpu),
    MUJOCO_GL="egl",
  )
  import viser

  checkpoint = resolve_checkpoint(
    Namespace(checkpoint=args.checkpoint, run_dir=args.run_dir, log_root=LOG_ROOT)
  ).resolve()
  library = {path.stem: path for path in run_library(checkpoint.parent, MOTIONS_DIR)}
  if args.clip not in library:
    parser.error(f"--clip must be a clip of the run's library: {sorted(library)}")
  motion = library[args.clip]
  print(f"[viser] {args.clip}: {checkpoint} ({TASK}) on {motion}", flush=True)
  states = rollout(checkpoint, motion)
  print(
    f"[viser] {len(states['step_reward'])} steps, failed {states['failed']}, "
    f"normalised return {states['normalized_return']:.4f}",
    flush=True,
  )

  # A clip of F frames scores frames 1 .. F - 1.
  num_steps = min(round(WINDOW_S / states["step_dt"]), states["num_frames"] - 1)
  first_frame = args.window_start
  if first_frame is None:
    first_frame = best_window(motion, num_steps)
  window = window_states(states, first_frame, num_steps)
  last_frame = first_frame + num_steps - 1
  print(
    f"[viser] window: scored frames {first_frame} .. {last_frame} "
    f"(t = {first_frame * states['step_dt']:.2f} .. "
    f"{last_frame * states['step_dt']:.2f} s)",
    flush=True,
  )
  (args.out_dir / args.clip).mkdir(parents=True, exist_ok=True)
  np.savez(
    args.out_dir / args.clip / "rollout.npz",
    qpos=states["sub_qpos"],
    ctrl=states["sub_ctrl"],
    frame=states["step_frame"],
    reward=states["step_reward"],
    ref_qpos=states["step_ref_qpos"],
    failed=states["failed"],
    normalized_return=states["normalized_return"],
    window_frames=(first_frame, num_steps),
    physics_dt=states["physics_dt"],
    step_dt=states["step_dt"],
    checkpoint=str(checkpoint),
    motion=str(motion),
  )

  server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
  try:
    for ghost in args.ghosts:
      render_video(args.clip, window, server, args, ghost)
  finally:
    server.stop()


if __name__ == "__main__":
  main()
