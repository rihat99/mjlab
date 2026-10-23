#!/usr/bin/env python3
"""Render a trained mjlab tracker's rollout through viser: mp4 videos from a headless
browser and an interactive recording. The simulated G1 plus an orange ghost G1 at the
policy's joint targets; no forces, no contact markers.

The rollout is the one ``render_force_ghost.py`` records: the clip's run ``--run``
(default: the newest run, which must be a ``_full`` run; final checkpoint) plays the
clip once from frame 0 in the play config of the task it was trained on
(``params/env.yaml``; 1 env, deterministic action), nominal robot (seed 42, no startup
randomization). After every physics substep (200 Hz) the robot pose, the actuator
targets ``data.ctrl``, the actuator torques and the PD gains in effect (constant for the
original tracker; the policy's choice for the Vic tasks) are kept. The ghost is the
robot's root with its 29 joints at those targets (``render_action_ghost.py``), so over
the 4 substeps of a policy step it holds that step's targets. Its meshes are moved
GHOST_INSET_M inward, so where ghost and robot coincide the ghost hides inside the robot
instead of z-fighting with it.

The G1 meshes and materials come from mjlab's viser bridge (``MjlabViserScene`` on
mjviser). Frames are rendered by a headless Chromium connected to a local viser server:
hardware WebGL through ANGLE/Vulkan on ``--gpu`` (``--software``: SwiftShader), at
PIXEL_RATIO times the frame size and downsampled. Each video frame is two renders,
without the ghost and with an opaque ghost, blended at GHOST_OPACITY; the interactive
page uses the browser's own transparency, where overlapping ghost parts can stack. Every
appearance setting (camera, colours, lights, floor, caption) is a constant under "Look"
below.

Outputs in ``<out-dir>/<clip>/<tag>/``, the tag from the run's name
``<date>_full[_<tag>][_fixstd]``: ``original`` for a plain ``_full`` run (the original
tracker), else ``<tag>`` (``vicfreq``, ``vicfreqdamp``, ``newrew``, ...):

- ``realtime.mp4``: 50 fps, the state at the end of every policy step (the frame times
  of the native videos);
- ``slow025.mp4``: 0.25x speed at 50 fps, one frame per physics substep;
- ``interactive/index.html``: viser's client with the recording embedded; works from any
  static file host or from disk, orbit with the mouse while it plays;
  ``recording.viser`` next to it is the same recording, for viser's stock client
  (``?playbackPath=recording.viser``);
- ``t<T>.png``: full-size stills at the ``--stills`` times; ``--only-stills`` renders
  only these;

and ``<rollouts-dir>/<clip>/<tag>.npz``: the rollout per substep (``qpos``, ``ctrl``,
``torque``, ``kp``, ``kd``, feet heights, ``substep_time``, policy ``step``, motion
``frame``) with the nominal gains and torque limits; ``--only-rollout`` writes only this
(no browser, no render), e.g. for ``plot_gains_torques.py``.

    PY=.venv/bin/python  # from the repo root
    $PY tools/render_viser.py --gpu 0 --chrome CHROME --clip amass_cmu_88_06 \\
        [--run 2026-10-21_00-32-59_full_vicfreq_fixstd] \\
        [--stills 0.8 1.9 3.0 [--only-stills]]
    $PY tools/render_viser.py --gpu 0 --only-rollout --clip amass_cmu_88_06 --run RUN

``CHROME`` is a Chromium headless shell or a wrapper script that execs one with
``--no-sandbox``. Get the shell with ``npx playwright install chromium-headless-shell``
(it lands under ``~/.cache/ms-playwright/chromium_headless_shell-*/``). On a machine
without ``libasound.so.2`` the wrapper puts an extracted copy of the library on
``LD_LIBRARY_PATH``:

    #!/bin/sh
    export LD_LIBRARY_PATH=/path/to/libasound${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    exec /path/to/chrome-headless-shell --no-sandbox "$@"
"""

from __future__ import annotations

import argparse
import base64
import copy
import os
import signal
import socket
import subprocess
import time
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import yaml
from paths import ROLLOUTS_DIR, RUNS_DIR, VIDEOS_DIR
from PIL import Image, ImageDraw, ImageFont
from render_action_ghost import SLOW_FACTOR, make_play_env
from render_force_ghost import nominal_cfg
from train_tracker import (
  TASK,
  ParamsLoader,
  resolve_checkpoint,
  run_motion_file,
  run_task_id,
)

if TYPE_CHECKING:
  import mujoco
  import viser

  from mjlab.viewer.viser.scene import MjlabViserScene

# Their heights (simulated and reference) go into the npz.
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")

# ---- Look ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080
# The browser renders at this multiple of the frame size; downsampling anti-aliases the
# edges.
PIXEL_RATIO = 2
CAMERA_FOV_DEG = 40.0  # vertical field of view
# Direction from the camera target to the camera, from world +x, counter-clockwise.
CAMERA_AZIMUTH_DEG = -60.0
CAMERA_ELEVATION_DEG = 10.0  # camera above the target
CAMERA_DISTANCE_M = 3.0
CAMERA_TARGET_HEIGHT_M = (
  0.65  # the camera looks at the low-passed root x/y at this height
)
FOLLOW_CUTOFF_HZ = (
  0.5  # zero-phase Butterworth low-pass of the root x/y the camera follows
)
# Initial camera of the interactive page, aimed at the middle of the root path.
INTERACTIVE_DISTANCE_M = 4.8
BACKGROUND_RGB = (244, 245, 247)  # also the fog colour the floor fades into
FLOOR_RGB = (160, 162, 165)  # unlit: the floor shows this colour exactly, bar shadows
# Section lines draw stronger than cell lines.
GRID_RGB, GRID_CELL_M, GRID_SECTION_M = (105, 107, 111), 0.25, 1.0
# Lines fade with the distance d from the camera as (1 - d / GRID_FADE_M) **
# GRID_FADE_STRENGTH (half at 6 m); the floor stays opaque until the fog has covered it.
GRID_FADE_M, GRID_FADE_STRENGTH = 100.0, 11.0
# Linear fog into BACKGROUND_RGB: the floor runs to the horizon without an edge.
FOG_NEAR_M, FOG_FAR_M = 5.0, 50.0
SHADOW_OPACITY = 0.5
# Image-based ambient light (not shown as background).
ENVIRONMENT_HDRI, ENVIRONMENT_INTENSITY = "studio", 0.3
# Directional, shines toward the origin; shadows.
KEY_LIGHT_POSITION, KEY_LIGHT_INTENSITY = (3.0, 1.0, 4.0), 2.4
# Directional, no shadows.
FILL_LIGHT_POSITION, FILL_LIGHT_INTENSITY = (-1.0, -3.0, 1.5), 0.3
GHOST_RGB, GHOST_OPACITY = (0.72, 0.12, 0.0), 0.4
# Ghost surfaces moved inward along their normals (no z-fighting where ghost = robot).
GHOST_INSET_M = 0.003
CAPTION_PX = HEIGHT // 32
CAPTION_RGB, CAPTION_GHOST_RGB = (55, 60, 68), (225, 95, 0)
# --------------------------------------------------------------------------------------

GPU_FLAGS = [
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--enable-gpu",
  "--ignore-gpu-blocklist",
]
SOFTWARE_FLAGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
NVIDIA_VULKAN_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"


def rollout(clip: str, log_root: Path, run: str | None) -> dict:
  """Play ``clip``'s tracker of run ``run`` (default: the newest run) once (nominal
  robot) and keep, after every physics substep, qpos, ctrl, the actuator torques, the
  gains in effect and the feet heights."""
  import torch

  from mjlab.envs.mdp.actions import JointGainAction
  from mjlab.tasks.tracking.mdp.commands import MotionCommand

  run_dir = None if run is None else log_root / clip / run
  checkpoint = resolve_checkpoint(
    Namespace(checkpoint=None, run_dir=run_dir, clip=clip, log_root=log_root)
  )
  if run is None and not checkpoint.parent.name.endswith("_full"):
    raise RuntimeError(
      f"{clip}: newest run {checkpoint.parent} is not a _full run; "
      "choose one with --run"
    )
  env_yaml = yaml.load(
    (checkpoint.parent / "params" / "env.yaml").read_text(), Loader=ParamsLoader
  )
  task = run_task_id(env_yaml)
  label = task.removeprefix(TASK).strip("-")  # "" for the original tracker, "Vic-Freq"
  tag = run_tag(checkpoint.parent.name)
  motion = run_motion_file(env_yaml, checkpoint.parent)
  motion_data = np.load(motion)
  num_frames = len(motion_data["joint_pos"])
  print(
    f"[viser] {clip}: {checkpoint} ({task}) on {motion} ({num_frames} frames)",
    flush=True,
  )

  vec_env, policy = make_play_env(checkpoint, motion, nominal_cfg, task=task)
  env = vec_env.unwrapped
  sim = env.sim
  robot = env.scene["robot"]
  feet = [robot.body_names.index(name) for name in FEET]
  feet_body_ids = robot.indexing.body_ids[feet].cpu().numpy()
  ref_feet_height = motion_data["body_pos_w"][:, feet, 2]  # (F, 2)
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  gain_term = (
    cast(JointGainAction, env.action_manager.get_term("joint_gain"))
    if "joint_gain" in env.action_manager.active_terms
    else None
  )
  gain_err = 0.0  # gains read from the simulator's model vs the gain term's own kp / kd
  sub_qpos, sub_ctrl, sub_torque, sub_kp, sub_kd, sub_feet, sub_frame = (
    [] for _ in range(7)
  )
  sim_step = sim.step  # the env's decimation loop calls sim.step once per substep

  def step_and_keep() -> None:
    sim_step()
    sub_qpos.append(sim.data.qpos[0].cpu().numpy().astype(np.float64))
    sub_ctrl.append(sim.data.ctrl[0].cpu().numpy().astype(np.float64))
    sub_torque.append(sim.data.actuator_force[0].cpu().numpy().astype(np.float64))
    # The position actuators' gains in effect: force = kp * ctrl - kp * q - kd * qdot.
    sub_kp.append(sim.model.actuator_gainprm[0, :, 0].cpu().numpy().astype(np.float64))
    sub_kd.append(-sim.model.actuator_biasprm[0, :, 2].cpu().numpy().astype(np.float64))
    sub_feet.append(sim.data.xpos[0, feet_body_ids, 2].cpu().numpy().astype(np.float64))

  sim.step = step_and_keep
  step_qpos, terminated = [], []
  obs = vec_env.get_observations()
  # Step num_frames - 1 would wrap the motion and teleport to frame 0.
  for step in range(1, num_frames - 1):
    with torch.no_grad():
      obs, _, dones, _ = vec_env.step(policy(obs))
    if bool(dones[0]):
      terminated.append(step)
    step_qpos.append(sim.data.qpos[0].cpu().numpy().astype(np.float64))
    motion_frame = int(command.time_steps[0])
    sub_frame += [motion_frame] * env.cfg.decimation
    if gain_term is not None:
      ctrl_ids = gain_term.ctrl_ids.cpu().numpy()
      gain_err = max(
        gain_err,
        np.abs(gain_term.kp[0].cpu().numpy() - sub_kp[-1][ctrl_ids]).max(),
        np.abs(gain_term.kd[0].cpu().numpy() - sub_kd[-1][ctrl_ids]).max(),
      )
  model = copy.deepcopy(sim.mj_model)
  actuator_joints = [
    model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)
  ]
  states = {
    "model": model,
    "free_q_adr": robot.indexing.free_joint_q_adr.cpu().numpy(),
    "sub_qpos": np.array(sub_qpos),
    "sub_ctrl": np.array(sub_ctrl),
    "step_qpos": np.array(step_qpos),
    "sub_torque": np.array(sub_torque),
    "sub_kp": np.array(sub_kp),
    "sub_kd": np.array(sub_kd),
    "sub_feet_height": np.array(sub_feet),
    "sub_ref_feet_height": ref_feet_height[sub_frame],
    "sub_frame": np.array(sub_frame),
    "kp0": model.actuator_gainprm[:, 0].copy(),
    "kd0": -model.actuator_biasprm[:, 2].copy(),
    "torque_limit": model.actuator_forcerange[:, 1].copy(),
    "actuator_joints": actuator_joints,
    "physics_dt": env.physics_dt,
    "step_dt": env.step_dt,
    "decimation": env.cfg.decimation,
    "terminated": terminated,
    "task": task,
    "label": label,
    "tag": tag,
    "checkpoint": str(checkpoint),
    "motion_file": str(motion),
  }
  if not model.actuator_forcelimited.all():
    raise RuntimeError(
      "expected force-limited actuators (torque_limit = actuator_forcerange[:, 1])"
    )
  if gain_term is not None:
    print(
      f"[viser] gain term {gain_term.cfg.mode}: "
      f"|model gains - term gains| max {gain_err:.2e}",
      flush=True,
    )
    if gain_err > 1e-4:
      raise RuntimeError("the gains read from the simulator's model are not the term's")
  else:  # the simulator's gains are float32, the compiled model's float64
    gain_err = max(
      np.abs(states["sub_kp"] / states["kp0"] - 1).max(),
      np.abs(states["sub_kd"] / states["kd0"] - 1).max(),
    )
    print(f"[viser] no gain term: |gains / nominal - 1| max {gain_err:.2e}", flush=True)
    if gain_err > 1e-5:
      raise RuntimeError(
        "the gains changed during the rollout of a task without a gain term"
      )
  vec_env.close()
  return states


def run_tag(run: str) -> str:
  """Output tag of run directory ``run`` (``<date>_full[_<tag>][_fixstd]``):
  ``original`` for a plain ``_full`` run, else ``<tag>``."""
  _, full, tag = run.partition("_full")
  if not full:
    raise RuntimeError(f"run {run} is not a _full run")
  return tag.removeprefix("_").removesuffix("_fixstd") or "original"


def save_rollout(states: dict, path: Path) -> None:
  """Write the rollout arrays (per substep; the ``sub_`` prefix dropped) and its
  provenance to ``path`` (npz)."""
  num_substeps = len(states["sub_qpos"])
  arrays = {
    key.removeprefix("sub_"): value
    for key, value in states.items()
    if key.startswith("sub_")
  }
  provenance = (
    "kp0", "kd0", "torque_limit", "actuator_joints", "physics_dt", "step_dt",
    "decimation", "task", "label", "tag", "checkpoint", "motion_file",
  )  # fmt: skip
  arrays.update(
    substep_time=(np.arange(num_substeps) + 1) * states["physics_dt"],
    step=np.arange(num_substeps) // states["decimation"] + 1,
    step_qpos=states["step_qpos"],
    terminated=np.array(states["terminated"], dtype=int),
    **{key: states[key] for key in provenance},
    feet=np.array(FEET),
  )
  np.savez(path, **arrays)
  print(f"[viser] rollout written to {path}", flush=True)


def check_rollout(clip: str, states: dict) -> None:
  """Print how far this rollout is from the one behind the native videos
  (``render_force_ghost.py``)."""
  decimation = states["decimation"]
  own = np.abs(
    states["sub_qpos"][decimation - 1 :: decimation] - states["step_qpos"]
  ).max()
  print(
    f"[viser] terminations {states['terminated']}; "
    f"|last substep - step end qpos| max {own:.2e}",
    flush=True,
  )
  native = ROLLOUTS_DIR / clip / "native.npz"
  if states["task"] != TASK:
    print(
      f"[viser] {states['task']}: pose check against the native videos "
      "(original tracker) skipped",
      flush=True,
    )
    return
  if not native.exists():
    print(
      f"[viser] no {native}: pose check against the native videos skipped", flush=True
    )
    return
  with np.load(native) as old:
    qpos_diff = np.abs(states["step_qpos"] - old["step_qpos"])
    ctrl_diff = np.abs(
      states["sub_ctrl"][decimation - 1 :: decimation] - old["step_ctrl"]
    ).max()
    # The native record keeps the state at the start of each substep, this one the
    # state at its end.
    sub_diff = np.abs(states["sub_qpos"][:-1] - old["sub_qpos"][1:]).max()
  root = qpos_diff[:, states["free_q_adr"][:3]].max()
  print(
    f"[viser] vs {native.name}: step qpos max |diff| {qpos_diff.max():.2e} "
    f"(root position {root:.2e} m), substep qpos {sub_diff:.2e}, "
    f"step ctrl {ctrl_diff:.2e}",
    flush=True,
  )


def camera_path(states: dict) -> tuple[np.ndarray, np.ndarray]:
  """Per substep frame the camera position (N, 3) and target (N, 3): a fixed offset
  from the low-passed root x/y."""
  from scipy.signal import butter, filtfilt

  b, a = cast(
    tuple[np.ndarray, np.ndarray],
    butter(2, FOLLOW_CUTOFF_HZ, fs=1.0 / states["physics_dt"]),
  )
  xy = filtfilt(b, a, states["sub_qpos"][:, states["free_q_adr"][:2]], axis=0)
  target = np.column_stack([xy, np.full(len(xy), CAMERA_TARGET_HEIGHT_M)])
  return target + CAMERA_DISTANCE_M * view_direction(), target


def view_direction() -> np.ndarray:
  """Unit vector (3,) from the camera target to the camera."""
  azimuth = np.radians(CAMERA_AZIMUTH_DEG)
  elevation = np.radians(CAMERA_ELEVATION_DEG)
  return np.array(
    [
      np.cos(elevation) * np.cos(azimuth),
      np.cos(elevation) * np.sin(azimuth),
      np.sin(elevation),
    ]
  )


def look_at_wxyz(position: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Camera orientation (4,) wxyz, viser's OpenCV convention (x right, y down, z
  forward), world z up."""
  import viser.transforms as vtf

  forward = (target - position) / np.linalg.norm(target - position)
  right = np.cross(forward, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)
  rotation = np.column_stack([right, np.cross(forward, right), forward])
  return vtf.SO3.from_matrix(rotation).wxyz


def ghost_model_of(model: mujoco.MjModel) -> mujoco.MjModel:
  """Copy of ``model`` drawn as the ghost: collision geoms hidden, visual geoms
  GHOST_RGB, meshes inset."""
  ghost = copy.deepcopy(model)
  collision = (ghost.geom_contype != 0) | (ghost.geom_conaffinity != 0)
  ghost.geom_rgba[collision, 3] = 0.0
  ghost.geom_rgba[~collision] = (*GHOST_RGB, GHOST_OPACITY)
  for mesh in range(ghost.nmesh):
    vert_adr, face_adr = ghost.mesh_vertadr[mesh], ghost.mesh_faceadr[mesh]
    verts = slice(vert_adr, vert_adr + ghost.mesh_vertnum[mesh])
    faces = ghost.mesh_face[face_adr : face_adr + ghost.mesh_facenum[mesh]]
    corners = ghost.mesh_vert[verts][faces]  # (F, 3, 3)
    # Area-weighted.
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal = np.zeros((ghost.mesh_vertnum[mesh], 3))
    np.add.at(normal, faces.ravel(), np.repeat(face_normal, 3, axis=0))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    ghost.mesh_vert[verts] -= GHOST_INSET_M * normal
  return ghost


def build_scene(
  api: viser.ViserServer | viser.ClientHandle, model: mujoco.MjModel
) -> MjlabViserScene:
  """The G1 (mjlab's viser bridge), lights and floor, added to ``api``'s scene (a
  server or one client)."""
  from mjlab.viewer.viser.scene import MjlabViserScene

  # mjviser edits the model's vis settings. It types its server as a ViserServer; the
  # calls made here (build, add_ghost_mesh, update_from_mjdata, clear) only use the
  # scene / atomic / flush API that a ClientHandle shares.
  scene = MjlabViserScene(api, copy.deepcopy(model), num_envs=1)  # pyright: ignore[reportArgumentType]
  # The world stays put; the video camera follows the robot instead.
  scene.camera_tracking_enabled = False
  scene.debug_visualization_enabled = True  # draws the queued ghosts
  # mjviser's floor (the terrain plane); "/floor" below replaces it.
  scene.fixed_bodies_frame.visible = False
  # mjviser decimates meshes seen from beyond 2 m (faceted head and hands).
  for group in scene._mesh_groups:
    group.handle.lod = "off"
  api.scene.configure_environment_map(
    ENVIRONMENT_HDRI, environment_intensity=ENVIRONMENT_INTENSITY
  )
  api.scene.configure_default_lights(enabled=False)
  api.scene.add_light_directional(
    "/lights/key",
    position=KEY_LIGHT_POSITION,
    intensity=KEY_LIGHT_INTENSITY,
    cast_shadow=True,
  )
  api.scene.add_light_directional(
    "/lights/fill", position=FILL_LIGHT_POSITION, intensity=FILL_LIGHT_INTENSITY
  )
  # Two grids, the floor and its lines, so that the lines can fade out sooner. Same
  # origin and size: the browser's back-to-front sort of transparent objects then ties
  # and keeps the order they were added in, and the lines lie at exactly the floor's
  # depth.
  floor = {"plane": "xy", "infinite_grid": True, "fade_distance": GRID_FADE_M}
  api.scene.add_grid(
    "/floor",
    **floor,
    cell_thickness=0.0,
    section_thickness=0.0,
    fade_strength=0.1,
    shadow_opacity=SHADOW_OPACITY,
    plane_color=FLOOR_RGB,
    plane_opacity=1.0,
  )
  api.scene.add_grid(
    "/floor_lines",
    **floor,
    cell_color=GRID_RGB,
    cell_size=GRID_CELL_M,
    section_color=GRID_RGB,
    section_size=GRID_SECTION_M,
    fade_strength=GRID_FADE_STRENGTH,
    shadow_opacity=0.0,
  )
  # three.js takes the fog colour as linear RGB and shows it sRGB-encoded.
  fog_rgb = ((np.array(BACKGROUND_RGB) / 255 + 0.055) / 1.055) ** 2.4
  api.scene.configure_fog(FOG_NEAR_M, FOG_FAR_M, color=tuple(fog_rgb))
  return scene


def show(
  scene: MjlabViserScene,
  data: mujoco.MjData,
  ghost: mujoco.MjModel,
  qpos: np.ndarray,
  ctrl: np.ndarray,
  ctrl_q_adr: np.ndarray,
  ghost_opacity: float | None,
) -> None:
  """Pose the robot at ``qpos`` (nq,) and the ghost (unless ``ghost_opacity`` is None)
  at the same root with the joints at ``ctrl`` (nu,)."""
  import mujoco

  data.qpos[:] = qpos
  mujoco.mj_kinematics(scene.mj_model, data)
  scene.clear()
  if ghost_opacity is not None:
    ghost_qpos = qpos.copy()
    ghost_qpos[ctrl_q_adr] = ctrl
    scene.add_ghost_mesh(ghost_qpos, ghost, alpha=ghost_opacity, label="targets")
  scene.update_from_mjdata(data)


def caption(frame: np.ndarray, title: str, legend: str) -> np.ndarray:
  """``frame`` (H, W, 3) uint8 with ``title`` at the top-left and ``legend`` (ghost
  colour) under it, both over the background (the grey floor would swallow the
  orange)."""
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default(size=CAPTION_PX)
  margin = CAPTION_PX
  draw.text((margin, margin), title, font=font, fill=CAPTION_RGB)
  draw.text(
    (margin, margin + 1.5 * CAPTION_PX), legend, font=font, fill=CAPTION_GHOST_RGB
  )
  return np.asarray(image)


class HeadlessBrowser:
  """A headless Chromium showing the viser server's page; ``client`` is its viser
  client. Exiting kills it."""

  def __init__(
    self,
    server: viser.ViserServer,
    chrome: str,
    gpu: int | None,
    timeout: float = 60.0,
  ) -> None:
    self.server, self.chrome, self.gpu, self.timeout = server, chrome, gpu, timeout
    self.client: viser.ClientHandle | None = None
    self._process: subprocess.Popen | None = None

  def __enter__(self) -> HeadlessBrowser:
    env = dict(os.environ)
    if self.gpu is not None:
      # Chromium's Vulkan takes the first device; Mesa's device-select layer puts the
      # GPU on this PCI bus first.
      bus = subprocess.check_output(
        [
          "nvidia-smi",
          "-i",
          str(self.gpu),
          "--query-gpu=pci.bus_id",
          "--format=csv,noheader",
        ],
        text=True,
      ).strip()  # e.g. 00000000:2B:00.0
      domain, bus_number, device_function = bus.lower().split(":")
      device, function = device_function.split(".")
      env.update(
        VK_ICD_FILENAMES=NVIDIA_VULKAN_ICD,
        MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE="1",
        DRI_PRIME=f"pci-{domain[-4:]}_{bus_number}_{device}_{function}",
      )
    url = f"http://127.0.0.1:{self.server.get_port()}/?fixedDpr={PIXEL_RATIO}"
    command = [
      self.chrome, "--headless", f"--window-size={WIDTH},{HEIGHT}",
      "--hide-scrollbars", "--remote-debugging-port=0",
      *(SOFTWARE_FLAGS if self.gpu is None else GPU_FLAGS), url,
    ]  # fmt: skip
    known = set(self.server.get_clients())
    # Own process group, so that exiting also kills the GPU and renderer processes.
    self._process = subprocess.Popen(
      command,
      env=env,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
    deadline = time.monotonic() + self.timeout
    while self.client is None:
      new = [
        client for key, client in self.server.get_clients().items() if key not in known
      ]
      if new:
        self.client = new[0]
      elif self._process.poll() is not None or time.monotonic() > deadline:
        self.__exit__()
        raise RuntimeError(
          f"headless Chromium did not connect to {url} "
          f"(exit code {self._process.returncode})"
        )
      else:
        time.sleep(0.05)
    return self

  def __exit__(self, *exc: object) -> None:
    if self._process is not None and self._process.poll() is None:
      os.killpg(self._process.pid, signal.SIGKILL)
      self._process.wait()

  def render(
    self, position: np.ndarray, target: np.ndarray, background: Image.Image
  ) -> np.ndarray:
    """Frame (H, W, 3) uint8 seen from ``position`` (3,) towards ``target`` (3,), over
    ``background`` (RGBA)."""
    assert self.client is not None  # set by __enter__
    fov = np.radians(CAMERA_FOV_DEG)
    # The browser's own camera goes to the same pose: the shadow cascades are fitted
    # to it.
    with self.client.atomic():
      self.client.camera.fov = fov
      self.client.camera.position = position
      self.client.camera.look_at = target
    rgba = self.client.get_render(
      HEIGHT,
      WIDTH,
      wxyz=look_at_wxyz(position, target),
      position=position,
      fov=fov,
      transport_format="png",
    )
    # The render is transparent where nothing is drawn.
    return np.asarray(Image.alpha_composite(background, Image.fromarray(rgba)))[..., :3]


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def render_videos(
  clip: str, states: dict, server: viser.ViserServer, args: Namespace
) -> None:
  """Render the substep frames in a headless browser; write both videos and the
  stills."""
  import imageio_ffmpeg
  import mediapy
  import mujoco

  # The ffmpeg shipped with imageio (there may be none on PATH).
  mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  model, decimation = states["model"], states["decimation"]
  physics_dt = states["physics_dt"]
  # qpos address of each actuator's joint.
  ctrl_q_adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
  ghost = ghost_model_of(model)
  data = mujoco.MjData(model)
  positions, targets = camera_path(states)
  background = Image.new("RGBA", (WIDTH, HEIGHT), (*BACKGROUND_RGB, 255))
  num_frames = len(states["sub_qpos"])
  stills = {round(t / physics_dt) - 1: t for t in args.stills}
  frames = sorted(stills) if args.only_stills else range(num_frames)

  def frame_at(browser: HeadlessBrowser, scene: MjlabViserScene, i: int) -> np.ndarray:
    # The ghost's transparency is blended here, from a render with an opaque ghost and
    # one without it: one ghost layer (its nearest surface) over whatever is behind it,
    # free of the browser's per-mesh sorting.
    qpos, ctrl = states["sub_qpos"][i], states["sub_ctrl"][i]
    show(scene, data, ghost, qpos, ctrl, ctrl_q_adr, None)
    plain = browser.render(positions[i], targets[i], background).astype(np.float32)
    show(scene, data, ghost, qpos, ctrl, ctrl_q_adr, 1.0)
    opaque = browser.render(positions[i], targets[i], background).astype(np.float32)
    return (plain + GHOST_OPACITY * (opaque - plain)).round().astype(np.uint8)

  start = time.monotonic()
  with ExitStack() as stack:
    browser = stack.enter_context(
      HeadlessBrowser(server, args.chrome, None if args.software else args.gpu)
    )
    # The client's scene, not the server's: its updates and the render requests then
    # reach the browser in order.
    assert browser.client is not None  # set by __enter__
    scene = build_scene(browser.client, model)
    previous = frame_at(browser, scene, 0)
    # Meshes load asynchronously in the browser: wait until two renders agree.
    for _ in range(40):
      time.sleep(0.25)
      current = frame_at(browser, scene, 0)
      if np.array_equal(current, previous):
        break
      previous = current
    else:
      raise RuntimeError("the browser's render of frame 0 did not settle")
    print(f"[viser] browser ready after {time.monotonic() - start:.1f} s", flush=True)

    out_dir = args.out_dir / clip / states["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    realtime_video = slow_video = None  # stay None with --only-stills
    if not args.only_stills:
      kwargs = {"shape": (HEIGHT, WIDTH), "fps": 1.0 / states["step_dt"], "crf": 18}
      realtime_video = stack.enter_context(
        mediapy.VideoWriter(out_dir / "realtime.mp4", **kwargs)
      )
      slow_video = stack.enter_context(
        mediapy.VideoWriter(out_dir / "slow025.mp4", **kwargs)
      )
    start = time.monotonic()
    title = f"{clip}   {states['label']}" if states["label"] else clip
    legend = "orange = commanded joint targets"
    for i in frames:
      frame = frame_at(browser, scene, i)
      t = (i + 1) * physics_dt
      realtime = caption(frame, f"{title}   t = {t:4.2f} s", legend)
      if i in stills:
        Image.fromarray(realtime).save(out_dir / f"t{stills[i]:.2f}.png")
      if realtime_video is None or slow_video is None:
        continue
      if (i + 1) % decimation == 0:
        realtime_video.add_image(realtime)
      slow_title = f"{title}   t = {t:5.3f} s   ({SLOW_FACTOR:g}x speed)"
      slow_video.add_image(caption(frame, slow_title, legend))
  elapsed = time.monotonic() - start
  print(
    f"[viser] {len(frames)} frames in {elapsed:.0f} s "
    f"({elapsed / max(len(frames), 1):.2f} s per frame)",
    flush=True,
  )


def record_interactive(
  clip: str, states: dict, server: viser.ViserServer, out_dir: Path
) -> None:
  """Replay the policy-step frames into a viser recording; write it and a page that
  plays it."""
  import mujoco
  import viser

  deadline = time.monotonic() + 10.0
  # A connected browser would show (and keep) the scene built below.
  while server.get_clients():
    if time.monotonic() > deadline:
      raise RuntimeError("a browser is still connected to the viser server")
    time.sleep(0.1)
  model, decimation = states["model"], states["decimation"]
  ctrl_q_adr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
  ghost = ghost_model_of(model)
  data = mujoco.MjData(model)
  scene = build_scene(server, model)
  # In the videos the background is composited after the render.
  server.scene.set_background_image(
    np.full((4, 4, 3), BACKGROUND_RGB, dtype=np.uint8), format="png"
  )
  center = np.append(
    states["sub_qpos"][:, states["free_q_adr"][:2]].mean(axis=0),
    CAMERA_TARGET_HEIGHT_M,
  )
  server.initial_camera.fov = np.radians(CAMERA_FOV_DEG)
  server.initial_camera.look_at = center
  server.initial_camera.position = center + INTERACTIVE_DISTANCE_M * view_direction()
  serializer = server.get_scene_serializer()  # holds the scene built so far
  for i in range(decimation - 1, len(states["sub_qpos"]), decimation):
    qpos, ctrl = states["sub_qpos"][i], states["sub_ctrl"][i]
    show(scene, data, ghost, qpos, ctrl, ctrl_q_adr, GHOST_OPACITY)
    serializer.insert_sleep(states["step_dt"])
  recording = serializer.serialize()

  folder = out_dir / clip / states["tag"] / "interactive"
  folder.mkdir(parents=True, exist_ok=True)
  (folder / "recording.viser").write_bytes(recording)
  # As viser's StateSerializer.as_html: the client build with the recording injected,
  # which it plays on load.
  client_html = (
    Path(viser.__file__).parent / "client" / "build" / "index.html"
  ).read_text()
  embedded = base64.b64encode(recording).decode("ascii")
  inject = (
    f'<script>window.__VISER_EMBED_DATA__="{embedded}";'
    "window.__VISER_EMBED_CONFIG__={darkMode:false};</script>"
  )
  head_end = client_html.index("</head>")
  (folder / "index.html").write_text(
    client_html[:head_end] + inject + client_html[head_end:]
  )
  print(
    f"[viser] {folder}: recording {len(recording) / 1e6:.1f} MB, "
    f"index.html {(folder / 'index.html').stat().st_size / 1e6:.1f} MB",
    flush=True,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--gpu", type=int, default=0, help="simulation and WebGL GPU")
  parser.add_argument(
    "--chrome",
    help="Chromium headless shell, or a wrapper that execs one (not --only-rollout)",
  )
  parser.add_argument("--clip", default="amass_cmu_88_06")
  parser.add_argument(
    "--run",
    help="run directory name under <log-root>/<clip> (default: the newest _full run)",
  )
  parser.add_argument(
    "--software", action="store_true", help="render with SwiftShader instead of the GPU"
  )
  parser.add_argument(
    "--stills", type=float, nargs="*", default=[], help="times (s) of full-size stills"
  )
  parser.add_argument(
    "--only-stills",
    action="store_true",
    help="render only the stills (to tune the look)",
  )
  parser.add_argument(
    "--only-rollout", action="store_true", help="write only the rollout npz (no render)"
  )
  parser.add_argument("--log-root", type=Path, default=RUNS_DIR)
  parser.add_argument(
    "--out-dir",
    type=Path,
    default=VIDEOS_DIR,
    help="videos, stills and interactive page",
  )
  parser.add_argument(
    "--rollouts-dir", type=Path, default=ROLLOUTS_DIR, help="the rollout npz"
  )
  args = parser.parse_args()
  if args.chrome is None and not args.only_rollout:
    parser.error("--chrome is required unless --only-rollout")
  # Set before mjlab imports torch / mujoco.
  os.environ.update(
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_EGL_DEVICE_ID=str(args.gpu),
    MUJOCO_GL="egl",
    WANDB_MODE="disabled",
  )
  import viser

  states = rollout(args.clip, args.log_root, args.run)
  check_rollout(args.clip, states)
  (args.rollouts_dir / args.clip).mkdir(parents=True, exist_ok=True)
  save_rollout(states, args.rollouts_dir / args.clip / f"{states['tag']}.npz")
  if args.only_rollout:
    return
  server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
  try:
    render_videos(args.clip, states, server, args)
    if not args.only_stills:
      record_interactive(args.clip, states, server, args.out_dir)
  finally:
    server.stop()


if __name__ == "__main__":
  main()
