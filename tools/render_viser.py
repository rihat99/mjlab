"""Video engine of ``tools/view.py --video``: a SuperMimic rollout of one clip,
rendered through viser in a headless browser.

The rollout (``rollout``) is ``evaluate_library.py``'s protocol on this one clip: the
task's play config (no observation noise, no randomization, foot friction 1.0, no
reset noise, clips start at frame 0) with the clip as a one-clip library, 1 env, the
checkpoint loaded with the task's runner (actor only), its deterministic policy,
``env.reset()``, then steps until the clip's ``motion_end`` time-out or a fall. Kept
after every physics substep (200 Hz): the robot pose and the actuator targets
``data.ctrl``; per policy step: the reward and the reference frame the step is scored
against (``MotionCommand.time_steps`` before the step; step k scores frame k), whose
root pose and joint angles are the reference ghost. Both ghosts hold over the 4
substeps of a policy step.

The G1 meshes and materials come from mjlab's viser bridge (``MjlabViserScene`` on
mjviser). Frames are rendered by a headless Chromium connected to a local viser
server: hardware WebGL through ANGLE/Vulkan on a given GPU (or SwiftShader), at
PIXEL_RATIO times the frame size and downsampled, one frame per policy step (50 fps,
the state at the end of the step). A ghost G1 (``ghost_model_of``) is drawn in one
colour with its meshes moved GHOST_INSET_M inward, so where ghost and robot coincide
the ghost hides inside the robot instead of z-fighting with it. Each frame blends a
render without the ghost and one with an opaque ghost at GHOST_OPACITY: one ghost
layer over whatever is behind it, free of the browser's per-mesh sorting. Every
appearance setting (camera, colours, lights, floor, caption) is a constant under
"Look" below.

The browser is Playwright's Chromium headless shell (``find_chrome``) or any Chromium
given by path, started with ``--no-sandbox``.
"""

from __future__ import annotations

import copy
import os
import shutil
import signal
import socket
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
  import mujoco
  import viser

  from mjlab.viewer.viser.scene import MjlabViserScene

# ---- Look ----------------------------------------------------------------------------
WIDTH, HEIGHT = 1920, 1080
# The browser renders at this multiple of the frame size; downsampling anti-aliases.
PIXEL_RATIO = 2
CAMERA_FOV_DEG = 40.0  # vertical field of view
# Direction from the camera target to the camera, from world +x, counter-clockwise.
CAMERA_AZIMUTH_DEG = -60.0
CAMERA_ELEVATION_DEG = 10.0  # camera above the target
CAMERA_DISTANCE_M = 3.0
CAMERA_TARGET_HEIGHT_M = 0.65  # the camera looks at the low-passed root x/y this high
FOLLOW_CUTOFF_HZ = 0.5  # zero-phase Butterworth low-pass of the root x/y it follows
BACKGROUND_RGB = (244, 245, 247)  # also the fog colour the floor fades into
FLOOR_RGB = (160, 162, 165)  # unlit: the floor shows this colour, darkened by shadows
# Section lines draw stronger than cell lines.
GRID_RGB, GRID_CELL_M, GRID_SECTION_M = (105, 107, 111), 0.25, 1.0
# Lines fade with the distance d from the camera as
# (1 - d / GRID_FADE_M) ** GRID_FADE_STRENGTH (half at 6 m); the floor stays opaque
# until the fog has covered it.
GRID_FADE_M, GRID_FADE_STRENGTH = 100.0, 11.0
# Linear fog into BACKGROUND_RGB: the floor runs to the horizon without an edge.
FOG_NEAR_M, FOG_FAR_M = 5.0, 50.0
SHADOW_OPACITY = 0.5
# Image-based ambient light (not shown as background).
ENVIRONMENT_HDRI, ENVIRONMENT_INTENSITY = "studio", 0.3
# Directional, shines toward the origin; casts the shadows.
KEY_LIGHT_POSITION, KEY_LIGHT_INTENSITY = (3.0, 1.0, 4.0), 2.4
# Directional, no shadows.
FILL_LIGHT_POSITION, FILL_LIGHT_INTENSITY = (-1.0, -3.0, 1.5), 0.3
GHOST_RGB, GHOST_OPACITY = (0.72, 0.12, 0.0), 0.4
GHOST_INSET_M = 0.003  # ghost surfaces moved inward along their normals
CAPTION_PX = HEIGHT // 32
CAPTION_RGB, CAPTION_GHOST_RGB = (55, 60, 68), (225, 95, 0)
# Blue-grey, deep enough to read at GHOST_OPACITY.
REFERENCE_RGB, CAPTION_REFERENCE_RGB = (0.12, 0.34, 0.72), (30, 90, 190)
GHOSTS = {  # ghost: (ghost colour, legend colour, legend)
  "reference": (REFERENCE_RGB, CAPTION_REFERENCE_RGB, "blue = reference motion"),
  "targets": (GHOST_RGB, CAPTION_GHOST_RGB, "orange = commanded joint targets"),
}
# --------------------------------------------------------------------------------------

WINDOW_S = 20.0  # rendered seconds (the whole clip if shorter)

GPU_FLAGS = [
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--enable-gpu",
  "--ignore-gpu-blocklist",
]
SOFTWARE_FLAGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
NVIDIA_VULKAN_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
# Playwright's browser cache unless PLAYWRIGHT_BROWSERS_PATH is set, and the headless
# shell's path in it.
PLAYWRIGHT_CACHE = Path.home() / ".cache" / "ms-playwright"
PLAYWRIGHT_SHELL = (
  "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
)
CHROME_INSTALL = "npx playwright install chromium-headless-shell"
LIBASOUND_RECIPE = """\
Chromium needs libasound.so.2, which this machine lacks. Install it
(sudo apt install libasound2t64; libasound2 before Ubuntu 24.04), or without root:

  cd $(mktemp -d) && apt-get download libasound2t64
  dpkg-deb -x libasound2t64_*.deb ~/.local/libasound
  export LD_LIBRARY_PATH=~/.local/libasound/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

and run again."""


def find_chrome() -> str:
  """The newest Chromium headless shell of the Playwright browser cache."""
  cache = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", PLAYWRIGHT_CACHE))
  shells = sorted(
    cache.glob(PLAYWRIGHT_SHELL), key=lambda path: int(path.parts[-3].split("-")[-1])
  )
  if not shells:
    raise SystemExit(
      f"no Chromium headless shell in {cache}: install one with `{CHROME_INSTALL}` "
      "or pass --chrome"
    )
  return str(shells[-1])


def check_chrome(chrome: str) -> None:
  """Exit with a clear message (and the libasound recipe) if ``chrome`` cannot run."""
  if shutil.which(chrome) is None:
    raise SystemExit(f"{chrome} is not an executable")
  result = subprocess.run(
    [chrome, "--no-sandbox", "--version"], capture_output=True, text=True, timeout=60
  )
  if result.returncode == 0:
    return
  if "libasound.so.2" in result.stderr:
    raise SystemExit(LIBASOUND_RECIPE)
  raise SystemExit(f"{chrome} --version failed: {result.stderr.strip()}")


def camera_path(states: dict) -> tuple[np.ndarray, np.ndarray]:
  """Camera position (N, 3) and target (N, 3) per substep frame of ``states``.

  A fixed offset from the low-passed root x/y; ``states`` holds ``sub_qpos`` (N, nq),
  ``free_q_adr`` and ``physics_dt``.
  """
  from scipy.signal import butter, filtfilt

  # scipy has no type stubs; butter returns (b, a) with its default output="ba".
  b, a = cast(
    "tuple[np.ndarray, np.ndarray]",
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
  """Camera orientation (4,) wxyz, viser's OpenCV convention, world z up.

  OpenCV convention: x right, y down, z forward.
  """
  import viser.transforms as vtf

  forward = (target - position) / np.linalg.norm(target - position)
  right = np.cross(forward, [0.0, 0.0, 1.0])
  right /= np.linalg.norm(right)
  rotation = np.column_stack([right, np.cross(forward, right), forward])
  return vtf.SO3.from_matrix(rotation).wxyz


def ghost_model_of(
  model: mujoco.MjModel, rgb: tuple[float, float, float] = GHOST_RGB
) -> mujoco.MjModel:
  """Copy of ``model`` drawn as the ghost.

  Collision geoms hidden, visual geoms ``rgb``, meshes inset by GHOST_INSET_M.
  """
  ghost = copy.deepcopy(model)
  collision = (ghost.geom_contype != 0) | (ghost.geom_conaffinity != 0)
  ghost.geom_rgba[collision, 3] = 0.0
  ghost.geom_rgba[~collision] = (*rgb, GHOST_OPACITY)
  for mesh in range(ghost.nmesh):
    first_vert = ghost.mesh_vertadr[mesh]
    verts = slice(first_vert, first_vert + ghost.mesh_vertnum[mesh])
    first_face = ghost.mesh_faceadr[mesh]
    faces = ghost.mesh_face[first_face : first_face + ghost.mesh_facenum[mesh]]
    corners = ghost.mesh_vert[verts][faces]  # (F, 3, 3)
    # Area-weighted face normals.
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normal = np.zeros((ghost.mesh_vertnum[mesh], 3))
    np.add.at(normal, faces.ravel(), np.repeat(face_normal, 3, axis=0))
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    ghost.mesh_vert[verts] -= GHOST_INSET_M * normal
  return ghost


def build_scene(
  api: viser.ViserServer | viser.ClientHandle, model: mujoco.MjModel
) -> MjlabViserScene:
  """The G1 (mjlab's viser bridge), lights and floor, added to ``api``'s scene.

  ``api`` is a server or one client.
  """
  from mjlab.viewer.viser.scene import MjlabViserScene

  # mjviser edits the model's vis settings. MjlabViserScene is annotated for a server;
  # outside its GUI it uses only ``scene``, ``atomic`` and ``flush``, which a client
  # has too.
  scene = MjlabViserScene(
    cast("viser.ViserServer", api), copy.deepcopy(model), num_envs=1
  )
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


def caption(
  frame: np.ndarray,
  title: str,
  legend: str,
  legend_rgb: tuple[int, int, int] = CAPTION_GHOST_RGB,
) -> np.ndarray:
  """``frame`` (H, W, 3) uint8 with ``title`` at the top-left and ``legend`` under it.

  The legend is drawn in ``legend_rgb`` (the ghost colour), both over the background
  (the grey floor would swallow the orange).
  """
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  font = ImageFont.load_default(size=CAPTION_PX)
  margin = CAPTION_PX
  draw.text((margin, margin), title, font=font, fill=CAPTION_RGB)
  draw.text((margin, margin + 1.5 * CAPTION_PX), legend, font=font, fill=legend_rgb)
  return np.asarray(image)


class HeadlessBrowser:
  """A headless Chromium showing the viser server's page.

  ``client`` is its viser client. Exiting kills the browser.
  """

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
      self.chrome,
      "--no-sandbox",
      "--headless",
      f"--window-size={WIDTH},{HEIGHT}",
      "--hide-scrollbars",
      "--remote-debugging-port=0",
      *(SOFTWARE_FLAGS if self.gpu is None else GPU_FLAGS),
      url,
    ]
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
    """Frame (H, W, 3) uint8 seen from ``position`` (3,) towards ``target`` (3,).

    Composited over ``background`` (RGBA).
    """
    assert self.client is not None, "enter the HeadlessBrowser before render()"
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
  """A free local TCP port."""
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def rollout(task: str, checkpoint: Path, motion: Path) -> dict:
  """Play ``motion`` once from frame 0 with ``checkpoint`` (see the module docstring).

  Keeps qpos and ctrl after every substep and, per policy step, the reward and the
  reference pose the step is scored against.
  """
  import torch

  import mjlab.tasks  # noqa: F401 -- registers the tasks
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp.commands import MotionCommand
  from mjlab.tasks.tracking.mdp.rewards import max_step_reward
  from mjlab.tasks.tracking.scripts.evaluate_library import make_library_env
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  agent_cfg = load_rl_cfg(task)
  vec_env = make_library_env(task, [str(motion)], "cuda:0", agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task)
  assert runner_cls is not None, f"{task} registers no runner"
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


def best_window(motion: Path, num_steps: int, last_frame: int) -> int:
  """First scored frame of the ``num_steps`` frames (of 1 .. ``last_frame``) that
  have the largest summed reference root speed."""
  root_vel = np.load(motion)["body_lin_vel_w"][1 : last_frame + 1, 0]  # (last, 3)
  speed = np.linalg.norm(root_vel, axis=1)
  return 1 + int(np.argmax(np.convolve(speed, np.ones(num_steps), mode="valid")))


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
  clip: str,
  states: dict,
  server: viser.ViserServer,
  out_dir: Path,
  ghost: str,
  chrome: str,
  gpu: int | None,
  stills: list[float],
) -> None:
  """Render the policy-step frames of ``states`` (a ``window_states`` window) with
  ``ghost`` (a GHOSTS key) into ``<out_dir>/<ghost>.mp4``.

  With ``stills`` (clip times, s) only the PNG stills ``<ghost>_t<T>.png`` at the end
  of the policy step nearest to each time. ``gpu`` None renders with SwiftShader.
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
  still_times = {
    round(t / states["step_dt"]) * decimation - 1 - offset: t for t in stills
  }
  if not all(0 <= i < len(states["sub_qpos"]) for i in still_times):
    raise ValueError(f"stills {stills} not all inside the window")
  if still_times:
    frames = sorted(still_times)
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
    browser = stack.enter_context(HeadlessBrowser(server, chrome, gpu))
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
      f"[view] {ghost}: browser ready after {time.monotonic() - start:.1f} s",
      flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    video = None
    if not still_times:
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
      if i in still_times:
        Image.fromarray(frame).save(out_dir / f"{ghost}_t{still_times[i]:.2f}.png")
      if video is not None:
        video.add_image(frame)
  elapsed = time.monotonic() - start
  print(
    f"[view] {ghost}: {len(frames)} frames in {elapsed:.0f} s "
    f"({elapsed / len(frames):.2f} s per frame)",
    flush=True,
  )
