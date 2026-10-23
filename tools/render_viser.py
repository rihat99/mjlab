"""Rendering library for viser videos of an mjlab G1 rollout from a headless browser.

The G1 meshes and materials come from mjlab's viser bridge (``MjlabViserScene`` on
mjviser). Frames are rendered by a headless Chromium connected to a local viser
server: hardware WebGL through ANGLE/Vulkan on a given GPU (or SwiftShader), at
PIXEL_RATIO times the frame size and downsampled. A ghost G1 (``ghost_model_of``) is
drawn in one colour with its meshes moved GHOST_INSET_M inward, so where ghost and
robot coincide the ghost hides inside the robot instead of z-fighting with it. A
caller blends a render without the ghost and one with an opaque ghost at
GHOST_OPACITY: one ghost layer over whatever is behind it, free of the browser's
per-mesh sorting. Every appearance setting (camera, colours, lights, floor, caption)
is a constant under "Look" below.

The browser is a Chromium headless shell (``npx playwright install
chromium-headless-shell``) or a wrapper script that execs one with ``--no-sandbox``;
on a machine without ``libasound.so.2`` the wrapper puts an extracted copy of the
library on ``LD_LIBRARY_PATH``.
"""

from __future__ import annotations

import copy
import os
import signal
import socket
import subprocess
import time
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
# --------------------------------------------------------------------------------------

GPU_FLAGS = [
  "--use-angle=vulkan",
  "--enable-features=Vulkan",
  "--enable-gpu",
  "--ignore-gpu-blocklist",
]
SOFTWARE_FLAGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
NVIDIA_VULKAN_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"


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
