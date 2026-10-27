"""Convert the BONES-SEED G1 CSVs to mjlab motion npz (120 -> 50 Hz), fast.

Source: BONES-SEED (https://huggingface.co/datasets/bones-studio/seed), ``g1.tar.gz``
extracted to ``<seed>/g1/csv/<date>/<filename>.csv``. Each CSV has a header line and
36 columns: ``Frame``, root translate xyz (cm), root rotate xyz (degrees, R = Rz Ry Rx,
scipy ``from_euler("xyz")``), 29 joint angles (degrees, the G1 model's joint order),
120 fps. Each CSV becomes ``<output-dir>/<filename>.npz`` (file names are unique across
the date folders); a clip whose npz exists is skipped.

The npz is the one ``mjlab.scripts.csv_to_npz`` writes: same keys, dtypes and shapes,
the same 50 Hz lerp / slerp resampling and finite-difference velocities, and body
states from the same G1 tracking scene. Instead of a scene compile and a simulation
per clip, this runs the same steps in float64 numpy and CPU MuJoCo kinematics
(``mj_kinematics``, ``mj_comPos``, ``mj_comVel`` per frame) on one compiled model,
with a process pool over clips. The root quaternions are made sign-continuous along
each clip before the slerp.

    PY=<worktree>/.venv/bin/python
    $PY tools/convert_seed.py <seed>/g1/csv [--output-dir DIR] [--clips a,b,...]
        [--every N] [--workers 48]
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import time
from pathlib import Path

import mujoco
import numpy as np
from paths import MOTIONS_DIR
from scipy.spatial.transform import Rotation

from mjlab.scene import Scene
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

INPUT_FPS = 120.0
OUTPUT_FPS = 50.0
ROOT_COLUMNS = (
  "Frame",
  "root_translateX",
  "root_translateY",
  "root_translateZ",
  "root_rotateX",
  "root_rotateY",
  "root_rotateZ",
)
# csv_to_npz's quat_slerp returns q0 when |dot| is within 4 eps of 1 (its dtype's eps).
SLERP_EPS = 4.0 * np.finfo(np.float64).eps

# Set in main() before the pool forks; each worker makes its own MjData.
MODEL: mujoco.MjModel | None = None
DATA: mujoco.MjData | None = None


def robot_bodies(model: mujoco.MjModel) -> tuple[int, int]:
  """First body id and count of the robot's bodies (csv_to_npz's 30 body states)."""
  names = [model.body(i).name for i in range(model.nbody)]
  ids = [i for i, name in enumerate(names) if name.startswith("robot/")]
  if ids != list(range(ids[0], ids[0] + len(ids))):
    raise ValueError("robot bodies are not contiguous in the model")
  return ids[0], len(ids)


def joint_names(model: mujoco.MjModel) -> list[str]:
  """Hinge joint names in qpos order, without the entity prefix."""
  if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("the first joint must be the robot's free joint")
  return [model.joint(i).name.split("/")[-1] for i in range(1, model.njnt)]


def read_seed_csv(
  path: Path, joints: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Root position (N, 3) m, root quaternion (N, 4) wxyz, joint angles (N, 29) rad."""
  with open(path) as file:
    header = file.readline().strip().split(",")
    expected = [*ROOT_COLUMNS, *(f"{name}_dof" for name in joints)]
    if header != expected:
      raise ValueError(f"{path}: unexpected header {header}")
    table = np.loadtxt(file, delimiter=",", ndmin=2)
  root_pos = table[:, 1:4] / 100.0
  quat = Rotation.from_euler("xyz", table[:, 4:7], degrees=True).as_quat()
  quat = quat[:, [3, 0, 1, 2]]
  # q and -q are the same rotation; keep neighbouring frames on one hemisphere.
  step_sign = np.where(np.sum(quat[1:] * quat[:-1], axis=1) < 0.0, -1.0, 1.0)
  first_sign = 1.0 if quat[0, 0] >= 0.0 else -1.0
  sign = first_sign * np.concatenate([[1.0], np.cumprod(step_sign)])
  return root_pos, quat * sign[:, None], np.deg2rad(table[:, 7:])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Hamilton product of wxyz quaternions, (..., 4)."""
  w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
  w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )


def axis_angle_from_quat(quat: np.ndarray) -> np.ndarray:
  """Rotation vector of wxyz quaternions (mjlab's axis_angle_from_quat), (..., 3)."""
  quat = np.where(quat[..., :1] < 0.0, -quat, quat)
  half_angle = np.arctan2(np.linalg.norm(quat[..., 1:], axis=-1), quat[..., 0])
  angle = 2.0 * half_angle
  safe_angle = np.where(np.abs(angle) > 1e-6, angle, 1.0)
  scale = np.where(
    np.abs(angle) > 1e-6, np.sin(half_angle) / safe_angle, 0.5 - angle * angle / 48
  )
  return quat[..., 1:] / scale[..., None]


def slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
  """Row-wise mjlab quat_slerp: shortest arc from q0 (N, 4) to q1 (N, 4)."""
  dot = np.sum(q0 * q1, axis=1)
  q1 = np.where(dot[:, None] < 0.0, -q1, q1)
  dot = np.abs(dot)
  angle = np.arccos(np.clip(dot, -1.0, 1.0))
  same = (np.abs(dot - 1.0) < SLERP_EPS) | (angle < SLERP_EPS)
  sin = np.where(same, 1.0, np.sin(angle))
  weight_0 = np.where(same, 1.0, np.sin((1.0 - blend) * angle) / sin)
  weight_1 = np.where(same, 0.0, np.sin(blend * angle) / sin)
  return weight_0[:, None] * q0 + weight_1[:, None] * q1


def resample(
  root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """120 -> 50 Hz on csv_to_npz's time grid (lerp positions and joints, slerp root)."""
  input_frames = root_pos.shape[0]
  duration = (input_frames - 1) * (1.0 / INPUT_FPS)
  output_dt = 1.0 / OUTPUT_FPS
  output_frames = math.floor(duration / output_dt + 1e-6) + 1
  position = np.arange(output_frames) * output_dt / duration * (input_frames - 1)
  index_0 = np.floor(position).astype(np.int64)
  index_1 = np.minimum(index_0 + 1, input_frames - 1)
  blend = position - index_0

  def lerp(values: np.ndarray) -> np.ndarray:
    return values[index_0] * (1.0 - blend[:, None]) + values[index_1] * blend[:, None]

  quat = slerp(root_quat[index_0], root_quat[index_1], blend)
  return lerp(root_pos), quat, lerp(joint_pos)


def root_angular_velocity(quat: np.ndarray, dt: float) -> np.ndarray:
  """World-frame central SO(3) difference, ends repeated (csv_to_npz), (T, 3)."""
  conjugate = quat[:-2] * np.array([1.0, -1.0, -1.0, -1.0])
  omega = axis_angle_from_quat(quat_mul(quat[2:], conjugate)) / (2.0 * dt)
  return np.concatenate([omega[:1], omega, omega[-1:]])


def convert_clip(csv_path: Path, output_path: Path) -> tuple[str, int, int, float]:
  """Write one clip's npz; returns (name, input frames, output frames, seconds)."""
  assert MODEL is not None
  global DATA
  if DATA is None:
    DATA = mujoco.MjData(MODEL)
  start = time.perf_counter()
  first_body, body_count = robot_bodies(MODEL)
  bodies = slice(first_body, first_body + body_count)
  root_pos_in, root_quat_in, joint_pos_in = read_seed_csv(csv_path, joint_names(MODEL))
  root_pos, root_quat, joint_pos = resample(root_pos_in, root_quat_in, joint_pos_in)
  dt = 1.0 / OUTPUT_FPS
  lin_vel = np.gradient(root_pos, dt, axis=0)
  joint_vel = np.gradient(joint_pos, dt, axis=0)
  ang_vel_w = root_angular_velocity(root_quat, dt)
  # Free joint qvel: world linear velocity, body-frame angular velocity.
  ang_vel_b = Rotation.from_quat(root_quat[:, [1, 2, 3, 0]]).inv().apply(ang_vel_w)
  qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
  qvel = np.concatenate([lin_vel, ang_vel_b, joint_vel], axis=1)

  frames = qpos.shape[0]
  xpos = np.empty((frames, body_count, 3))
  xquat = np.empty((frames, body_count, 4))
  cvel = np.empty((frames, body_count, 6))
  subtree_com = np.empty((frames, 3))
  for frame in range(frames):
    DATA.qpos[:] = qpos[frame]
    DATA.qvel[:] = qvel[frame]
    mujoco.mj_kinematics(MODEL, DATA)
    mujoco.mj_comPos(MODEL, DATA)
    mujoco.mj_comVel(MODEL, DATA)
    xpos[frame] = DATA.xpos[bodies]
    xquat[frame] = DATA.xquat[bodies]
    cvel[frame] = DATA.cvel[bodies]
    subtree_com[frame] = DATA.subtree_com[first_body]
  # cvel is about the root subtree com; move the linear part to each body origin.
  offset = subtree_com[:, None, :] - xpos
  body_lin_vel = cvel[..., 3:] - np.cross(cvel[..., :3], offset)

  # Write then rename, so an interrupted run never leaves a truncated npz.
  tmp_path = output_path.with_name(output_path.name + ".tmp")
  with open(tmp_path, "wb") as file:
    np.savez(
      file,
      fps=np.array([OUTPUT_FPS]),
      joint_pos=joint_pos.astype(np.float32),
      joint_vel=joint_vel.astype(np.float32),
      body_pos_w=xpos.astype(np.float32),
      body_quat_w=xquat.astype(np.float32),
      body_lin_vel_w=body_lin_vel.astype(np.float32),
      body_ang_vel_w=cvel[..., :3].astype(np.float32),
    )
  os.replace(tmp_path, output_path)
  return csv_path.stem, root_pos_in.shape[0], frames, time.perf_counter() - start


def convert_star(job: tuple[Path, Path]) -> tuple[str, int, int, float]:
  return convert_clip(*job)


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("input_dir", type=Path, help="<seed>/g1/csv")
  parser.add_argument("--output-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument("--clips", help="comma-separated clip names (default: all)")
  parser.add_argument("--every", type=int, default=1, help="every N-th clip by name")
  parser.add_argument("--workers", type=int, default=48)
  args = parser.parse_args()

  csvs = {path.stem: path for path in args.input_dir.glob("*/*.csv")}
  if args.clips is None:
    names = sorted(csvs)[:: args.every]
  else:
    names = args.clips.split(",")
  missing = [name for name in names if name not in csvs]
  if not names or missing:
    parser.error(f"no CSVs in {args.input_dir} or missing: {missing[:10]}")
  args.output_dir.mkdir(parents=True, exist_ok=True)
  jobs = [
    (csvs[name], args.output_dir / f"{name}.npz")
    for name in names
    if not (args.output_dir / f"{name}.npz").exists()
  ]
  print(
    f"[convert_seed] {len(names)} clips, {len(names) - len(jobs)} exist", flush=True
  )

  global MODEL
  MODEL = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu").compile()
  start = time.time()
  input_frames = output_frames = 0
  context = multiprocessing.get_context("fork")
  with context.Pool(args.workers) as pool:
    results = pool.imap_unordered(convert_star, jobs, chunksize=16)
    for number, (_, frames_in, frames_out, _) in enumerate(results, 1):
      input_frames += frames_in
      output_frames += frames_out
      if number % 2000 == 0 or number == len(jobs):
        elapsed = time.time() - start
        print(
          f"[convert_seed] {number}/{len(jobs)} clips, {output_frames} frames at"
          f" 50 Hz, {elapsed:.0f} s, {output_frames / max(elapsed, 1e-9):.0f}"
          " frames/s",
          flush=True,
        )
  print(
    f"[convert_seed] done: {len(jobs)} clips, {input_frames} frames in,"
    f" {output_frames} out ({output_frames / OUTPUT_FPS / 3600:.2f} h),"
    f" {time.time() - start:.0f} s"
  )


if __name__ == "__main__":
  main()
