"""Apply the SuperMimic paper's BONES-SEED filters (App. E) and write a manifest.

Paper: 142,220 raw clips -> 129,667 after the name filter -> 129,514 after the
feasibility filter (264.56 h). A clip is kept when neither stage flags it.

1. Name filter: case-insensitive substring match on the metadata ``filename`` with
   SONIC's 40 keywords (gear_sonic ``filter_and_copy_bones_data.py``) plus "50cm" and
   "look_over_fence" (the paper's examples jump_off_50cm, look_over_fence).
2. Feasibility filter on the 50 Hz npz of ``tools/convert_seed.py``. zmin(t) is the
   lowest body-link origin (``body_pos_w`` z); a foot's height is the smallest
   ``mj_geomDistance`` between its 7 collision capsules and the floor plane.
   - elevated support: zmin > 0.20 m for >= 0.6 s (30 contiguous frames);
   - ground penetration: zmin < -0.12 m anywhere, or < -0.08 m in > 80 % of frames;
   - floating feet: both feet > 0.02 m for >= 0.5 s (25 contiguous frames) while the
     pelvis is upright, at standing height and nearly still. The paper gives no gate
     thresholds; these are SuperMimic-Reproduced's: pelvis z-axis world z > 0.9,
     0.60 m < pelvis z < 0.85 m, pelvis speed < 0.1 m/s, joint speed RMS < 0.5 rad/s.

Writes ``<motions-dir>/manifest.json``: a summary (counts per stage and per reason,
hours, hits per name term, the lower foot's height on gated frames of kept clips) and,
per clip, kept, reasons, frames at 50 Hz, hours and the feasibility numbers.

    PY=<worktree>/.venv/bin/python
    $PY tools/filter_seed.py <seed> [--motions-dir DIR] [--workers 48]
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import time
from pathlib import Path

import mujoco
import numpy as np
from paths import MOTIONS_SEED_DIR

from mjlab.scene import Scene
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

SONIC_TERMS = (
  "bed",
  "bike",
  "chair",
  "climb",
  "com_up_50cm",
  "sitting",
  "step_on",
  "seat",
  "table",
  "_sit_",
  "sit_",
  "ladder",
  "crutch",
  "_bed_",
  "_ride_",
  "scooter",
  "stepdown",
  "acrobatics_",
  "box_hspu",
  "cartwheel",
  "50cm_box_",
  "on_box",
  "fall_from",
  "handstand_ff_",
  "on_1m",
  "form_box",
  "off_1m",
  "230m",
  "jump_over_obstacle_",
  "lift_crate_come_up_",
  "jump_to_shoulder_roll",
  "kozak_dance",
  "stair",
  "handstand",
  "box_jump",
  "monkey_jump",
  "safety_roll",
  "box_dips",
  "walking_on_edge",
  "push_obstacle",
)
NAME_TERMS = (*SONIC_TERMS, "50cm", "look_over_fence")
FPS = 50
ELEVATED_Z = 0.20
ELEVATED_FRAMES = 30
PENETRATION_Z = -0.12
PENETRATION_SOFT_Z = -0.08
PENETRATION_FRACTION = 0.8
FLOATING_Z = 0.02
FLOATING_FRAMES = 25
UPRIGHT_Z = 0.9
STANDING_Z = (0.60, 0.85)
STILL_SPEED = 0.1
STILL_JOINT_RMS = 0.5
FEASIBILITY = ("elevated_support", "ground_penetration", "floating_feet")

# Set in main() before the pool forks; each worker makes its own MjData.
MODEL: mujoco.MjModel | None = None
DATA: mujoco.MjData | None = None


def foot_geoms(model: mujoco.MjModel) -> tuple[list[int], list[int], int]:
  """Left and right foot collision capsule ids and the floor plane id."""
  feet = []
  for side in ("left", "right"):
    names = [f"robot/{side}_foot{i}_collision" for i in range(1, 8)]
    feet.append([model.geom(name).id for name in names])
  floor = model.geom("terrain").id
  if model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE:
    raise ValueError("the terrain geom is not a plane")
  return feet[0], feet[1], floor


def longest_run(mask: np.ndarray) -> int:
  """Length of the longest run of True in a 1-D mask."""
  edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
  return int(np.max(edges[1::2] - edges[::2])) if edges.size else 0


def feasibility(path: Path) -> dict:
  """Feasibility numbers and flags of one clip, plus its gated lower-foot heights."""
  assert MODEL is not None
  global DATA
  if DATA is None:
    DATA = mujoco.MjData(MODEL)
  left, right, floor = foot_geoms(MODEL)
  with np.load(path) as clip:
    joint_pos = clip["joint_pos"].astype(np.float64)
    joint_vel = clip["joint_vel"].astype(np.float64)
    body_pos = clip["body_pos_w"].astype(np.float64)
    pelvis_quat = clip["body_quat_w"][:, 0].astype(np.float64)
    pelvis_vel = clip["body_lin_vel_w"][:, 0].astype(np.float64)
  frames = joint_pos.shape[0]
  qpos = np.concatenate([body_pos[:, 0], pelvis_quat, joint_pos], axis=1)
  feet = np.empty((frames, 2))
  for frame in range(frames):
    DATA.qpos[:] = qpos[frame]
    mujoco.mj_kinematics(MODEL, DATA)
    for side, geoms in enumerate((left, right)):
      feet[frame, side] = min(
        mujoco.mj_geomDistance(MODEL, DATA, geom, floor, 10.0, None) for geom in geoms
      )

  zmin = body_pos[:, :, 2].min(axis=1)
  elevated_frames = longest_run(zmin > ELEVATED_Z)
  soft_fraction = float(np.mean(zmin < PENETRATION_SOFT_Z))
  # World z of the pelvis z-axis (wxyz quaternion).
  upright = 1.0 - 2.0 * (pelvis_quat[:, 1] ** 2 + pelvis_quat[:, 2] ** 2) > UPRIGHT_Z
  pelvis_z = body_pos[:, 0, 2]
  standing = (pelvis_z > STANDING_Z[0]) & (pelvis_z < STANDING_Z[1])
  still = (np.linalg.norm(pelvis_vel, axis=1) < STILL_SPEED) & (
    np.sqrt(np.mean(joint_vel**2, axis=1)) < STILL_JOINT_RMS
  )
  gate = upright & standing & still
  lower_foot = feet.min(axis=1)
  floating_frames = longest_run(gate & (lower_foot > FLOATING_Z))
  flags = {
    "elevated_support": elevated_frames >= ELEVATED_FRAMES,
    "ground_penetration": bool(zmin.min() < PENETRATION_Z)
    or soft_fraction > PENETRATION_FRACTION,
    "floating_feet": floating_frames >= FLOATING_FRAMES,
  }
  return {
    "name": path.stem,
    "frames": frames,
    "flags": [reason for reason in FEASIBILITY if flags[reason]],
    "zmin": round(float(zmin.min()), 4),
    "zmax_of_zmin": round(float(zmin.max()), 4),
    "elevated_frames": elevated_frames,
    "below_soft_fraction": round(soft_fraction, 4),
    "gated_frames": int(gate.sum()),
    "floating_frames": floating_frames,
    "stance": lower_foot[gate].astype(np.float32),
  }


def read_names(seed_dir: Path) -> list[str]:
  """Clip names (metadata ``filename``) of the release, in metadata order."""
  path = seed_dir / "metadata" / "seed_metadata_v004.csv"
  with open(path, newline="") as file:
    return [row["filename"] for row in csv.DictReader(file)]


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("seed_dir", type=Path, help="BONES-SEED root (metadata/)")
  parser.add_argument("--motions-dir", type=Path, default=MOTIONS_SEED_DIR)
  parser.add_argument("--workers", type=int, default=48)
  args = parser.parse_args()

  names = read_names(args.seed_dir)
  missing = [name for name in names if not (args.motions_dir / f"{name}.npz").exists()]
  if missing:
    parser.error(f"{len(missing)} clips not converted, e.g. {missing[:5]}")
  name_hits = {
    name: [term for term in NAME_TERMS if term in name.lower()] for name in names
  }

  global MODEL
  MODEL = Scene(unitree_g1_flat_tracking_env_cfg().scene, device="cpu").compile()
  start = time.time()
  paths = [args.motions_dir / f"{name}.npz" for name in names]
  with multiprocessing.get_context("fork").Pool(args.workers) as pool:
    results = {
      result["name"]: result
      for result in pool.imap_unordered(feasibility, paths, chunksize=32)
    }
  print(f"[filter_seed] feasibility pass: {time.time() - start:.0f} s")

  clips = {}
  stance = []
  for name in names:
    result = results[name]
    reasons = [f"name:{term}" for term in name_hits[name]] + result["flags"]
    kept = not reasons
    if kept:
      stance.append(result.pop("stance"))
    else:
      result.pop("stance")
    result.pop("name")
    result.pop("flags")
    frames = result["frames"]
    clips[name] = {"kept": kept, "reasons": reasons, "hours": frames / FPS / 3600}
    clips[name].update(result)

  survivors = [name for name in names if not name_hits[name]]
  flagged = {
    reason: sum(reason in clips[name]["reasons"] for name in survivors)
    for reason in FEASIBILITY
  }
  first_match = {reason: 0 for reason in FEASIBILITY}
  for name in survivors:
    found = [reason for reason in FEASIBILITY if reason in clips[name]["reasons"]]
    if found:
      first_match[found[0]] += 1
  kept = [name for name in names if clips[name]["kept"]]
  kept_frames = sum(clips[name]["frames"] for name in kept)
  terms = {
    term: {
      "hits": sum(term in hits for hits in name_hits.values()),
      "only_term": sum(hits == [term] for hits in name_hits.values()),
    }
    for term in NAME_TERMS
  }
  stance_all = np.concatenate(stance).astype(np.float64)
  summary = {
    "raw": len(names),
    "after_name_filter": len(survivors),
    "after_feasibility_filter": len(kept),
    "removed_by_name": len(names) - len(survivors),
    "removed_by_feasibility": len(survivors) - len(kept),
    "feasibility_flags_among_name_survivors": flagged,
    "feasibility_first_match_elevated_penetration_floating": first_match,
    "kept_frames_50hz": kept_frames,
    "kept_hours_frames": kept_frames / FPS / 3600,
    "kept_hours_duration": (kept_frames - len(kept)) / FPS / 3600,
    "lower_foot_on_gated_frames_of_kept_clips_m": {
      "frames": int(stance_all.size),
      "mean": float(stance_all.mean()),
      "std": float(stance_all.std()),
      "median": float(np.median(stance_all)),
    },
    "name_terms": terms,
  }
  output = args.motions_dir / "manifest.json"
  with open(output, "w") as file:
    json.dump({"summary": summary, "clips": clips}, file, indent=1)

  print(f"{'term':<24} {'hits':>6} {'only':>6}")
  for term, count in terms.items():
    print(f"{term:<24} {count['hits']:>6} {count['only_term']:>6}")
  print(json.dumps({k: v for k, v in summary.items() if k != "name_terms"}, indent=1))
  print(f"[filter_seed] wrote {output}")


if __name__ == "__main__":
  main()
