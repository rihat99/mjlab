#!/usr/bin/env python3
"""Convert retargeted G1 motions (BetterRetarget result files, format version 2) into
mjlab motion-tracking npz files at 50 Hz.

Input contract, one clip = ``<clip>.npz`` plus ``<clip>.json`` side by side (written by
BetterRetarget's ``save_result``):

- ``<clip>.json``: run metadata; must hold ``"status": "ok"`` and
  ``"format_version": 2``;
- ``<clip>.npz`` (read with ``allow_pickle=False``): ``time`` (T,) s from the clip
  start, uniformly spaced at ``fps``; ``fps`` () frames per second; ``root_pos`` (T, 3)
  pelvis position, world z up, m; ``root_quat`` (T, 4) pelvis orientation, unit
  quaternions xyzw, sign-continuous; ``dof_pos`` (T, 29) joint angles, rad;
  ``dof_names`` (29,) str, the joint order, which must be mjlab's G1 order
  (``mjlab.scripts.csv_to_npz.G1_JOINT_NAMES``); ``dof_mask`` (29,) bool, which must be
  all True (mjlab needs every joint).

The conversion is mjlab's own csv_to_npz code path (lerp/slerp resampling to 50 Hz,
finite-difference velocities, forward kinematics of mjlab's G1 in the simulator): each
result is written as the CSV that csv_to_npz reads (root pos, root quat xyzw, 29 joint
angles in mjlab order, no header) and passed to ``mjlab.scripts.csv_to_npz.main``:

    PY=.venv/bin/python  # from the repo root
    $PY tools/convert_motion.py --gpu 0 $MJLAB_DATA/retarget/betterretarget/<clip>.npz
    $PY tools/convert_motion.py --gpu 0 --input-dir $MJLAB_DATA/retarget/betterretarget

Output: ``<output-dir>/<clip>.npz`` (default <data root>/motions/), plus ``<clip>.mp4``
with ``--render``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from paths import MOTIONS_DIR

# mjlab tracking plays one motion frame per env step (sim dt 0.005 x decimation 4).
OUTPUT_FPS = 50.0
FORMAT_VERSION = 2
RESULT_KEYS = (
  "time",
  "fps",
  "root_pos",
  "root_quat",
  "dof_pos",
  "dof_names",
  "dof_mask",
)


def load_result(path: Path) -> dict[str, np.ndarray]:
  """Arrays ``RESULT_KEYS`` of one result npz whose json says status ok, version 2."""
  meta = json.loads(path.with_suffix(".json").read_text())
  if meta.get("status") != "ok":
    raise ValueError(
      f"{path}: json status {meta.get('status')!r} ({meta.get('error')})"
    )
  if meta.get("format_version") != FORMAT_VERSION:
    raise ValueError(
      f"{path}: format_version {meta.get('format_version')!r}, "
      f"expected {FORMAT_VERSION}"
    )
  with np.load(path, allow_pickle=False) as data:
    return {key: data[key] for key in RESULT_KEYS}


def convert(input_path: Path, output_path: Path, render: bool) -> None:
  """Write the mjlab motion npz for one result npz."""
  from mjlab.scripts import csv_to_npz

  result = load_result(input_path)
  fps = float(result["fps"])
  if tuple(str(name) for name in result["dof_names"]) != csv_to_npz.G1_JOINT_NAMES:
    raise ValueError(f"{input_path}: dof_names differ from mjlab's G1 joint order")
  if not result["dof_mask"].all():
    masked = np.flatnonzero(~result["dof_mask"])
    raise ValueError(f"{input_path}: masked dofs {masked}; mjlab needs all 29")
  if not np.allclose(np.diff(result["time"]), 1.0 / fps, rtol=0.0, atol=1e-6):
    raise ValueError(f"{input_path}: frames are not uniformly spaced at {fps} fps")
  table = np.concatenate(
    [result["root_pos"], result["root_quat"], result["dof_pos"]], axis=1
  )
  with tempfile.TemporaryDirectory() as tmp_dir:
    csv_path = Path(tmp_dir) / f"{input_path.stem}.csv"
    np.savetxt(csv_path, table, delimiter=",")  # %.18e: exact float64 round trip
    csv_to_npz.main(
      input_file=str(csv_path),
      output_file=str(output_path),
      input_fps=fps,
      output_fps=OUTPUT_FPS,
      device="cuda:0",
      render=render,
    )


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("inputs", nargs="*", type=Path, help="result npz files")
  parser.add_argument(
    "--input-dir", type=Path, help="convert every <clip>.npz directly in this dir"
  )
  parser.add_argument("--output-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument(
    "--render", action="store_true", help="also write <clip>.mp4 next to the npz"
  )
  parser.add_argument(
    "--overwrite", action="store_true", help="replace existing output npz files"
  )
  args = parser.parse_args()

  inputs = list(args.inputs)
  if args.input_dir is not None:
    inputs += sorted(args.input_dir.glob("*.npz"))
  if not inputs:
    parser.error("give result npz files and/or --input-dir")
  # Set before mjlab imports torch / mujoco. EGL device ids are physical GPU ids (EGL
  # ignores CUDA_VISIBLE_DEVICES), so rendering would otherwise land on GPU 0.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(args.gpu)
  os.environ["WANDB_MODE"] = "disabled"

  failed = []
  for input_path in inputs:
    output_path = args.output_dir / f"{input_path.stem}.npz"
    if output_path.exists() and not args.overwrite:
      print(f"[skip] {output_path} exists (pass --overwrite to replace)")
      continue
    print(f"[convert] {input_path} -> {output_path}", flush=True)
    try:
      convert(input_path, output_path, args.render)
    except Exception as error:  # noqa: BLE001 -- batch mode: report every failing clip
      print(f"[failed] {input_path}: {error!r}", flush=True)
      failed.append(input_path.stem)
  if failed:
    sys.exit(f"{len(failed)} clip(s) failed: {failed}")


if __name__ == "__main__":
  main()
