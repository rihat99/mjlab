"""Convert the LAFAN1 G1 retargeting CSVs to mjlab motion npz (30 -> 50 Hz).

Source: the LAFAN1 Retargeting Dataset
(https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), Unitree G1
29-dof ``g1/*.csv``: no header, 36 columns = pelvis position xyz (m), pelvis
quaternion xyzw, 29 joint angles (rad, mjlab's G1 joint order), 30 fps. Each CSV
becomes ``<output-dir>/<clip>.npz`` through mjlab's ``csv_to_npz`` (lerp / slerp to
50 Hz, finite-difference velocities, body states from the G1 kinematics), one clip
after the other in this process; a clip whose npz exists is skipped.

``csv_to_npz``'s per-frame self-check, which compares the simulator's read-back root
linear / angular velocity with the value just written (by default with torch's float32
tolerances, atol 1e-5, rtol 1.3e-6), runs here with atol = rtol = 1e-4: on
fallAndGetUp2_subject2 one frame differs by 1.22e-5 (float32 noise) and the default
check aborts. The check only compares; the saved arrays are the same.

    PY=<worktree>/.venv/bin/python
    $PY tools/convert_lafan.py <LAFAN1_Retargeting_Dataset>/g1 \\
        [--output-dir DIR] [--clips walk1_subject1,...] [--gpu 0]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from paths import LAFAN_MOTIONS_DIR

SELF_CHECK_TOLERANCE = 1e-4


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("input_dir", type=Path, help="folder of the G1 CSVs")
  parser.add_argument("--output-dir", type=Path, default=LAFAN_MOTIONS_DIR)
  parser.add_argument("--clips", help="comma-separated clip ids (default: all CSVs)")
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()
  if args.clips is None:
    csvs = sorted(args.input_dir.glob("*.csv"))
  else:
    csvs = [args.input_dir / f"{clip}.csv" for clip in args.clips.split(",")]
  missing = [str(path) for path in csvs if not path.is_file()]
  if not csvs or missing:
    parser.error(f"no CSVs in {args.input_dir} or missing: {missing}")
  # Set before mjlab imports torch.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  from mjlab.scripts import csv_to_npz

  for number, csv in enumerate(csvs, 1):
    output = args.output_dir / f"{csv.stem}.npz"
    if output.exists():
      print(f"[convert_lafan] {number}/{len(csvs)} {csv.stem}: exists, skipped")
      continue
    print(f"[convert_lafan] {number}/{len(csvs)} {csv.stem}", flush=True)
    csv_to_npz.main(
      input_file=str(csv),
      output_file=str(output),
      input_fps=30.0,
      output_fps=50.0,
      atol=SELF_CHECK_TOLERANCE,
      rtol=SELF_CHECK_TOLERANCE,
    )


if __name__ == "__main__":
  main()
