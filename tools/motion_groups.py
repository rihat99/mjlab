"""Motion groups of the SuperMimic paper's Table A.8 on the BONES-SEED library.

The paper counts failed motions per motion group but does not define the groups. We
define each group by the metadata ``content_name`` (a mirrored clip, ``<name>_M``,
goes with its original) and take its kept clips from ``<motions-dir>/manifest.json``
(``tools/filter_seed.py``). ``exact`` groups have the paper's size; the others
approximate a paper group (flips, lying/resting/standing up) or are ours, with no
paper column (the cartwheels, handstands and kozak dance that our name filter keeps).
The paper's dance routine (8 motions) cannot be identified by name and has no group.
Writes ``<motions-dir>/groups.json``::

  {"paper_columns": {<paper row>: {"size": ..., "sonic": ..., "supermimic_1gpu_3h": ...,
                                   "supermimic_2gpu_9.8h": ...}, ...},
   "groups": {<group>: {"exact": true|false, "clips": [<npz stem>, ...]}, ...}}

``paper_columns`` holds Table A.8: the group size and the failed motions of SONIC and
of SuperMimic after 3 h on one GPU and after 9.8 h on two GPUs, plus "all other" and
"total" rows.

    PY=<worktree>/.venv/bin/python
    $PY tools/motion_groups.py [--motions-dir DIR] [--metadata CSV]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from paths import MOTIONS_DIR, RAW_SEED_DIR

PAPER_COLUMNS = ("size", "sonic", "supermimic_1gpu_3h", "supermimic_2gpu_9.8h")
PAPER_TABLE = {
  "side rolls": (22, 22, 2, 0),
  "flips": (44, 44, 1, 0),
  "fire-in-the-hole turns": (20, 3, 3, 0),
  "crossed legs": (10, 9, 0, 0),
  "lying, resting and standing up": (99, 28, 8, 0),
  "broad jumps": (24, 15, 0, 0),
  "dance routine": (8, 8, 0, 0),
  "all other": (129_287, 50, 11, 0),
  "total": (129_514, 179, 25, 0),
}
# Group: (exact, regex that the whole content_name, without "_M", matches).
GROUPS = {
  "side rolls": (True, r"side_roll_R"),
  "flips": (False, r"flip_(090|180_R|360|360_hero_landing)"),
  "fire-in-the-hole turns": (True, r"fire_in_the_hole_.*"),
  "crossed legs": (True, r"crossed_legs_R"),
  "lying, resting and standing up": (
    False,
    r"(faint_)?stand_up_lying(_side|_stomach)?_R|fetal_position_R|spread_out_R"
    r"|rest_R|faint_side_R|balled_up_R",
  ),
  "broad jumps": (True, r"broad_jump"),
  "cartwheels": (False, r"cartwheel_R|cartwheelin"),
  "handstands": (False, r".*handstand.*"),
  "kozak dance": (False, r"kozak_dance"),
}


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--motions-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument(
    "--metadata",
    type=Path,
    default=RAW_SEED_DIR / "metadata" / "seed_metadata_v004.csv",
  )
  args = parser.parse_args()

  clips = json.loads((args.motions_dir / "manifest.json").read_text())["clips"]
  with open(args.metadata, newline="") as file:
    rows = list(csv.DictReader(file))
  groups = {}
  for group, (exact, pattern) in GROUPS.items():
    regex = re.compile(f"(?:{pattern})(?:_M)?")
    members = [row for row in rows if regex.fullmatch(row["content_name"])]
    kept = sorted(row["filename"] for row in members if clips[row["filename"]]["kept"])
    size = PAPER_TABLE[group][0] if group in PAPER_TABLE else None
    print(f"{group}: {len(kept)} kept of {len(members)} (paper {size}, exact {exact})")
    for name in sorted({row["content_name"] for row in members}):
      named = [row["filename"] for row in members if row["content_name"] == name]
      count = sum(clips[filename]["kept"] for filename in named)
      print(f"  {name:<44} {count:4d} kept of {len(named):4d}")
    if exact and len(kept) != size:
      raise ValueError(f"{group}: {len(kept)} kept clips, the paper has {size}")
    groups[group] = {"exact": exact, "clips": kept}

  paper_columns = {
    group: dict(zip(PAPER_COLUMNS, values, strict=True))
    for group, values in PAPER_TABLE.items()
  }
  output = args.motions_dir / "groups.json"
  output.write_text(
    json.dumps({"paper_columns": paper_columns, "groups": groups}, indent=1)
  )
  print(f"[motion_groups] wrote {output}")


if __name__ == "__main__":
  main()
