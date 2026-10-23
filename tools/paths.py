"""Data locations of the SuperMimic tools.

Everything lives under one data root: ``$MJLAB_DATA``, by default
``<repo root>/../retarget_test/mjlab``.
"""

import os
from pathlib import Path

DATA_ROOT = Path(
  os.environ.get(
    "MJLAB_DATA", Path(__file__).resolve().parents[1].parent / "retarget_test/mjlab"
  )
)
MOTIONS_DIR = DATA_ROOT / "motions_lafan"  # the LAFAN1 clips, tools/convert_lafan.py
LOG_ROOT = DATA_ROOT / "runs" / "lafan_supermimic"  # one folder per training run
VISER_DIR = DATA_ROOT / "viser" / "supermimic"  # tools/render_viser_supermimic.py
