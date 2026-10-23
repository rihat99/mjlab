"""Data and environment locations shared by the scripts in ``tools/``.

The data root is ``$MJLAB_DATA``, by default ``<repo root>/../retarget_test/mjlab``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_BIN = REPO_ROOT / ".venv" / "bin"  # mjlab's train / play entry points
DATA_ROOT = Path(
  os.environ.get("MJLAB_DATA", REPO_ROOT.parent / "retarget_test" / "mjlab")
)
MOTIONS_DIR = DATA_ROOT / "motions"  # <clip>.npz, mjlab motion files at 50 Hz
RUNS_DIR = DATA_ROOT / "runs"  # <clip>/<run>/
QUEUE_DIR = DATA_ROOT / "queue"  # train_queue.py status tables and stdout logs
VIDEOS_DIR = DATA_ROOT / "videos"  # <clip>/<tag>/
ROLLOUTS_DIR = DATA_ROOT / "rollouts"  # <clip>/<tag>.npz
FIGURES_DIR = DATA_ROOT / "figures"  # <clip>/*.png
MPS_DIR = DATA_ROOT / "mps"  # gpu<N>/{pipe,log}, one CUDA MPS daemon per GPU
