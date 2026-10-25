"""Data locations of the SuperMimic tools, all under the repo root (gitignored).

Motion data lives under ``$MJLAB_DATA``, by default ``<repo root>/data``; training
runs under ``<repo root>/runs``, rendered videos under ``<repo root>/videos``.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("MJLAB_DATA", REPO_ROOT / "data"))
MOTIONS_DIR = DATA_ROOT / "motions" / "lafan"  # LAFAN1 clips, tools/convert_lafan.py
MOTIONS_SEED_DIR = DATA_ROOT / "motions" / "seed"  # BONES-SEED, tools/convert_seed.py
MOTIONS_TEST_DIR = DATA_ROOT / "motions" / "test"  # clips in neither library
RAW_SEED_DIR = DATA_ROOT / "raw" / "bones_seed"  # raw BONES-SEED (g1/csv, metadata)
RUNS_ROOT = REPO_ROOT / "runs"
LOG_ROOT = RUNS_ROOT / "lafan"  # one folder per LAFAN1 training run
SEED_LOG_ROOT = RUNS_ROOT / "seed"  # one folder per BONES-SEED training run
VIDEOS_DIR = REPO_ROOT / "videos"  # rendered videos, stills and rollout.npz per clip
