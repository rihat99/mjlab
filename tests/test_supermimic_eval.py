"""Evaluation tooling: training time per checkpoint, the checkpoints to evaluate
(skip, overwrite, watch) and tools/report_groups.py."""

import csv
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from mjlab.rl.local_log_writer import TRAINING_SECONDS, LocalLogWriter
from mjlab.tasks.tracking.scripts.evaluate_library import (
  EvaluateLibraryConfig,
  checkpoint_files,
  results_json,
  training_running,
  training_seconds,
  watched_checkpoints,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_training_seconds(tmp_path: Path) -> None:
  """The writer's wall-clock column is read at the row before the save."""
  run = tmp_path / "run"
  writer = LocalLogWriter(str(run))
  for iteration in (49, 99):
    writer.add_scalar("Perf/collection_time", 1.0, iteration)
    writer.add_scalar("Perf/learning_time", 2.0, iteration)
  writer.stop()
  with open(run / "metrics.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
  assert [row["iteration"] for row in rows] == ["49", "99"]
  seconds = training_seconds(run / "model_100.pt")
  assert seconds is not None and seconds == float(rows[1][TRAINING_SECONDS])
  assert 0 <= float(rows[0][TRAINING_SECONDS]) <= seconds
  assert training_seconds(run / "model_50.pt") == float(rows[0][TRAINING_SECONDS])
  assert training_seconds(run / "model_40.pt") is None  # before the first row
  assert training_seconds(run / "best.pt") is None

  for iteration in (8000, 400, 12000):
    (run / f"model_{iteration}.pt").touch()
  cfg = EvaluateLibraryConfig(motion_files="library.txt", run_dir=str(run))
  assert [path.name for path in checkpoint_files(cfg)] == [
    "model_400.pt",
    "model_8000.pt",
    "model_12000.pt",
  ]
  with pytest.raises(ValueError):
    checkpoint_files(EvaluateLibraryConfig(motion_files="library.txt"))


def test_skip_overwrite_and_watch(tmp_path: Path) -> None:
  """--run-dir skips checkpoints that have a results JSON unless --overwrite; --watch
  yields the pending ones, then each new one once settled while the training runs,
  and ends after the training process has exited."""
  run = tmp_path / "run"
  run.mkdir()
  old = time.time() - 3600
  for iteration in (2000, 4000):
    (run / f"model_{iteration}.pt").touch()
    os.utime(run / f"model_{iteration}.pt", (old, old))
  results_json(run / "model_2000.pt").write_text("{}")
  cfg = EvaluateLibraryConfig(motion_files="library.txt", run_dir=str(run))
  assert [path.name for path in checkpoint_files(cfg)] == ["model_4000.pt"]
  assert [path.name for path in checkpoint_files(replace(cfg, overwrite=True))] == [
    "model_2000.pt",
    "model_4000.pt",
  ]
  with pytest.raises(ValueError):
    checkpoint_files(replace(cfg, run_dir=None, checkpoint_file="a.pt", watch=True))

  trainer = subprocess.Popen(["sleep", "60"])
  watched = watched_checkpoints(
    replace(cfg, watch=True, pid=trainer.pid, poll_seconds=0.01)
  )
  first = next(watched)
  assert first.name == "model_4000.pt"
  results_json(first).write_text("{}")
  (run / "model_6000.pt").touch()  # still being written while the training runs
  (run / "model_8000.pt").touch()
  os.utime(run / "model_8000.pt", (old, old))
  assert next(watched).name == "model_8000.pt"
  trainer.kill()
  trainer.wait()
  assert [path.name for path in watched] == ["model_6000.pt"]

  # Without a pid: running while a model_<it>.pt or metrics.csv (none yet: the start
  # of the watch) is younger than stale_minutes.
  fresh_run = tmp_path / "fresh"
  fresh_run.mkdir()
  assert training_running(fresh_run, None, 1.0, since=time.time())
  assert not training_running(fresh_run, None, 1.0, since=old)
  (fresh_run / "metrics.csv").touch()
  assert training_running(fresh_run, None, 1.0, since=old)
  os.utime(fresh_run / "metrics.csv", (old, old))
  assert not training_running(fresh_run, None, 1.0, since=time.time())


def fake_eval(iteration: int, failed: set[str], seconds: float) -> dict:
  """Eval JSON of 6 clips: return 0.5 for a failed clip, else 0.9."""
  names = ("roll_a", "roll_b", "flip_a", "walk_a", "walk_b", "walk_c")
  returns = [0.5 if name in failed else 0.9 for name in names]
  clips = [
    {"clip": name, "failed": name in failed, "normalized_return": value}
    for name, value in zip(names, returns, strict=True)
  ]
  return {
    "num_clips": len(clips),
    "num_failed": len(failed),
    "mean_normalized_return": sum(returns) / len(returns),
    "frame_weighted_normalized_return": 0.8,
    "mean_normalized_return_survived": 0.85,
    "frame_weighted_normalized_return_survived": 0.82,
    "iteration": iteration,
    "training_seconds": seconds,
    "clips": clips,
  }


def test_report_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Group rows (exact, approximate, ours, not identified, all other, total) and the
  written table, CSV and figures."""
  monkeypatch.syspath_prepend(str(TOOLS))
  report_groups = importlib.import_module("report_groups")
  paper = {"size": 2, "sonic": 2, "supermimic_1gpu_3h": 1, "supermimic_2gpu_9.8h": 0}
  groups = {
    "paper_columns": {
      "rolls": paper,
      "flips": {**paper, "size": 4},
      "dance": paper,
      "all other": {**paper, "size": 3},
      "total": {**paper, "size": 6},
    },
    "groups": {
      "rolls": {"exact": True, "clips": ["roll_a", "roll_b"]},
      # A group clip outside the evaluated library is not counted.
      "flips": {"exact": False, "clips": ["flip_a", "flip_missing"]},
      "ours": {"exact": False, "clips": ["walk_a"]},
    },
  }
  groups_file = tmp_path / "groups.json"
  groups_file.write_text(json.dumps(groups))
  run = tmp_path / "run"
  run.mkdir()
  early = run / "eval_library_model_4000.json"
  late = run / "eval_library_model_8000.json"
  early.write_text(json.dumps(fake_eval(4000, {"roll_a", "walk_b", "flip_a"}, 900)))
  late.write_text(json.dumps(fake_eval(8000, {"roll_a"}, 1800)))

  rows = report_groups.group_rows(json.loads(late.read_text()), groups)
  table = {row["group"]: row for row in rows}
  assert list(table) == ["rolls", "flips", "dance", "ours", "all other", "total"]
  assert [row["definition"] for row in rows] == [
    "exact",
    "approx.",
    "not identified",
    "ours",
    "rest",
    "all",
  ]
  assert (table["rolls"]["clips"], table["rolls"]["failed"]) == (2, 1)
  assert table["rolls"]["mean_normalized_return"] == pytest.approx(0.7)
  assert table["flips"]["clips"] == 1
  assert table["dance"]["clips"] is None and table["dance"]["paper_sonic"] == 2
  assert (table["all other"]["clips"], table["all other"]["failed"]) == (2, 0)
  assert (table["total"]["clips"], table["total"]["failed"]) == (6, 1)
  assert table["total"]["paper_size"] == 6 and table["ours"]["paper_size"] is None
  only_total = report_groups.group_rows(json.loads(late.read_text()), {})
  assert [row["group"] for row in only_total] == ["total"]

  out = tmp_path / "out"
  argv = ["report_groups.py", str(late), str(early), "--groups", str(groups_file)]
  monkeypatch.setattr(sys, "argv", [*argv, "--out-dir", str(out)])
  report_groups.main()
  markdown = (out / "eval_library_model_8000_groups.md").read_text()
  assert "| rolls | exact | 2 | 1 | 50.00 | 0.7000 | 2 | 2 | 1 | 0 |" in markdown
  with open(out / "eval_library_model_8000_groups.csv", newline="") as handle:
    assert [row["group"] for row in csv.DictReader(handle)] == list(table)
  for figure in (
    "eval_library_model_8000_groups.png",
    "library_eval_curves.png",
    "per_clip_eval.png",
  ):
    assert (out / figure).stat().st_size > 0
