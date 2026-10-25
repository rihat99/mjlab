"""Evaluation tooling: training time per checkpoint and tools/report_groups.py."""

import csv
import importlib
import json
import sys
from pathlib import Path

import pytest

from mjlab.rl.local_log_writer import TRAINING_SECONDS, LocalLogWriter
from mjlab.tasks.tracking.scripts.evaluate_library import (
  ITERATION_TIMES,
  EvaluateLibraryConfig,
  checkpoint_files,
  training_seconds,
)

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_training_seconds(tmp_path: Path) -> None:
  """The writer's wall-clock column is read at the row before the save; an older
  metrics.csv without it gives the sum of the logged iteration times."""
  run = tmp_path / "run"
  writer = LocalLogWriter(str(run))
  for iteration in (49, 99):
    writer.add_scalar("Perf/collection_time", 1.0, iteration)
    writer.add_scalar("Perf/learning_time", 2.0, iteration)
  writer.stop()
  with open(run / "metrics.csv", newline="") as handle:
    rows = list(csv.DictReader(handle))
  assert [row["iteration"] for row in rows] == ["49", "99"]
  seconds, column = training_seconds(run / "model_100.pt")
  assert column == TRAINING_SECONDS and seconds is not None
  assert seconds == float(rows[1][TRAINING_SECONDS])
  assert 0 <= float(rows[0][TRAINING_SECONDS]) <= seconds
  assert training_seconds(run / "model_50.pt")[0] == float(rows[0][TRAINING_SECONDS])

  with open(run / "metrics.csv", "w", newline="") as handle:
    fields = ["iteration", "Perf/collection_time", "Perf/learning_time"]
    old = csv.DictWriter(handle, fieldnames=fields)
    old.writeheader()
    old.writerows({field: row[field] for field in fields} for row in rows)
  assert training_seconds(run / "model_100.pt") == (6.0, ITERATION_TIMES)
  assert training_seconds(run / "best.pt") == (None, None)

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
    "iteration": iteration,
    "training_seconds": seconds,
    "training_seconds_from": TRAINING_SECONDS,
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
  for figure in ("eval_library_model_8000_groups.png", "library_eval_curves.png"):
    assert (out / figure).stat().st_size > 0
