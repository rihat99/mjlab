#!/usr/bin/env python3
"""Per-group results of a library evaluation (the paper's Table A.8) and run curves.

Reads ``eval_library_model_<it>.json`` files (``evaluate_library``) and a
``groups.json`` (``tools/motion_groups.py``: Table A.8's columns and the clips of each
motion group). For the last evaluation by iteration it writes, in ``--out-dir``
(default ``<folder of the first JSON>/figures``):

- ``<json stem>_groups.md`` and ``.csv``: one row per group, then "all other" (clips in
  no group) and "total": evaluated clips, failed clips, failed fraction and mean
  normalised return, next to the paper's group size and failed motions of SONIC and of
  SuperMimic (1 GPU x 3 h, 2 GPUs x 9.8 h) where the group has them. ``definition``
  says whether a group has the paper's size (exact), approximates a paper group
  (approx.), is ours (no paper column), or is a paper group we cannot identify
  (not identified: paper columns only);
- ``<json stem>_groups.png``: failed fraction and mean normalised return per group.

Given several JSONs of a run, also ``library_eval_curves.png``: mean (and
frame-weighted) normalised return and failed clips vs checkpoint iteration and vs
training wall-clock hours (the JSON's ``training_seconds``), as the paper's Fig. 3(a)
and A.1;
when the evaluated library is within 10 % of the paper's benchmark size, the paper's
failed counts are drawn as reference lines.

``--groups`` defaults to ``groups.json`` in the folder of the evaluated clips (from the
JSON's provenance); without one the table has the total only.

    PY=<worktree>/.venv/bin/python
    $PY tools/report_groups.py RUN/eval_library_model_*.json [--groups FILE] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150
# groups.json "paper_columns" keys and their table headers.
PAPER_COLUMNS = {
  "size": "paper size",
  "sonic": "SONIC failed",
  "supermimic_1gpu_3h": "SuperMimic 1 GPU x 3 h failed",
  "supermimic_2gpu_9.8h": "SuperMimic 2 GPUs x 9.8 h failed",
}
# Paper rows that are not motion groups.
REST_ROWS = ("all other", "total")
# Reference lines need a library of about the paper's benchmark size.
REFERENCE_SIZE_TOLERANCE = 0.1


def default_groups_file(evaluation: dict) -> Path | None:
  """``groups.json`` in the folder of the evaluated clips, from the provenance."""
  motion_files = evaluation.get("provenance", {}).get("motion_files", [])
  if not motion_files:
    return None
  first = Path(motion_files[0])
  if first.suffix == ".txt":
    if not first.is_file():
      return None
    first = Path(first.read_text().split("\n", 1)[0])
  return first.parent / "groups.json"


def summary_row(name: str, definition: str, records: list[dict], paper: dict) -> dict:
  """Table row of the clip ``records`` of one group; nan return for no clips."""
  failed = sum(record["failed"] for record in records)
  return {
    "group": name,
    "definition": definition,
    "clips": len(records),
    "failed": failed,
    "failed_fraction": failed / len(records) if records else math.nan,
    "mean_normalized_return": (
      float(np.mean([record["normalized_return"] for record in records]))
      if records
      else math.nan
    ),
    **{f"paper_{key}": paper.get(name, {}).get(key) for key in PAPER_COLUMNS},
  }


def group_rows(evaluation: dict, groups: dict) -> list[dict]:
  """Rows in the paper's order, then our own groups, "all other" and "total".

  ``groups`` is a groups.json (empty: the total only). A clip of the groups that was
  not evaluated is not counted.
  """
  clips = {record["clip"]: record for record in evaluation["clips"]}
  paper = groups.get("paper_columns", {})
  definitions = groups.get("groups", {})
  if not definitions:
    return [summary_row("total", "all", list(clips.values()), paper)]
  rows = []
  names = [name for name in paper if name not in REST_ROWS]
  names += [name for name in definitions if name not in paper]
  for name in names:
    if name not in definitions:
      rows.append(
        summary_row(name, "not identified", [], paper) | {"clips": None, "failed": None}
      )
      continue
    group = definitions[name]
    definition = "exact" if group["exact"] else "approx." if name in paper else "ours"
    records = [clips[clip] for clip in group["clips"] if clip in clips]
    rows.append(summary_row(name, definition, records, paper))
  grouped = {clip for group in definitions.values() for clip in group["clips"]}
  other = [record for clip, record in clips.items() if clip not in grouped]
  rows.append(summary_row("all other", "rest", other, paper))
  rows.append(summary_row("total", "all", list(clips.values()), paper))
  return rows


def format_cell(value: object, digits: int = 4) -> str:
  """Markdown cell: '-' for None or nan, fixed digits for floats."""
  if value is None or (isinstance(value, float) and math.isnan(value)):
    return "-"
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def markdown_table(rows: list[dict], caption: str) -> str:
  """Markdown table of ``group_rows`` under a one-line caption."""
  header = [
    "group",
    "definition",
    "clips",
    "failed",
    "failed %",
    "mean normalised return",
    *PAPER_COLUMNS.values(),
  ]
  lines = [
    caption,
    "",
    "| " + " | ".join(header) + " |",
    "|" + "---|" * len(header),
  ]
  for row in rows:
    cells = [
      row["group"],
      row["definition"],
      format_cell(row["clips"]),
      format_cell(row["failed"]),
      format_cell(100 * row["failed_fraction"], 2),
      format_cell(row["mean_normalized_return"]),
      *(format_cell(row[f"paper_{key}"]) for key in PAPER_COLUMNS),
    ]
    lines.append("| " + " | ".join(cells) + " |")
  return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
  """``group_rows`` as CSV, one column per row key; empty cells for None."""
  with open(path, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def plot_groups(rows: list[dict], out_path: Path, title: str) -> None:
  """Horizontal bars per group: failed fraction and mean normalised return."""
  rows = [row for row in rows if row["clips"]]
  labels = [f"{row['group']} ({row['definition']})" for row in rows][::-1]
  failed = [100 * row["failed_fraction"] for row in rows][::-1]
  returns = [row["mean_normalized_return"] for row in rows][::-1]
  counts = [f"{row['failed']} / {row['clips']}" for row in rows][::-1]
  positions = np.arange(len(rows))
  fig, (ax_failed, ax_return) = plt.subplots(
    1, 2, figsize=(14, 1.5 + 0.45 * len(rows)), sharey=True
  )
  ax_failed.barh(positions, failed, color="tab:red")
  for position, value, count in zip(positions, failed, counts, strict=True):
    ax_failed.text(value, position, f" {count}", va="center", fontsize=8)
  ax_failed.set_xlim(0, max(100.0, 1.25 * max(failed)))
  ax_failed.set_xlabel("failed clips (% of the group; label: failed / clips)")
  ax_return.barh(positions, returns, color="tab:blue")
  for position, value in zip(positions, returns, strict=True):
    ax_return.text(value, position, f" {value:.3f}", va="center", fontsize=8)
  ax_return.set_xlim(0, 1.12)
  ax_return.set_xlabel("mean normalised return")
  ax_failed.set_yticks(positions, labels, fontsize=9)
  for ax in (ax_failed, ax_return):
    ax.grid(axis="x", alpha=0.3)
  fig.suptitle(title, fontsize=13)
  fig.tight_layout()
  fig.savefig(out_path, dpi=DPI)
  plt.close(fig)


def plot_curves(evals: list[dict], groups: dict, out_path: Path, title: str) -> None:
  """Return and failed clips vs iteration and vs training hours (2 x 2 panels).

  The hours column is left empty when an evaluation lacks ``training_seconds``.
  """
  iterations = [evaluation["iteration"] for evaluation in evals]
  hours: list[float] | None = [
    evaluation["training_seconds"] / 3600
    for evaluation in evals
    if evaluation["training_seconds"] is not None
  ]
  if len(hours) < len(evals):
    hours = None
  total = groups.get("paper_columns", {}).get("total", {})
  num_clips = evals[-1]["num_clips"]
  reference = bool(total) and (
    abs(num_clips - total["size"]) <= REFERENCE_SIZE_TOLERANCE * total["size"]
  )
  fig, axes = plt.subplots(2, 2, figsize=(14, 9))
  for column, (x, xlabel) in enumerate(
    ((iterations, "checkpoint iteration"), (hours, "training wall-clock (h)"))
  ):
    ax_return, ax_failed = axes[0, column], axes[1, column]
    if x is None:
      for ax in (ax_return, ax_failed):
        ax.text(0.5, 0.5, "no training_seconds in the JSONs", ha="center")
        ax.set_axis_off()
      continue
    ax_return.plot(
      x, [e["mean_normalized_return"] for e in evals], "o-", label="mean over clips"
    )
    ax_return.plot(
      x,
      [e["frame_weighted_normalized_return"] for e in evals],
      "s--",
      label="frame-weighted",
    )
    ax_return.set_ylabel("normalised return")
    ax_return.legend(fontsize=9)
    ax_failed.plot(x, [e["num_failed"] for e in evals], "o-", color="tab:red")
    if reference:
      for key, color in (("sonic", "black"), ("supermimic_1gpu_3h", "tab:gray")):
        ax_failed.axhline(
          total[key],
          color=color,
          linestyle=":",
          label=f"paper: {PAPER_COLUMNS[key].removesuffix(' failed')}, {total[key]}",
        )
      ax_failed.legend(fontsize=9)
    ax_failed.set_ylabel(f"failed clips (of {num_clips})")
    ax_failed.set_ylim(bottom=0)
    for ax in (ax_return, ax_failed):
      ax.set_xlabel(xlabel)
      ax.grid(alpha=0.3)
  fig.suptitle(title, fontsize=14)
  fig.tight_layout()
  fig.savefig(out_path, dpi=DPI)
  plt.close(fig)


def main() -> None:
  parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  parser.add_argument("evals", type=Path, nargs="+", help="eval_library_*.json files")
  parser.add_argument("--groups", type=Path, help="groups.json (default: see above)")
  parser.add_argument(
    "--out-dir", type=Path, help="output folder (default <JSON folder>/figures)"
  )
  args = parser.parse_args()

  loaded = [(json.loads(path.read_text()), path) for path in args.evals]
  loaded.sort(key=lambda pair: (pair[0]["iteration"] is None, pair[0]["iteration"]))
  evals = [evaluation for evaluation, _ in loaded]
  last, last_path = loaded[-1]
  groups_file = args.groups or default_groups_file(last)
  groups = {}
  if groups_file is not None and groups_file.is_file():
    groups = json.loads(groups_file.read_text())
  elif args.groups is not None:
    raise SystemExit(f"no groups file {args.groups}")
  else:
    print(f"[report_groups] no groups.json ({groups_file}): total only")
  out_dir = args.out_dir or args.evals[0].parent / "figures"
  out_dir.mkdir(parents=True, exist_ok=True)

  rows = group_rows(last, groups)
  hours = last.get("training_seconds")
  title = (
    f"{last_path.parent.name}, iteration {last['iteration']}"
    + (f" ({hours / 3600:.2f} h)" if hours is not None else "")
    + f": {last['num_clips']} clips"
  )
  caption = f"{title}; groups {groups_file if groups else 'none'}"
  table = markdown_table(rows, caption)
  (out_dir / f"{last_path.stem}_groups.md").write_text(table)
  write_csv(rows, out_dir / f"{last_path.stem}_groups.csv")
  plot_groups(rows, out_dir / f"{last_path.stem}_groups.png", title)
  if len(evals) > 1:
    plot_curves(
      evals, groups, out_dir / "library_eval_curves.png", last_path.parent.name
    )
  print(table)
  print(f"written to {out_dir}")


if __name__ == "__main__":
  main()
