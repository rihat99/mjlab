#!/usr/bin/env python3
"""Figures of one SuperMimic training run: training curves and the library evaluations.

Reads ``<run_dir>/metrics.csv`` (one row per logged iteration) and
``<run_dir>/eval_library_model_<it>.json`` (one per checkpoint: per-clip normalised
return, failed flag, steps, mean and frame-weighted summary; newer files also hold the
return per survived step, ``*_survived``). Writes three PNGs (150 dpi) in ``--out``
(default ``<run_dir>/figures/``) and prints a markdown table of the checkpoint
evaluations:

- ``training_curves.png``: 8 panels of a 3x3 grid vs iteration (world-model loss,
  policy / critic losses, imagined rollout statistics, simulator return, episode
  length, sampler statistics, time per iteration split by phase); curves a run has
  not logged (the policy's before the world-model warm-up ends) are left out;
- ``library_eval.png``: mean and frame-weighted normalised return per clip length
  (solid) and per survived step (dashed, when the JSON has it), and number of failed
  clips, vs checkpoint iteration;
- ``per_clip_eval.png``: heatmap clips (rows, sorted by name) x checkpoints (columns)
  of the normalised return, a black cross on the cells of failed clips; for a library
  of more than 60 clips only the 60 with the lowest return at the last checkpoint.

``--hours``: the training curves and ``library_eval.png`` against training wall-clock
hours (metrics.csv's ``training_wall_seconds``; runs logged before that column use the
sum of the logged iteration times, about 1 % less) instead of iterations.

The run directory is only read, so it can be called on a run that is still training.

    PY=<worktree>/.venv/bin/python
    $PY tools/plot_supermimic_run.py <run_dir> [--out DIR] [--hours]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150
MAX_HEATMAP_CLIPS = 60
EVAL_NAME = re.compile(r"eval_library_model_(\d+)\.json")
# metrics.csv column of mjlab.rl.local_log_writer.LocalLogWriter.
TRAINING_SECONDS = "training_wall_seconds"


def read_metrics(path: Path) -> dict[str, np.ndarray]:
  """``{column: (n_rows,) float array}`` of a metrics csv; empty cells become nan."""
  with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))
  return {
    column: np.array(
      [float(row[column]) if row[column] != "" else np.nan for row in rows]
    )
    for column in rows[0]
  }


def training_hours(metrics: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
  """Training wall-clock hours at each metrics row ``(n_rows,)`` and its axis label.

  Runs logged before the ``TRAINING_SECONDS`` column: the cumulative logged iteration
  time, which leaves out the replay prefill and checkpoint saves.
  """
  if TRAINING_SECONDS in metrics:
    return metrics[TRAINING_SECONDS] / 3600, "training wall-clock (h)"
  seconds = np.cumsum(metrics["Perf/collection_time"] + metrics["Perf/learning_time"])
  return seconds / 3600, "training time (h, sum of iteration times)"


def read_evals(run_dir: Path) -> list[dict]:
  """The ``eval_library_model_<it>.json`` contents, sorted by checkpoint iteration.

  Each gets the key ``iteration``. Diagnostic variants
  (``eval_library_model_<it>_<variant>.json``) are skipped.
  """
  evals = []
  for path in run_dir.glob("eval_library_model_*.json"):
    match = EVAL_NAME.fullmatch(path.name)
    if match is None:
      continue
    with open(path) as handle:
      evaluation = json.load(handle)
    evaluation["iteration"] = int(match.group(1))
    evals.append(evaluation)
  return sorted(evals, key=lambda evaluation: evaluation["iteration"])


def plot_line(
  ax,
  x: np.ndarray,
  values: np.ndarray | None,
  label: str,
  color: str,
  linestyle: str = "-",
) -> None:
  """Plot ``values`` against ``x``, dropping the nan rows; nothing for None.

  The policy columns are empty before the policy trains and absent from a run that
  stopped before it did.
  """
  if values is None:
    return
  valid = ~np.isnan(values)
  ax.plot(
    x[valid],
    values[valid],
    color=color,
    linestyle=linestyle,
    linewidth=1.6,
    label=label,
  )


def style_axis(ax, title: str, xlabel: str, ylabel: str, log: bool = False) -> None:
  """Title, labels and grid of every training-curve panel; ``log``: log-scale y."""
  ax.set_title(title, fontsize=13)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.grid(alpha=0.3)
  if log:
    ax.set_yscale("log")


def twin(ax, ylabel: str, color: str):
  """Right-hand y axis in ``color`` sharing the x axis of ``ax``."""
  twin_ax = ax.twinx()
  twin_ax.set_ylabel(ylabel, color=color)
  twin_ax.tick_params(axis="y", colors=color)
  return twin_ax


def merge_legends(ax, twin_ax) -> None:
  """One legend for a panel with a twin axis."""
  handles, labels = ax.get_legend_handles_labels()
  twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
  ax.legend(handles + twin_handles, labels + twin_labels, fontsize=9, loc="best")


def plot_training_curves(
  metrics: dict[str, np.ndarray], x: np.ndarray, xlabel: str, out_path: Path, title: str
) -> None:
  """3x3 grid of the training curves of ``metrics`` (``read_metrics``) against ``x``
  (one value per row, e.g. the iteration), saved."""
  blue, orange, green, red = "tab:blue", "tab:orange", "tab:green", "tab:red"
  fig, axes = plt.subplots(3, 3, figsize=(18, 14))

  ax = axes[0, 0]
  plot_line(ax, x, metrics.get("Loss/world_model_state"), "state loss", blue)
  style_axis(
    ax,
    "World model: state loss and rollout error",
    xlabel,
    "state loss (log)",
    log=True,
  )
  ax.tick_params(axis="y", colors=blue)
  twin_ax = twin(ax, "error of the last step of the imagined rollout (log)", orange)
  plot_line(
    twin_ax,
    x,
    metrics.get("Metrics/world_model_last_step_error"),
    "last-step error",
    orange,
  )
  twin_ax.set_yscale("log")
  merge_legends(ax, twin_ax)

  ax = axes[0, 1]
  plot_line(ax, x, metrics.get("Loss/policy"), "policy loss", blue)
  style_axis(ax, "Policy loss and gradient norm", xlabel, "policy loss")
  ax.tick_params(axis="y", colors=blue)
  twin_ax = twin(ax, "policy gradient norm", orange)
  plot_line(twin_ax, x, metrics.get("Metrics/policy_grad_norm"), "grad norm", orange)
  merge_legends(ax, twin_ax)

  ax = axes[0, 2]
  plot_line(
    ax,
    x,
    metrics.get("Metrics/imagined_reward_normalized"),
    "imagined reward (normalised by max)",
    blue,
  )
  plot_line(ax, x, metrics.get("Metrics/imagined_survival"), "imagined survival", green)
  style_axis(ax, "Imagined rollouts", xlabel, "fraction (0 to 1)")
  ax.legend(fontsize=9, loc="best")

  ax = axes[1, 0]
  plot_line(ax, x, metrics.get("Loss/critic"), "critic loss", blue)
  style_axis(ax, "Critic loss", xlabel, "critic loss (log)", log=True)

  ax = axes[1, 1]
  plot_line(ax, x, metrics.get("Train/normalized_return"), "normalised return", blue)
  plot_line(ax, x, metrics.get("Train/termination_rate"), "termination rate", red)
  style_axis(
    ax,
    "Simulator training episodes",
    xlabel,
    "normalised return / termination rate (0 to 1)",
  )
  ax.legend(fontsize=9, loc="best")

  ax = axes[1, 2]
  plot_line(
    ax, x, metrics.get("Train/mean_episode_length"), "mean episode length", blue
  )
  style_axis(ax, "Episode length", xlabel, "mean episode length (steps)")

  ax = axes[2, 0]
  plot_line(ax, x, metrics.get("Metrics/sampler_reset_entropy"), "reset entropy", blue)
  style_axis(ax, "Reset sampler", xlabel, "entropy of the reset distribution")
  ax.tick_params(axis="y", colors=blue)
  twin_ax = twin(ax, "mean bin return", orange)
  plot_line(
    twin_ax,
    x,
    metrics.get("Metrics/sampler_mean_bin_return"),
    "mean bin return",
    orange,
  )
  handles, labels = ax.get_legend_handles_labels()
  twin_handles, twin_labels = twin_ax.get_legend_handles_labels()
  # Both curves run along the top and bottom edges; the middle-right is free.
  ax.legend(
    handles + twin_handles,
    labels + twin_labels,
    fontsize=9,
    loc="center right",
    bbox_to_anchor=(1, 0.62),
  )

  ax = axes[2, 1]
  phases = [
    ("Perf/sim_time", "simulation"),
    ("Perf/world_model_time", "world model"),
    ("Perf/policy_time", "policy"),
    ("Perf/critic_time", "critic"),
  ]
  ax.stackplot(
    x,
    [np.nan_to_num(metrics[key]) for key, _ in phases],
    labels=[label for _, label in phases],
    alpha=0.85,
  )
  total_time = sum(np.nan_to_num(metrics[key]) for key, _ in phases)
  # One-off spikes (first policy iteration, checkpoint saves) would hide the
  # steady-state split, so clip them.
  ax.set_ylim(0, 1.6 * np.median(total_time))
  style_axis(
    ax,
    "Time per iteration (one-off spikes clipped)",
    xlabel,
    "seconds per iteration (stacked)",
  )
  ax.legend(fontsize=9, loc="upper left")
  axes[2, 2].axis("off")

  fig.suptitle(title, fontsize=16)
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out_path, dpi=DPI)
  plt.close(fig)


def plot_library_eval(
  evals: list[dict],
  out_path: Path,
  title: str,
  hours: tuple[list[float], str] | None = None,
) -> None:
  """Normalised return and failed clip count vs checkpoint iteration, or vs
  ``hours = (training hours per eval, axis label)``.

  Mean and frame-weighted, per clip length and (when the JSON has it) per survived step.
  """
  iterations = [evaluation["iteration"] for evaluation in evals]
  x = iterations if hours is None else hours[0]
  num_clips = max(evaluation["num_clips"] for evaluation in evals)
  fig, (ax_return, ax_failed) = plt.subplots(1, 2, figsize=(14, 5.5))

  ax_return.plot(
    x,
    [e["mean_normalized_return"] for e in evals],
    "o-",
    color="tab:blue",
    label="mean over clips, per clip length",
  )
  ax_return.plot(
    x,
    [e["frame_weighted_normalized_return"] for e in evals],
    "s-",
    color="tab:orange",
    label="frame-weighted, per clip length",
  )
  # Older eval JSONs lack the per-survived-step keys: nan leaves a gap.
  for key, marker, color, label in (
    (
      "mean_normalized_return_survived",
      "o--",
      "tab:blue",
      "mean over clips, per survived step",
    ),
    (
      "frame_weighted_normalized_return_survived",
      "s--",
      "tab:orange",
      "frame-weighted, per survived step",
    ),
  ):
    values = [e.get(key, np.nan) for e in evals]
    if not np.all(np.isnan(values)):
      ax_return.plot(x, values, marker, color=color, label=label)
  ax_return.set_ylim(0, 1)
  ax_return.set_title("Library evaluation: normalised return", fontsize=13)
  ax_return.set_ylabel("normalised return (1 = full reward at every frame)")
  ax_return.legend()

  ax_failed.plot(x, [e["num_failed"] for e in evals], "o-", color="tab:red")
  ax_failed.set_ylim(0, num_clips)
  ax_failed.set_title(f"Library evaluation: failed clips (of {num_clips})", fontsize=13)
  ax_failed.set_ylabel("number of failed clips")

  for ax in (ax_return, ax_failed):
    if hours is None:
      ax.set_xlabel("checkpoint iteration")
      ax.set_xticks(iterations)
    else:
      ax.set_xlabel(hours[1])
    ax.grid(alpha=0.3)
  fig.suptitle(title, fontsize=15)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(out_path, dpi=DPI)
  plt.close(fig)


def plot_per_clip_eval(evals: list[dict], out_path: Path, title: str) -> None:
  """Heatmap clips (rows, by name) x checkpoints (columns) of the normalised return.

  Crosses mark failed clips. Large libraries: the ``MAX_HEATMAP_CLIPS`` clips with the
  lowest return at the last checkpoint.
  """
  clips = sorted({clip["clip"] for evaluation in evals for clip in evaluation["clips"]})
  if len(clips) > MAX_HEATMAP_CLIPS:
    last = sorted(evals[-1]["clips"], key=lambda clip: clip["normalized_return"])
    title += (
      f"\n{MAX_HEATMAP_CLIPS} lowest of {len(clips)} clips at the last checkpoint"
    )
    clips = sorted(clip["clip"] for clip in last[:MAX_HEATMAP_CLIPS])
  row_of = {name: row for row, name in enumerate(clips)}
  returns = np.full((len(clips), len(evals)), np.nan)
  failed = np.zeros((len(clips), len(evals)), dtype=bool)
  for column, evaluation in enumerate(evals):
    for clip in evaluation["clips"]:
      if clip["clip"] in row_of:
        returns[row_of[clip["clip"]], column] = clip["normalized_return"]
        failed[row_of[clip["clip"]], column] = clip["failed"]

  fig, ax = plt.subplots(figsize=(3.0 + 1.1 * len(evals), 1.5 + 0.3 * len(clips)))
  image = ax.imshow(returns, aspect="auto", cmap="viridis", vmin=0, vmax=1)
  for row, column in zip(*np.nonzero(failed), strict=True):
    ax.plot(column, row, marker="x", color="black", markersize=9, markeredgewidth=1.8)
  for row in range(len(clips)):
    for column in range(len(evals)):
      if not np.isnan(returns[row, column]):
        colour = "white" if returns[row, column] < 0.55 else "black"
        ax.text(
          column + 0.3,
          row,
          f"{returns[row, column]:.2f}",
          ha="center",
          va="center",
          fontsize=7,
          color=colour,
        )
  ax.set_xticks(
    range(len(evals)), [str(evaluation["iteration"]) for evaluation in evals]
  )
  ax.set_yticks(range(len(clips)), clips, fontsize=8)
  ax.set_xlabel("checkpoint iteration")
  ax.set_title(title + "\nnormalised return per clip (x = failed clip)", fontsize=12)
  fig.colorbar(image, ax=ax, label="normalised return", fraction=0.04, pad=0.02)
  fig.tight_layout()
  fig.savefig(out_path, dpi=DPI)
  plt.close(fig)


def markdown_table(evals: list[dict]) -> str:
  """Markdown table of the checkpoint evaluations."""
  lines = [
    (
      "| iteration | mean normalised return | frame-weighted normalised return "
      "| mean per survived step | frame-weighted per survived step | failed / N |"
    ),
    "|---|---|---|---|---|---|",
  ]
  for e in evals:
    survived = [
      e.get(key, np.nan)
      for key in (
        "mean_normalized_return_survived",
        "frame_weighted_normalized_return_survived",
      )
    ]
    lines.append(
      f"| {e['iteration']} | {e['mean_normalized_return']:.4f} "
      f"| {e['frame_weighted_normalized_return']:.4f} | {survived[0]:.4f} "
      f"| {survived[1]:.4f} | {e['num_failed']} / {e['num_clips']} |"
    )
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  parser.add_argument(
    "run_dir",
    type=Path,
    help="run directory with metrics.csv and eval_library_model_<it>.json",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="output directory (default <run_dir>/figures)",
  )
  parser.add_argument(
    "--hours",
    action="store_true",
    help="x axis in training wall-clock hours instead of iterations",
  )
  args = parser.parse_args()

  out_dir = args.out if args.out is not None else args.run_dir / "figures"
  out_dir.mkdir(parents=True, exist_ok=True)
  title = args.run_dir.name

  metrics = read_metrics(args.run_dir / "metrics.csv")
  x, xlabel = metrics["iteration"], "training iteration"
  if args.hours:
    x, xlabel = training_hours(metrics)
  plot_training_curves(metrics, x, xlabel, out_dir / "training_curves.png", title)
  evals = read_evals(args.run_dir)
  if not evals:
    raise SystemExit(f"no eval_library_model_*.json in {args.run_dir}")
  hours = None
  if args.hours:
    # model_<it>.pt is saved right after the row of iteration it - 1.
    eval_hours = np.interp([e["iteration"] - 1 for e in evals], metrics["iteration"], x)
    hours = (eval_hours.tolist(), xlabel)
  plot_library_eval(evals, out_dir / "library_eval.png", title, hours)
  plot_per_clip_eval(evals, out_dir / "per_clip_eval.png", title)
  print(markdown_table(evals))
  print(f"\nfigures in {out_dir}")


if __name__ == "__main__":
  main()
