#!/usr/bin/env python3
"""Training curves of one SuperMimic run, from its metrics.csv.

Reads ``<run_dir>/metrics.csv`` (one row per logged iteration) and writes
``training_curves.png`` (150 dpi) in ``--out`` (default ``<run_dir>/figures/``): 8
panels of a 3x3 grid vs iteration (world-model loss, policy / critic losses, imagined
rollout statistics, simulator return, episode length, sampler statistics, time per
iteration split by phase); curves a run has not logged (the policy's before the
world-model warm-up ends) are left out. The library evaluations of the run
(``eval_library_model_<it>.json``) are plotted by ``tools/report_groups.py``.

``--hours``: against training wall-clock hours (metrics.csv's
``training_wall_seconds``) instead of iterations.

The run directory is only read, so it can be called on a run that is still training.

    PY=<worktree>/.venv/bin/python
    $PY tools/plot_supermimic_run.py <run_dir> [--out DIR] [--hours]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 150
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


def main() -> None:
  parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  parser.add_argument(
    "run_dir",
    type=Path,
    help="run directory with metrics.csv",
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
    x, xlabel = metrics[TRAINING_SECONDS] / 3600, "training wall-clock (h)"
  plot_training_curves(metrics, x, xlabel, out_dir / "training_curves.png", title)
  print(f"wrote {out_dir / 'training_curves.png'}")


if __name__ == "__main__":
  main()
