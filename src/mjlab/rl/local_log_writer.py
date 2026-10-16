"""Run logger that keeps everything on disk: TensorBoard, CSV and a PNG of curves."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from matplotlib.figure import Figure
from rsl_rl.utils import LogWriter
from torch.utils.tensorboard import SummaryWriter

# Tags drawn in curves.png, in panel order (metrics.csv keeps every tag).
_PLOTTED_PREFIXES = (
  "Train/",
  "Episode_Reward/",
  "Episode_Termination/",
  "Metrics/",
  "Loss/",
  "Policy/",
)
# Tags under these prefixes share units and are drawn in one panel; every other
# plotted tag gets its own panel.
_GROUPED_PREFIXES = ("Episode_Reward/", "Episode_Termination/")
_NUM_PANEL_COLUMNS = 4


class LocalLogWriter(SummaryWriter, LogWriter):
  """TensorBoard writer that also keeps ``metrics.csv`` and ``curves.png``.

  ``metrics.csv`` has one row per iteration and one column per scalar tag. A tag
  that first appears late gets a new column and the file is rewritten. A row is
  written once the next iteration starts logging, or at a checkpoint save or at
  stop. ``curves.png`` is redrawn at every checkpoint save and at stop.
  """

  def __init__(self, log_dir: str, flush_secs: int = 10) -> None:
    super().__init__(log_dir=log_dir, flush_secs=flush_secs)
    self._csv_path = Path(log_dir) / "metrics.csv"
    self._png_path = Path(log_dir) / "curves.png"
    self._columns: list[str] = ["iteration"]
    self._rows: list[dict[str, float]] = []
    self._current: dict[str, float] = {}

  def add_scalar(
    self,
    tag,
    scalar_value,
    global_step=None,
    walltime=None,
    new_style=False,
    double_precision=False,
  ) -> None:
    super().add_scalar(
      tag, scalar_value, global_step, walltime, new_style, double_precision
    )
    # rsl_rl logs "<tag>/time" against elapsed seconds, not iterations: TB only.
    if global_step is None or tag.endswith("/time"):
      return
    if self._current and self._current["iteration"] != global_step:
      self._write_row()
    self._current["iteration"] = global_step
    self._current[tag] = float(scalar_value)

  def save_model(self, model_path: str, it: int) -> None:
    del model_path, it
    self._write_row()
    self._plot()

  def stop(self) -> None:
    self._write_row()
    self._plot()
    self.close()

  def _write_row(self) -> None:
    if not self._current:
      return
    row, self._current = self._current, {}
    self._rows.append(row)
    new_columns = [tag for tag in row if tag not in self._columns]
    if new_columns:
      self._columns += new_columns
      tmp_path = self._csv_path.with_suffix(".csv.tmp")
      with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=self._columns)
        writer.writeheader()
        writer.writerows(self._rows)
      os.replace(tmp_path, self._csv_path)
    else:
      with open(self._csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=self._columns).writerow(row)

  def _plot(self) -> None:
    panels: dict[str, list[str]] = {}
    for tag in self._columns:
      if not tag.startswith(_PLOTTED_PREFIXES):
        continue
      group = next((p for p in _GROUPED_PREFIXES if tag.startswith(p)), tag)
      panels.setdefault(group, []).append(tag)
    if not panels:
      return
    order = sorted(
      panels,
      key=lambda name: next(
        i for i, prefix in enumerate(_PLOTTED_PREFIXES) if name.startswith(prefix)
      ),
    )
    num_rows = math.ceil(len(order) / _NUM_PANEL_COLUMNS)
    fig = Figure(figsize=(4 * _NUM_PANEL_COLUMNS, 2.8 * num_rows), layout="constrained")
    axes = fig.subplots(num_rows, _NUM_PANEL_COLUMNS, squeeze=False).ravel()
    for ax, name in zip(axes, order, strict=False):
      tags = panels[name]
      for tag in tags:
        points = [(row["iteration"], row[tag]) for row in self._rows if tag in row]
        label = tag.removeprefix(name) if len(tags) > 1 else None
        ax.plot(*zip(*points, strict=True), label=label, linewidth=1)
      ax.set_title(name.rstrip("/"), fontsize=9)
      ax.set_xlabel("iteration", fontsize=8)
      ax.tick_params(labelsize=7)
      ax.grid(alpha=0.3)
      if len(tags) > 1:
        ax.legend(fontsize=6)
    for ax in axes[len(order) :]:
      ax.set_visible(False)
    tmp_path = self._png_path.with_suffix(".tmp.png")
    fig.savefig(tmp_path, dpi=100)
    os.replace(tmp_path, self._png_path)
