#!/usr/bin/env python3
"""Figures of a Vic tracker's PD gains and actuator torques over one clip, from the
rollout npz files that ``render_viser.py`` writes (one rollout: 1 env, nominal robot,
200 Hz substeps).

Reads ``<rollouts-dir>/<clip>/<tag>.npz`` per run tag (and ``original.npz``, the
original tracker, for the torque comparison); writes three PNGs (200 dpi) per tag in
``--out-dir`` (default <data root>/figures/<clip>/), the time axis the clip time of the
videos, each with a title naming the run and the clip and a legend for every line and
shading:

- ``gains_<tag>.png``: kp / kp0 per substep for the left leg and the waist (one panel
  per joint, log y from 0.25 to 4, nominal = 1 dashed); for a ``freq_damp`` run a
  second block of panels with kd / kd0;
- ``gains_heatmap_<tag>.png``: log m = ln(kp / kp0) / 2 for all 29 joints over time
  (centred diverging colours, limits +-ln 2 = the policy's bound); a ``freq_damp`` run
  gets a second heatmap of log r = ln(kd / kd0) - log m;
- ``torques_<tag>.png``: actuator torque of the same joints, the Vic run in colour, the
  original tracker's rollout in thin grey, the torque limits dashed.

Phases are shaded on every time plot: a reference foot is in contact when its height is
below its minimum over the clip plus CONTACT_MARGIN; flight = both feet off (orange),
double stance = both on (blue-grey), single stance unshaded. Thin vertical lines mark
``--stills`` (the videos' stills).

    PY=.venv/bin/python  # from the repo root
    $PY tools/plot_gains_torques.py --clip amass_cmu_88_06 --tag vicfreq vicfreqdamp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from paths import FIGURES_DIR, ROLLOUTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CONTACT_MARGIN = 0.03  # m above the foot's minimum reference height
PANEL_JOINTS = (  # the figures' 3 x 3 panels
  "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
  "left_knee", "left_ankle_pitch", "left_ankle_roll",
  "waist_yaw", "waist_roll", "waist_pitch",
)  # fmt: skip
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = (
  "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
  "wrist_roll", "wrist_pitch", "wrist_yaw",
)  # fmt: skip
HEATMAP_JOINTS = (  # robot joint order
  *(f"left_{j}" for j in LEG),
  *(f"right_{j}" for j in LEG),
  "waist_yaw",
  "waist_roll",
  "waist_pitch",
  *(f"left_{j}" for j in ARM),
  *(f"right_{j}" for j in ARM),
)
FLIGHT_COLOR, DOUBLE_COLOR = ("orange", 0.2), ("steelblue", 0.12)
VIC_COLOR, KD_COLOR, BASELINE_COLOR = "tab:blue", "tab:green", (0.45, 0.45, 0.45)
LOG_RANGE = np.log(2.0)  # the gain term's bound on |log m| and |log r|
DPI = 200


class Rollout:
  """One rollout npz: arrays per substep plus the joint names of the actuators
  (``robot/`` prefix dropped)."""

  def __init__(self, path: Path) -> None:
    data = np.load(path)
    self.t: np.ndarray = data["substep_time"]  # (S,)
    self.joints = [
      str(name).split("/")[-1].removesuffix("_joint")
      for name in data["actuator_joints"]
    ]
    self.torque, self.kp, self.kd = data["torque"], data["kp"], data["kd"]  # (S, nu)
    # (nu,) each.
    self.kp0, self.kd0, self.torque_limit = (
      data["kp0"],
      data["kd0"],
      data["torque_limit"],
    )
    self.label, self.task = str(data["label"]), str(data["task"])
    self.run = Path(str(data["checkpoint"])).parent.name  # the policy's run directory
    ref_feet = data["ref_feet_height"]  # (S, 2)
    contact = ref_feet < ref_feet.min(axis=0) + CONTACT_MARGIN
    self.flight, self.double = ~contact.any(axis=1), contact.all(axis=1)  # (S,) bool

  def column(self, joint: str) -> int:
    return self.joints.index(joint)

  @property
  def log_m(self) -> np.ndarray:
    """(S, nu) ln(kp / kp0) / 2: the frequency multiplier the policy chose."""
    return np.log(self.kp / self.kp0) / 2

  @property
  def log_r(self) -> np.ndarray:
    """(S, nu) ln(kd / kd0) - log m: the damping-ratio multiplier."""
    return np.log(self.kd / self.kd0) - self.log_m


def shade_phases(ax: Axes, rollout: Rollout, stills: list[float]) -> None:
  """Flight and double-stance spans of ``rollout`` as background bands, plus a line
  per still time."""
  for mask, (color, alpha) in (
    (rollout.flight, FLIGHT_COLOR),
    (rollout.double, DOUBLE_COLOR),
  ):
    # Starts and ends of the true runs.
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    dt = rollout.t[1] - rollout.t[0]
    for start, end in edges.reshape(-1, 2):
      ax.axvspan(
        rollout.t[start] - dt, rollout.t[end - 1], color=color, alpha=alpha, lw=0
      )
  for still in stills:
    ax.axvline(still, color="k", lw=0.5, alpha=0.6)


def phase_handles(stills: list[float]) -> list[Patch | Line2D]:
  """Legend entries for what ``shade_phases`` draws."""
  times = ", ".join(f"{still:g}" for still in stills)
  return [
    Patch(
      color=FLIGHT_COLOR[0],
      alpha=FLIGHT_COLOR[1],
      label="flight (both reference feet off)",
    ),
    Patch(color=DOUBLE_COLOR[0], alpha=DOUBLE_COLOR[1], label="double stance"),
    Patch(facecolor="white", edgecolor="0.7", label="unshaded: single stance"),
    Line2D([], [], color="k", lw=0.5, alpha=0.6, label=f"video stills ({times} s)"),
  ]


def panel_grid(
  parent: Figure | SubFigure,
  rows: int,
  rollout: Rollout,
  stills: list[float],
  ylabel: str,
) -> np.ndarray:
  """``rows`` x 3 panels in ``parent`` with shared x = clip time, phases shaded,
  ``ylabel`` on the left column."""
  axes = parent.subplots(rows, 3, sharex=True, squeeze=False)
  for ax in axes.ravel():
    shade_phases(ax, rollout, stills)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
  for ax in axes[-1]:
    ax.set_xlabel("clip time (s)", fontsize=9)
  for ax in axes[:, 0]:
    ax.set_ylabel(ylabel, fontsize=9)
  axes[0, 0].set_xlim(0, rollout.t[-1])
  return axes


def plot_gains(
  rollout: Rollout, clip: str, tag: str, stills: list[float], out_dir: Path
) -> Path:
  blocks = [("kp / kp0", "gain", rollout.kp / rollout.kp0, VIC_COLOR)]
  if rollout.task.endswith("FreqDamp"):
    blocks.append(("kd / kd0", "damping", rollout.kd / rollout.kd0, KD_COLOR))
  fig = plt.figure(figsize=(15, 7.6 * len(blocks) + 0.4), layout="constrained")
  fig.suptitle(
    f"{rollout.label} run {rollout.run}, clip {clip}: PD gains chosen by the policy, "
    "relative to the nominal gains (left leg and waist)",
    fontsize=11,
  )
  subfigs = np.atleast_1d(fig.subfigures(len(blocks), 1))
  for subfig, (name, short, ratio, color) in zip(subfigs, blocks, strict=True):
    axes = panel_grid(subfig, 3, rollout, stills, f"{name} (log scale)")
    for ax, joint in zip(axes.ravel(), PANEL_JOINTS, strict=True):
      ax.plot(rollout.t, ratio[:, rollout.column(joint)], color=color, lw=1.0)
      ax.axhline(1.0, color="k", ls="--", lw=0.7)
      ax.set_yscale("log")
      ax.set_ylim(0.25, 4.0)
      ax.set_yticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
      ax.set_title(f"{joint}: {name}", fontsize=9)
    handles = [
      Line2D([], [], color=color, lw=1.0, label=f"policy {short} multiplier {name}"),
      Line2D([], [], color="k", ls="--", lw=0.7, label=f"nominal {short} (1)"),
      *phase_handles(stills),
    ]
    subfig.legend(
      handles=handles, loc="outside upper center", ncol=len(handles), fontsize=8
    )
  path = out_dir / f"gains_{tag}.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  return path


def plot_heatmap(
  rollout: Rollout, clip: str, tag: str, stills: list[float], out_dir: Path
) -> Path:
  maps = [
    (
      "log m = ½·ln(kp/kp0)",
      "natural-frequency multiplier m (kp = m² kp0); red = stiffer than nominal, "
      "blue = softer",
      rollout.log_m,
    )
  ]
  if rollout.task.endswith("FreqDamp"):
    maps.append(
      (
        "log r = ln(kd/kd0) − log m",
        "damping-ratio multiplier r; red = more damped than nominal",
        rollout.log_r,
      )
    )
  fig = plt.figure(figsize=(14, 7.5 * len(maps) + 0.5), layout="constrained")
  # Holds the legend: a figure-level one would overlap the figure's title.
  panel = fig.subfigures(1, 1)
  axes = panel.subplots(
    2 * len(maps), 1, sharex=True, gridspec_kw={"height_ratios": [1, 24] * len(maps)}
  )
  order = [rollout.column(joint) for joint in HEATMAP_JOINTS]
  dt = rollout.t[1] - rollout.t[0]
  for (name, meaning, values), band, ax in zip(
    maps, axes[0::2], axes[1::2], strict=True
  ):
    shade_phases(band, rollout, [])
    band.set_yticks([])
    band.set_ylabel("phase", rotation=0, ha="right", va="center", fontsize=8)
    band.set_title(f"{name.split(' =')[0]}: {meaning}", fontsize=10)
    image = ax.imshow(
      values[:, order].T,
      aspect="auto",
      cmap="RdBu_r",
      vmin=-LOG_RANGE,
      vmax=LOG_RANGE,
      extent=(rollout.t[0] - dt, rollout.t[-1], len(order) - 0.5, -0.5),
      interpolation="nearest",
    )
    ax.set_yticks(range(len(order)), HEATMAP_JOINTS, fontsize=7)
    ax.set_ylabel("joint")
    for still in stills:
      ax.axvline(still, color="k", lw=0.5, alpha=0.6)
    panel.colorbar(image, ax=[band, ax], label=f"{name} (bounds ±ln 2)", pad=0.01)
  fig.suptitle(
    f"{rollout.label} run {rollout.run}, clip {clip}: gain multipliers chosen by the "
    "policy, all joints over time",
    fontsize=11,
  )
  panel.legend(
    handles=phase_handles(stills), loc="outside upper center", ncol=4, fontsize=9
  )
  axes[-1].set_xlabel("clip time (s)")
  axes[-1].set_xlim(0, rollout.t[-1])
  path = out_dir / f"gains_heatmap_{tag}.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  return path


def plot_torques(
  rollout: Rollout,
  baseline: Rollout | None,
  clip: str,
  tag: str,
  stills: list[float],
  out_dir: Path,
) -> Path:
  fig = plt.figure(figsize=(15, 7.6), layout="constrained")
  # Holds the legend: a figure-level one would overlap the figure's title.
  panel = fig.subfigures(1, 1)
  axes = panel_grid(panel, 3, rollout, stills, "torque (Nm)")
  for ax, joint in zip(axes.ravel(), PANEL_JOINTS, strict=True):
    column = rollout.column(joint)
    if baseline is not None:
      ax.plot(
        baseline.t,
        baseline.torque[:, baseline.column(joint)],
        color=BASELINE_COLOR,
        lw=0.6,
      )
    ax.plot(rollout.t, rollout.torque[:, column], color=VIC_COLOR, lw=1.0)
    limit = rollout.torque_limit[column]
    for sign in (-1, 1):
      ax.axhline(sign * limit, color="k", ls="--", lw=0.7)
    ax.set_ylim(-1.1 * limit, 1.1 * limit)
    ax.set_title(f"{joint}: torque, limit ±{limit:g} Nm", fontsize=9)
  handles = [
    Line2D([], [], color=VIC_COLOR, lw=1.0, label=f"{rollout.label} (this run)")
  ]
  if baseline is not None:
    handles.append(
      Line2D(
        [],
        [],
        color=BASELINE_COLOR,
        lw=0.6,
        label=f"original tracker (fixed gains), run {baseline.run}",
      )
    )
  handles += [
    Line2D([], [], color="k", ls="--", lw=0.7, label="torque limit"),
    *phase_handles(stills),
  ]
  fig.suptitle(
    f"{rollout.label} run {rollout.run}, clip {clip}: actuator torques "
    "(left leg and waist)",
    fontsize=11,
  )
  panel.legend(handles=handles, loc="outside upper center", ncol=4, fontsize=9)
  path = out_dir / f"torques_{tag}.png"
  fig.savefig(path, dpi=DPI)
  plt.close(fig)
  return path


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--clip", default="amass_cmu_88_06")
  parser.add_argument(
    "--tag",
    nargs="+",
    default=["vicfreq", "vicfreqdamp"],
    help="run tags of the rollout npz files",
  )
  parser.add_argument(
    "--stills",
    type=float,
    nargs="*",
    default=[0.8, 1.9, 3.0],
    help="still times to mark (s)",
  )
  parser.add_argument(
    "--rollouts-dir",
    type=Path,
    default=ROLLOUTS_DIR,
    help="<clip>/<tag>.npz from render_viser.py",
  )
  parser.add_argument(
    "--out-dir", type=Path, help="default: <data root>/figures/<clip>"
  )
  args = parser.parse_args()
  out_dir = args.out_dir or FIGURES_DIR / args.clip
  out_dir.mkdir(parents=True, exist_ok=True)
  baseline_path = args.rollouts_dir / args.clip / "original.npz"
  baseline = Rollout(baseline_path) if baseline_path.exists() else None
  if baseline is None:
    print(f"no {baseline_path}: torque figures without the original tracker's trace")
  for tag in args.tag:
    rollout = Rollout(args.rollouts_dir / args.clip / f"{tag}.npz")
    for path in (
      plot_gains(rollout, args.clip, tag, args.stills, out_dir),
      plot_heatmap(rollout, args.clip, tag, args.stills, out_dir),
      plot_torques(rollout, baseline, args.clip, tag, args.stills, out_dir),
    ):
      print(path)


if __name__ == "__main__":
  main()
