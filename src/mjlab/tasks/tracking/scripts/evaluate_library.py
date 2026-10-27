"""Evaluate a tracking policy on every clip of a motion library.

Protocol of the SuperMimic paper (App. F.1): each clip is played once from its first
frame, one env per clip, with the task's play cfg (no observation noise, no domain
randomization, foot friction 1.0, no reset noise) and a deterministic policy. An
episode ends at the clip end (the ``motion_end`` time-out) or at a training
termination, which counts as failed. Normalised return = sum of rewards / (max step
reward x (clip frames - 1)): an episode from frame 0 scores frames 1 .. last. The
per-survived-step normalised return divides by the steps the episode lasted instead
(the metric of CuboiLeo/SuperMimic-Reproduced); both are equal on a clip that did not
fail. Aggregates: the mean over clips (headline) and the frame-weighted mean (sum of
returns / sum of maximal returns).

Each clip record also holds the terminations that fired (``fired``; the record's
``steps`` is then the frame they fired at) and the return split by reward term
(``reward_terms``).

A library with more clips than ``--num-envs`` is played in waves of at most that many,
longest first so that the clips of a wave end at similar steps; an env that finishes
its clip early replays it unscored until the wave ends. The results JSON (written
compact) holds the aggregates and one record per clip.

Several checkpoints are evaluated in one process, on one env (the library loads once):
a tuple of ``--checkpoint-file`` paths, or ``--run-dir`` for every ``model_<it>.pt`` of
a run that has no results JSON yet (``--overwrite``: every one). ``--watch`` keeps
polling the run dir while it trains and evaluates each new checkpoint, until the
training has ended (``--pid`` exited, or without it no new checkpoint or metrics.csv
row for ``--stale-minutes``) and every checkpoint is evaluated. Each checkpoint gets
its own JSON, which also records the checkpoint's iteration, file mtime and the run's
training wall-clock at that iteration (``training_seconds``), and in its
``provenance`` the git commit of the evaluating code (with a ``dirty`` flag) and the
robot's collision setting in the play cfg (``self_collision``, ``collisions``).

  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --checkpoint-file <run>/model_4000.pt --motion-files "('a.npz', 'b.npz')"
  # large library: a .txt file listing the npz paths, one per line
  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --checkpoint-file <run>/model_4000.pt --motion-files library.txt
  # every checkpoint of a run (or --checkpoint-file "('<run>/a.pt', '<run>/b.pt')")
  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --run-dir <run> --motion-files library.txt
  # and every checkpoint the training process <pid> writes until it ends
  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --run-dir <run> --motion-files library.txt --watch True --pid <pid>
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.local_log_writer import TRAINING_SECONDS
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, expand_motion_files
from mjlab.tasks.tracking.mdp.rewards import max_step_reward
from mjlab.utils.torch import configure_torch_backends

# Failed clips listed on stdout; all of them are in the JSON.
MAX_PRINTED_FAILURES = 50
CHECKPOINT_NAME = re.compile(r"model_(\d+)\.pt")
# Training time of runs whose metrics.csv predates the TRAINING_SECONDS column.
ITERATION_TIMES = "Perf/collection_time + Perf/learning_time"
# --watch reads a checkpoint of a running training once its file is this old (s):
# torch.save writes it in place.
SETTLE_SECONDS = 30.0


@dataclass(frozen=True)
class EvaluateLibraryConfig:
  """Configuration for library evaluation."""

  # tuple first, as MotionCommandCfg.motion_file: a single path parses as a str.
  motion_files: tuple[str, ...] | str
  """Motion npz files of the library as a Python tuple literal, or a .txt file listing
  them (one path per line)."""
  checkpoint_file: tuple[str, ...] | str = ()
  """Local checkpoint to evaluate (e.g. '<run_dir>/model_4000.pt'), or a Python tuple
  literal of several, evaluated in that order."""
  run_dir: str | None = None
  """Evaluate every 'model_<it>.pt' of this run directory that has no results JSON yet,
  in iteration order, instead of --checkpoint-file."""
  overwrite: bool = False
  """With --run-dir: also evaluate the checkpoints that already have a results JSON."""
  watch: bool = False
  """With --run-dir: keep evaluating the checkpoints the training writes, until it
  has ended and every checkpoint is evaluated."""
  pid: int | None = None
  """--watch: the training process; the training has ended when it exits."""
  stale_minutes: float = 30.0
  """--watch without --pid: the training has ended when no model_<it>.pt or
  metrics.csv of the run (else the watch itself) is younger than this."""
  poll_seconds: float = 60.0
  """--watch: seconds between two looks at the run dir."""
  num_envs: int = 4096
  """Parallel envs; more clips than this are played in waves."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_json: str | None = None
  """Path of the results JSON of a single checkpoint. Defaults to
  'eval_library_<checkpoint stem>.json' next to each checkpoint."""


def make_library_env(
  task_id: str,
  motion_files: Sequence[str],
  device: str,
  clip_actions: float | None = None,
  num_envs: int | None = None,
) -> RslRlVecEnvWrapper:
  """Play-cfg env of ``task_id`` on the library ``motion_files`` (see
  ``expand_motion_files``), with ``num_envs`` envs (default one per clip), env i on
  clip i mod clips until ``evaluate_library`` assigns the waves.
  """
  env_cfg = load_env_cfg(task_id, play=True)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")
  if "motion_end" not in env_cfg.terminations:
    raise ValueError(f"Task {task_id} has no 'motion_end' termination.")
  motion_cmd.motion_file = tuple(motion_files)
  motion_cmd.sampling_mode = "start"
  clips = len(expand_motion_files(tuple(motion_files)))
  env_cfg.scene.num_envs = min(clips, num_envs or clips)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=clip_actions)


def evaluate_library(
  env: RslRlVecEnvWrapper, policy: Callable[[TensorDict], torch.Tensor]
) -> dict[str, Any]:
  """Play each clip once from its first frame; return per-clip and aggregate results.

  ``env`` comes from ``make_library_env``; ``policy`` maps the observation TensorDict
  to actions (e.g. a runner's ``get_inference_policy()``). Clips are played in waves
  of ``env.num_envs``, longest first; ``num_failed`` counts failed clips.
  """
  unwrapped = env.unwrapped
  command = cast(MotionCommand, unwrapped.command_manager.get_term("motion"))
  motion = command.motion
  num_clips = len(motion.clip_names)
  num_envs = env.num_envs
  max_step = max_step_reward(unwrapped)

  rewards = unwrapped.reward_manager
  terminations = unwrapped.termination_manager
  fall_terms = [
    name
    for name in terminations.active_terms
    if not terminations.get_term_cfg(name).time_out
  ]

  device = env.device
  frames = motion.clip_end - motion.clip_start
  total_reward = torch.zeros(num_clips, device=device)
  term_reward = torch.zeros(num_clips, len(rewards.active_terms), device=device)
  steps = torch.zeros(num_clips, dtype=torch.long, device=device)
  failed = torch.zeros(num_clips, dtype=torch.bool, device=device)
  fired = torch.zeros(num_clips, len(fall_terms), dtype=torch.bool, device=device)
  order = torch.argsort(frames, descending=True, stable=True)
  env_ids = torch.arange(num_envs, device=device)
  start_time = time.time()
  env_steps = 0
  for first in range(0, num_clips, num_envs):
    wave_clips = order[first : first + num_envs]
    count = len(wave_clips)
    # Envs past the last clip of the final wave replay its clips, unscored.
    command.start_clip[:] = wave_clips[env_ids % count]
    wave_reward = torch.zeros(num_envs, device=device)
    wave_terms = torch.zeros(num_envs, len(rewards.active_terms), device=device)
    wave_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    wave_failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    wave_fired = torch.zeros(num_envs, len(fall_terms), dtype=torch.bool, device=device)
    done = env_ids >= count
    obs, _ = env.reset()
    while not done.all():
      with torch.no_grad():
        actions = policy(obs)
      obs, reward, dones, _ = env.step(actions)
      env_steps += 1
      active = ~done
      wave_reward += torch.where(active, reward, 0.0)
      # Per-term reward rates of this step (scaled by dt as the reward is).
      step_terms = rewards._step_reward * unwrapped.step_dt
      wave_terms += torch.where(active[:, None], step_terms, 0.0)
      wave_steps += active.long()
      newly_done = dones.bool() & active
      newly_failed = newly_done & terminations.terminated
      wave_failed |= newly_failed
      fall_flags = torch.stack([terminations.get_term(name) for name in fall_terms], 1)
      wave_fired |= newly_failed[:, None] & fall_flags
      done |= newly_done
    total_reward[wave_clips] = wave_reward[:count]
    term_reward[wave_clips] = wave_terms[:count]
    steps[wave_clips] = wave_steps[:count]
    failed[wave_clips] = wave_failed[:count]
    fired[wave_clips] = wave_fired[:count]
    print(
      f"[INFO] {first + count}/{num_clips} clips done, {int(failed.sum())} failed "
      f"({env_steps} steps, {time.time() - start_time:.0f} s)",
      flush=True,
    )

  max_return = max_step * (frames - 1)
  normalized = total_reward / max_return
  survived = total_reward / (max_step * steps)
  # One host copy per column: per-element reads of CUDA tensors sync 129k times each.
  columns = {
    "clip": list(motion.clip_names),
    "frames": frames.tolist(),
    "steps": steps.tolist(),
    "return": total_reward.tolist(),
    "normalized_return": normalized.tolist(),
    "normalized_return_survived": survived.tolist(),
    "failed": failed.tolist(),
    "fired": [
      [name for name, hit in zip(fall_terms, row, strict=True) if hit]
      for row in fired.tolist()
    ],
    "reward_terms": [
      dict(zip(rewards.active_terms, row, strict=True)) for row in term_reward.tolist()
    ],
  }
  clips = [
    dict(zip(columns, row, strict=True)) for row in zip(*columns.values(), strict=True)
  ]
  return {
    "num_clips": num_clips,
    "num_failed": int(failed.sum()),
    "mean_normalized_return": float(normalized.mean()),
    "frame_weighted_normalized_return": float(total_reward.sum() / max_return.sum()),
    "mean_normalized_return_survived": float(survived.mean()),
    "frame_weighted_normalized_return_survived": float(
      total_reward.sum() / (max_step * steps.sum())
    ),
    "max_step_reward": max_step,
    "num_envs": num_envs,
    "env_steps": env_steps,
    "seconds": time.time() - start_time,
    "clips": clips,
  }


def results_json(checkpoint: Path) -> Path:
  """The default results JSON of ``checkpoint``: ``eval_library_<stem>.json`` next to
  it."""
  return checkpoint.with_name(f"eval_library_{checkpoint.stem}.json")


def checkpoint_files(cfg: EvaluateLibraryConfig) -> list[Path]:
  """The checkpoints of ``cfg``: every ``model_<it>.pt`` of ``cfg.run_dir`` by
  iteration without those that have a results JSON (unless ``cfg.overwrite``), else
  ``cfg.checkpoint_file`` in the given order."""
  given = (
    (cfg.checkpoint_file,)
    if isinstance(cfg.checkpoint_file, str)
    else cfg.checkpoint_file
  )
  if (cfg.run_dir is None) == (not given):
    raise ValueError("Give either --checkpoint-file or --run-dir.")
  if cfg.run_dir is None:
    if cfg.watch:
      raise ValueError("--watch needs --run-dir.")
    return [Path(path) for path in given]
  found = {
    int(match.group(1)): path
    for path in Path(cfg.run_dir).glob("model_*.pt")
    if (match := CHECKPOINT_NAME.fullmatch(path.name))
  }
  if not found and not cfg.watch:
    raise FileNotFoundError(f"No model_<it>.pt in {cfg.run_dir}")
  return [
    found[iteration]
    for iteration in sorted(found)
    if cfg.overwrite or not results_json(found[iteration]).is_file()
  ]


def training_running(
  run_dir: Path, pid: int | None, stale_minutes: float, since: float
) -> bool:
  """Whether the training of ``run_dir`` still runs: process ``pid`` exists, or
  without ``pid``, the newest ``model_<it>.pt`` or ``metrics.csv`` of the run (none
  yet: the time ``since``) is younger than ``stale_minutes``."""
  if pid is not None:
    return Path(f"/proc/{pid}").exists()
  outputs = [*run_dir.glob("model_*.pt"), run_dir / "metrics.csv"]
  newest = max(
    (path.stat().st_mtime for path in outputs if path.is_file()), default=since
  )
  return time.time() - newest < stale_minutes * 60


def watched_checkpoints(cfg: EvaluateLibraryConfig) -> Iterator[Path]:
  """``--watch``: the checkpoints of ``checkpoint_files(cfg)``, then each one the
  training writes later (once ``SETTLE_SECONDS`` old), polled every
  ``cfg.poll_seconds``; ends when the training has ended (``training_running``) and
  every checkpoint was yielded. The caller writes a checkpoint's JSON before taking
  the next one."""
  assert cfg.run_dir is not None
  start = time.time()
  yielded: set[Path] = set()
  while True:
    running = training_running(Path(cfg.run_dir), cfg.pid, cfg.stale_minutes, start)
    settled = time.time() - SETTLE_SECONDS
    ready = [
      path
      for path in checkpoint_files(cfg)
      if path not in yielded and (not running or path.stat().st_mtime < settled)
    ]
    if not ready and not running:
      return
    if not ready:
      time.sleep(cfg.poll_seconds)
    for path in ready:
      yielded.add(path)
      yield path


def training_seconds(checkpoint: Path) -> tuple[float | None, str | None]:
  """Training wall-clock of the run at ``model_<it>.pt`` and the column it comes from.

  Read from ``metrics.csv`` next to the checkpoint, on the last row logged before the
  save (iteration it - 1). A run logged before the ``TRAINING_SECONDS`` column existed
  gives the sum of its logged iteration times (``ITERATION_TIMES``, without the replay
  prefill and checkpoint saves). ``(None, None)`` without metrics.csv or for another
  file name.
  """
  match = CHECKPOINT_NAME.fullmatch(checkpoint.name)
  metrics = checkpoint.parent / "metrics.csv"
  if match is None or not metrics.is_file():
    return None, None
  with open(metrics, newline="") as handle:
    rows = [
      row
      for row in csv.DictReader(handle)
      if row["iteration"] and int(row["iteration"]) < int(match.group(1))
    ]
  if not rows:
    return None, None
  if rows[-1].get(TRAINING_SECONDS):
    return float(rows[-1][TRAINING_SECONDS]), TRAINING_SECONDS
  iteration_times = sum(
    float(row["Perf/collection_time"]) + float(row["Perf/learning_time"])
    for row in rows
  )
  return iteration_times, ITERATION_TIMES


def source_commit() -> dict[str, str | bool | None]:
  """``commit``: HEAD of the git checkout this file runs from; ``dirty``: whether its
  tracked files differ from HEAD. Both None when no checkout tracks this file (an
  export or an installed package)."""
  here = Path(__file__).resolve()

  def git(*args: str) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(here.parent), *args]
    return subprocess.run(command, capture_output=True, text=True)

  if git("ls-files", "--error-unmatch", here.name).returncode != 0:
    return {"commit": None, "dirty": None}
  status = git("status", "--porcelain", "--untracked-files=no")
  return {
    "commit": git("rev-parse", "HEAD").stdout.strip(),
    "dirty": bool(status.stdout.strip()),
  }


def self_collision(env: ManagerBasedRlEnv, entity_name: str) -> bool:
  """Whether two geoms of the entity on different bodies pass MuJoCo's contype /
  conaffinity filter in the compiled model (parent-child and excluded pairs aside)."""
  model = env.sim.mj_model
  geoms = env.scene[entity_name].indexing.geom_ids.cpu().numpy()
  contype, conaffinity = model.geom_contype[geoms], model.geom_conaffinity[geoms]
  bodies = model.geom_bodyid[geoms]
  can_touch = (contype[:, None] & conaffinity[None, :]) != 0
  return bool((can_touch & (bodies[:, None] != bodies[None, :])).any())


def run_evaluate_library(
  task_id: str, cfg: EvaluateLibraryConfig
) -> list[dict[str, Any]]:
  """Load the library once and evaluate each checkpoint of ``cfg`` on it with the
  task's runner (``--watch``: also those the training writes later); return the
  results in evaluation order."""
  configure_torch_backends()
  source = source_commit()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoints = checkpoint_files(cfg)
  if cfg.output_json is not None and (len(checkpoints) > 1 or cfg.watch):
    raise ValueError("--output-json needs a single checkpoint.")
  if not checkpoints and not cfg.watch:
    print(
      f"[INFO] every checkpoint of {cfg.run_dir} has its results JSON "
      "(--overwrite True evaluates them again)"
    )
    return []
  motion_files = (
    (cfg.motion_files,) if isinstance(cfg.motion_files, str) else cfg.motion_files
  )
  for path in (*checkpoints, *map(Path, motion_files)):
    if not path.is_file():
      raise FileNotFoundError(f"File not found: {path}")
  missing = [f for f in expand_motion_files(motion_files) if not Path(f).is_file()]
  if missing:
    raise FileNotFoundError(f"{len(missing)} motion files not found: {missing[:5]}")

  agent_cfg = load_rl_cfg(task_id)
  env = make_library_env(
    task_id,
    motion_files,
    device,
    agent_cfg.clip_actions,
    num_envs=cfg.num_envs,
  )
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  unwrapped = env.unwrapped
  command_cfg = cast(MotionCommandCfg, unwrapped.cfg.commands["motion"])
  entity_name = command_cfg.entity_name
  collisions = {
    "self_collision": self_collision(unwrapped, entity_name),
    "collisions": [
      asdict(c) for c in unwrapped.cfg.scene.entities[entity_name].collisions
    ],
  }

  results = []
  for checkpoint in watched_checkpoints(cfg) if cfg.watch else checkpoints:
    print(f"[INFO] {time.strftime('%H:%M:%S')} evaluating {checkpoint}", flush=True)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    result = evaluate_library(env, policy)
    match = CHECKPOINT_NAME.fullmatch(checkpoint.name)
    seconds, seconds_from = training_seconds(checkpoint)
    result |= {
      "iteration": int(match.group(1)) if match else None,
      "checkpoint_mtime": checkpoint.stat().st_mtime,
      "training_seconds": seconds,
      "training_seconds_from": seconds_from,
      "provenance": {
        "task": task_id,
        "checkpoint_file": str(checkpoint.resolve()),
        "motion_files": [str(Path(f).resolve()) for f in motion_files],
        **source,
        **collisions,
      },
    }
    output = Path(cfg.output_json or results_json(checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))
    print(
      f"[INFO] normalised return per clip length: mean "
      f"{result['mean_normalized_return']:.4f}, frame-weighted "
      f"{result['frame_weighted_normalized_return']:.4f}; per survived step: mean "
      f"{result['mean_normalized_return_survived']:.4f}, frame-weighted "
      f"{result['frame_weighted_normalized_return_survived']:.4f}; "
      f"failed {result['num_failed']}/{len(result['clips'])}; saved {output}"
    )
    failures = [record for record in result["clips"] if record["failed"]]
    for record in failures[:MAX_PRINTED_FAILURES]:
      print(
        f"[INFO] failed: {record['clip']} at frame {record['steps']}: "
        f"{', '.join(record['fired'])}"
      )
    if len(failures) > MAX_PRINTED_FAILURES:
      print(f"[INFO] ... {len(failures) - MAX_PRINTED_FAILURES} more failed clips")
    results.append(result)
  env.close()
  return results


def main():
  import mjlab.tasks  # noqa: F401

  tracking_tasks = [t for t in list_tasks() if "Tracking" in t]
  if not tracking_tasks:
    print("No tracking tasks found.")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    EvaluateLibraryConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_evaluate_library(chosen_task, args)


if __name__ == "__main__":
  main()
