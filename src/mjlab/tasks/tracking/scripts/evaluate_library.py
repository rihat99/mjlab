"""Evaluate a tracking policy on every clip of a motion library.

Protocol of the SuperMimic paper (App. F.1): each clip is played once from its first
frame, one env per clip, with the task's play cfg (no observation noise, no domain
randomization, foot friction 1.0, no reset noise) and a deterministic policy. An
episode ends at the clip end (the ``motion_end`` time-out) or at a training
termination, which counts as failed. Normalised return = sum of rewards / (max step
reward x (clip frames - 1)): an episode from frame 0 scores frames 1 .. last. The
per-survived-step normalised return divides by the steps the episode lasted instead
(the metric of CuboiLeo/SuperMimic-Reproduced); both are equal on a clip that did not
fail. Aggregates: the mean over rollouts (headline) and the frame-weighted mean (sum
of returns / sum of maximal returns).

Each rollout record also holds the terminations that fired (``fired``; the record's
``steps`` is then the frame they fired at) and the return split by reward term
(``reward_terms``). Diagnostic variants: another anchor body (old checkpoints), the
fall test on another body, self-collisions off, several rollouts per clip with the
training-time observation and reset noise.

A library with more rollouts than ``--num-envs`` is played in waves of at most that
many, longest clips first so that the rollouts of a wave end at similar steps; an env
that finishes its clip early replays it unscored until the wave ends. The results
JSON (written compact) holds the aggregates and one record per rollout.

  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --checkpoint-file <run>/model_4000.pt --motion-files "('a.npz', 'b.npz')"
  # large library: a .txt file listing the npz paths, one per line
  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --checkpoint-file <run>/model_4000.pt --motion-files library.txt
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import (
  MotionCommandCfg,
  bad_anchor_ori,
  bad_anchor_pos_z_only,
)
from mjlab.tasks.tracking.mdp.commands import MotionCommand, expand_motion_files
from mjlab.utils.torch import configure_torch_backends

# Failed rollouts listed on stdout; all of them are in the JSON.
MAX_PRINTED_FAILURES = 50


@dataclass(frozen=True)
class EvaluateLibraryConfig:
  """Configuration for library evaluation."""

  checkpoint_file: str
  """Local checkpoint to evaluate (e.g. '<run_dir>/model_4000.pt')."""
  # tuple first, as MotionCommandCfg.motion_file: a single path parses as a str.
  motion_files: tuple[str, ...] | str
  """Motion npz files of the library as a Python tuple literal, or a .txt file listing
  them (one path per line)."""
  num_envs: int = 4096
  """Parallel envs; more rollouts than this are played in waves."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_json: str | None = None
  """Path of the results JSON. Defaults to 'eval_library_<checkpoint stem>.json'
  next to the checkpoint."""
  anchor_body: str | None = None
  """Motion anchor body (goal observation, anchor rewards, relative body targets).
  Defaults to the task cfg's; a checkpoint needs the anchor it was trained with
  ('torso_link' for SuperMimic runs before the cfg moved to 'pelvis')."""
  fall_body: str | None = None
  """Motion body of the fall test (height and tilt terminations). Defaults to the
  anchor body."""
  self_collision: bool = True
  """False zeroes the robot geoms' contype: no self-contacts, ground contacts stay."""
  rollouts_per_clip: int = 1
  """Rollouts of each clip, run in parallel (rollout r of clip c is item r x clips +
  c)."""
  train_noise: bool = False
  """Training-time observation noise and reset noise (root pose and velocity, joint
  positions); the other play settings stay."""


def make_library_env(
  task_id: str,
  motion_files: Sequence[str],
  device: str,
  clip_actions: float | None = None,
  anchor_body: str | None = None,
  fall_body: str | None = None,
  self_collision: bool = True,
  rollouts_per_clip: int = 1,
  train_noise: bool = False,
  num_envs: int | None = None,
) -> RslRlVecEnvWrapper:
  """Play-cfg env of ``task_id`` on the library ``motion_files`` (see
  ``expand_motion_files``), with ``num_envs`` envs (default one per rollout), env i on
  clip i mod clips until ``evaluate_library`` assigns the waves. The other arguments
  are those of ``EvaluateLibraryConfig``.
  """
  env_cfg = load_env_cfg(task_id, play=True)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")
  if "motion_end" not in env_cfg.terminations:
    raise ValueError(f"Task {task_id} has no 'motion_end' termination.")
  for body in (anchor_body, fall_body):
    if body is not None and body not in motion_cmd.body_names:
      raise ValueError(f"{body!r} is not a motion body: {motion_cmd.body_names}.")
  motion_cmd.motion_file = tuple(motion_files)
  motion_cmd.sampling_mode = "start"
  if anchor_body is not None:
    motion_cmd.anchor_body_name = anchor_body
  if fall_body is not None:
    for term in env_cfg.terminations.values():
      if term.func in (bad_anchor_pos_z_only, bad_anchor_ori):
        term.params["body_name"] = fall_body
  if not self_collision:
    robot = env_cfg.scene.entities[motion_cmd.entity_name]
    robot.collisions = tuple(replace(c, contype=0) for c in robot.collisions)
  if train_noise:
    train_cfg = load_env_cfg(task_id, play=False)
    train_cmd = train_cfg.commands["motion"]
    assert isinstance(train_cmd, MotionCommandCfg)
    motion_cmd.pose_range = train_cmd.pose_range
    motion_cmd.velocity_range = train_cmd.velocity_range
    motion_cmd.joint_position_range = train_cmd.joint_position_range
    for name, group in env_cfg.observations.items():
      group.enable_corruption = train_cfg.observations[name].enable_corruption
  rollouts = len(expand_motion_files(tuple(motion_files))) * rollouts_per_clip
  env_cfg.scene.num_envs = min(rollouts, num_envs or rollouts)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=clip_actions)


def evaluate_library(
  env: RslRlVecEnvWrapper,
  policy: Callable[[TensorDict], torch.Tensor],
  rollouts_per_clip: int = 1,
) -> dict[str, Any]:
  """Run each clip ``rollouts_per_clip`` times from its first frame; return
  per-rollout and aggregate results.

  ``env`` comes from ``make_library_env``; ``policy`` maps the observation TensorDict
  to actions (e.g. a runner's ``get_inference_policy()``). Rollouts are played in
  waves of ``env.num_envs``, longest clips first. Aggregates are over all rollouts
  (each clip has the same number); ``num_failed`` counts failed rollouts.
  """
  unwrapped = env.unwrapped
  command = cast(MotionCommand, unwrapped.command_manager.get_term("motion"))
  motion = command.motion
  num_clips = len(motion.clip_names)
  num_envs = env.num_envs
  # Largest reward of one step: all tracking terms at 1, penalties at 0.
  max_step_reward = unwrapped.step_dt * sum(
    term.weight for term in unwrapped.cfg.rewards.values() if term.weight > 0
  )

  rewards = unwrapped.reward_manager
  terminations = unwrapped.termination_manager
  fall_terms = [
    name
    for name in terminations.active_terms
    if not terminations.get_term_cfg(name).time_out
  ]

  device = env.device
  num_items = num_clips * rollouts_per_clip
  items = torch.arange(num_items, device=device)
  item_clip = items % num_clips
  frames = (motion.clip_end - motion.clip_start)[item_clip]
  total_reward = torch.zeros(num_items, device=device)
  term_reward = torch.zeros(num_items, len(rewards.active_terms), device=device)
  steps = torch.zeros(num_items, dtype=torch.long, device=device)
  failed = torch.zeros(num_items, dtype=torch.bool, device=device)
  fired = torch.zeros(num_items, len(fall_terms), dtype=torch.bool, device=device)
  order = torch.argsort(frames, descending=True, stable=True)
  env_ids = torch.arange(num_envs, device=device)
  start_time = time.time()
  env_steps = 0
  for first in range(0, num_items, num_envs):
    wave_items = order[first : first + num_envs]
    count = len(wave_items)
    # Envs past the last rollout of the final wave replay its clips, unscored.
    command.start_clip[:] = item_clip[wave_items[env_ids % count]]
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
    total_reward[wave_items] = wave_reward[:count]
    term_reward[wave_items] = wave_terms[:count]
    steps[wave_items] = wave_steps[:count]
    failed[wave_items] = wave_failed[:count]
    fired[wave_items] = wave_fired[:count]
    print(
      f"[INFO] {first + count}/{num_items} rollouts done, {int(failed.sum())} failed "
      f"({env_steps} steps, {time.time() - start_time:.0f} s)",
      flush=True,
    )

  max_return = max_step_reward * (frames - 1)
  normalized = total_reward / max_return
  survived = total_reward / (max_step_reward * steps)
  # One host copy per column: per-element reads of CUDA tensors sync 129k times each.
  columns = {
    "clip": [motion.clip_names[c] for c in item_clip.tolist()],
    "rollout": (items // num_clips).tolist(),
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
  failure_fraction = (
    torch.zeros(num_clips, device=device).index_add_(0, item_clip, failed.float())
    / rollouts_per_clip
  )
  return {
    "num_clips": num_clips,
    "rollouts_per_clip": rollouts_per_clip,
    "num_failed": int(failed.sum()),
    "clip_failure_fraction": dict(
      zip(motion.clip_names, failure_fraction.tolist(), strict=True)
    ),
    "mean_normalized_return": float(normalized.mean()),
    "frame_weighted_normalized_return": float(total_reward.sum() / max_return.sum()),
    "mean_normalized_return_survived": float(survived.mean()),
    "frame_weighted_normalized_return_survived": float(
      total_reward.sum() / (max_step_reward * steps.sum())
    ),
    "max_step_reward": max_step_reward,
    "num_envs": num_envs,
    "env_steps": env_steps,
    "seconds": time.time() - start_time,
    "clips": clips,
  }


def run_evaluate_library(task_id: str, cfg: EvaluateLibraryConfig) -> dict[str, Any]:
  """Load the checkpoint with the task's runner and evaluate it on the library."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = Path(cfg.checkpoint_file)
  motion_files = (
    (cfg.motion_files,) if isinstance(cfg.motion_files, str) else cfg.motion_files
  )
  for path in (checkpoint, *map(Path, motion_files)):
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
    anchor_body=cfg.anchor_body,
    fall_body=cfg.fall_body,
    self_collision=cfg.self_collision,
    rollouts_per_clip=cfg.rollouts_per_clip,
    train_noise=cfg.train_noise,
    num_envs=cfg.num_envs,
  )
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  result = evaluate_library(env, policy, cfg.rollouts_per_clip)
  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  anchor_body = command.cfg.anchor_body_name
  # Effective value: the task cfg itself may already have self-collisions off.
  robot = env.unwrapped.cfg.scene.entities[command.cfg.entity_name]
  self_collision = any(
    any(c.contype.values()) if isinstance(c.contype, dict) else c.contype
    for c in robot.collisions
  )
  result["provenance"] = {
    "task": task_id,
    "checkpoint_file": str(checkpoint.resolve()),
    "motion_files": [str(Path(f).resolve()) for f in motion_files],
    "anchor_body": anchor_body,
    "fall_body": cfg.fall_body or anchor_body,
    "self_collision": self_collision,
    "train_noise": cfg.train_noise,
  }
  output = Path(
    cfg.output_json or checkpoint.with_name(f"eval_library_{checkpoint.stem}.json")
  )
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
      f"[INFO] failed: {record['clip']} rollout {record['rollout']} at frame "
      f"{record['steps']}: {', '.join(record['fired'])}"
    )
  if len(failures) > MAX_PRINTED_FAILURES:
    print(f"[INFO] ... {len(failures) - MAX_PRINTED_FAILURES} more failed rollouts")
  env.close()
  return result


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
