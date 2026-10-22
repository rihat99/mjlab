"""Evaluate a tracking policy on every clip of a motion library.

Protocol of the SuperMimic paper (App. F.1): each clip is played once from its first
frame, one env per clip, with the task's play cfg (no observation noise, no domain
randomization, foot friction 1.0, no reset noise) and a deterministic policy. An
episode ends at the clip end (the ``motion_end`` time-out) or at a training
termination, which counts as failed. Normalised return = sum of rewards / (max step
reward x (clip frames - 1)): an episode from frame 0 scores frames 1 .. last.

  python -m mjlab.tasks.tracking.scripts.evaluate_library <task> \
    --checkpoint-file <run>/model_4000.pt --motion-files "('a.npz', 'b.npz')"
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateLibraryConfig:
  """Configuration for library evaluation."""

  checkpoint_file: str
  """Local checkpoint to evaluate (e.g. '<run_dir>/model_4000.pt')."""
  motion_files: tuple[str, ...]
  """Motion npz files of the library, one env each, as a Python tuple literal."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_json: str | None = None
  """Path of the results JSON. Defaults to 'eval_library_<checkpoint stem>.json'
  next to the checkpoint."""


def make_library_env(
  task_id: str,
  motion_files: Sequence[str],
  device: str,
  clip_actions: float | None = None,
) -> RslRlVecEnvWrapper:
  """Play-cfg env of ``task_id`` with one env per motion file, env i on clip i."""
  env_cfg = load_env_cfg(task_id, play=True)
  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")
  if "motion_end" not in env_cfg.terminations:
    raise ValueError(f"Task {task_id} has no 'motion_end' termination.")
  motion_cmd.motion_file = tuple(motion_files)
  motion_cmd.sampling_mode = "start"
  env_cfg.scene.num_envs = len(motion_files)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=clip_actions)


def evaluate_library(
  env: RslRlVecEnvWrapper, policy: Callable[[TensorDict], torch.Tensor]
) -> dict[str, Any]:
  """Run each env once through its clip; return per-clip and aggregate results.

  ``env`` comes from ``make_library_env``; ``policy`` maps the observation TensorDict
  to actions (e.g. a runner's ``get_inference_policy()``).
  """
  unwrapped = env.unwrapped
  command = cast(MotionCommand, unwrapped.command_manager.get_term("motion"))
  motion = command.motion
  num_clips = len(motion.clip_names)
  if env.num_envs != num_clips:
    raise ValueError(f"Need one env per clip: {env.num_envs} envs, {num_clips} clips.")
  # Largest reward of one step: all tracking terms at 1, penalties at 0.
  max_step_reward = unwrapped.step_dt * sum(
    term.weight for term in unwrapped.cfg.rewards.values() if term.weight > 0
  )

  device = env.device
  total_reward = torch.zeros(num_clips, device=device)
  steps = torch.zeros(num_clips, dtype=torch.long, device=device)
  done = torch.zeros(num_clips, dtype=torch.bool, device=device)
  failed = torch.zeros(num_clips, dtype=torch.bool, device=device)

  obs, _ = env.reset()
  while not done.all():
    with torch.no_grad():
      actions = policy(obs)
    obs, reward, dones, _ = env.step(actions)
    active = ~done
    total_reward += torch.where(active, reward, 0.0)
    steps += active.long()
    newly_done = dones.bool() & active
    failed |= newly_done & unwrapped.termination_manager.terminated
    done |= newly_done
    if newly_done.any():
      print(
        f"[INFO] {int(done.sum())}/{num_clips} clips done (step {int(steps.max())})"
      )

  frames = motion.clip_end - motion.clip_start
  max_return = max_step_reward * (frames - 1)
  normalized = total_reward / max_return
  clips = [
    {
      "clip": name,
      "frames": int(frames[c]),
      "steps": int(steps[c]),
      "return": float(total_reward[c]),
      "normalized_return": float(normalized[c]),
      "failed": bool(failed[c]),
    }
    for c, name in enumerate(motion.clip_names)
  ]
  return {
    "num_clips": num_clips,
    "num_failed": int(failed.sum()),
    "mean_normalized_return": float(normalized.mean()),
    "frame_weighted_normalized_return": float(total_reward.sum() / max_return.sum()),
    "max_step_reward": max_step_reward,
    "clips": clips,
  }


def run_evaluate_library(task_id: str, cfg: EvaluateLibraryConfig) -> dict[str, Any]:
  """Load the checkpoint with the task's runner and evaluate it on the library."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = Path(cfg.checkpoint_file)
  for path in (checkpoint, *map(Path, cfg.motion_files)):
    if not path.is_file():
      raise FileNotFoundError(f"File not found: {path}")

  agent_cfg = load_rl_cfg(task_id)
  env = make_library_env(task_id, cfg.motion_files, device, agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  result = evaluate_library(env, policy)
  result["provenance"] = {
    "task": task_id,
    "checkpoint_file": str(checkpoint.resolve()),
    "motion_files": [str(Path(f).resolve()) for f in cfg.motion_files],
  }
  output = Path(
    cfg.output_json or checkpoint.with_name(f"eval_library_{checkpoint.stem}.json")
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(result, indent=2))
  print(
    f"[INFO] mean normalised return {result['mean_normalized_return']:.4f}, "
    f"frame-weighted {result['frame_weighted_normalized_return']:.4f}, "
    f"failed {result['num_failed']}/{result['num_clips']}; saved {output}"
  )
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
