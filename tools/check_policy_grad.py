"""Gradient check of the compiled SuperMimic learner against the eager one.

A compiled policy objective once matched the eager loss to 1e-5 with a policy gradient
at cosine ~0.5 to the eager one, and every run trained with it stalled; this check
compares gradients, not losses, on the real task. It builds the SuperMimic env on a
motion library, loads a checkpoint into an eager reference learner (no compile, TF32)
and fills a replay buffer with ``--fill-steps`` sim steps of its policy.
Each candidate learner loads the same checkpoint and runs ``update_policy`` with the
reference's torch seed (same seeds, noise and batches), optimiser steps disabled; the
policy gradient is compared as stored in ``.grad`` (after both clips), and likewise the
world-model gradient after ``fit_world_model`` (its last batch). Candidates:

  compiled         the default build (``precision.compile``)
  whole_objective  the policy objective compiled as one function: the known-bad build,
                   which the check must catch

Exit status 1 when ``compiled`` has a mean policy-gradient cosine below
``--min-cosine`` or ``whole_objective`` reaches it (the check would not catch the
bug).
Single trials vary: 32-step rollouts are chaotic, TF32 alone moves single trials to
cosine ~0.995. About 10 minutes on one GPU:

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python tools/check_policy_grad.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from paths import LOG_ROOT, MOTIONS_DIR

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.supermimic.algorithm import SuperMimic
from mjlab.rl.supermimic.config import SuperMimicRunnerCfg
from mjlab.rl.supermimic.runner import SuperMimicRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

TASK = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"
CHECKPOINT = LOG_ROOT / "2026-10-23_00-33-44_final_learnedhead" / "model_14000.pt"
PARTS = {"actor": True, "critic": True, "world_model": True, "optimizer": True}
CANDIDATES = ("compiled", "whole_objective")


def log(message: str) -> None:
  """Timestamped line on stdout."""
  print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def flat_grad(module: torch.nn.Module) -> torch.Tensor:
  """All parameter gradients of ``module`` as one float64 vector."""
  grads = [p.grad.flatten() for p in module.parameters() if p.grad is not None]
  return torch.cat(grads).double()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
  """Cosine similarity of two vectors (nan when one is not finite)."""
  return float(a @ b / (a.norm() * b.norm()))


def disable_steps(alg: SuperMimic) -> None:
  """Keep the learner's weights and optimiser states fixed across calls."""
  for optimizer in (alg.policy_optimizer, alg.world_model_optimizer):
    optimizer.step = lambda *args, **kwargs: None


def main() -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
  parser.add_argument(
    "--anchor-body",
    default="torso_link",
    help="motion anchor body the checkpoint was trained with (the default "
    "checkpoint: torso_link; the task cfg now uses pelvis)",
  )
  parser.add_argument("--motions-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument("--candidates", default=",".join(CANDIDATES))
  parser.add_argument("--trials", type=int, default=3)
  parser.add_argument("--warmup-calls", type=int, default=3)
  parser.add_argument("--fill-steps", type=int, default=200)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument(
    "--analytic-termination",
    choices=("cfg", "true", "false"),
    default="cfg",
    help="policy.analytic_termination for every learner (cfg: the task default)",
  )
  parser.add_argument("--min-cosine", type=float, default=0.995)
  parser.add_argument("--out", type=Path, default=None, help="JSON of all trials")
  args = parser.parse_args()
  device = "cuda:0"
  started = time.time()

  env_cfg = load_env_cfg(TASK)
  motions = sorted(args.motions_dir.glob("*.npz"))
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = tuple(str(p) for p in motions)
  motion_cfg.anchor_body_name = args.anchor_body
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = 0
  agent_cfg = load_rl_cfg(TASK)
  assert isinstance(agent_cfg, SuperMimicRunnerCfg)
  if args.analytic_termination != "cfg":
    agent_cfg.policy.analytic_termination = args.analytic_termination == "true"
  agent_cfg.precision.compile = False
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner = SuperMimicRunner(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(str(args.checkpoint), load_cfg={**PARTS, "sampler": True}, strict=True)
  reference, iface = runner.alg, runner.interface
  disable_steps(reference)
  checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
  # fit_world_model updates the running statistics: every call starts from the
  # checkpoint's (copied in place, so the CUDA graphs stay valid).
  world_model_state = checkpoint["world_model_state_dict"]
  env.reset()
  runner._init_collection()
  # Past the normaliser freeze: the fill leaves the policy normalisers as loaded.
  runner._collect(args.fill_steps, 10**9)
  assert runner.buffer is not None
  buffer = runner.buffer
  log(
    f"{len(motions)} clips, {args.num_envs} envs, anchor {args.anchor_body}, "
    "analytic termination "
    f"{runner.policy_cfg.analytic_termination}, replay filled in "
    f"{time.time() - started:.0f} s"
  )

  def build(candidate: str) -> SuperMimic:
    alg = SuperMimic(
      iface,
      runner.world_model_cfg,
      runner.policy_cfg,
      runner.critic_cfg,
      device,
      compile=True,
      action_noise_std=runner.collection_cfg.action_noise_std,
    )
    alg.load_state_dict(checkpoint, PARTS, strict=True)
    disable_steps(alg)
    if candidate == "whole_objective":
      # The pre-fix build: reward and termination inlined into one compiled objective.
      alg._reward = iface.reward
      alg._termination = alg.termination
      alg._policy_objective = torch.compile(alg.policy_objective)
    return alg

  results: list[dict] = []
  failures: list[str] = []
  for candidate in args.candidates.split(","):
    if candidate not in CANDIDATES:
      raise ValueError(f"Unknown candidate {candidate!r}; choose from {CANDIDATES}.")
    alg = build(candidate)
    # Compile and record the CUDA graphs.
    for _ in range(args.warmup_calls):
      alg.fit_world_model(buffer)
      alg.update_policy(buffer)
    skipped = (alg.skipped_world_model_steps, alg.skipped_policy_steps)
    policy_cosines, world_model_cosines = [], []
    for trial in range(args.trials):
      grads = {}
      for name, learner in (("reference", reference), ("candidate", alg)):
        learner.world_model.load_state_dict(world_model_state)
        torch.manual_seed(1000 + trial)
        learner.fit_world_model(buffer)
        world_model_grad = flat_grad(learner.world_model)
        torch.manual_seed(2000 + trial)
        metrics, rollout, _ = learner.update_policy(buffer)
        grads[name] = (world_model_grad, flat_grad(learner.policy), rollout.rewards)
      ref, cand = grads["reference"], grads["candidate"]
      row = {
        "candidate": candidate,
        "trial": trial,
        "policy_grad_cosine": cosine(ref[1], cand[1]),
        "world_model_grad_cosine": cosine(ref[0], cand[0]),
        "imagined_reward_max_abs_diff": float((ref[2] - cand[2]).abs().max()),
      }
      policy_cosines.append(row["policy_grad_cosine"])
      world_model_cosines.append(row["world_model_grad_cosine"])
      results.append(row)
      log(json.dumps(row))
    trial_skips = (
      alg.skipped_world_model_steps - skipped[0],
      alg.skipped_policy_steps - skipped[1],
    )
    mean_policy = sum(policy_cosines) / len(policy_cosines)
    mean_world_model = sum(world_model_cosines) / len(world_model_cosines)
    summary = {
      "candidate": candidate,
      "mean_policy_grad_cosine": mean_policy,
      "mean_world_model_grad_cosine": mean_world_model,
      "skipped_steps_during_trials_world_model_policy": trial_skips,
    }
    results.append(summary)
    log(json.dumps(summary))
    if candidate == "compiled" and not mean_policy >= args.min_cosine:
      failures.append(f"compiled: mean policy cosine {mean_policy:.5f}")
    if candidate == "whole_objective" and mean_policy >= args.min_cosine:
      failures.append(
        f"whole_objective: mean policy cosine {mean_policy:.5f}; the check no longer "
        "catches the known-bad compiled objective"
      )
    del alg
    gc.collect()
    torch.cuda.empty_cache()

  if args.out is not None:
    args.out.write_text(json.dumps(results, indent=1))
  log(f"done in {time.time() - started:.0f} s")
  for failure in failures:
    print(f"FAIL {failure}", file=sys.stderr)
  if not failures:
    print("PASS")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
