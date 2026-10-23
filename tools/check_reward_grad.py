"""Compiled vs eager reward and termination of the SuperMimic learner, per function.

The policy objective runs eagerly but calls the reward (``iface.reward``) and the
termination (``SuperMimic.termination``: the analytic soft rule or the learned head)
compiled, each on its own (``PrecisionCfg.compile``). This check feeds identical
inputs to the eager and the compiled function and compares the forward values and the
gradients with respect to the inputs, on replay states of a checkpoint's policy and on
a perturbed copy (root drops, tilts, joint noise: the soft termination is active
there). Each function runs in fp32 (TF32 off, so that compiled and eager differ only
in fusion and summation order) and under fp16 autocast as in training (compiled and
eager both under autocast). The reward is also checked term by term.

Unlike tools/check_policy_grad.py no rollout amplifies the differences, so the bar is
tight: relative error |compiled - eager| / |eager| (vector norms) of the values and of
every input gradient below ``--tol``. The exception is the learned head under fp16
(``--tol-fp16-head``): it is an MLP, and eager autocast rounds every layer output to
fp16 while the compiled kernels keep fused intermediates in fp32 (~2e-3 apart; 1e-6
in fp32). Rows also give the norm of each eager input gradient (a zero gradient
compares as error 0). Exit status 1 when a row exceeds its bar. About 5 minutes on
one GPU:

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python tools/check_reward_grad.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import torch
from paths import LOG_ROOT, MOTIONS_DIR

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.supermimic.config import SuperMimicRunnerCfg
from mjlab.rl.supermimic.runner import SuperMimicRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg

TASK = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"
CHECKPOINT = LOG_ROOT / "2026-10-23_21-20-15_uni_env" / "model_14000.pt"
PARTS = {"actor": True, "critic": True, "world_model": True, "optimizer": True}
HORIZON = 32  # The termination sees (H, B) features; any split of the batch works.


def log(message: str) -> None:
  """Timestamped line on stdout."""
  print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
  """|value - reference| / |reference| over all elements (float64)."""
  diff = (value.double() - reference.double()).norm()
  return float(diff / reference.double().norm().clamp_min(1e-30))


def evaluate(
  fn: Callable[..., torch.Tensor],
  inputs: list[torch.Tensor],
  grad_inputs: tuple[int, ...],
  upstream: torch.Tensor,
  fp16: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
  """Value of ``fn(*inputs)`` and the gradients of ``sum(value * upstream)`` with
  respect to the inputs ``grad_inputs``."""
  leaves = [x.clone().requires_grad_(i in grad_inputs) for i, x in enumerate(inputs)]
  with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
    value = fn(*leaves)
  grads = torch.autograd.grad(
    (value.float() * upstream).sum(),
    [leaves[i] for i in grad_inputs],
    allow_unused=True,
  )
  grads = [
    torch.zeros_like(leaves[i]) if g is None else g
    for g, i in zip(grads, grad_inputs, strict=True)
  ]
  return value.detach().float(), grads


def main() -> int:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
  parser.add_argument(
    "--anchor-body",
    default="pelvis",
    help="motion anchor body the checkpoint was trained with",
  )
  parser.add_argument("--motions-dir", type=Path, default=MOTIONS_DIR)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--fill-steps", type=int, default=150)
  parser.add_argument("--batch", type=int, default=32768, help="states per check")
  parser.add_argument("--tol", type=float, default=1e-5)
  parser.add_argument("--tol-fp16-head", type=float, default=1e-2)
  parser.add_argument("--out", type=Path, default=None, help="JSON of all rows")
  args = parser.parse_args()
  device = "cuda:0"
  started = time.time()

  env_cfg = load_env_cfg(TASK)
  motion_cfg = env_cfg.commands["motion"]
  assert isinstance(motion_cfg, MotionCommandCfg)
  motion_cfg.motion_file = tuple(str(p) for p in sorted(args.motions_dir.glob("*.npz")))
  motion_cfg.anchor_body_name = args.anchor_body
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = 0
  agent_cfg = load_rl_cfg(TASK)
  assert isinstance(agent_cfg, SuperMimicRunnerCfg)
  agent_cfg.precision.compile = False
  agent_cfg.precision.autocast = "none"
  agent_cfg.precision.tf32 = False
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions
  )
  runner = SuperMimicRunner(env, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(str(args.checkpoint), load_cfg={**PARTS, "sampler": True}, strict=True)
  alg, iface = runner.alg, runner.interface
  for module in (alg.world_model, alg.critic, alg.policy):
    module.requires_grad_(False)
  env.reset()
  runner._init_collection()
  runner._collect(args.fill_steps, 10**9)
  assert runner.buffer is not None
  torch.manual_seed(0)
  batch = next(iter(runner.buffer.iter_transitions(args.batch)))
  states = batch["next_state"].clone()
  actions = batch["action"].clone()
  prev_actions = actions.roll(1, 0)
  frames = batch["frame"].clone()
  # Root drops and tilts, joint noise: the soft termination's active range.
  perturbed = states.clone()
  perturbed[:, 2] -= 0.45 * torch.rand(len(perturbed), device=device)
  quat = perturbed[:, 3:7] + 0.25 * torch.randn_like(perturbed[:, 3:7])
  perturbed[:, 3:7] = quat / quat.norm(dim=-1, keepdim=True)
  perturbed[:, 13:] += 0.2 * torch.randn_like(perturbed[:, 13:])
  log(
    f"{len(states)} replay states of {args.checkpoint} in {time.time() - started:.0f} s"
  )

  def reward_terms(s, a, a_prev, frame) -> torch.Tensor:
    return torch.stack(list(iface.reward_terms(s, a, a_prev, frame).values()))

  term_names = list(
    iface.reward_terms(states[:2], actions[:2], prev_actions[:2], frames[:2])
  )

  def termination(s, frame, features) -> torch.Tensor:
    return alg.termination(s, frame, features)

  rows: list[dict] = []
  failures: list[str] = []

  def check(name, fn, inputs, grad_inputs, upstreams) -> None:
    compiled = torch.compile(fn)
    for fp16 in (False, True):
      for label, upstream in upstreams.items():
        value, grads = evaluate(fn, inputs, grad_inputs, upstream, fp16)
        value_c, grads_c = evaluate(compiled, inputs, grad_inputs, upstream, fp16)
        row = {
          "function": f"{name}{label}",
          "precision": "fp16" if fp16 else "fp32",
          "value": relative_error(value_c, value),
        }
        for i, grad, grad_c in zip(grad_inputs, grads, grads_c, strict=True):
          row[f"grad_input{i}"] = relative_error(grad_c, grad)
        worst = max(v for k, v in row.items() if k not in ("function", "precision"))
        for i, grad in zip(grad_inputs, grads, strict=True):
          row[f"grad_input{i}_norm"] = float(grad.double().norm())
        rows.append(row)
        log(json.dumps(row))
        tol = args.tol_fp16_head if fp16 and "head" in name else args.tol
        if not worst <= tol:
          failures.append(
            f"{row['function']} {row['precision']}: {worst:.2e} > {tol:g}"
          )
    torch._dynamo.reset()

  for data_name, data in (("real", states), ("perturbed", perturbed)):
    upstream = 0.5 + torch.rand(len(data), device=device)
    reward_inputs = [data, actions, prev_actions, frames]
    check(f"reward/{data_name}", iface.reward, reward_inputs, (0, 1, 2), {"": upstream})
    one_hot = {}
    for j, term in enumerate(term_names):
      weights = torch.zeros(len(term_names), len(data), device=device)
      weights[j] = upstream
      one_hot[f"/{term}"] = weights
    check(f"reward_term/{data_name}", reward_terms, reward_inputs, (0, 1, 2), one_hot)
    features = iface.invariant_features(data).view(HORIZON, len(data) // HORIZON, -1)
    for analytic in (True, False):
      alg.policy_cfg.analytic_termination = analytic
      check(
        f"termination_{'analytic' if analytic else 'head'}/{data_name}",
        termination,
        [data, frames, features],
        (0, 2),
        {"": upstream.view(HORIZON, -1)},
      )

  if args.out is not None:
    args.out.write_text(json.dumps(rows, indent=1))
  log(f"done in {time.time() - started:.0f} s")
  for failure in failures:
    print(f"FAIL {failure}", file=sys.stderr)
  if not failures:
    print("PASS")
  return 1 if failures else 0


if __name__ == "__main__":
  sys.exit(main())
