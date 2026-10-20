"""Evaluate a trained tracking policy and compute metrics."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointGainAction
from mjlab.envs.mdp.rewards import action_term_slice
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.metrics import (
  compute_ee_orientation_error,
  compute_ee_position_error,
  compute_joint_velocity_error,
  compute_mpkpe,
  compute_root_relative_mpkpe,
)
from mjlab.tasks.tracking.rl.runner import GAIN_ACTION_NAME, GAIN_BOUND_THRESHOLD
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvaluateConfig:
  """Configuration for policy evaluation."""

  checkpoint_file: str
  """Local checkpoint to evaluate (e.g. '<run_dir>/model_4000.pt')."""
  motion_file: str
  """Local motion npz the policy tracks."""
  num_envs: int = 1024
  """Number of parallel environments (= number of episodes to evaluate)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  output_file: str | None = None
  """Path of the metrics JSON. Defaults to 'eval_<checkpoint stem>.json' next to
  the checkpoint."""
  freeze_gains: bool = False
  """Gain-head tasks only: after the normal pass, re-run with every joint's gain
  pinned to its mean log multiplier over the first pass (metrics under
  'frozen_gains' in the JSON)."""


def _rollout(
  env: RslRlVecEnvWrapper,
  policy,
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
  gain_term: JointGainAction | None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
  """Run one episode in every env; return metrics and per-step arrays."""
  num_envs = env.num_envs
  device = env.device
  robot = env.unwrapped.scene["robot"]
  action_manager = env.unwrapped.action_manager
  joint_cols = action_term_slice(env.unwrapped, "joint_pos")
  gain_cols = (
    action_term_slice(env.unwrapped, GAIN_ACTION_NAME)
    if gain_term is not None
    else slice(0, 0)
  )

  # Metric accumulators.
  all_mpkpe: list[torch.Tensor] = []
  all_r_mpkpe: list[torch.Tensor] = []
  all_joint_vel_error: list[torch.Tensor] = []
  all_ee_pos_error: list[torch.Tensor] = []
  all_ee_ori_error: list[torch.Tensor] = []
  all_joint_acc_rms: list[torch.Tensor] = []
  all_torque_rate_rms: list[torch.Tensor] = []
  all_action_rate_joint: list[torch.Tensor] = []
  all_active: list[torch.Tensor] = []
  all_smooth: list[torch.Tensor] = []
  all_log_m: list[torch.Tensor] = []
  all_log_r: list[torch.Tensor] = []
  all_squash: list[torch.Tensor] = []

  done_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
  success = torch.zeros(num_envs, dtype=torch.bool, device=device)
  prev_force = robot.data.actuator_force.clone()

  obs = env.get_observations()

  print(f"[INFO] Running {num_envs} evaluation episodes...")

  step = 0
  while not done_envs.all():
    # Snapshot the reference frame the upcoming step will be scored against.
    # env.step computes the reward (against the current reference) and only
    # afterwards advances the command's motion frame, so reading the
    # reference after stepping would pair the robot with the *next* frame.
    # We snapshot here and pair it with the post-step robot state below,
    # matching how the reward is computed.
    ref = SimpleNamespace(
      num_envs=command.num_envs,
      device=command.device,
      cfg=command.cfg,
      body_pos_w=command.body_pos_w.clone(),
      body_pos_relative_w=command.body_pos_relative_w.clone(),
      body_quat_relative_w=command.body_quat_relative_w.clone(),
      joint_vel=command.joint_vel.clone(),
    )

    with torch.no_grad():
      actions = policy(obs)
    if env.clip_actions is not None:
      actions = torch.clamp(actions, -env.clip_actions, env.clip_actions)
    if gain_term is not None:
      # The gains this step is simulated with. After env.step the envs that just
      # ended are already reset to nominal, so reading the term then would log
      # the next episode's first state on every terminal row.
      log_m, log_r = gain_term.log_gains_for(actions[:, gain_cols])
      all_log_m.append(log_m.clone())
      all_log_r.append(log_r.clone())
      all_squash.append(gain_term.squash_of(log_m, log_r))
    obs, _, dones, _ = env.step(actions)

    # Pair the snapshotted reference with the post-step robot state.
    ref.robot_body_pos_w = command.robot_body_pos_w
    ref.robot_body_quat_w = command.robot_body_quat_w
    ref.robot_joint_vel = command.robot_joint_vel
    ref_command = cast(MotionCommand, ref)

    # Accumulate metrics for envs still running this step. active.any() is
    # always true here: the loop runs only while some env is not done, and
    # done_envs is updated below after this point.
    active = ~done_envs
    all_active.append(active.float())
    all_mpkpe.append(torch.where(active, compute_mpkpe(ref_command), 0.0))
    all_r_mpkpe.append(
      torch.where(active, compute_root_relative_mpkpe(ref_command), 0.0)
    )
    all_joint_vel_error.append(
      torch.where(active, compute_joint_velocity_error(ref_command), 0.0)
    )
    all_ee_pos_error.append(
      torch.where(active, compute_ee_position_error(ref_command, ee_body_names), 0.0)
    )
    all_ee_ori_error.append(
      torch.where(active, compute_ee_orientation_error(ref_command, ee_body_names), 0.0)
    )
    # Smoothness: RMS over joints / actuators of the post-step quantities. Envs
    # that just ended were reset inside env.step, so their row holds the reset
    # state (zero forces, large accelerations) and is left out.
    newly_done = dones.bool() & ~done_envs
    smooth = active & ~newly_done
    all_smooth.append(smooth.float())
    force = robot.data.actuator_force
    joint_acc_rms = torch.sqrt(torch.mean(torch.square(robot.data.joint_acc), dim=-1))
    torque_rate_rms = torch.sqrt(torch.mean(torch.square(force - prev_force), dim=-1))
    action_rate_joint = torch.sum(
      torch.square(
        action_manager.action[:, joint_cols] - action_manager.prev_action[:, joint_cols]
      ),
      dim=-1,
    )
    prev_force = force.clone()
    all_joint_acc_rms.append(torch.where(smooth, joint_acc_rms, 0.0))
    # The first step has no previous force; its row is excluded below.
    all_torque_rate_rms.append(torch.where(smooth & (step > 0), torque_rate_rms, 0.0))
    all_action_rate_joint.append(torch.where(smooth, action_rate_joint, 0.0))

    # Track completions.
    terminated = env.unwrapped.termination_manager.terminated
    truncated = env.unwrapped.termination_manager.time_outs

    if newly_done.any():
      success = success | (newly_done & truncated & ~terminated)
      done_envs = done_envs | newly_done
      print(
        f"[INFO] {done_envs.sum().item()}/{num_envs} episodes completed "
        f"(step {step}, truncated={(newly_done & truncated).sum().item()}, "
        f"terminated={(newly_done & terminated).sum().item()})"
      )
    step += 1

  # Compute mean metrics over the steps each env was active.
  stacks = [
    all_mpkpe,
    all_r_mpkpe,
    all_joint_vel_error,
    all_ee_pos_error,
    all_ee_ori_error,
    all_joint_acc_rms,
    all_torque_rate_rms,
    all_action_rate_joint,
  ]
  stacks = [torch.stack(s, dim=0) for s in stacks]
  active_mask = torch.stack(all_active, dim=0)  # (T, N)
  smooth_mask = torch.stack(all_smooth, dim=0)
  active_steps = active_mask.sum(dim=0).clamp(min=1)
  smooth_steps = smooth_mask.sum(dim=0).clamp(min=1)
  torque_steps = smooth_mask[1:].sum(dim=0).clamp(min=1)  # step 0 has no rate
  steps = [active_steps] * 5 + [smooth_steps, torque_steps, smooth_steps]
  means = [s.sum(dim=0) / n for s, n in zip(stacks, steps, strict=True)]

  metrics = {
    "success_rate": success.float().mean().item(),
    "mpkpe": means[0].mean().item(),
    "r_mpkpe": means[1].mean().item(),
    "joint_vel_error": means[2].mean().item(),
    "ee_pos_error": means[3].mean().item(),
    "ee_ori_error": means[4].mean().item(),
    "joint_acc_rms": means[5].mean().item(),
    "torque_rate_rms": means[6].mean().item(),
    "action_rate_joint": means[7].mean().item(),
  }
  arrays = {"active": active_mask.T.cpu().numpy().astype(bool)}  # (N, T)

  if gain_term is not None:
    log_m = torch.stack(all_log_m, dim=0)  # (T, N, J)
    log_r = torch.stack(all_log_r, dim=0)
    squash = torch.stack(all_squash, dim=0)  # (T, N, A)
    mask = active_mask.bool()

    def env_mean(x: torch.Tensor) -> torch.Tensor:
      """Mean over the active steps of each env: (T, N, K) -> (N, K)."""
      return (x * mask[..., None]).sum(0) / active_steps[:, None]

    # Per env over its active steps, then mean over envs (and joints).
    log_m_mean_env = env_mean(log_m)
    log_m_var_env = env_mean(torch.square(log_m - log_m_mean_env))
    metrics["log_m_mean"] = log_m_mean_env.mean().item()
    metrics["log_m_std_time"] = torch.sqrt(log_m_var_env).mean().item()
    metrics["gain_frac_bound"] = (
      env_mean((squash.abs() > GAIN_BOUND_THRESHOLD).float()).mean().item()
    )
    if gain_term.cfg.mode == "freq_damp":
      metrics["log_r_mean"] = env_mean(log_r).mean().item()
    arrays["log_m"] = log_m.transpose(0, 1).cpu().numpy()  # (N, T, J)
    arrays["log_r"] = log_r.transpose(0, 1).cpu().numpy()
    arrays["log_m_time_mean"] = log_m[mask].mean(dim=0).cpu().numpy()  # (J,)
    arrays["log_r_time_mean"] = log_r[mask].mean(dim=0).cpu().numpy()

  return metrics, arrays


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
  print("\n" + "=" * 50)
  print(title)
  print("=" * 50)
  for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
  print("=" * 50)


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> dict[str, float]:
  """Run policy evaluation and compute metrics."""
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load configs.
  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)

  motion_cmd = env_cfg.commands.get("motion")
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id} is not a tracking task.")

  resume_path = Path(cfg.checkpoint_file)
  for path in (resume_path, Path(cfg.motion_file)):
    if not path.is_file():
      raise FileNotFoundError(f"File not found: {path}")
  motion_cmd.motion_file = cfg.motion_file

  # Evaluation config.
  motion_cmd.sampling_mode = "start"
  env_cfg.observations["actor"].enable_corruption = True
  env_cfg.events.pop("push_robot", None)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  print(f"[INFO] Loading checkpoint: {resume_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  ee_body_names = env_cfg.terminations["ee_body_pos"].params["body_names"]
  print(f"[INFO] End effector bodies: {ee_body_names}")

  gain_term: JointGainAction | None = None
  if GAIN_ACTION_NAME in env.unwrapped.action_manager.active_terms:
    term = env.unwrapped.action_manager.get_term(GAIN_ACTION_NAME)
    assert isinstance(term, JointGainAction)
    gain_term = term
  elif cfg.freeze_gains:
    raise ValueError(f"--freeze-gains needs a '{GAIN_ACTION_NAME}' action term.")

  metrics, arrays = _rollout(env, policy, command, ee_body_names, gain_term)
  _print_metrics("Evaluation Results", metrics)

  output_path = Path(
    cfg.output_file or resume_path.with_name(f"eval_{resume_path.stem}.json")
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  result: dict[str, Any] = dict(metrics)

  if gain_term is not None:
    gains_path = output_path.with_name(output_path.stem + "_gains.npz")
    np.savez_compressed(
      gains_path,
      joint_names=np.array(gain_term.joint_names),
      **arrays,  # type: ignore[invalid-argument-type]
    )
    print(f"[INFO] Per-step gains saved to {gains_path}")
    result["gains_file"] = str(gains_path)
    result["log_m_time_mean"] = arrays["log_m_time_mean"].tolist()

  if cfg.freeze_gains:
    assert gain_term is not None
    gain_term.freeze(
      torch.as_tensor(arrays["log_m_time_mean"]),
      torch.as_tensor(arrays["log_r_time_mean"]),
    )
    env.unwrapped.reset()
    frozen_metrics, _ = _rollout(env, policy, command, ee_body_names, gain_term)
    gain_term.freeze(None)
    _print_metrics(
      "Evaluation Results (gains frozen at their time mean)", frozen_metrics
    )
    result["frozen_gains"] = frozen_metrics

  motion_path = Path(cfg.motion_file).resolve()
  result["provenance"] = {
    "task": task_id,
    "checkpoint_file": str(resume_path.resolve()),
    "motion_file": str(motion_path),
    "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
    "num_envs": cfg.num_envs,
    "episode_length_s": env_cfg.episode_length_s,
    "max_episode_length_steps": env.unwrapped.max_episode_length,
  }
  with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
  print(f"[INFO] Metrics saved to {output_path}")

  env.close()
  return metrics


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
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
