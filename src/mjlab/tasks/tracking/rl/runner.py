import os
from dataclasses import dataclass
from typing import cast

import torch
from rsl_rl.env import VecEnv
from rsl_rl.modules import GaussianDistribution
from torch import nn

from mjlab.envs.mdp.actions import JointGainAction
from mjlab.envs.mdp.rewards import action_term_slice
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommand

GAIN_ACTION_NAME = "joint_gain"
GAIN_BOUND_THRESHOLD = 0.95


@dataclass
class TrackingRunnerCfg(RslRlOnPolicyRunnerCfg):
  gain_head_init_std: float | None = None
  """Initial policy std of the gain-head dims (``joint_gain`` action term). None
  keeps the actor's ``init_std`` for every dim."""


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    gain_head_init_std = train_cfg.pop("gain_head_init_std", None)
    super().__init__(env, train_cfg, log_dir, device)
    action_manager = self.env.unwrapped.action_manager
    self._gain_term: JointGainAction | None = None
    if GAIN_ACTION_NAME in action_manager.active_terms:
      term = action_manager.get_term(GAIN_ACTION_NAME)
      assert isinstance(term, JointGainAction)
      self._gain_term = term
      self._gain_slice = action_term_slice(self.env.unwrapped, GAIN_ACTION_NAME)
      if gain_head_init_std is not None:
        distribution = self.alg.get_policy().distribution
        assert isinstance(distribution, GaussianDistribution)
        with torch.no_grad():
          distribution.std_param[self._gain_slice] = gain_head_init_std
      # rsl_rl's Logger is built inside OnPolicyRunner and only knows the mean
      # std; add the gain statistics of the rollout just collected.
      base_log = self.logger.log

      def log_with_gains(*args, **kwargs):
        self._log_gain_stats(kwargs["it"])
        base_log(*args, **kwargs)

      self.logger.log = log_with_gains  # type: ignore

  def _log_gain_stats(self, it: int) -> None:
    writer = self.logger.writer
    if writer is None or self._gain_term is None:
      return
    joint_slice = slice(0, self._gain_slice.start)
    std = self.alg.get_policy().output_std
    writer.add_scalar("Gains/std_joint_head", std[..., joint_slice].mean().item(), it)
    writer.add_scalar(
      "Gains/std_gain_head", std[..., self._gain_slice].mean().item(), it
    )
    # Rollout means (T, N, A): the deterministic gains the policy would apply.
    params = self.alg.storage.distribution_params
    assert params is not None
    mean = params[0][..., self._gain_slice]
    squash = torch.tanh(mean / self._gain_term.cfg.s0)
    log_m, log_r = self._gain_term.squash_actions(mean)
    stats = {"log_m": log_m}
    if self._gain_term.cfg.mode == "freq_damp":
      stats["log_r"] = log_r
    for name, value in stats.items():
      writer.add_scalar(f"Gains/{name}_mean", value.mean().item(), it)
      # Std over time and envs, per joint, then averaged over joints.
      writer.add_scalar(
        f"Gains/{name}_std", value.flatten(0, 1).std(0).mean().item(), it
      )
    writer.add_scalar(
      "Gains/frac_bound",
      (squash.abs() > GAIN_BOUND_THRESHOLD).float().mean().item(),
      it,
    )

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=["obs", "time_step"],
      output_names=[
        "actions",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ],
      dynamic_axes={},
      dynamo=False,
    )

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      metadata = get_base_metadata(self.env.unwrapped, "local")
      motion_term = cast(
        MotionCommand, self.env.unwrapped.command_manager.get_term("motion")
      )
      metadata.update(
        {
          "anchor_body_name": motion_term.cfg.anchor_body_name,
          "body_names": list(motion_term.cfg.body_names),
        }
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
