from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor

from . import kernels
from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int] | slice:
  # All bodies: a slice, since indexing with a list copies it to the device and syncs.
  if body_names is None:
    return slice(None)
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return kernels.root_pos_reward(command.robot_anchor_pos_w, command.anchor_pos_w, std)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return kernels.root_ori_reward(
    command.robot_anchor_quat_w, command.anchor_quat_w, std
  )


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  return kernels.body_pos_reward(
    command.robot_body_pos_w[:, body_indexes],
    command.body_pos_relative_w[:, body_indexes],
    std,
  )


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  return kernels.body_ori_reward(
    command.robot_body_quat_w[:, body_indexes],
    command.body_quat_relative_w[:, body_indexes],
    std,
  )


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  return kernels.body_lin_vel_reward(
    command.robot_body_lin_vel_w[:, body_indexes],
    command.body_lin_vel_w[:, body_indexes],
    std,
  )


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  return kernels.body_ang_vel_reward(
    command.robot_body_ang_vel_w[:, body_indexes],
    command.body_ang_vel_w[:, body_indexes],
    std,
  )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def max_step_reward(env: ManagerBasedRlEnv) -> float:
  """Largest reward of one env step, every positive term at 1 (the maximum of the
  tracking terms) and the penalties at 0, scaled by dt as the env reward is; the
  normaliser of the normalised return."""
  manager = env.reward_manager
  weights = [manager.get_term_cfg(name).weight for name in manager.active_terms]
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  return scale * sum(w for w in weights if w > 0)
