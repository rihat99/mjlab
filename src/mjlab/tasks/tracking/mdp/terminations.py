from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from . import kernels
from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_name: str | None = None,
) -> torch.Tensor:
  """Height error of the anchor, or of ``body_name`` (a motion body), > threshold."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if body_name is None:
    robot_pos_w, ref_pos_w = command.robot_anchor_pos_w, command.anchor_pos_w
  else:
    body = command.cfg.body_names.index(body_name)
    robot_pos_w = command.robot_body_pos_w[:, body]
    ref_pos_w = command.body_pos_w[:, body]
  error = kernels.anchor_height_error(robot_pos_w, ref_pos_w)
  return error > threshold


def bad_anchor_ori(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str,
  threshold: float,
  body_name: str | None = None,
) -> torch.Tensor:
  """Tilt error of the anchor, or of ``body_name`` (a motion body), > threshold."""
  # Gravity is the fixed -z axis (EntityData.gravity_vec_w); asset_cfg stays in the
  # signature so existing cfgs keep working.
  del asset_cfg
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if body_name is None:
    robot_quat_w, ref_quat_w = command.robot_anchor_quat_w, command.anchor_quat_w
  else:
    body = command.cfg.body_names.index(body_name)
    robot_quat_w = command.robot_body_quat_w[:, body]
    ref_quat_w = command.body_quat_w[:, body]
  error = kernels.anchor_tilt_error(robot_quat_w, ref_quat_w)
  return error > threshold


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """True on the step scored against the last frame of the env's clip.

  Use with ``time_out=True``: the env then resets through the normal path instead of
  ``MotionCommand`` wrapping to a new frame inside the episode.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.time_steps >= command.clip_end_of(command.time_steps) - 1
