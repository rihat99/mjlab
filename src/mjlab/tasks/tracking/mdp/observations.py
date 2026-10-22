from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from . import kernels
from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = kernels.pose_in_frame(
    command.anchor_pos_w,
    command.anchor_quat_w,
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = kernels.pose_in_frame(
    command.anchor_pos_w,
    command.anchor_quat_w,
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
  )
  return ori


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = kernels.pose_in_frame(
    command.robot_body_pos_w,
    command.robot_body_quat_w,
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = kernels.pose_in_frame(
    command.robot_body_pos_w,
    command.robot_body_quat_w,
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
  )
  return ori_b.reshape(env.num_envs, -1)
