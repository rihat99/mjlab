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


def motion_future_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str, future_frames: int = 10
) -> torch.Tensor:
  """Reference anchor orientation of frames k_t .. k_t + K - 1 (clamped to the clip
  end) seen from the robot anchor, 6D per frame: SONIC's ``motion_anchor_ori_b_mf``.

  See ``MotionCommand.future_anchor_ori``. Returns (N, K * 6).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.future_anchor_ori(
    command.robot_anchor_quat_w, command.time_steps, future_frames
  )


def motion_future_frames_reference(
  env: ManagerBasedRlEnv, command_name: str, future_frames: int = 10
) -> torch.Tensor:
  """Reference frames k_t .. k_t + K - 1 (clamped to the clip end) in the heading
  frame of the current reference frame k_t - 1; independent of the robot state.

  See ``MotionCommand.reference_goal`` and ``kernels.reference_heading_goal``.
  Returns (N, K * (15 + 2 J)).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.reference_goal(command.time_steps, future_frames)
