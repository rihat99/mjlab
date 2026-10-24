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


def motion_current_anchor_ori_b(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Reference anchor orientation of the current frame k_t - 1 seen from the robot
  anchor, 6D (N, 6): Uni-SuperMimic's orientation-error observation. k_t - 1 is the
  frame the state is aligned with (``MotionCommand.current_frames``); the next
  reward is scored against k_t."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  frames = command.current_frames(command.time_steps)
  ref_quat = command.motion.body_quat_w[frames, command.motion_anchor_body_index]
  return kernels.relative_rotation_6d(command.robot_anchor_quat_w, ref_quat.float())


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


def motion_future_frames_b(
  env: ManagerBasedRlEnv, command_name: str, future_frames: int = 10
) -> torch.Tensor:
  """Reference frames k_t .. k_t + K - 1 (clamped to the clip end) seen from the robot.

  See ``kernels.future_frames_goal``. Positions are taken env-local (reference as
  stored, robot minus its env origin), so the goal does not lose float32 precision
  to large env origins. Returns (N, K * (9 + 2 J)).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  frames = command.future_frames(command.time_steps, future_frames)
  anchor = command.motion_anchor_body_index
  return kernels.future_frames_goal(
    command.motion.body_pos_w[frames, anchor],
    command.motion.body_quat_w[frames, anchor].float(),
    command.motion.joint_pos[frames].float(),
    command.motion.joint_vel[frames].float(),
    command.robot_anchor_pos_w - env.scene.env_origins,
    command.robot_anchor_quat_w,
  )
