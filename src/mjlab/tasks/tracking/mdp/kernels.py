"""Pure tensor kernels behind the motion tracking rewards, terminations and observations.

The env terms in ``rewards.py``, ``terminations.py``, ``observations.py`` and
``MotionCommand.update_relative_body_poses`` gather tensors from the env and call these
functions, so the same math can run on imagined states (e.g. world-model rollouts) and
stay differentiable. Every kernel is batched over leading dims and has no env access.

Shapes: ``B`` = batch, ``T`` = tracked bodies, ``J`` = joints. Quaternions are wxyz.
Positions and velocities are in the world frame unless the name says otherwise.

This module is not star-exported from ``mdp``: ``action_rate_l2`` would shadow the
generic env term of the same name. Import it as ``mdp.kernels``.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_inv,
  quat_mul,
  subtract_frame_transforms,
  yaw_quat,
)


def quat_error_magnitude(
  q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-14
) -> torch.Tensor:
  """Rotation angle between two quaternions, with a finite gradient at zero error.

  Same value as ``mjlab.utils.lab_api.math.quat_error_magnitude`` (within 1e-6 in
  float32) and the same division order, but ``eps`` sits inside the square root of
  the axis norm, so the norm and ``sin(half) / angle`` never see an exact zero (the
  original gives NaN gradients when ``q1 == q2``).

  Args:
    q1: (..., 4) wxyz.
    q2: (..., 4) wxyz.
    eps: Added to the squared axis norm; the angle at zero error is 2 sqrt(eps).

  Returns:
    (...,) angle in radians, in [0, pi].
  """
  quat = quat_mul(q1, quat_conjugate(q2))
  axis_norm = torch.sqrt(torch.sum(torch.square(quat[..., 1:]), dim=-1) + eps)
  half_angle = torch.atan2(axis_norm, torch.abs(quat[..., 0]))
  angle = 2.0 * half_angle
  return axis_norm / (torch.sin(half_angle) / angle)


##
# Anchor frames.
##


def anchor_frame(
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  ref_anchor_pos_w: torch.Tensor,
  ref_anchor_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """BeyondMimic re-anchoring of the reference onto the robot anchor.

  The reference is moved so that its anchor sits at the robot anchor's xy (keeping
  the reference height) and rotated by the yaw difference robot vs reference.

  Args:
    anchor_pos_w: (B, 3) robot anchor position.
    anchor_quat_w: (B, 4) robot anchor orientation.
    ref_anchor_pos_w: (B, 3) reference anchor position.
    ref_anchor_quat_w: (B, 4) reference anchor orientation.

  Returns:
    pos_delta: (B, 3) where the reference anchor is placed, robot xy + reference z.
    quat_delta: (B, 4) yaw-only rotation applied to the reference.
  """
  pos_delta = torch.cat([anchor_pos_w[..., :2], ref_anchor_pos_w[..., 2:]], dim=-1)
  quat_delta = yaw_quat(quat_mul(anchor_quat_w, quat_inv(ref_anchor_quat_w)))
  return pos_delta, quat_delta


def relative_body_targets(
  ref_body_pos_w: torch.Tensor,
  ref_body_quat_w: torch.Tensor,
  ref_anchor_pos_w: torch.Tensor,
  pos_delta: torch.Tensor,
  quat_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference body poses re-expressed in the frame given by ``anchor_frame``.

  Args:
    ref_body_pos_w: (B, T, 3) reference body positions.
    ref_body_quat_w: (B, T, 4) reference body orientations.
    ref_anchor_pos_w: (B, 3) reference anchor position.
    pos_delta: (B, 3) from ``anchor_frame``.
    quat_delta: (B, 4) from ``anchor_frame``.

  Returns:
    body_pos_relative_w: (B, T, 3) target body positions.
    body_quat_relative_w: (B, T, 4) target body orientations.
  """
  quat_delta = quat_delta.unsqueeze(-2).expand_as(ref_body_quat_w)
  body_quat = quat_mul(quat_delta, ref_body_quat_w)
  body_pos = pos_delta.unsqueeze(-2) + quat_apply(
    quat_delta, ref_body_pos_w - ref_anchor_pos_w.unsqueeze(-2)
  )
  return body_pos, body_quat


def pose_in_frame(
  pos_w: torch.Tensor,
  quat_w: torch.Tensor,
  frame_pos_w: torch.Tensor,
  frame_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Pose expressed in another frame, orientation as two rotation-matrix columns.

  Args:
    pos_w: (..., 3) position.
    quat_w: (..., 4) orientation.
    frame_pos_w: (..., 3) frame origin, same leading shape as ``pos_w``.
    frame_quat_w: (..., 4) frame orientation, same leading shape as ``quat_w``.

  Returns:
    pos_b: (..., 3) position in the frame.
    ori_6d: (..., 6) first two columns of the relative rotation matrix, flattened
      row by row ``[r00, r01, r10, r11, r20, r21]`` (mjlab's layout).
  """
  pos_b, quat_b = subtract_frame_transforms(frame_pos_w, frame_quat_w, pos_w, quat_w)
  mat = matrix_from_quat(quat_b)
  return pos_b, mat[..., :2].reshape(*mat.shape[:-2], 6)


##
# Rewards (unweighted, not scaled by dt).
##


def exp_tracking(error: torch.Tensor, std: float) -> torch.Tensor:
  """``exp(-error / std^2)`` for a squared error ``error`` of shape (B,)."""
  return torch.exp(-error / std**2)


def root_pos_reward(
  anchor_pos_w: torch.Tensor, ref_anchor_pos_w: torch.Tensor, std: float
) -> torch.Tensor:
  """Global anchor position tracking: (B, 3), (B, 3) -> (B,)."""
  error = torch.sum(torch.square(ref_anchor_pos_w - anchor_pos_w), dim=-1)
  return exp_tracking(error, std)


def root_ori_reward(
  anchor_quat_w: torch.Tensor, ref_anchor_quat_w: torch.Tensor, std: float
) -> torch.Tensor:
  """Global anchor orientation tracking: (B, 4), (B, 4) -> (B,)."""
  error = quat_error_magnitude(ref_anchor_quat_w, anchor_quat_w) ** 2
  return exp_tracking(error, std)


def body_pos_reward(
  body_pos_w: torch.Tensor, target_pos_w: torch.Tensor, std: float
) -> torch.Tensor:
  """Body position tracking, mean over bodies: (B, T, 3), (B, T, 3) -> (B,)."""
  error = torch.sum(torch.square(target_pos_w - body_pos_w), dim=-1)
  return exp_tracking(error.mean(-1), std)


def body_ori_reward(
  body_quat_w: torch.Tensor, target_quat_w: torch.Tensor, std: float
) -> torch.Tensor:
  """Body orientation tracking, mean over bodies: (B, T, 4), (B, T, 4) -> (B,)."""
  error = quat_error_magnitude(target_quat_w, body_quat_w) ** 2
  return exp_tracking(error.mean(-1), std)


def body_lin_vel_reward(
  body_lin_vel_w: torch.Tensor, ref_body_lin_vel_w: torch.Tensor, std: float
) -> torch.Tensor:
  """Body linear velocity tracking, mean over bodies: (B, T, 3) x2 -> (B,)."""
  error = torch.sum(torch.square(ref_body_lin_vel_w - body_lin_vel_w), dim=-1)
  return exp_tracking(error.mean(-1), std)


def body_ang_vel_reward(
  body_ang_vel_w: torch.Tensor, ref_body_ang_vel_w: torch.Tensor, std: float
) -> torch.Tensor:
  """Body angular velocity tracking, mean over bodies: (B, T, 3) x2 -> (B,)."""
  error = torch.sum(torch.square(ref_body_ang_vel_w - body_ang_vel_w), dim=-1)
  return exp_tracking(error.mean(-1), std)


def action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
  """Squared change of the raw actions: (B, A), (B, A) -> (B,)."""
  return torch.sum(torch.square(action - prev_action), dim=-1)


def joint_limit_penalty(q: torch.Tensor, soft_limits: torch.Tensor) -> torch.Tensor:
  """Summed distance outside the soft joint limits: (B, J), (B, J, 2) -> (B,)."""
  below = -(q - soft_limits[..., 0]).clip(max=0.0)
  above = (q - soft_limits[..., 1]).clip(min=0.0)
  return torch.sum(below + above, dim=-1)


##
# Terminations.
##


def anchor_height_error(
  anchor_pos_w: torch.Tensor, ref_anchor_pos_w: torch.Tensor
) -> torch.Tensor:
  """Absolute anchor height error: (B, 3), (B, 3) -> (B,)."""
  return torch.abs(ref_anchor_pos_w[..., -1] - anchor_pos_w[..., -1])


def anchor_tilt_error(
  anchor_quat_w: torch.Tensor, ref_anchor_quat_w: torch.Tensor
) -> torch.Tensor:
  """Difference of the gravity z component in the two anchor frames: (B, 4) x2 -> (B,).

  Gravity is the unit vector -z, as ``EntityData.gravity_vec_w``.
  """
  gravity_w = anchor_quat_w.new_tensor([0.0, 0.0, -1.0]).expand_as(
    anchor_quat_w[..., :3]
  )
  ref_gravity_b = quat_apply_inverse(ref_anchor_quat_w, gravity_w)
  gravity_b = quat_apply_inverse(anchor_quat_w, gravity_w)
  return (ref_gravity_b[..., 2] - gravity_b[..., 2]).abs()


def anchor_termination(
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  ref_anchor_pos_w: torch.Tensor,
  ref_anchor_quat_w: torch.Tensor,
  z_thresh: float = 0.25,
  tilt_thresh: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """The rule of ``bad_anchor_pos_z_only`` and ``bad_anchor_ori``.

  Args:
    anchor_pos_w: (B, 3) robot anchor position.
    anchor_quat_w: (B, 4) robot anchor orientation.
    ref_anchor_pos_w: (B, 3) reference anchor position.
    ref_anchor_quat_w: (B, 4) reference anchor orientation.
    z_thresh: Height error above which the episode fails.
    tilt_thresh: Tilt error above which the episode fails.

  Returns:
    bad_z: (B,) bool. bad_tilt: (B,) bool. e_z: (B,) height error. e_tilt: (B,)
    tilt error.
  """
  e_z = anchor_height_error(anchor_pos_w, ref_anchor_pos_w)
  e_tilt = anchor_tilt_error(anchor_quat_w, ref_anchor_quat_w)
  return e_z > z_thresh, e_tilt > tilt_thresh, e_z, e_tilt


##
# Observations.
##


def proprio_observation(
  q: torch.Tensor,
  qd: torch.Tensor,
  root_quat_w: torch.Tensor,
  root_ang_vel_b: torch.Tensor,
  encoder_bias: torch.Tensor,
  default_q: torch.Tensor,
  a_prev: torch.Tensor,
) -> torch.Tensor:
  """Noise-free proprioceptive observation, one frame.

  Same values as the env terms ``joint_pos_rel(biased=True)``, ``joint_vel_rel``,
  the ``imu_ang_vel`` gyro (its site has the pelvis orientation), ``projected_gravity``
  and ``last_action``, concatenated in that order.

  Args:
    q: (B, J) joint positions.
    qd: (B, J) joint velocities.
    root_quat_w: (B, 4) root orientation.
    root_ang_vel_b: (B, 3) root angular velocity in the root frame.
    encoder_bias: (B, J) per-env encoder offsets.
    default_q: (B, J) or (J,) default joint positions.
    a_prev: (B, A) last raw action.

  Returns:
    (B, 2 J + 6 + A); 93 for the G1 (J = A = 29).
  """
  gravity_w = root_quat_w.new_tensor([0.0, 0.0, -1.0]).expand_as(root_quat_w[..., :3])
  projected_gravity = quat_apply_inverse(root_quat_w, gravity_w)
  joint_pos = q + encoder_bias - default_q
  return torch.cat([joint_pos, qd, root_ang_vel_b, projected_gravity, a_prev], dim=-1)
