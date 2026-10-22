"""Differentiable forward kinematics of a floating-base robot, built from an MjModel."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np
import torch

from mjlab.utils.lab_api.math import normalize, quat_apply, quat_mul


class TorchKinematics:
  """Batched torch forward kinematics of a free-floating tree of hinge joints.

  Reproduces MuJoCo's ``mj_kinematics`` body poses (``xpos``, ``xquat``) and the
  body-origin velocities that mjlab reports as ``body_link_lin_vel_w`` /
  ``body_link_ang_vel_w`` (``cvel`` shifted from the subtree COM to ``xpos``), so
  the tracking rewards can be evaluated on (imagined) generalized states.

  Supported structure: one body with a free joint (the root) and, below it, bodies
  with exactly one hinge joint each, anchored at the body origin. Anything else
  raises ``NotImplementedError``. The hinge columns of ``q`` / ``qd`` follow the
  model's ``qposadr`` order (``joint_names``), i.e. mjlab's ``EntityData.joint_pos``.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    body_names: Sequence[str],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
  ) -> None:
    """Index the root subtree of ``mj_model``.

    Args:
      mj_model: Compiled model, e.g. ``env.sim.mj_model`` (names carry the entity
        prefix, ``"robot/pelvis"``) or ``Entity.compile()`` (no prefix).
      body_names: Bodies returned by ``forward``, in this order (T of them).
      device: Torch device of the constants and outputs.
      dtype: Torch dtype of the constants; inputs must match it.
    """
    body_ids = [mj_model.body(name).id for name in body_names]
    root = body_ids[0]
    while not _has_free_joint(mj_model, root):
      if root == 0:
        raise NotImplementedError(f"No free-joint ancestor for {body_names[0]!r}.")
      root = int(mj_model.body_parentid[root])

    depth = {root: 0}
    for body in range(root + 1, mj_model.nbody):
      parent = int(mj_model.body_parentid[body])
      if parent in depth:
        depth[body] = depth[parent] + 1
    missing = [n for n, b in zip(body_names, body_ids, strict=True) if b not in depth]
    if missing:
      raise ValueError(f"Bodies outside the subtree of the free joint: {missing}")

    hinge_ids = []
    for body in depth:
      if body == root:
        continue
      jnt = int(mj_model.body_jntadr[body])
      if (
        mj_model.body_jntnum[body] != 1
        or mj_model.jnt_type[jnt] != mujoco.mjtJoint.mjJNT_HINGE
        or np.any(mj_model.jnt_pos[jnt] != 0.0)
      ):
        name = mj_model.body(body).name
        raise NotImplementedError(f"Body {name!r} is not one hinge at its origin.")
      hinge_ids.append(jnt)
    hinge_ids.sort(key=lambda j: int(mj_model.jnt_qposadr[j]))
    column = {jnt: col for col, jnt in enumerate(hinge_ids)}
    self.joint_names: tuple[str, ...] = tuple(mj_model.joint(j).name for j in hinge_ids)

    def const(values, kind=dtype) -> torch.Tensor:
      return torch.tensor(np.asarray(values), dtype=kind, device=device)

    # Bodies grouped by depth; a body's parent is always in the previous level.
    levels: list[list[int]] = [[] for _ in range(max(depth.values()) + 1)]
    for body, d in depth.items():
      levels[d].append(body)
    self._levels: list[dict[str, torch.Tensor]] = []
    for d in range(1, len(levels)):
      bodies = levels[d]
      parents = [levels[d - 1].index(mj_model.body_parentid[b]) for b in bodies]
      joints = [int(mj_model.body_jntadr[b]) for b in bodies]
      self._levels.append(
        {
          "parent": const(parents, torch.long),
          "body_pos": const(mj_model.body_pos[bodies]),
          "body_quat": const(mj_model.body_quat[bodies]),
          "axis": const(mj_model.jnt_axis[joints]),
          "column": const([column[j] for j in joints], torch.long),
          "qpos0": const(mj_model.qpos0[mj_model.jnt_qposadr[joints]]),
        }
      )
    # Position of each requested body in the depth-ordered concatenation.
    order = [b for level in levels for b in level]
    self._output = const([order.index(b) for b in body_ids], torch.long)

  def forward(
    self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, q: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Body poses from the root pose and joint positions.

    Args:
      root_pos_w: (B, 3) free-joint position (qpos[0:3]).
      root_quat_w: (B, 4) free-joint quaternion wxyz (qpos[3:7]); normalized here.
      q: (B, J) hinge positions.

    Returns:
      body_pos_w: (B, T, 3). body_quat_w: (B, T, 4) unit wxyz.
    """
    all_pos, all_quat = self._level_poses(root_pos_w, root_quat_w, q)
    return self._select(all_pos), self._select(all_quat)

  def forward_with_velocity(
    self,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    q: torch.Tensor,
    qd: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Body poses and body-origin velocities.

    Args:
      root_pos_w: (B, 3) free-joint position.
      root_quat_w: (B, 4) free-joint quaternion wxyz.
      root_lin_vel_w: (B, 3) root origin velocity, world frame (qvel[0:3]).
      root_ang_vel_w: (B, 3) root angular velocity, world frame (R qvel[3:6]).
      q: (B, J) hinge positions.
      qd: (B, J) hinge velocities.

    Returns:
      body_pos_w (B, T, 3), body_quat_w (B, T, 4), body_lin_vel_w (B, T, 3) of the
      body origins, body_ang_vel_w (B, T, 3); all in the world frame.
    """
    all_pos, all_quat = self._level_poses(root_pos_w, root_quat_w, q)
    lin_vel, ang_vel = root_lin_vel_w.unsqueeze(1), root_ang_vel_w.unsqueeze(1)
    all_lin_vel, all_ang_vel = [lin_vel], [ang_vel]
    for i, level in enumerate(self._levels):
      parent = level["parent"]
      pos, quat = all_pos[i + 1], all_quat[i + 1]
      parent_ang_vel = ang_vel[:, parent]
      # The hinge sits at the body origin, so it adds no linear velocity there.
      offset = pos - all_pos[i][:, parent]
      lin_vel = lin_vel[:, parent] + torch.cross(parent_ang_vel, offset, dim=-1)
      axis_w = quat_apply(quat, level["axis"].expand_as(pos))
      ang_vel = parent_ang_vel + axis_w * qd[:, level["column"], None]
      all_lin_vel.append(lin_vel)
      all_ang_vel.append(ang_vel)
    return (
      self._select(all_pos),
      self._select(all_quat),
      self._select(all_lin_vel),
      self._select(all_ang_vel),
    )

  def _level_poses(
    self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, q: torch.Tensor
  ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-level body poses, (B, n_level, 3) and (B, n_level, 4), root first."""
    pos = root_pos_w.unsqueeze(1)
    quat = normalize(root_quat_w).unsqueeze(1)
    all_pos, all_quat = [pos], [quat]
    for level in self._levels:
      shape = (root_pos_w.shape[0], level["parent"].numel())
      parent_quat = quat[:, level["parent"]]
      pos = pos[:, level["parent"]] + quat_apply(
        parent_quat, level["body_pos"].expand(*shape, 3)
      )
      quat = quat_mul(parent_quat, level["body_quat"].expand(*shape, 4))
      half_angle = 0.5 * (q[:, level["column"]] - level["qpos0"])
      joint_quat = torch.cat(
        [
          torch.cos(half_angle)[..., None],
          torch.sin(half_angle)[..., None] * level["axis"],
        ],
        dim=-1,
      )
      quat = normalize(quat_mul(quat, joint_quat))
      all_pos.append(pos)
      all_quat.append(quat)
    return all_pos, all_quat

  def _select(self, per_level: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-level tensors and pick the requested bodies: (B, T, D)."""
    return torch.cat(per_level, dim=1)[:, self._output]


def _has_free_joint(mj_model: mujoco.MjModel, body: int) -> bool:
  jnt = int(mj_model.body_jntadr[body])
  return jnt >= 0 and mj_model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE
