"""Terminal-state capture for SuperMimic: the sim state of envs right before auto-reset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def floating_base_state(env: ManagerBasedRlEnv, entity_name: str) -> torch.Tensor:
  """State (N, 13 + 2 nq) of a floating-base entity, read from qpos / qvel only.

  Layout of :class:`mjlab.rl.supermimic.FloatingBaseState`: root position minus the
  env origin (3), root quaternion wxyz (4), root linear velocity in world (3), root
  angular velocity in world (3), joint positions (nq), joint velocities (nq). Derived
  quantities (xpos, cvel) are not used: they lag qpos by one substep inside
  ``env.step()``, where this also runs (pre-reset).
  """
  robot: Entity = env.scene[entity_name]
  root_pose = env.sim.data.qpos[:, robot.indexing.free_joint_q_adr]
  root_vel = env.sim.data.qvel[:, robot.indexing.free_joint_v_adr]
  quat = root_pose[:, 3:7]
  return torch.cat(
    [
      root_pose[:, :3] - env.scene.env_origins,
      quat,
      root_vel[:, :3],
      # The free joint's angular velocity is in the root frame.
      quat_apply(quat, root_vel[:, 3:6]),
      robot.data.joint_pos,
      robot.data.joint_vel,
    ],
    dim=-1,
  )


class TerminalStateRecorder(RecorderTerm):
  """Keeps the state of every env at its last auto-reset, before it is overwritten.

  ``record_pre_reset`` runs inside ``env.step()`` after terminations and rewards and
  before ``_reset_idx``, so qpos / qvel still hold the terminal state s_T. Params:
  ``entity_name`` of the floating-base robot.
  """

  def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._entity_name: str = cfg.params["entity_name"]
    self._snapshot: torch.Tensor | None = None

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    state = floating_base_state(self._env, self._entity_name)
    if self._snapshot is None:
      self._snapshot = torch.zeros_like(state)
    self._snapshot[env_ids] = state[env_ids]

  def terminal_state_snapshot(self) -> torch.Tensor | None:
    """(N, 13 + 2 nq) state at each env's last auto-reset; None before the first.

    Rows are current for the envs reset in the last ``env.step()``; other rows hold
    older terminal states. The tensor is updated in place.
    """
    return self._snapshot
