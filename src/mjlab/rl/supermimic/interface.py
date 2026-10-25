"""Contract between a task and the SuperMimic learner, plus floating-base state helpers.

The learner never touches the simulator state directly: everything it needs from the
task (state read-out, analytic observation / goal / reward heads on imagined states,
reset-sampler bookkeeping) goes through a :class:`ModelInterface`. The runner builds the
interface as ``cls(env.unwrapped)`` from the class named in
``SuperMimicRunnerCfg.model_interface``.
"""

from __future__ import annotations

from typing import Protocol

import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)

# Below this squared rotation angle the log/exp maps switch to their Taylor forms, which
# keeps gradients finite at exactly zero rotation (the dynamics output starts at zero).
_SMALL_ANGLE_SQ = 1e-12


class ModelInterface(Protocol):
  """Task side of SuperMimic: state access and differentiable heads on imagined states.

  Shapes: N = number of envs, B = any batch. ``s`` is a state laid out as described by
  :class:`FloatingBaseState`. Frames are global reference-frame indices (int64). Methods
  marked (grad) must be differentiable with respect to ``s`` (and ``a`` where present).

  Frame convention: at observation time the env shows frame k_t; the transition
  (s_t, a_t -> s_{t+1}) is rewarded and terminated against frame k_t; then
  k_{t+1} = k_t + 1.
  """

  state_dim: int
  """Size of ``s`` (13 + 2 nq)."""
  context_dim: int
  """Size of the normalised sim context xi."""
  obs_dim: int
  """Size of one proprioceptive observation frame."""
  obs_term_dims: tuple[int, ...]
  """Sizes of the observation terms in the order the env concatenates them; they sum to
  ``obs_dim``. mjlab flattens a history group term-major (each term's frames oldest to
  newest, then the next term), so the policy needs them to recover frames
  (history, obs_dim)."""
  goal_dim: int
  """Size of the goal vector."""
  action_dim: int
  """Size of the action."""
  feature_dim: int
  """Size of :meth:`invariant_features`."""
  nq: int
  """Number of actuated joints."""
  max_step_reward: float
  """Largest reward one step can earn (already multiplied by dt); normalises returns."""

  def state_from_sim(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Current sim state of the envs, (N, state_dim) (or (len(env_ids), state_dim))."""
    ...

  def terminal_state_snapshot(self) -> torch.Tensor | None:
    """Pre-reset state of the last step, (N, state_dim).

    Rows are valid for envs that were reset (done) in the last ``env.step``; other rows
    are arbitrary. None only if no env has been reset yet.
    """
    ...

  def frame_from_sim(self) -> torch.Tensor:
    """Reference frame k_t the current observation targets, (N,) int64."""
    ...

  def context_normalized(self) -> torch.Tensor:
    """Normalised per-env sim context xi, (N, context_dim); fixed per env."""
    ...

  def invariant_features(self, s: torch.Tensor) -> torch.Tensor:
    """(grad) Translation/yaw-invariant features (B, feature_dim) of ``s`` (B, S)."""
    ...

  def observation(
    self,
    s: torch.Tensor,
    a_prev: torch.Tensor,
    env_idx: torch.Tensor,
    frame: torch.Tensor,
  ) -> torch.Tensor:
    """(grad) Noise-free observation frame.

    Args:
      s: (B, state_dim) state.
      a_prev: (B, action_dim) action executed before reaching ``s`` (zeros at episode
        start).
      env_idx: (B,) int64 env of each row (selects per-env constants such as encoder
        bias).
      frame: (B,) int64 frame k_t the observation targets (terms that compare ``s``
        with the reference use it; the state is aligned with k_t - 1).

    Returns:
      (B, obs_dim) observation, terms in ``obs_term_dims`` order.
    """
    ...

  def add_observation_noise(self, o: torch.Tensor) -> torch.Tensor:
    """Observation (B, obs_dim) plus fresh sensor noise as the env applies it."""
    ...

  def goal(self, s: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """(grad) Goal (B, goal_dim) of state ``s`` (B, S) at frame ``frame`` (B,)."""
    ...

  def reward(
    self,
    s_next: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    frame: torch.Tensor,
  ) -> torch.Tensor:
    """(grad) Reward (B,) of reaching ``s_next`` with action ``a`` against ``frame``.

    Already multiplied by dt, as the env reward is.
    """
    ...

  def reward_terms(
    self,
    s_next: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    frame: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Per-term rewards, each (B,), for logging and parity checks."""
    ...

  def termination_soft(self, s_next: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """(grad) Termination probability (B,) of s_next against frame: the env's
    termination errors through sigmoids. Imagination uses it as d_{t+1}."""
    ...

  def clip_end(self, frame: torch.Tensor) -> torch.Tensor:
    """Exclusive end frame (B,) int64 of the clip each frame (B,) belongs to."""
    ...

  def bin_of(self, frame: torch.Tensor) -> torch.Tensor:
    """Reset-sampler bin (B,) int64 of each frame (B,)."""
    ...

  def update_bin_returns(self, bins: torch.Tensor, returns: torch.Tensor) -> None:
    """Feed observed discounted returns (M,) from bins (M,) to the reset sampler."""
    ...

  def update_past_weights(self) -> None:
    """Advance the sampler's memory of past sampling weights; once per iteration."""
    ...

  def sampler_state_dict(self) -> dict[str, torch.Tensor]:
    """Reset-sampler state to checkpoint (bin returns, past weights)."""
    ...

  def load_sampler_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Restore the reset-sampler state saved by :meth:`sampler_state_dict`."""
    ...

  def sampler_log(self) -> dict[str, float]:
    """Scalars about the reset sampler to log (e.g. reset entropy, mean bin return)."""
    ...


def _quat_log(quat: torch.Tensor) -> torch.Tensor:
  """Rotation vector (..., 3) of unit quaternions (..., 4) wxyz, shortest arc."""
  quat = torch.where(quat[..., :1] < 0.0, -quat, quat)
  w = quat[..., 0]
  v = quat[..., 1:]
  norm_sq = (v * v).sum(-1)
  small = norm_sq < _SMALL_ANGLE_SQ
  norm = torch.sqrt(torch.where(small, torch.ones_like(norm_sq), norm_sq))
  w_safe = torch.where(small, w, torch.ones_like(w))
  scale = torch.where(small, 2.0 / w_safe, 2.0 * torch.atan2(norm, w) / norm)
  return v * scale.unsqueeze(-1)


def _quat_exp(rotvec: torch.Tensor) -> torch.Tensor:
  """Unit quaternion (..., 4) wxyz of rotation vectors (..., 3)."""
  angle_sq = (rotvec * rotvec).sum(-1, keepdim=True)
  small = angle_sq < _SMALL_ANGLE_SQ
  angle = torch.sqrt(torch.where(small, torch.ones_like(angle_sq), angle_sq))
  w = torch.where(small, 1.0 - angle_sq / 8.0, torch.cos(0.5 * angle))
  scale = torch.where(small, 0.5 - angle_sq / 48.0, torch.sin(0.5 * angle) / angle)
  return torch.cat((w, rotvec * scale), dim=-1)


class FloatingBaseState:
  """Layout helpers for a floating-base robot state with ``nq`` hinge joints.

  State ``s`` (..., 13 + 2 nq): root position (3, env-local world frame), root
  quaternion wxyz (4), root linear velocity in world (3), root angular velocity in world
  (3), joint positions q (nq), joint velocities qd (nq).

  Delta (..., 12 + 2 nq) from ``s`` to ``s_next``: position change in the root frame of
  ``s`` (3), rotation vector of R_s^T R_next (3), linear and angular velocity changes in
  the root frame of ``s`` (3 + 3), q change (nq), qd change (nq). It contains no world
  position or yaw, so a model predicting it is translation- and heading-invariant.
  """

  def __init__(self, nq: int) -> None:
    self.nq = nq
    self.state_dim = 13 + 2 * nq
    self.delta_dim = 12 + 2 * nq
    self.feature_dim = 10 + 2 * nq

  def split(
    self, s: torch.Tensor
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Split ``s`` (..., state_dim) into (pos, quat, lin_vel_w, ang_vel_w, q, qd)."""
    pos, quat, lin_vel, ang_vel, q, qd = torch.split(
      s, [3, 4, 3, 3, self.nq, self.nq], dim=-1
    )
    return pos, quat, lin_vel, ang_vel, q, qd

  @staticmethod
  def to_root_frame(quat: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    """Express world vectors (..., 3) in the root frame given by ``quat`` (..., 4)."""
    return quat_apply_inverse(quat, vec_w)

  def delta(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """Increment (..., delta_dim) taking ``s`` to ``s_next`` (both (..., state_dim))."""
    pos, quat, lin_vel, ang_vel, q, qd = self.split(s)
    pos1, quat1, lin_vel1, ang_vel1, q1, qd1 = self.split(s_next)
    rel = quat_mul(quat_conjugate(quat), quat1)
    return torch.cat(
      (
        self.to_root_frame(quat, pos1 - pos),
        _quat_log(rel),
        self.to_root_frame(quat, lin_vel1 - lin_vel),
        self.to_root_frame(quat, ang_vel1 - ang_vel),
        q1 - q,
        qd1 - qd,
      ),
      dim=-1,
    )

  def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Apply an increment (..., delta_dim) to ``s`` (..., S); inverse of delta."""
    pos, quat, lin_vel, ang_vel, q, qd = self.split(s)
    d_pos, d_rot, d_lin_vel, d_ang_vel, d_q, d_qd = torch.split(
      delta, [3, 3, 3, 3, self.nq, self.nq], dim=-1
    )
    quat1 = quat_mul(quat, _quat_exp(d_rot))
    quat1 = quat1 / torch.linalg.norm(quat1, dim=-1, keepdim=True)
    return torch.cat(
      (
        pos + quat_apply(quat, d_pos),
        quat1,
        lin_vel + quat_apply(quat, d_lin_vel),
        ang_vel + quat_apply(quat, d_ang_vel),
        q + d_q,
        qd + d_qd,
      ),
      dim=-1,
    )

  def invariant_features(self, s: torch.Tensor) -> torch.Tensor:
    """Features (..., feature_dim) of ``s``: root height, gravity, linear and angular
    velocity in the root frame, q, qd. Free of world xy and yaw."""
    pos, quat, lin_vel, ang_vel, q, qd = self.split(s)
    gravity_w = torch.zeros_like(lin_vel)
    gravity_w[..., 2] = -1.0
    return torch.cat(
      (
        pos[..., 2:3],
        self.to_root_frame(quat, gravity_w),
        self.to_root_frame(quat, lin_vel),
        self.to_root_frame(quat, ang_vel),
        q,
        qd,
      ),
      dim=-1,
    )

  def expand_blocks(self, per_block: tuple[float, ...] | list[float]) -> torch.Tensor:
    """Repeat six per-block values (position, rotation, linear velocity, angular
    velocity, q, qd) to a (delta_dim,) tensor."""
    sizes = (3, 3, 3, 3, self.nq, self.nq)
    assert len(per_block) == len(sizes), "expected one value per delta block"
    return torch.cat(
      [torch.full((n,), float(v)) for n, v in zip(sizes, per_block, strict=True)]
    )
