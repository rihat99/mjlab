"""Action term that lets the policy modulate the PD gains of position actuators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import BuiltinPositionActuator
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_GAIN_FIELDS = ("actuator_gainprm", "actuator_biasprm")


@dataclass(kw_only=True)
class JointGainActionCfg(ActionTermCfg):
  """Configuration for policy-predicted PD gains (variable impedance control).

  Each raw action ``u`` is squashed to a bounded log-multiplier
  ``log m = log_range * tanh(u / s0)`` of the natural frequency, so ``u = 0``
  is exactly the nominal gain set: ``kp = m^2 kp0`` and ``kd = r m kd0`` with
  ``r = 1`` in ``"freq"`` mode (fixed damping ratio) and
  ``log r = damp_log_range * tanh(u_r / s0)`` in ``"freq_damp"`` mode.
  """

  actuator_names: tuple[str, ...] | list[str] = (".*",)
  """Actuator names (regex) whose joints get a gain multiplier."""

  mode: Literal["freq", "freq_damp"] = "freq"
  """``"freq"``: one frequency multiplier per joint. ``"freq_damp"``: plus one
  damping-ratio multiplier per joint (action dim doubles)."""

  log_range: float = math.log(2.0)
  """Bound of ``|log m|``: ``m`` lies in ``[exp(-log_range), exp(log_range)]``."""

  damp_log_range: float = math.log(2.0)
  """Bound of ``|log r|`` in ``"freq_damp"`` mode."""

  s0: float = 2.5
  """Pre-scale of the raw action inside the tanh."""

  def build(self, env: ManagerBasedRlEnv) -> JointGainAction:
    return JointGainAction(self, env)


class JointGainAction(ActionTerm):
  """Write per-env PD gains into the MuJoCo position actuators every step."""

  cfg: JointGainActionCfg
  _entity: Entity

  def __init__(self, cfg: JointGainActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._joint_ids, self._joint_names = self._entity.find_joints_by_actuator_names(
      cfg.actuator_names
    )
    self._num_joints = len(self._joint_ids)
    self._action_dim = (
      2 * self._num_joints if cfg.mode == "freq_damp" else self._num_joints
    )
    self._ctrl_ids = self._resolve_ctrl_ids()

    # Nominal gains of the controlled actuators, from the compiled C model.
    gainprm0 = self._env.sim.get_default_field("actuator_gainprm")
    biasprm0 = self._env.sim.get_default_field("actuator_biasprm")
    self._kp0 = gainprm0[self._ctrl_ids, 0].clone()
    self._kd0 = -biasprm0[self._ctrl_ids, 2].clone()

    # Per-world gain memory: without this the writes below are silently shared
    # by all worlds (mjwarp fields default to a leading dim of 1).
    self._env.sim.expand_model_fields(_GAIN_FIELDS)
    for field in _GAIN_FIELDS:
      tensor = getattr(self._env.sim.model, field)
      assert tensor.shape[0] == self.num_envs and tensor.stride(0) != 0, (
        f"{field} is not expanded per world: shape {tuple(tensor.shape)}, "
        f"stride {tensor.stride()}."
      )

    shape = (self.num_envs, self._num_joints)
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._squash = torch.zeros_like(self._raw_actions)
    self._log_m = torch.zeros(shape, device=self.device)
    self._log_r = torch.zeros(shape, device=self.device)
    self._prev_log_m = torch.zeros(shape, device=self.device)
    self._prev_log_r = torch.zeros(shape, device=self.device)
    self._frozen: tuple[torch.Tensor, torch.Tensor] | None = None
    self._write_gains()

  def _resolve_ctrl_ids(self) -> torch.Tensor:
    """Map each controlled joint to its MuJoCo actuator (ctrl) index."""
    joint_to_ctrl: dict[int, int] = {}
    for actuator in self._entity.actuators:
      if not isinstance(actuator, BuiltinPositionActuator):
        raise TypeError(
          f"JointGainAction needs BuiltinPositionActuator, got {type(actuator).__name__}."
        )
      for joint_id, ctrl_id in zip(
        actuator.target_ids.tolist(), actuator.global_ctrl_ids.tolist(), strict=True
      ):
        joint_to_ctrl[joint_id] = ctrl_id
    ctrl_ids = [joint_to_ctrl[j] for j in self._joint_ids]
    # The pairing above assumes actuator.target_ids[i] drives global_ctrl_ids[i].
    trn_joint = self._env.sim.mj_model.actuator_trnid[ctrl_ids, 0]
    global_joint = self._entity.indexing.joint_ids[self._joint_ids].cpu().numpy()
    assert (trn_joint == global_joint).all(), "joint -> ctrl map mismatch"
    return torch.tensor(ctrl_ids, device=self.device, dtype=torch.long)

  # Properties.

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def joint_names(self) -> list[str]:
    return self._joint_names

  @property
  def ctrl_ids(self) -> torch.Tensor:
    """Global actuator indices, in the order of the gain dims."""
    return self._ctrl_ids

  @property
  def squash(self) -> torch.Tensor:
    """Bounded values in (-1, 1) behind the gains in effect, shape (N, action_dim):
    ``tanh(u / s0)`` of the raw actions, or the equivalent of the frozen values."""
    return self._squash

  @property
  def log_m(self) -> torch.Tensor:
    """Log frequency multiplier in effect, shape (N, J)."""
    return self._log_m

  @property
  def log_r(self) -> torch.Tensor:
    """Log damping-ratio multiplier in effect (zeros in ``"freq"`` mode)."""
    return self._log_r

  @property
  def prev_log_m(self) -> torch.Tensor:
    return self._prev_log_m

  @property
  def prev_log_r(self) -> torch.Tensor:
    return self._prev_log_r

  @property
  def kp(self) -> torch.Tensor:
    return torch.exp(2.0 * self._log_m) * self._kp0

  @property
  def kd(self) -> torch.Tensor:
    return torch.exp(self._log_r + self._log_m) * self._kd0

  # Methods.

  def squash_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Map raw actions (..., action_dim) to (log m, log r), each (..., J)."""
    z = torch.tanh(actions / self.cfg.s0)
    log_m = self.cfg.log_range * z[..., : self._num_joints]
    if self.cfg.mode == "freq_damp":
      log_r = self.cfg.damp_log_range * z[..., self._num_joints :]
    else:
      log_r = torch.zeros_like(log_m)
    return log_m, log_r

  def log_gains_for(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log m, log r) that ``process_actions(actions)`` puts in effect: the frozen
    values when frozen, else ``squash_actions(actions)``."""
    if self._frozen is not None:
      return self._frozen
    return self.squash_actions(actions)

  def squash_of(self, log_m: torch.Tensor, log_r: torch.Tensor) -> torch.Tensor:
    """Bounded values (..., action_dim) that map to the given log multipliers."""
    z = log_m / self.cfg.log_range
    if self.cfg.mode == "freq_damp":
      z = torch.cat([z, log_r / self.cfg.damp_log_range], dim=-1)
    return z

  def freeze(self, log_m: torch.Tensor | None, log_r: torch.Tensor | None = None):
    """Pin the gains to fixed (J,) or (N, J) log multipliers; ``None`` unfreezes."""
    if log_m is None:
      self._frozen = None
      return
    log_m = log_m.to(self.device).expand(self.num_envs, self._num_joints).clone()
    if log_r is None:
      log_r = torch.zeros_like(log_m)
    log_r = log_r.to(self.device).expand(self.num_envs, self._num_joints).clone()
    self._frozen = (log_m, log_r)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    self._prev_log_m[:] = self._log_m
    self._prev_log_r[:] = self._log_r
    self._log_m[:], self._log_r[:] = self.log_gains_for(self._raw_actions)
    self._squash[:] = self.squash_of(self._log_m, self._log_r)
    self._write_gains()

  def apply_actions(self) -> None:
    pass

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._log_m[env_ids] = 0.0
    self._log_r[env_ids] = 0.0
    self._prev_log_m[env_ids] = 0.0
    self._prev_log_r[env_ids] = 0.0
    if self._frozen is not None:
      self._log_m[env_ids] = self._frozen[0][env_ids]
      self._log_r[env_ids] = self._frozen[1][env_ids]
    self._squash[env_ids] = self.squash_of(self._log_m[env_ids], self._log_r[env_ids])
    self._write_gains()

  def _write_gains(self) -> None:
    kp = self.kp
    kd = self.kd
    model = self._env.sim.model
    model.actuator_gainprm[:, self._ctrl_ids, 0] = kp
    model.actuator_biasprm[:, self._ctrl_ids, 1] = -kp
    model.actuator_biasprm[:, self._ctrl_ids, 2] = -kd
