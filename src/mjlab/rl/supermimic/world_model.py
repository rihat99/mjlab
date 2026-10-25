"""World model: deterministic dynamics f(s, a, xi)."""

from __future__ import annotations

import torch
from rsl_rl.modules import EmpiricalNormalization
from torch import nn

from mjlab.rl.supermimic.config import WorldModelCfg
from mjlab.rl.supermimic.interface import FloatingBaseState, ModelInterface
from mjlab.rl.supermimic.networks import _update_in_place
from mjlab.rl.supermimic.replay_buffer import ReplayBuffer

# Transitions per chunk of the end-of-prefill statistics fit.
_STATISTICS_CHUNK = 16384


class WorldModel(nn.Module):
  """Deterministic dynamics in normalised increments.

  The MLP maps [features(s), a, xi] (normalised) to a normalised increment y; the
  increment is y * nominal_scale and the next state is ``integrate(s, increment)``. The
  last layer is zero-initialised so an untrained model predicts "no change". Input
  statistics and nominal scales are fitted from the replay buffer at the end of the
  prefill (:meth:`fit_statistics`) and then kept running on the logged transitions of
  every fit's windows (:meth:`update_statistics_from_windows`,
  ``WorldModelCfg.statistics_horizon``); they are buffers (saved with the state dict,
  no gradient) and are updated in place, because the CUDA graphs of the compiled loss
  and imagination step read them by address.
  """

  nominal_scale: torch.Tensor
  nominal_floor: torch.Tensor
  statistics_fitted: torch.Tensor

  def __init__(
    self,
    state: FloatingBaseState,
    action_dim: int,
    context_dim: int,
    cfg: WorldModelCfg,
  ) -> None:
    super().__init__()
    self.state = state
    self.statistics_horizon = cfg.statistics_horizon
    input_dim = state.feature_dim + action_dim + context_dim
    self.input_normalizer = EmpiricalNormalization(input_dim)
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in cfg.hidden_dims:
      layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU()]
      width = hidden
    last = nn.Linear(width, state.delta_dim)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    self.dynamics = nn.Sequential(*layers, last)
    # Running statistics of the logged increments; the nominal scale is their std.
    self.increment_statistics = EmpiricalNormalization(state.delta_dim)
    self.register_buffer("nominal_scale", torch.ones(state.delta_dim))
    self.register_buffer(
      "nominal_floor", state.expand_blocks(cfg.nominal_floor), persistent=False
    )
    self.register_buffer("statistics_fitted", torch.tensor(False))

  @staticmethod
  def dynamics_input(
    features: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    """Unnormalised dynamics input (..., F + A + C) = [features(s), a, xi]."""
    return torch.cat((features, action, context), dim=-1)

  def normalized_increment(
    self, features: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    """Predicted increment divided by the nominal scale, (B, delta_dim).

    Args:
      features: (B, F) invariant features of the current state.
      action: (B, A) action.
      context: (B, C) normalised sim context.
    """
    x = self.input_normalizer(self.dynamics_input(features, action, context))
    return self.dynamics(x)

  def step(
    self,
    s: torch.Tensor,
    features: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
  ) -> torch.Tensor:
    """Next state (B, S) from state ``s`` (B, S) and its features (B, F)."""
    y = self.normalized_increment(features, action, context)
    return self.state.integrate(s, y * self.nominal_scale)

  @torch.no_grad()
  def fit_statistics(
    self, buffer: ReplayBuffer, iface: ModelInterface, context: torch.Tensor
  ) -> None:
    """Fit input statistics and nominal scales from every transition in the buffer
    (end of prefill); see :meth:`update_statistics`.

    Args:
      buffer: the replay buffer.
      iface: the task's model interface (invariant features).
      context: (N, C) normalised sim context of every env.
    """
    for chunk in buffer.iter_transitions(_STATISTICS_CHUNK):
      s, s_next = chunk["state"], chunk["next_state"]
      inputs = self.dynamics_input(
        iface.invariant_features(s), chunk["action"], context[chunk["env_idx"]]
      )
      self.update_statistics(inputs, self.state.delta(s, s_next))
    self.statistics_fitted.fill_(True)

  @torch.no_grad()
  def update_statistics_from_windows(
    self,
    batches: list[dict[str, torch.Tensor]],
    iface: ModelInterface,
    context: torch.Tensor,
  ) -> None:
    """Running update of the statistics from the logged transitions of sampled windows
    (real data only, never imagined states).

    Transition m of a window goes from the logged state s_m (s_0, then the targets) to
    targets[:, m] under actions[:, m]. The moving-average horizon is
    ``statistics_horizon`` updates of this size.

    Args:
      batches: outputs of ``ReplayBuffer.sample_windows``.
      iface: the task's model interface (invariant features).
      context: (N, C) normalised sim context of every env.
    """

    def cat(key: str) -> torch.Tensor:
      return torch.cat([windows[key] for windows in batches])

    s0, actions, targets = cat("s0"), cat("actions"), cat("targets")
    valid = cat("valid") > 0  # (B, k)
    num, k = valid.shape
    next_features = iface.invariant_features(targets.flatten(0, 1)).view(num, k, -1)
    features = torch.cat(
      (iface.invariant_features(s0).unsqueeze(1), next_features[:, :-1]), dim=1
    )
    states = torch.cat((s0.unsqueeze(1), targets[:, :-1]), dim=1)
    window_context = context[cat("env_idx")].unsqueeze(1).expand(-1, k, -1)
    inputs = self.dynamics_input(features, actions, window_context)[valid]
    increments = self.state.delta(states, targets)[valid]
    self.update_statistics(
      inputs, increments, max_count=self.statistics_horizon * inputs.shape[0]
    )

  @torch.no_grad()
  def update_statistics(
    self,
    inputs: torch.Tensor,
    increments: torch.Tensor,
    max_count: int | None = None,
  ) -> None:
    """Update the running statistics with logged transitions, in place.

    rsl_rl's running mean / variance of the dynamics inputs (M, F + A + C) and the
    increments (M, delta_dim); the nominal scale is the per-dim std of the increments,
    floored by ``nominal_floor``. With ``max_count`` each statistic's sample count is
    first capped there, which turns the running average into an exponential moving
    average: M new samples get weight M / (max_count + M).
    """
    pairs = (
      (self.input_normalizer, inputs),
      (self.increment_statistics, increments),
    )
    for normalizer, x in pairs:
      if max_count is not None:
        normalizer.get_buffer("count").clamp_(max=max_count)
      _update_in_place(normalizer, x)
    self.nominal_scale.copy_(
      torch.maximum(self.increment_statistics.std, self.nominal_floor)
    )


def world_model_loss(
  model: WorldModel,
  iface: ModelInterface,
  windows: dict[str, torch.Tensor],
  context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """k-step autoregressive loss of the world model on replay windows (Eq. 1).

  Starting from the logged s_0, the model is rolled out on the logged actions. At step
  m the state error is the difference between the predicted increment and the increment
  that would take the *predicted* previous state to the logged one, both divided by the
  nominal scale::

    e_m = (delta(s_hat_{m-1}, s_hat_m) - delta(s_hat_{m-1}, s_m)) / nominal
        = y_m - delta(s_hat_{m-1}, s_m) / nominal

  so the model learns to steer its own drift back. The state loss is the mean of e_m^2
  over the delta dims and the valid steps. Gradients flow through the whole window (no
  detach).

  Args:
    model: the world model.
    iface: the task's model interface (invariant features).
    windows: output of ``ReplayBuffer.sample_windows``.
    context: (B, C) normalised sim context of the window envs.

  Returns:
    state loss (), mean e^2 at the last window step ().
  """
  s_hat = windows["s0"]
  valid = windows["valid"]
  prev_states, increments = [], []
  for m in range(valid.shape[1]):
    features = iface.invariant_features(s_hat)
    y = model.normalized_increment(features, windows["actions"][:, m], context)
    prev_states.append(s_hat)
    increments.append(y)
    s_hat = model.state.integrate(s_hat, y * model.nominal_scale)

  # The error terms do not feed back into the rollout, so they run once over the whole
  # window (B, k, ...): far fewer kernel launches than per step.
  prev = torch.stack(prev_states, dim=1)
  target = model.state.delta(prev, windows["targets"]) / model.nominal_scale
  state_error = (torch.stack(increments, dim=1) - target).square().mean(dim=-1)
  state_loss = (state_error * valid).sum() / valid.sum().clamp(min=1.0)
  last_error = (state_error[:, -1] * valid[:, -1]).sum() / valid[:, -1].sum().clamp(
    min=1.0
  )
  return state_loss, last_error
