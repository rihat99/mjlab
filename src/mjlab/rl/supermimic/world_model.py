"""World model: deterministic dynamics f(s, a, xi) and a learned termination head."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from rsl_rl.modules import MLP, EmpiricalNormalization
from torch import nn

from mjlab.rl.supermimic.config import WorldModelCfg
from mjlab.rl.supermimic.interface import FloatingBaseState, ModelInterface
from mjlab.rl.supermimic.networks import _update_in_place


class TerminationHead(nn.Module):
  """Termination probability of a state given its goal: d = sigmoid(head(f, g)).

  The input statistics come from logged transitions
  (:meth:`WorldModel.update_statistics`).
  """

  def __init__(
    self, feature_dim: int, goal_dim: int, hidden_dims: tuple[int, ...]
  ) -> None:
    super().__init__()
    self.normalizer = EmpiricalNormalization(feature_dim + goal_dim)
    self.mlp = MLP(feature_dim + goal_dim, 1, hidden_dims, activation="elu")

  def forward(self, features: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Termination logit (B,) from features (B, F) and goal (B, G)."""
    x = self.normalizer(torch.cat((features, goal), dim=-1))
    return self.mlp(x).squeeze(-1)


class WorldModel(nn.Module):
  """Deterministic dynamics in normalised increments, plus the termination head.

  The MLP maps [features(s), a, xi] (normalised) to a normalised increment y; the
  increment is y * nominal_scale and the next state is ``integrate(s, increment)``. The
  last layer is zero-initialised so an untrained model predicts "no change". Input
  statistics and nominal scales are fitted from the replay buffer at the end of the
  prefill (:meth:`fit_statistics`) and then either frozen or kept running
  (:meth:`update_statistics`, ``WorldModelCfg.running_statistics``); they are buffers
  (saved with the state dict, no gradient) and are updated in place, because the CUDA
  graphs of the compiled loss and imagination step read them by address.
  """

  nominal_scale: torch.Tensor
  nominal_floor: torch.Tensor
  statistics_fitted: torch.Tensor

  def __init__(
    self,
    state: FloatingBaseState,
    action_dim: int,
    context_dim: int,
    goal_dim: int,
    cfg: WorldModelCfg,
  ) -> None:
    super().__init__()
    self.state = state
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
    self.termination_head = TerminationHead(
      state.feature_dim, goal_dim, cfg.termination_head_hidden
    )
    # Running statistics of the logged increments; the nominal scale is their std.
    self.increment_statistics = EmpiricalNormalization(state.delta_dim)
    self.register_buffer("nominal_scale", torch.ones(state.delta_dim))
    self.register_buffer(
      "nominal_floor", state.expand_blocks(cfg.nominal_floor), persistent=False
    )
    self.register_buffer("statistics_fitted", torch.tensor(False))

  def normalized_increment(
    self, features: torch.Tensor, action: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    """Predicted increment divided by the nominal scale, (B, delta_dim).

    Args:
      features: (B, F) invariant features of the current state.
      action: (B, A) action.
      context: (B, C) normalised sim context.
    """
    x = self.input_normalizer(torch.cat((features, action, context), dim=-1))
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
    self, chunks: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
  ) -> None:
    """Fit input statistics and nominal scales from logged transitions (end of prefill).

    Each chunk is (inputs (M, F + A + C) = [features(s), a, xi], increments
    (M, delta_dim) = delta(s, s_next), head inputs (M, F + G) = [features(s_next),
    goal]); see :meth:`update_statistics`.
    """
    for inputs, increments, head_inputs in chunks:
      self.update_statistics(inputs, increments, head_inputs)
    self.statistics_fitted.fill_(True)

  @torch.no_grad()
  def update_statistics(
    self,
    inputs: torch.Tensor,
    increments: torch.Tensor,
    head_inputs: torch.Tensor,
    max_count: int | None = None,
  ) -> None:
    """Update the running statistics with logged transitions, in place.

    rsl_rl's running mean / variance of the dynamics inputs (M, F + A + C), the
    increments (M, delta_dim) and the head inputs (M, F + G); the nominal scale is the
    per-dim std of the increments, floored by ``nominal_floor``. With ``max_count`` each
    statistic's sample count is first capped there, which turns the running average
    into an exponential moving average: M new samples get weight M / (max_count + M).
    """
    pairs = (
      (self.input_normalizer, inputs),
      (self.termination_head.normalizer, head_inputs),
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
  pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """k-step autoregressive loss of the world model on replay windows (Eq. 1).

  Starting from the logged s_0, the model is rolled out on the logged actions. At step
  m the state error is the difference between the predicted increment and the increment
  that would take the *predicted* previous state to the logged one, both divided by the
  nominal scale::

    e_m = (delta(s_hat_{m-1}, s_hat_m) - delta(s_hat_{m-1}, s_m)) / nominal
        = y_m - delta(s_hat_{m-1}, s_m) / nominal

  so the model learns to steer its own drift back. The state loss is the mean of e_m^2
  over the delta dims and the valid steps. The termination head sees the predicted
  state s_hat_m with its goal at frame k_{m-1}, the frame the logged ``terminated``
  flag was judged against, and is trained with BCE on that flag. Gradients flow
  through the whole window (no detach).

  Args:
    model: the world model.
    iface: the task's model interface (features, goal, clip ends).
    windows: output of ``ReplayBuffer.sample_windows``.
    context: (B, C) normalised sim context of the window envs.
    pos_weight: positive-class weight of the BCE.

  Returns:
    state loss (), termination loss (), mean e^2 at the last window step ().
  """
  s_hat = windows["s0"]
  valid = windows["valid"]
  num, k = valid.shape
  features = iface.invariant_features(s_hat)
  prev_states, increments, next_states, next_features = [], [], [], []
  for m in range(k):
    y = model.normalized_increment(features, windows["actions"][:, m], context)
    prev_states.append(s_hat)
    increments.append(y)
    s_hat = model.state.integrate(s_hat, y * model.nominal_scale)
    features = iface.invariant_features(s_hat)
    next_states.append(s_hat)
    next_features.append(features)

  # The error and head terms do not feed back into the rollout, so they run once over
  # the whole window (B, k, ...): far fewer kernel launches than per step.
  prev = torch.stack(prev_states, dim=1)
  target = model.state.delta(prev, windows["targets"]) / model.nominal_scale
  state_error = (torch.stack(increments, dim=1) - target).square().mean(dim=-1)
  frames = windows["frames"].flatten()
  goal = iface.goal(torch.stack(next_states, dim=1).flatten(0, 1), frames)
  logit = model.termination_head(
    torch.stack(next_features, dim=1), goal.view(num, k, -1)
  )
  bce = F.binary_cross_entropy_with_logits(
    logit,
    windows["terminated"].float(),
    pos_weight=torch.tensor(pos_weight, device=logit.device),
    reduction="none",
  )
  num_valid = valid.sum().clamp(min=1.0)
  state_loss = (state_error * valid).sum() / num_valid
  termination_loss = (bce * valid).sum() / num_valid
  last_error = (state_error[:, -1] * valid[:, -1]).sum() / valid[:, -1].sum().clamp(
    min=1.0
  )
  return state_loss, termination_loss, last_error
