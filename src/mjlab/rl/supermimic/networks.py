"""SuperMimic policy (motion encoder + shared decoder) and critic."""

from __future__ import annotations

import torch
from rsl_rl.modules import MLP, EmpiricalNormalization
from torch import nn
from torch.distributed import distributed_c10d

from mjlab.rl.supermimic.config import PolicyCfg


class MotionPolicy(nn.Module):
  """Deterministic policy a = decoder([tanh(encoder(goal)), observation history]).

  The env hands the observation history flattened term-major (mjlab history groups:
  each term's frames oldest to newest, then the next term); :meth:`forward` takes that
  layout, :meth:`forward_frames` takes frames (B, history, obs_dim) as imagination
  builds them. Observation frames share one running normaliser across the history; the
  goal has its own. Both are updated from simulator data only, by the runner.
  """

  def __init__(
    self,
    obs_term_dims: tuple[int, ...],
    goal_dim: int,
    action_dim: int,
    cfg: PolicyCfg,
  ) -> None:
    super().__init__()
    self.obs_term_dims = tuple(obs_term_dims)
    self.obs_dim = sum(self.obs_term_dims)
    self.history_length = cfg.history_length
    self.encoder = MLP(
      goal_dim,
      cfg.latent_dim,
      cfg.encoder_hidden,
      activation=cfg.activation,
      last_activation="tanh",
    )
    self.decoder = MLP(
      cfg.latent_dim + self.obs_dim * self.history_length,
      action_dim,
      cfg.decoder_hidden,
      activation=cfg.activation,
    )
    # A small output layer keeps the first actions near the default pose.
    last = self.decoder[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
      last.weight.mul_(0.01)
      last.bias.zero_()
    self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
    self.goal_normalizer = EmpiricalNormalization(goal_dim)

  def frames_from_env_layout(self, obs_hist: torch.Tensor) -> torch.Tensor:
    """Term-major history (B, history * obs_dim) -> frames (B, history, obs_dim)."""
    sizes = [dim * self.history_length for dim in self.obs_term_dims]
    terms = torch.split(obs_hist, sizes, dim=-1)
    return torch.cat(
      [
        term.reshape(-1, self.history_length, dim)
        for term, dim in zip(terms, self.obs_term_dims, strict=True)
      ],
      dim=-1,
    )

  def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Action (B, A) from the env's history (B, history * obs_dim) and goal (B, G)."""
    return self.forward_frames(self.frames_from_env_layout(obs_hist), goal)

  def forward_frames(self, frames: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Action (B, A) from observation frames (B, history, obs_dim), oldest first."""
    z = self.encoder(self.goal_normalizer(goal))
    o = self.obs_normalizer(frames).flatten(start_dim=1)
    return self.decoder(torch.cat((z, o), dim=-1))

  @torch.no_grad()
  def update_normalizers(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> None:
    """Update the running statistics with the newest env history frame and the goal."""
    _update_in_place(self.obs_normalizer, self.frames_from_env_layout(obs_hist)[:, -1])
    _update_in_place(self.goal_normalizer, goal)


def _update_in_place(normalizer: EmpiricalNormalization, x: torch.Tensor) -> None:
  """rsl_rl's ``normalizer.update(x)`` (one process, no ``until``), in place: the std
  buffer keeps its storage (rsl_rl assigns a new tensor), so CUDA graphs that read the
  normaliser stay valid, and the batch count is made on the device (rsl_rl copies it
  from the host, which waits for the GPU)."""
  if not normalizer.training:
    return
  # rsl_rl all-reduces the batch statistics across processes; this does not.
  assert not distributed_c10d.is_initialized(), "multi-GPU is not supported"
  count = normalizer.get_buffer("count")
  mean = normalizer.get_buffer("_mean")
  var = normalizer.get_buffer("_var")
  count_x = torch.full_like(count, x.shape[0])
  var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
  mean_x = torch.mean(x, dim=0, keepdim=True)
  count += count_x
  rate = count_x / count
  delta_mean = mean_x - mean
  mean += rate * delta_mean
  var += rate * (var_x - var + delta_mean * (mean_x - mean))
  normalizer.get_buffer("_std").copy_(torch.sqrt(var))


class Critic(nn.Module):
  """State value V(features, goal, xi); running input normaliser, no target net."""

  def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
    super().__init__()
    self.normalizer = EmpiricalNormalization(input_dim)
    self.mlp = MLP(input_dim, 1, hidden_dims, activation="elu")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Value (...,) of critic inputs (..., input_dim)."""
    return self.mlp(self.normalizer(x)).squeeze(-1)
