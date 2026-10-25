"""Return-based reset sampler over the bins of a motion library (paper App. D.1)."""

from __future__ import annotations

import math

import torch


class ReturnSampler:
  """Reset distribution over bins, weighted towards bins of low observed return.

  Bin b covers the frames [bin_start[b], bin_end[b]) of one clip. Each bin keeps a
  return estimate G_b, an exponential moving average of the discounted returns
  observed from it (:meth:`credit_episodes`, fed by the runner), and an EMA of past
  softmax weights (:meth:`update_past_weights`, once per iteration). A reset draws a
  bin from

    p = 0.5 uniform + 0.4 softmax(-25 G_b / G_max) + 0.1 past weights

  (:meth:`probabilities`) and a frame uniformly inside it, never a clip's last frame
  (:meth:`sample_frames`, called by the env's command term). The state tensors are
  updated in place.
  """

  def __init__(
    self,
    bin_of_frame: torch.Tensor,
    bin_start: torch.Tensor,
    bin_end: torch.Tensor,
    bin_clip_end: torch.Tensor,
    g_max: float,
    return_alpha: float,
    past_weight_alpha: float,
  ) -> None:
    """Args:
    bin_of_frame: (F,) int64 bin of every library frame.
    bin_start, bin_end: (Nb,) int64 frame range of each bin.
    bin_clip_end: (Nb,) int64 exclusive end frame of each bin's clip.
    g_max: largest attainable discounted return (scales the softmax).
    return_alpha: EMA factor of the bin return estimates.
    past_weight_alpha: EMA factor of the past softmax weights, per iteration.
    """
    self.bin_of_frame = bin_of_frame
    self.bin_start = bin_start
    # Exclusive end of the reset frames of each bin: never a clip's last frame (the
    # reset's frame advance would leave the clip).
    self._reset_stop = torch.minimum(bin_end, bin_clip_end - 1)
    self.bin_count = int(bin_start.numel())
    self.g_max = g_max
    self.return_alpha = return_alpha
    self.past_weight_alpha = past_weight_alpha
    device = bin_start.device
    self.bin_returns = torch.zeros(self.bin_count, device=device)
    self.bin_past_weights = torch.full(
      (self.bin_count,), 1.0 / self.bin_count, device=device
    )

  def probabilities(self) -> torch.Tensor:
    """Reset distribution over bins (Nb,)."""
    return 0.5 / self.bin_count + 0.4 * self._softmax() + 0.1 * self.bin_past_weights

  def sample_frames(self, num: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reset frames (num,) int64 and the distribution (Nb,) their bins were drawn
    from."""
    probabilities = self.probabilities()
    bins = torch.multinomial(probabilities, num, replacement=True)
    start = self.bin_start[bins]
    stop = self._reset_stop[bins]
    offset = (torch.rand(num, device=start.device) * (stop - start)).long()
    return torch.minimum(start + offset, stop - 1), probabilities

  def credit_episodes(
    self, frames: torch.Tensor, returns: torch.Tensor, valid: torch.Tensor
  ) -> None:
    """Feed the returns of finished episodes to the bin estimates: every bin an
    episode entered gets the return-to-go at its first step in that bin.

    Args:
      frames: (M, L) int64 frame of each episode step.
      returns: (M, L) return-to-go G_t of each step.
      valid: (M, L) bool, True for the steps of the episode (L >= its length).
    """
    bins = self.bin_of_frame[frames.flatten()].view_as(frames)
    entered = torch.ones_like(valid)
    entered[:, 1:] = bins[:, 1:] != bins[:, :-1]
    first = valid & entered
    first_returns = returns[first].float()
    if not torch.isfinite(first_returns).all():
      raise RuntimeError("Non-finite episode return for the reset sampler.")
    self.update_returns(bins[first], first_returns)

  def update_returns(self, bins: torch.Tensor, returns: torch.Tensor) -> None:
    """EMA of the observed returns per bin; duplicate bins are averaged first.

    Args:
      bins: (M,) int64 bin indexes.
      returns: (M,) returns observed from those bins.
    """
    # No bincount or boolean-mask indexing: both wait for the GPU.
    total = torch.zeros_like(self.bin_returns).index_add_(0, bins, returns)
    count = torch.zeros_like(self.bin_returns).index_add_(
      0, bins, torch.ones_like(returns)
    )
    alpha = self.return_alpha
    update = (1 - alpha) * self.bin_returns + alpha * (total / count.clamp(min=1.0))
    self.bin_returns.copy_(torch.where(count > 0, update, self.bin_returns))

  def update_past_weights(self) -> None:
    """EMA of the softmax distribution; called once per training iteration."""
    self.bin_past_weights.lerp_(self._softmax(), self.past_weight_alpha)

  def _softmax(self) -> torch.Tensor:
    return torch.softmax(-25.0 * self.bin_returns / self.g_max, dim=0)

  def state_dict(self) -> dict[str, torch.Tensor]:
    """Bin returns G_b and past weights, (Nb,) each."""
    return {
      "bin_returns": self.bin_returns.clone(),
      "bin_past_weights": self.bin_past_weights.clone(),
    }

  def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    """Restore :meth:`state_dict`; the library must have the same bins."""
    self.bin_returns.copy_(state["bin_returns"])
    self.bin_past_weights.copy_(state["bin_past_weights"])

  def log(self) -> dict[str, float]:
    """Normalised entropy of the reset distribution and mean bin return."""
    probs = self.probabilities()
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    count = probs.numel()
    return {
      "reset_entropy": float(entropy) / math.log(count) if count > 1 else 1.0,
      "mean_bin_return": float(self.bin_returns.mean()),
    }
