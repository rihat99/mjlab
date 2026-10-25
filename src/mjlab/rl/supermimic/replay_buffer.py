"""Replay buffer of simulator transitions, a ring over time per env, kept on device."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

# Rejection sampling draws this many candidates per missing sample; valid fractions are
# high (most rows are not near a done or a clip end), so one round is the usual case.
_OVERSAMPLE = 2
_MAX_SAMPLING_ROUNDS = 100


class ReplayBuffer:
  """Ring buffer of the last ``capacity`` transitions of every env.

  Row j of env n holds the transition that started at state s_j: the state, the executed
  (noisy) action, the reference frame k_j, the normalised sim context xi of the
  episode, whether the step ended the episode
  (``done``: terminated or truncated) and whether it ended it by a failure
  (``terminated``), the captured pre-reset state s_{j+1} when it did (the next row then
  starts a new episode) and the step index within the episode. Storage is
  time-major, (capacity, num_envs, ...). Ages count rows from the oldest stored one
  (age 0) to the newest (age size - 1).
  """

  def __init__(
    self,
    capacity: int,
    num_envs: int,
    state_dim: int,
    action_dim: int,
    context_dim: int,
    device: str | torch.device,
  ) -> None:
    self.capacity = capacity
    self.num_envs = num_envs
    self.device = torch.device(device)
    shape = (capacity, num_envs)
    self.state = torch.zeros((*shape, state_dim), device=self.device)
    self.action = torch.zeros((*shape, action_dim), device=self.device)
    self.frame = torch.zeros(shape, dtype=torch.long, device=self.device)
    self.context = torch.zeros((*shape, context_dim), device=self.device)
    self.done = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.terminated = torch.zeros(shape, dtype=torch.bool, device=self.device)
    self.terminal_state = torch.zeros((*shape, state_dim), device=self.device)
    self.episode_step = torch.zeros(shape, dtype=torch.int32, device=self.device)
    self.ptr = 0
    self.size = 0

  def add(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    frame: torch.Tensor,
    context: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor,
    terminal_state: torch.Tensor,
    episode_step: torch.Tensor,
  ) -> None:
    """Append one step of all envs; every argument has leading dim num_envs."""
    row = self.ptr
    self.state[row] = state
    self.action[row] = action
    self.frame[row] = frame
    self.context[row] = context
    self.done[row] = done
    self.terminated[row] = terminated
    self.terminal_state[row] = terminal_state
    self.episode_step[row] = episode_step
    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  def _rows(self, age: torch.Tensor) -> torch.Tensor:
    """Storage rows of ages (any shape)."""
    return (self.ptr - self.size + age) % self.capacity

  def _next_state(self, rows: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """State reached by the transitions at (rows, env): the next row's state, or the
    captured terminal state where the transition ended the episode."""
    next_rows = (rows + 1) % self.capacity
    return torch.where(
      self.done[rows, env].unsqueeze(-1),
      self.terminal_state[rows, env],
      self.state[next_rows, env],
    )

  def _rejection_sample(
    self,
    batch_size: int,
    max_age: int,
    is_valid: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw (age, env) pairs uniformly among those accepted by ``is_valid``.

    Ages are drawn from [0, max_age). Returns ages (B,) and envs (B,).
    """
    ages, envs = [], []
    missing = batch_size
    for _ in range(_MAX_SAMPLING_ROUNDS):
      num = _OVERSAMPLE * missing
      age = torch.randint(0, max_age, (num,), device=self.device)
      env = torch.randint(0, self.num_envs, (num,), device=self.device)
      keep = torch.nonzero(is_valid(age, env)).squeeze(-1)[:missing]
      ages.append(age[keep])
      envs.append(env[keep])
      missing -= keep.numel()
      if missing == 0:
        return torch.cat(ages), torch.cat(envs)
    raise RuntimeError(
      "Replay buffer: too few valid rows to sample from (are all clips shorter than "
      "the imagination horizon plus the seed margin, or is the buffer nearly empty?)."
    )

  def sample_windows(self, batch_size: int, k: int) -> dict[str, torch.Tensor]:
    """Sample k-step windows for the world-model loss.

    A window starting at row j uses rows j..j+k-1; it is valid when none of rows
    j..j+k-2 ended an episode (so it never crosses a reset) and every state it needs is
    stored. Only its last transition may be a done; its target is then the captured
    terminal state.

    Returns:
      s0 (B, S) start state; actions (B, k, A) executed actions; targets (B, k, S)
      states reached by each transition; valid (B, k) float (1 up to and including
      the first done); context (B, C) the window's sim context (one episode);
      env_idx (B,) int64.
    """
    assert self.size >= k + 1, "replay buffer holds fewer than k + 1 steps"
    offsets = torch.arange(k, device=self.device)

    def is_valid(age: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
      rows = self._rows(age.unsqueeze(-1) + offsets[:-1])
      return ~self.done[rows, env.unsqueeze(-1)].any(dim=-1)

    # Rows j..j+k must be stored so that every transition's next state is known.
    age, env = self._rejection_sample(batch_size, self.size - k, is_valid)
    rows = self._rows(age.unsqueeze(-1) + offsets)
    env_k = env.unsqueeze(-1).expand_as(rows)
    done = self.done[rows, env_k]
    valid = torch.cumprod(
      torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), dim=-1).float(), dim=-1
    )
    return {
      "s0": self.state[rows[:, 0], env],
      "actions": self.action[rows, env_k],
      "targets": self._next_state(rows, env_k),
      "valid": valid,
      "context": self.context[rows[:, 0], env],
      "env_idx": env,
    }

  def sample_seeds(
    self,
    batch_size: int,
    history: int,
    horizon: int,
    clip_end_fn: Callable[[torch.Tensor], torch.Tensor],
  ) -> dict[str, torch.Tensor]:
    """Sample start states for imagined rollouts, with their observation history.

    A row j is a valid seed when it is not a done, its frame satisfies
    ``frame + horizon < clip_end(frame)`` (every frame of the rollout, including the
    bootstrap frame k_0 + horizon, stays inside the clip) and the rows of its history,
    plus the row before the oldest, are still stored. History rows before the episode
    start repeat the episode's first row with a zero previous action, as mjlab's
    history backfill does at reset.

    Returns:
      s0 (B, S); frame0 (B,) int64; context (B, C) the sim context of the seed's
      episode (its history rows share it); env_idx (B,) int64; state_hist
      (B, history, S) oldest first, ending at s0; prev_action_hist (B, history, A) the
      action executed before each history state (zeros at episode start) and
      frame_hist (B, history) the frame shown with it, so frame m of the observation
      history is observation(state_hist[:, m], prev_action_hist[:, m], context,
      frame_hist[:, m]); a_prev (B, A) the action executed before s0
      (= prev_action_hist[:, -1]).
    """

    def is_valid(age: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
      rows = self._rows(age)
      frame = self.frame[rows, env]
      ep_step = self.episode_step[rows, env].long()
      return (
        ~self.done[rows, env]
        & (frame + horizon < clip_end_fn(frame))
        & (age >= ep_step.clamp(max=history))
      )

    age, env = self._rejection_sample(batch_size, self.size, is_valid)
    rows = self._rows(age)
    ep_step = self.episode_step[rows, env].long()
    back = torch.arange(history - 1, -1, -1, device=self.device)
    back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # (B, history)
    hist_age = age.unsqueeze(-1) - back
    env_h = env.unsqueeze(-1).expand_as(hist_age)
    hist_ep_step = ep_step.unsqueeze(-1) - back
    prev_action = self.action[self._rows(hist_age - 1), env_h]
    prev_action_hist = torch.where(
      (hist_ep_step > 0).unsqueeze(-1), prev_action, torch.zeros_like(prev_action)
    )
    return {
      "s0": self.state[rows, env],
      "frame0": self.frame[rows, env],
      "context": self.context[rows, env],
      "env_idx": env,
      "state_hist": self.state[self._rows(hist_age), env_h],
      "prev_action_hist": prev_action_hist,
      "frame_hist": self.frame[self._rows(hist_age), env_h],
      "a_prev": prev_action_hist[:, -1],
    }

  def iter_transitions(self, chunk_size: int) -> Iterator[dict[str, torch.Tensor]]:
    """Yield every stored transition whose next state is known, in chunks.

    Each chunk has state (M, S), action (M, A), next_state (M, S), context (M, C),
    env_idx (M,).
    """
    num_ages = self.size - 1
    ages = torch.arange(num_ages, device=self.device).repeat_interleave(self.num_envs)
    envs = torch.arange(self.num_envs, device=self.device).repeat(num_ages)
    for start in range(0, ages.numel(), chunk_size):
      rows = self._rows(ages[start : start + chunk_size])
      env = envs[start : start + chunk_size]
      yield {
        "state": self.state[rows, env],
        "action": self.action[rows, env],
        "next_state": self._next_state(rows, env),
        "context": self.context[rows, env],
        "env_idx": env,
      }
