"""SuperMimic updates: world-model fit, BPTT policy update, critic fit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

from mjlab.rl.supermimic.config import CriticCfg, PolicyCfg, WorldModelCfg
from mjlab.rl.supermimic.interface import FloatingBaseState, ModelInterface
from mjlab.rl.supermimic.networks import Critic, MotionPolicy
from mjlab.rl.supermimic.replay_buffer import ReplayBuffer
from mjlab.rl.supermimic.world_model import WorldModel, world_model_loss

_STATISTICS_CHUNK = 16384


@dataclass
class ImaginedRollout:
  """Detached imagined trajectories for the critic fit.

  features (H + 1, B, F) and goals (H + 1, B, G) of s_0..s_H; rewards (H, B) r_1..r_H;
  terminations (H, B) predicted d_1..d_H; context (B, C).
  """

  features: torch.Tensor
  goals: torch.Tensor
  rewards: torch.Tensor
  terminations: torch.Tensor
  context: torch.Tensor


def lambda_returns(
  rewards: torch.Tensor,
  terminations: torch.Tensor,
  values: torch.Tensor,
  gamma: float,
  lam: float,
) -> torch.Tensor:
  """Lambda-return critic targets (paper Eq. 3), computed backwards.

  R_t = r_{t+1} + gamma (1 - d_{t+1}) ((1 - lam) V_{t+1} + lam R_{t+1}),  R_H = V_H.

  Args:
    rewards: (H, B) r_{t+1} for t = 0..H-1.
    terminations: (H, B) termination probabilities d_{t+1}.
    values: (H + 1, B) V_0..V_H.

  Returns:
    (H + 1, B) targets R_0..R_H.
  """
  returns = [values[-1]]
  for t in range(rewards.shape[0] - 1, -1, -1):
    blend = (1.0 - lam) * values[t + 1] + lam * returns[-1]
    returns.append(rewards[t] + gamma * (1.0 - terminations[t]) * blend)
  return torch.stack(returns[::-1])


def _maybe_compile(fn: Callable, enabled: bool, mode: str | None = None) -> Callable:
  """``torch.compile(fn, mode=mode)`` when enabled, else ``fn`` itself."""
  return torch.compile(fn, mode=mode) if enabled else fn


def clip_policy_gradients(
  parameters: Iterable[nn.Parameter], max_norm: float, max_value: float
) -> torch.Tensor:
  """Clip gradients to a global norm, then element-wise; returns the pre-clip norm."""
  params = [p for p in parameters if p.grad is not None]
  norm = nn.utils.clip_grad_norm_(params, max_norm)
  nn.utils.clip_grad_value_(params, max_value)
  return norm


class SuperMimic:
  """World model, policy and critic with their optimisers and update rules.

  The task enters only through ``iface`` (a :class:`ModelInterface`).
  """

  def __init__(
    self,
    iface: ModelInterface,
    world_model_cfg: WorldModelCfg,
    policy_cfg: PolicyCfg,
    critic_cfg: CriticCfg,
    device: str | torch.device,
    compile: bool = False,
  ) -> None:
    self.iface = iface
    self.world_model_cfg = world_model_cfg
    self.policy_cfg = policy_cfg
    self.critic_cfg = critic_cfg
    self.device = torch.device(device)
    self.state = FloatingBaseState(iface.nq)
    assert self.state.state_dim == iface.state_dim
    assert self.state.feature_dim == iface.feature_dim
    self.world_model = WorldModel(
      self.state, iface.action_dim, iface.context_dim, iface.goal_dim, world_model_cfg
    ).to(self.device)
    self.policy = MotionPolicy(
      iface.obs_term_dims, iface.goal_dim, iface.action_dim, policy_cfg
    ).to(self.device)
    self.critic = Critic(
      iface.feature_dim + iface.goal_dim + iface.context_dim, critic_cfg.hidden
    ).to(self.device)
    # With compile, Adam runs fused (one kernel per step, same update rule).
    fused = True if compile else None
    self.world_model_optimizer = torch.optim.Adam(
      self.world_model.parameters(), lr=world_model_cfg.lr, fused=fused
    )
    self.policy_optimizer = torch.optim.Adam(
      self.policy.parameters(),
      lr=policy_cfg.lr,
      betas=tuple(policy_cfg.betas),
      fused=fused,
    )
    self.critic_optimizer = torch.optim.Adam(
      self.critic.parameters(), lr=critic_cfg.lr, fused=fused
    )
    self.skipped_policy_steps = 0
    # Compiled twins of the eager hot paths (same math; random draws stay eager).
    # "reduce-overhead" adds CUDA graphs, which read parameters and buffers in place:
    # used where every buffer read keeps its storage (Adam and the policy normalisers
    # update in place); the objective reads the critic normaliser, which rsl_rl
    # replaces each iteration, and the state heads also serve variable-size batches.
    graphs = "reduce-overhead"
    self._world_model_loss = _maybe_compile(world_model_loss, compile, graphs)
    self._state_heads = _maybe_compile(self.state_heads, compile)
    self._imagine_step = _maybe_compile(self.imagine_step, compile, graphs)
    self._policy_objective = _maybe_compile(self.policy_objective, compile)
    self._critic_loss = _maybe_compile(self.critic_loss, compile, graphs)

  def set_learning_rate_scale(self, scale: float) -> None:
    """Set every optimiser's learning rate to its configured value times ``scale``."""
    pairs = (
      (self.world_model_optimizer, self.world_model_cfg.lr),
      (self.policy_optimizer, self.policy_cfg.lr),
      (self.critic_optimizer, self.critic_cfg.lr),
    )
    for optimizer, base_lr in pairs:
      for group in optimizer.param_groups:
        group["lr"] = base_lr * scale

  @torch.no_grad()
  def act(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Deterministic action (N, A) from the env history (N, H * O) and goal (N, G)."""
    return self.policy(obs_hist, goal)

  def critic_input(
    self, features: torch.Tensor, goal: torch.Tensor, context: torch.Tensor
  ) -> torch.Tensor:
    """Critic input (..., F + G + C)."""
    return torch.cat((features, goal, context), dim=-1)

  @torch.no_grad()
  def value(
    self, s: torch.Tensor, frame: torch.Tensor, env_idx: torch.Tensor
  ) -> torch.Tensor:
    """V (M,) of states s (M, S) whose next observation targets ``frame`` (M,)."""
    features, goal = self._state_heads(s, frame)
    context = self.iface.context_normalized()[env_idx]
    x = self.critic_input(features, goal, context)
    return self.critic(x)

  # World model.

  def _statistics_chunks(
    self, buffer: ReplayBuffer, context_all: torch.Tensor
  ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    iface = self.iface
    for chunk in buffer.iter_transitions(_STATISTICS_CHUNK):
      s, s_next = chunk["state"], chunk["next_state"]
      context = context_all[chunk["env_idx"]]
      inputs = torch.cat((iface.invariant_features(s), chunk["action"], context), -1)
      increments = self.state.delta(s, s_next)
      frame = chunk["frame"]
      goal = iface.goal(s_next, torch.minimum(frame + 1, iface.clip_end(frame) - 1))
      head_inputs = torch.cat((iface.invariant_features(s_next), goal), dim=-1)
      yield inputs, increments, head_inputs

  @torch.no_grad()
  def fit_world_model_statistics(self, buffer: ReplayBuffer) -> None:
    """Fit the world model's frozen input statistics and nominal scales (end of
    prefill) from every transition in the buffer."""
    context_all = self.iface.context_normalized()
    self.world_model.fit_statistics(self._statistics_chunks(buffer, context_all))

  def fit_world_model(self, buffer: ReplayBuffer) -> dict[str, torch.Tensor]:
    """``batches_per_update`` Adam steps of the k-step loss; returns mean metrics."""
    cfg = self.world_model_cfg
    context_all = self.iface.context_normalized()
    sums = torch.zeros(3, device=self.device)
    # Sampling syncs with the host; drawing every batch up front (same random stream,
    # nothing else draws in between) keeps those syncs out of the update loop.
    batches = [
      buffer.sample_windows(cfg.batch_size, cfg.window_k)
      for _ in range(cfg.batches_per_update)
    ]
    for windows in batches:
      state_loss, termination_loss, last_error = self._world_model_loss(
        self.world_model,
        self.iface,
        windows,
        context_all[windows["env_idx"]],
        cfg.termination_pos_weight,
      )
      loss = state_loss + termination_loss
      self.world_model_optimizer.zero_grad(set_to_none=True)
      loss.backward()
      nn.utils.clip_grad_norm_(self.world_model.parameters(), cfg.max_grad_norm)
      self.world_model_optimizer.step()
      sums += torch.stack((state_loss, termination_loss, last_error)).detach()
    means = sums / cfg.batches_per_update
    return {
      "world_model_state": means[0],
      "world_model_termination": means[1],
      "world_model_last_step_error": means[2],
    }

  # Policy.

  def state_heads(
    self, s: torch.Tensor, frame: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Invariant features (B, F) and goal (B, G) of states s (B, S) at frame (B,)."""
    return self.iface.invariant_features(s), self.iface.goal(s, frame)

  def imagine_step(
    self,
    frames: torch.Tensor,
    goal: torch.Tensor,
    s: torch.Tensor,
    features: torch.Tensor,
    context: torch.Tensor,
    env_idx: torch.Tensor,
    next_frame: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One imagined step without its observation noise (the caller adds it).

    Args:
      frames: (B, history, obs_dim) noisy observation frames, oldest first.
      goal: (B, G) goal of ``s``.
      s: (B, S) state; features: (B, F) its invariant features.
      context: (B, C) normalised sim context; env_idx: (B,) env of each row.
      next_frame: (B,) frame the next observation targets.

    Returns:
      action (B, A), s_next (B, S), noise-free observation of s_next (B, obs_dim),
      features (B, F) and goal (B, G) of s_next.
    """
    action = self.policy.forward_frames(frames, goal)
    s_next = self.world_model.step(s, features, action, context)
    obs = self.iface.observation(s_next, action, env_idx)
    next_features, next_goal = self.state_heads(s_next, next_frame)
    return action, s_next, obs, next_features, next_goal

  def policy_objective(
    self,
    states: list[torch.Tensor],
    actions: list[torch.Tensor],
    prev_actions: list[torch.Tensor],
    reward_frames: list[torch.Tensor],
    feature_list: list[torch.Tensor],
    goal_list: list[torch.Tensor],
    context: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    """Policy loss -mean J of an imagined rollout (Eq. 2), see :meth:`update_policy`.

    Lists hold step t at index t: states s_{t+1}, actions a_t, previous actions,
    reward frames k_t (H entries of (B, ...)); features and goals of s_0..s_H (H + 1).

    Returns:
      loss (), rewards (H, B), terminations (H, B), weighted rewards (H, B),
      survival c_1..c_H (H, B), features (H + 1, B, F), goals (H + 1, B, G).
    """
    cfg = self.policy_cfg
    horizon = len(states)
    num = states[0].shape[0]
    # Rewards and terminations do not feed back into the rollout: one batched call each
    # over (horizon, B), far fewer kernel launches than per step.
    reward = self.iface.reward(
      torch.cat(states),
      torch.cat(actions),
      torch.cat(prev_actions),
      torch.cat(reward_frames),
    ).view(horizon, num)
    all_features = torch.stack(feature_list)
    all_goals = torch.stack(goal_list)
    termination = torch.sigmoid(
      self.world_model.termination_head(all_features[1:], all_goals[1:])
    )  # (horizon, B): d_1..d_H
    survival_next = torch.cumprod(1.0 - termination, dim=0)  # c_1..c_H
    survival = torch.cat((torch.ones_like(survival_next[:1]), survival_next[:-1]))
    discount = cfg.gamma ** torch.arange(horizon, device=reward.device).unsqueeze(-1)
    weighted_reward = survival * reward
    terminal_value = self.critic(
      self.critic_input(feature_list[-1], goal_list[-1], context)
    )
    objective = (discount * weighted_reward).sum(dim=0)
    objective = objective + cfg.gamma**horizon * survival_next[-1] * terminal_value
    loss = -objective.mean()
    return (
      loss,
      reward,
      termination,
      weighted_reward,
      survival_next,
      all_features,
      all_goals,
    )

  def update_policy(
    self, buffer: ReplayBuffer
  ) -> tuple[dict[str, torch.Tensor], ImaginedRollout]:
    """One BPTT step of the policy on imagined rollouts from replay states (Eq. 2).

    J = sum_t gamma^t c_t r_{t+1} + gamma^H c_H V(s_H, g_H, xi),
    c_0 = 1,  c_{t+1} = c_t (1 - d_{t+1}).
    The world model, termination head and critic are frozen (requires_grad False) but
    gradients flow through their inputs. Gradients are clipped by global norm, then
    element-wise; a step with a non-finite norm is skipped and counted.
    """
    cfg = self.policy_cfg
    iface = self.iface
    seeds = buffer.sample_seeds(
      cfg.num_rollouts, cfg.history_length, cfg.horizon, iface.clip_end
    )
    env_idx = seeds["env_idx"]
    num = env_idx.shape[0]
    context = iface.context_normalized()[env_idx]
    with torch.no_grad():
      hist_states = seeds["state_hist"].flatten(0, 1)
      hist_actions = seeds["prev_action_hist"].flatten(0, 1)
      hist_env = env_idx.repeat_interleave(cfg.history_length)
      frames = iface.add_observation_noise(
        iface.observation(hist_states, hist_actions, hist_env)
      ).view(num, cfg.history_length, iface.obs_dim)

    frozen = (self.world_model, self.critic)
    for module in frozen:
      module.requires_grad_(False)

    s = seeds["s0"]
    frame = seeds["frame0"]
    a_prev = seeds["a_prev"]
    features, goal = self._state_heads(s, frame)
    # Index t of these lists is step t of the rollout (s_t -> s_{t+1}); features and
    # goals also hold the start state, so they have horizon + 1 entries.
    states, actions, prev_actions, reward_frames = [], [], [], []
    feature_list, goal_list = [features], [goal]
    for _ in range(cfg.horizon):
      action, s_next, obs, features, goal = self._imagine_step(
        frames, goal, s, features, context, env_idx, frame + 1
      )
      obs = iface.add_observation_noise(obs)
      frames = torch.cat((frames[:, 1:], obs.unsqueeze(1)), dim=1)
      states.append(s_next)
      actions.append(action)
      prev_actions.append(a_prev)
      reward_frames.append(frame)
      frame = frame + 1
      feature_list.append(features)
      goal_list.append(goal)
      s, a_prev = s_next, action

    (
      loss,
      reward,
      termination,
      weighted_reward,
      survival_next,
      all_features,
      all_goals,
    ) = self._policy_objective(
      states,
      actions,
      prev_actions,
      reward_frames,
      feature_list,
      goal_list,
      context,
    )

    self.policy_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = clip_policy_gradients(
      self.policy.parameters(), cfg.max_grad_norm, cfg.max_grad_value
    )
    if torch.isfinite(grad_norm):
      self.policy_optimizer.step()
    else:
      self.skipped_policy_steps += 1
    for module in frozen:
      module.requires_grad_(True)

    metrics = {
      "policy": loss.detach(),
      "policy_grad_norm": grad_norm.detach(),
      "imagined_reward_per_step": weighted_reward.detach().mean(),
      "imagined_survival": survival_next[-1].detach().mean(),
    }
    rollout = ImaginedRollout(
      features=all_features.detach(),
      goals=all_goals.detach(),
      rewards=reward.detach(),
      terminations=termination.detach(),
      context=context,
    )
    return metrics, rollout

  # Critic.

  def critic_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error () of V on normalised critic inputs x (M, D) vs (M,)."""
    return (self.critic.mlp(x).squeeze(-1) - targets).square().mean()

  def fit_critic(self, rollout: ImaginedRollout) -> dict[str, torch.Tensor]:
    """Fit V to lambda-return targets of the imagined rollout (Eq. 3).

    Targets are computed once with the pre-update critic. The fit runs ``epochs`` x
    ``minibatches`` Adam steps over all (H + 1) x B imagined states, the last state's
    target being its own pre-update value.
    """
    cfg = self.critic_cfg
    steps = rollout.features.shape[0]
    context = rollout.context.unsqueeze(0).expand(steps, -1, -1)
    x = self.critic_input(rollout.features, rollout.goals, context)
    x = x.flatten(0, 1)
    self.critic.normalizer.update(x)
    # The normaliser stays fixed during the fit: normalise every input once.
    x = self.critic.normalizer(x)
    with torch.no_grad():
      values = self.critic.mlp(x).view(steps, -1)
      targets = lambda_returns(
        rollout.rewards, rollout.terminations, values, self.policy_cfg.gamma, cfg.lam
      ).flatten()
    num = x.shape[0]
    minibatch = num // cfg.minibatches
    loss_sum = torch.zeros((), device=self.device)
    for _ in range(cfg.epochs):
      perm = torch.randperm(num, device=self.device)
      for i in range(cfg.minibatches):
        idx = perm[i * minibatch : (i + 1) * minibatch]
        loss = self._critic_loss(x[idx], targets[idx])
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()
        loss_sum += loss.detach()
    return {"critic": loss_sum / (cfg.epochs * cfg.minibatches)}

  # Checkpoints.

  def state_dict(self) -> dict:
    """Networks (with their normalisers and nominal scales) and optimiser states."""
    return {
      "policy_state_dict": self.policy.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "world_model_state_dict": self.world_model.state_dict(),
      "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
      "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
      "world_model_optimizer_state_dict": self.world_model_optimizer.state_dict(),
      "skipped_policy_steps": self.skipped_policy_steps,
    }

  def load_state_dict(self, state: dict, load_cfg: dict, strict: bool) -> None:
    """Load the parts selected by ``load_cfg`` (keys actor, critic, world_model,
    optimizer)."""
    if load_cfg.get("actor"):
      self.policy.load_state_dict(state["policy_state_dict"], strict=strict)
    if load_cfg.get("critic"):
      self.critic.load_state_dict(state["critic_state_dict"], strict=strict)
    if load_cfg.get("world_model"):
      self.world_model.load_state_dict(state["world_model_state_dict"], strict=strict)
    if load_cfg.get("optimizer"):
      self.policy_optimizer.load_state_dict(state["policy_optimizer_state_dict"])
      self.critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
      self.world_model_optimizer.load_state_dict(
        state["world_model_optimizer_state_dict"]
      )
      self.skipped_policy_steps = state["skipped_policy_steps"]
