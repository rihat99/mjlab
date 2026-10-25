"""SuperMimicRunner: collection loop, replay, episode returns for the reset sampler,
schedule, logging and checkpoints. Honours the runner contract of mjlab's scripts."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

import torch
from rsl_rl.utils import resolve_callable
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

from mjlab.rl.supermimic.algorithm import SuperMimic
from mjlab.rl.supermimic.config import (
  CollectionCfg,
  CriticCfg,
  PolicyCfg,
  PrecisionCfg,
  ScheduleCfg,
  WorldModelCfg,
)
from mjlab.rl.supermimic.interface import ModelInterface
from mjlab.rl.supermimic.replay_buffer import ReplayBuffer
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

_ALL_PARTS = {
  "actor": True,
  "critic": True,
  "world_model": True,
  "optimizer": True,
  "iteration": True,
  "sampler": True,
}


def sim_steps_at(iteration: int, steps_per_iteration: float) -> int:
  """Sim steps collected at ``iteration``; a fractional rate alternates (2.5: 2, 3)."""
  return math.floor((iteration + 1) * steps_per_iteration) - math.floor(
    iteration * steps_per_iteration
  )


def learning_rate_scale(iteration: int, cfg: ScheduleCfg) -> float:
  """Cosine decay from 1 to ``lr_final_ratio`` over ``lr_decay_iterations``, then
  constant."""
  if iteration >= cfg.lr_decay_iterations:
    return cfg.lr_final_ratio
  cosine = 0.5 * (1.0 + math.cos(math.pi * iteration / cfg.lr_decay_iterations))
  return cfg.lr_final_ratio + (1.0 - cfg.lr_final_ratio) * cosine


class SuperMimicRunner:
  """Model-based trainer: collect, fit the world model, BPTT policy update, critic fit.

  The task supplies a :class:`ModelInterface` named by ``train_cfg["model_interface"]``.
  The env must return a TensorDict with ``actor`` (N, history * obs_dim) and ``goal``
  (N, goal_dim); failures are read from ``env.unwrapped.termination_manager``.
  """

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
      raise NotImplementedError("SuperMimicRunner trains on a single GPU only.")
    if not train_cfg["model_interface"]:
      raise ValueError(
        "SuperMimicRunnerCfg.model_interface must name the task's ModelInterface "
        'class, e.g. "package.module:ClassName".'
      )
    self.env = env
    self.cfg = train_cfg
    self.device = device
    self.world_model_cfg = WorldModelCfg(**train_cfg["world_model"])
    self.policy_cfg = PolicyCfg(**train_cfg["policy"])
    self.critic_cfg = CriticCfg(**train_cfg["critic"])
    self.collection_cfg = CollectionCfg(**train_cfg["collection"])
    self.schedule_cfg = ScheduleCfg(**train_cfg["schedule"])
    precision_cfg = PrecisionCfg(**train_cfg["precision"])
    configure_torch_backends(allow_tf32=precision_cfg.tf32)

    interface_cls = resolve_callable(train_cfg["model_interface"])
    self.interface: ModelInterface = interface_cls(env.unwrapped)
    iface = self.interface
    obs = env.get_observations()
    history_dim = self.policy_cfg.history_length * iface.obs_dim
    if obs["actor"].shape[-1] != history_dim or obs["goal"].shape[-1] != iface.goal_dim:
      raise ValueError(
        f"Env observations actor {tuple(obs['actor'].shape)}, goal "
        f"{tuple(obs['goal'].shape)} do not match history_length "
        f"{self.policy_cfg.history_length} x obs_dim {iface.obs_dim} and goal_dim "
        f"{iface.goal_dim}."
      )
    if sum(iface.obs_term_dims) != iface.obs_dim:
      raise ValueError("ModelInterface.obs_term_dims must sum to obs_dim.")

    self.alg = SuperMimic(
      iface,
      self.world_model_cfg,
      self.policy_cfg,
      self.critic_cfg,
      device,
      compile=True,
      action_noise_std=self.collection_cfg.action_noise_std,
      autocast=precision_cfg.autocast,
    )

    # rsl_rl's Logger reads PPO-shaped keys; give it its own view of the cfg.
    writer_cfg = train_cfg["logger"]
    if writer_cfg == "local":
      writer_cfg = {"class_name": "mjlab.rl.local_log_writer:LocalLogWriter"}
    logger_cfg = {**train_cfg, "logger": writer_cfg, "algorithm": {}}
    self.logger = Logger(
      log_dir=log_dir,
      cfg=logger_cfg,
      env_cfg=env.cfg,
      num_envs=env.num_envs,
      is_distributed=False,
      gpu_world_size=1,
      gpu_global_rank=0,
      device=device,
    )
    self.current_learning_iteration = 0
    # Built on the first learn() so that play/evaluate do not allocate the replay.
    self.buffer: ReplayBuffer | None = None

  # Contract.

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run ``num_learning_iterations`` iterations (prefilling the replay first)."""
    col = self.collection_cfg
    if col.prefill_steps <= self.world_model_cfg.window_k:
      raise ValueError("collection.prefill_steps must exceed world_model.window_k.")
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )
    # get_inference_policy() may have left the policy in eval mode, which would stop
    # its normalisers from updating.
    self.alg.policy.train()
    self.logger.init_logging_writer()
    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    if self.buffer is None:
      self._init_collection()
      self._collect(col.prefill_steps, start_it)
    if not bool(self.alg.world_model.statistics_fitted):
      assert self.buffer is not None
      self.alg.fit_world_model_statistics(self.buffer)

    last_saved = -1
    for it in range(start_it, total_it):
      lr_scale = learning_rate_scale(it, self.schedule_cfg)
      self.alg.set_learning_rate_scale(lr_scale)
      t0 = self._clock()
      self._collect(sim_steps_at(it, col.sim_steps_per_iteration), it)
      self.interface.return_sampler.update_past_weights()
      t1 = self._clock()
      assert self.buffer is not None
      metrics = self.alg.fit_world_model(self.buffer)
      t2 = self._clock()
      t3 = t4 = t2
      if it >= col.warmup_world_model_iterations:
        policy_metrics, rollout = self.alg.update_policy(self.buffer)
        t3 = self._clock()
        metrics |= policy_metrics | self.alg.fit_critic(rollout)
        t4 = self._clock()
      # A crash beats training on: a non-finite update is never skipped or saved.
      if not bool(torch.isfinite(torch.stack(list(metrics.values()))).all()):
        values = {key: float(value) for key, value in metrics.items()}
        raise RuntimeError(f"Non-finite loss or gradient at iteration {it}: {values}")
      self._accumulate(metrics, (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
      self.current_learning_iteration = it + 1
      if (it + 1) % self.cfg["log_interval"] == 0 or it + 1 == total_it:
        self._log(it, start_it, total_it, lr_scale)
      if self.logger.writer is not None and (it + 1) % self.cfg["save_interval"] == 0:
        self.save(self._checkpoint_path())
        last_saved = it + 1

    if self.logger.writer is not None:
      if last_saved != self.current_learning_iteration:
        self.save(self._checkpoint_path())
      self.logger.stop_logging_writer()

  def save(self, path: str, infos: dict | None = None) -> None:
    """Save networks, normalisers, nominal scales, optimisers, the reset-sampler state
    and the iteration (number of completed iterations). The replay is not saved."""
    saved = self.alg.state_dict()
    saved["sampler_state_dict"] = self.interface.return_sampler.state_dict()
    saved["iter"] = self.current_learning_iteration
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    saved["infos"] = {**(infos or {}), "env_state": env_state}
    torch.save(saved, path)
    self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load a checkpoint; ``load_cfg`` selects parts (actor, critic, world_model,
    optimizer, iteration, sampler), None loads all."""
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    parts = _ALL_PARTS if load_cfg is None else load_cfg
    self.alg.load_state_dict(loaded, parts, strict)
    if parts.get("sampler"):
      self.interface.return_sampler.load_state_dict(loaded["sampler_state_dict"])
    if parts.get("iteration"):
      self.current_learning_iteration = loaded["iter"]
    infos = loaded["infos"]
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos

  def get_inference_policy(
    self, device: str | None = None
  ) -> Callable[[TensorDict], torch.Tensor]:
    """Deterministic ``obs -> policy(obs["actor"], obs["goal"])`` in eval mode."""
    policy = self.alg.policy
    policy.eval()
    if device is not None:
      policy.to(device)

    def act(obs: TensorDict) -> torch.Tensor:
      with torch.no_grad():
        return policy(obs["actor"], obs["goal"])

    return act

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    """Register a repository whose git state is stored with the run."""
    self.logger.git_status_repos.append(repo_file_path)

  # Collection.

  def _init_collection(self) -> None:
    num_envs = self.env.num_envs
    self.buffer = ReplayBuffer(
      self.collection_cfg.replay_steps,
      num_envs,
      self.interface.state_dim,
      self.interface.action_dim,
      self.device,
    )
    self.obs = self.env.get_observations().to(self.device)
    self.episode_step = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
    max_len = int(self.env.max_episode_length)
    self.episode_rewards = torch.zeros(num_envs, max_len, device=self.device)
    self.episode_frames = torch.zeros(
      num_envs, max_len, dtype=torch.long, device=self.device
    )
    self._reset_log_stats()

  @torch.no_grad()
  def _collect(self, num_steps: int, iteration: int) -> None:
    """Step all envs ``num_steps`` times with the noisy policy and store the steps."""
    assert self.buffer is not None
    iface = self.interface
    col = self.collection_cfg
    clip = self.cfg["clip_actions"]
    env_ids = torch.arange(self.env.num_envs, device=self.device)
    for _ in range(num_steps):
      state = iface.state_from_sim()
      frame = iface.frame_from_sim()
      if iteration < self.policy_cfg.normalizer_freeze_iteration:
        self.alg.policy.update_normalizers(self.obs["actor"], self.obs["goal"])
      action = self.alg.act(self.obs["actor"], self.obs["goal"])
      action = action + col.action_noise_std * torch.randn_like(action)
      if clip is not None:
        action = action.clamp(-clip, clip)
      obs, reward, dones, extras = self.env.step(action)
      self.obs = obs.to(self.device)
      done = dones.bool()
      # A failure on the same step as a time-out counts as terminated.
      terminated = self.env.unwrapped.termination_manager.terminated.clone()
      truncated = done & ~terminated
      terminal = iface.terminal_state_snapshot()
      if terminal is None:
        if done.any():
          raise RuntimeError("Envs were reset but terminal_state_snapshot() is None.")
        terminal = state
      self.buffer.add(
        state=state,
        action=action,
        frame=frame,
        done=done,
        terminated=terminated,
        terminal_state=terminal,
        episode_step=self.episode_step,
      )
      self.logger.process_env_step(reward, dones, extras)
      step = self.episode_step.long()
      self.episode_rewards[env_ids, step] = reward
      self.episode_frames[env_ids, step] = frame
      done_ids = torch.nonzero(done).squeeze(-1)
      if done_ids.numel() > 0:
        self._finish_episodes(done_ids, truncated[done_ids], terminal[done_ids])
      self.episode_step = torch.where(done, 0, self.episode_step + 1)
      self._log_stats["env_steps"] += 1

  def _finish_episodes(
    self, env_ids: torch.Tensor, truncated: torch.Tensor, terminal: torch.Tensor
  ) -> None:
    """Credit the discounted returns of finished episodes to the reset sampler.

    The return-to-go G_t is computed backwards from the episode end, bootstrapped with
    V(s_T) for truncated episodes (0 for terminated ones, and for all episodes before
    the critic's first fit); ``ReturnSampler.credit_episodes`` gives every bin the
    episode entered G at its first step in that bin.
    """
    iface = self.interface
    gamma = self.policy_cfg.gamma
    lengths = self.episode_step[env_ids].long() + 1  # (M,)
    max_len = int(lengths.max())
    steps = torch.arange(max_len, device=self.device)
    mask = steps.unsqueeze(0) < lengths.unsqueeze(1)  # (M, L)
    rewards = self.episode_rewards[env_ids, :max_len].double() * mask
    frames = self.episode_frames[env_ids, :max_len]
    rows = torch.arange(env_ids.numel(), device=self.device)
    last_frame = frames[rows, lengths - 1]
    next_frame = torch.minimum(last_frame + 1, iface.clip_end(last_frame) - 1)
    if self.alg.critic_fitted:
      value = self.alg.value(terminal, next_frame, env_ids)
    else:
      value = torch.zeros(env_ids.numel(), device=self.device)
    bootstrap = torch.where(truncated, value, 0.0).double()
    # Return-to-go by a reverse cumulative sum of discounted rewards, in float64 because
    # it divides by gamma^t (down to ~0.007 for 500 steps).
    discount = gamma ** steps.double()
    tail = torch.flip(torch.cumsum(torch.flip(rewards * discount, [1]), 1), [1])
    returns = tail / discount
    returns += (gamma ** (lengths.unsqueeze(1) - steps).double()) * bootstrap[:, None]
    iface.return_sampler.credit_episodes(frames, returns, mask)

    stats = self._log_stats
    stats["episode_reward_sum"] += rewards.sum()
    stats["episode_length_sum"] += lengths.sum()
    stats["episodes"] += env_ids.numel()
    stats["terminated"] += (~truncated).sum()

  # Logging.

  def _clock(self) -> float:
    if torch.device(self.device).type == "cuda":
      torch.cuda.synchronize(self.device)
    return time.perf_counter()

  def _reset_log_stats(self) -> None:
    zero = torch.zeros((), device=self.device)
    self._log_stats: dict = {
      "env_steps": 0,
      "iterations": 0,
      "times": [0.0, 0.0, 0.0, 0.0],
      "metric_sums": {},
      "metric_counts": {},
      "episode_reward_sum": zero.double(),
      "episode_length_sum": zero.long(),
      "episodes": 0,
      "terminated": zero.long(),
    }

  def _accumulate(
    self, metrics: dict[str, torch.Tensor], times: tuple[float, ...]
  ) -> None:
    stats = self._log_stats
    stats["iterations"] += 1
    stats["times"] = [a + b for a, b in zip(stats["times"], times, strict=True)]
    for key, value in metrics.items():
      stats["metric_sums"][key] = stats["metric_sums"].get(key, 0.0) + value
      stats["metric_counts"][key] = stats["metric_counts"].get(key, 0) + 1

  def _log(self, it: int, start_it: int, total_it: int, lr_scale: float) -> None:
    stats = self._log_stats
    writer = self.logger.writer
    if writer is None:
      self._reset_log_stats()
      return
    means = {
      key: float(total) / stats["metric_counts"][key]
      for key, total in stats["metric_sums"].items()
    }
    loss_keys = ("world_model_state", "policy", "critic")
    loss_dict = {key: means[key] for key in loss_keys if key in means}
    max_step_reward = self.interface.max_step_reward
    scalars = {
      "Metrics/world_model_last_step_error": means.get("world_model_last_step_error"),
      "Metrics/imagined_reward_normalized": (
        means["imagined_reward_per_step"] / max_step_reward
        if "imagined_reward_per_step" in means
        else None
      ),
      "Metrics/imagined_survival": means.get("imagined_survival"),
      "Metrics/policy_grad_norm": means.get("policy_grad_norm"),
      "Metrics/world_model_grad_norm": means.get("world_model_grad_norm"),
    }
    if stats["episodes"] > 0:
      length_sum = float(stats["episode_length_sum"])
      scalars["Train/normalized_return"] = float(stats["episode_reward_sum"]) / (
        max_step_reward * length_sum
      )
      scalars["Train/termination_rate"] = float(stats["terminated"]) / stats["episodes"]
    for key, value in self.interface.return_sampler.log().items():
      scalars[f"Metrics/sampler_{key}"] = value
    names = ("sim", "world_model", "policy", "critic")
    for name, total in zip(names, stats["times"], strict=True):
      scalars[f"Perf/{name}_time"] = total / stats["iterations"]
    for tag, value in scalars.items():
      if value is not None:
        writer.add_scalar(tag, value, it)

    collect_time = stats["times"][0]
    learn_time = sum(stats["times"][1:])
    # The Logger counts env steps as num_steps_per_env x num_envs per call.
    self.logger.cfg["num_steps_per_env"] = stats["env_steps"]
    self.logger.log(
      it=it,
      start_it=start_it,
      total_it=total_it,
      collect_time=collect_time,
      learn_time=learn_time,
      loss_dict=loss_dict,
      learning_rate=self.policy_cfg.lr * lr_scale,
      action_std=torch.tensor(self.collection_cfg.action_noise_std),
      rnd_weight=None,
    )
    self._reset_log_stats()

  def _checkpoint_path(self) -> str:
    assert self.logger.log_dir is not None
    return os.path.join(
      self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
    )
