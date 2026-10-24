"""Tests for the task-agnostic SuperMimic package (mjlab.rl.supermimic).

A synthetic system stands in for the tracking task: random linear dynamics of the
body-frame root velocities and joints of a 29-joint floating base (71-d state), a toy
reference library of two clips, a toy reward and a height-error termination. The fake
env mimics RslRlVecEnvWrapper (term-major observation history, auto-reset, time-outs in
extras) and the synthetic interface implements the ModelInterface protocol on it.
"""

import math
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from conftest import get_test_device
from tensordict import TensorDict
from torch import nn

from mjlab.rl.supermimic.algorithm import (
  SuperMimic,
  clip_policy_gradients,
  lambda_returns,
)
from mjlab.rl.supermimic.config import (
  CollectionCfg,
  CriticCfg,
  PolicyCfg,
  SuperMimicRunnerCfg,
  WorldModelCfg,
)
from mjlab.rl.supermimic.interface import FloatingBaseState
from mjlab.rl.supermimic.replay_buffer import ReplayBuffer
from mjlab.rl.supermimic.runner import SuperMimicRunner, sim_steps_at
from mjlab.rl.supermimic.world_model import world_model_loss
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply

NQ = 29
STATE_DIM = 13 + 2 * NQ
OBS_TERM_DIMS = (NQ, NQ, 3, 3, NQ)
OBS_DIM = sum(OBS_TERM_DIMS)
HISTORY = 10
FUTURE_FRAMES = 10
GOAL_DIM = FUTURE_FRAMES * (3 + 6 + 2 * NQ)
CONTEXT_DIM = 33
DT = 0.02
CLIP_LENGTHS = (130, 170)
BIN_SIZE = 50
MAX_EPISODE_LENGTH = 40
HEIGHT_TOLERANCE = 0.05
NOISE_WIDTHS = (0.01, 0.5, 0.2, 0.05, 0.0)


@pytest.fixture(scope="module")
def device() -> str:
  return get_test_device()


def _term_major(frames: torch.Tensor) -> torch.Tensor:
  """(N, history, obs) -> (N, history * obs) flattened like an mjlab history group."""
  terms = torch.split(frames, list(OBS_TERM_DIMS), dim=-1)
  return torch.cat([term.flatten(start_dim=1) for term in terms], dim=-1)


class _SyntheticInterface:
  """ModelInterface of the synthetic env (pure torch, differentiable heads)."""

  state_dim = STATE_DIM
  context_dim = CONTEXT_DIM
  obs_dim = OBS_DIM
  obs_term_dims: tuple[int, ...] = OBS_TERM_DIMS
  goal_dim = GOAL_DIM
  action_dim = NQ
  feature_dim = 10 + 2 * NQ
  nq = NQ
  max_step_reward = 2 * DT

  def __init__(self, env: "_SyntheticEnv") -> None:
    self.env = env
    self.layout = FloatingBaseState(NQ)
    widths = [
      torch.full((n,), w) for n, w in zip(OBS_TERM_DIMS, NOISE_WIDTHS, strict=True)
    ]
    self.noise_widths = torch.cat(widths).to(env.device)

  def state_from_sim(self, env_ids=None):
    return self.env.state if env_ids is None else self.env.state[env_ids]

  def terminal_state_snapshot(self):
    return self.env.terminal_state

  def frame_from_sim(self):
    return self.env.frame

  def context_normalized(self):
    return self.env.context

  def invariant_features(self, s):
    return self.layout.invariant_features(s)

  def observation(self, s, a_prev, env_idx, frame):
    del frame  # No term compares the state with the reference.
    _, quat, _, ang_vel, q, qd = self.layout.split(s)
    gravity = torch.zeros_like(ang_vel)
    gravity[:, 2] = -1.0
    return torch.cat(
      (
        q + self.env.encoder_bias[env_idx],
        qd,
        self.layout.to_root_frame(quat, ang_vel),
        self.layout.to_root_frame(quat, gravity),
        a_prev,
      ),
      dim=-1,
    )

  def add_observation_noise(self, o):
    return o + (2.0 * torch.rand_like(o) - 1.0) * self.noise_widths

  def goal(self, s, frame):
    env = self.env
    offsets = torch.arange(FUTURE_FRAMES, device=frame.device)
    future = torch.minimum(
      frame.unsqueeze(-1) + offsets, self.clip_end(frame).unsqueeze(-1) - 1
    )
    pos = s[:, None, :3]
    per_frame = torch.cat(
      (
        env.ref_pos[future] - pos,
        env.ref_ori[future],
        env.ref_q[future],
        env.ref_qd[future],
      ),
      dim=-1,
    )
    return per_frame.flatten(start_dim=1)

  def reward_terms(self, s_next, a, a_prev, frame):
    pos, _, _, _, q, _ = self.layout.split(s_next)
    env = self.env
    joint_error = (q - env.ref_q[frame]).square().mean(dim=-1)
    height_error = (pos[:, 2] - env.ref_pos[frame, 2]).square()
    return {
      "joint": DT * torch.exp(-joint_error / 0.25),
      "height": DT * torch.exp(-height_error / 0.01),
      "action_rate": -0.1 * DT * (a - a_prev).square().sum(dim=-1),
    }

  def reward(
    self,
    s_next: torch.Tensor,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    frame: torch.Tensor,
  ) -> torch.Tensor:
    terms = self.reward_terms(s_next, a, a_prev, frame)
    return terms["joint"] + terms["height"] + terms["action_rate"]

  def termination_hard(self, s_next, frame):
    height_error = s_next[:, 2] - self.env.ref_pos[frame, 2]
    return height_error.abs() > HEIGHT_TOLERANCE

  def termination_soft(self, s_next, frame):
    height_error = s_next[:, 2] - self.env.ref_pos[frame, 2]
    return torch.sigmoid((height_error.abs() - HEIGHT_TOLERANCE) / 0.01)

  def clip_end(self, frame):
    return self.env.clip_end_of_frame[frame]

  def bin_of(self, frame):
    return self.env.bin_of_frame[frame]

  def update_bin_returns(self, bins, returns):
    env = self.env
    total = torch.zeros_like(env.bin_returns).index_add_(0, bins, returns)
    count = torch.zeros_like(env.bin_returns).index_add_(
      0, bins, torch.ones_like(returns)
    )
    seen = count > 0
    env.bin_returns[seen] = (
      0.9 * env.bin_returns[seen] + 0.1 * total[seen] / count[seen]
    )
    env.bin_update_calls += 1

  def update_past_weights(self):
    self.env.past_weight_updates += 1

  def sampler_state_dict(self):
    return {"bin_returns": self.env.bin_returns.clone()}

  def load_sampler_state_dict(self, state):
    self.env.bin_returns.copy_(state["bin_returns"])

  def sampler_log(self):
    return {"mean_bin_return": float(self.env.bin_returns.mean())}


class _SyntheticEnv:
  """Fake RslRlVecEnvWrapper (and its own ``unwrapped``) over the synthetic system."""

  def __init__(self, num_envs: int, device: str, seed: int = 0) -> None:
    gen = torch.Generator().manual_seed(seed)
    self.num_envs = num_envs
    self.device = torch.device(device)
    self.unwrapped = self
    self.termination_manager = SimpleNamespace(
      terminated=torch.zeros(num_envs, dtype=torch.bool, device=device)
    )
    self.cfg: dict = {}
    self.max_episode_length = MAX_EPISODE_LENGTH
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.common_step_counter = 0
    self.layout = FloatingBaseState(NQ)

    # Reference library: clips back to back, 1 s bins per clip.
    clip_end, bin_of, first_bin, start = [], [], 0, 0
    for length in CLIP_LENGTHS:
      clip_end += [start + length] * length
      bin_of += [first_bin + i // BIN_SIZE for i in range(length)]
      first_bin += math.ceil(length / BIN_SIZE)
      start += length
    self.clip_end_of_frame = torch.tensor(clip_end, device=device)
    self.bin_of_frame = torch.tensor(bin_of, device=device)
    self.bin_returns = torch.zeros(first_bin, device=device)
    self.bin_update_calls = 0
    self.past_weight_updates = 0
    t = torch.arange(start, dtype=torch.float32).unsqueeze(-1) * DT
    phase = torch.rand(1, NQ, generator=gen) * 6.28
    self.ref_q = (0.3 * torch.sin(2.0 * t + phase)).to(device)
    self.ref_qd = (0.6 * torch.cos(2.0 * t + phase)).to(device)
    height = 0.8 + 0.03 * torch.sin(t)
    self.ref_pos = torch.cat((0.1 * t, torch.zeros_like(t), height), -1).to(device)
    self.ref_ori = (torch.rand(start, 6, generator=gen) * 2 - 1).to(device)

    # u' = A u + B a + C xi on u = [lin vel root frame, ang vel root frame, q, qd].
    dim_u = 6 + 2 * NQ
    noise = torch.randn(dim_u, dim_u, generator=gen) / math.sqrt(dim_u)
    self.A = (0.85 * torch.eye(dim_u) + 0.05 * noise).to(device)
    self.B = (0.1 * torch.randn(dim_u, NQ, generator=gen) / math.sqrt(NQ)).to(device)
    c_scale = 0.02 / math.sqrt(CONTEXT_DIM)
    self.C = (c_scale * torch.randn(dim_u, CONTEXT_DIM, generator=gen)).to(device)
    self.context = (torch.rand(num_envs, CONTEXT_DIM, generator=gen) * 2 - 1).to(device)
    self.encoder_bias = (0.02 * torch.rand(num_envs, NQ, generator=gen) - 0.01).to(
      device
    )

    self.iface = _SyntheticInterface(self)
    self.state = torch.zeros(num_envs, STATE_DIM, device=device)
    self.frame = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.a_prev = torch.zeros(num_envs, NQ, device=device)
    self.history = torch.zeros(num_envs, HISTORY, OBS_DIM, device=device)
    self.terminal_state: torch.Tensor | None = None
    self._reset(torch.arange(num_envs, device=device))

  def _reset(self, env_ids: torch.Tensor) -> None:
    num = env_ids.numel()
    frame = torch.randint(0, self.clip_end_of_frame.numel(), (num,), device=self.device)
    last = frame == self.clip_end_of_frame[frame] - 1
    frame = torch.where(last, frame - 1, frame)
    identity = torch.zeros(num, STATE_DIM, device=self.device)
    identity[:, 3] = 1.0
    delta = torch.zeros(num, self.layout.delta_dim, device=self.device)
    delta[:, 2] = self.ref_pos[frame, 2]
    delta[:, 3:6] = 0.05 * torch.randn(num, 3, device=self.device)
    delta[:, 6:12] = 0.05 * torch.randn(num, 6, device=self.device)
    delta[:, 12 : 12 + NQ] = self.ref_q[frame] + 0.05 * torch.randn(
      num, NQ, device=self.device
    )
    delta[:, 12 + NQ :] = self.ref_qd[frame]
    self.state[env_ids] = self.layout.integrate(identity, delta)
    self.frame[env_ids] = frame
    self.a_prev[env_ids] = 0.0
    self.episode_length_buf[env_ids] = 0
    obs = self._noisy_obs(env_ids)
    self.history[env_ids] = obs.unsqueeze(1).expand(-1, HISTORY, -1)

  def _noisy_obs(self, env_ids: torch.Tensor) -> torch.Tensor:
    obs = self.iface.observation(
      self.state[env_ids], self.a_prev[env_ids], env_ids, self.frame[env_ids]
    )
    return self.iface.add_observation_noise(obs)

  def _dynamics(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    _, quat, lin_vel, ang_vel, q, qd = self.layout.split(s)
    lin_b = self.layout.to_root_frame(quat, lin_vel)
    ang_b = self.layout.to_root_frame(quat, ang_vel)
    u = torch.cat((lin_b, ang_b, q, qd), dim=-1)
    u = u @ self.A.T + action @ self.B.T + self.context @ self.C.T
    lin_b, ang_b, q1, qd1 = torch.split(u, [3, 3, NQ, NQ], dim=-1)
    zeros = torch.zeros_like(lin_b)
    delta = torch.cat((DT * lin_b, DT * ang_b, zeros, zeros, q1 - q, qd1 - qd), -1)
    s1 = self.layout.integrate(s, delta)
    quat1 = s1[:, 3:7]
    s1[:, 7:10] = quat_apply(quat1, lin_b)
    s1[:, 10:13] = quat_apply(quat1, ang_b)
    return s1

  def get_observations(self) -> TensorDict:
    goal = self.iface.goal(self.state, self.frame)
    obs = {"actor": _term_major(self.history), "goal": goal}
    return TensorDict(obs, batch_size=[self.num_envs])

  def step(self, actions: torch.Tensor):
    frame = self.frame
    s1 = self._dynamics(self.state, actions)
    reward = self.iface.reward(s1, actions, self.a_prev, frame)
    self.episode_length_buf += 1
    self.common_step_counter += 1
    terminated = self.iface.termination_hard(s1, frame)
    motion_end = frame >= self.clip_end_of_frame[frame] - 1
    truncated = (self.episode_length_buf >= self.max_episode_length) | motion_end
    done = terminated | truncated
    self.termination_manager.terminated = terminated
    self.terminal_state = s1.clone()
    self.state = s1
    self.a_prev = actions.clone()
    self.frame = frame + 1
    keep = torch.nonzero(~done).squeeze(-1)
    new_obs = self._noisy_obs(keep)
    self.history[keep] = torch.cat((self.history[keep, 1:], new_obs.unsqueeze(1)), 1)
    self._reset(torch.nonzero(done).squeeze(-1))
    extras = {
      "time_outs": truncated,
      "log": {"Episode_Termination/height": terminated.float().mean()},
    }
    return self.get_observations(), reward, done.long(), extras


def _small_cfgs() -> tuple[WorldModelCfg, PolicyCfg, CriticCfg]:
  world_model = WorldModelCfg(
    hidden_dims=(256, 256), window_k=4, batch_size=256, batches_per_update=1, lr=1e-3
  )
  policy = PolicyCfg(
    encoder_hidden=(64,), decoder_hidden=(64,), num_rollouts=64, horizon=8
  )
  critic = CriticCfg(hidden=(32,), epochs=2, minibatches=2)
  return world_model, policy, critic


def _fill_buffer(
  env: _SyntheticEnv, steps: int, action_std: float = 0.3
) -> ReplayBuffer:
  """Collect ``steps`` random-action steps into a buffer, as the runner stores them."""
  buffer = ReplayBuffer(steps + 1, env.num_envs, STATE_DIM, NQ, env.device)
  episode_step = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
  for _ in range(steps):
    state, frame = env.state.clone(), env.frame.clone()
    action = action_std * torch.randn(env.num_envs, NQ, device=env.device)
    _, reward, dones, _ = env.step(action)
    done = dones.bool()
    assert env.terminal_state is not None
    buffer.add(
      state=state,
      action=action,
      frame=frame,
      reward=reward,
      terminated=env.termination_manager.terminated,
      done=done,
      terminal_state=env.terminal_state,
      episode_step=episode_step,
    )
    episode_step = torch.where(done, 0, episode_step + 1)
  return buffer


def test_floating_base_delta_integrate_round_trip(device):
  # float64: the check is that delta and integrate are exact inverses; in float32 the
  # round trip of states with velocities of a few m/s differs by a few 1e-6 (roundoff).
  torch.manual_seed(0)
  layout = FloatingBaseState(NQ)
  num = 512
  f64 = torch.float64
  s = torch.randn(num, STATE_DIM, dtype=f64, device=device)
  s[:, 3:7] = s[:, 3:7] / s[:, 3:7].norm(dim=-1, keepdim=True)
  delta = 0.3 * torch.randn(num, layout.delta_dim, dtype=f64, device=device)
  delta[:, 3:6] = 2.0 * (torch.rand(num, 3, dtype=f64, device=device) - 0.5)
  s_next = layout.integrate(s, delta)
  torch.testing.assert_close(layout.delta(s, s_next), delta, atol=1e-6, rtol=0)

  other = torch.randn(num, STATE_DIM, dtype=f64, device=device)
  other[:, 3:7] = other[:, 3:7] / other[:, 3:7].norm(dim=-1, keepdim=True)
  back = layout.integrate(s, layout.delta(s, other))
  quat_sign = torch.sign((back[:, 3:7] * other[:, 3:7]).sum(-1, keepdim=True))
  back[:, 3:7] = back[:, 3:7] * quat_sign
  torch.testing.assert_close(back, other, atol=1e-6, rtol=0)

  # The dynamics output starts at zero: gradients through a zero rotation stay finite.
  zero = torch.zeros(
    num, layout.delta_dim, dtype=f64, device=device, requires_grad=True
  )
  layout.delta(s, layout.integrate(s, zero)).sum().backward()
  assert zero.grad is not None and torch.isfinite(zero.grad).all()


def test_replay_buffer_windows_and_seeds_around_dones_and_wrap():
  # 9 steps into a 6-row ring: ages 0..5 hold global steps 3..8. Env 0 ends an episode
  # at step 4 (terminal state 100 + step); env 1 runs one long episode. State = (step,
  # env), action = step, so every sampled row can be traced back.
  buffer = ReplayBuffer(6, 2, state_dim=2, action_dim=1, device="cpu")
  episode_step = torch.zeros(2, dtype=torch.int32)
  for t in range(9):
    done = torch.tensor([t == 4, False])
    buffer.add(
      state=torch.tensor([[t, 0.0], [t, 1.0]]),
      action=torch.full((2, 1), float(t)),
      frame=torch.full((2,), t),
      reward=torch.zeros(2),
      terminated=done,
      done=done,
      terminal_state=torch.tensor([[100.0 + t, 0.0], [100.0 + t, 1.0]]),
      episode_step=episode_step,
    )
    episode_step = torch.where(done, 0, episode_step + 1)
  assert buffer.size == 6 and buffer.ptr == 3

  def next_state(t: int, env: int) -> list[float]:
    return [100.0 + t, env] if (env == 0 and t == 4) else [t + 1.0, env]

  windows = buffer.sample_windows(2000, k=2)
  starts = set()
  for i in range(2000):
    t, env = int(windows["s0"][i, 0]), int(windows["env_idx"][i])
    starts.add((env, t))
    for m in range(2):
      assert windows["targets"][i, m].tolist() == next_state(t + m, env)
      assert windows["frames"][i, m] == t + m
    assert bool(windows["terminated"][i, 1]) == (env == 0 and t + 1 == 4)
  assert windows["valid"].all()
  # Starts need rows t, t+1, t+2 stored and no done at row t.
  assert starts == {(0, 3), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6)}

  seeds = buffer.sample_seeds(2000, history=3, horizon=2, clip_end_fn=lambda f: f + 100)
  starts = set()
  for i in range(2000):
    t, env = int(seeds["s0"][i, 0]), int(seeds["env_idx"][i])
    starts.add((env, t))
    ep = t if env == 1 else t - 5
    hist = [max(t - back, t - ep) for back in (2, 1, 0)]
    assert seeds["state_hist"][i, :, 0].tolist() == hist
    prev = [float(h - 1) if h > t - ep else 0.0 for h in hist]
    assert seeds["prev_action_hist"][i, :, 0].tolist() == prev
    assert seeds["a_prev"][i, 0] == prev[-1]
  # Env 0 step 3 lacks its history (episode start overwritten), step 4 is a done; env 1
  # steps 3..5 lack the row before their 3-row history.
  assert starts == {(0, 5), (0, 6), (0, 7), (0, 8), (1, 6), (1, 7), (1, 8)}

  # Rollouts must end strictly inside the clip: frame + horizon < clip_end.
  seeds = buffer.sample_seeds(50, 3, 2, clip_end_fn=lambda f: torch.full_like(f, 8))
  assert set(seeds["s0"][:, 0].tolist()) == {5.0}


def test_world_model_k_step_loss_decreases(device):
  torch.manual_seed(0)
  env = _SyntheticEnv(64, device)
  wm_cfg, policy_cfg, critic_cfg = _small_cfgs()
  alg = SuperMimic(env.iface, wm_cfg, policy_cfg, critic_cfg, device)
  buffer = _fill_buffer(env, 60)
  alg.fit_world_model_statistics(buffer)
  windows = buffer.sample_windows(1024, wm_cfg.window_k)
  context = env.context[windows["env_idx"]]

  def state_loss() -> float:
    with torch.no_grad():
      loss, _, _ = world_model_loss(alg.world_model, env.iface, windows, context, 1.0)
    return float(loss)

  initial = state_loss()
  for _ in range(300):
    alg.fit_world_model(buffer)
  final = state_loss()
  assert final < 0.1 * initial, (initial, final)


def test_running_world_model_statistics(device):
  torch.manual_seed(0)
  env = _SyntheticEnv(64, device)
  wm_cfg, policy_cfg, critic_cfg = _small_cfgs()
  prefill = _fill_buffer(env, 40)
  wider = _fill_buffer(env, 40, action_std=3.0)
  actions = slice(env.iface.feature_dim, env.iface.feature_dim + NQ)
  model = None
  for running in (False, True):
    cfg = replace(wm_cfg, running_statistics=running)
    alg = SuperMimic(env.iface, cfg, policy_cfg, critic_cfg, device)
    alg.fit_world_model_statistics(prefill)
    model = alg.world_model
    before = {name: b.clone() for name, b in model.named_buffers()}
    storage = {name: b.data_ptr() for name, b in model.named_buffers()}
    alg.fit_world_model(wider)
    after = dict(model.named_buffers())
    # Buffers keep their storage: CUDA graphs read them by address.
    assert storage == {name: b.data_ptr() for name, b in after.items()}
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    if not running:
      assert not changed
      continue
    for name in ("input_normalizer", "termination_head.normalizer"):
      assert {f"{name}._mean", f"{name}._var", f"{name}._std"} <= changed
    assert "nominal_scale" in changed
    std_before = before["input_normalizer._std"][0, actions]
    assert (after["input_normalizer._std"][0, actions] > 2.0 * std_before).all()
    assert (model.nominal_scale >= model.nominal_floor).all()

  # Moving average: the count is capped at max_count before each update (weight 1/2
  # here), so increments of zero shrink the scale geometrically down to the floor.
  assert model is not None  # the running-statistics model of the last iteration
  num = 256
  inputs = torch.randn(num, model.input_normalizer.get_buffer("_mean").shape[-1])
  heads = torch.randn(
    num, model.termination_head.normalizer.get_buffer("_mean").shape[-1]
  )
  zeros = torch.zeros(num, model.state.delta_dim)
  for _ in range(60):
    model.update_statistics(
      inputs.to(device), zeros.to(device), heads.to(device), max_count=num
    )
  assert int(model.increment_statistics.get_buffer("count")) == 2 * num
  torch.testing.assert_close(model.nominal_scale, model.nominal_floor, rtol=0, atol=0)


def test_clip_policy_gradients_norm_then_value():
  # Norm first: (0.3, 0.4) -> (0.06, 0.08) -> clamp 0.08 leaves it. Value first would
  # give (0.08, 0.08) -> renormalised (0.0707, 0.0707).
  param = nn.Parameter(torch.zeros(2))
  param.grad = torch.tensor([0.3, 0.4])
  norm = clip_policy_gradients([param], max_norm=0.1, max_value=0.08)
  assert float(norm) == pytest.approx(0.5)
  torch.testing.assert_close(param.grad, torch.tensor([0.06, 0.08]))
  param.grad = torch.tensor([0.5, 0.0])
  clip_policy_gradients([param], max_norm=1.0, max_value=0.1)
  torch.testing.assert_close(param.grad, torch.tensor([0.1, 0.0]))


def test_policy_bptt_update(device):
  torch.manual_seed(0)
  env = _SyntheticEnv(64, device)
  wm_cfg, policy_cfg, critic_cfg = _small_cfgs()
  alg = SuperMimic(env.iface, wm_cfg, policy_cfg, critic_cfg, device)
  buffer = _fill_buffer(env, 40)
  alg.fit_world_model_statistics(buffer)
  alg.fit_world_model(buffer)
  before = [p.detach().clone() for p in alg.policy.parameters()]
  frozen_grads = {
    module: [None if p.grad is None else p.grad.clone() for p in module.parameters()]
    for module in (alg.world_model, alg.critic)
  }

  metrics, rollout, finite = alg.update_policy(buffer)

  grads = [p.grad for p in alg.policy.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
  # The applied gradient went through both clips.
  assert max(float(g.abs().max()) for g in grads) <= policy_cfg.max_grad_value + 1e-7
  total_norm = torch.cat([g.flatten() for g in grads]).norm()
  assert float(total_norm) <= policy_cfg.max_grad_norm + 1e-5
  assert torch.isfinite(metrics["policy_grad_norm"]) and metrics["policy_grad_norm"] > 0
  assert finite
  assert alg.skipped_policy_steps == 0
  assert any(
    not torch.equal(b, p) for b, p in zip(before, alg.policy.parameters(), strict=True)
  )
  # Frozen modules got no gradient and are trainable again afterwards.
  for module, grads_before in frozen_grads.items():
    for p, g in zip(module.parameters(), grads_before, strict=True):
      if g is None:
        assert p.grad is None
      else:
        assert p.grad is not None and torch.equal(p.grad, g)
      assert p.requires_grad
  horizon, num = policy_cfg.horizon, policy_cfg.num_rollouts
  assert rollout.features.shape == (horizon + 1, num, 10 + 2 * NQ)
  assert rollout.goals.shape == (horizon + 1, num, GOAL_DIM)
  assert rollout.rewards.shape == rollout.terminations.shape == (horizon, num)
  assert "critic" in alg.fit_critic(rollout)


@pytest.mark.parametrize(
  "flags", [False, True], ids=["frozen_clean", "running_noisy_analytic"]
)
def test_compiled_iteration_matches_eager(device, flags):
  if device == "cpu":
    pytest.skip("the compiled path (torch.compile + CUDA graphs) targets CUDA")
  env = _SyntheticEnv(64, device)
  wm_cfg, policy_cfg, critic_cfg = _small_cfgs()
  # flags: running world-model statistics, the action noise and the analytic
  # termination in imagination.
  wm_cfg = replace(wm_cfg, running_statistics=flags)
  policy_cfg = replace(
    policy_cfg, imagine_action_noise=flags, analytic_termination=flags
  )
  buffer = _fill_buffer(env, 40)
  losses = {}
  for compile in (False, True):
    torch.manual_seed(1)  # Same initial weights and the same sampled batches.
    alg = SuperMimic(
      env.iface, wm_cfg, policy_cfg, critic_cfg, device, compile, action_noise_std=0.1
    )
    alg.fit_world_model_statistics(buffer)
    # Two iterations: the second one's graphs must read the statistics updated in
    # place by the first.
    for _ in range(2):
      metrics = alg.fit_world_model(buffer)
      policy_metrics, rollout, _ = alg.update_policy(buffer)
      losses[compile] = metrics | policy_metrics | alg.fit_critic(rollout)
  for key, eager in losses[False].items():
    torch.testing.assert_close(losses[True][key], eager, rtol=1e-4, atol=1e-8)


@pytest.mark.parametrize("analytic", [False, True], ids=["learned_head", "analytic"])
def test_compiled_policy_gradient_matches_eager(device, analytic):
  # The losses can agree while the gradients do not (a compiled objective once gave
  # the right loss with a gradient at cosine ~0.5 to the eager one).
  if device == "cpu":
    pytest.skip("the compiled path (torch.compile + CUDA graphs) targets CUDA")
  env = _SyntheticEnv(64, device)
  wm_cfg, policy_cfg, critic_cfg = _small_cfgs()
  policy_cfg = replace(policy_cfg, analytic_termination=analytic)
  buffer = _fill_buffer(env, 40)
  grads = []
  for compile in (False, True):
    torch.manual_seed(1)  # Same weights, batches, seeds and noise.
    alg = SuperMimic(
      env.iface, wm_cfg, policy_cfg, critic_cfg, device, compile, action_noise_std=0.1
    )
    alg.fit_world_model_statistics(buffer)
    alg.fit_world_model(buffer)
    alg.update_policy(buffer)
    params = alg.policy.parameters()
    grads.append(torch.cat([cast(torch.Tensor, p.grad).flatten() for p in params]))
  cosine = torch.nn.functional.cosine_similarity(grads[0], grads[1], dim=0)
  assert cosine > 0.99, f"compiled vs eager policy gradient cosine {cosine:.3f}"


def test_lambda_returns_hand_computed_h3():
  gamma, lam = 0.9, 0.8
  r = [0.1, 0.2, 0.3]  # r_1, r_2, r_3
  d = [0.0, 0.5, 0.25]  # d_1, d_2, d_3
  v = [1.0, 2.0, 3.0, 4.0]  # V_0..V_3
  r3 = v[3]
  r2 = r[2] + gamma * (1 - d[2]) * ((1 - lam) * v[3] + lam * r3)
  r1 = r[1] + gamma * (1 - d[1]) * ((1 - lam) * v[2] + lam * r2)
  r0 = r[0] + gamma * (1 - d[0]) * ((1 - lam) * v[1] + lam * r1)
  expected = torch.tensor([[r0], [r1], [r2], [r3]], dtype=torch.float64)
  out = lambda_returns(
    torch.tensor(r, dtype=torch.float64).unsqueeze(-1),
    torch.tensor(d, dtype=torch.float64).unsqueeze(-1),
    torch.tensor(v, dtype=torch.float64).unsqueeze(-1),
    gamma,
    lam,
  )
  torch.testing.assert_close(out, expected)


def test_runner_learn_save_load_and_inference(device, tmp_path):
  torch.manual_seed(0)
  assert [sim_steps_at(i, 2.5) for i in range(4)] == [2, 3, 2, 3]
  env = _SyntheticEnv(64, device)
  wm_cfg, policy_cfg, critic_cfg = _small_cfgs()
  cfg = SuperMimicRunnerCfg(
    max_iterations=30,
    save_interval=10,
    log_interval=10,
    model_interface=_SyntheticInterface,  # type: ignore[arg-type]
    world_model=wm_cfg,
    policy=policy_cfg,
    critic=critic_cfg,
    collection=CollectionCfg(
      replay_steps=200, prefill_steps=16, warmup_world_model_iterations=5
    ),
  )
  wrapped = cast(RslRlVecEnvWrapper, env)
  runner = SuperMimicRunner(wrapped, asdict(cfg), log_dir=str(tmp_path), device=device)
  runner.learn(num_learning_iterations=30, init_at_random_ep_len=True)

  # Prefill + 15 x 2 + 15 x 3 sim steps.
  assert env.common_step_counter == 16 + 75
  assert runner.current_learning_iteration == 30
  assert env.past_weight_updates == 30 and env.bin_update_calls > 0
  assert runner.alg.skipped_policy_steps == 0
  for it in (10, 20, 30):
    assert (tmp_path / f"model_{it}.pt").is_file()
  header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
  for tag in (
    "Loss/world_model_state",
    "Loss/policy",
    "Loss/critic",
    "Train/normalized_return",
  ):
    assert tag in header

  obs = env.get_observations()
  actions = runner.get_inference_policy(device)(obs)
  assert actions.shape == (64, NQ) and torch.isfinite(actions).all()

  restored = SuperMimicRunner(wrapped, asdict(cfg), log_dir=None, device=device)
  restored.load(str(tmp_path / "model_30.pt"), map_location=device)
  assert restored.current_learning_iteration == 30
  torch.testing.assert_close(restored.get_inference_policy(device)(obs), actions)
  for a, b in zip(
    restored.alg.world_model.state_dict().values(),
    runner.alg.world_model.state_dict().values(),
    strict=False,
  ):
    torch.testing.assert_close(a, b)
