"""TrackingModelInterface vs the live SuperMimic env (2 synthetic G1 clips, 8 envs,
noise off, random actions, 100 steps): observation, goal, termination and reward
heads, state read-out, terminal snapshots, frame bookkeeping and gradients.

The env scores a step on body kinematics one physics substep stale (the forward inside
the last ``mj_step``) and on relative body targets built from the previous state's
anchor. A per-substep probe records the stale state so the env's numbers can be
reproduced exactly; the interface itself scores the reached state s_next.
"""

from typing import cast

import pytest
import torch
from conftest import get_test_device, write_g1_motion_npz

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.rl.supermimic import FloatingBaseState
from mjlab.rl.supermimic.algorithm import _maybe_compile
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, kernels
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.kinematics import TorchKinematics
from mjlab.tasks.tracking.supermimic import TrackingModelInterface
from mjlab.tasks.tracking.supermimic.recorder import floating_base_state

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"
OBS_TERM_DIMS = (29, 29, 3, 3, 29, 6)
OBS_DIM, GOAL_DIM = 99, 730
NUM_ENVS = 8
NUM_STEPS = 100
# Envs started 3 frames before their clip end: motion_end fires on their third step.
NEAR_END = {0: 0, 1: 1}  # env -> clip


class _SubstepState:
  """Metrics term evaluated every physics substep; keeps the last two states."""

  instance: "_SubstepState | None" = None

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.states: list[torch.Tensor] = []
    self._zeros = torch.zeros(env.num_envs, device=env.device)
    _SubstepState.instance = self

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    self.states = [*self.states[-1:], floating_base_state(env, "robot")]
    return self._zeros


class _PreResetQpos(RecorderTerm):
  """Independent copy of qpos at the auto-reset hook."""

  def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.qpos: torch.Tensor | None = None

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    self.qpos = self._env.sim.data.qpos.clone()


def _newest_frame(actor: torch.Tensor, term_dims: tuple[int, ...]) -> torch.Tensor:
  """Last history frame (N, 99) of a term-major flattened history (N, 10 * 99)."""
  history = actor.shape[-1] // sum(term_dims)
  parts = torch.split(actor, [history * d for d in term_dims], dim=-1)
  return torch.cat(
    [p.view(-1, history, d)[:, -1] for p, d in zip(parts, term_dims, strict=True)], -1
  )


@pytest.fixture(scope="module")
def rollout(tmp_path_factory: pytest.TempPathFactory):
  root = tmp_path_factory.mktemp("clips")
  clips = (
    write_g1_motion_npz(root / "walk.npz", num_frames=150),
    write_g1_motion_npz(root / "sway.npz", num_frames=120, amplitude=0.4, speed=0.0),
  )
  cfg = load_env_cfg(TASK_ID)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = tuple(str(clip) for clip in clips)
  cfg.observations["actor"].enable_corruption = False
  cfg.metrics["substep_state"] = MetricsTermCfg(func=_SubstepState, per_substep=True)
  cfg.recorders["pre_reset_qpos"] = RecorderTermCfg(func=_PreResetQpos, params={})
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  device = get_test_device()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  iface = TrackingModelInterface(env)
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  for env_id, clip in NEAR_END.items():
    end = int(command.motion.clip_end[clip])
    command.reset_to_frame(torch.tensor([env_id], device=device), end - 3)
  env.sim.forward()
  command.update_relative_body_poses()

  substep = _SubstepState.instance
  assert substep is not None
  pre_reset = cast(_PreResetQpos, env.recorder_manager.get_term("pre_reset_qpos"))
  generator = torch.Generator(device=device).manual_seed(0)
  steps = []
  for _ in range(NUM_STEPS):
    record = {
      "s": iface.state_from_sim(),
      "frame": iface.frame_from_sim(),
      "context": iface.context_normalized(),
      "a_prev": env.action_manager.action.clone(),
      "target_pos": command.body_pos_relative_w.clone(),
      "target_quat": command.body_quat_relative_w.clone(),
    }
    action = torch.randn(NUM_ENVS, iface.action_dim, generator=generator, device=device)
    pre_reset.qpos = None
    obs, reward, _, truncated, _ = env.step(action)
    terminated = env.termination_manager.terminated.clone()  # incl. time-out overlap
    done = terminated | truncated
    snapshot = iface.terminal_state_snapshot()
    s_post = iface.state_from_sim()
    record |= {
      "a": action,
      "reward": reward.clone(),
      "terms": env.reward_manager._step_reward.clone() * env.step_dt,
      "terminated": terminated,
      "truncated": truncated.clone(),
      "s_post": s_post,
      "frame_post": iface.frame_from_sim(),
      "context_post": iface.context_normalized(),
      "robot_anchor_pos": command.robot_anchor_pos_w - env.scene.env_origins,
      "robot_anchor_quat": command.robot_anchor_quat_w.clone(),
      "a_post": env.action_manager.action.clone(),
      "actor": cast(torch.Tensor, obs["actor"]).clone(),
      "goal": cast(torch.Tensor, obs["goal"]).clone(),
      "stale": substep.states[0],
      "s_next": s_post
      if snapshot is None
      else torch.where(done[:, None], snapshot, s_post),
      "snapshot": None if snapshot is None else snapshot.clone(),
      "pre_reset_qpos": pre_reset.qpos,
    }
    steps.append(record)
  yield env, iface, steps
  env.close()


def _stack(steps: list[dict], key: str) -> torch.Tensor:
  return torch.stack([step[key] for step in steps])


def test_dims_and_context(rollout) -> None:
  env, iface, _ = rollout
  assert (iface.state_dim, iface.obs_dim, iface.goal_dim) == (71, OBS_DIM, GOAL_DIM)
  assert (iface.context_dim, iface.feature_dim, iface.action_dim) == (33, 68, 29)
  assert iface.critic_dim == 68 + GOAL_DIM + 33 + 9
  assert iface.obs_term_dims == OBS_TERM_DIMS
  group_dims = env.observation_manager.group_obs_dim
  assert group_dims["actor"] == (10 * OBS_DIM,) and group_dims["goal"] == (GOAL_DIM,)
  assert iface.max_step_reward == pytest.approx(0.1)
  context = iface.context_normalized()
  assert context.shape == (NUM_ENVS, 33)
  assert context.abs().max() <= 1.0
  # Friction is shared by the foot geoms; the context reads one of them.
  robot = env.scene["robot"]
  feet = robot.indexing.geom_ids[
    robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")[0]
  ]
  friction = env.sim.model.geom_friction[:, feet, 0]
  assert torch.equal(friction, friction[:, :1].expand_as(friction))
  torch.testing.assert_close(context[:, 0], (friction[:, 0] - 0.75) / 0.45)
  torch.testing.assert_close(context[:, 1:30], robot.data.encoder_bias / 0.01)


def test_observation_and_goal_match_env(rollout) -> None:
  _, iface, steps = rollout
  obs_err = goal_err = 0.0
  for step in steps:
    newest = _newest_frame(step["actor"], iface.obs_term_dims)
    obs = iface.actor_observation(
      step["s_post"], step["a_post"], step["context_post"], step["frame_post"]
    )
    goal = iface.goal(step["s_post"], step["frame_post"])
    obs_err = max(obs_err, float((obs - newest).abs().max()))
    goal_err = max(goal_err, float((goal - step["goal"]).abs().max()))
  assert obs_err <= 1e-5
  assert goal_err <= 1e-5


def test_context_resampled_per_episode(rollout) -> None:
  """Friction and encoder bias change exactly at resets; the CoM offset is fixed."""
  _, _, steps = rollout
  resets = 0
  for step in steps:
    done = step["terminated"] | step["truncated"]
    before, after = step["context"], step["context_post"]
    episode_part = (before[:, :30] != after[:, :30]).any(-1)
    assert torch.equal(episode_part, done)
    assert torch.equal(before[:, 30:], after[:, 30:])
    resets += int(done.sum())
  assert resets >= 5


def test_critic_root_pose_error_matches_env(rollout) -> None:
  """The privileged critic input: the reference anchor of frame k_t - 1 in the robot
  anchor frame, as the env's command computes it on the post-step state."""
  env, iface, steps = rollout
  command = cast(MotionCommand, env.command_manager.get_term("motion"))
  anchor = command.motion_anchor_body_index
  err = 0.0
  for step in steps:
    s, frame = step["s_post"], step["frame_post"]
    x = iface.critic_input(
      s,
      frame,
      iface.invariant_features(s),
      iface.goal(s, frame),
      step["context_post"],
    )
    current = command.current_frames(frame)
    ref_quat = command.motion.body_quat_w[current, anchor].float()
    pos_b, ori_b = kernels.pose_in_frame(
      command.motion.body_pos_w[current, anchor],
      ref_quat / ref_quat.norm(dim=-1, keepdim=True),
      step["robot_anchor_pos"],
      step["robot_anchor_quat"],
    )
    expected = torch.cat((pos_b, ori_b), dim=-1)
    err = max(err, float((x[:, -9:] - expected).abs().max()))
  assert err <= 1e-4


def test_compiled_critic_input_matches_eager(rollout) -> None:
  """The learner's compiled critic_input (FK of the anchor) against the eager one:
  values and the gradient to the state, in fp32 and under bf16 autocast."""
  env, iface, steps = rollout
  if env.device == "cpu":
    pytest.skip("the learner compiles for CUDA")
  compiled = _maybe_compile(iface.critic_input, True)
  s = _stack(steps, "s_post").flatten(0, 1)
  frame = _stack(steps, "frame_post").flatten()
  context = _stack(steps, "context_post").flatten(0, 1)
  weights = torch.linspace(-1.0, 1.0, iface.critic_dim, device=s.device)

  def value_and_grad(fn, autocast: bool) -> tuple[torch.Tensor, torch.Tensor]:
    x = s.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
      out = fn(x, frame, iface.invariant_features(x), iface.goal(x, frame), context)
    (grad,) = torch.autograd.grad((out * weights).sum(), x)
    return out.detach(), grad

  for autocast in (False, True):
    out, grad = value_and_grad(compiled, autocast)
    out_eager, grad_eager = value_and_grad(iface.critic_input, autocast)
    torch.testing.assert_close(out, out_eager, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad, grad_eager, rtol=1e-4, atol=1e-4)


def test_termination_matches_env(rollout, monkeypatch: pytest.MonkeyPatch) -> None:
  env, _, steps = rollout
  terminated = _stack(steps, "terminated").flatten()
  frames = _stack(steps, "frame").flatten()
  assert int(terminated.sum()) >= 3
  # The env's rule saw the state one substep before s_next. At vanishing
  # temperatures the soft rule is the env's threshold rule.
  command_cfg = env.command_manager.get_term("motion").cfg
  monkeypatch.setattr(command_cfg, "termination_z_temperature", 1e-6)
  monkeypatch.setattr(command_cfg, "termination_tilt_temperature", 1e-6)
  sharp = TrackingModelInterface(env)
  stale = _stack(steps, "stale").flatten(0, 1)
  soft = sharp.termination_soft(stale, frames)
  assert torch.equal(soft > 0.5, terminated)


def test_reward_matches_env(rollout) -> None:
  _, iface, steps = rollout
  gaps = []
  for step in steps:
    terms = iface.reward_terms(step["s_next"], step["a"], step["a_prev"], step["frame"])
    reward = iface.reward(step["s_next"], step["a"], step["a_prev"], step["frame"])
    torch.testing.assert_close(reward, torch.stack(list(terms.values())).sum(0))
    gaps.append(float((reward - step["reward"]).abs().max()))
  # The gap is the env's timing (next test reproduces the env to 1e-6). Under these
  # violent random actions it reaches ~6e-3 (6 % of the max step reward).
  assert max(gaps) <= 1e-2


def test_env_reward_reproduced_from_stale_state(rollout) -> None:
  """Pins the cause of the reward gap: on the stale state, with the relative targets
  the env built from the previous anchor, every env term is reproduced."""
  env, iface, steps = rollout
  names = env.reward_manager.active_terms
  kinematics = TorchKinematics(
    env.sim.mj_model,
    [f"robot/{name}" for name in env.cfg.commands["motion"].body_names],
    device=env.device,
  )
  state = FloatingBaseState(iface.nq)
  err = dict.fromkeys(names, 0.0)
  for step in steps:
    on_stale = iface.reward_terms(
      step["stale"], step["a"], step["a_prev"], step["frame"]
    )
    on_next = iface.reward_terms(
      step["s_next"], step["a"], step["a_prev"], step["frame"]
    )
    pos, quat, _, _, q, _ = state.split(step["stale"])
    body_pos, body_quat = kinematics.forward(pos, quat, q)
    origins = env.scene.env_origins[:, None]
    weight = {n: env.reward_manager.get_term_cfg(n).weight * env.step_dt for n in names}
    expected = {
      "motion_body_pos": weight["motion_body_pos"]
      * kernels.body_pos_reward(body_pos, step["target_pos"] - origins, 0.3),
      "motion_body_ori": weight["motion_body_ori"]
      * kernels.body_ori_reward(body_quat, step["target_quat"], 0.4),
      # q-based terms read the final qpos, not the stale kinematics.
      "action_rate_l2": on_next["action_rate_l2"],
      "joint_limit": on_next["joint_limit"],
    }
    for i, name in enumerate(names):
      value = expected.get(name, on_stale[name])
      err[name] = max(err[name], float((value - step["terms"][:, i]).abs().max()))
  assert max(err.values()) <= 1e-6


def test_state_read_out(rollout) -> None:
  env, iface, _ = rollout
  robot = env.scene["robot"]
  s = iface.state_from_sim()
  pos, quat, lin_vel, ang_vel, q, qd = FloatingBaseState(iface.nq).split(s)
  data = robot.data
  torch.testing.assert_close(pos + env.scene.env_origins, data.root_link_pos_w)
  sign = torch.sign((quat * data.root_link_quat_w).sum(-1, keepdim=True))
  torch.testing.assert_close(quat * sign, data.root_link_quat_w)
  torch.testing.assert_close(lin_vel, data.root_link_lin_vel_w, atol=1e-4, rtol=0)
  torch.testing.assert_close(ang_vel, data.root_link_ang_vel_w, atol=1e-4, rtol=0)
  assert torch.equal(q, data.joint_pos) and torch.equal(qd, data.joint_vel)
  assert torch.equal(
    iface.state_from_sim(torch.tensor([2, 5], device=s.device)), s[[2, 5]]
  )


def test_terminal_snapshot(rollout) -> None:
  env, iface, steps = rollout
  robot = env.scene["robot"]
  checked = 0
  for step in steps:
    done = step["terminated"] | step["truncated"]
    if not done.any():
      continue
    ids = done.nonzero().squeeze(-1)
    snapshot, qpos = step["snapshot"][ids], step["pre_reset_qpos"][ids]
    assert not torch.allclose(snapshot, step["s_post"][ids])
    pos, quat, _, _, q, _ = FloatingBaseState(iface.nq).split(snapshot)
    root = qpos[:, robot.indexing.free_joint_q_adr]
    torch.testing.assert_close(pos + env.scene.env_origins[ids], root[:, :3])
    assert torch.equal(quat, root[:, 3:7])
    assert torch.equal(q, qpos[:, robot.indexing.joint_q_adr])
    checked += len(ids)
  assert checked >= 5


def test_frame_bookkeeping(rollout) -> None:
  _, iface, steps = rollout
  frame, frame_post = _stack(steps, "frame"), _stack(steps, "frame_post")
  truncated = _stack(steps, "truncated")
  done = _stack(steps, "terminated") | truncated
  clip_end = iface.clip_end(frame.flatten()).view_as(frame)
  advanced = frame_post == frame + 1
  assert torch.all(advanced | done)
  # Never past the clip end: the next frame stays in the clip.
  assert torch.all((frame_post < clip_end) | done)
  # The only time-out here is motion_end, on the clip's last frame.
  assert torch.equal(truncated, truncated & (frame == clip_end - 1))
  for env_id in NEAR_END:
    assert bool(truncated[2, env_id]) and not bool(done[:2, env_id].any())


def test_gradients_are_finite(rollout) -> None:
  _, iface, steps = rollout
  s_next = _stack(steps, "s_next").flatten(0, 1).requires_grad_(True)
  frames = _stack(steps, "frame").flatten()
  a, a_prev = _stack(steps, "a").flatten(0, 1), _stack(steps, "a_prev").flatten(0, 1)
  context = _stack(steps, "context").flatten(0, 1)
  reward = iface.reward(s_next, a, a_prev, frames)
  next_frames = torch.minimum(frames + 1, iface.clip_end(frames) - 1)
  goal = iface.goal(s_next, next_frames)
  obs = iface.add_observation_noise(
    iface.actor_observation(s_next, a_prev, context, next_frames)
  )
  features = iface.invariant_features(s_next)
  critic_x = iface.critic_input(s_next, next_frames, features, goal, context)
  (reward.sum() + goal.sum() + obs.sum() + critic_x.sum()).backward()
  grad = s_next.grad
  assert grad is not None and torch.isfinite(grad).all()
  # Every block of the state (pose, velocities, q, qd) reaches the reward.
  reward_grad = torch.autograd.grad(
    iface.reward(s_next, a, a_prev, frames).sum(), s_next
  )
  assert torch.all(reward_grad[0].abs().sum(0)[[0, 3, 7, 10, 13, 42]] > 0)


def test_termination_soft(rollout) -> None:
  _, iface, steps = rollout
  # Reset states sit on their reference (anchor errors of a few cm / centiradians).
  s, frames = steps[0]["s"], steps[0]["frame"]
  states = torch.cat(
    [s + torch.eye(71, device=s.device)[2] * dz for dz in (0, 0.25, 2)]
  )
  states.requires_grad_(True)
  soft = iface.termination_soft(states, frames.repeat(3)).view(3, NUM_ENVS)
  inside, near, outside = soft
  assert torch.all((soft >= 0) & (soft <= 1))
  assert torch.all(inside < 1e-3) and torch.all(outside > 1 - 1e-3)
  # Root raised by the height threshold: the anchor error is near it, d in (0, 1).
  assert torch.all((near > 0.01) & (near < 0.99))
  grad = torch.autograd.grad(soft.sum(), states)[0]
  assert torch.isfinite(grad).all()
  assert torch.all(grad.view(3, NUM_ENVS, -1)[1, :, 2] > 0)
