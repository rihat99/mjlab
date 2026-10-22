"""Tests for the pure tracking kernels behind the tracking env terms."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import get_test_device, write_g1_motion_npz

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg, kernels
from mjlab.tasks.tracking.mdp.kinematics import TorchKinematics
from mjlab.utils.lab_api.math import quat_apply, quat_error_magnitude


def _random_quat(n: int, generator: torch.Generator) -> torch.Tensor:
  return torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)


def test_quat_error_magnitude_matches_original() -> None:
  generator = torch.Generator().manual_seed(0)
  q1 = _random_quat(100_000, generator)
  q2 = _random_quat(100_000, generator)
  near = torch.nn.functional.normalize(
    q1 + 1e-4 * torch.randn(q1.shape, generator=generator), dim=-1
  )
  for a, b in ((q1, q2), (q1, near), (q1, q1), (q1, -q1)):
    torch.testing.assert_close(
      kernels.quat_error_magnitude(a, b), quat_error_magnitude(a, b), atol=1e-6, rtol=0
    )


def test_quat_error_magnitude_gradient_finite_at_zero_error() -> None:
  """mjlab's quat_error_magnitude gives NaN gradients when q1 == q2."""
  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], requires_grad=True)
  error = kernels.quat_error_magnitude(quat, quat)
  (grad,) = torch.autograd.grad(error.sum() + (error**2).sum(), quat)
  assert torch.isfinite(grad).all()


def test_tracking_rewards_have_finite_gradients_at_zero_error() -> None:
  """Rewards of an FK state against itself (perfect tracking) backpropagate."""
  model = Entity(get_g1_robot_cfg()).compile()
  fk = TorchKinematics(model, ["pelvis", "torso_link", "left_wrist_yaw_link"])
  generator = torch.Generator().manual_seed(0)
  q = (0.3 * torch.randn(3, 29, generator=generator)).requires_grad_(True)
  root_quat = _random_quat(3, generator).requires_grad_(True)
  pos, quat = fk.forward(torch.zeros(3, 3), root_quat, q)
  reward = (
    kernels.root_ori_reward(quat[:, 1], quat[:, 1].detach(), std=0.4)
    + kernels.body_ori_reward(quat, quat.detach(), std=0.4)
    + kernels.body_pos_reward(pos, pos.detach(), std=0.3)
  )
  torch.testing.assert_close(reward, torch.full_like(reward, 3.0))
  grads = torch.autograd.grad(reward.sum(), (q, root_quat))
  assert all(torch.isfinite(g).all() for g in grads)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory):
  motion = write_g1_motion_npz(tmp_path_factory.mktemp("motion") / "motion.npz")
  cfg = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1")
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(motion)
  cfg.scene.num_envs = 4
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


def _call_term(env: ManagerBasedRlEnv, term_cfg) -> torch.Tensor:
  return term_cfg.func(env, **term_cfg.params)


def test_kernels_on_fk_state_match_env_terms(env: ManagerBasedRlEnv) -> None:
  """Kernels fed with FK of (qpos, qvel) and the reference reproduce the env terms.

  The env terms are re-evaluated after each step, when the sim kinematics are
  fresh, so both sides see the same state; this is the imagined-state path.
  """
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  robot = env.scene["robot"]
  tracked = [f"robot/{name}" for name in command.cfg.body_names]
  fk = TorchKinematics(env.sim.mj_model, tracked, device=env.device)
  anchor = command.motion_anchor_body_index
  rewards = {
    n: env.reward_manager.get_term_cfg(n) for n in env.reward_manager.active_terms
  }
  std: dict[str, float] = {
    n: cfg.params["std"] for n, cfg in rewards.items() if "std" in cfg.params
  }
  terminations = env.termination_manager

  generator = torch.Generator().manual_seed(0)
  for _ in range(30):
    action = 0.3 * torch.randn(env.num_envs, 29, generator=generator)
    env.step(action.to(env.device))
    qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
    root_ang_vel_w = quat_apply(qpos[:, 3:7], qvel[:, 3:6])
    pos, quat, lin_vel, ang_vel = fk.forward_with_velocity(
      qpos[:, :3], qpos[:, 3:7], qvel[:, :3], root_ang_vel_w, qpos[:, 7:], qvel[:, 6:]
    )
    ref_anchor_pos, ref_anchor_quat = command.anchor_pos_w, command.anchor_quat_w
    pos_delta, quat_delta = kernels.anchor_frame(
      pos[:, anchor], quat[:, anchor], ref_anchor_pos, ref_anchor_quat
    )
    target_pos, target_quat = kernels.relative_body_targets(
      command.body_pos_w, command.body_quat_w, ref_anchor_pos, pos_delta, quat_delta
    )
    expected = {
      "motion_global_root_pos": kernels.root_pos_reward(
        pos[:, anchor], ref_anchor_pos, std["motion_global_root_pos"]
      ),
      "motion_global_root_ori": kernels.root_ori_reward(
        quat[:, anchor], ref_anchor_quat, std["motion_global_root_ori"]
      ),
      "motion_body_pos": kernels.body_pos_reward(
        pos, target_pos, std["motion_body_pos"]
      ),
      "motion_body_ori": kernels.body_ori_reward(
        quat, target_quat, std["motion_body_ori"]
      ),
      "motion_body_lin_vel": kernels.body_lin_vel_reward(
        lin_vel, command.body_lin_vel_w, std["motion_body_lin_vel"]
      ),
      "motion_body_ang_vel": kernels.body_ang_vel_reward(
        ang_vel, command.body_ang_vel_w, std["motion_body_ang_vel"]
      ),
      "action_rate_l2": kernels.action_rate_l2(
        env.action_manager.action, env.action_manager.prev_action
      ),
      "joint_limit": kernels.joint_limit_penalty(
        qpos[:, 7:], robot.data.soft_joint_pos_limits
      ),
    }
    for name, value in expected.items():
      torch.testing.assert_close(
        value, _call_term(env, rewards[name]), atol=1e-4, rtol=0, msg=name
      )

    bad_z, bad_tilt, e_z, e_tilt = kernels.anchor_termination(
      pos[:, anchor], quat[:, anchor], ref_anchor_pos, ref_anchor_quat
    )
    for name, bad, error, threshold in (
      ("anchor_pos", bad_z, e_z, 0.25),
      ("anchor_ori", bad_tilt, e_tilt, 0.8),
    ):
      term_cfg = terminations.get_term_cfg(name)
      assert term_cfg.params["threshold"] == threshold
      clear = (error - threshold).abs() > 1e-4
      assert torch.equal(bad[clear], _call_term(env, term_cfg)[clear])
      # Done envs are already reset at observation time, so the cfg thresholds
      # almost never fire here; thresholds just below / above each env's kernel
      # error check that the env term sees the same error.
      for i, value in enumerate(error.tolist()):
        for margin, fires in ((-1e-4, True), (1e-4, False)):
          params = {**term_cfg.params, "threshold": value + margin}
          assert term_cfg.func(env, **params)[i].item() is fires

    anchor_pos_b, anchor_ori_b = kernels.pose_in_frame(
      ref_anchor_pos, ref_anchor_quat, pos[:, anchor], quat[:, anchor]
    )
    body_pos_b, body_ori_b = kernels.pose_in_frame(
      pos,
      quat,
      pos[:, anchor : anchor + 1].expand_as(pos),
      quat[:, anchor : anchor + 1].expand_as(quat),
    )
    critic = {
      n: env.observation_manager.get_term_cfg("critic", n)
      for n in ("motion_anchor_pos_b", "motion_anchor_ori_b", "body_pos", "body_ori")
    }
    for value, name in (
      (anchor_pos_b, "motion_anchor_pos_b"),
      (anchor_ori_b, "motion_anchor_ori_b"),
      (body_pos_b.reshape(env.num_envs, -1), "body_pos"),
      (body_ori_b.reshape(env.num_envs, -1), "body_ori"),
    ):
      torch.testing.assert_close(
        value, _call_term(env, critic[name]), atol=1e-5, rtol=0, msg=name
      )

    proprio = kernels.proprio_observation(
      qpos[:, 7:],
      qvel[:, 6:],
      qpos[:, 3:7],
      qvel[:, 3:6],
      robot.data.encoder_bias,
      robot.data.default_joint_pos,
      env.action_manager.action,
    )
    env_proprio = torch.cat(
      [
        envs_mdp.joint_pos_rel(env, biased=True),
        envs_mdp.joint_vel_rel(env),
        envs_mdp.builtin_sensor(env, "robot/imu_ang_vel"),
        envs_mdp.projected_gravity(env),
        envs_mdp.last_action(env),
      ],
      dim=-1,
    )
    torch.testing.assert_close(proprio, env_proprio, atol=1e-5, rtol=0)


def _parity_files() -> tuple[Path, Path] | None:
  before = os.environ.get("TRACKING_PARITY_BEFORE")
  after = os.environ.get("TRACKING_PARITY_AFTER")
  if before is None or after is None:
    return None
  paths = Path(before), Path(after)
  return paths if all(path.is_file() for path in paths) else None


@pytest.mark.skipif(
  _parity_files() is None,
  reason="set TRACKING_PARITY_BEFORE/AFTER to rollouts recorded by record_baseline.py",
)
def test_recorded_rollout_parity() -> None:
  """A fixed-seed CPU rollout recorded before the kernel refactor matches one after."""
  files = _parity_files()
  assert files is not None
  before, after = (np.load(path) for path in files)
  for key in ("reward_term_names", "termination_term_names", "time_steps"):
    np.testing.assert_array_equal(before[key], after[key])
  np.testing.assert_array_equal(before["terminations"], after["terminations"])
  for key in ("step_reward", "actor", "critic", "qpos", "qvel"):
    np.testing.assert_allclose(before[key], after[key], atol=1e-6, rtol=0, err_msg=key)
