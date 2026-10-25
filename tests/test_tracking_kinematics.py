"""Tests for the differentiable forward kinematics of the tracking task."""

import pytest
import torch
from conftest import get_test_device, write_g1_motion_npz

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.mdp.kinematics import TorchKinematics
from mjlab.utils.lab_api.math import quat_apply

POS_TOL, QUAT_TOL, VEL_TOL = 1e-5, 1e-5, 1e-4


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


def _fk_from_state(fk: TorchKinematics, qpos: torch.Tensor, qvel: torch.Tensor):
  """FK of mjlab's generalized state: qvel[3:6] is the root angular velocity in
  the root frame, the FK takes it in the world frame."""
  root_ang_vel_w = quat_apply(qpos[:, 3:7], qvel[:, 3:6])
  return fk.forward_with_velocity(
    qpos[:, :3], qpos[:, 3:7], qvel[:, :3], root_ang_vel_w, qpos[:, 7:], qvel[:, 6:]
  )


def _assert_quat_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
  error = torch.minimum(
    (actual - expected).abs().amax(-1), (actual + expected).abs().amax(-1)
  )
  assert error.max().item() < QUAT_TOL


def test_fk_matches_live_env(env: ManagerBasedRlEnv) -> None:
  """FK of (qpos, qvel) equals the sim's xpos/xquat and mjlab's body_link_* views."""
  robot = env.scene["robot"]
  command = env.command_manager.get_term("motion")
  assert isinstance(command, MotionCommand)
  tracked = [f"robot/{name}" for name in command.cfg.body_names]
  fk_tracked = TorchKinematics(env.sim.mj_model, tracked, device=env.device)
  fk_all = TorchKinematics(
    env.sim.mj_model, [f"robot/{n}" for n in robot.body_names], device=env.device
  )
  assert fk_tracked.joint_names == tuple(f"robot/{n}" for n in robot.joint_names)

  generator = torch.Generator().manual_seed(0)
  for _ in range(20):
    action = 0.3 * torch.randn(env.num_envs, 29, generator=generator)
    env.step(action.to(env.device))
    qpos, qvel = env.sim.data.qpos, env.sim.data.qvel

    pos, quat, _, _ = _fk_from_state(fk_all, qpos, qvel)
    torch.testing.assert_close(
      pos, env.sim.data.xpos[:, robot.indexing.body_ids], atol=POS_TOL, rtol=0
    )
    _assert_quat_close(quat, env.sim.data.xquat[:, robot.indexing.body_ids])

    pos, quat, lin_vel, ang_vel = _fk_from_state(fk_tracked, qpos, qvel)
    torch.testing.assert_close(pos, command.robot_body_pos_w, atol=POS_TOL, rtol=0)
    _assert_quat_close(quat, command.robot_body_quat_w)
    torch.testing.assert_close(
      lin_vel, command.robot_body_lin_vel_w, atol=VEL_TOL, rtol=0
    )
    torch.testing.assert_close(
      ang_vel, command.robot_body_ang_vel_w, atol=VEL_TOL, rtol=0
    )


def test_fk_gradcheck() -> None:
  """Analytic gradients of poses and velocities match finite differences."""
  model = Entity(get_g1_robot_cfg()).compile()
  names = ["pelvis", "left_ankle_roll_link", "torso_link", "right_wrist_yaw_link"]
  fk = TorchKinematics(model, names, dtype=torch.float64)
  generator = torch.Generator().manual_seed(0)

  def rand(*shape: int) -> torch.Tensor:
    x = torch.randn(*shape, dtype=torch.float64, generator=generator)
    return x.requires_grad_(True)

  inputs = (rand(2, 3), rand(2, 4), rand(2, 3), rand(2, 3), rand(2, 29), rand(2, 29))
  assert torch.autograd.gradcheck(fk.forward_with_velocity, inputs)
