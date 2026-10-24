"""Motion library: multi-clip loading, per-clip bins, return-based reset sampling, the
motion_end time-out, the future-frame goal, the SuperMimic env and library evaluation."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from conftest import get_test_device, write_g1_motion_npz

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionLoader
from mjlab.tasks.tracking.scripts.evaluate_library import (
  evaluate_library,
  make_library_env,
)

TASK_ID = "Mjlab-Tracking-Flat-Unitree-G1-SuperMimic"
CLIP_LENGTHS = (120, 75, 101)
NUM_ENVS = 16


@pytest.fixture(scope="module")
def motion_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, ...]:
  root = tmp_path_factory.mktemp("library")
  return tuple(
    str(write_g1_motion_npz(root / f"clip{i}.npz", num_frames=n))
    for i, n in enumerate(CLIP_LENGTHS)
  )


@pytest.fixture(scope="module")
def env(motion_files: tuple[str, ...]):
  """SuperMimic env on the 3-clip library; episodes end only at clip ends."""
  cfg = load_env_cfg(TASK_ID)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_files
  del cfg.terminations["anchor_pos"], cfg.terminations["anchor_ori"]
  cfg.scene.num_envs = NUM_ENVS
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  yield env
  env.close()


def _command(env: ManagerBasedRlEnv) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term("motion"))


def test_motion_loader_concatenates_clips(motion_files: tuple[str, ...]) -> None:
  body_indexes = torch.arange(30)
  loader = MotionLoader(motion_files, body_indexes)
  ends = np.cumsum(CLIP_LENGTHS)
  assert loader.clip_end.tolist() == ends.tolist()
  assert loader.clip_start.tolist() == [0, *ends[:-1].tolist()]
  assert loader.clip_names == ("clip0", "clip1", "clip2")
  assert loader.clip_id_of_frame.tolist() == sum(
    ([c] * n for c, n in enumerate(CLIP_LENGTHS)), []
  )
  for c, file in enumerate(motion_files):
    data = np.load(file)
    frames = slice(int(loader.clip_start[c]), int(loader.clip_end[c]))
    # Stored in float16, except the body positions.
    np.testing.assert_array_equal(
      loader.joint_vel[frames].numpy(), data["joint_vel"].astype(np.float16)
    )
    np.testing.assert_array_equal(
      loader.body_quat_w[frames].numpy(), data["body_quat_w"].astype(np.float16)
    )
    np.testing.assert_array_equal(loader.body_pos_w[frames].numpy(), data["body_pos_w"])
  # Only the tracked bodies are kept, in the given order.
  subset = MotionLoader(motion_files[0], torch.tensor([3, 0, 7]))
  np.testing.assert_array_equal(
    subset.body_pos_w.numpy(), np.load(motion_files[0])["body_pos_w"][:, [3, 0, 7]]
  )
  single = MotionLoader(motion_files[1], body_indexes)
  assert single.clip_end.tolist() == [CLIP_LENGTHS[1]]
  assert single.time_step_total == CLIP_LENGTHS[1]


def test_bins_are_per_clip(env: ManagerBasedRlEnv) -> None:
  command = _command(env)
  clip_of_frame = command.motion.clip_id_of_frame
  num_frames = command.motion.time_step_total
  start, end = command.bin_start, command.bin_end
  assert start[0] == 0 and end[-1] == num_frames
  assert torch.equal(start[1:], end[:-1])
  assert torch.equal(clip_of_frame[start], clip_of_frame[end - 1])
  assert ((end - start) <= 50).all()
  bins_per_clip = torch.bincount(clip_of_frame[start]).tolist()
  assert bins_per_clip == [n // 50 + 1 for n in CLIP_LENGTHS]
  frames = torch.arange(num_frames, device=env.device)
  assert torch.equal(
    command.bin_of(frames), torch.searchsorted(end, frames, right=True)
  )


def test_future_frames_clamp_to_clip_end(env: ManagerBasedRlEnv) -> None:
  command = _command(env)
  ends = np.cumsum(CLIP_LENGTHS)
  frames = [0, 112, 119, 120, 190, 194, 295]
  result = command.future_frames(torch.tensor(frames, device=env.device), 10)
  last = [int(ends[np.searchsorted(ends, f, side="right")]) - 1 for f in frames]
  expected = [
    [min(f + k, end) for k in range(10)] for f, end in zip(frames, last, strict=True)
  ]
  assert result.tolist() == expected


def test_return_sampling(env: ManagerBasedRlEnv) -> None:
  command = _command(env)
  assert command.cfg.sampling_mode == "return"
  num_bins = command.bin_count
  probs = command.bin_sampling_probabilities()
  torch.testing.assert_close(probs, torch.full_like(probs, 1 / num_bins))

  bins = torch.tensor([0, 0, 3], device=env.device)
  command.update_bin_returns(bins, torch.tensor([8.0, 4.0, 1.0], device=env.device))
  expected_returns = torch.zeros(num_bins, device=env.device)
  expected_returns[0], expected_returns[3] = 0.1 * 6.0, 0.1 * 1.0
  torch.testing.assert_close(command.bin_returns, expected_returns)
  softmax = torch.softmax(-25.0 * expected_returns / 10.0, dim=0)
  probs = command.bin_sampling_probabilities()
  torch.testing.assert_close(probs, 0.5 / num_bins + 0.4 * softmax + 0.1 / num_bins)
  torch.testing.assert_close(probs.sum(), torch.tensor(1.0, device=env.device))
  assert probs[0] < probs[3] < probs[1]

  command.update_past_weights()
  torch.testing.assert_close(
    command.bin_past_weights, (1 - 1e-4) / num_bins + 1e-4 * softmax
  )

  # One hard bin: it receives 0.5 / Nb + 0.4 + 0.1 / Nb of the resets.
  hard_bin = 5
  command.bin_returns[:] = 10.0
  command.bin_returns[hard_bin] = 0.0
  command.bin_past_weights[:] = 1 / num_bins
  env_ids = torch.arange(NUM_ENVS, device=env.device)
  starts = []
  for _ in range(200):
    command.reset(env_ids)
    starts.append(command.time_steps.clone())
  starts = torch.cat(starts)
  assert (starts < command.clip_end_of(starts) - 1).all()
  hard_share = (command.bin_of(starts) == hard_bin).float().mean().item()
  assert abs(hard_share - (0.6 / num_bins + 0.4)) < 0.05

  command.bin_returns.zero_()
  command.bin_past_weights[:] = 1 / num_bins
  env.reset()


def test_existing_sampling_modes_on_library(env: ManagerBasedRlEnv) -> None:
  command = _command(env)
  env_ids = torch.arange(NUM_ENVS, device=env.device)
  try:
    command.cfg.sampling_mode = "start"
    command.reset(env_ids)
    assert torch.equal(command.time_steps, command.motion.clip_start[env_ids % 3])
    for mode in ("adaptive", "uniform"):
      command.cfg.sampling_mode = mode
      for _ in range(20):
        command.reset(env_ids)
        frames = command.time_steps
        assert ((frames >= 0) & (frames < command.motion.time_step_total)).all()
  finally:
    command.cfg.sampling_mode = "return"
    env.reset()


def test_motion_end_never_wraps(env: ManagerBasedRlEnv) -> None:
  """With motion_end, every resample is a reset: frames advance by one inside a clip."""
  command = _command(env)
  env.reset()
  resampled = 0
  resample_command = command._resample_command

  def counting_resample(env_ids: torch.Tensor) -> None:
    nonlocal resampled
    resampled += len(env_ids)
    resample_command(env_ids)

  command._resample_command = counting_resample  # type: ignore[method-assign]
  try:
    resets, clip_ends = 0, 0
    actions = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim)
    for _ in range(300):
      frame = command.time_steps.clone()
      _, _, terminated, truncated, _ = env.step(actions.to(env.device))
      done = terminated | truncated
      at_clip_end = env.termination_manager.get_term("motion_end")
      assert torch.equal(at_clip_end, frame == command.clip_end_of(frame) - 1)
      resets += int(done.sum())
      clip_ends += int(at_clip_end.sum())
      next_frame = command.time_steps[~done]
      assert torch.equal(next_frame, frame[~done] + 1)
      clip_of_frame = command.motion.clip_id_of_frame
      assert torch.equal(clip_of_frame[next_frame], clip_of_frame[frame[~done]])
    assert clip_ends > NUM_ENVS
    assert resampled == resets

    # Control: the counter sees MotionCommand's wrap when the clip end is not a done.
    command.time_steps[0] = command.clip_end_of(command.time_steps[:1])[0] - 1
    env.command_manager.compute(dt=env.step_dt)
    assert resampled > resets
  finally:
    del command._resample_command
    env.reset()


def _matrix(quat: np.ndarray) -> np.ndarray:
  w, x, y, z = quat / np.linalg.norm(quat)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def test_goal_matches_hand_computation(env: ManagerBasedRlEnv) -> None:
  command = _command(env)
  obs, _ = env.reset()
  actions = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim)
  for _ in range(5):
    obs, *_ = env.step(actions.to(env.device))
  # The default task's goal is the reference layout; the anchor layout's term is
  # checked below by hand.
  goal = mdp.motion_future_frames_reference(env, "motion", future_frames=10)
  assert goal.shape == (NUM_ENVS, 730)
  torch.testing.assert_close(obs["goal"], goal, atol=0, rtol=0)
  goal = mdp.motion_future_frames_b(env, "motion", future_frames=10)
  assert goal.shape == (NUM_ENVS, 670)

  # Near its clip end, so the last future frames are clamped.
  env_id = 1
  frame = int(command.clip_end_of(command.time_steps[env_id : env_id + 1])[0]) - 4
  command.time_steps[env_id] = frame
  goal = mdp.motion_future_frames_b(env, "motion", future_frames=10)[env_id]

  motion, anchor = command.motion, command.motion_anchor_body_index
  last = int(command.clip_end_of(command.time_steps[env_id : env_id + 1])[0]) - 1
  robot_pos = (command.robot_anchor_pos_w - env.scene.env_origins)[env_id]
  robot_pos = robot_pos.double().cpu().numpy()
  robot_rot = _matrix(command.robot_anchor_quat_w[env_id].double().cpu().numpy())
  expected = []
  for k in range(10):
    f = min(frame + k, last)
    ref_pos = motion.body_pos_w[f, anchor].double().cpu().numpy()
    ref_rot = _matrix(motion.body_quat_w[f, anchor].double().cpu().numpy())
    expected += [
      robot_rot.T @ (ref_pos - robot_pos),
      (robot_rot.T @ ref_rot)[:, :2].reshape(-1),
      motion.joint_pos[f].double().cpu().numpy(),
      motion.joint_vel[f].double().cpu().numpy(),
    ]
  np.testing.assert_allclose(
    goal.double().cpu().numpy(), np.concatenate(expected), atol=1e-5, rtol=0
  )
  env.reset()


def test_supermimic_env_builds_and_steps(motion_files: tuple[str, ...]) -> None:
  cfg = load_env_cfg(TASK_ID)
  assert "push_robot" not in cfg.events
  assert "self_collisions" not in cfg.rewards
  assert cfg.rewards["joint_limit"].weight == -2.0
  assert "ee_body_pos" not in cfg.terminations
  assert cfg.terminations["motion_end"].time_out
  assert set(cfg.observations) == {"actor", "goal"}
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_files[:2]
  cfg.scene.num_envs = 4
  cfg.seed = 0
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    obs, _ = env.reset()
    shapes = {k: tuple(cast(torch.Tensor, v).shape) for k, v in obs.items()}
    assert shapes == {"actor": (4, 990), "goal": (4, 730)}
    for _ in range(50):
      obs, reward, *_ = env.step(0.3 * torch.randn(4, 29, device=env.device))
      assert torch.isfinite(reward).all()
      for group in ("actor", "goal"):
        assert torch.isfinite(cast(torch.Tensor, obs[group])).all()
  finally:
    env.close()


def test_evaluate_library_zero_policy(
  motion_files: tuple[str, ...], tmp_path: Path
) -> None:
  """A walking clip (the robot holding still falls behind) and a standing clip."""
  standing = write_g1_motion_npz(tmp_path / "stand.npz", 60, amplitude=0.0, speed=0.0)
  files = (motion_files[1], str(standing))

  def evaluate(
    motion_file: tuple[str, ...], num_envs: int | None = None, rollouts: int = 1
  ) -> dict:
    env = make_library_env(
      TASK_ID,
      motion_file,
      get_test_device(),
      rollouts_per_clip=rollouts,
      num_envs=num_envs,
    )
    try:

      def zero_policy(obs) -> torch.Tensor:
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)

      return evaluate_library(env, zero_policy, rollouts)
    finally:
      env.close()

  result = evaluate(files)
  # 2 rollouts per clip on 3 envs (two waves), from a list file: the same results.
  listing = tmp_path / "library.txt"
  listing.write_text("".join(f"{file}\n" for file in files))
  waves = evaluate((str(listing),), num_envs=3, rollouts=2)
  assert (result["num_envs"], waves["num_envs"]) == (2, 3)
  assert [c["rollout"] for c in waves["clips"]] == [0, 0, 1, 1]
  assert waves["clip_failure_fraction"] == result["clip_failure_fraction"]
  for item, wave_clip in enumerate(waves["clips"]):
    clip = dict(result["clips"][item % 2])
    terms, wave_terms = clip.pop("reward_terms"), wave_clip.pop("reward_terms")
    assert sum(terms.values()) == pytest.approx(clip["return"], rel=1e-4)
    assert clip | {"rollout": item // 2} == pytest.approx(wave_clip, rel=1e-4)
    assert terms == pytest.approx(wave_terms, rel=1e-3, abs=1e-5)
  clips = result["clips"]
  assert [c["clip"] for c in clips] == ["clip1", "stand"]
  assert [c["frames"] for c in clips] == [CLIP_LENGTHS[1], 60]
  assert result["max_step_reward"] == pytest.approx(0.1)
  for clip in clips:
    assert clip["steps"] <= clip["frames"] - 1
    assert clip["failed"] or clip["steps"] == clip["frames"] - 1
    assert clip["normalized_return"] == pytest.approx(
      clip["return"] / (0.1 * (clip["frames"] - 1))
    )
  assert not clips[1]["failed"] and clips[1]["normalized_return"] > 0.5
  assert result["num_failed"] == sum(c["failed"] for c in clips)
  assert result["mean_normalized_return"] == pytest.approx(
    np.mean([c["normalized_return"] for c in clips])
  )
  assert result["frame_weighted_normalized_return"] == pytest.approx(
    sum(c["return"] for c in clips) / sum(0.1 * (c["frames"] - 1) for c in clips)
  )
