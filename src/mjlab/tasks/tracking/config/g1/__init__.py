from mjlab.rl.supermimic import SuperMimicRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_flat_tracking_supermimic_env_cfg,
)
from .rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_tracking_supermimic_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-SuperMimic",
  env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)

# The goal layout of the checkpoints before 2026-10-24: reference frames seen from
# the robot anchor (goal 670, actor 930), no orientation-error term.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-AnchorGoal",
  env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(goal="anchor"),
  play_env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(play=True, goal="anchor"),
  rl_cfg=unitree_g1_tracking_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)

# The contact physics of the 14k torso-anchor runs: self-collision on, friction
# randomised on the feet only.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-SelfCollision",
  env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(
    self_collision=True, friction_all_geoms=False
  ),
  play_env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(
    play=True, self_collision=True, friction_all_geoms=False
  ),
  rl_cfg=unitree_g1_tracking_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)

# Both: the physics and the goal layout of those runs' checkpoints.
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-SuperMimic-SelfCollision-AnchorGoal",
  env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(
    self_collision=True, friction_all_geoms=False, goal="anchor"
  ),
  play_env_cfg=unitree_g1_flat_tracking_supermimic_env_cfg(
    play=True, self_collision=True, friction_all_geoms=False, goal="anchor"
  ),
  rl_cfg=unitree_g1_tracking_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)
