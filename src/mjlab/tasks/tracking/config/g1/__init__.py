from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_flat_tracking_vic_env_cfg,
)
from .rl_cfg import unitree_g1_tracking_ppo_runner_cfg

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

# Policy-predicted PD gains (variable impedance control) and its control arm.
# The gain head starts at half the joint head's exploration std.
for _suffix, _gain_mode, _gain_head_init_std in (
  ("Vic-Freq", "freq", 0.5),
  ("Vic-FreqDamp", "freq_damp", 0.5),
  ("NewRew", None, None),
):
  register_mjlab_task(
    task_id=f"Mjlab-Tracking-Flat-Unitree-G1-{_suffix}",
    env_cfg=unitree_g1_flat_tracking_vic_env_cfg(_gain_mode),
    play_env_cfg=unitree_g1_flat_tracking_vic_env_cfg(_gain_mode, play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(_gain_head_init_std),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
