from typing import Literal

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
# The gain head explores at half the joint head's initial std, held fixed: a
# learnable gain-head std grew without bound under the entropy bonus once the
# squashed output saturated.
_VIC_TASKS: tuple[
  tuple[str, Literal["freq", "freq_damp"] | None, float | None, bool], ...
] = (
  ("Vic-Freq", "freq", 0.5, True),
  ("Vic-FreqDamp", "freq_damp", 0.5, True),
  ("NewRew", None, None, False),
)
for _suffix, _gain_mode, _gain_head_init_std, _gain_head_std_fixed in _VIC_TASKS:
  register_mjlab_task(
    task_id=f"Mjlab-Tracking-Flat-Unitree-G1-{_suffix}",
    env_cfg=unitree_g1_flat_tracking_vic_env_cfg(_gain_mode),
    play_env_cfg=unitree_g1_flat_tracking_vic_env_cfg(_gain_mode, play=True),
    rl_cfg=unitree_g1_tracking_ppo_runner_cfg(
      _gain_head_init_std, _gain_head_std_fixed
    ),
    runner_cls=MotionTrackingOnPolicyRunner,
  )
