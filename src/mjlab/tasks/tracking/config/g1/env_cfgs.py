"""Unitree G1 flat tracking environment configurations."""

from typing import Literal

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointGainActionCfg, JointPositionActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


##
# Variable impedance control (policy-predicted PD gains).
##

VIC_TARGET_TRACKING_WEIGHT = -0.1
VIC_GAIN_RATE_WEIGHT = -0.1
VIC_STIFFNESS_WEIGHT = -1e-2
VIC_TORQUE_RATE_WEIGHT = -1e-4


def unitree_g1_flat_tracking_vic_env_cfg(
  gain_mode: Literal["freq", "freq_damp"] | None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Baseline tracking cfg plus the VIC reward set and, optionally, a gain head.

  ``gain_mode=None`` is the control arm: fixed gains, new rewards only.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  robot_cfg = SceneEntityCfg("robot")

  # The joint head keeps its own action-rate term; gain dims get gain_rate_l2.
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=mdp.action_rate_l2, weight=-1e-1, params={"action_name": "joint_pos"}
  )
  cfg.rewards["target_tracking_l2"] = RewardTermCfg(
    func=mdp.target_tracking_l2,
    weight=VIC_TARGET_TRACKING_WEIGHT,
    params={"asset_cfg": robot_cfg},
  )
  cfg.rewards["torque_rate_l2"] = RewardTermCfg(
    func=mdp.actuator_force_rate_l2,
    weight=VIC_TORQUE_RATE_WEIGHT,
    params={"asset_cfg": robot_cfg},
  )
  cfg.metrics = {
    "target_tracking_sq": MetricsTermCfg(
      func=mdp.target_tracking_sq, params={"asset_cfg": robot_cfg}
    ),
    "torque_saturation": MetricsTermCfg(
      func=mdp.torque_saturation, params={"asset_cfg": robot_cfg}, per_substep=True
    ),
  }

  if gain_mode is not None:
    cfg.actions["joint_gain"] = JointGainActionCfg(
      entity_name="robot", actuator_names=(".*",), mode=gain_mode
    )
    gain = {"action_name": "joint_gain"}
    cfg.rewards["gain_rate_l2"] = RewardTermCfg(
      func=mdp.gain_rate_l2, weight=VIC_GAIN_RATE_WEIGHT, params=gain
    )
    cfg.rewards["stiffness_cost"] = RewardTermCfg(
      func=mdp.stiffness_cost, weight=VIC_STIFFNESS_WEIGHT, params=gain
    )
    cfg.metrics["gain_log_m_mean"] = MetricsTermCfg(
      func=mdp.gain_log_m_mean, params=gain
    )
    cfg.metrics["gain_log_m_std"] = MetricsTermCfg(func=mdp.gain_log_m_std, params=gain)
    cfg.metrics["gain_frac_bound"] = MetricsTermCfg(
      func=mdp.gain_frac_bound, params=gain
    )
    if gain_mode == "freq_damp":
      cfg.metrics["gain_log_r_mean"] = MetricsTermCfg(
        func=mdp.gain_log_r_mean, params=gain
      )

  return cfg
