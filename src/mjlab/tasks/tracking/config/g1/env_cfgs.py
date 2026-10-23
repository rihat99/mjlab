"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.supermimic.recorder import TerminalStateRecorder
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


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
# SuperMimic (world-model policy optimization over a motion library).
##


def unitree_g1_flat_tracking_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Baseline tracking cfg changed to the SuperMimic paper's env (Tables A.1-A.3).

  No pushes, no self-collision penalty, joint-limit weight -2, no end-effector
  termination, and the clip end is a time-out (``motion_end``). Observation groups:
  ``actor`` = 10-frame history of [joint_pos, joint_vel, base_ang_vel,
  projected_gravity, actions] (930, flattened term-major, each term's frames oldest
  first) and ``goal`` = reference frames k_t .. k_t + 9 (670), k_t being the frame
  the next reward is scored against. There is no critic group.
  ``commands.motion.motion_file`` takes a tuple of npz paths (the clip library).
  The ``terminal_state`` recorder keeps the pre-reset state of auto-reset envs.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  cfg.recorders["terminal_state"] = RecorderTermCfg(
    func=TerminalStateRecorder, params={"entity_name": "robot"}
  )

  cfg.events.pop("push_robot", None)
  del cfg.rewards["self_collisions"]
  cfg.rewards["joint_limit"].weight = -2.0
  del cfg.terminations["ee_body_pos"]
  cfg.terminations["motion_end"] = TerminationTermCfg(
    func=mdp.motion_end, params={"command_name": "motion"}, time_out=True
  )

  actor_terms = cfg.observations["actor"].terms
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "joint_pos": actor_terms["joint_pos"],
        "joint_vel": actor_terms["joint_vel"],
        "base_ang_vel": actor_terms["base_ang_vel"],
        "projected_gravity": ObservationTermCfg(
          func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        ),
        "actions": actor_terms["actions"],
      },
      enable_corruption=not play,
      history_length=10,
      flatten_history_dim=True,
    ),
    "goal": ObservationGroupCfg(
      terms={
        "motion_future_frames": ObservationTermCfg(
          func=mdp.motion_future_frames_b,
          params={"command_name": "motion", "future_frames": 10},
        ),
      },
    ),
  }

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # Paper App. A: termination, the anchor rewards and the relative body targets use
  # the root (pelvis), not BeyondMimic's torso_link.
  motion_cmd.anchor_body_name = "pelvis"
  if play:
    # Paper App. F.1 evaluation: friction 1.0, no randomization, no reset noise.
    cfg.events.pop("base_com")
    cfg.events.pop("encoder_bias")
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    motion_cmd.joint_position_range = (0.0, 0.0)
  else:
    motion_cmd.sampling_mode = "return"
    cfg.scene.num_envs = 4096  # Paper Table A.5.

  return cfg
