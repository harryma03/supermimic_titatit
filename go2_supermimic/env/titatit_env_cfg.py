"""The same three chase/dodge tasks with TITATIT and mixed leg/wheel control."""
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from go2_supermimic.env.env_cfg import go2_chase_env_cfg
from go2_supermimic.env import mdp
from go2_supermimic.robot.titatit import (
  LEG_JOINTS, WHEEL_JOINTS, LEG_ACTION_SCALE, WHEEL_ACTION_SCALE,
  WHEEL_RADIUS, get_titatit_robot_cfg,
)
from go2_supermimic.supermimic.state import joint_positions, joint_velocities


def titatit_chase_env_cfg(play=False, terrain=False, obstacles=False):
  cfg = go2_chase_env_cfg(play=play, terrain=terrain, obstacles=obstacles)
  cfg.scene.entities["robot"] = get_titatit_robot_cfg()
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=LEG_JOINTS,
                                         scale=LEG_ACTION_SCALE),
    "wheel_vel": JointVelocityActionCfg(entity_name="robot", actuator_names=WHEEL_JOINTS,
                                         scale=WHEEL_ACTION_SCALE),
  }
  # This optional warning API is absent in mjlab 1.6's mujoco-warp 3.11 dependency.
  cfg.events.pop("quiet_solver_warnings", None)
  terms = cfg.observations["actor"].terms
  terms["joint_pos"].func = joint_positions
  terms["joint_vel"].func = joint_velocities
  cfg.rewards["joint_deviation_l2"].func = mdp.leg_joint_deviation_l2
  cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=LEG_JOINTS)
  target = cfg.commands["target"].target
  # Initial kinematic estimate; calibrate against settled standing height.
  target.stand_height = 0.37
  target.head_offset = 0.50556
  target.trunk_bounds = (-0.505375, 0.50556, -0.15, 0.15, -0.079, 0.111)
  target.thigh_radius = 0.05
  target.calf_radius = 0.043
  target.wheel_radius = WHEEL_RADIUS
  cfg.rewards["hard_landing"].params["foot_radius"] = WHEEL_RADIUS
  for family, action_term, joints in (("legs", "joint_pos", LEG_JOINTS),
                                       ("wheels", "wheel_vel", WHEEL_JOINTS)):
    for reward in ("action_rate_l2", "joint_acc_l2", "joint_torques_l2"):
      cfg.metrics[f"penalty/{reward}/{family}"] = MetricsTermCfg(
        func=mdp.joint_penalty_contribution,
        params={"reward_name": reward, "action_term": action_term,
                "asset_cfg": SceneEntityCfg("robot", joint_names=joints, actuator_names=joints)},
      )
  return cfg
