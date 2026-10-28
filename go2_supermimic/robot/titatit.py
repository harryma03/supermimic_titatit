"""TITATIT wheel-legged robot, using the supplied mjlab MJCF unchanged on disk."""
from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

LEGS = ("FR", "FL", "RR", "RL")
LEG_JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf"))
WHEEL_JOINTS = tuple(f"{leg}_foot_joint" for leg in LEGS)
JOINT_ORDER = LEG_JOINTS + WHEEL_JOINTS
FOOT_GEOM_NAMES = tuple(f"{leg}_foot_collision" for leg in LEGS)
XML = Path(__file__).with_suffix("") / "mujoco/wheeled_titatit_mjlab.xml"
WHEEL_RADIUS = 0.0948
WHEEL_SPEED_LIMIT = 25.0  # rad/s, supplied URDF
# User-supplied leg gains and action scales.
LEG_KP, LEG_KD = 60.0, 1.5
WHEEL_KD = 0.5
LEG_ACTION_SCALE = {".*_hip_joint": 0.125, ".*_thigh_joint": 0.25, ".*_calf_joint": 0.25}
# Original controller zeros wheel position error; foot stiffness=10 contributes no torque.
# Effective wheel control is damping * (action * vel_scale - velocity).
WHEEL_ACTION_SCALE = 5.0


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(XML))
  base = spec.body("base_link")
  base.name = "base"
  for i, geom in enumerate(spec.geoms):
    if geom.contype or geom.conaffinity:
      geom.name = geom.name or f"body_collision{i}"
      geom.group = 3
  for leg in LEGS:
    wheel = spec.body(f"{leg}_foot")
    for geom in wheel.geoms:
      if geom.contype or geom.conaffinity:
        geom.name = f"{leg}_foot_collision"
    # On the axle: rotation about local Y does not move the site.
    wheel.add_site(name=leg, pos=(0, -0.04 if leg[1] == "R" else 0.04, 0), group=5)
  base.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 pos=(1.2, 0.9, 3), dir=(-0.4, -0.3, -1),
                 mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM, castshadow=True,
                 diffuse=(0.45, 0.45, 0.43), specular=(0.05, 0.05, 0.05))
  base.add_site(name="head", pos=(0.50556, 0, 0.016), group=5)
  base.add_site(name="imu", pos=(0, 0, 0), group=5)
  spec.add_sensor(name="imu_ang_vel", type=mujoco.mjtSensor.mjSENS_GYRO,
                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
  return spec


def get_titatit_robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0, 0, 0.50),
      joint_pos={".*L_hip_joint": 0.1, ".*R_hip_joint": -0.1,
                 "F.*_thigh_joint": 1.0, "F.*_calf_joint": -1.5,
                 "R.*_thigh_joint": -0.8, "R.*_calf_joint": 1.5, ".*_foot_joint": 0.0},
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(actuators=(
      IdealPdActuatorCfg(target_names_expr=LEG_JOINTS, stiffness=LEG_KP,
                         damping=LEG_KD, effort_limit=55.0),
      IdealPdActuatorCfg(target_names_expr=WHEEL_JOINTS, stiffness=0.0,
                         damping=WHEEL_KD, effort_limit=10.0),
    ), soft_joint_pos_limit_factor=0.9),
    collisions=(CollisionCfg(geom_names_expr=FOOT_GEOM_NAMES, contype=1, conaffinity=1,
                             condim=6, priority=1, friction=(1.0, 5e-3, 5e-4)),),
  )
