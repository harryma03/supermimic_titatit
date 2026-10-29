"""Shared velocity observation/reward contract; no target objects.

Auxiliary dynamics are measured in simulation and predicted by the world model.
Slip is the material velocity at the strongest wheel/ground contact, not axle
velocity or a wheel-speed mismatch. Flat stationary ground only.
"""
from dataclasses import dataclass, field
import math
import torch
from go2_supermimic.supermimic import state as st

CORE_DIM = 78
SLIP = slice(78, 86)       # four contact velocities, horizontal yaw frame, m/s
CONTACT = slice(86, 90)    # four wheel contact indicators
BODY_CONTACT = slice(90, 91)
TORQUE = slice(91, 107)    # generalized actuator torque in action order, Nm
ACC = slice(107, 123)      # joint acceleration, rad/s^2
STATE_DIM = 123
EXTRA_DIM = STATE_DIM - CORE_DIM
LAYOUT = st.StateLayout(16)


@dataclass
class VelocityRewards:
  sigma_lin: float = 0.5
  sigma_yaw: float = 0.5
  height: float = 0.3622  # scripts/calibrate_velocity_height.py, loaded standing at 50 Hz
  wheel_speed_limit: float = 25.0
  weights: dict[str, float] = field(default_factory=lambda: {
    'track_linear': 1.5, 'track_yaw': 1.0, 'upright': 0.5,
    'tilt_rate': -0.05, 'height': -10.0, 'vertical_speed': -0.5,
    'action_rate': -0.05, 'joint_acc': -2.5e-7, 'joint_torque': -2e-5,
    'joint_limits': -1.0, 'leg_deviation': -0.1,
    'wheel_slip': -0.2, 'wheel_speed': -0.01, 'body_contact': -2.0,
    'contact_change': -0.02, 'stand_motion': -1.0, 'fall': -200.0,
  })


def vertical_yaw_rate(s):
  # Body gyro projected onto world vertical, not body z when tilted.
  g = torch.nn.functional.normalize(s[:, st.GRAV], dim=-1)
  return -(s[:, st.W_B] * g).sum(-1)


def fall_probability(s):
  gz = (-torch.nn.functional.normalize(s[:, st.GRAV], dim=-1)[:, 2]).clamp(-1+1e-4, 1-1e-4)
  return torch.sigmoid((torch.acos(gz) - math.radians(70)) / math.radians(5))


def reward_terms(s0, s, a, previous, commands, default_q, limits, cfg):
  """Identical SI units and smooth formulas in real and imagined rewards."""
  q = s[:, LAYOUT.q] + default_q
  contact = s[:, CONTACT].clamp(0, 1)
  zero = (commands == 0).all(-1).to(s.dtype)
  return {
    'track_linear': torch.exp(-(s[:, 4:6]-commands[:, :2]).square().sum(-1)/cfg.sigma_lin**2),
    'track_yaw': torch.exp(-(vertical_yaw_rate(s)-commands[:, 2]).square()/cfg.sigma_yaw**2),
    'upright': torch.exp(-s[:, 1:3].square().sum(-1)/0.4**2),
    'tilt_rate': s[:, 7:9].square().sum(-1),
    'height': (s[:, 0]-cfg.height).square(),
    'vertical_speed': s[:, 6].square(),
    'action_rate': (a-previous).square().sum(-1),
    'joint_acc': s[:, ACC].square().sum(-1),
    'joint_torque': s[:, TORQUE].square().sum(-1),
    'joint_limits': ((limits[:, :12, 0]-q[:, :12]).relu() + (q[:, :12]-limits[:, :12, 1]).relu()).sum(-1),
    'leg_deviation': s[:, 10:22].square().sum(-1),
    'wheel_slip': (s[:, SLIP].reshape(-1,4,2).square().sum(-1)*contact).sum(-1),
    'wheel_speed': (s[:, LAYOUT.qd][:, 12:].abs()-cfg.wheel_speed_limit).relu().square().sum(-1),
    'body_contact': s[:, BODY_CONTACT].squeeze(-1).clamp(0, 1),
    'contact_change': (contact-s0[:, CONTACT].clamp(0,1)).square().sum(-1),
    'stand_motion': zero*(s[:, 4:6].square().sum(-1)+vertical_yaw_rate(s).square()+0.01*s[:, LAYOUT.qd].square().sum(-1)),
    'fall': fall_probability(s),
  }


def observation_frame(s, previous, commands, bias, noise=False):
  frame = st.obs_frame(s[:, :CORE_DIM], previous, commands, bias, noise=False, command_name="velocity")
  if noise:
    for name in ('base_ang_vel','projected_gravity','joint_vel'):
      frame[name] = frame[name] + (2*torch.rand_like(frame[name])-1)*st.OBS_NOISE[name]
  # Continuous wheel angles AND encoder/noise on those unused slots are masked.
  frame['joint_pos'] = torch.cat([frame['joint_pos'][:, :12], torch.zeros_like(previous[:, 12:])], -1)
  return frame


def contact_point_velocity(linear, angular, origin, point, normal):
  v = linear + torch.cross(angular, point-origin, dim=-1)
  return v - (v*normal).sum(-1, keepdim=True)*normal


def auxiliary_state(env):
  robot = env.scene['robot']; d = robot.data
  sensor = env.scene['velocity_wheel_contact']
  # Sensor expansion follows model order; explicitly resolve by name.
  names = sensor.primary_names
  order = [next(i for i,n in enumerate(names) if n.split('/')[-1] == f'{leg}_foot_collision') for leg in st.LEGS]
  cs = sensor.data
  ids, _ = robot.find_bodies(tuple(f'{leg}_foot' for leg in st.LEGS), preserve_order=True)
  found = (cs.found[:, order] > 0).float()
  v = contact_point_velocity(d.body_link_lin_vel_w[:, ids], d.body_link_ang_vel_w[:, ids],
                            d.body_link_pos_w[:, ids], cs.pos[:, order], cs.normal[:, order])
  v = v * found.unsqueeze(-1)
  v = torch.stack([st.yaw_rotate_inverse(v[:,i], d.heading_w)[:, :2] for i in range(4)], 1)
  body = (env.scene['velocity_body_contact'].data.found > 0).any(-1, keepdim=True).float()
  ids = st.action_joint_ids(env)
  return torch.cat([v.flatten(1), found, body, d.qfrc_actuator[:, ids], d.joint_acc[:, ids]], -1)
