"""TITATIT velocity task; same robot/control, independent commands and rewards."""
import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from go2_supermimic.env.titatit_env_cfg import titatit_chase_env_cfg
from go2_supermimic.env.velocity_command import VelocityCommandCfg
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.velocity import reward_terms
from go2_supermimic.robot.titatit import FOOT_GEOM_NAMES


def masked_joint_positions(env, biased=True):
  return st.joint_positions(env, biased)


def velocity_reward(env, name):
  # All reward terms see one snapshot before command resampling and auto reset.
  key = env.common_step_counter
  if getattr(env, '_velocity_reward_step', None) != key:
    s = st.robot_state(env)
    prev = getattr(env, '_velocity_reward_state', s)
    prev = torch.where((env.episode_length_buf <= 1)[:,None], s, prev)
    c = getattr(env, '_velocity_consts', None)
    if c is None:
      c = env._velocity_consts = st.robot_consts(env)
    env._velocity_reward_values = reward_terms(prev,s,env.action_manager.action,
      env.action_manager.prev_action,env.command_manager.get_command('velocity'),
      c.default_q,c.soft_limits,env.cfg.commands['velocity'].rewards)
    env._velocity_reward_state = s.detach().clone()
    env._velocity_reward_step = key
  return env._velocity_reward_values[name]


def titatit_velocity_env_cfg(play=False):
  cfg = titatit_chase_env_cfg(play=play)
  cfg.commands = {'velocity': VelocityCommandCfg()}
  # Re-enable the existing MJCF trunk boxes: the inherited wheel-only
  # CollisionCfg otherwise disables them, making body-contact sensing inert.
  from mjlab.utils.spec_config import CollisionCfg
  robot = cfg.scene.entities['robot']
  robot.collisions += (CollisionCfg(geom_names_expr=('(front|rear)_tita_base_.*',),
    contype=1, conaffinity=1, condim=3, disable_other_geoms=False),)
  terms = cfg.observations['actor'].terms
  terms['velocity'] = terms.pop('target')
  terms['velocity'].params = {'command_name':'velocity'}
  terms['joint_pos'].func = masked_joint_positions
  # Mask unused wheel position slots after corruption through a dedicated term.
  terms['joint_pos'].noise = None
  cfg.rewards = {name: RewardTermCfg(func=velocity_reward,weight=w,params={'name':name})
                 for name,w in cfg.commands['velocity'].rewards.weights.items()}
  cfg.metrics = {}
  cfg.scene.sensors += (
    ContactSensorCfg(name='velocity_wheel_contact',
      primary=ContactMatch(mode='geom',pattern=FOOT_GEOM_NAMES,entity='robot'),
      secondary=ContactMatch(mode='body',pattern='terrain'),
      fields=('found','pos','normal'),reduce='maxforce',num_slots=1),
    ContactSensorCfg(name='velocity_body_contact',
      primary=ContactMatch(mode='body',pattern='base',entity='robot'),
      secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found',)),
  )
  # DR is applied explicitly at stage 5 by Trainer, with fresh replay context.
  for name in ('foot_friction_slide','foot_friction_spin','foot_friction_roll','base_com'):
    cfg.events.pop(name, None)
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  cfg.events['velocity_dr'] = EventTermCfg(mode='startup', func=velocity_domain_randomization,
    params={'feet':SceneEntityCfg('robot',geom_names=FOOT_GEOM_NAMES),
            'base':SceneEntityCfg('robot',body_names=('base',))})
  # Keep random reset yaw, but start from the supplied joint posture at rest.
  cfg.events['reset_robot_joints'].params['position_range'] = (0.,0.)
  cfg.events['reset_robot_joints'].params['velocity_range'] = (0.,0.)
  return cfg

# Called once at the explicit robustness stage; the decorator prepares expanded
# model fields at env construction even before randomization is enabled.
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.envs.mdp import dr

@requires_model_fields('geom_friction', 'body_ipos', recompute=RecomputeLevel.set_const)
def velocity_domain_randomization(env, env_ids, feet, base):
  if not getattr(env, 'velocity_dr_enabled', False):
    return
  dr.geom_friction(env, env_ids, asset_cfg=feet, operation='abs', axes=[0],
                   ranges=(.5,1.3), shared_random=True)
  dr.body_com_offset(env, env_ids, asset_cfg=base, operation='add',
                     ranges={0:(-.015,.015),1:(-.015,.015),2:(-.015,.015)})
