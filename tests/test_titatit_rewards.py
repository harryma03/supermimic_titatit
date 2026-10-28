"""Robot-specific height rewards and diagnostic-only leg/wheel contributions."""
from types import SimpleNamespace as NS

import pytest
import torch
import go2_supermimic.env
from mjlab.tasks.registry import load_env_cfg
from go2_supermimic.env import mdp
from go2_supermimic.target import collision
from go2_supermimic.target.generator import LOW_BAR, HIGH_BAR, BallTarget, BallTargetCfg
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import Imagination, ImagineCfg


def target_state(cfg):
  gen = BallTarget(cfg, 1, 'cpu')
  gen.spawn(torch.arange(1), torch.zeros(1,2), torch.zeros(1), torch.zeros(1,2),
            torch.ones(1,dtype=torch.bool), torch.ones(1,dtype=torch.bool))
  g = gen.state()
  g['dodge_pos'][:] = torch.tensor([.5,0.,.3])
  g['flying'][:] = True
  g['spent'][:] = False
  g['kind'][:] = LOW_BAR
  return g


def test_height_defaults_and_titatit_override():
  for suffix in ('Flat','Obstacles','Terrain'):
    go2 = load_env_cfg('Go2-Chase-' + suffix)
    titatit = load_env_cfg('Titatit-Chase-' + suffix)
    assert go2.commands['target'].target.stand_height == .27
    c = titatit.commands['target'].target
    assert (c.stand_height,c.crouch_min_height,c.base_min_height) == (.37,.23,.18)
    assert go2.rewards['base_height_low'].params == titatit.rewards['base_height_low'].params == {}
    assert {k:v.weight for k,v in go2.rewards.items()} == {k:v.weight for k,v in titatit.rewards.items()}
    assert sum(k.startswith('penalty/') for k in titatit.metrics) == 6


def test_jump_and_duck_use_robot_standing_height():
  cfg = BallTargetCfg(stand_height=.37)
  g = target_state(cfg)
  xy = torch.zeros(1,2)
  def hint(z):
    return collision.dodge_shaping(g,xy,torch.tensor([z]),stand_height=cfg.stand_height).item()
  assert hint(.37) == pytest.approx(0.)
  assert hint(.47) == pytest.approx(.10)
  g['kind'][:] = HIGH_BAR
  assert hint(.37) == pytest.approx(0.)
  assert hint(.32) == pytest.approx(.05)
  # Existing callers retain the Go2 default.
  assert collision.dodge_shaping(g,xy,torch.tensor([.22])).item() == pytest.approx(.05)


def test_environment_and_imagination_share_height_thresholds():
  cfg = BallTargetCfg(stand_height=.43,crouch_min_height=.32,base_min_height=.29)
  g = target_state(cfg)
  n=16
  const = st.RobotConsts(torch.zeros(1,n),torch.ones(1,n),torch.zeros(n),torch.zeros(n),
                         torch.ones(n)*55,torch.tensor([-3.,3.]).expand(1,n,2))
  imag = Imagination(ImagineCfg(),const,cfg,5,'cpu')
  s=torch.zeros(1,78);s[:,0]=.25;s[:,3]=-1
  pose=torch.zeros(1,3)
  k={'default_q':const.default_q,'action_scale':const.action_scale,'soft_limits':const.soft_limits,
     'hint_weight':torch.tensor(1.)}
  _, terms=imag._rewards(s,s,pose,g,torch.zeros(1,n),torch.zeros(1,n),k)
  pos=torch.tensor([[0.,0.,.25]])
  term=NS(cfg=NS(target=cfg),gen=NS(state=lambda:g),base_local=lambda:pos,height_field=None)
  env=NS(command_manager=NS(get_term=lambda name:term,active_terms=['target']),
         scene={'robot':NS(data=NS(root_link_pos_w=pos))})
  torch.testing.assert_close(terms['crouch'],mdp.crouch(env,'target'))
  torch.testing.assert_close(terms['base_height_low'],mdp.base_height_low(env))
  assert terms['crouch'].item() == pytest.approx(.07)
  assert terms['base_height_low'].item() == pytest.approx(.04)
  # Check the actual imagination caller passes stand_height too.
  s[:,0]=.53
  _, terms=imag._rewards(s,s,pose,g,torch.zeros(1,n),torch.zeros(1,n),k)
  assert terms['dodge_hint'].item() == pytest.approx(.10)


@pytest.mark.parametrize('reward', ['action_rate_l2','joint_acc_l2','joint_torques_l2'])
def test_split_penalties_sum_to_original_weighted_term(reward):
  from mjlab.envs import mdp as builtin
  a=torch.tensor([[1.,2.,3.,4.],[2.,3.,4.,5.]])
  data=NS(joint_acc=a*2,actuator_force=a*3)
  weight={'action_rate_l2':-.05,'joint_acc_l2':-2.5e-7,'joint_torques_l2':-2e-5}[reward]
  env=NS(action_manager=NS(action=a,prev_action=torch.zeros_like(a),
         active_terms=['joint_pos','wheel_vel'],get_term=lambda name:NS(action_dim=2)),
         scene={'robot':NS(data=data)},step_dt=.02,cfg=NS(rewards={reward:NS(weight=weight)}))
  # Joint storage is interleaved, actuator and action storage is grouped.
  legs=NS(name='robot',joint_ids=[0,2],actuator_ids=[0,1])
  wheels=NS(name='robot',joint_ids=[1,3],actuator_ids=[2,3])
  split=sum(mdp.joint_penalty_contribution(env,reward,term,asset)
            for term,asset in [('joint_pos',legs),('wheel_vel',wheels)])
  if reward == 'action_rate_l2':
    original=builtin.action_rate_l2(env)
  else:
    original=getattr(builtin,reward)(env,NS(name='robot',joint_ids=slice(None),actuator_ids=slice(None)))
  torch.testing.assert_close(split, original*weight*env.step_dt)
