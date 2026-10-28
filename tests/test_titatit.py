"""TITATIT registration, model, mixed control and 16-action learner checks."""
import mujoco
import numpy as np
import pytest
import torch

import go2_supermimic.env  # registers both robot families
from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg
from go2_supermimic.robot.titatit import get_titatit_robot_cfg, LEG_JOINTS, WHEEL_JOINTS
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.buffer import ReplayBuffer, WindowArchive, pack_generator
from go2_supermimic.supermimic.imagine import ImagineCfg, Imagination
from go2_supermimic.supermimic.train import Actor, Critic
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.target.generator import BallTarget
from go2_supermimic.target.collision import robot_geometry, robot_clearance
from go2_supermimic.target.terrain import SCAN_POINTS


@pytest.mark.parametrize('suffix,dim', [('Flat', 410), ('Obstacles', 490), ('Terrain', 795)])
def test_task_preserves_scene_and_rewards(suffix, dim):
  cfg = load_env_cfg('Titatit-Chase-' + suffix)
  go2 = load_env_cfg('Go2-Chase-' + suffix)
  assert {k: v.weight for k,v in cfg.rewards.items()} == {k: v.weight for k,v in go2.rewards.items()}
  assert cfg.commands['target'].mode_probs == go2.commands['target'].mode_probs
  assert cfg.scene.terrain.terrain_type == go2.scene.terrain.terrain_type
  assert cfg.scene.terrain.terrain_generator == go2.scene.terrain.terrain_generator
  assert sum(d for _, d in st.obs_terms(cfg)) * 5 == dim
  assert len(cfg.actions) == 2 and len(go2.actions) == 1
  assert cfg.actions['joint_pos'].actuator_names == LEG_JOINTS
  assert cfg.actions['wheel_vel'].actuator_names == WHEEL_JOINTS
  assert cfg.actions['wheel_vel'].scale == 5.0
  assert 'quiet_solver_warnings' not in cfg.events


def test_model_preserves_limits_mass_and_default_pose():
  entity = Entity(get_titatit_robot_cfg())
  model = entity.spec.compile()
  assert (model.nq, model.nv, model.nu) == (23, 22, 16)
  np.testing.assert_allclose(model.body_mass[model.body('base').id], 26.085)
  for leg in ('FR','FL','RR','RL'):
    for kind in ('hip','thigh','calf','foot'):
      jid = model.joint(f'{leg}_{kind}_joint').id
      expected = 10 if kind == 'foot' else 55
      np.testing.assert_allclose(model.jnt_actfrcrange[jid], [-expected, expected])
    assert model.site(leg).id >= 0
    assert model.geom(f'{leg}_foot_collision').contype == 1
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  for name, expected in [('FR_hip_joint', -.1), ('FL_hip_joint', .1),
                          ('FR_thigh_joint', 1.), ('RL_thigh_joint', -.8),
                          ('FL_calf_joint', -1.5), ('RR_calf_joint', 1.5)]:
    assert data.qpos[model.jnt_qposadr[model.joint(name).id]] == pytest.approx(expected)
  mujoco.mj_forward(model, data)
  for _ in range(10):
    mujoco.mj_step(model, data)
  assert np.isfinite(data.qpos).all()


@pytest.mark.parametrize('n', [12, 16])
def test_world_model_buffer_and_imagination_backward(n):
  torch.manual_seed(4)
  b, layout = 2, st.StateLayout(n)
  cfg = load_env_cfg(('Go2' if n == 12 else 'Titatit') + '-Chase-Flat')
  terms = st.obs_terms(cfg)
  obs_dim = sum(d for _,d in terms) * 5
  mask = torch.arange(n) < 12
  c = st.RobotConsts(torch.zeros(b,n), torch.ones(b,n),
                     torch.where(mask,60.,0.), torch.where(mask,1.5,.5),
                     torch.where(mask,55.,10.), torch.tensor([-2.,2.]).expand(b,n,2),
                     mask, torch.full((n,),float('inf')))
  wm = WorldModel((32,32), num_joints=n)
  actor = Actor(obs_dim,(32,),n)
  imag = Imagination(ImagineCfg(horizon=2,compile=False),c,cfg.commands['target'].target,5,'cpu',terms)
  critic = Critic(imag.critic_dim,(32,))
  s = torch.zeros(b,layout.state_dim)
  s[:,0] = .4
  s[:,3] = -1.
  s[:,-36:] = torch.randn(b,36) * .05
  a = torch.zeros(b,n)
  xi = torch.zeros(b,4)
  relief = torch.zeros(b,SCAN_POINTS)
  gen = BallTarget(cfg.commands['target'].target,b,'cpu')
  gen.spawn(torch.arange(b),torch.zeros(b,2),torch.zeros(b),torch.zeros(b,2),
             torch.ones(b,dtype=torch.bool),torch.ones(b,dtype=torch.bool))
  buf = ReplayBuffer(b,5,obs_dim,'cpu',num_joints=n)
  for _ in range(4):
    buf.add(s,torch.zeros(b,3),a,a,torch.zeros(b,obs_dim),pack_generator(gen),torch.zeros(b),relief)
  env,idx,valid = buf.sample_windows(b,2)
  windows = dict(s=buf.s[env[:,None],idx],pose=buf.pose[env[:,None],idx],
                 a=buf.a[env[:,None],idx[:,:2]],a_prev=buf.a_prev[env[:,None],idx[:,:2]],
                 xi=xi,valid=valid,relief=buf.relief[env[:,None],idx[:,:2]])
  archive = WindowArchive(4,2,4,'cpu',num_joints=n)
  archive.add(windows)
  assert archive.sample(b)['s'].shape == (b,3,layout.state_dim)
  loss = wm.loss(s,windows['a'],windows['a_prev'],xi,windows['relief'].float(),
                 windows['s'][:,1:],torch.zeros(b,2,3),valid)
  loss.backward()
  assert torch.isfinite(loss)
  seed = dict(s=s,pose=torch.zeros(b,3),a_prev=a,obs=torch.zeros(b,obs_dim),
              xi=xi,enc_bias=a,env=torch.arange(b),gen=pack_generator(gen))
  objective,_ = imag.rollout(wm,actor,critic,seed)
  (-objective.mean()).backward()
  assert torch.isfinite(objective).all()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_titatit_collision_includes_large_trunk_and_wheels():
  cfg=load_env_cfg('Titatit-Chase-Flat').commands['target'].target
  limbs={k: torch.tensor([[[0.,.3,.1]] * 4]) for k in ('hips','knees','feet')}
  geo=robot_geometry(torch.zeros(1,2),torch.tensor([.4]),torch.zeros(1),limbs,cfg)
  assert robot_clearance(torch.tensor([[.49,0.,.4]]),.02,geo).item() < 0
  assert robot_clearance(torch.tensor([[0.,.3,.02]]),.02,geo).item() < 0
