"""Sensor fidelity, RSSM reset/timing, and real SuperMimic update regression."""
import copy
import torch
import pytest
from go2_supermimic.supermimic.depth_train import DepthTrainCfg,DepthTrainer
from go2_supermimic.supermimic.perception import PerceptionRSSM,proprio
from go2_supermimic.env.depth_env_cfg import read_depth


def test_rssm_reset_camera_hold_and_action_gradients():
  torch.manual_seed(9)
  model=PerceptionRSSM(16,8)
  h,z=model.initial(2,'cpu')
  action=torch.randn(2,16);prop=torch.randn(2,54)
  image=torch.rand(2,1,32,32);yes=torch.ones(2,dtype=torch.bool);no=~yes
  expected=model.observe(h,z,action,prop,image,yes,yes)
  actual=model.observe(h+5,z-3,action+8,prop,image,yes,yes)
  for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
  held=model.observe(h,z,action,prop,image,no,no)
  changed=model.observe(h,z,action,prop,1-image,no,no)
  torch.testing.assert_close(held[1],changed[1])
  fresh=model.observe(h,z,action,prop,1-image,yes,no)
  assert not torch.equal(fresh[1],held[1])
  action.requires_grad_();hn,zn=model.imagine(h,z,action)
  (hn.square().mean()+zn.square().mean()).backward()
  assert action.grad.abs().sum()>0 and action.grad.isfinite().all()


@pytest.fixture
def depth_trainer(monkeypatch,tmp_path):
  from mjlab.tasks import registry
  cfg=registry.load_env_cfg('Titatit-Velocity-Depth')
  cfg.scene.terrain.terrain_generator.size=(4.,4.)
  cfg.scene.terrain.terrain_generator.num_rows=2
  cfg.scene.terrain.terrain_generator.border_width=1.
  cfg.commands['velocity'].field_res=.08
  cfg.commands['velocity'].cache_dir=str(tmp_path/'heightfields')
  monkeypatch.setattr(registry,'load_env_cfg',lambda *a,**kw:copy.deepcopy(cfg))
  tc=DepthTrainCfg(device='cpu',num_envs=9,buffer_len=32,wm_hidden=(32,32),
    actor_hidden=(32,),critic_hidden=(32,),wm_batch=4,wm_updates=1,imag_batch=2,
    critic_epochs=1,critic_minibatches=1,archive_size=4,rssm_deter=16,rssm_stoch=8,
    perception_batch=2,perception_burn=4,log_root=str(tmp_path))
  t=DepthTrainer(tc)
  yield t
  t.writer.close();t.env.close()


def test_real_depth_geometry_timing_training_and_checkpoint(depth_trainer,monkeypatch):
  t=depth_trainer;env=t.env
  assert t.obs.shape==(9,309) and 'height_scan' not in dict(t.terms)
  assert t.imag.critic_dim==247
  depth=read_depth(env)
  assert depth.shape==(9,1,32,32) and depth.isfinite().all()
  assert depth.min()<.9 and depth.max()<=1
  # Actual camera ray hits must match independent MuJoCo rays at several pixels.
  import mujoco
  import numpy as np
  from mjlab.sensor import PinholeCameraPatternCfg
  from go2_supermimic.env.depth_env_cfg import DEPTH_FOV
  sensor=env.scene['forward_depth'];model=env.sim.mj_model
  data=mujoco.MjData(model)
  data.qpos[:]=env.sim.data.qpos[0].cpu().numpy()
  mujoco.mj_forward(model,data)
  site=model.site('robot/depth_camera').id
  _,directions=PinholeCameraPatternCfg(width=32,height=32,fovy=DEPTH_FOV).generate_rays(model,'cpu')
  groups=np.array([1,0,0,0,0,0],dtype=np.uint8)
  for pixel in (0,512,1023):
    direction=data.site_xmat[site].reshape(3,3)@directions[pixel].numpy()
    distance=mujoco.mj_ray(model,data,data.site_xpos[site],direction,groups,1,-1,np.zeros(1,dtype=np.int32))
    observed=float(sensor.data.distances[0,pixel])
    if 0<=distance<=3: assert observed==pytest.approx(distance,abs=2e-4)
    else: assert observed<0
  # Holding 4 steps then updating on the fifth; reset always forces a new frame.
  held=t.depth.clone()
  for _ in range(4):
    t.collect();assert not t.fresh.any();torch.testing.assert_close(t.depth,held)
  t.collect();assert t.fresh.all()
  done=torch.zeros(9,dtype=torch.bool);done[0]=True
  t.a_prev[0]=0
  t.advance_perception(t.obs[:,:285],done)
  assert t.first[0] and t.fresh[0] and not t.first[1:].any()
  while t.buf.size<16:t.collect()
  before={k:v.detach().clone() for k,v in t.perception.state_dict().items()}
  assert torch.isfinite(torch.tensor(t.fit_world_model()))
  assert any(not torch.equal(v,t.perception.state_dict()[k]) for k,v in before.items())
  frozen={k:v.detach().clone() for k,v in t.perception.state_dict().items()}
  actor_before={k:v.detach().clone() for k,v in t.actor.state_dict().items()}
  t.it=2001
  result=t.improve_policy()
  assert result['actor_grad_norm']>0 and all(torch.isfinite(torch.tensor(v)) for v in result.values())
  assert any(not torch.equal(v,t.actor.state_dict()[k]) for k,v in actor_before.items())
  for k,v in frozen.items():torch.testing.assert_close(t.perception.state_dict()[k],v)
  assert all(p.grad is None for p in t.perception.parameters())
  seed=t.policy_seed()
  # Replay seeds recompute the latent from frames: no cached feature dependence.
  assert seed['history'].shape==(2,285) and seed['latent'][0].shape==(2,16)
  path=t.save(3);t.cfg.reset_critic=False;t.load(str(path))
  t._velocity_curriculum(15000)
  assert t.first.all() and t.obs.shape==(9,309)
  from scripts.eval_depth import evaluate
  report=evaluate(str(path),device='cpu',num_envs=9,seconds=.04,terrain_level=1,mask_depth=True)
  assert len(report['per_terrain'])==9
