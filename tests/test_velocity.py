"""Velocity contracts, real CPU Warp integration, and BPTT/TD(lambda) regression."""
import copy
import torch
import pytest
import go2_supermimic.env  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.train import Trainer, TrainCfg
from go2_supermimic.supermimic.velocity import (
  STATE_DIM, observation_frame, contact_point_velocity, vertical_yaw_rate, reward_terms,
)
from go2_supermimic.env.velocity_command import sample_commands, VelocityCommandCfg


def test_registration_control_and_observation_contract():
  cfg=load_env_cfg('Titatit-Velocity-Flat')
  chase=load_env_cfg('Titatit-Chase-Flat')
  assert list(cfg.commands)==['velocity']
  assert not any(n in cfg.rewards for n in ('hit','chase_tag','crouch'))
  assert cfg.actions==chase.actions
  assert cfg.scene.entities['robot'].articulation==chase.scene.entities['robot'].articulation
  assert cfg.scene.entities['robot'].spec_fn==chase.scene.entities['robot'].spec_fn
  assert cfg.scene.entities['robot'].collisions[0]==chase.scene.entities['robot'].collisions[0]
  assert cfg.decimation==4 and cfg.sim.mujoco.timestep==.005
  assert sum(n for _,n in st.obs_terms(cfg))*5==285
  assert cfg.commands['velocity'].rewards.height==pytest.approx(.3622)


def test_stratified_commands_ranges_and_curriculum():
  torch.manual_seed(1);cfg=VelocityCommandCfg()
  full=sample_commands(cfg,20000,20000,'cpu')
  assert (full.abs()<=torch.tensor([1.,.4,1.])).all()
  assert (full==0).all(-1).float().mean().item()==pytest.approx(.25,abs=.015)
  for axis,prob in enumerate((.2,.15,.2)):
    pure=(full!=0).sum(-1)==1
    assert (pure & (full[:,axis]!=0)).float().mean().item()==pytest.approx(prob,abs=.015)
  first=sample_commands(cfg,2000,0,'cpu')
  assert (first[:,0]>=0).all() and (first[:,0]<=.25).all()
  assert not first[:,1:].any()
  second=sample_commands(cfg,2000,3000,'cpu')
  assert (second[:,0]<0).any() and not second[:,1:].any()
  third=sample_commands(cfg,2000,6000,'cpu')
  assert ((third!=0).sum(-1)<=1).all() and third[:,1:].any()


def test_slip_is_material_contact_velocity_and_yaw_is_vertical():
  v=torch.tensor([[[1.,0.,0.]]]);w=torch.tensor([[[0.,10.,0.]]])
  origin=torch.tensor([[[0.,0.,.1]]]);point=torch.zeros_like(origin);n=torch.tensor([[[0.,0.,1.]]])
  assert torch.allclose(contact_point_velocity(v,w,origin,point,n),torch.zeros_like(v))
  assert contact_point_velocity(v,torch.zeros_like(w),origin,point,n).square().sum()==1
  s=torch.zeros(1,STATE_DIM);s[:,1:4]=torch.tensor([[0.,-.6,-.8]]);s[:,7:10]=torch.tensor([[0.,.6,.8]])
  assert vertical_yaw_rate(s).item()==pytest.approx(1.)


@pytest.fixture(scope='module')
def trainer(tmp_path_factory):
  # This is a real simulator smoke test, deliberately tiny and on CPU.
  cfg=TrainCfg(task='Titatit-Velocity-Flat',device='cpu',num_envs=2,buffer_len=48,
    wm_hidden=(32,32),actor_hidden=(32,),critic_hidden=(32,),wm_batch=4,wm_updates=1,
    imag_batch=2,critic_epochs=1,critic_minibatches=1,archive_size=8,compile=False,
    log_root=str(tmp_path_factory.mktemp('velocity')),warmup_iterations=0)
  t=Trainer(cfg)
  yield t
  t.writer.close();t.env.close()


def test_real_obs_rewards_replay_and_command_timing(trainer):
  t=trainer;env=t.env
  # Disable corruption for exact simulator vs imagined reconstruction.
  env.observation_manager.cfg['actor'].enable_corruption=False
  for name in env.observation_manager.active_terms['actor']:
    env.observation_manager.get_term_cfg('actor',name).noise=None
  obs,_=env.reset();t.obs=obs['actor'];t.a_prev.zero_()
  before=t.obs.clone();a=torch.randn(2,16)*.05
  previous=env.action_manager.action.clone()
  obs,r,done,_,_=env.step(a)
  assert not done.any()
  s=st.robot_state(env)
  frame=observation_frame(s,a,t.term.command,t.enc_bias,noise=False)
  rebuilt=st.push_history(before,frame,5,t.terms)
  torch.testing.assert_close(obs['actor'],rebuilt,atol=1e-5,rtol=1e-5)
  assert s.shape==(2,STATE_DIM) and torch.isfinite(s).all()
  assert not frame['joint_pos'][:,12:].any()
  # Reward manager and shared reward implementation use the identical snapshot.
  measured=env._velocity_reward_state
  values=reward_terms(measured,measured,a,previous,t.term.command,t.consts.default_q,t.consts.soft_limits,t.term.cfg.rewards)
  expected=sum(t.term.cfg.rewards.weights[k]*v for k,v in values.items())*env.step_dt
  torch.testing.assert_close(r,expected)
  t.obs=obs['actor'];t.a_prev=a
  expected_cmd=t.term.command.clone();expected_obs=t.obs.clone()
  t.collect();slot=(t.buf.ptr-1)%t.buf.L
  torch.testing.assert_close(t.buf.command[:,slot],expected_cmd)
  torch.testing.assert_close(t.buf.obs[:,slot].float(),expected_obs.half().float())
  assert t.buf.gen.shape[-1]==0
  assert ((t.term.time_left>=2-env.step_dt*2)&(t.term.time_left<=5)).all()
  old=t.term.command.clone();t.term.time_left[0]=0.;t.term.time_left[1]=4.
  t.term.compute(.02)
  assert 2<=t.term.time_left[0]<=5 and t.term.time_left[1]==pytest.approx(3.98)
  torch.testing.assert_close(old[1],t.term.command[1])


def test_real_eight_step_wm_and_32step_bptt_critic_and_checkpoint(trainer):
  t=trainer
  while t.buf.size<12:t.collect()
  assert t.wm.net[0].in_features==236 and t.wm.net[-1].out_features==126
  assert t.imag.critic_dim==146
  loss=t.fit_world_model()
  assert torch.isfinite(torch.tensor(loss))
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in t.wm.parameters())
  wm_before=copy.deepcopy(t.wm.state_dict());actor_before=copy.deepcopy(t.actor.state_dict())
  critic_before=copy.deepcopy(t.critic.state_dict());t.it=1
  result=t.improve_policy()
  assert all(torch.isfinite(torch.tensor(x)) for x in result.values())
  assert result['actor_grad_norm']>0
  for p in t.actor.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
  assert any(not torch.equal(v,t.actor.state_dict()[k]) for k,v in actor_before.items())
  assert any(not torch.equal(v,t.critic.state_dict()[k]) for k,v in critic_before.items())
  for k,v in wm_before.items():torch.testing.assert_close(v,t.wm.state_dict()[k],rtol=0,atol=0)
  assert all(p.grad is None for p in t.wm.parameters())
  e,idx=t.buf.sample_states(2)
  seed=dict(env=e,s=t.buf.s[e,idx],pose=t.buf.pose[e,idx],a_prev=t.buf.a_prev[e,idx],
            obs=t.buf.obs[e,idx].float(),command=t.buf.command[e,idx],xi=t.buf.xi[e,idx],enc_bias=t.buf.enc_bias[e,idx])
  obj,data=t.imag.rollout(t.wm,t.actor,t.critic,seed)
  assert obj.shape==(2,) and data['rewards'].shape==(2,32)
  assert data['critic_in'].shape==(2,33,146) and data['surv'].shape==(2,33)
  assert torch.isfinite(obj).all()
  targets=t.imag.gae_targets(t.critic,data)
  assert targets.shape==(2,32) and torch.isfinite(targets).all()
  path=t.save(1);t.load(str(path))
  from scripts.export_actor import export
  result=export(path,t.log_dir/'actor.onnx')
  assert result['input']==[1,285] and result['output']==[1,16]
  assert result['max_abs_error']<1e-5


def test_stage_five_randomization_preserves_replay_boundary(trainer):
  t=trainer;before=t.xi.clone();old_bias=t.buf.enc_bias.clone()
  t._velocity_curriculum(15000)
  assert t.env.velocity_dr_enabled
  assert not torch.equal(t.xi,before)
  assert (t.buf.done[:,(t.buf.ptr-1)%t.buf.L]==1).all()
  torch.testing.assert_close(t.buf.enc_bias,old_bias)
  assert torch.isfinite(t.obs).all()
