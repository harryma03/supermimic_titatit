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


def test_existing_trunk_boxes_really_produce_contact(trainer):
  env=trainer.env;robot=env.scene['robot']
  pose=robot.data.root_link_pose_w.clone();pose[:,2]=.04
  robot.write_root_link_pose_to_sim(pose)
  env.sim.forward();env.sim.sense()
  assert (env.scene['velocity_body_contact'].data.found>0).any()
  assert (st.robot_state(env)[:,90]>0).all()
  env.reset()


def test_fixed_command_changes_enter_history_on_next_observation(trainer):
  t=trainer;env=t.env
  t.term.cfg.fixed_command=(.5,0.,0.)
  obs,_=env.reset();before=obs['actor'].clone()
  t.term.cfg.fixed_command=(0.,0.,.8)
  obs,_,_,_,_=env.step(torch.zeros(env.num_envs,16,device=env.device))
  # Command terms are term-major, oldest -> newest; no duplicated state frame.
  torch.testing.assert_close(obs['actor'][:,-15:-3],before[:,-12:])
  torch.testing.assert_close(obs['actor'][:,-3:],torch.tensor([[0.,0.,.8]]).expand(env.num_envs,-1))
  t.term.cfg.fixed_command=None


def test_reward_finetune_requires_new_critic_and_preserves_dynamics():
  from go2_supermimic.supermimic.train import validate_velocity_resume
  old={'state_dim':123,'command':{'ranges':[[-1,1]],'rewards':{'sigma_lin':.5}}}
  new=copy.deepcopy(old);new['command']['rewards']['sigma_lin']=.25
  with pytest.raises(ValueError,match='reset-critic'):
    validate_velocity_resume(old,new,False)
  validate_velocity_resume(old,new,True)
  new['state_dim']=78
  with pytest.raises(ValueError,match='contract differs'):
    validate_velocity_resume(old,new,True)
  assert old['command']['rewards']['sigma_lin']==.5


def test_resumed_critic_warmup_is_relative_to_checkpoint(trainer):
  t=trainer;path=t.save(5000)
  old_continue,old_reset=t.cfg.continue_run,t.cfg.reset_critic
  try:
    t.cfg.continue_run=True;t.cfg.reset_critic=True
    t.load(str(path))
    assert t.start_it==5000
    assert t.actor_frozen_until>=5000+t.cfg.critic_warmup_iterations
  finally:
    t.cfg.continue_run=old_continue;t.cfg.reset_critic=old_reset


def test_trace_summary_reports_signed_response():
  from scripts.eval_velocity import summarize_trace
  row={'velocity':torch.tensor([[0.,.05,0.]]),'command':torch.tensor([[0.,.2,0.]]),
       'height':torch.tensor([.36]),'reward_terms':{'track_linear':torch.tensor([1.2])},
       'done':torch.tensor([False])}
  result=summarize_trace([row,row])
  assert result['velocity_mean'][1]==pytest.approx(.05)
  assert result['velocity_rmse'][1]==pytest.approx(.15)
  assert result['reset_count']==0


def test_terrain_config_speed_and_explicit_transfer_contract(trainer):
  from go2_supermimic.supermimic.train import validate_velocity_transfer
  cfg=load_env_cfg('Titatit-Velocity-Terrain')
  assert cfg.commands['velocity'].ranges[0]==(-1.5,1.5)
  assert cfg.scene.terrain.max_init_terrain_level==0
  assert sum(d for _,d in st.obs_terms(cfg))*5==670
  assert set(cfg.scene.terrain.terrain_generator.sub_terrains)=={
    'flat','stairs','stairs_inv','boxes','rough','slope','slope_inv','gap','climb'}
  commands=sample_commands(cfg.commands['velocity'],20000,20000,'cpu')
  assert (commands.abs()<=torch.tensor([1.5,.4,1.])).all()
  assert (commands[:,0].abs()>1.4).any()
  assert cfg.actions==load_env_cfg('Titatit-Velocity-Flat').actions
  old=trainer.velocity_contract();new=copy.deepcopy(old)
  new.update(version=3,terrain={})
  new['terms']=list(new['terms'])+[('height_scan',77)]
  new['command'].update(terrain=True,field_res=.04,cache_dir='logs/heightfields',ranges=cfg.commands['velocity'].ranges)
  validate_velocity_transfer(old,new)
  new['dt']=.01
  with pytest.raises(ValueError,match='contract differs'):
    validate_velocity_transfer(old,new)


def test_real_terrain_transfer_scan_height_bptt_and_eval(trainer, monkeypatch, tmp_path):
  from mjlab.tasks import registry
  from go2_supermimic.target import terrain as tr
  from go2_supermimic.env.velocity_env_cfg import velocity_terrain_boundary
  flat_path=trainer.save(40000)
  cfg=load_env_cfg('Titatit-Velocity-Terrain')
  # Keep all nine real terrain types; shrink the map only for CPU integration.
  cfg.scene.terrain.terrain_generator.num_rows=2
  cfg.scene.terrain.terrain_generator.size=(4.,4.)
  cfg.scene.terrain.terrain_generator.border_width=1.
  cfg.commands['velocity'].field_res=.08
  cfg.commands['velocity'].cache_dir=str(tmp_path/'heightfields')
  original_load=registry.load_env_cfg
  monkeypatch.setattr(registry,'load_env_cfg',lambda *a,**kw: copy.deepcopy(cfg))
  train_cfg=TrainCfg(task='Titatit-Velocity-Terrain',device='cpu',num_envs=9,buffer_len=24,
    wm_hidden=(32,32),actor_hidden=(32,),critic_hidden=(32,),wm_batch=4,wm_updates=1,
    imag_batch=2,critic_epochs=1,critic_minibatches=1,archive_size=8,compile=False,
    log_root=str(tmp_path),resume=str(flat_path),velocity_transfer=True,reset_critic=True)
  t=Trainer(train_cfg)
  try:
    assert t.obs.shape==(9,670) and t.imag.critic_dim==223
    assert t.start_it==0 and t.policy_start>=1500 and t.actor_frozen_until>=2000
    assert t.field is t.term.height_field
    # Adding the scan preserves the old actor's initial function exactly.
    old_actor=trainer.actor
    obs=torch.randn(9,285);extra=torch.randn(9,385)
    torch.testing.assert_close(t.actor(torch.cat([obs,extra],-1)),old_actor(obs))
    env=t.env;terrain=env.scene.terrain
    assert terrain.terrain_origins.shape==(2,9,3)
    assert not terrain.terrain_levels.any()
    assert len(terrain.terrain_types.unique())==9
    env.observation_manager.cfg['actor'].enable_corruption=False
    for name in env.observation_manager.active_terms['actor']:
      env.observation_manager.get_term_cfg('actor',name).noise=None
    obs,_=env.reset();t.obs=obs['actor'];t.a_prev.zero_()
    before=t.obs.clone();a=torch.zeros(9,16)
    obs,_,_,_,_=env.step(a)
    s=st.robot_state(env,t.field);pose=st.planar_pose(env)
    d=env.scene['robot'].data
    torch.testing.assert_close(s[:,0],d.root_link_pos_w[:,2]-tr.ground(pose[:,:2],t.field))
    assert tr.ground(pose[:,:2],t.field).abs().max()>.01
    scan=s[:,:1]+t.imag.relief(pose)
    torch.testing.assert_close(scan,t.term.scan(),atol=1e-5,rtol=1e-5)
    assert t.imag.relief(pose).abs().max()>.01
    frame=observation_frame(s,a,t.term.command,t.enc_bias,height_scan=scan)
    torch.testing.assert_close(obs['actor'],st.push_history(before,frame,5,t.terms),atol=1e-5,rtol=1e-5)
    # Real rewards must use the same ground-relative height as replay.
    measured=env._velocity_reward_state
    torch.testing.assert_close(env._velocity_reward_values['height'],(measured[:,0]-.3622).square())
    t.obs=obs['actor']
    while t.buf.size<12:t.collect()
    assert t.buf.relief.abs().max()>.01
    assert torch.isfinite(torch.tensor(t.fit_world_model()))
    t.it=2001
    stats=t.improve_policy()
    assert stats['actor_grad_norm']>0 and all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    # Height scan must retain gradients through the imagined pose.
    pose=pose.detach().requires_grad_()
    t.imag.scan(s,pose,True).sum().backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
    assert not velocity_terrain_boundary(env).any()
    root=d.root_link_pose_w.clone();root[0,0]=env.scene.env_origins[0,0]+2.
    env.scene['robot'].write_root_link_pose_to_sim(root);env.sim.forward();env.sim.sense()
    assert velocity_terrain_boundary(env)[0]
    env.common_step_counter=12000
    env.reset()
    assert terrain.terrain_levels.max()>0
    path=t.save(1);t.load(str(path))
    from scripts.export_actor import export
    assert export(path,tmp_path/'terrain.onnx')['input']==[1,670]
    # Exercise the real evaluator against the same saved terrain recipe.
    import scripts.eval_velocity as ev
    monkeypatch.setattr(ev,'load_env_cfg',lambda *a,**kw: copy.deepcopy(cfg))
    from argparse import Namespace
    report=ev.evaluate(Namespace(checkpoint=str(path),device='cpu',seed=0,num_envs=9,
      width=320,height=240,suite=False,vx=1.5,vy=0.,yaw=0.,seconds=.04,
      terrain_level=1,viewer=False,video=None,output=None,trace=None))
    assert report['task']=='Titatit-Velocity-Terrain' and report['terrain_level']==1
  finally:
    t.writer.close();t.env.close()
    monkeypatch.setattr(registry,'load_env_cfg',original_load)
