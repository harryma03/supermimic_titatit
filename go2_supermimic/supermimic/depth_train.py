"""WMP-inspired perception with the existing SuperMimic dynamics/BPTT learner."""
import json
from dataclasses import dataclass,asdict,replace
import torch
from go2_supermimic.env.depth_env_cfg import read_depth,depth_contract,DEPTH_INTERVAL,DEPTH_SIZE
from go2_supermimic.supermimic.train import Trainer,TrainCfg,Actor,Critic,clip_grads,lr_at
from go2_supermimic.supermimic.buffer import ReplayBuffer
from go2_supermimic.supermimic.perception import PerceptionRSSM,DepthImagination,proprio,HISTORY_DIM
from go2_supermimic.supermimic.velocity import EXTRA_DIM


@dataclass(kw_only=True)
class DepthTrainCfg(TrainCfg):
  task: str = 'Titatit-Velocity-Depth'
  num_envs: int = 256
  buffer_len: int = 512
  wm_batch: int = 1024
  imag_batch: int = 2048
  archive_size: int = 20000
  compile: bool = False
  warmup_iterations: int = 1500
  reset_critic: bool = True
  actor_lr_scale: float = .25
  terrain_curriculum_iterations: int = 20000
  depth_transfer: bool = False
  rssm_deter: int = 128
  rssm_stoch: int = 32
  perception_batch: int = 32
  perception_burn: int = 16


class DepthTrainer(Trainer):
  def __init__(self,cfg):
    if cfg.task != 'Titatit-Velocity-Depth' or cfg.velocity_transfer or cfg.collect_steps != 1:
      raise ValueError('Use Depth task, collect-steps 1 and depth-transfer instead of velocity-transfer')
    if cfg.compile:
      raise ValueError('Depth BPTT currently requires --compile false')
    if cfg.perception_burn < 1 or cfg.buffer_len <= cfg.perception_burn+cfg.wm_window+1:
      raise ValueError('Replay must fit perception burn-in plus a full training window')
    if min(cfg.rssm_deter,cfg.rssm_stoch,cfg.perception_batch) < 1:
      raise ValueError('RSSM dimensions and perception batch must be positive')
    if cfg.depth_transfer and (not cfg.resume or cfg.continue_run or not cfg.reset_critic):
      raise ValueError('depth-transfer requires resume, reset-critic true, continue-run false')
    super().__init__(replace(cfg,resume=''))
    self.cfg = cfg
    self.perception = PerceptionRSSM(cfg.rssm_deter,cfg.rssm_stoch).to(self.dev)
    self.opt_perception = torch.optim.Adam(self.perception.parameters(),lr=cfg.lr)
    self.actor = Actor(HISTORY_DIM+self.perception.feature_dim,cfg.actor_hidden,16).to(self.dev)
    self.opt_actor = torch.optim.Adam(self.actor.parameters(),lr=cfg.lr,betas=(.9,.95))
    self.imag = DepthImagination(self.imag,self.perception)
    self.critic = Critic(self.imag.critic_dim,cfg.critic_hidden).to(self.dev)
    self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=cfg.lr)
    self.buf = ReplayBuffer(cfg.num_envs,cfg.buffer_len,HISTORY_DIM+self.perception.feature_dim,
      self.dev,gen_dim=0,num_joints=16,extra_state_dim=EXTRA_DIM,command_dim=3)
    self.depth_frames = torch.zeros(cfg.num_envs,cfg.buffer_len,1,DEPTH_SIZE,DEPTH_SIZE,
                                    dtype=torch.uint8,device=self.dev)
    self.fresh_frames = torch.zeros(cfg.num_envs,cfg.buffer_len,dtype=torch.bool,device=self.dev)
    self.first_frames = torch.zeros_like(self.fresh_frames)
    self.policy_start = max(self.policy_start,cfg.perception_burn+cfg.wm_window+2)
    self.actor_frozen_until = self.policy_start+cfg.critic_warmup_iterations
    if cfg.resume:
      self.load(cfg.resume)
    self.reset_perception(self.obs[:,:HISTORY_DIM])
    (self.log_dir/'config.json').write_text(json.dumps(
      {'train':asdict(cfg),'velocity_contract':self.velocity_contract()},indent=2))

  def velocity_contract(self):
    contract = super().velocity_contract()
    contract.update(version=4,depth=depth_contract(),
      perception={'type':'gaussian_rssm_v1','deter':self.cfg.rssm_deter,
                  'stoch':self.cfg.rssm_stoch,'history_dim':HISTORY_DIM,
                  'actor_input':'proprio_history_then_deter_then_stoch'})
    return contract

  @torch.no_grad()
  def reset_perception(self,history):
    self.h,self.z = self.perception.initial(self.cfg.num_envs,self.dev)
    self.age = torch.zeros(self.cfg.num_envs,dtype=torch.long,device=self.dev)
    self.fresh = torch.ones_like(self.age,dtype=torch.bool)
    self.first = self.fresh.clone()
    self.depth = read_depth(self.env)
    self.h,self.z,_ = self.perception.observe(self.h,self.z,self.a_prev,proprio(history),
                                             self.depth,self.fresh,self.first)
    self.obs = torch.cat([history,self.h,self.z],-1)

  @torch.no_grad()
  def advance_perception(self,history,done,measured_depth=None):
    self.age += 1
    self.fresh = (self.age>=DEPTH_INTERVAL)|done
    self.first = done.clone()
    measured_depth = read_depth(self.env) if measured_depth is None else measured_depth
    self.depth = torch.where(self.fresh[:,None,None,None],measured_depth,self.depth)
    self.age[self.fresh] = 0
    self.h,self.z,_ = self.perception.observe(self.h,self.z,self.a_prev,proprio(history),
                                             self.depth,self.fresh,self.first)
    self.obs = torch.cat([history,self.h,self.z],-1)

  @torch.no_grad()
  def collect(self):
    slot = self.buf.ptr
    self.depth_frames[:,slot] = (self.depth*255).round().to(torch.uint8)
    self.fresh_frames[:,slot],self.first_frames[:,slot] = self.fresh,self.first
    super().collect()
    self.advance_perception(self.obs,self.buf.done[:,slot].bool())

  def _velocity_curriculum(self,it):
    before = getattr(self.env,'velocity_dr_enabled',False)
    super()._velocity_curriculum(it)
    if not before and getattr(self.env,'velocity_dr_enabled',False):
      self.reset_perception(self.obs)

  def perception_window(self,batch,length):
    env,idx,_ = self.buf.sample_windows(batch,length-1)
    e = env[:,None]
    return env,idx,dict(history=self.buf.obs[e,idx,:HISTORY_DIM].float(),
      previous=self.buf.a_prev[e,idx],depths=self.depth_frames[e,idx].float()/255,
      fresh=self.fresh_frames[e,idx],first=self.first_frames[e,idx],
      scans=self.buf.s[e,idx,:1]+self.buf.relief[e,idx].float())

  def fit_world_model(self):
    loss = super().fit_world_model()
    length = self.cfg.perception_burn+self.cfg.wm_window
    if self.buf.size>length:
      _,_,window = self.perception_window(self.cfg.perception_batch,length)
      perception_loss = self.perception.sequence_loss(**window,burn=self.cfg.perception_burn)
      for group in self.opt_perception.param_groups:
        group['lr'] = lr_at(self.cfg,getattr(self,'it',0))*self.lr_scale
      self.opt_perception.zero_grad(set_to_none=True)
      perception_loss.backward()
      norm = clip_grads(self.perception.parameters(),self.cfg.grad_clip,float('inf'))
      if torch.isfinite(norm) and torch.isfinite(perception_loss):
        self.opt_perception.step()
      else:
        self.nonfinite_steps += 1
      self.ep_stats['Metrics/perception/loss'].append(float(perception_loss.detach()))
    return loss

  @torch.no_grad()
  def policy_seed(self):
    env,idx,w = self.perception_window(self.cfg.imag_batch,self.cfg.perception_burn+1)
    h,z = self.perception.initial(len(env),self.dev)
    for j in range(idx.shape[1]):
      h,z,_ = self.perception.observe(h,z,w['previous'][:,j],proprio(w['history'][:,j]),
        w['depths'][:,j],w['fresh'][:,j],w['first'][:,j])
    t = idx[:,-1];b = self.buf
    return dict(env=env,s=b.s[env,t],pose=b.pose[env,t],a_prev=b.a_prev[env,t],
      history=w['history'][:,-1],latent=(h,z),command=b.command[env,t],
      xi=b.xi[env,t],enc_bias=b.enc_bias[env,t])

  def policy_models(self):
    return self.wm,self.perception

  def checkpoint_extra(self):
    return dict(perception=self.perception.state_dict(),opt_perception=self.opt_perception.state_dict())

  def load(self,path):
    checkpoint = torch.load(path,map_location=self.dev,weights_only=False)
    task = checkpoint['cfg']['task']
    if task == self.cfg.task:
      if checkpoint.get('velocity_contract') != self.velocity_contract():
        raise ValueError('Depth sensor/perception/dynamics contract differs')
      self.actor.load_state_dict(checkpoint['actor'])
      self.perception.load_state_dict(checkpoint['perception'])
      self.wm.load_state_dict(checkpoint['world_model'])
      self.opt_actor.load_state_dict(checkpoint['opt_actor'])
      self.opt_perception.load_state_dict(checkpoint['opt_perception'])
      self.opt_wm.load_state_dict(checkpoint['opt_wm'])
      if not self.cfg.reset_critic:
        self.critic.load_state_dict(checkpoint['critic'])
        self.opt_critic.load_state_dict(checkpoint['opt_critic'])
      if self.cfg.continue_run:
        self.start_it = int(checkpoint['iteration'])
        self.env.common_step_counter = self.start_it
        if self.cfg.reset_critic:
          self.actor_frozen_until = self.start_it+self.cfg.critic_warmup_iterations
    else:
      if not self.cfg.depth_transfer or task not in ('Titatit-Velocity-Flat','Titatit-Velocity-Terrain'):
        raise ValueError('Use --depth-transfer true for Flat/Terrain initialization')
      old = checkpoint.get('velocity_contract',{})
      new = self.velocity_contract()
      if old.get('version') != (2 if task.endswith('Flat') else 3):
        raise ValueError('Expected Flat v2 or Terrain v3 checkpoint')
      for key in ('trunk_contacts','state_dim','history','dt'):
        if old.get(key)!=new[key]:
          raise ValueError(f'Incompatible velocity dynamics: {key}')
      if tuple(tuple(t) for t in old['terms'][:6]) != tuple(new['terms']):
        raise ValueError('Incompatible proprioceptive observation terms')
      self.wm.load_state_dict(checkpoint['world_model'])
      sd = self.actor.state_dict();source = checkpoint['actor']
      for key,value in source.items():
        if key not in ('net.0.weight','norm.mean','norm.var'):
          sd[key] = value.clone()
      sd['net.0.weight'].zero_()
      sd['net.0.weight'][:,:HISTORY_DIM] = source['net.0.weight'][:,:HISTORY_DIM]
      for key in ('norm.mean','norm.var'):
        sd[key][:HISTORY_DIM] = source[key][:HISTORY_DIM]
      self.actor.load_state_dict(sd)
      print('Depth transfer: proprio weights/WM loaded; new RSSM, critic and feature weights',flush=True)
    print(f'Depth checkpoint loaded: {path}; curriculum starts at {self.start_it}',flush=True)
