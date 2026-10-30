"""Velocity BPTT with constant replay commands for the 32-step horizon."""
import torch
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.imagine import Imagination
from go2_supermimic.supermimic.velocity import (
  CORE_DIM, STATE_DIM, observation_frame, reward_terms, fall_probability,
)
from go2_supermimic.target import terrain as tr


class VelocityImagination:
  # Share the existing TD(lambda) implementation, without building BallTarget.
  gae_targets = Imagination.gae_targets

  def __init__(self, cfg, consts, rewards, history, terms, noise=True, field=None):
    self.cfg,self.c,self.rewards,self.h,self.terms = cfg,consts,rewards,history,terms
    self.noise = noise
    self.field = field
    self.use_scan = 'height_scan' in dict(terms)
    self.critic_dim = STATE_DIM+3+16+st.CONTEXT_DIM+(tr.SCAN_POINTS if self.use_scan else 0)
    self._step_fn = torch.compile(self._step, dynamic=False) if cfg.compile else self._step

  def relief(self, pose, grad=False):
    return tr.relief(pose[:, :2],pose[:, 2],self.field,grad=grad)

  def scan(self, s, pose, grad=False):
    return s[:, st.Z] + self.relief(pose,grad)

  def critic_input(self, s, commands, previous, xi, pose=None):
    parts = [s,commands,previous,xi]
    if self.use_scan:
      parts.append(self.scan(s,pose))
    return torch.cat(parts,-1)

  def _step(self, wm, actor, s, pose, previous, hist, commands, xi, bias, q, limits):
    a = actor(hist)
    sn, delta = wm(s,a,previous,xi,self.relief(pose,self.cfg.sensor_grad))
    next_pose = st.apply_pose_delta(pose,delta)
    # Wheels have no position target or observed cumulative angle.
    sn = torch.cat([sn[:,:22],torch.zeros_like(sn[:,22:26]),sn[:,26:]],-1)
    values = reward_terms(s,sn,a,previous,commands,q,limits,self.rewards)
    r = sum(self.rewards.weights[n]*v for n,v in values.items())*self.cfg.dt
    scan = self.scan(sn,next_pose,self.cfg.sensor_grad) if self.use_scan else None
    frame = observation_frame(sn,a,commands,bias,noise=self.noise,height_scan=scan)
    return sn,next_pose,a,st.push_history(hist,frame,self.h,self.terms),r,fall_probability(sn)

  def rollout(self,wm,actor,critic,seed,hint_weight=0.):
    s,pose,previous,hist = (seed[k] for k in ('s','pose','a_prev','obs'))
    commands,xi,bias = (seed[k] for k in ('command','xi','enc_bias'))
    ids = seed['env']; q,limits = self.c.default_q[ids],self.c.soft_limits[ids]
    survival = torch.ones_like(s[:,0]); ret = torch.zeros_like(survival)
    rewards,survivals,inputs = [],[survival],[]
    discount = 1.
    for _ in range(self.cfg.horizon):
      inputs.append(self.critic_input(s,commands,previous,xi,pose))
      s,pose,previous,hist,r,d = self._step_fn(wm,actor,s,pose,previous,hist,commands,xi,bias,q,limits)
      ret = ret+discount*survival*r
      survival = survival*(1-d)
      rewards.append(r); survivals.append(survival); discount *= self.cfg.gamma
    inputs.append(self.critic_input(s,commands,previous,xi,pose))
    objective = ret+discount*survival*critic(inputs[-1]).squeeze(-1)
    return objective,dict(rewards=torch.stack(rewards,1),surv=torch.stack(survivals,1),critic_in=torch.stack(inputs,1))
