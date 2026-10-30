"""Compact WMP-inspired Gaussian RSSM, trained separately from policy BPTT.

Reference: https://github.com/harryma03/wmp/blob/main/dreamer/models.py
This is an independent implementation, not compatible with WMP checkpoints.
The prior advances at control rate; depth corrects it only on fresh frames.
"""
import torch
from torch import nn
from torch.nn import functional as F
from go2_supermimic.supermimic import state as st

HISTORY_DIM = 285
PROP_DIM = 54


def proprio(history):
  """Latest noisy, biased sensor frame; excludes command and privileged state."""
  dims = (3,3,16,16,16)
  out,offset = [],0
  for dim in dims:
    out.append(history[:,offset+4*dim:offset+5*dim])
    offset += 5*dim
  return torch.cat(out,-1)


class PerceptionRSSM(nn.Module):
  def __init__(self, deter=128, stoch=32):
    super().__init__()
    self.deter,self.stoch = deter,stoch
    self.feature_dim = deter+stoch
    self.encoder = nn.Sequential(nn.Conv2d(1,16,4,2,1),nn.ELU(),
      nn.Conv2d(16,32,4,2,1),nn.ELU(),nn.Conv2d(32,32,4,2,1),nn.ELU(),
      nn.Flatten(),nn.Linear(512,128),nn.ELU())
    self.cell = nn.GRUCell(stoch+16,deter)
    self.prior = nn.Linear(deter,2*stoch)
    self.posterior = nn.Sequential(nn.Linear(deter+128+PROP_DIM,128),nn.ELU(),nn.Linear(128,2*stoch))
    self.depth_decoder = nn.Sequential(nn.Linear(self.feature_dim,256),nn.ELU(),nn.Linear(256,1024))
    self.prop_decoder = nn.Linear(self.feature_dim,PROP_DIM)
    self.scan_decoder = nn.Sequential(nn.Linear(self.feature_dim,128),nn.ELU(),nn.Linear(128,77))

  def initial(self, batch, device):
    return torch.zeros(batch,self.deter,device=device),torch.zeros(batch,self.stoch,device=device)

  @staticmethod
  def distribution(stats):
    mean,raw = stats.chunk(2,-1)
    return mean, F.softplus(raw)+.1

  @staticmethod
  def draw(mean, std, sample):
    return mean+std*torch.randn_like(std) if sample else mean

  def imagine(self,h,z,action,sample=False):
    h = self.cell(torch.cat([z,action],-1),h)
    mean,std = self.distribution(self.prior(h))
    return h,self.draw(mean,std,sample)

  def observe(self,h,z,previous,prop,depth,fresh,first,sample=False):
    keep = (~first).unsqueeze(-1)
    h = self.cell(torch.cat([z*keep,previous*keep],-1),h*keep)
    pm,ps = self.distribution(self.prior(h))
    qm,qs = self.distribution(self.posterior(torch.cat([h,self.encoder(depth),prop],-1)))
    assimilate = (fresh|first).unsqueeze(-1)
    mean,std = torch.where(assimilate,qm,pm),torch.where(assimilate,qs,ps)
    z = self.draw(mean,std,sample)
    kl = (torch.log(ps/qs)+(qs.square()+(qm-pm).square())/(2*ps.square())-.5).mean(-1)
    return h,z,kl*(fresh|first)

  def sequence_loss(self, history, previous, depths, fresh, first, scans, burn):
    h,z = self.initial(history.shape[0],history.device)
    losses = []
    for j in range(history.shape[1]):
      if j < burn:
        with torch.no_grad():
          h,z,_ = self.observe(h,z,previous[:,j],proprio(history[:,j]),depths[:,j],
                               fresh[:,j],first[:,j])
        continue
      prop = proprio(history[:,j])
      h,z,kl = self.observe(h,z,previous[:,j],prop,depths[:,j],fresh[:,j],first[:,j],sample=True)
      feat = torch.cat([h,z],-1)
      # Scale joint velocities independently; all targets come from recorded data.
      scale = prop.new_ones(PROP_DIM);scale[22:38] = 10.
      prop_loss = ((self.prop_decoder(feat)-prop)/scale).square().mean(-1)
      scan_loss = (self.scan_decoder(feat)-scans[:,j]).square().mean(-1)
      image_loss = (self.depth_decoder(feat).sigmoid()-depths[:,j].flatten(1)).square().mean(-1)
      losses.append((prop_loss+scan_loss+image_loss*fresh[:,j]+.01*kl).mean())
    return torch.stack(losses).mean()


class DepthImagination:
  from go2_supermimic.supermimic.imagine import Imagination
  gae_targets = Imagination.gae_targets

  def __init__(self, base, perception):
    self.base,self.perception,self.cfg = base,perception,base.cfg
    self.base.use_scan = True
    self.critic_dim = 223+perception.feature_dim

  def rollout(self,wm,actor,critic,seed,hint_weight=0.):
    from go2_supermimic.supermimic.velocity import observation_frame,reward_terms,fall_probability
    s,pose,previous,hist = (seed[k] for k in ('s','pose','a_prev','history'))
    h,z = seed['latent']
    commands,xi,bias = (seed[k] for k in ('command','xi','enc_bias'))
    q,limits = self.base.c.default_q[seed['env']],self.base.c.soft_limits[seed['env']]
    survival = torch.ones_like(s[:,0]);ret = torch.zeros_like(survival)
    rewards,survivals,inputs = [],[survival],[]
    discount = 1.
    for _ in range(self.cfg.horizon):
      feature = torch.cat([h,z],-1)
      inputs.append(torch.cat([self.base.critic_input(s,commands,previous,xi,pose),feature],-1))
      a = actor(torch.cat([hist,feature],-1))
      sn,delta = wm(s,a,previous,xi,self.base.relief(pose,self.cfg.sensor_grad))
      sn = torch.cat([sn[:,:22],torch.zeros_like(sn[:,22:26]),sn[:,26:]],-1)
      values = reward_terms(s,sn,a,previous,commands,q,limits,self.base.rewards)
      r = sum(self.base.rewards.weights[n]*v for n,v in values.items())*self.cfg.dt
      # No future image or oracle scan enters the policy's recurrent state.
      h,z = self.perception.imagine(h,z,a)
      frame = observation_frame(sn,a,commands,bias,noise=self.base.noise)
      hist = st.push_history(hist,frame,self.base.h,self.base.terms)
      s,pose,previous = sn,st.apply_pose_delta(pose,delta),a
      ret = ret+discount*survival*r
      survival = survival*(1-fall_probability(s))
      rewards.append(r);survivals.append(survival);discount *= self.cfg.gamma
    inputs.append(torch.cat([self.base.critic_input(s,commands,previous,xi,pose),h,z],-1))
    return ret+discount*survival*critic(inputs[-1]).squeeze(-1),dict(
      rewards=torch.stack(rewards,1),surv=torch.stack(survivals,1),critic_in=torch.stack(inputs,1))
