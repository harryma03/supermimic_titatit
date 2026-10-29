"""Explicit staged, mixed-mode horizontal body-frame velocity commands."""
from dataclasses import dataclass, field
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from go2_supermimic.supermimic.velocity import VelocityRewards


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (2., 5.)
  ranges: tuple[tuple[float,float], ...] = ((-1.,1.),(-.4,.4),(-1.,1.))
  # zero, forward/backward, lateral, rotation, mixed
  mode_probs: tuple[float,...] = (.25,.2,.15,.2,.2)
  stage_iterations: tuple[int,...] = (0, 3000, 6000, 10000, 15000)
  rewards: VelocityRewards = field(default_factory=VelocityRewards)
  fixed_command: tuple[float,float,float] | None = None

  def build(self, env):
    return VelocityCommand(self, env)


def sample_commands(cfg, count, iteration, device):
  stage = sum(iteration >= x for x in cfg.stage_iterations)-1
  probs = torch.tensor(cfg.mode_probs,device=device)
  if stage < 2: probs[2:] = 0
  elif stage < 3: probs[4] = 0
  mode = torch.multinomial(probs, count, replacement=True)
  ranges = torch.tensor(cfg.ranges,device=device)
  scale = (.25,.5,.5,1.,1.)[stage]
  cmd = ranges[:,0] + torch.rand(count,3,device=device)*(ranges[:,1]-ranges[:,0])
  cmd *= scale
  if stage == 0: cmd[:,0] = cmd[:,0].abs()
  masks = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],device=device)
  return cmd*masks[mode]


class VelocityCommand(CommandTerm):
  def __init__(self,cfg,env):
    super().__init__(cfg,env)
    self._command = torch.zeros(self.num_envs,3,device=self.device)
    self.iteration = 0

  @property
  def command(self): return self._command

  def _resample_command(self,env_ids):
    if self.cfg.fixed_command is not None:
      self._command[env_ids] = torch.tensor(self.cfg.fixed_command,device=self.device)
    else:
      self._command[env_ids] = sample_commands(self.cfg,len(env_ids),self.iteration,self.device)

  def _update_command(self,env_ids): pass
  def _update_metrics(self): pass
