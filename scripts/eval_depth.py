"""Actual simulator evaluation of the depth/RSSM actor, with per-terrain metrics."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import torch
from go2_supermimic.supermimic.depth_train import DepthTrainer,DepthTrainCfg
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.velocity import vertical_yaw_rate


def evaluate(checkpoint,device='cpu',num_envs=9,seconds=10.,vx=.5,vy=0.,yaw=0.,terrain_level=0,
             mask_depth=False,output=None):
  saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
  if saved['cfg']['task']!='Titatit-Velocity-Depth':
    raise ValueError('Use a Depth checkpoint')
  cfg = replace(DepthTrainCfg(**saved['cfg']),device=device,num_envs=num_envs,
    buffer_len=max(32,saved['cfg']['perception_burn']+saved['cfg']['wm_window']+2),
    archive_size=1,resume=checkpoint,depth_transfer=False,continue_run=False,
    reset_critic=False,log_root='/tmp/supermimic-depth-eval',run_name='eval')
  t = DepthTrainer(cfg)
  try:
    env = t.env;terrain = env.scene.terrain
    if not 0<=terrain_level<terrain.terrain_origins.shape[0]:
      raise ValueError('Invalid terrain level')
    # Keep one explicit row across all resets for reproducible comparisons.
    def fixed_level(env,ids,steps):
      terrain.terrain_levels[ids] = terrain_level
      terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids],terrain.terrain_types[ids]]
      return {}
    env.curriculum_manager.get_term_cfg('terrain_levels').func = fixed_level
    t.term.cfg.fixed_command = (vx,vy,yaw)
    obs,_ = env.reset();t.a_prev.zero_();t.reset_perception(obs['actor'])
    if mask_depth:
      t.depth.fill_(1.)
      h,z = t.perception.initial(num_envs,t.dev)
      from go2_supermimic.supermimic.perception import proprio
      t.h,t.z,_ = t.perception.observe(h,z,t.a_prev,proprio(obs['actor']),t.depth,t.fresh,t.first)
      t.obs = torch.cat([obs['actor'],t.h,t.z],-1)
    steps = max(1,round(seconds/env.step_dt))
    linear = torch.zeros(num_envs,device=t.dev);angular=linear.clone();falls=linear.clone()
    for step in range(steps):
      with torch.no_grad():
        action = t.actor(t.obs)
        obs,_,done,timeout,_ = env.step(action)
        # On auto-reset steps omit tracking samples; count actual terminations.
        s = st.robot_state(env,t.field)
        valid = ~(done|timeout)
        linear += (s[:,4:6]-s.new_tensor([vx,vy])).square().sum(-1)*valid
        angular += (vertical_yaw_rate(s)-yaw).square()*valid
        if step == 0: counts = valid.float()
        else: counts += valid
        falls += done
        t.a_prev = torch.where((done|timeout)[:,None],torch.zeros_like(action),action)
        t.advance_perception(obs['actor'],done|timeout,
                             measured_depth=torch.ones_like(t.depth) if mask_depth else None)
    names = list(env.cfg.scene.terrain.terrain_generator.sub_terrains)
    report = dict(checkpoint=checkpoint,command=[vx,vy,yaw],terrain_level=terrain_level,
      mask_depth=mask_depth,seconds=steps*env.step_dt,per_terrain={})
    for index,name in enumerate(names):
      selected = terrain.terrain_types==index
      if selected.any():
        report['per_terrain'][name] = dict(
          linear_rmse=float((linear[selected].sum()/counts[selected].sum().clamp(min=1)).sqrt()),
          yaw_rmse=float((angular[selected].sum()/counts[selected].sum().clamp(min=1)).sqrt()),
          falls=int(falls[selected].sum()),valid_samples=int(counts[selected].sum()))
    if output:Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report
  finally:
    t.writer.close();t.env.close()


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0')
  p.add_argument('--num-envs',type=int,default=9);p.add_argument('--seconds',type=float,default=30.)
  p.add_argument('--vx',type=float,default=.5);p.add_argument('--vy',type=float,default=0.)
  p.add_argument('--yaw',type=float,default=0.);p.add_argument('--terrain-level',type=int,default=0)
  p.add_argument('--mask-depth',action='store_true');p.add_argument('--output')
  args = p.parse_args()
  if args.seconds<=0 or args.num_envs<1:p.error('Positive seconds and num-envs required')
  evaluate(**vars(args))
