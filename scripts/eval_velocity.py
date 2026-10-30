"""Simulator metrics, held-command imagination, and open-loop dynamics errors."""
import argparse
import json
from pathlib import Path
import torch
import go2_supermimic.env  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.train import Actor, Critic
from go2_supermimic.supermimic.world_model import WorldModel
from go2_supermimic.supermimic.imagine import ImagineCfg
from go2_supermimic.supermimic.velocity_imagine import VelocityImagination
from go2_supermimic.supermimic.velocity import EXTRA_DIM, SLIP, CONTACT, vertical_yaw_rate

SUITE = [(0.,0.,0.),(.5,0.,0.),(-.5,0.,0.),(0.,.25,0.),(0.,-.25,0.),
         (0.,0.,.8),(0.,0.,-.8),(.5,0.,.5),(0.,.2,.5),(0.,0.,0.)]


def summarize_trace(rows):
  """Keep signed per-axis response separate from pooled tracking rewards."""
  velocity=torch.stack([r['velocity'] for r in rows])
  command=torch.stack([r['command'] for r in rows])
  return {
    'velocity_mean':velocity.mean((0,1)).tolist(),
    'velocity_rmse':(velocity-command).square().mean((0,1)).sqrt().tolist(),
    'height_mean_m':float(torch.stack([r['height'] for r in rows]).mean()),
    'reward_terms_per_second':{k:float(torch.stack([r['reward_terms'][k] for r in rows]).mean())
                               for k in rows[0]['reward_terms']},
    'reset_count':int(torch.stack([r['done'] for r in rows]).sum()),
  }


def evaluate(args):
  ckpt=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
  task=ckpt['cfg']['task']
  if task not in ('Titatit-Velocity-Flat','Titatit-Velocity-Terrain'):
    raise ValueError('Use a TITATIT velocity checkpoint')
  cfg=load_env_cfg(task,play=True)
  contract=ckpt.get('velocity_contract')
  if contract is None or contract.get('version')!=(3 if task.endswith('Terrain') else 2):
    raise ValueError('Expected velocity contract v2 (Flat) or v3 (Terrain)')
  # Evaluate the checkpoint's reward scales, rather than silently using new ones.
  from go2_supermimic.supermimic.velocity import VelocityRewards
  cfg.commands['velocity'].rewards=VelocityRewards(**contract['command']['rewards'])
  if cfg.commands['velocity'].terrain:
    cfg.commands['velocity'].field_res=contract['command']['field_res']
    cfg.curriculum={}
    if not 0 <= args.terrain_level < cfg.scene.terrain.terrain_generator.num_rows:
      raise ValueError('terrain-level must be between 0 and 4')
  for name, reward in cfg.rewards.items():
    reward.weight=cfg.commands['velocity'].rewards.weights[name]
  cfg.seed=args.seed
  cfg.scene.num_envs=args.num_envs
  cfg.auto_reset=False
  cfg.viewer.width,cfg.viewer.height=args.width,args.height
  schedule=SUITE if args.suite else [(args.vx,args.vy,args.yaw)]
  cfg.commands['velocity'].fixed_command=schedule[0]
  env=ManagerBasedRlEnv(cfg,device=args.device,render_mode='rgb_array' if args.video else None)
  if cfg.commands['velocity'].terrain:
    from go2_supermimic.env.terrains import velocity_terrain_contract
    if json.loads(json.dumps(velocity_terrain_contract(cfg.scene.terrain))) != json.loads(json.dumps(contract['terrain'])):
      raise ValueError('Terrain recipe differs from checkpoint')
    terrain=env.scene.terrain
    terrain.terrain_levels.fill_(args.terrain_level)
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
  obs,_=env.reset();hist=obs['actor'];term=env.command_manager.get_term('velocity')
  c=st.robot_consts(env);ctx=st.ContextReader(env);xi=ctx.read()
  bias=env.scene['robot'].data.encoder_bias[:,st.action_joint_ids(env)]*c.position_mask
  actor=Actor(hist.shape[-1],tuple(ckpt['cfg']['actor_hidden']),16).to(args.device).eval()
  actor.load_state_dict(ckpt['actor'])
  wm=WorldModel(tuple(ckpt['cfg']['wm_hidden']),16,EXTRA_DIM).to(args.device).eval()
  wm.load_state_dict(ckpt['world_model'])
  imag=VelocityImagination(ImagineCfg(horizon=32,dt=env.step_dt),c,term.cfg.rewards,5,
                           st.obs_terms(cfg),noise=False,field=term.height_field)
  critic=Critic(imag.critic_dim,tuple(ckpt['cfg']['critic_hidden'])).to(args.device).eval()
  critic.load_state_dict(ckpt['critic'])
  if args.viewer:
    if args.suite: raise ValueError('--viewer uses a single fixed command; use --video for suite metrics')
    # Interactive viewer owns stepping and auto reset.
    env.cfg.auto_reset=True
    from mjlab.viewer import NativeMujocoViewer
    NativeMujocoViewer(env,lambda observation: actor(observation['actor'])).run()
    env.close();return
  metrics={};frames=[];previous=torch.zeros(env.num_envs,16,device=env.device)
  horizon_errors={h:[] for h in (1,8,16,32)}
  imagined=[];real_windows=[];real_total=torch.zeros(env.num_envs,device=env.device)
  pred=None;alive=None;window_return=None;window_step=0
  phase_stats=[]
  trace=[]
  def add(name,value): metrics.setdefault(name,[]).append(value.detach().cpu().reshape(-1))
  with torch.inference_mode():
    for phase,cmd in enumerate(schedule):
      before=len(metrics.get('lin_sq',[]))
      phase_trace=[]
      phase_steps=max(1,round(args.seconds/env.step_dt))
      for step in range(phase_steps):
        s=st.robot_state(env,term.height_field);pose=st.planar_pose(env)
        if pred is None or window_step == 32:
          seed=dict(s=s,pose=pose,a_prev=previous,obs=hist,command=term.command.clone(),
                    xi=xi,enc_bias=bias,env=torch.arange(env.num_envs,device=env.device))
          objective,_=imag.rollout(wm,actor,critic,seed)
          imagined.append(objective.cpu())
          pred=s.clone();pred_pose=pose.clone();alive=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
          window_return=torch.zeros(env.num_envs,device=env.device);window_step=0
        action=actor(hist)
        pred,delta=wm(pred,action,previous,xi,imag.relief(pred_pose))
        pred_pose=st.apply_pose_delta(pred_pose,delta)
        executed_command=term.command.clone()
        # Commands update after reward, before observation/history construction.
        if step==phase_steps-1 and phase+1<len(schedule):
          term.cfg.fixed_command=schedule[phase+1]
        obs,reward,done,timeout,_=env.step(action)
        # auto_reset=False: read terminal state, never substitute reset posture.
        sn=st.robot_state(env,term.height_field)
        add('lin_sq',(sn[:,4:6]-executed_command[:,:2]).square().sum(-1))
        add('yaw_sq',(vertical_yaw_rate(sn)-executed_command[:,2]).square())
        add('height',sn[:,0])
        gravity=torch.nn.functional.normalize(sn[:,1:4],dim=-1)
        add('roll',torch.atan2(-gravity[:,1],-gravity[:,2]))
        add('pitch',torch.asin(gravity[:,0].clamp(-1,1)))
        add('slip_sq',(sn[:,SLIP].reshape(-1,4,2).square().sum(-1)*sn[:,CONTACT]).sum(-1))
        add('contacts',sn[:,CONTACT].sum(-1))
        add('action_rate',(action-previous).square().sum(-1))
        add('falls',done.float())
        row={'time':(phase*phase_steps+step+1)*env.step_dt,
             'phase':phase,'command':executed_command.cpu(),
             'velocity':torch.cat([sn[:,4:6],vertical_yaw_rate(sn)[:,None]],-1).cpu(),
             'height':sn[:,0].cpu(),'action':action.cpu(),
             'joint_velocity':sn[:,26:42].cpu(),'torque':sn[:,91:107].cpu(),
             'contact':sn[:,CONTACT].cpu(),'slip':sn[:,SLIP].reshape(-1,4,2).cpu(),
             'reward_terms':{k:v.cpu()*term.cfg.rewards.weights[k]
                             for k,v in env._velocity_reward_values.items()},
             'done':(done|timeout).cpu()}
        phase_trace.append(row)
        if args.trace:
          row.update(state_before=s.cpu(),state_after=sn.cpu(),observation=hist.cpu(),
                     previous_action=previous.cpu(),context=xi.cpu(),encoder_bias=bias.cpu(),
                     pose=pose.cpu())
          trace.append(row)
        real_total+=reward
        window_return+=imag.cfg.gamma**window_step*alive*reward
        alive &= ~(done|timeout)
        window_step+=1
        if window_step in horizon_errors and alive.any():
          normalized=(pred-sn)/(wm.out_norm.var[:wm.state_dim].sqrt()+1e-6)
          horizon_errors[window_step].append({
            'normalized_rmse':float(normalized[alive].square().mean().sqrt()),
            'linear_velocity_rmse_m_s':float((pred[alive,4:6]-sn[alive,4:6]).square().mean().sqrt()),
            'height_rmse_m':float((pred[alive,0]-sn[alive,0]).square().mean().sqrt()),
            'pose_xy_rmse_m':float((pred_pose[alive,:2]-st.planar_pose(env)[alive,:2]).square().mean().sqrt()),
            'valid_envs':int(alive.sum()),
          })
        if window_step==32: real_windows.append(window_return.cpu())
        if args.video: frames.append(env.render())
        reset=done|timeout
        if reset.any():
          obs,_=env.reset(env_ids=reset.nonzero().flatten())
        previous=torch.where(reset[:,None],torch.zeros_like(action),action)
        hist=obs['actor']
      # Dynamics windows never straddle a suite command switch.
      pred=None
      phase_stats.append({'command':cmd,
        'linear_tracking_rmse_m_s':float(torch.cat(metrics['lin_sq'][before:]).mean().sqrt()),
        'yaw_tracking_rmse_rad_s':float(torch.cat(metrics['yaw_sq'][before:]).mean().sqrt()),
        'last_half':summarize_trace(phase_trace[len(phase_trace)//2:])})
  m={k:torch.cat(v) for k,v in metrics.items()}
  duration=max(1,round(args.seconds/env.step_dt))*env.step_dt*len(schedule)
  report={
    'linear_tracking_rmse_m_s':float(m['lin_sq'].mean().sqrt()),
    'yaw_tracking_rmse_rad_s':float(m['yaw_sq'].mean().sqrt()),
    'base_height_mean_m':float(m['height'].mean()),'base_height_std_m':float(m['height'].std(unbiased=False)),
    'roll_mean_rad':float(m['roll'].mean()),'roll_std_rad':float(m['roll'].std(unbiased=False)),
    'pitch_mean_rad':float(m['pitch'].mean()),'pitch_std_rad':float(m['pitch'].std(unbiased=False)),
    'fall_rate_per_env_minute':float(m['falls'].sum()/env.num_envs/duration*60),
    'wheel_contact_slip_rms_m_s':float((m['slip_sq'].sum()/m['contacts'].sum().clamp(min=1)).sqrt()),
    'action_delta_squared_mean':float(m['action_rate'].mean()),
    'actual_simulator_return_mean':float(real_total.mean()),
    'actual_32step_discounted_reward_mean':float(torch.cat(real_windows).mean()) if real_windows else None,
    'imagined_32step_bootstrapped_return_mean':float(torch.cat(imagined).mean()),
    'world_model_open_loop_errors':{h:({k:sum(r[k] for r in rows)/len(rows) for k in rows[0]} if rows else None)
                                    for h,rows in horizon_errors.items()},
    'phases':phase_stats,'device':args.device,'task':task,
    'terrain_level':args.terrain_level if term.cfg.terrain else None,
    'note':'Actual total and imagined bootstrapped return use different horizons; neither is a convergence certificate.',
  }
  print(json.dumps(report,indent=2))
  if args.output: Path(args.output).write_text(json.dumps(report,indent=2))
  if args.trace: torch.save({'checkpoint':args.checkpoint,'dt':env.step_dt,'samples':trace},args.trace)
  if args.video:
    import mediapy
    mediapy.write_video(args.video,frames,fps=round(1/env.step_dt))
  env.close()
  return report


def main():
  p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
  for name in ('vx','vy','yaw'):p.add_argument('--'+name,type=float,default=0.)
  p.add_argument('--num-envs',type=int,default=64);p.add_argument('--seconds',type=float,default=10.,help='Seconds per suite phase')
  p.add_argument('--device',default='cuda:0');p.add_argument('--suite',action='store_true')
  p.add_argument('--seed',type=int,default=0);p.add_argument('--trace',help='Save aligned per-step tensors to a .pt file')
  p.add_argument('--terrain-level',type=int,default=0,help='Terrain difficulty row, 0 (easy) to 4 (hard)')
  p.add_argument('--viewer',action='store_true');p.add_argument('--video');p.add_argument('--output')
  p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=720)
  args=p.parse_args()
  if args.seconds<=0 or args.num_envs<1:p.error('seconds and num-envs must be positive')
  evaluate(args)

if __name__=='__main__':main()
