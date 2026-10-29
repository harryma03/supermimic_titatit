"""Native MuJoCo standing-height calibration, not an RL policy or demonstration.

Uses the task's unmodified robot and torque limits. A diagnostic support-force
controller supplies leg position targets at 50 Hz; the original 60/1.5 PD runs
at 200 Hz. Wheels retain zero velocity targets with damping 0.5. The resulting
height is a reachable standing reference, not evidence that zero action stands.
"""
import json
import mujoco
import numpy as np
from scipy.optimize import least_squares
import go2_supermimic.env  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.scene import Scene


def calibrate(seconds=20.):
  cfg=load_env_cfg('Titatit-Velocity-Flat',play=True)
  model=Scene(cfg.scene,'cpu').compile()
  model.opt.timestep=cfg.sim.mujoco.timestep
  model.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  model.opt.cone=mujoco.mjtCone.mjCONE_ELLIPTIC
  data=mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model,data,0)
  sites=[model.site('robot/'+leg).id for leg in ('FR','FL','RR','RL')]
  from go2_supermimic.robot.titatit import WHEEL_RADIUS
  def residual(x):
    data.qpos[2]=x[0]
    data.qpos[3:7]=[np.cos(x[1]/2),0,np.sin(x[1]/2),0]
    mujoco.mj_forward(model,data)
    return data.site_xpos[sites,2]-WHEEL_RADIUS
  fit=least_squares(residual,[.36,.02]);residual(fit.x)
  reference=data.qpos.copy();target=reference.copy();heights=[];tilts=[]
  for t in range(round(seconds/model.opt.timestep)):
    if t % cfg.decimation == 0:
      mujoco.mj_forward(model,data)
      jac=[]
      for site in sites:
        point=data.site_xpos[site].copy();point[2]-=WHEEL_RADIUS
        jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(model,data,jp,jr,point,model.site_bodyid[site]);jac.append(jp)
      jac=np.concatenate(jac)
      error=np.zeros(model.nv)
      mujoco.mj_differentiatePos(model,error,1,data.qpos,reference)
      wrench=(data.qfrc_bias[:6]+np.array([30,30,300,100,100,30])*error[:6]
              -np.array([20,20,50,20,20,10])*data.qvel[:6])
      forces=np.linalg.lstsq(jac[:,:6].T,wrench,rcond=None)[0]
      torque=data.qfrc_bias-jac.T@forces
      for i in range(model.nu):
        j=model.actuator_trnid[i,0]
        target[model.jnt_qposadr[j]]=reference[model.jnt_qposadr[j]]+torque[model.jnt_dofadr[j]]/60
    for i in range(model.nu):
      j=model.actuator_trnid[i,0];qa=model.jnt_qposadr[j];va=model.jnt_dofadr[j]
      wheel='foot' in model.joint(j).name
      torque=-.5*data.qvel[va] if wheel else 60*(target[qa]-data.qpos[qa])-1.5*data.qvel[va]
      data.ctrl[i]=np.clip(torque,-10 if wheel else -55,10 if wheel else 55)
    mujoco.mj_step(model,data)
    if t*model.opt.timestep >= seconds-5:
      heights.append(data.qpos[2]);tilts.append(2*np.arccos(np.clip(abs(data.qpos[3]),0,1)))
  result={'height_mean_m':float(np.mean(heights)), 'height_std_m':float(np.std(heights)),
          'orientation_angle_max_rad':float(max(tilts)), 'contacts_final':data.ncon,
          'seconds':seconds, 'controller':'diagnostic support forces -> leg position targets at 50 Hz'}
  if not .2 < result['height_mean_m'] < .5 or max(tilts)>.2 or result['height_std_m'] > .002:
    raise RuntimeError(f'Unstable calibration: {result}')
  return result

if __name__=='__main__':
  print(json.dumps(calibrate(),indent=2))
