"""Forward, body-mounted ray depth; no oracle terrain in actor observations."""
import math
import mujoco
import numpy as np
import torch
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg
from go2_supermimic.env.velocity_env_cfg import titatit_velocity_env_cfg
from go2_supermimic.robot.titatit import get_spec

DEPTH_SIZE = 32
DEPTH_FOV = 70.
DEPTH_FAR = 3.
DEPTH_INTERVAL = 5


def depth_robot_spec():
  spec = get_spec()
  angle = math.radians(20)
  c, s = math.cos(angle), math.sin(angle)
  # Camera -Z points forward/down; local X is robot right, local Y is up.
  rotation = np.array([[0,s,-c],[-1,0,0],[0,c,s]],dtype=float)
  quat = np.empty(4)
  mujoco.mju_mat2Quat(quat, rotation.ravel())
  spec.body('base').add_site(name='depth_camera',pos=(.45,0,.08),quat=quat,group=5)
  return spec


def titatit_depth_env_cfg(play=False):
  cfg = titatit_velocity_env_cfg(play=play,terrain=True)
  cfg.observations['actor'].terms.pop('height_scan')
  cfg.scene.entities['robot'].spec_fn = depth_robot_spec
  cfg.scene.sensors += (RayCastSensorCfg(name='forward_depth',
    frame=ObjRef(type='site',name='depth_camera',entity='robot'),
    pattern=PinholeCameraPatternCfg(width=DEPTH_SIZE,height=DEPTH_SIZE,fovy=DEPTH_FOV),
    ray_alignment='base',max_distance=DEPTH_FAR,include_geom_groups=(0,)),)
  return cfg


def read_depth(env):
  """Normalized camera-axis depth (not ray length); misses are far plane."""
  distance = env.scene['forward_depth'].data.distances
  grid = torch.linspace(-1,1,DEPTH_SIZE,device=distance.device)
  x,y = torch.meshgrid(grid,grid,indexing='xy')
  cosine = (1+(x.square()+y.square())*math.tan(math.radians(DEPTH_FOV/2))**2).rsqrt()
  depth = torch.where(distance>=0,distance*cosine.flatten(),DEPTH_FAR)
  return depth.clamp(.05,DEPTH_FAR).reshape(-1,1,DEPTH_SIZE,DEPTH_SIZE)/DEPTH_FAR


def depth_contract():
  return dict(version=1,size=DEPTH_SIZE,fovy=DEPTH_FOV,far=DEPTH_FAR,
              interval=DEPTH_INTERVAL,position=(.45,0,.08),pitch_down_degrees=20,
              geometry_groups=(0,),depth='camera_axis_normalized')
