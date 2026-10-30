"""Actual collision geometry: safe spawn, trench floor and raised platforms."""
import mujoco
import numpy as np
import pytest
from go2_supermimic.env.terrains import GapClimbTerrainCfg, rough_terrain


@pytest.mark.parametrize('kind', ['gap', 'climb'])
@pytest.mark.parametrize('difficulty', [0., 1.])
def test_gap_climb_geometry(kind, difficulty):
  cfg = GapClimbTerrainCfg(kind=kind, size=(12.,12.))
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name='terrain')
  output = cfg.function(difficulty, spec, np.random.default_rng(0))
  model = spec.compile(); data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  def height(x, y=6.):
    distance = mujoco.mj_ray(model, data, np.array([x,y,2.]),
      np.array([0.,0.,-1.]), None, 1, -1, np.zeros(1,dtype=np.int32))
    assert distance >= 0
    return 2-distance
  np.testing.assert_allclose(output.origin, [6.,6.,0.])
  assert height(6.) == pytest.approx(0.)
  extent = (.1+.3*difficulty) if kind=='gap' else .8
  expected = -.6 if kind=='gap' else .05+.2*difficulty
  for x in (5.25-extent/2, 6.75+extent/2):
    for y in (.1,6.,11.9):
      assert height(x,y) == pytest.approx(expected)
  assert height(5.25-extent-.05) == pytest.approx(0.)
  assert height(6.75+extent+.05) == pytest.approx(0.)


def test_only_velocity_terrain_opts_into_gap_climb():
  assert len(rough_terrain().terrain_generator.sub_terrains)==7
  cfg=rough_terrain(gap_climb=True).terrain_generator
  assert cfg.num_cols==len(cfg.sub_terrains)==9


def test_mixed_terrain_constraint_capacity():
  from go2_supermimic.env.velocity_env_cfg import titatit_velocity_env_cfg
  from go2_supermimic.env.depth_env_cfg import titatit_depth_env_cfg
  assert titatit_velocity_env_cfg(terrain=True).sim.njmax == 1024
  assert titatit_depth_env_cfg().sim.njmax == 1024
