"""Generated terrain and the scene look.

The mix follows Isaac Lab's rough-terrain set (pyramid stairs up and down, box
grids, random rough ground, slopes), with 12 m tiles that leave room for the
chase. One column per terrain type, five rows of rising difficulty: steps go
from 2 to 10 cm, boxes from +-2 to +-5 cm (neighbours differ by up to 10 cm),
and the flat platforms are small so the robot spends its time on the features. Sharp edges are where the world model is least
accurate, so robots start on the easiest row and harder rows open up during
training (mdp.terrain_levels_by_progress).

The look is a light matte ground under a sky gradient with a sun that casts
shadows, instead of mjlab's height-coloured terrain.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import (
  box_random_grid,
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import (TerrainGeneratorCfg, SubTerrainCfg,
                                              TerrainGeometry, TerrainOutput)
from mjlab.utils.spec_config import LightCfg, MaterialCfg, TextureCfg

ROUGH_TERRAINS = TerrainGeneratorCfg(
  seed=7,
  size=(12.0, 12.0),
  border_width=10.0,
  num_rows=5,
  num_cols=7,
  curriculum=True,
  color_scheme="none",
  sub_terrains={
    "flat": flat(proportion=0.10),
    "stairs": pyramid_stairs(proportion=0.20, step_height_range=(0.02, 0.10), step_width=0.4,
                             platform_width=1.5),
    "stairs_inv": pyramid_stairs_inv(proportion=0.20, step_height_range=(0.02, 0.10),
                                     step_width=0.4, platform_width=1.5),
    "boxes": box_random_grid(proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.05),
                             platform_width=1.0),
    "rough": random_rough(proportion=0.15, noise_range=(0.01, 0.04), noise_step=0.01),
    "slope": hf_pyramid_slope(proportion=0.10, slope_range=(0.0, 0.3)),
    "slope_inv": hf_pyramid_slope_inv(proportion=0.10, slope_range=(0.0, 0.3)),
  },
  add_lights=False,
)

@dataclass(kw_only=True)
class GapClimbTerrainCfg(SubTerrainCfg):
  """Two transverse obstacles around a flat, ground-level spawn platform."""

  kind: str = "gap"
  gap_width_range: tuple[float, float] = (0.10, 0.40)
  gap_depth: float = 0.60
  climb_height_range: tuple[float, float] = (0.05, 0.25)
  climb_width: float = 0.80
  platform_width: float = 1.50

  def function(self, difficulty, spec, rng):
    del rng
    if self.kind not in ("gap", "climb"):
      raise ValueError(f"Unknown terrain kind: {self.kind}")
    length, width = self.size
    d = float(np.clip(difficulty, 0., 1.))
    obstacle_width = (self.gap_width_range[0] + d *
      (self.gap_width_range[1] - self.gap_width_range[0])) if self.kind == "gap" else self.climb_width
    left = (length - self.platform_width) / 2
    right = (length + self.platform_width) / 2
    if left <= obstacle_width or width <= 0 or self.gap_depth <= 0:
      raise ValueError("Tile must fit the spawn platform and two obstacles")
    body = spec.body("terrain")
    geometries = []

    def box(x0, x1, bottom, top):
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=((x0+x1)/2, width/2, (bottom+top)/2),
        size=((x1-x0)/2, width/2, (top-bottom)/2))
      geometries.append(TerrainGeometry(geom=geom, color=(.62,.62,.61,1.)))

    if self.kind == "gap":
      # A finite trench floor keeps both physical contacts and height rays defined.
      box(0, length, -self.gap_depth-.1, -self.gap_depth)
      for a, b in ((0, left-obstacle_width), (left, right),
                   (right+obstacle_width, length)):
        box(a, b, -self.gap_depth, 0.)
    else:
      box(0, length, -.1, 0.)
      height = self.climb_height_range[0] + d * (self.climb_height_range[1]-self.climb_height_range[0])
      for a, b in ((left-obstacle_width, left), (right, right+obstacle_width)):
        box(a, b, 0., height)
    return TerrainOutput(origin=np.array((length/2, width/2, 0.)), geometries=geometries)


GROUND_RGBA = (0.62, 0.62, 0.61, 1.0)
GRID_RGB = ((0.66, 0.66, 0.65), (0.62, 0.62, 0.61), (0.46, 0.47, 0.50))  # cells, lines
SKY_RGB = ((0.93, 0.95, 0.98), (0.62, 0.74, 0.87))


def clean_look(terrain: TerrainEntityCfg) -> TerrainEntityCfg:
  """Light grey matte ground with a faint 0.5 m grid (its lines break at step
  edges and bend over slopes, so the terrain's shape reads on video), a sky
  gradient and soft fill light."""
  terrain.textures = (
    TextureCfg(name="sky", type="skybox", builtin="gradient", rgb1=SKY_RGB[0], rgb2=SKY_RGB[1],
               width=512, height=3072),
    TextureCfg(name="grid", type="2d", builtin="checker", mark="edge", rgb1=GRID_RGB[0],
               rgb2=GRID_RGB[1], markrgb=GRID_RGB[2], width=512, height=512),
  )
  terrain.materials = (
    MaterialCfg(name="ground", texture="grid", texuniform=True, texrepeat=(1.0, 1.0),
                reflectance=0.0, geom_names_expr=(".*",)),
  )
  # Fill light only: the shadow-casting sun follows the robot (robot/go2.xml),
  # so its shadow map stays sharp wherever the robot runs.
  terrain.lights = (
    LightCfg(name="fill", type="directional", pos=(0.0, 0.0, 10.0), dir=(0.3, 0.2, -1.0),
             diffuse=(0.12, 0.12, 0.12), specular=(0.0, 0.0, 0.0), ambient=(0.12, 0.13, 0.15),
             castshadow=False),
  )
  return terrain


def rough_terrain(*, gap_climb: bool = False) -> TerrainEntityCfg:
  generator = copy.deepcopy(ROUGH_TERRAINS)
  if gap_climb:
    generator.sub_terrains.update(gap=GapClimbTerrainCfg(kind="gap", proportion=.15),
                                 climb=GapClimbTerrainCfg(kind="climb", proportion=.15))
    generator.num_cols = len(generator.sub_terrains)
  return clean_look(TerrainEntityCfg(
    terrain_type="generator", terrain_generator=generator,
    max_init_terrain_level=None,
  ))


def velocity_terrain_contract(terrain):
  """Serializable effective terrain geometry for checkpoints and eval."""
  from dataclasses import asdict
  cfg = terrain.terrain_generator
  return {key: value for key, value in asdict(cfg).items()
          if key in ('seed','size','num_rows','num_cols','border_width','curriculum','sub_terrains')}
