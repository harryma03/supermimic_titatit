"""Train TITATIT depth perception + SuperMimic: python scripts/train_depth.py --help."""
import torch
from go2_supermimic.supermimic.train import main
from go2_supermimic.supermimic.depth_train import DepthTrainCfg,DepthTrainer

if __name__ == '__main__':
  torch.set_float32_matmul_precision('high')
  main(DepthTrainCfg,DepthTrainer)
