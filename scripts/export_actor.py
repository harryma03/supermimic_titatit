"""Export just Actor+RunningNorm and verify the ONNX output on the same input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from go2_supermimic.supermimic.train import Actor


def export(checkpoint, output):
  import onnx
  from onnx.reference import ReferenceEvaluator
  ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if ckpt['cfg']['task'] != 'Titatit-Velocity-Flat' or not ckpt.get('velocity_contract'):
    raise ValueError('Expected a velocity checkpoint with an observation contract')
  actor=Actor(ckpt['obs_dim'],tuple(ckpt['cfg']['actor_hidden']),ckpt['act_dim']).eval()
  actor.load_state_dict(ckpt['actor'])
  x=torch.randn(1,ckpt['obs_dim'])
  output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
  torch.onnx.export(actor,x,str(output),input_names=['observation'],output_names=['action'],
                    opset_version=17,dynamo=False)
  model=onnx.load(str(output));onnx.checker.check_model(model)
  # ReferenceEvaluator avoids a dependency on a particular ONNX Runtime wheel.
  session=ReferenceEvaluator(model)
  errors=[]
  for x in (x,torch.zeros_like(x),actor.norm.mean[None].clone()):
    with torch.no_grad(): expected=actor(x).numpy()
    actual=session.run(None,{'observation':x.numpy()})[0]
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
    errors.append(float(np.max(np.abs(actual-expected))))
  metadata={'input':[1,ckpt['obs_dim']], 'output':[1,ckpt['act_dim']],
            'max_abs_error':max(errors),'validator':'onnx.reference.ReferenceEvaluator',
            'contract':ckpt['velocity_contract']}
  output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
  print(json.dumps({k:v for k,v in metadata.items() if k!='contract'},indent=2))
  return metadata

if __name__=='__main__':
  ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True)
  args=ap.parse_args();export(args.checkpoint,args.output)
