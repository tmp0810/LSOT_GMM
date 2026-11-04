"""External timing harness; official GHAP source is never patched or wrapped."""
import gc
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import gmm_sampler
import scene as scene_module
import pytorch3d
from scene.gaussian_model import GaussianModel

job = Path('/home/zpxiang/ghap_pristine_compare_20260921_01')
ratio = float(os.environ['KEEP_RATIO'])
root = Path(os.environ['LOWRATIO_ROOT'])/'pre'
root.mkdir(parents=True, exist_ok=True)
assert Path(gmm_sampler.__file__).resolve() == job/'code/gmm_sampler.py'
assert Path(scene_module.__file__).resolve() == job/'code/scene/__init__.py'
assert str(Path(pytorch3d.__file__).resolve()).startswith(str(job/'pytorch3d_official')+'/')
provenance = dict(commit='27b3a581c861069cc97e5629e216dc3d1f42a98a',
                  sampler=gmm_sampler.__file__, scene=scene_module.__file__,
                  pytorch3d=pytorch3d.__file__, torch=torch.__version__)
(root/'imports.json').write_text(json.dumps(provenance,indent=2))
# Import our implementation only after the official modules have been verified.
import hcp_block_lloyd_batched as batch
rows=[]
for name, checkpoint in [
 ('train','/data/zpxiang/ghap_hcp_train_20260920_01/input/chkpnt30000.pth'),
 ('truck','/data/zpxiang/ghap_sanc_hot_20260904_01/truck/backbone_15k/chkpnt15000.pth')]:
 cap,_=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=GaussianModel(3)
 for attr,index in [('_xyz',1),('_features_dc',2),('_features_rest',3),('_scaling',4),('_rotation',5),('_opacity',6)]:
  setattr(model,attr,cap[index].detach().cuda())
 del cap
 target=int(len(model._xyz)*ratio)
 for repeat in range(-1,3):
  for method in (['official_ghap','hilbert_optimized'] if repeat%2 else ['hilbert_optimized','official_ghap']):
   gc.collect(); torch.cuda.empty_cache(); torch.manual_seed(42); np.random.seed(42)
   torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
   start=perf_counter()
   with torch.no_grad():
    if method=='official_ghap':
     result=gmm_sampler.gaussian_model_reduction(model,ratio,random_seed=42)
     record={}
    else:
     tensors,record=batch.payload(model,target,42)
     result=GaussianModel(3)
     for attr,key in [('_xyz','xyz'),('_scaling','scaling'),('_rotation','rotation'),('_features_dc','f_dc'),('_features_rest','f_rest'),('_opacity','opacity')]:
      setattr(result,attr,tensors[key])
     del tensors
   torch.cuda.synchronize()
   elapsed=perf_counter()-start
   row=dict(scene=name,method=method,repeat=repeat,warmup=repeat<0,seconds=elapsed,
     source_count=len(model._xyz),target_count=target,output_count=len(result._xyz),
     retention=ratio,batch_size=int(os.environ.get("HCP_BATCH_SIZE","16")),
     peak_memory_bytes=torch.cuda.max_memory_allocated(),stages_seconds=record.get('stages_seconds'))
   rows.append(row); (root/'timings.json').write_text(json.dumps(rows,indent=2))
   print('TIMING_RESULT',json.dumps(row),flush=True)
   if repeat==0:
    out=root/name/method
    out.mkdir(parents=True,exist_ok=True)
    result.save_ply(str(out/'point_cloud/iteration_1/point_cloud.ply'))
    old=Path('/data/zpxiang/ghap_gmm_fixed_20260920_01')/name/'gmm_mm'
    (out/'cfg_args').write_text((old/'cfg_args').read_text().replace(str(old),str(out)))
    (out/'reduction_stats.json').write_text(json.dumps(row,indent=2))
    (out/'algorithm_record.json').write_text(json.dumps(record,indent=2))
   del result,record
 del model
(root/'benchmark_complete.txt').write_text('4 warmups and 12 measured runs; fixed seed 42; no finetuning\n')
