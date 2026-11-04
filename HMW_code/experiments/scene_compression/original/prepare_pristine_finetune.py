"""Convert existing compression PLYs to official checkpoints; reset Adam equally."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel

root = Path('/data/zpxiang/ghap_pristine_compare_20260921_01')
out = Path('/data/zpxiang/ghap_pristine_finetune_20260921_01')
parser = argparse.ArgumentParser()
op = OptimizationParams(parser)
opt = op.extract(parser.parse_args([]))
opt.iterations = 35000
for scene, checkpoint in [
    ('train', '/data/zpxiang/ghap_hcp_train_20260920_01/input/chkpnt30000.pth'),
    ('truck', '/data/zpxiang/ghap_sanc_hot_20260904_01/truck/backbone_15k/chkpnt15000.pth')]:
    cap, _ = torch.load(checkpoint, map_location='cpu', weights_only=False)
    spatial_scale = cap[-1]
    del cap
    for method in ['official_ghap', 'hilbert_optimized']:
        source = root/scene/method/'point_cloud/iteration_1/point_cloud.ply'
        dest = out/scene/method
        dest.mkdir(parents=True, exist_ok=True)
        model = GaussianModel(3)
        model.load_ply(str(source))
        model.spatial_lr_scale = spatial_scale
        model.max_radii2D = torch.zeros(len(model._xyz), device='cuda')
        # Exposure is not captured; official Scene initializes it from cameras.
        model._exposure = torch.nn.Parameter(torch.eye(3, 4, device='cuda')[None])
        model.training_setup(opt)
        torch.save((model.capture(), 30000), dest/'start.pth')
        (dest/'protocol.json').write_text(json.dumps(dict(
            source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            count=len(model._xyz), first_iter=30000, last_iter=35000,
            loop_iterations=5000, optimizer_updates=4999,
            schedule='common official 30001..35000 for both scenes and arms',
            optimizer='fresh Adam for both arms', geometry_trainable=True,
            densification=False, compaction=False, seed=0,
            note='Official loop skips optimizer update on the final iteration.'), indent=2))
        del model
        torch.cuda.empty_cache()
