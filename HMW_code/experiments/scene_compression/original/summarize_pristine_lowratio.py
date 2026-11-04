"""Require all 16 offline evaluations and summarize measured compression runs."""
import json
import math
import statistics
from pathlib import Path

root = Path('/data/zpxiang/ghap_pristine_lowratio_20260923_01')
summary = []
for ratio in [0.01, 0.001]:
    r = root/f'rho_{ratio}'
    timings = json.loads((r/'pre/timings.json').read_text())
    assert len(timings) == 16
    for scene in ['train', 'truck']:
        for method in ['official_ghap', 'hilbert_optimized']:
            runs = [x for x in timings if x['scene'] == scene and
                    x['method'] == method and not x['warmup']]
            assert len(runs) == 3
            pre = json.loads((r/'pre'/scene/method/'results.json').read_text())[method]
            post = json.loads((r/'post'/scene/method/'results.json').read_text())[method]
            for metric in [pre, post]:
                assert all(math.isfinite(metric[k]) for k in ['PSNR', 'SSIM', 'LPIPS'])
            assert all(x['output_count'] > 0 for x in runs)
            if method == 'hilbert_optimized':
                assert all(x['output_count'] == x['target_count'] for x in runs)
            wall = float((r/'post'/scene/method/'train_wall.txt').read_text().splitlines()[0].split()[1])
            summary.append(dict(retention=ratio, scene=scene, method=method,
                compression_median_seconds=statistics.median(x['seconds'] for x in runs),
                compression_runs_seconds=[x['seconds'] for x in runs],
                train_process_seconds=wall, output_count=runs[0]['output_count'],
                target_count=runs[0]['target_count'], pre=pre, post=post))
(root/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
