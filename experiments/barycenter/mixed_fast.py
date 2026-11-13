"""Reproduce the selected fast LP-free initialization + Riemannian pipelines.

These settings were tuned on K=30/data-seed=0 and are not universal defaults.
All initializer construction/selection and optimizer validation enter timing.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .data import load_gmm, save_gmm
from .mixed_fast_solver import balanced_search, pad, solve_mixed
from .scaling import evaluate


SETTINGS={
    'images':dict(initialization=dict(L=1024,banks=1,width=3,root_width=3),
                  optimization=dict(rate=1.,sampling='partial',mean=1e-6,
                                    mass=.01,steps=100,L=64)),
    'synthetic':dict(initialization=dict(L=256,banks=1,width=2,root_width=2),
                     optimization=dict(rate=10.,sampling='fixed',mean=1e-6,
                                       mass=.01,steps=200,L=256)),
}


def main():
    with threadpool_limits(1):
        run()


def run():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=list(SETTINGS),default=list(SETTINGS))
    p.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent / 'assets' / 'mixed_fast')
    p.add_argument('--output-dir',type=Path,default=Path('results/mixed_fast'))
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be positive')
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'settings.json').write_text(json.dumps(SETTINGS,indent=2))
    rows=[]
    for case in args.cases:
        source=args.source_dir/case
        inputs=[load_gmm(source/'inputs'/f'input_{j}.npz') for j in range(4)]
        ref=load_gmm(source/'MW2/barycenter.npz')
        cfg=SETTINGS[case];out=args.output_dir/case;out.mkdir(parents=True,exist_ok=True)
        ea=SimpleNamespace(test_kinds=['B'],test_L=512,eval_samples=1024)
        for repeat in range(args.repeats):
            tick=time.perf_counter()
            candidate,count=balanced_search(inputs,**cfg['initialization'])
            initial=pad(candidate.gmm,sum(g.count for g in inputs)-3,0)
            init_ms=1000*(time.perf_counter()-tick)
            result,history,stats=solve_mixed(inputs,initial,cfg['optimization'])
            row=dict(case=case,repeat=repeat,candidates=count,init_ms=init_ms,
                     total_ms=init_ms+stats['optimizer_ms'],**stats,
                     **evaluate(result,inputs,[.25]*4,ea,0,ref))
            init_metrics=evaluate(initial,inputs,[.25]*4,ea,0,ref)
            row.update(initial_mw2_objective=init_metrics['mw2_objective'],
                       initial_test_min_B=init_metrics['test_min_B'],
                       improvement_percent=100*(1-row['mw2_objective']/init_metrics['mw2_objective']))
            rows.append(row);pd.DataFrame(rows).to_csv(args.output_dir/'metrics.csv',index=False)
            save_gmm(out/'initial.npz',initial);save_gmm(out/'optimized.npz',result)
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
