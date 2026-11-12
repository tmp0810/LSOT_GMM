"""Image-GMM component scaling: full-tuple MW2 versus halving LSMT.

Run --config experiments/barycenter/configs/scaling.json. Each barycenter
solver is isolated; failures never prevent independent methods from running.
"""
import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from lsot.gaussians import GMM
from lsot.projections import sample_projection_bank
from .data import SHAPE_NAMES, bundled_masks, image_cloud, save_gmm, load_gmm
from .fast_solver import optimize_halving, optimize_fast, checkpoint_gmm, FastObjective, synchronize
from .reference import shared_initializations, mw2_barycenter
from .solver import on_device, mw2_objective, mw2_distance_squared
from .run import write_csv, empirical_sw2
from .scaling_resources import MemoryMeter, run_worker, write_json

METHODS = ['MW2'] + [f'{mode}-LSOT-{kind}' for mode in ('min', 'avg')
                    for kind in ('B', 'B1D', 'Mix', 'SMix')]


def prepare_data(folder, k, seed, args):
    """Fit all foreground points in [0,1]^2; shared cached inputs for all solvers."""
    from PIL import Image
    from sklearn.mixture import GaussianMixture
    folder.mkdir(parents=True, exist_ok=True)
    masks, provenance = bundled_masks()
    paths, fits = [], []
    for index, name in enumerate(SHAPE_NAMES):
        if args.image_dir:
            pixels = np.asarray(Image.open(Path(args.image_dir)/f'{name}.png'))
            origin = 'user_original_png'
        else:
            pixels = np.repeat(np.where(masks[name], 0, 255)[..., None], 3, axis=2).astype(np.uint8)
            origin = 'bundled_notebook_output_recovered_mask'
        cloud, mask = image_cloud(pixels)
        if mask.shape != (128, 128):
            raise ValueError(f'{name}: expected 128x128 image')
        cloud = cloud/127.  # SAME coordinate scale for all shapes, K and methods.
        settings = dict(K=k, seed=seed, cloud_sha256=hashlib.sha256(cloud.tobytes()).hexdigest(),
                        coordinate_divisor=127, max_iter=args.em_steps,
                        tol=args.em_tolerance, reg_covar=1e-6, n_init=1, covariance_type='full')
        path, meta_path = folder/f'input_{index}.npz', folder/f'input_{index}.json'
        cached = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if path.exists() and cached and cached['settings'] == settings:
            info = cached
        else:
            tick = time.perf_counter()
            fit = GaussianMixture(n_components=k, random_state=seed, n_init=1,
                                  max_iter=args.em_steps, tol=args.em_tolerance,
                                  reg_covar=1e-6, covariance_type='full')
            with warnings.catch_warnings(record=True) as messages:
                fit.fit(cloud)
            info = dict(settings=settings, name=name, origin=origin,
                        points=len(cloud), fit_ms=1000*(time.perf_counter()-tick),
                        converged=bool(fit.converged_), iterations=int(fit.n_iter_),
                        warnings=[str(w.message) for w in messages])
            save_gmm(path, GMM.from_numpy(fit.weights_, fit.means_, fit.covariances_))
            write_json(meta_path, info)
        Image.fromarray(pixels).save(folder/f'{name}.png')
        paths.append(str(path.resolve()))
        fits.append(info)
    write_json(folder/'manifest.json', dict(provenance=provenance['provenance'], fits=fits))
    return paths, fits


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec_path).parent
    torch.set_num_threads(spec['threads'])
    device = 'cpu' if spec['method'] == 'MW2' else spec['device']
    result = dict(method=spec['method'], runtime_device=device, total_ms=None)
    try:
        cpu_inputs = [load_gmm(p) for p in spec['inputs']]
        inputs = [on_device(g, device) for g in cpu_inputs]
        # Warm libraries with a tiny, independent problem, before measured window.
        warmup_started = time.perf_counter()
        tiny = GMM.from_numpy([1.], [[.5, .5]], [[[.01, 0.], [0., .01]]])
        mw2_barycenter([tiny]*4, spec['weights'], iterations=2, fast=True)
        if spec['method'] != 'MW2':
            mode, _, kind = spec['method'].split('-')
            small = on_device(tiny, device)
            optimize_fast([small]*4, spec['weights'], small, kind=kind, mode=mode,
                          steps=1, L=2, validation_L=2, patience=0)
        synchronize(device)
        result['warmup_ms'] = 1000*(time.perf_counter()-warmup_started)
        write_json(out/'ready.json', dict(monotonic=time.monotonic()))
        with threadpool_limits(limits=spec['threads']), MemoryMeter(device) as meter:
            tick = time.perf_counter()
            if spec['method'] == 'MW2':
                timings = {}
                solved = mw2_barycenter(cpu_inputs, spec['weights'],
                    iterations=spec['gaussian_iterations'], tolerance=spec['gaussian_tolerance'],
                    fast=True, timings=timings)
                candidate = solved.gmm
                result.update(status='ok', phase_ms=timings, marginal_error=solved.marginal_error,
                              covariance_iterations=solved.covariance_iterations,
                              covariance_relative_change=solved.covariance_relative_change,
                              covariance_tolerance_met=solved.covariance_tolerance_met,
                              tuple_objective=solved.cost_squared)
            else:
                mode, _, kind = spec['method'].split('-')
                initials = [on_device(g, device) for g in shared_initializations(
                    cpu_inputs, spec['weights'], spec['Kb'], starts=4,
                    iterations=spec['initialization_iterations'], seed=spec['seed'])]
                synchronize(device)
                init_ms = 1000*(time.perf_counter()-tick)
                solved = optimize_halving(inputs, spec['weights'], initials,
                    probe_steps=tuple(spec['probe_steps']), kind=kind, mode=mode,
                    steps=spec['steps'], learning_rate=spec['learning_rate'],
                    L=spec['L'], validation_L=spec['validation_L'], seed=spec['seed'],
                    sampling=spec['sampling'], validate_every=spec['validate_every'],
                    refresh_every=spec['refresh_every'], refresh_fraction=spec['refresh_fraction'],
                    patience=0, checkpoints=tuple(spec['checkpoints']), coordinate_scale=1.)
                candidate = solved.gmm
                result.update(status='numerical_stop' if solved.status.startswith('numerical_stop') else 'ok',
                              optimizer_status=solved.status, initialization_ms=init_ms,
                              validation_ms=solved.validation_ms, validation_objective=solved.objective,
                              total_updates=solved.total_updates, selected_start=solved.selected_start,
                              selected_step=solved.selected_step, gradient_inf=solved.gradient_inf,
                              start_summaries=solved.start_summaries, halving_decisions=solved.halving_decisions)
            synchronize(device)
            result['total_ms'] = 1000*(time.perf_counter()-tick)
        result.update(meter.stats, output_components=candidate.count)
        # Serialization, density plots, and quality evaluation are OUTSIDE timer.
        save_gmm(out/'barycenter.npz', candidate)
        if spec['method'] != 'MW2':
            write_csv(out/'history.csv', solved.history)
            points = {point['iteration']: point for point in solved.checkpoints}
            cp_rows = []
            for step, point in sorted(points.items()):
                g = checkpoint_gmm(initials[0], point)
                save_gmm(out/f'step{step:05d}.npz', g)
                cp_rows.append({key: value for key, value in point.items() if key != 'state'})
            write_csv(out/'checkpoints.csv', cp_rows)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        status = ('oom' if isinstance(exc, (MemoryError, torch.cuda.OutOfMemoryError))
                  else 'solver_error')
        result.update(status=status, error=f'{type(exc).__name__}: {exc}', total_ms=None)
    write_json(out/'result.json', result)


def evaluate(candidate, inputs, weights, args, seed, reference=None):
    from .plotting import density_grid
    row = dict(mw2_objective=mw2_objective(inputs, weights, candidate))
    for kind in args.test_kinds:
        bank = sample_projection_bank(2, args.test_L, kind=kind, seed=seed+2000003)
        with torch.no_grad():
            row[f'test_min_{kind}'] = float(FastObjective(inputs, weights, bank, kind, 'min')(candidate))
    row.update(relative_mw2_gap=None, mw2_to_reference=None,
               sw2_to_reference=None, density_l1_to_reference=None)
    if reference is not None:
        ref_cost = mw2_objective(inputs, weights, reference)
        _, a = density_grid(candidate, (0., 1.), 128)
        _, b = density_grid(reference, (0., 1.), 128)
        row.update(relative_mw2_gap=(row['mw2_objective']-ref_cost)/max(ref_cost, 1e-15),
                   mw2_to_reference=np.sqrt(max(0., mw2_distance_squared(reference, candidate))),
                   sw2_to_reference=empirical_sw2(reference, candidate, count=args.eval_samples,
                                                 directions=100, seed=seed+100000),
                   density_l1_to_reference=float(np.abs(a/a.sum()-b/b.sum()).sum()))
    return row


def plan_benchmark(inputs, args, seed, k):
    """Six unordered input pairs, full ground-cost work included for all methods."""
    from .benchmark import plan_once, elapsed_call
    rows = []
    for i, j in itertools.combinations(range(4), 2):
        reference_cost = None
        for method in ['MW2'] + [m for m in args.methods if m != 'MW2']:
            # Exact LP uses CPU/POT; sliced plans use requested accelerator.
            device = 'cpu' if method == 'MW2' else args.device
            a, b = [on_device(inputs[x], device) for x in (i, j)]
            samples = []
            for repeat in range(args.plan_warmups + args.plan_repeats):
                outcome, ms = elapsed_call(lambda: plan_once(a, b, method, args.L,
                                                            seed+repeat), device)
                if repeat >= args.plan_warmups:
                    samples.append(ms)
            cost, left, right = outcome
            error = float(torch.maximum((left-a.weights).abs().max(), (right-b.weights).abs().max()))
            if error > 1e-8:
                raise RuntimeError(f'Invalid plan marginals: {method}: {error}')
            if method == 'MW2':
                reference_cost = cost
            rows.append(dict(K=k, seed=seed, pair=f'{i}-{j}', method=method,
                             runtime_device=device, median_ms=float(np.median(samples)),
                             q25_ms=float(np.quantile(samples,.25)), q75_ms=float(np.quantile(samples,.75)),
                             cost_squared=cost, cost_ratio_to_mw2=cost/max(reference_cost,1e-15),
                             marginal_error=error, repeats=args.plan_repeats, scope='full_plan_and_cost', quality_sample='last_timing_repeat'))
    return rows


def make_outputs(root, rows, plans):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if rows:
        write_csv(root/'metrics.csv', rows)
        df = pd.DataFrame(rows)
        summaries = []
        for (k, method), group in df.groupby(['K', 'method']):
            ok = group[group.status == 'ok']
            item = dict(K=int(k), method=method, runs=len(group), successes=len(ok),
                        status_counts=json.dumps(group.status.value_counts().to_dict()),
                        em_nonconverged_runs=int((group.em_all_converged == False).sum()))
            for key in ['total_ms','cpu_peak_rss_bytes','cuda_peak_allocated_bytes',
                        'mw2_objective','relative_mw2_gap','test_min_B','test_min_B1D']:
                if key in ok:
                    values = pd.to_numeric(ok[key],errors='coerce').dropna()
                    item[key+'_mean'] = float(values.mean()) if len(values) else None
                    item[key+'_std'] = float(values.std()) if len(values)>1 else None
            summaries.append(item)
        write_csv(root/'summary.csv', summaries)
        # Paired speedups only where both methods actually succeeded.
        successes = df[df.status == 'ok']
        exact = successes[successes.method == 'MW2'][['K','seed','total_ms']].rename(
            columns={'total_ms':'mw2_ms'})
        comparisons = successes[successes.method != 'MW2'].merge(exact,on=['K','seed'])
        if len(comparisons):
            comparisons['speedup_over_mw2'] = comparisons.mw2_ms / comparisons.total_ms
            comparisons[['K','seed','method','mw2_ms','total_ms','speedup_over_mw2',
                         'relative_mw2_gap']].to_csv(root/'paired_speedups.csv',index=False)

        for metric, ylabel in [('total_ms','Total solver time (ms)'),
                               ('cpu_peak_rss_bytes','Peak CPU RSS (bytes)'),
                               ('cuda_peak_allocated_bytes','Peak CUDA allocation (bytes)'),
                               ('relative_mw2_gap','Relative MW2 objective gap')]:
            if metric not in df: continue
            fig, ax = plt.subplots(figsize=(7,4))
            for method, group in df.groupby('method'):
                group=group[group.status=='ok'].copy()
                group[metric]=pd.to_numeric(group[metric],errors='coerce')
                values=group.groupby('K')[metric].mean().dropna()
                if metric != 'relative_mw2_gap': values=values[values>0]
                if len(values): ax.plot(values.index,values.values,'o-',label=method)
            ax.set(xscale='log',xlabel='Components per input GMM K',ylabel=ylabel)
            if metric!='relative_mw2_gap': ax.set_yscale('log')
            if ax.lines: ax.legend()
            ax.grid(alpha=.25);fig.tight_layout();fig.savefig(root/f'{metric}.png',dpi=180);plt.close(fig)
    if plans:
        write_csv(root/'plans.csv',plans)
        df=pd.DataFrame(plans)
        fig,ax=plt.subplots(figsize=(7,4))
        for method,group in df.groupby('method'):
            values=group.groupby('K').median_ms.median()
            ax.plot(values.index,values.values,'o-',label=method)
        ax.set(xscale='log',yscale='log',xlabel='K',ylabel='Median full-plan time (ms)')
        ax.legend();fig.tight_layout();fig.savefig(root/'plans_time.png',dpi=180);plt.close(fig)


def run(args):
    import os
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ('output_dir','config','no_resume','worker')}
    # Record source version to prevent mixing runs after algorithm changes.
    sources=['scaling.py','scaling_resources.py','reference.py','fast_solver.py','fast_geometry.py',
             'data.py','solver.py','benchmark.py']
    settings['sources_sha256']=hashlib.sha256(b''.join(
        (Path(__file__).parent/name).read_bytes() for name in sources)).hexdigest()
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['settings'] != settings:
        raise ValueError('Settings/source changed: use a new output directory')
    write_json(manifest,dict(settings=settings,python=platform.python_version(),
        torch=str(torch.__version__),numpy=np.__version__,cpu=platform.processor(),
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
        timing='resident input GMMs; includes initialization, all starts, validation; excludes EM, startup, metrics, plots',
        memory='sampled solver RSS absolute and extra; CUDA allocated/reserved; host watchdog includes startup',
        interpretation='comparison to full-tuple MW2 implementation, not all possible MW2 algorithms'))
    torch.set_num_threads(args.threads)
    rows,plans,trajectory=[],[],[]
    for k in args.component_counts:
        for seed in args.seeds:
            case=root/f'K_{k}'/f'seed_{seed}';case.mkdir(parents=True,exist_ok=True)
            with threadpool_limits(limits=args.threads):
                paths,fits=prepare_data(case/'inputs',k,seed,args)
            inputs=[load_gmm(p) for p in paths]
            fit_ms=sum(f['fit_ms'] for f in fits)
            kb=4*k-3
            records={}
            if args.task in ('all','barycenters'):
                for method in args.methods:
                    folder=case/method;folder.mkdir(exist_ok=True)
                    spec=dict(settings,method=method,inputs=paths,Kb=kb,seed=seed)
                    write_json(folder/'spec.json',spec)
                    result_path=folder/'result.json'
                    # Minimum bytes of materialized tuple indices, coordinates,
                    # covariances, barycenter parameters and costs, not peak estimate.
                    minimum_bytes=k**4*(4+4*2+4*4+2+4+1)*8
                    if result_path.exists() and not args.no_resume:
                        result=json.loads(result_path.read_text())
                    elif method=='MW2' and args.max_tuples and k**4>args.max_tuples:
                        result=dict(status='skipped_tuple_budget',total_ms=None)
                        write_json(result_path,result)
                    elif method=='MW2' and args.memory_limit_gb and minimum_bytes>args.memory_limit_gb*1024**3:
                        result=dict(status='skipped_memory_estimate',total_ms=None,
                                    tuple_array_lower_bound_bytes=minimum_bytes)
                        write_json(result_path,result)
                    else:
                        print(f'K={k} seed={seed} {method}, tuples={k**4:,}, Kb={kb}',flush=True)
                        result=run_worker(folder/'spec.json',args.timeout_s,args.memory_limit_gb,args.threads)
                    result.update(result.get('phase_ms', {}))
                    records[method]=dict(K=k,seed=seed,tuple_count=k**4,
                                         requested_Kb=kb,fit_ms=fit_ms,
                                         em_all_converged=all(f['converged'] for f in fits),**dict(result, method=method))
                    print(f"  {method}: {result['status']}, total_ms={result.get('total_ms')}",flush=True)
                    # Persist failures immediately; do not wait for evaluation.
                    write_csv(root/'metrics.csv',rows+list(records.values()))
                reference=None
                if records.get('MW2',{}).get('status')=='ok':
                    reference=load_gmm(case/'MW2'/'barycenter.npz')
                from .plotting import density_grid,draw_density
                import matplotlib.pyplot as plt
                fig,axes=plt.subplots(1,4+len(args.methods),figsize=(3*(4+len(args.methods)),3))
                for ax,g,name in zip(axes[:4],inputs,SHAPE_NAMES):
                    x,d=density_grid(g,(0.,1.),128);draw_density(ax,x,d);ax.set_title(name)
                for ax,method in zip(axes[4:],args.methods):
                    record=records[method];folder=case/method;ax.set_title(method)
                    if record['status'] in ('ok','numerical_stop') and (folder/'barycenter.npz').exists():
                        g=load_gmm(folder/'barycenter.npz')
                        with threadpool_limits(limits=args.threads):
                            record.update(evaluate(g,inputs,args.weights,args,seed,reference))
                        if method=='MW2':
                            record.update(relative_mw2_gap=0.,mw2_to_reference=0.,sw2_to_reference=0.,density_l1_to_reference=0.)
                        x,d=density_grid(g,(0.,1.),128);draw_density(ax,x,d)
                        if args.evaluate_checkpoints and (folder/'checkpoints.csv').exists():
                            import csv
                            for point in csv.DictReader((folder/'checkpoints.csv').open()):
                                step=int(point['iteration']);cp=load_gmm(folder/f'step{step:05d}.npz')
                                with threadpool_limits(limits=args.threads):
                                    metrics=evaluate(cp,inputs,args.weights,args,seed,reference)
                                trajectory.append(dict(K=k,seed=seed,method=method,step=step,
                                    elapsed_ms=float(point['elapsed_ms'])+record['initialization_ms'],
                                    selected_step=int(point['selected_step']),**metrics))
                    else:
                        ax.axis('off');ax.text(.5,.5,record['status'],ha='center',wrap=True)
                    write_json(folder/'metrics.json',record)
                fig.tight_layout();fig.savefig(case/'comparison.png',dpi=160);plt.close(fig)
                rows.extend(records.values())
                if trajectory: write_csv(root/'checkpoints.csv',trajectory)
            if args.task in ('all','plans'):
                plan_file=case/'plans.json'
                if plan_file.exists() and not args.no_resume:
                    these=json.loads(plan_file.read_text())
                else:
                    with threadpool_limits(limits=args.threads):
                        these=plan_benchmark(inputs,args,seed,k)
                    write_json(plan_file,these)
                plans.extend(these)
            make_outputs(root,rows,plans)
    print(f'Saved results to {root}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('results/barycenter_scaling'))
    parser.add_argument('--task',choices=('all','barycenters','plans'),default='all')
    parser.add_argument('--component-counts',type=int,nargs='+',default=[5,10,20,30,50,100])
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=['MW2','min-LSOT-B','min-LSOT-B1D'])
    parser.add_argument('--weights',type=float,nargs=4,default=[.25]*4)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--image-dir',type=Path)
    parser.add_argument('--em-steps',type=int,default=300)
    parser.add_argument('--em-tolerance',type=float,default=.001)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--probe-steps',type=int,nargs=2,default=[25,50])
    parser.add_argument('--learning-rate',type=float,default=.003)
    parser.add_argument('--L',type=int,default=100)
    parser.add_argument('--validation-L',type=int,default=512)
    parser.add_argument('--test-L',type=int,default=512)
    parser.add_argument('--test-kinds',nargs='+',choices=['B','B1D','Mix','SMix'],default=['B','B1D'])
    parser.add_argument('--sampling',choices=['fixed','fresh','partial'],default='fresh')
    parser.add_argument('--validate-every',type=int,default=10)
    parser.add_argument('--refresh-every',type=int,default=10)
    parser.add_argument('--refresh-fraction',type=float,default=.25)
    parser.add_argument('--checkpoints',type=int,nargs='*',default=[25,50,100,200])
    parser.add_argument('--evaluate-checkpoints',action='store_true',help='Post-hoc metrics, excluded from timer')
    parser.add_argument('--initialization-iterations',type=int,default=10)
    parser.add_argument('--gaussian-iterations',type=int,default=100)
    parser.add_argument('--gaussian-tolerance',type=float,default=1e-8)
    parser.add_argument('--eval-samples',type=int,default=1024)
    parser.add_argument('--plan-repeats',type=int,default=5)
    parser.add_argument('--plan-warmups',type=int,default=1)
    parser.add_argument('--timeout-s',type=float,default=600.)
    parser.add_argument('--memory-limit-gb',type=float,default=8.)
    parser.add_argument('--max-tuples',type=int,default=0,help='0 disables optional tuple cap; skips are never OOM')
    parser.add_argument('--no-resume',action='store_true')
    preliminary,_=parser.parse_known_args(argv)
    if preliminary.worker:
        worker(preliminary.worker);return
    if preliminary.config:
        config=json.loads(preliminary.config.read_text())
        unknown=set(config)-{a.dest for a in parser._actions}
        if unknown: parser.error(f'Unknown config keys: {sorted(unknown)}')
        parser.set_defaults(**config)
    args=parser.parse_args(argv)
    args.output_dir=Path(args.output_dir)
    if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    if (min(*args.component_counts,args.threads,args.em_steps,args.L,args.validation_L,args.test_L,
            args.validate_every,args.refresh_every,args.initialization_iterations,args.gaussian_iterations,
            args.eval_samples,args.plan_repeats)<1 or min(args.seeds)<0 or args.plan_warmups<0
            or not 0<args.probe_steps[0]<args.probe_steps[1]<args.steps
            or min(args.learning_rate,args.em_tolerance)<=0
            or min(args.gaussian_tolerance,args.timeout_s,args.memory_limit_gb,args.max_tuples)<0
            or not 0<args.refresh_fraction<=1 or min(args.weights)<=0
            or not np.isclose(sum(args.weights),1) or any(s<0 for s in args.checkpoints)):
        parser.error('Invalid settings; use positive four-input weights and 0 < probe1 < probe2 < steps')
    if len(set(args.methods)) != len(args.methods):parser.error('Duplicate methods')
    run(args)


if __name__=='__main__':
    main()
