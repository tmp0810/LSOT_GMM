"""Reproduce the selected fast LP-free initialization + Riemannian pipelines.

These settings were tuned on K=30/data-seed=0 and are not universal defaults.
All initializer construction/selection and optimizer validation enter timing.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .data import load_gmm, save_gmm
from .mixed_fast_solver import balanced_search, pad, solve_mixed
from .plotting import density_grid, plt
from .scaling import evaluate
from .solver import mw2_objective


PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE / 'configs' / 'mixed_fast.json'
CASES = ('images', 'synthetic')


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--cases', nargs='+', choices=CASES)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--no-plots', dest='plots', action='store_false', default=None)
    args = parser.parse_args(argv)
    try:
        defaults = json.loads(DEFAULT_CONFIG.read_text())
        config = json.loads(args.config.read_text())
        unknown = config.keys() - (defaults.keys() | {'source_dir'})
        if unknown:
            raise ValueError(f'Unknown config keys: {", ".join(sorted(unknown))}')
        settings = defaults | config
        settings.setdefault('source_dir', str(PACKAGE / 'assets' / 'mixed_fast'))
        for key in ('cases', 'source_dir', 'output_dir', 'repeats', 'plots'):
            value = getattr(args, key)
            if value is not None:
                settings[key] = str(value) if isinstance(value, Path) else value
        if type(settings['repeats']) is not int or settings['repeats'] < 1:
            raise ValueError('repeats must be a positive integer')
        if not settings['cases'] or any(case not in CASES for case in settings['cases']):
            raise ValueError('cases must contain images and/or synthetic')
        if type(settings['plots']) is not bool:
            raise ValueError('plots must be true or false')
        for case in settings['cases']:
            cfg = settings['settings'][case]
            for section in ('initialization', 'optimization'):
                expected = defaults['settings'][case][section].keys()
                if cfg[section].keys() != expected:
                    raise ValueError(f'{case}.{section} requires {", ".join(sorted(expected))}')
            for key, value in cfg['initialization'].items():
                if type(value) is not int or value < 1:
                    raise ValueError(f'{case}.initialization.{key} must be a positive integer')
            opt = cfg['optimization']
            if type(opt['steps']) is not int or opt['steps'] < 1 or type(opt['L']) is not int or opt['L'] < 1:
                raise ValueError(f'{case}: steps and L must be positive integers')
            if opt['sampling'] not in ('fixed', 'partial') or (opt['sampling'] == 'partial' and opt['L'] < 4):
                raise ValueError(f'{case}: use fixed sampling or partial sampling with L >= 4')
            if any(not np.isfinite(opt[key]) or opt[key] < 0 for key in ('rate', 'mean', 'mass')):
                raise ValueError(f'{case}: rate, mean and mass must be finite and nonnegative')
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(f'Invalid mixed_fast config: {exc}')
    return settings


def save_plots(out, case, models, history, selected_step):
    # Use common axes and contour levels so density heights are comparable.
    grids = [density_grid(gmm, (0., 1.), 160) for _, gmm, _ in models]
    levels = np.linspace(0, max(values.max() for _, values in grids), 10)[1:]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), layout='constrained')
    for ax, (title, _, value), (x, values) in zip(axes, models, grids):
        ax.contour(x, x, values, levels=levels, cmap='viridis')
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', title=f'{title}\nMW2 objective: {value:.6f}')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(case.capitalize())
    fig.savefig(out / 'comparison.png', dpi=170)
    fig.savefig(out / 'comparison.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5), layout='constrained')
    ax.plot(history['step'], history['validation'], label='Validation')
    ax.plot(history['step'], history['best_validation'], '--', label='Best so far')
    ax.axvline(selected_step, color='gray', linestyle=':', label=f'Selected step {selected_step}')
    ax.set(xlabel='Optimizer step', ylabel='Validation min-LSOT-B objective')
    ax.legend()
    fig.savefig(out / 'convergence.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    settings = parse_config(argv)
    with threadpool_limits(1):
        run(settings)


def run(settings):
    torch.set_num_threads(1)
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    rows=[]
    for case in settings['cases']:
        source=Path(settings['source_dir'])/case
        inputs=[load_gmm(source/'inputs'/f'input_{j}.npz') for j in range(4)]
        ref=load_gmm(source/'MW2/barycenter.npz')
        cfg=settings['settings'][case];out=output_dir/case;out.mkdir(parents=True,exist_ok=True)
        ea=SimpleNamespace(test_kinds=['B'],test_L=512,eval_samples=1024)
        for repeat in range(settings['repeats']):
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
            rows.append(row);pd.DataFrame(rows).to_csv(output_dir/'metrics.csv',index=False)
            save_gmm(out/'initial.npz',initial);save_gmm(out/'optimized.npz',result)
            trace = pd.DataFrame(history)
            trace.to_csv(out/'history.csv',index=False)
            print(json.dumps(row),flush=True)
        if settings['plots']:
            save_plots(out, case, [
                ('MW2 reference', ref, mw2_objective(inputs, [.25] * 4, ref)),
                ('Initialization', initial, row['initial_mw2_objective']),
                ('Mixed fast', result, row['mw2_objective']),
            ], trace, row['selected_step'])


if __name__=='__main__':
    main()
