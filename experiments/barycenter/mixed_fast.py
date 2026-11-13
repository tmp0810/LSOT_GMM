"""Compare mixed min-LSOT-B and multi-marginal MW2 barycenters.

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

from lsot.gaussians import GMM
from .data import load_gmm, save_gmm
from .mixed_fast_solver import balanced_search, pad, solve_mixed
from .plotting import density_grid, plt
from .reference import mw2_barycenter
from .scaling import evaluate
from .solver import mw2_objective


PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE / 'configs' / 'mixed_fast.json'
CASES = ('images', 'synthetic')
WEIGHTS = (.25,) * 4
MIXED = 'min-LSOT-B'


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--cases', nargs='+', choices=CASES)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--no-plots', dest='plots', action='store_false', default=None)
    parser.add_argument('--skip-mw2', dest='benchmark_mw2', action='store_false', default=None,
                        help='Use the saved MW2 reference without benchmarking its runtime')
    args = parser.parse_args(argv)
    try:
        defaults = json.loads(DEFAULT_CONFIG.read_text())
        config = json.loads(args.config.read_text())
        unknown = config.keys() - (defaults.keys() | {'source_dir'})
        if unknown:
            raise ValueError(f'Unknown config keys: {", ".join(sorted(unknown))}')
        settings = defaults | config
        settings.setdefault('source_dir', str(PACKAGE / 'assets' / 'mixed_fast'))
        for key in ('cases', 'source_dir', 'output_dir', 'repeats', 'plots', 'benchmark_mw2'):
            value = getattr(args, key)
            if value is not None:
                settings[key] = str(value) if isinstance(value, Path) else value
        if type(settings['repeats']) is not int or settings['repeats'] < 1:
            raise ValueError('repeats must be a positive integer')
        if not settings['cases'] or any(case not in CASES for case in settings['cases']):
            raise ValueError('cases must contain images and/or synthetic')
        for key in ('plots', 'benchmark_mw2'):
            if type(settings[key]) is not bool:
                raise ValueError(f'{key} must be true or false')
        mw2 = settings['mw2'] = defaults['mw2'] | settings['mw2']
        if mw2.keys() != defaults['mw2'].keys():
            raise ValueError('mw2 accepts gaussian_iterations and gaussian_tolerance')
        if type(mw2['gaussian_iterations']) is not int or mw2['gaussian_iterations'] < 1:
            raise ValueError('mw2.gaussian_iterations must be a positive integer')
        tol = mw2['gaussian_tolerance']
        if not np.isfinite(tol) or tol < 0:
            raise ValueError('mw2.gaussian_tolerance must be finite and nonnegative')
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


def fit_mw2(inputs, config, weights=WEIGHTS):
    timings = {}
    start = time.perf_counter()
    solved = mw2_barycenter(inputs, weights, iterations=config['gaussian_iterations'],
                            tolerance=config['gaussian_tolerance'], fast=True,
                            timings=timings, lp_threads=1)
    timings['total_ms'] = 1000 * (time.perf_counter() - start)
    return solved, timings


def fit_mixed(inputs, config):
    start = time.perf_counter()
    candidate, count = balanced_search(inputs, **config['initialization'])
    initial = pad(candidate.gmm, sum(g.count for g in inputs) - 3, 0)
    init_ms = 1000 * (time.perf_counter() - start)
    result, history, stats = solve_mixed(inputs, initial, config['optimization'])
    return initial, result, history, dict(
        candidates=count, init_ms=init_ms, total_ms=init_ms + stats['optimizer_ms'], **stats)


def write_metrics(output_dir, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / 'metrics.csv', index=False)
    columns = ['total_ms', 'mw2_objective', 'test_min_B', 'relative_mw2_gap',
               'mw2_to_reference', 'sw2_to_reference', 'density_l1_to_reference',
               'init_ms', 'optimizer_ms', 'tuple_setup_ms', 'gaussian_cost_ms',
               'constraint_setup_ms', 'lp_ms']
    grouped = frame.groupby(['case', 'method'], sort=False)
    summary = grouped[[key for key in columns if key in frame]].agg(['mean', 'std'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary['repeats'] = grouped.size()
    summary = summary.reset_index()
    mw2_ms = summary[summary['method'] == 'MW2'].set_index('case')['total_ms_mean']
    summary['speedup_vs_mw2'] = summary['case'].map(mw2_ms) / summary['total_ms_mean']
    summary.to_csv(output_dir / 'summary.csv', index=False)
    return summary


def save_runtime(out, case_summary):
    fig, ax = plt.subplots(figsize=(5, 3.6), layout='constrained')
    x = case_summary['method'].tolist()
    seconds = case_summary['total_ms_mean'] / 1000
    bars = ax.bar(x, seconds, yerr=case_summary['total_ms_std'].fillna(0) / 1000, capsize=4)
    ax.bar_label(bars, fmt='%.2f s', padding=4)
    ax.margins(y=.2)
    ax.set(ylabel='Total solver time (s)', title='Mean over sequential repeats')
    fig.savefig(out / 'runtime.png', dpi=170)
    plt.close(fig)


def run(settings):
    torch.set_num_threads(1)
    output_dir = Path(settings['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    # Warm both solver paths on an independent tiny problem outside the timers.
    tiny = GMM.from_numpy([1.], [[.5, .5]], [[[.01, 0.], [0., .01]]])
    if settings['benchmark_mw2']:
        fit_mw2([tiny] * 4, dict(gaussian_iterations=2, gaussian_tolerance=0.))
    solve_mixed([tiny] * 4, tiny, dict(L=4, steps=1, sampling='fixed', rate=1., mean=1e-6, mass=.01))

    rows = []
    for case in settings['cases']:
        source = Path(settings['source_dir']) / case
        inputs = [load_gmm(source / 'inputs' / f'input_{j}.npz') for j in range(4)]
        benchmark = settings['benchmark_mw2']
        if not benchmark:
            ref = load_gmm(source / 'MW2/barycenter.npz')
        cfg = settings['settings'][case]
        out = output_dir / case
        out.mkdir(parents=True, exist_ok=True)
        ea = SimpleNamespace(test_kinds=['B'], test_L=512, eval_samples=1024)
        for repeat in range(settings['repeats']):
            print(f'{case}: repeat {repeat + 1}/{settings["repeats"]}'
                  + (' (MW2 and min-LSOT-B)' if benchmark else ' (min-LSOT-B)'), flush=True)
            # Alternate the solver order. Neither fit receives the other's output.
            if benchmark and repeat % 2 == 0:
                solved, timings = fit_mw2(inputs, settings['mw2'])
            initial, result, history, stats = fit_mixed(inputs, cfg)
            if benchmark:
                if repeat % 2 == 1:
                    solved, timings = fit_mw2(inputs, settings['mw2'])
                ref = solved.gmm
                mw2_row = dict(case=case, repeat=repeat, method='MW2', status='ok',
                               output_components=ref.count,
                               num_tuples=int(np.prod([g.count for g in inputs])),
                               covariance_iterations=solved.covariance_iterations,
                               covariance_tolerance_met=solved.covariance_tolerance_met,
                               marginal_error=solved.marginal_error, **timings,
                               **evaluate(ref, inputs, WEIGHTS, ea, 0, ref))
                rows.append(mw2_row)
                save_gmm(out / 'mw2.npz', ref)
                print(json.dumps(mw2_row), flush=True)
            row = dict(case=case, repeat=repeat, method=MIXED, output_components=result.count,
                       **stats, **evaluate(result, inputs, WEIGHTS, ea, 0, ref))
            init_metrics = evaluate(initial, inputs, WEIGHTS, ea, 0, ref)
            row.update(initial_mw2_objective=init_metrics['mw2_objective'],
                       initial_test_min_B=init_metrics['test_min_B'],
                       improvement_percent=100*(1-row['mw2_objective']/init_metrics['mw2_objective']))
            rows.append(row)
            summary = write_metrics(output_dir, rows)
            save_gmm(out/'initial.npz',initial);save_gmm(out/'optimized.npz',result)
            trace = pd.DataFrame(history)
            trace.to_csv(out/'history.csv',index=False)
            print(json.dumps(row),flush=True)
        if settings['plots']:
            save_plots(out, case, [
                ('MW2 reference', ref, mw2_objective(inputs, WEIGHTS, ref)),
                ('Initialization', initial, row['initial_mw2_objective']),
                ('Mixed fast', result, row['mw2_objective']),
            ], trace, row['selected_step'])
            if benchmark:
                save_runtime(out, summary[summary['case'] == case])


if __name__=='__main__':
    main()
