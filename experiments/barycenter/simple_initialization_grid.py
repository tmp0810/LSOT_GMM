"""Bilinear barycenter grids with full MW2 and greedy min-LSOT-B."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from lsot.gaussians import GMM
from .data import gaussian_example, grid_nodes, load_gmm, save_gmm
from .mixed_fast import PACKAGE, fit_mw2
from .mixed_fast_solver import solve_mixed
from .plotting import density_grid, plt, save_gaussian_overlay, save_method_grid
from .run import fingerprint, read_record, write_json
from .scaling import evaluate, prepare_data
from .simple_initialization import PHASES, configurations, fit_initialized, save_case_plots


DEFAULT = PACKAGE/'configs/simple_initialization_grid.json'
METHODS = ('MW2', 'mixed_fast', 'greedy_mean_spread0.25')
CASES = ('gaussian', 'synthetic', 'images')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--cases', nargs='+', choices=CASES)
    parser.add_argument('--component-counts', nargs='+', type=int)
    parser.add_argument('--grid-size', type=int)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--steps', type=int, help='Override the optimization budget for all cases')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plots', dest='plots', action='store_false', default=None)
    args = parser.parse_args(argv)
    defaults = json.loads(DEFAULT.read_text())
    loaded = json.loads(args.config.read_text())
    if loaded.keys()-defaults.keys():
        parser.error('Unknown configuration keys')
    config = defaults | loaded
    for key in ('cases', 'component_counts', 'grid_size', 'repeats', 'steps', 'output_dir', 'plots'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    for key in ('cases', 'component_counts', 'methods'):
        values = config[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            parser.error(f'{key} must be a nonempty list of distinct entries')
    if set(config['cases'])-set(CASES) or set(config['methods'])-set(METHODS):
        parser.error('Unknown case or method')
    if 'gaussian' in config['cases'] and 'mixed_fast' in config['methods']:
        parser.error('The Gaussian control supports MW2 and greedy_mean_spread0.25')
    if 'MW2' not in config['methods']:
        parser.error('Include MW2 to evaluate and compare the grids')
    if any(type(k) is not int or k < 4 for k in config['component_counts']):
        parser.error('Component counts must be integers >= 4')
    for key, minimum in [('grid_size', 3), ('repeats', 1), ('density_grid', 16)]:
        if type(config[key]) is not int or config[key] < minimum:
            parser.error(f'{key} must be an integer >= {minimum}')
    steps = config['steps']
    if steps is not None and (type(steps) is not int or steps < 1):
        parser.error('steps must be a positive integer')
    overrides = config['optimization']
    if not isinstance(overrides, dict) or set(overrides)-set(CASES):
        parser.error('optimization must contain per-case settings for gaussian, images or synthetic')
    for case, settings in overrides.items():
        if not isinstance(settings, dict) or set(settings)-{'L', 'sampling'}:
            parser.error(f'optimization.{case} accepts only L and sampling')
        if 'L' in settings and (type(settings['L']) is not int or settings['L'] < 4):
            parser.error(f'optimization.{case}.L must be an integer >= 4')
        if 'sampling' in settings and settings['sampling'] not in ('fixed', 'partial'):
            parser.error(f'optimization.{case}.sampling must be fixed or partial')
    for key in ('mw2', 'data'):
        if not isinstance(config[key], dict) or config[key].keys()-defaults[key].keys():
            parser.error(f'Invalid {key} options')
        config[key] = defaults[key] | config[key]
    mw2, data = config['mw2'], config['data']
    if (type(config['plots']) is not bool or type(mw2['gaussian_iterations']) is not int
            or mw2['gaussian_iterations'] < 1 or not np.isfinite(mw2['gaussian_tolerance'])
            or mw2['gaussian_tolerance'] < 0 or not 0 < data['synthetic_spread'] < 1
            or type(data['em_steps']) is not int or data['em_steps'] < 1
            or not np.isfinite(data['em_tolerance']) or data['em_tolerance'] <= 0):
        parser.error('Invalid plot, MW2, or data settings')
    return config


def load_inputs(root, case, k, config):
    if k == 30 and config['data'] == json.loads(DEFAULT.read_text())['data']:
        source = PACKAGE/'assets/mixed_fast'/case/'inputs'
        return [load_gmm(source/f'input_{j}.npz') for j in range(4)], 'bundled mixed_fast K=30'
    paths, _ = prepare_data(root/'inputs', k, 0,
                            SimpleNamespace(data=case, image_dir=None, **config['data']))
    return [load_gmm(p) for p in paths], 'scaling data preparation'


def fit_node(name, inputs, weights, initializers, optimization, mw2):
    if len(inputs) == 1:
        return inputs[0], None, [], dict(status='anchor', total_ms=0.)
    if name == 'MW2':
        solved, times = fit_mw2(inputs, mw2, weights)
        times['mw2_finalize_ms'] = times['total_ms']-sum(
            times[k] for k in ('tuple_setup_ms', 'gaussian_cost_ms', 'constraint_setup_ms', 'lp_ms'))
        return solved.gmm, None, [], dict(
            status='ok', num_tuples=int(np.prod([g.count for g in inputs])),
            covariance_iterations=solved.covariance_iterations,
            covariance_tolerance_met=solved.covariance_tolerance_met,
            marginal_error=solved.marginal_error, **times)
    initial, result, history, stats = fit_initialized(inputs, initializers[name], optimization, weights)
    return result, initial, history, stats


def write_tables(root, records):
    frame = pd.DataFrame(records)
    frame.to_csv(root/'metrics.csv', index=False)
    # Corners are copied, so leave them out of averages and speedup.
    active = frame[frame['active_inputs'] > 1]
    groups = ['case', 'K', 'name']
    columns = ['total_ms', 'final_mw2', 'initial_mw2', 'test_min_B',
               'relative_mw2_gap', *(key for key, _ in PHASES)]
    columns = [key for key in columns if key in active]
    summary = active.groupby(groups, sort=False)[columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    # Summing first per repeat gives end-to-end solver time over all grid nodes.
    per_repeat = active.groupby([*groups, 'repeat'])['total_ms'].sum()
    grid = per_repeat.groupby(level=groups).agg(['mean', 'std', 'size']).rename(columns={
        'mean': 'grid_total_ms_mean', 'std': 'grid_total_ms_std', 'size': 'repeats'})
    summary = summary.merge(grid.reset_index(), on=groups)
    mw2_times = summary[summary['name']=='MW2'].set_index(['case', 'K'])['grid_total_ms_mean']
    summary['speedup_vs_mw2'] = [mw2_times.loc[c, k]/t for c,k,t in
                                  zip(summary['case'], summary['K'], summary['grid_total_ms_mean'])]
    summary.to_csv(root/'summary.csv', index=False)
    return summary


def run_grid(root, case, k, config, settings):
    root.mkdir(parents=True, exist_ok=True)
    gaussian = gaussian_example() if case == 'gaussian' else None
    if gaussian is None:
        inputs, provenance = load_inputs(root, case, k, config)
        size = config['grid_size']
        bounds = (0., 1.)
        nodes = grid_nodes(size)
    else:
        inputs, provenance = gaussian.inputs, 'original three-Gaussian control'
        size = 1
        bounds = gaussian.bounds
        nodes = [(0, 0, None, None, np.full(3, 1/3))]
    optimization = settings['optimization'] | config['optimization'].get(case, {})
    if config['steps'] is not None:
        optimization['steps'] = config['steps']
    options = {c['name']: c for c in configurations(settings['initialization'])}
    methods = config['methods']
    manifest = dict(schema=1, case=case, K=k, seed=0, provenance=provenance,
                    size=size, repeats=config['repeats'], density_grid=config['density_grid'],
                    initializers={n: options[n] for n in methods if n != 'MW2'}, methods=methods,
                    optimization=optimization, mw2=config['mw2'], cpu_threads=1)
    key = fingerprint(manifest, SimpleNamespace(inputs=inputs))
    previous = read_record(root/'run_manifest.json')
    if previous and previous['fingerprint'] != key:
        raise ValueError(f'Settings changed in {root}; choose a new --output-dir')
    write_json(root/'run_manifest.json', dict(fingerprint=key, settings=manifest))
    # Preserve the exact prepared cache; load_gmm renormalizes its weights.
    if gaussian is not None or provenance == 'bundled mixed_fast K=30':
        (root/'inputs').mkdir(exist_ok=True)
        for j, g in enumerate(inputs):
            save_gmm(root/'inputs'/f'input_{j}.npz', g)
    evaluation = SimpleNamespace(test_kinds=['B'], test_L=512, eval_samples=1024)
    records = []
    for repeat in range(config['repeats']):
        rep = root/f'repeat_{repeat:02d}'
        for row, col, tx, ty, weights in nodes:
            tag = f'r{row:02d}_c{col:02d}'
            active = np.flatnonzero(weights)
            gs, omega = [inputs[j] for j in active], weights[active]
            shift = (row*size + col + repeat) % len(methods)
            fitted, cached = {}, {}
            for name in methods[shift:]+methods[:shift]:
                path = rep/name/'barycenters'/f'{tag}.npz'
                record = read_record(rep/name/'records'/f'{tag}.json')
                if record is not None and path.is_file():
                    cached[name] = record
                    continue
                print(f'{case} K={k} repeat={repeat+1} {tag} {name}', flush=True)
                fitted[name] = fit_node(name, gs, omega, options, optimization, config['mw2'])
            if fitted:
                ref = fitted['MW2'][0] if 'MW2' in fitted else load_gmm(
                    rep/'MW2'/'barycenters'/f'{tag}.npz')
                for name, (result, initial, history, stats) in fitted.items():
                    folder = rep/name
                    for sub in ('barycenters', 'records', 'initializations', 'history'):
                        (folder/sub).mkdir(parents=True, exist_ok=True)
                    metrics = evaluate(result, gs, omega, evaluation, 0, ref, bounds=bounds)
                    record = dict(case=case, K=k, name=name, repeat=repeat, row=row, col=col,
                                  tx=tx, ty=ty, input_weights=weights.tolist(),
                                  active_inputs=len(active), output_components=result.count,
                                  final_mw2=metrics['mw2_objective'], **stats, **metrics)
                    x, z = density_grid(result, bounds, config['density_grid'])
                    save_gmm(folder/'barycenters'/f'{tag}.npz', result, x=x, density=z, input_weights=weights)
                    if initial is not None:
                        save_gmm(folder/'initializations'/f'{tag}.npz', initial, input_weights=weights)
                        start = evaluate(initial, gs, omega, evaluation, 0)
                        record.update(initial_mw2=start['mw2_objective'],
                                      initial_test_min_B=start['test_min_B'],
                                      initial_validation=history[0]['validation'])
                        pd.DataFrame(history).to_csv(folder/'history'/f'{tag}.csv', index=False)
                    write_json(folder/'records'/f'{tag}.json', record)
                    cached[name] = record
            records.extend(cached[n] for n in methods)
    summary = write_tables(root, records)
    if config['plots']:
        rep = root/f'repeat_{config["repeats"]-1:02d}'
        middle = size//2
        models = {}
        for name in methods:
            subset = [r for r in records if r['name']==name and r['repeat']==config['repeats']-1]
            save_method_grid(rep/name, subset, size)
            center = next(r for r in subset if r['row']==r['col']==middle)
            model = load_gmm(rep/name/'barycenters'/f'r{middle:02d}_c{middle:02d}.npz')
            if gaussian is not None:
                save_gaussian_overlay(rep/name, gaussian, model)
            models[name] = (model, center['final_mw2'])
        save_case_plots(root, summary, models, grid_nodes=None if gaussian else len(nodes)-4,
                        bounds=bounds)
    return records


def plot_scaling(root, summary):
    for case, frame in summary[summary['case'] != 'gaussian'].groupby('case'):
        fig, ax = plt.subplots(figsize=(5.5, 3.8), layout='constrained')
        for name, group in frame.groupby('name', sort=False):
            group = group.sort_values('K')
            ax.errorbar(group['K'], group['grid_total_ms_mean']/1000,
                        yerr=group['grid_total_ms_std'].fillna(0)/1000,
                        marker='o', capsize=3, label=name)
        ax.set(xlabel='Components per input GMM (K)', ylabel='Total grid solver time (s)', yscale='log')
        ax.set_xticks(sorted(frame['K'].unique()))
        ax.legend(fontsize=8)
        fig.savefig(root/case/'scaling_runtime.png', dpi=170)
        plt.close(fig)


def main(argv=None):
    config = parse_args(argv)
    torch.set_num_threads(1)
    root = Path(config['output_dir'])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'experiment.json', config)
    defaults = json.loads((PACKAGE/'configs/mixed_fast.json').read_text())['settings']
    defaults['gaussian'] = dict(initialization={}, optimization=dict(
        steps=100, L=100, sampling='fixed', rate=10., mean=1e-6, mass=.01))
    tiny = GMM.from_numpy([1.], [[.5, .5]], [[[.01, 0.], [0., .01]]])
    rows = []
    with threadpool_limits(1):
        fit_mw2([tiny]*2, dict(gaussian_iterations=2, gaussian_tolerance=0.), [.5, .5])
        if set(config['methods']) != {'MW2'}:
            solve_mixed([tiny]*2, tiny, dict(L=4, steps=1, sampling='fixed', rate=1., mean=1e-6, mass=.01))
        for case in config['cases']:
            for k in (1,) if case == 'gaussian' else config['component_counts']:
                folder = root/case if case == 'gaussian' else root/case/f'K{k}'
                rows.extend(run_grid(folder, case, k, config, defaults[case]))
                summary = write_tables(root, rows)
        if config['plots']:
            plot_scaling(root, summary)


if __name__ == '__main__':
    main()
