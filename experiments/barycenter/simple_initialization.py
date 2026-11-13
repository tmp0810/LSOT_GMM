"""Controlled K=30/seed=0 initializer ablation; no LP in initialization/training.

Run with python -m experiments.barycenter.simple_initialization.
MW2 is a separately timed baseline; it never selects a min-LSOT-B start/iterate.
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

from lsot.plans import SparsePlan
from lsot.gaussians import GMM
from .data import load_gmm, save_gmm
from .fast_geometry import pair_costs
from .mixed_fast import PACKAGE, WEIGHTS, density_grid, fit_mw2, plt, save_plots
from .mixed_fast_solver import (balanced_search, from_tuples, singleton, join,
                                pad, solve_mixed, TupleCandidate)
from .scaling import evaluate


def directions(n):
    angle = np.arange(n) * np.pi / n
    return np.column_stack([np.cos(angle), np.sin(angle)])


def quantile_candidate(inputs, coordinates):
    orders = [np.argsort(x, kind='stable') for x in coordinates]
    cumulative = [np.r_[0., np.cumsum(g.weights.numpy()[o])[:-1], 1.]
                  for g, o in zip(inputs, orders)]
    edges = np.unique(np.concatenate(cumulative))
    mass = np.diff(edges)
    mid = (edges[:-1] + edges[1:]) / 2
    indices = np.column_stack([o[np.searchsorted(c[1:], mid, side='right').clip(max=len(o)-1)]
                               for o, c in zip(orders, cumulative)])
    return from_tuples(inputs, indices, mass)


def greedy_plan(a, b, geometry):
    """Exhaust residual masses along globally sorted pair costs (not an LP)."""
    if geometry == 'mean':
        costs = ((a.means[:, None] - b.means[None]) ** 2).sum(-1).numpy()
    else:
        costs = pair_costs(a, b).detach().numpy()
    aa, bb = a.weights.numpy().copy(), b.weights.numpy().copy()
    rows, cols, masses = [], [], []
    for index in np.argsort(costs.ravel(), kind='stable'):
        i, j = divmod(int(index), b.count)
        mass = min(aa[i], bb[j])
        if mass > 1e-14:
            rows.append(i); cols.append(j); masses.append(mass)
            aa[i] -= mass; bb[j] -= mass
    p = SparsePlan(torch.tensor(rows), torch.tensor(cols),
                   torch.tensor(masses, dtype=torch.float64), (a.count, b.count))
    if p.marginal_error(a, b) > 1e-9:
        raise ValueError('Greedy marginal error')
    return p


def mean_slice_plan(a, b, n):
    candidates = [quantile_candidate([a, b], [g.means.numpy() @ v for g in (a, b)])
                  for v in directions(n)]
    best = min(candidates, key=lambda c: c.cost)
    return SparsePlan(torch.tensor(best.indices[:, 0]), torch.tensor(best.indices[:, 1]),
                      torch.tensor(best.mass), (a.count, b.count))


def arithmetic_join(a, b, plan, moment=False, spread=1., weights=None):
    """Direct moment averages, avoiding iterative Gaussian barycenters."""
    inputs = a.inputs + b.inputs
    indices = np.column_stack([a.indices[plan.rows.numpy()], b.indices[plan.cols.numpy()]])
    mass = plan.mass.numpy()
    means = np.stack([g.means.numpy()[indices[:, j]] for j, g in enumerate(inputs)], 1)
    covs = np.stack([g.covariances.numpy()[indices[:, j]] for j, g in enumerate(inputs)], 1)
    omega = np.ones(len(inputs)) if weights is None else np.asarray(weights, dtype=float)
    uniform = np.all(omega == omega[0])
    omega = omega/omega.sum()
    mean = means.mean(1) if uniform else np.einsum('j,tji->ti', omega, means)
    cov = covs.mean(1) if uniform else np.einsum('j,tjkl->tkl', omega, covs)
    if moment:
        delta = means - mean[:, None]
        if uniform:
            cov += spread * np.einsum('tji,tjk->tik', delta, delta) / len(inputs)
        else:
            cov += spread * np.einsum('j,tji,tjk->tik', omega, delta, delta)
    cross = np.sqrt(np.maximum(np.einsum('tjab,tba->tj', covs, cov) +
                     2*np.sqrt(np.maximum(np.linalg.det(covs)*np.linalg.det(cov)[:, None], 0)), 0))
    costs = ((means-mean[:, None])**2).sum(-1) + np.trace(covs, axis1=-2, axis2=-1) + \
            np.trace(cov, axis1=-2, axis2=-1)[:, None] - 2*cross
    result = GMM.from_numpy(mass, mean, cov)
    for j, g in enumerate(inputs):
        if np.max(np.abs(np.bincount(indices[:, j], mass, minlength=g.count)-g.weights.numpy())) > 1e-9:
            raise ValueError('Arithmetic tuple marginal error')
    return TupleCandidate(inputs, indices, mass, result,
                          float(mass @ (costs.mean(1) if uniform else costs @ omega)))


def initialize(inputs, cfg, weights=None):
    kind = cfg['kind']
    if kind == 'baseline':
        return balanced_search(inputs, weights=weights, **cfg['params'])
    if weights is not None and kind != 'greedy':
        raise ValueError('Weighted initialization supports the greedy and lifted tree strategies')
    if kind == 'quantile':
        vv = directions(cfg['n']) if cfg['n'] != 4 else np.array([[1, 2**.5], [1, 0], [0, 1], [1, -1]])
        candidates = [quantile_candidate(inputs, [g.means.numpy() @ v for g in inputs]) for v in vv]
    elif kind == 'pca':
        m = np.concatenate([g.means.numpy() for g in inputs])
        a = np.concatenate([g.weights.numpy() / 4 for g in inputs])
        centered = m - a @ m
        _, vec = np.linalg.eigh(centered.T @ (a[:, None] * centered))
        vv = [vec[:, 1], vec[:, 0], vec[:, 1]+vec[:, 0], vec[:, 1]-vec[:, 0]]
        candidates = [quantile_candidate(inputs, [g.means.numpy() @ v for g in inputs]) for v in vv]
    else:
        def merge(x, y, w):
            plan = (mean_slice_plan(x.gmm, y.gmm, cfg['n']) if kind == 'mean_tree'
                    else greedy_plan(x.gmm, y.gmm, cfg['geometry']))
            if cfg.get('representative') in ('arithmetic', 'moment'):
                return arithmetic_join(x, y, plan, cfg['representative'] == 'moment',
                                       cfg.get('spread', 1.), weights=w)
            return join(x, y, plan, w)

        if len(inputs) == 2:
            a, b = map(singleton, inputs)
            return merge(a, b, weights), 1
        omega = np.ones(len(inputs)) if weights is None else np.asarray(weights, dtype=float)
        candidates = []
        for order in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]:
            a, b, c, d = [singleton(inputs[i]) for i in order]
            w = omega[list(order)]
            candidates.append(merge(merge(a, b, w[:2]), merge(c, d, w[2:]), w))
    return min(candidates, key=lambda c: c.cost), len(candidates)


def configurations(baseline):
    yield dict(name='mixed_fast', kind='baseline', params=baseline)
    for n in [1, 4, 16, 64]:
        yield dict(name=f'mean_quantile_{n}', kind='quantile', n=n)
    yield dict(name='pca_quantile_4', kind='pca')
    for n in [4, 16, 64]:
        yield dict(name=f'mean_tree_{n}', kind='mean_tree', n=n)
    for geometry in ['mean', 'gaussian']:
        yield dict(name=f'greedy_{geometry}', kind='greedy', geometry=geometry)
        for representative in ['arithmetic', 'moment']:
            yield dict(name=f'greedy_{geometry}_{representative}', kind='greedy',
                       geometry=geometry, representative=representative)
        for spread in [.25, .5, .75]:
            yield dict(name=f'greedy_{geometry}_spread{spread}', kind='greedy',
                       geometry=geometry, representative='moment', spread=spread)
    for n in [64, 256, 1024]:
        yield dict(name=f'B_tree_{n}', kind='baseline',
                   params=dict(L=n, banks=1, width=1, root_width=1))


PHASES = (
    ('tuple_setup_ms', 'Tuple enumeration'),
    ('gaussian_cost_ms', 'Gaussian costs'),
    ('constraint_setup_ms', 'LP constraints'),
    ('lp_ms', 'LP solve'),
    ('mw2_finalize_ms', 'GMM construction / overhead'),
    ('init_ms', 'Initialization'),
    ('optimizer_ms', 'Optimization + checkpoint selection'),
)


def write_summary(root, rows, summary_rows=None):
    frame = pd.DataFrame(rows)
    frame.to_csv(root/'metrics.csv', index=False)
    if summary_rows is not None:
        frame = pd.DataFrame(summary_rows)
    columns = ('initial_mw2', 'final_mw2', 'total_ms', 'improvement_percent',
               'validation_objective', 'test_min_B', 'relative_mw2_gap',
               'sw2_to_reference', 'density_l1_to_reference',
               *(key for key, _ in PHASES))
    grouped = frame.groupby(['case', 'name'], sort=False)
    summary = grouped[[c for c in columns if c in frame]].agg(['mean', 'std'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary['repeats'] = grouped.size()
    summary = summary.reset_index()
    mw2_times = summary.loc[summary['name'] == 'MW2'].set_index('case')['total_ms_mean']
    summary['speedup_vs_mw2'] = summary['case'].map(mw2_times) / summary['total_ms_mean']
    summary.to_csv(root/'summary.csv', index=False)
    return summary


def save_case_plots(out, summary, models, *, grid_nodes=None):
    """Compare methods with common contour levels and separate timing scales."""
    preferred = ['MW2'] + [name for name in models if name != 'MW2']
    summary = summary.set_index('name').loc[[n for n in preferred if n in summary['name'].values]]
    order = summary.index.tolist()
    labels = {'mixed_fast': 'Mixed fast', 'greedy_mean_spread0.25': 'Greedy mean'}
    names = [labels.get(n, n) for n in order]
    fig, ax = plt.subplots(figsize=(max(5, len(order)*1.35), 3.8), layout='constrained')
    total = 'total_ms' if grid_nodes is None else 'grid_total_ms'
    seconds = summary[total+'_mean']/1000
    bars = ax.bar(range(len(order)), seconds, yerr=summary[total+'_std'].fillna(0)/1000, capsize=4)
    ax.bar_label(bars, fmt='%.2f s', padding=4)
    ax.set_xticks(range(len(names)), names, rotation=20 if len(names)>3 else 0)
    ax.set_ylabel('Total solver time (s)' if grid_nodes is None else 'Total grid solver time (s)')
    ax.margins(y=.2)
    fig.savefig(out/'runtime.png', dpi=170)
    plt.close(fig)

    groups = [[n for n in order if n == 'MW2'], [n for n in order if n != 'MW2']]
    groups = [g for g in groups if g]
    fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 4.2),
                             squeeze=False, layout='constrained')
    for ax, group in zip(axes[0], groups):
        base = np.zeros(len(group))
        for key, title in PHASES:
            col = key + '_mean'
            if col in summary:
                values = summary.loc[group, col].fillna(0).to_numpy()*(grid_nodes or 1)/1000
                if np.any(values):
                    ax.bar(range(len(group)), values, bottom=base, label=title)
                    base += values
        ax.set_xticks(range(len(group)), [labels.get(n, n) for n in group],
                      rotation=20 if len(group)>3 else 0)
        ax.set_ylabel('Mean time (s)' if grid_nodes is None else 'Mean time per grid (s)')
        ax.legend(fontsize=8)
    fig.savefig(out/'runtime_breakdown.png', dpi=170)
    plt.close(fig)

    # Individual comparisons remain available when running the full ablation.
    if len(models) > 5:
        return
    grids = [density_grid(models[n][0], (0., 1.), 160) for n in preferred]
    levels = np.linspace(0, max(z.max() for _, z in grids), 10)[1:]
    fig, axes = plt.subplots(1, len(grids), figsize=(3*len(grids), 3.3),
                             squeeze=False, layout='constrained')
    for ax, name, (x, z) in zip(axes[0], preferred, grids):
        ax.contour(x, x, z, levels=levels, cmap='viridis')
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xticks=[], yticks=[],
               title=f'{labels.get(name, name)}\nMW2 objective: {models[name][1]:.6f}')
    for suffix in ('png', 'pdf'):
        fig.savefig(out/f'comparison.{suffix}', dpi=170)
    plt.close(fig)


def fit_initialized(inputs, cfg, optimization, weights=None):
    tick = time.perf_counter()
    candidate, count = initialize(inputs, cfg, weights)
    initial = pad(candidate.gmm, sum(g.count for g in inputs)-len(inputs)+1, 0)
    init_ms = (time.perf_counter()-tick)*1000
    result, history, stats = solve_mixed(inputs, initial, optimization, weights)
    return initial, result, history, dict(candidates=count, init_ms=init_ms,
                                         total_ms=(time.perf_counter()-tick)*1000, **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--cases', nargs='+', choices=['images', 'synthetic'])
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--steps', type=int, help='Optional separate optimizer-budget ablation')
    parser.add_argument('--skip-mw2', dest='benchmark_mw2', action='store_false', default=None)
    parser.add_argument('--no-plots', dest='plots', action='store_false', default=None)
    args = parser.parse_args()
    defaults = json.loads((PACKAGE/'configs/mixed_fast.json').read_text())
    config = dict(cases=['images', 'synthetic'], output_dir='results/simple_initialization',
                  names=None, repeats=1, steps=None, benchmark_mw2=False, plots=True,
                  mw2=defaults['mw2'])
    if args.config:
        loaded = json.loads(args.config.read_text())
        if loaded.keys()-config.keys():
            parser.error('Unknown config keys')
        config.update(loaded)
    for key in config:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
        setattr(args, key, config[key])
    if not args.cases or any(c not in ('images', 'synthetic') for c in args.cases) or args.repeats < 1:
        parser.error('Invalid cases or repeats')
    if any(type(config[key]) is not bool for key in ('benchmark_mw2', 'plots')):
        parser.error('benchmark_mw2 and plots must be booleans')
    if (not isinstance(args.mw2, dict) or set(args.mw2) - set(defaults['mw2'])):
        parser.error('mw2 accepts gaussian_iterations and gaussian_tolerance')
    args.mw2 = config['mw2'] = defaults['mw2'] | args.mw2
    if (type(args.mw2['gaussian_iterations']) is not int or args.mw2['gaussian_iterations'] < 1
            or not np.isfinite(args.mw2['gaussian_tolerance']) or args.mw2['gaussian_tolerance'] < 0):
        parser.error('Invalid MW2 iterations or tolerance')
    args.output_dir = Path(args.output_dir)
    torch.set_num_threads(1)
    settings = defaults['settings']
    if args.steps is not None:
        if args.steps < 1:
            parser.error('steps must be positive')
        for case in args.cases:
            settings[case]['optimization']['steps'] = args.steps
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'experiment.json').write_text(json.dumps(dict(
        **config, settings=settings, K=30, data_seed=0, threads=1, device='cpu',
        component_budget=117, checkpoint_selection='fixed min-LSOT-B validation bank'), indent=2))
    rows = []
    eval_args = SimpleNamespace(test_kinds=['B'], test_L=512, eval_samples=1024)
    if args.benchmark_mw2:
        tiny = GMM.from_numpy([1.], [[.5, .5]], [[[.01, 0.], [0., .01]]])
        fit_mw2([tiny]*4, dict(gaussian_iterations=2, gaussian_tolerance=0.))
    for case_index, case in enumerate(args.cases):
        source = PACKAGE/'assets/mixed_fast'/case
        inputs = [load_gmm(source/'inputs'/f'input_{j}.npz') for j in range(4)]
        case_out = args.output_dir/case
        case_out.mkdir(parents=True, exist_ok=True)
        if not args.benchmark_mw2:
            ref = load_gmm(source/'MW2/barycenter.npz')
            ref_metrics = evaluate(ref, inputs, WEIGHTS, eval_args, 0, ref)
        # Warm the optimizer on a real input, excluding this run from timing.
        warm = dict(settings[case]['optimization'], steps=1, L=4, sampling='fixed')
        solve_mixed(inputs, inputs[0], warm)
        configs = list(configurations(settings[case]['initialization']))
        if args.names:
            configs = [c for c in configs if c['name'] in args.names]
            if len(configs) != len(set(args.names)):
                raise ValueError('Unknown initializer name')
        methods = [None, *configs] if args.benchmark_mw2 else configs
        models = {}
        for repeat in range(args.repeats):
            shift = (repeat + case_index) % len(methods)
            fitted = {}
            for cfg in methods[shift:]+methods[:shift]:
                name = 'MW2' if cfg is None else cfg['name']
                print(f'{case}: repeat {repeat+1}/{args.repeats}, {name}', flush=True)
                if cfg is None:
                    solved, timings = fit_mw2(inputs, args.mw2)
                else:
                    fitted[name] = fit_initialized(inputs, cfg, settings[case]['optimization'])
            if args.benchmark_mw2:
                ref = solved.gmm
                ref_metrics = evaluate(ref, inputs, WEIGHTS, eval_args, 0, ref)
                timings['mw2_finalize_ms'] = timings['total_ms']-sum(
                    timings[k] for k in ('tuple_setup_ms', 'gaussian_cost_ms', 'constraint_setup_ms', 'lp_ms'))
                row = dict(case=case, name='MW2', repeat=repeat, status='ok',
                           final_mw2=ref_metrics['mw2_objective'],
                           output_components=ref.count, num_tuples=int(np.prod([g.count for g in inputs])),
                           covariance_iterations=solved.covariance_iterations,
                           covariance_tolerance_met=solved.covariance_tolerance_met,
                           marginal_error=solved.marginal_error, **timings, **ref_metrics)
                rows.append(row)
                print(json.dumps(row), flush=True)
            ref_obj = ref_metrics['mw2_objective']
            save_gmm(case_out/'mw2.npz', ref)
            models['MW2'] = (ref, ref_obj)
            for cfg in configs:
                initial, result, history, stats = fitted[cfg['name']]
                init_metrics = evaluate(initial, inputs, WEIGHTS, eval_args, 0, ref)
                metrics = evaluate(result, inputs, WEIGHTS, eval_args, 0, ref)
                initial_obj, final_obj = init_metrics['mw2_objective'], metrics['mw2_objective']
                row = dict(case=case, name=cfg['name'], repeat=repeat, output_components=result.count,
                           initial_mw2=initial_obj, final_mw2=final_obj, reference_mw2=ref_obj,
                           initial_test_min_B=init_metrics['test_min_B'],
                           initial_validation=history[0]['validation'],
                           improvement_percent=100*(1-final_obj/initial_obj), **stats, **metrics)
                rows.append(row)
                out = case_out/cfg['name']; out.mkdir(parents=True, exist_ok=True)
                save_gmm(out/'initial.npz', initial); save_gmm(out/'optimized.npz', result)
                trace = pd.DataFrame(history); trace.to_csv(out/'history.csv', index=False)
                (out/'settings.json').write_text(json.dumps(dict(initialization=cfg,
                        optimization=settings[case]['optimization']), indent=2))
                if args.plots and repeat == args.repeats-1:
                    save_plots(out, case, [('MW2', ref, ref_obj), ('Initialization', initial, initial_obj),
                        (cfg['name'], result, final_obj)], trace, stats['selected_step'])
                print(json.dumps(row), flush=True)
                models[cfg['name']] = (result, final_obj)
            summary = write_summary(args.output_dir, rows)
        if args.plots:
            save_case_plots(case_out, summary[summary['case'] == case], models)


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
