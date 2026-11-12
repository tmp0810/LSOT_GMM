# MW2 and variational LSOT barycenters

This experiment follows Julie Delon's
[GMM_OT_Barycenters.ipynb](https://github.com/judelo/gmmot/blob/0984edc826b113e35c3260b699a4ff49ab39d25f/python/GMM_OT_Barycenters.ipynb).
It retains MW2 and adds **eight** methods: avg/min LSOT with Mix, SMix, B,
and B1D. 

## Fast solver and fair runtime benchmarks

See [FAST_EXPERIMENT.md](FAST_EXPERIMENT.md) for the new single-start solver,
fixed/fresh/partial projection banks, fixed validation selection, the additional
MW2-variational baseline, component-count sweeps, and timing definitions.
The original MW2 multi-marginal reference and legacy optimizer remain available.
The code retains `LSOT` method names for compatibility; the paper calls them LSMT.

```bash
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/fast.json \
  --cases synthetic --device cuda --output-dir results/barycenter_fast

# Same full grid with newly sampled directions at every Adam update:
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/fast.json \
  --cases synthetic --sampling fresh --device cuda \
  --output-dir results/barycenter_fast_fresh
```

`fast.json` is an experiment starting point, not an empirically optimal setting:
one screened initialization, 200 Adam updates, learning rate .003, training
L=100, fixed validation L=512. Early stopping is **off** in this preset so that
short plateaus are not automatically treated as convergence. Set
`--early-stopping-patience 5` to test stopping after five regular validation
checks without a relative improvement of 1e-4. All eight LSOT variants work
with the fast engine; specify `--methods` to include averages.

Each method still saves its grid at `<output>/<case>/<method>/barycenters.png`
and PDF, individual panels, GMMs, histories and checkpoint GMMs.

## Optimizer checks and revised solver

### Adam-only experiment

Pure Adam is now available without constructing L-BFGS or running a second
optimizer. It uses a fixed update budget, Adam learning rate 0.01 by default,
and returns the best exact hard-objective candidate encountered. Means,
covariances, weights, projection formulas and banks are unchanged.

```bash
# Quick equal-weight synthetic check: MW2 + all eight LSOT methods.
python -m experiments.barycenter.run --config experiments/barycenter/configs/adam.json \
  --cases synthetic --center-only --output-dir results/barycenter_adam_check

# Full 7x7 grids for the original Gaussian, synthetic and image experiments.
python -m experiments.barycenter.run --config experiments/barycenter/configs/adam.json \
  --output-dir results/barycenter_adam

# Paired optimizer comparison, keeping inputs/start fixed across bank seeds.
python -m experiments.barycenter.compare_optimizers --cases synthetic \
  --seeds 0 1 2 --output-dir results/barycenter_optimizer_comparison
```

`adam.json` selects one start, 500 Adam steps, L=500 and no warmup. Override
`--steps` and `--L` explicitly for smaller budgets, or `--starts 4` for shared spatial
starts. The explicit CLI equivalent is `--optimizer adam --steps 500 --L 500
--starts 1 --adam-learning-rate 0.01`. With Adam, `--steps` counts Adam updates;
with L-BFGS, it counts outer L-BFGS iterations. Adam rejects a nonzero
`--warmup-steps` to avoid accidentally running two Adam stages.

The comparison script checks Adam200, Adam280 and Adam200+L-BFGS80. It saves
`comparison.csv`, `paired.csv`, `summary.csv`, center comparison images, and
loss/gradient histories. Each paired solve uses identical inputs, start and
bank. Input EM and initialization seeds are fixed at 0; only projection bank
seeds vary. Objective/gradient evaluation counts are separate. Solver times
include all optimizer stages and exclude shared setup, reference evaluation,
plots and one-time optimizer import overhead. Equal outer-update budgets are
not equal compute budgets; use measured time/evaluation counts.

`max_steps` is expected for a fixed-budget Adam run. It does not establish
convergence. Neither optimizer uses soft sorting or changes the variational
objective. Use a new output folder when switching optimizer/configuration.

The sampled variational objectives are unchanged:

```
avg: sum_j omega_j * mean_l C(mu_j, nu; s_l)
min: sum_j omega_j * min_l  C(mu_j, nu; s_l)
```

`C` uses the lifted plan and the full Gaussian W2 component cost. Each
input pair selects its own slice in the min objective. Banks remain fixed;
means, Cholesky covariance parameters, and softmax mixture weights are
optimized. Hard sorting and mass boundaries can obstruct smooth L-BFGS
line searches. A flat loss trace is not a convergence certificate.

The revised default remains **pure L-BFGS**, with four method-independent
common-quantile initializations (historical direction, x axis, y axis, and
the alternating-sign direction). All eight methods receive the same starts,
saved under `initializations/`. No MW2 solution is used as a warm start.
The single-Gaussian example uses one start. `--steps 80` is the budget
**per start**, so the default can use up to 320 L-BFGS outer iterations.
All starts and recovery work are included in the reported solver time.

The solver reevaluates gradients at retained iterates, clears curvature
history when accepting another line-search candidate, and tries block
gradient backtracking before abandoning a stalled start. PyTorch's internal
change threshold no longer blocks small-gradient refinement; stopping is
handled by the outer loop. The exact loss is divided by the squared spatial
coordinate scale for numerical conditioning; reported costs stay in original
units. `gradient_inf` is the gradient of this scaled loss with respect to the
optimizer parameters, evaluated at the returned candidate.

`gradient_tolerance_met` only reports that branch-gradient test. `stalled`
means recovery found no sufficient progress; `max_steps` means the winning
start exhausted its budget. Neither establishes global optimality, and at
hard ordering boundaries a classical gradient test is not a general
nonsmooth stationarity certificate.

### Quick center check

```bash
python -m experiments.barycenter.run --cases gaussian synthetic \
  --center-only --output-dir results/barycenter_optimizer_check
```

This checks the Gaussian control and the equal-weight center of the default
7x7 synthetic grid, without running all 49 nodes. Inspect `metrics.csv`,
`center_comparison.png`, and `center_convergence.png`. The convergence figure
shows best loss, current loss, and gradients at the best/current candidates.
Individual history CSVs include start, phase, evaluation count, and curvature
restarts; per-start statuses are saved in each JSON record.

### Full experiment

```bash
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/mw2_reference.json \
  --output-dir results/barycenter_v2
```

Use a new output directory when moving from the old solver: schema/version
checks prevent reusing old optimization results. Completed v2 runs can resume
normally. Each method still produces its 7x7 `barycenters.png` and PDF.

For an explicitly different optimization strategy, optional Adam warmup can
cross hard ordering boundaries before L-BFGS refinement:

```bash
python -m experiments.barycenter.run --cases synthetic --center-only \
  --starts 1 --warmup-steps 200 --output-dir results/barycenter_warmup_check
```

This is **Adam + L-BFGS**, not pure L-BFGS; the exact hard objective and banks
stay unchanged. Warmup is off by default and its time/iterations are counted.
`--freeze-weights` keeps the weights of each shared start fixed.

Image runs use K=12 input GMMs and foreground masks recovered from the pinned
notebook display: redcross/duck support matches POT's 128x128 data; original
star/batman PNG identity cannot be verified. `--image-dir` accepts the four
original PNGs. Preprocessing uses every nonzero `1-blue` pixel after a vertical
flip, with no resizing or subsampling, and seeded full-covariance EM.

### Staged historical starts (4 -> 2 -> 1)

```bash
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/halving.json \
  --cases synthetic --device cuda \
  --output-dir results/barycenter_halving
```

This preset keeps MW2 as the main baseline and runs the four min-LSOT families.
It builds the four original common-quantile initializations (no PCA screening).
All four run 25 Adam updates; the best two run 25 more; the best one continues
from update 50 to update 500. Ranking uses the best validated iterate on the
same fixed independent bank (L=512). The live model and Adam moments are
preserved, not reset to a checkpoint when promoted. The final returned GMM is
that survivor's best validated iterate. Fresh L=100 training directions use the
same per-update seeds across starts. `--sampling fixed` and `partial` also work;
partial retained directions depend on each start's winners.

Defaults: learning rate 0.003, `--probe-steps 25 50`, `--steps 500`, no early
stopping. This costs 600 total updates rather than 2000 for four complete starts,
excluding extra validation/initialization work. Actual wall time includes all
starts, screening and validation; initialization is included in total_ms.
No quality or speedup guarantee is assumed. Single-Gaussian cases use one start;
exact grid corners reuse the inputs. Use a new output directory for each setup.

Each per-node JSON records `halving_decisions`, `selected_start`, `total_updates`
and all `start_summaries`; history includes all starts with their identifiers.
Checkpoint files describe the surviving trajectory. `iterations` is the final
survivor's update count; `total_updates` includes discarded trajectories.
For a quick central-node check append `--center-only --methods MW2 min-LSOT-B1D`.

### Image-GMM component scaling

The complete K=5..100 protocol, isolated MW2 resource limits, normalized image
preprocessing, halving optimizer and pairwise-plan benchmark are documented in
[SCALING.md](SCALING.md). Run:

```bash
python -m experiments.barycenter.scaling \
  --config experiments/barycenter/configs/scaling.json \
  --device cuda --output-dir results/barycenter_scaling
```
