# MW2 and variational LSOT barycenters

This experiment follows Julie Delon's
[GMM_OT_Barycenters.ipynb](https://github.com/judelo/gmmot/blob/0984edc826b113e35c3260b699a4ff49ab39d25f/python/GMM_OT_Barycenters.ipynb).
It retains MW2 and adds **eight** methods: avg/min LSOT with Mix, SMix, B,
and B1D. 

## Optimizer checks and revised solver

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
