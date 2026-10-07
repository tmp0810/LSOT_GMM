# Revised barycenter solver validation

CPU, float64, PyTorch 2.14.1+cpu, NumPy 2.3.5, POT 0.9.7.post1.

## Tests

```bash
python -m pytest -q experiments/barycenter/test_barycenter.py \
  tests/test_transport.py tests/test_distribution_projections.py \
  tests/test_linalg_batching.py
```

**69 passed, 9 skipped** (CUDA unavailable).

The MW2 reference matches the upstream notebook's N=10 tuple costs, Gaussian
means/covariances, and multi-marginal LP. All eight hard sampled losses match
the existing LSOT plans; finite differences match weight/mean/covariance
gradients away from rank and cumulative-mass boundaries. Regression tests
cover a rejected stationary line-search trial, curvature reset after accepting
a better trial, the returned-point gradient, and shared reproducible starts.

## Bugs corrected

- The stopping gradient could come from a rejected line-search trial. It is
  now recomputed at the retained and finally returned candidate.
- Restoring another trial without resetting L-BFGS would reuse curvature
  from a different iterate history. Restoration now clears that history.
- PyTorch uses tolerance_change for the directional derivative as well as
  function changes. With a 1e-10 threshold this blocked reaching a 1e-7
  gradient on the smooth Gaussian control. The internal threshold is zero;
  the outer patience/recovery monitor handles small changes.
- The old best-loss-only plot could not diagnose stationarity. The new figure
  shows best/current loss and their branch gradients; the returned best point
  is explicitly recorded.

## Center checks at full L=100

All three cases and all nine methods ran at the center: **27 saved results**.
The synthetic/image solver uses four common spatial starts, 80 L-BFGS steps
per start, patience 8, at most two curvature recoveries per start, and no
Adam warmup. Gaussian inputs use one start.

For the smooth single-Gaussian control, **all eight methods reach the scaled
parameter-gradient infinity norm <= 1e-7** (5.25e-8 to 9.72e-8), and agree
with a 100-iteration Gaussian covariance reference to 1e-8 in objective.

Synthetic center objective values (compare old/new only within a row):

| Method | Old single start | Revised four starts | Final status |
| --- | ---: | ---: | --- |
| min-LSOT-Mix | 0.03455371 | 0.03106456 | stalled |
| min-LSOT-SMix | 0.03615215 | 0.03028743 | max_steps |
| min-LSOT-B | 0.03010434 | 0.03009431 | stalled |
| min-LSOT-B1D | 0.03392471 | 0.03030588 | stalled |
| avg-LSOT-Mix | 0.08365242 | 0.08289859 | stalled |
| avg-LSOT-SMix | 0.08442245 | 0.07759855 | stalled |
| avg-LSOT-B | 0.06632906 | 0.06469284 | stalled |
| avg-LSOT-B1D | 0.06967546 | 0.06816204 | stalled |

All eight synthetic costs decrease versus the old single-start run. This
comparison uses more initialization/solver work, not equal runtime. **None
of the eight synthetic methods meets the 1e-7 branch-gradient threshold**;
final gradients remain approximately 1.7e-3 to 1.0e-2. Seven report stalled
and one max_steps. This is improvement, not a claim of certified convergence.
At finite hard sorting boundaries, branch gradients are not a general
nonsmooth stationarity test. The full lifted cost can jump when ranks swap.

For image GMMs, six methods improve versus the old run, while min-Mix and
min-B have higher costs. None meets the gradient threshold. Nonconvex local
solves and changed conditioning can select different basins; the fix does
not claim uniform improvement for every case or equality with MW2. All
returned candidates are finite, SPD, and no worse than their first starts.
Exact costs, gradients, statuses, and costs of the MW2 candidate evaluated
under each LSOT objective are committed in `validation/center_results.json`.

## Pipeline and limits

The final smoke config completes **171 method/node results**, creates all
individual plots, grids, PDFs and GIFs, and reuses completed solves on rerun.
The full 7x7, 891-result experiment was not run. CUDA was not tested.

The default remains pure L-BFGS. Optional --warmup-steps 200 is explicitly
Adam followed by L-BFGS and keeps the exact hard objective; it is not a
soft-sorting surrogate or a pure L-BFGS comparison.

Original shape PNGs are absent from the upstream MW2 repository. Tests use
the documented recovered masks, with exact POT foreground checks for
redcross/duck; original star/batman pixel identity remains unverified.

## Adam-only paired comparison

Pure Adam now has an explicit optimizer flag and fixed step budget; no
L-BFGS object is constructed. Regression tests verify this, best-iterate
restoration, frozen weights, exact parity with the same warmup prefix, and
rejection of an ambiguous extra warmup. The targeted suite now reports
**72 passed, 9 skipped** (CPU only). Objective and gradient evaluations are
logged separately.

At L=100 and one identical common start, Adam200 and Adam280 were compared
with Adam200 + L-BFGS80. Inputs/initialization stay fixed at seed 0 while
projection bank seeds vary. Synthetic checks cover all eight methods and
three bank seeds (24 paired settings); image checks cover all eight methods
at bank seed 0 (8 settings). All optimizer stages are included in timings;
setup/reference evaluations/plots and one-time optimizer imports are not.
There is one timing measurement per setting, so ratios are diagnostic CPU
measurements. Equal outer-iteration counts are not equal computation.

| Data | Adam updates | Mean loss gap to hybrid | Maximum gap | Median hybrid/Adam time | Adam lower/equal |
| --- | ---: | ---: | ---: | ---: | ---: |
| synthetic | Adam200 | 0.111% | 0.619% | 2.69 | 0/24 |
| synthetic | Adam280 | -0.127% | 0.619% | 1.90 | 10/24 |
| images | Adam200 | 1.523% | 4.265% | 2.98 | 0/8 |
| images | Adam280 | 1.386% | 4.265% | 2.11 | 2/8 |

A positive gap means Adam has a higher own-objective value. Adam280 improves
the synthetic mean but is not uniformly better. Image gaps are larger;
min-B retains the largest hybrid advantage (~4.27%). Adam200 is a simpler
and faster fixed-budget choice, with a measured loss tradeoff. These are
local center checks, not global convergence or full-grid claims.

Exact paired costs/timings and settings are in
`validation/optimizer_comparison.json`. The reproduction command is:

```bash
python -m experiments.barycenter.compare_optimizers --cases synthetic \
  --seeds 0 1 2 --output-dir results/adam_comparison
```

The pure-Adam config also completes a 27-result all-case center run and a
171-result 3x3 smoke run, including grids/PDFs/GIFs and resumed checkpoints.
The full 7x7, 891-result pure-Adam run was not performed.
