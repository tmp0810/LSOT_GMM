# Simple initialization ablation: K=30, data seed 0

This is a development experiment on the existing bundled images and synthetic
inputs, with four equal barycentric weights and 117 output components. The
published `mixed_fast` defaults and solver have not been changed.

## Run

```bash
python -m experiments.barycenter.simple_initialization \
  --config experiments/barycenter/configs/simple_initialization.json
```

The config runs the full multi-marginal MW2 baseline, the old initializer
(`mixed_fast`), and `greedy_mean_spread0.25`, with three sequential repetitions,
CPU float64 and one thread. The methods run in rotating order after an untimed
warmup. Running without a config explores all 24 initializer configurations per
case using the bundled reference. Use `--names` to select a subset,
`--repeats 1` for a quick run, `--skip-mw2` to use the bundled reference without
timing a new solve, or `--no-plots` to write numerical results only.

Under `results/simple_initialization_confirm/`:

- `metrics.csv`: one row per method, case and repeat. MW2 phase times cover
  tuple enumeration, Gaussian costs, constraints, the LP solve, and remaining
  time (including construction of the output GMM). The min-LSOT-B rows report
  initialization, optimization including checkpoint selection, and total time.
- `summary.csv`: means, sample standard deviations, and speedup relative to MW2
  when it was timed in the same run. `experiment.json` records resolved settings.
- `<case>/comparison.png` and `.pdf`: final barycenters side by side, using shared
  contour levels. `runtime.png` and `runtime_breakdown.png` show totals and phases.
- `<case>/<initializer>/comparison.png` and `.pdf`: MW2, initial GMM, and optimized
  GMM. `convergence.png`, `history.csv`, and the GMM snapshots record the solve.

Solver timing excludes input fitting/loading, warmup, evaluation (including
diagnostic MW2 LPs), plotting, and file writing. MW2 results are used only
after both min-LSOT-B fits, for evaluation and plotting.

## Full 7×7 grids and component sweep

```bash
python -m experiments.barycenter.simple_initialization_grid \
  --config experiments/barycenter/configs/simple_initialization_grid.json
```

The grid config runs **synthetic and images**, at **K=5, 10, 30** components
per input GMM and data seed 0. K=30 reuses the bundled input fits; K=5 and 10
use the scaling experiment's synthetic refinement and image EM fitting. All
methods share exactly the same four inputs at each K. No download is needed.

Columns vary x, rows vary y; the node weights are
`[(1-x)(1-y), x(1-y), (1-x)y, xy]`. Corner nodes copy the corresponding
input, edges fit two inputs, and interior nodes fit four. The weighted
initializer normalizes weights within each group for its Gaussian
representative, and uses the original node weights at the final merge.
Greedy min-LSOT-B and MW2 use the same node weights. The variational
component budget is the sum of the active input counts minus the number
of active inputs plus one. The main config uses partial direction refresh
for both cases (25% replaced every 10 steps, retaining the winning directions).
Images uses 64 training directions and 100 steps; synthetic uses 128 training
directions and 200 steps. Both use the same covariance update as the center
experiment and choose a checkpoint on a fixed 512-direction validation bank.
The per-case `optimization` entries override the inherited training bank
size and sampling mode. These settings were selected from the K=30 center
sweep, rather than an exhaustive grid search.

The default is one repeat (588 method/node rows). The full run can take
tens of minutes on one CPU thread: there are 50 four-input K=30 MW2 LPs
across the two cases. Rerun the same command to resume from saved nodes.
Use a separate output directory when changing settings. A quick check:

```bash
python -m experiments.barycenter.simple_initialization_grid \
  --cases synthetic --component-counts 5 --steps 1 \
  --output-dir results/simple_initialization_grid_smoke
```

The output root is `results/simple_initialization_grid_greedy_partial/`:

- `metrics.csv`: one row per node and method, including bilinear weights,
  measured solve phases, MW2 objective, and independent min-LSOT-B test loss.
- `summary.csv`: average per fitted node, summed solver time for the entire
  grid, and speedup relative to MW2. Copied corners do not enter the averages.
  Standard deviation of total grid time requires multiple repeats.
- `<case>/K<K>/repeat_00/<method>/barycenters.png` and `.pdf`: full 7×7 grid;
  the folder also saves the individual GMMs, initializations, and loss history.
- `<case>/K<K>/comparison.png` and `.pdf`: center cell comparison at shared
  contour levels. `runtime.png` and `runtime_breakdown.png` report whole-grid
  timings. `<case>/scaling_runtime.png` compares K values.

Timing excludes data preparation, warmup, evaluation, and rendering. With
multiple repeats, the displayed grids are from the last repeat.

## New initializer

1. Consider the three pairings of four input GMMs.
2. Within each pair, sort all pairs of component means by squared Euclidean
   distance. Visit them in that order and allocate the minimum remaining source
   and target mass. This is a greedy feasible coupling, not an optimal LP.
3. Represent each group by its average mean and a covariance consisting of the
   average input covariance plus rho times the covariance of the grouped means.
   Use these intermediate GMMs to greedily join the two pairs. Retain the original
   component indices and recompute the representative from all four components.
4. Score the three resulting feasible tuple plans by their Gaussian W2 tuple
   costs and keep one. Optimize only that one GMM with the unchanged mixed solver.

For a group of r equally weighted input Gaussians:

    m = sum_j m_j / r
    Sigma = sum_j Sigma_j / r
            + rho * sum_j (m_j-m)(m_j-m)^T / r.

rho=0 is a covariance average; rho=1 matches the first two moments of the
equally weighted mixture. rho=0.25 is a tuned shrinkage heuristic, not an exact
Gaussian Wasserstein barycenter. It is used identically for both cases. No
iterative Gaussian barycenter calculation, direction sampling or LP is used in
this initializer. Gaussian W2 costs still score its three tuple candidates.

Each greedy join has at most n+m-1 positive entries (ignoring negligible numeric
mass). Thus four K-component inputs produce at most 4K-3 components; the existing
padding rule handles fewer components. Pair cost sorting uses quadratic storage
and O(K^2 log K) work for fixed four inputs. This is not a linear-memory sliced
initializer and has not been benchmarked at larger K.

## Measurements

In the development ablation below, we ran 24 initializer configurations on each case (48 optimization trials),
including common quantiles, PCA, mean projection trees, greedy mean/Gaussian
matching, covariance averages/moments/shrinkage, and narrower lifted B trees.
Then the selected setting and baseline were rerun sequentially three times.
Quality reproduced exactly; repetitions measure runtime variation, not variation
across data seeds. Optimizer settings are unchanged: 100 steps for images, 200
for synthetic, Adam for means/weights and affine-invariant covariance updates.

| Case | Initializer | Initial MW2 objective | Final MW2 objective | Init (ms), mean +/- std | Total (ms), mean +/- std |
|---|---|---:|---:|---:|---:|
| Images | mixed_fast | 0.013402057 | 0.013067631 | 1061.86 +/- 37.81 | 2530.84 +/- 90.28 |
| Images | Greedy mean, rho=.25 | 0.012985528 | 0.012330998 | 18.34 +/- 0.43 | 1588.63 +/- 43.26 |
| Synthetic | mixed_fast | 0.018581245 | 0.017763354 | 215.88 +/- 4.02 | 5833.11 +/- 158.04 |
| Synthetic | Greedy mean, rho=.25 | 0.018003589 | 0.016807616 | 18.81 +/- 2.96 | 5872.29 +/- 111.14 |

The new optimizer outputs reduce MW2 by 5.04% and 6.64% relative to their own
initializers. Compared with the old optimized outputs they reduce MW2 by about
5.64% and 5.38%. Means remain nearly fixed (learning rate 1e-6).

The synthetic winner is step 40. A separate three-repeat 50-step experiment
reproduced the same final objective, at total times 1550.17, 1516.52 and 1410.42
ms (mean 1492.37 ms). This is a budget change, separate from the controlled
initializer comparison above. Reproduce using:

```bash
python -m experiments.barycenter.simple_initialization \
  --config experiments/barycenter/configs/simple_initialization.json \
  --cases synthetic --names greedy_mean_spread0.25 --steps 50 --skip-mw2 \
  --output-dir results/simple_initialization_synthetic50
```

## Interpretation and limitations

Checkpoint selection uses fixed-bank min-LSOT-B validation, never MW2. MW2 was
used to choose development configurations across experiments, as requested.
These are not held-out quality estimates or convergence guarantees.

The new solution has a worse independent min-LSOT-B test objective than the old
solution: images 0.051887 versus 0.032406, synthetic 0.041516 versus 0.037901.
Thus improved MW2 is not evidence of a better optimum for the min-LSOT objective.
Final SW2/density L1 to the reference improve versus old final outputs, but both
can worsen versus the new initializer. No claim of improvement on every metric
is justified. Exact MW2 reference objectives remain lower (0.00834904 images,
0.01419297 synthetic).

The independent quality audit, all search results and repeated timings are in
`validation/simple_initialization_k30/`. Numeric checks verified unequal-weight
tuple marginals, the component bound, the covariance formula, positive
definiteness and agreement of tuple scores with Gaussian W2 costs.
