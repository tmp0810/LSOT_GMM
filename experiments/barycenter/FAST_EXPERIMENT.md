# Barycenter optimization and runtime experiments

These additions retain the original notebook data and MW2 multi-marginal LP.
They do not claim that min-LSMT must be faster, converge globally, or return
the MW2 barycenter. Compare time and quality together. Historical `LSOT`
filenames/method identifiers are retained for compatibility with earlier runs.

## Colab

```python
%cd /content
!test -d LSOT_GMM || git clone https://github.com/tmp0810/LSOT_GMM.git
%cd /content/LSOT_GMM
!git pull --ff-only origin main
!pip -q install -e .
```

Quick center check (all four min projections and both MW2 baselines):

```python
!python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/fast.json \
  --cases synthetic --center-only --device cuda \
  --output-dir results/barycenter_fast_center
```

Full original 7x7 synthetic and image experiments:

```python
!python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/fast.json \
  --cases synthetic images --device cuda \
  --output-dir results/barycenter_fast_full
```

Use `--cases synthetic` for synthetic only. To include averages add
`--methods MW2 MW2-variational min-LSOT-Mix min-LSOT-SMix min-LSOT-B min-LSOT-B1D avg-LSOT-Mix avg-LSOT-SMix avg-LSOT-B avg-LSOT-B1D`.
Directions, starts, steps and learning rate are separate controls.

## Mathematical and numerical protocol

The candidate is a GMM with learned means, Cholesky covariance factors and
softmax weights (unless `--freeze-weights` is explicitly used). The default
budget is `sum(K_j)-J+1`; corners return the original GMM directly. The grid
weights and all input GMMs are unchanged.

* `MW2`: original multi-marginal LP. The full-grid default retains the
  notebook's ten covariance iterations. `--gaussian-iterations 100
  --gaussian-tolerance 1e-10` requests a more accurate reference, stopping
  within that maximum iteration budget. Benchmark scripts use this setting.
* `MW2-variational`: additional local Adam baseline, with the same candidate
  parameterization and initializer as LSMT. Its pairwise distances use POT's
  exact discrete LP. Cost gradients use the optimal plan; gradients of learned
  masses use the OT dual potentials. It does **not** replace the original MW2.
* `min-LSOT-*`: `sum_j lambda_j min_l C(mu_j, nu; s_l)`, selecting a slice
  separately for every input pair. Average methods replace min by mean.
  Every lifted cost uses the original Gaussian W2 ground cost.

The fast implementation uses the exact trace/determinant expression for 2-D
Gaussian costs and B projections. Small component products use a dense cost
matrix (cheaper than repeated sparse bookkeeping at that size). Large problems
deduplicate scored edges, and min backpropagates only through winning slices.
Exact tied minima share gradients; projection fibers retain proportional mass
splitting. Higher-dimensional Gaussian costs use the existing spectral path.
The optimizer avoids revalidating all SPD eigenvalues at every forward; inputs,
validation checkpoints and the final returned GMM are fully checked.

Hard orders are recomputed, not differentiated. Derivatives are local branch
derivatives, including interval-mass dependence on trainable weights. No soft
sorting, entropic regularization or optimizer change is hidden in the fast path.

## One screened initialization

`--initialization pca-screen --starts 1` uses weighted PCA of the component
means, two diagonal combinations of the PCA axes, and the historical direction.
In 2D this screens five inexpensive common-quantile candidates and runs Adam
**once**. The single-Gaussian case needs only one candidate. The score is the
weighted Gaussian tuple cost **before** budget merges/splits; it is a shared
geometric proxy, not a method-specific LSMT objective. The selected initializer,
directions and scores are saved. No MW2 LP solution enters initialization.

`--initialization historical --starts 1` provides the original single start.
The older four-start protocol remains available with the legacy engine.
Screening can improve or worsen the final local solution; the ablation measures
this rather than assuming that the lowest initial proxy guarantees the best solve.

## Projection policies and selection

* `--sampling fixed`: sample once per solve; reuse the bank.
* `--sampling fresh`: independently sample L slices before every update.
* `--sampling partial`: every `--refresh-every 10` updates retain current
  per-pair winning directions when capacity permits, fill retained slots in
  deterministic bank order, and replace a fraction (`--refresh-fraction .25`).
  For averages there are no winning directions; the first retained slots stay.
  This is an adaptive search heuristic, not an unbiased Monte Carlo estimator.

Fresh averages give an unbiased loss estimate for a fixed candidate. A finite
random minimum is generally biased above the full slice infimum. No unbiased
population-gradient claim is made for hard lifted sorting.

All fast policies select the returned iterate on **one fixed validation bank**,
independent of training (seed + 1000003). `--validation-L 512` and
`--validate-every 10` control its size and frequency. Checkpoints at 25, 50, 100
and 200 updates are also evaluated. Selection never compares training losses
from different random banks. Final benchmark test banks use seed + 2000003
and `--test-L 2048`; test scores do not select a checkpoint or stop training.

`--early-stopping-patience 0` disables early stopping. Positive values count
regular validation checks without significant improvement. Extra trajectory
checkpoints do not consume patience. `validation_plateau` and `max_steps` are
budget/stopping descriptions, not convergence certificates. The gradient of the
returned candidate is recomputed on the validation objective. Earlier gradient
diagnostics are explicitly labeled as training gradients before the update.

## Controlled solver ablation

```bash
python -m experiments.barycenter.benchmark --task ablation \
  --cases synthetic images --kinds Mix SMix B B1D --seeds 0 1 2 \
  --steps 200 --L 100 --patience 0 --device cuda \
  --output-dir results/barycenter_ablation
```

Variants: `reference-fixed` (old objective kernel in the same single-start
Adam/validation loop), `fast-historical` (fast kernel, same historical start),
`fast-fixed` (screened start), `fast-fresh`, `fast-partial`. MW2 multi-marginal
and MW2-variational are also included. `reference-fixed` is a kernel ablation,
not a timing of the entire legacy four-start program. Select variants with
`--variants fast-fixed fast-fresh fast-partial` and add `--aggregations min avg`
when averages are desired. Case data are fixed; each seed is an independent
run, not an extra start from which the best result is selected.

Outputs:

* `solvers.csv`: measured total/init/solve/validation times, common MW2
  objective, reference gap, SW2, independent test objective, status and steps.
* `trajectory.csv`: common MW2 objective versus cumulative time at saved
  checkpoints. A common objective is needed because own losses differ.
* `<case>_time_quality.png`: all seed trajectories, plus the MW2 reference.
* `summary.csv`: mean/std across runs; `time_to_quality.csv`: first **observed**
  checkpoint within 1%, 5%, 10% of the reference MW2 objective (empty if missed).
  These are approximation-to-reference criteria, not LSMT optimality gaps.
* Per-run GMM NPZ files, initialization metadata and validation histories.

To examine whether 200 steps are enough, repeat with `--steps 500 --patience 0
--checkpoints 25 50 100 200 500` in a new output folder. Do not label the first
run as converged merely because its loss or image looks stable.

## Component-count sweeps

Pairwise plan/transport computation, analogous to the color-transfer timing:

```bash
python -m experiments.barycenter.benchmark --task plans \
  --component-counts 5 10 20 50 100 200 500 --kinds Mix SMix B B1D \
  --seeds 0 1 2 --L 100 --repeats 10 --device cuda \
  --output-dir results/lifted_plan_scaling
```

`plans.csv` and `plans_*.png` distinguish total pairwise computation from a
controlled cached-ground-cost measurement. Total includes bank generation,
projection, sorting, lifting and scoring for LSMT; MW2 includes ground costs and
LP. Cached-cost measurements give **both** methods the same precomputed costs.
Marginals and cost ratios are checked/reported. Add `--aggregations min avg` for
eight LSMT methods. Repeat on CPU to separate hardware effects from the algorithm.

Four-input barycenter scaling with distinct random SPD Gaussian components:

```bash
python -m experiments.barycenter.benchmark --task scaling \
  --component-counts 3 5 8 12 20 --kinds B1D --variants fast-fixed fast-fresh \
  --seeds 0 1 2 --steps 200 --patience 0 --device cuda \
  --output-dir results/barycenter_scaling
```

Every input has K components; Kb defaults to 4K-3, and all four input weights
are 1/4. These are new scaling datasets, not copies of the original Figure 4
GMMs. The original 7x7 experiment is preserved by `run.py`. `--max-tuples 200000`
prevents allocating an excessive multi-marginal LP; skipped references are
explicitly recorded and never assigned a zero runtime or quality gap.

## Timing boundaries and hardware

All solver timers start from resident, fitted input GMMs. Fit/preprocessing and
one-time input placement are shared/excluded. Total includes initializer
screening and candidate transfer, projection-bank setup, every update,
validation and best-state selection. Validation time is a subset of solve time,
not added twice. Output serialization, plots and post-hoc metrics are excluded.
Cached initializers are charged their full measured setup cost to every method.

CUDA timers synchronize before/after wall-clock measurement. Throwaway warmup
states remove lazy optimizer/kernel initialization; measured solves start fresh.
MW2's multi-marginal LP runs on CPU; the variational POT LP also uses CPU, even
when tensors live on CUDA, and its required transfer overhead is timed. The
manifest records hardware, thread count, seeds and numerical settings. Pairwise
and reference timings use repeated measurements; optimizer variability is
reported across independent seeds. Run timing commands without competing jobs.

The fast grid's `objective_squared` is its validation objective, while legacy
records use their training bank. `objective_role` records this distinction.
Use a new output folder after changing settings; resume rejects mismatches.

## Tests

```bash
pytest -q experiments/barycenter/test_fast.py experiments/barycenter/test_barycenter.py \
  tests/test_transport.py tests/test_distribution_projections.py tests/test_linalg_batching.py
```

Checks cover dense/sparse fast values and gradients for all eight methods,
projection-fiber ties, MW2 dual gradients for masses, algebraic Gaussian/B
geometry, one-start screening, resampling/validation reproducibility, checkpoint
restoration, and stopping semantics. CUDA checks run only when available.
