# MW2 and variational LSOT barycenters

This experiment follows Julie Delon's
[GMM_OT_Barycenters.ipynb](https://github.com/judelo/gmmot/blob/0984edc826b113e35c3260b699a4ff49ab39d25f/python/GMM_OT_Barycenters.ipynb).
It retains MW2 and adds **eight** methods: avg/min LSOT with Mix, SMix, B,
and B1D. There is no min-opt method and no projection training.

## Data and settings

| Notebook experiment | Inputs | Barycenter weights | Display |
| --- | --- | --- | --- |
| Gaussian example | Three 2D Gaussians, exact means/covariances from cell 3 | `(1/3,1/3,1/3)` | Domain `[-2,2]^2`, 50 x 50 density grid |
| Synthetic GMMs | Four 2D GMMs, component counts `(3,4,4,3)`, exact arrays from cell 8 | Bilinear weights on a 7 x 7 grid | Domain `[0,1]^2`, 50 x 50 density grid |
| Image GMMs | redcross, duck, star, batman; 12 full-covariance components per image | Same 7 x 7 grid | Domain `[0,128]^2`, 128 x 128 density grid |

The grid uses

```
omega = ((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty)
tx, ty in {0, 1/6, 2/6, ..., 1}.
```

Columns vary `tx`; rows vary `ty`. The corner layout is
`mu0, mu1` on the top row and `mu2, mu3` on the bottom row, matching the
notebook's displayed indexing `gmminterp[i + nb_images*j]`.
Corner barycenters are the exact input GMMs and need no optimization.

**Image provenance:** the gmmot repository does not commit the four original
PNG files. To make the experiment self-contained, `assets/notebook_masks.json`
contains foreground masks recovered from the notebook's saved input-image
display. The redcross and duck masks were checked against POT's 128 x 128
images and match their foreground support exactly. Original pixel identity
for star and batman cannot be verified. These are explicitly recorded as
recovered data, rather than original PNGs. Provide `--image-dir` with the
four original PNGs to reproduce the raw-image experiment without this
limitation. No unrelated replacement shapes are generated.

Preprocessing follows the notebook: `u = 1 - blue_channel`, vertically flip
`u`, take **every nonzero pixel** as a point `(column, flipped_row)`, and fit
`sklearn.mixture.GaussianMixture(n_components=12, covariance_type='full')`.
There is no resizing, pixel subsampling, intensity weighting, or spatial
normalization. Seed 0 is added because the original notebook leaves EM
unseeded. Thus even with original PNGs, an identical historical unseeded EM
fit cannot be guaranteed. Both MW2 and all LSOT methods use the same saved
input GMMs.

## Algorithms

**MW2:** compute the Gaussian barycenter and its weighted Gaussian W2 cost
for every active component tuple, then solve the discrete multi-marginal
LP. `reference.py` vectorizes the notebook's matrix computations and uses
sparse marginal constraints with SciPy HiGHS. The mathematical finite LP
is unchanged. Gaussian covariances use the notebook's fixed-point update
starting from the identity, with **N=10** by default. This is the notebook's
numerical MW2 reference, not an exact Gaussian covariance solution; increase
`--gaussian-iterations` for greater accuracy.

**LSOT:** optimize a candidate GMM `nu` using the sampled variational losses

```
avg: sum_j omega_j * mean_l C(mu_j, nu; s_l)
min: sum_j omega_j * min_l  C(mu_j, nu; s_l)
```

`C` is the Gaussian W2 cost evaluated with the lifted component plan.
It is not the projected scalar cost. Each pair in the min objective selects
its own minimizing slice. The four projection formulas and their sampling
laws are reused directly from the repository.

L-BFGS updates means, full covariances, and mixture weights. Cholesky factors
with positive diagonals ensure SPD covariances; softmax ensures positive
normalized weights. Projected ranks/fibers and lifted masses are recomputed
at every closure evaluation. The projection bank is fixed throughout
optimization. Gradients through transport masses are retained when mixture
weights change. Hard ordering and cumulative-mass coincidences make the
objective piecewise smooth; this is a local numerical solver without a
global optimality guarantee. `stalled` is a stopping condition, not a
certificate of optimality. The best feasible iterate is restored.

All eight methods start from the same method-independent common-quantile
Gaussian initialization. It uses a fixed spatial ordering and **does not
use the MW2 LP solution**. The default component budget is `sum(Kj)-J+1`:
1 for the Gaussian example, 11 for synthetic GMMs, and 45 for image GMMs.
This is the basic feasible support bound of the MW2 component LP. Override
it with `--barycenter-components`. Means/covariance parameters are internally
scaled for conditioning; all projections and costs still use the original
physical coordinates.

## Run from the repository root

```bash
python -m pip install -e .

# Full notebook grid, MW2 + eight LSOT methods, L=100.
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/mw2_reference.json \
  --output-dir results/barycenter

# Faster pipeline check; retains K=12 image fits but uses a 3x3 grid,
# L=8 and three L-BFGS outer steps. Do not report it as the full experiment.
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/smoke.json \
  --output-dir results/barycenter_smoke

# Original PNGs, if available.
python -m experiments.barycenter.run \
  --config experiments/barycenter/configs/mw2_reference.json \
  --image-dir /path/to/original/images \
  --output-dir results/barycenter_original

# Prepare image clouds/GMMs without running barycenters.
python -m experiments.barycenter.prepare --output-dir data/barycenter

# Optional checks.
python -m pytest -q experiments/barycenter/test_barycenter.py
```

CPU is the default for these small 2D problems. `--device cuda` or
`--device auto` runs LSOT tensors/L-BFGS on an available GPU; the MW2 LP
remains on CPU. This implementation was validated on CPU; CUDA tests require
a CUDA runtime. Use `--mp4` to also generate the notebook-style perimeter
movies if ffmpeg is installed; GIF export needs no ffmpeg.

Completed node/method results are reused on rerunning the same command.
Changed settings or input GMMs are detected by a fingerprint: use a new
output directory or `--no-resume` to recompute. The latter replaces saved
results in that directory. A partially completed run can continue from its
saved node records.

## Colab

Open `colab.ipynb`, or run:

```python
%cd /content
!git clone https://github.com/tmp0810/LSOT_GMM.git
%cd /content/LSOT_GMM
!git pull --ff-only origin main
!pip -q install -e .

!python -m experiments.barycenter.run \
    --config experiments/barycenter/configs/mw2_reference.json \
    --output-dir /content/LSOT_GMM/results/barycenter
```

If the repository was already cloned, skip the clone command. The full run
has **891 method/node results**: 9 for the Gaussian example and 441 for each
7 x 7 experiment. Plotting and evaluations are excluded from solver timing.

## Output

```
results/barycenter/
  image_data/                  # Prepared PNG masks, point clouds, fitted input GMMs, provenance
  metrics.csv                  # One row per method and grid node
  summary.csv                  # Per-case/method means, excluding exact corners
  environment.json
  synthetic/                   # Also gaussian/ and images/
    inputs/                    # Input NPZs, clean individual images, inputs.png
    initializations/           # Same starting GMM for all eight LSOT methods
    bank_Mix.npz               # Also SMix, B, B1D
    MW2/                       # Same layout for each LSOT method
      barycenters.png          # Complete 7x7 contour grid, notebook-style
      barycenters.pdf
      barycenters/r03_c03.npz   # GMM parameters, density grid, input weights
      images/r03_c03.png        # Individual panel without caption or x/y labels
      records/r03_c03.json
      history/r03_c03.csv       # L-BFGS loss history for LSOT methods
      perimeter.gif            # Corner-to-corner traversal, as in the notebook
    center_comparison.png      # MW2 + all eight LSOT methods at equal weights
    center_convergence.png     # Each objective normalized by its own initial value
```

Use each method's `barycenters.png`/`.pdf` to reproduce the notebook's main
figures. Individual images default to no axes; `--show-axes` retains numeric
ticks without adding titles or axis labels.

Metric meanings:

| Field | Meaning |
| --- | --- |
| `objective_squared` | Own-method weighted variational objective; for MW2, the multi-marginal LP cost with N Gaussian iterations |
| `initial_objective_squared` | LSOT objective before optimization |
| `mw2_objective_squared` | Common evaluation: weighted sum of pairwise MW2 squared distances from inputs to the candidate |
| `mw2_to_reference` | Mixture Wasserstein distance to this node's numerical MW2 barycenter |
| `sw2_to_reference` | Empirical sliced W2 distance to the MW2 barycenter, using 1024 samples and 100 spatial directions by default |
| `density_l1_to_reference` | L1 difference of densities normalized on the displayed grid; affected by domain truncation |
| `solve_ms` | MW2 Gaussian tuple computations + LP, or LSOT objective setup + L-BFGS |
| `initialization_ms` | Shared LSOT initialization and device transfer |
| `total_ms` | `solve_ms + initialization_ms`; shared projection-bank generation is outside this timing |
| `status`, `iterations`, `evaluations` | Numerical stopping status, outer L-BFGS steps and closure evaluations |

The notebook primarily reports contour grids and movies. The metrics above
are added diagnostics. Different avg/min losses are different objectives;
their raw loss values should not be treated as a common accuracy score.
Distances to MW2 measure agreement with that reference and do not prove that
an LSOT barycenter is incorrect when it differs.

## Reproduce the bundled recovery

Download the pinned original notebook **with its saved outputs**, then run:

```bash
python -m experiments.barycenter.prepare \
  --recover-notebook-output GMM_OT_Barycenters.ipynb \
  --mask-output recovered_masks.json
```

The recovery uses inverse viridis lookup at the centers of the 128 x 128
displayed pixels. Foreground digests, pixel counts, source notebook blob,
cell number, and sampling coordinates are recorded in the bundled manifest.
The stripped notebook without output images cannot be used for recovery.
