# Texture synthesis with GMM transport

Compare MW2 and the eight min/avg LSOT variants on the same one-scale
texture synthesis pipeline. Each method gets the same two fitted patch
GMMs and changes only the component coupling used to build the
conditional-mean map. The code keeps the original patch locations and
averages overlapping patches to reconstruct an RGB image.

The baseline follows Delon and Desolneux's [MW2 paper](https://arxiv.org/abs/1907.05254),
Sections 6.3 and 7.2: RGB 3×3 patches, stride one, one scale, direct
`T_mean`, and no nearest-neighbor step. For ADSN, patch extraction, and
aggregation, see the [StoRI notebook](https://github.com/storimaging/Notebooks/blob/1127497deec4a3875dcd1c9eef3fddc493a40b1d/ImageGeneration/Texture_synthesis_with_GMMOT.ipynb)
by Arthur Leclaire and Lucía Bouza and its
[helpers](https://github.com/storimaging/Notebooks/blob/1127497deec4a3875dcd1c9eef3fddc493a40b1d/ImageGeneration/AuxiliarFunctions/Texture_synthesis_with_GMMOT.py).
This implementation uses the same ADSN and overlap-average formulas and
the repo's existing `BarycentricMap`. It does not import the StoRI helpers
at runtime. The StoRI demo itself uses 5×5 patches, K=4, and 1,000 exemplar
patches; our presets do not use those parameters.

## Run

From the repository root:

```bash
python -m pip install -e .
python -m experiments.texture.run --config experiments/texture/configs/smoke.json
python -m experiments.texture.run --config experiments/texture/configs/paper.json
```

The main preset uses three StoRI exemplars (stone, fabric, and a pattern),
with maximum side length 256, seed 0, K=10, 30, 50, and L=100. It downloads
only the selected images from a pinned revision of
[storimaging/Images](https://github.com/storimaging/Images) and checks their
SHA-256 hashes. Cached files go into `data/texture/stori/`. Pass an explicit
local path and `--no-download` for an offline run.

```bash
python -m experiments.texture.run --config experiments/texture/configs/paper.json \
  --images ground1013_small.png --component-counts 10 30 50 \
  --methods MW2 min-LSOT-B1D avg-LSOT-B1D --output-dir results/texture_b1d
```

Use `--seeds 0 1 2` for multiple realizations. The code draws one ADSN per
exemplar and seed and keeps it for all K and methods. Both GMMs use full
covariance EM. Each is fitted to at most 10,000 patches selected without
replacement, held fixed across K; `--fit-patches 0` fits on every patch.
All source patches are transformed regardless of the fitting limit. The
10,000-patch cap and the K=30, 50 cases are part of this benchmark, so the
main preset is not an exact reproduction of the paper's Figure 12.
Fit convergence and iteration counts are saved for both GMMs.

MW2 builds the dense Gaussian cost matrix with the same kernel that
min-LSOT uses for requested component pairs and solves it with POT
`ot.emd`. Min-LSOT chooses the lifted plan with lowest original Gaussian
cost, and avg-LSOT averages the lifted plans, not just their costs.
Min and avg share the same sampled bank within each projection family.
The map collapses the pairwise affine maps into one affine map per
source component, using the plan's conditional weights. This is
algebraically the same as the direct sum in the StoRI helper.

## What is measured

- **Patch SW2 and W2 (lower is better).** Extract overlapping patches
  from the final clipped RGB image and the exemplar. SW2 uses up to
  2,048 patches and 128 independent spatial directions in the patch
  vector space. W2 is an exact empirical OT solve on up to 256 patches
  from each image with squared Euclidean ground cost, then a square
  root. It is not the W2 of all patches. Output locations, exemplar
  locations, and evaluation directions are fixed across K and methods.
- **RGB autocovariance RMSE (lower is better).** Root mean squared
  difference of RGB cross-covariance matrices at zero displacement
  and valid horizontal, vertical and diagonal lags of 1, 2, 4, 8, 16
  pixels. This checks some spatial statistics beyond a single patch.
  We compare distributions and spatial statistics, so no pixelwise
  ground truth is assumed for the newly generated texture.
- **Plan diagnostics.** Lifted squared Gaussian cost, nonzero entries,
  and marginal error. The code checks that each plan has the prescribed
  marginals, MW2 minimizes the component cost, and min costs no more
  than avg on the same bank. Plan cost is a diagnostic and does not
  measure the quality of the reconstructed image.
- **Time.** CPU float64, one thread by default. Record image loading,
  ADSN, patch preparation, shared EM, projection-bank generation,
  component-plan solve, map setup, application, and patch aggregation.
  `plan_ms` includes projecting and lifting for LSOT and Gaussian costs
  plus the LP for MW2. Bank generation is separate. Avg-LSOT can make
  its plan without computing Gaussian pair costs, so its diagnostic
  cost is evaluated outside the timer. Method phases use the median
  of three sequential runs after one untimed warmup per method;
  method order rotates. `total_ms` charges each method the full cost
  of shared preparation, fitting, and its own bank in addition to the
  measured plan/map/aggregation. Downloads, evaluation, and output
  writing are outside timing. Compare both plan and total times:
  a faster plan may be offset by a denser map or shared EM.

## Outputs

Under `results/texture/` (or `--output-dir`):

- `settings.json`, `metrics.csv` (one row per exemplar/seed/K/method),
  `timings.csv` (per repeat), `summary.csv` (mean/std by K and method),
  and `adsn_metrics.csv` (the untransported baseline). Delta columns
  are method minus ADSN. Speedups are ratios against MW2 on the same
  exemplar/seed/K.
- `<image>/seed_<seed>/`: exemplar, ADSN, and input metadata.
- `<image>/seed_<seed>/K<K>/`: reconstructed PNGs, comparison PNG/PDF,
  fitted GMMs, EM status, and sparse component plans in NPZ.
- `sweep.png` and `sweep.pdf`: quality and time versus K.
