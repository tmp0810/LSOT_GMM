# LSOT for Gaussian mixture models

Averaged **LSOT-Mix** and **LSOT-SMix**, compared with the original **MW2**
baseline on color transfer. The experiment follows the RGB/GMM/Tmean setting
of [Delon and Desolneux](https://arxiv.org/abs/1907.05254) and their
[reference notebook](https://github.com/judelo/gmmot/blob/master/python/GMM_OT_color_transfer.ipynb).

[Open the runnable notebook in Colab](https://colab.research.google.com/github/tmp0810/LSOT_GMM/blob/main/notebooks/color_transfer.ipynb).

## Run

Python 3.10 or later, from this repository's root:

```bash
python -m pip install -e .
python -m experiments.color_transfer.run --config experiments/color_transfer/configs/smoke.json
```

The smoke run downloads the two original example images, uses K0=K1=10,
resizes to a maximum side of 96 pixels, fits at most 4,000 pixels per image,
and tests L=4,8. It checks the pipeline; it is not a paper benchmark.

Run the full reference image setting:

```bash
python -m experiments.color_transfer.run --config experiments/color_transfer/configs/mw2_reference.json
```

This uses Renoir -> Gauguin at the original image resolutions, all pixels
for EM, full covariances, K0=K1=10, kmeans initialization, and n_init=1.
EM uses scikit-learn's standard max_iter=100, tol=1e-3, reg_covar=1e-6.
Fixed random states are added for reproducibility. There is no retraining
of the GMMs for different methods or projection budgets.

To use your own images, change sizes, or repeat data seeds:

```bash
python -m experiments.color_transfer.run --source source.png --target target.jpg --components 10 --projections 10 50 100 --seeds 0 1 2 --output-dir results/my_pair
```

Source and target image sizes may differ. `--components 10 15` sets different
source/target component counts. `--max-side` and `--fit-pixels` are explicit
approximations; omit them to keep the full image setting.

In Colab, `%cd /content/LSOT_GMM` before installing/running shell commands.
Unlike `!cd`, `%cd` persists across cells. The provided notebook handles
cloning and the working directory automatically.

## What stays the same, and what changes

`gmmot.py` is preserved unchanged. Its legacy `GW2` function solves the
**Mixture Wasserstein** component LP and returns a squared cost (it is not
Gromov--Wasserstein). It is called directly for the MW2 baseline.

All methods use the same Gaussian W2-squared ground cost:

$$c_{ij}=\|m_i-n_j\|^2+\mathrm{tr}\left(\Sigma_i+\Lambda_j-2(\Sigma_i^{1/2}\Lambda_j\Sigma_i^{1/2})^{1/2}\right).$$

MW2 minimizes `sum(P * C)` over component couplings. LSOT replaces this
component plan by a lifted plan from each scalar parameter projection:

$$p_s^{\rm Mix}(m,\Sigma)=\psi_1\theta^\top m+\psi_2\langle A,\log\Sigma\rangle_F,$$
$$p_s^{\rm SMix}(m,\Sigma)=\psi_1\theta^\top m+\psi_2\log\sqrt{\theta^\top\Sigma\theta}.$$

The original Mix sampling law in `param_proj/sot_gms.py` is preserved:
`A = Q diag(v) Q.T`, with `v` uniform on the unit sphere and `Q` Haar
orthogonal via sign-corrected Gaussian QR. This is **not** a replacement by
uniform sampling of the entire symmetric Frobenius sphere. Mean direction
`theta` and mixing direction `psi` are sampled independently on their spheres.
Mix and SMix share theta/psi; all budgets use prefixes of one maximum-size
bank. Regenerating banks with different maximum lengths is not the same as
taking prefixes; saved banks are the definitive reproducibility record.

Within each projection, equal scalar locations are aggregated into fibers.
For sorted fiber masses A_r,B_t and cumulative masses U_r,V_t,

$$\lambda_{rt}=[\min(U_r,V_t)-\max(U_{r-1},V_{t-1})]_+,$$
$$\bar P^s_{ij}=\lambda_{rt}\frac{\alpha_i}{A_r}\frac{\beta_j}{B_t},\quad i\in I_r,\ j\in J_t.$$

We implement this original proportional lift, including exact collisions;
there is no zero-cost masking and no arbitrary index tie breaking. No
positive tolerance merges nearby but unequal projected locations. The fast
path batches directions without ties; exact ties use the fiber formula.
Consequently an intentionally constant projection has a product self-plan,
not an identity plan. Identity tests use separating random projections.

The averaged plan and its true lifted cost are

$$P_L=L^{-1}\sum_{\ell=1}^L\bar P^{s_\ell},\qquad \widehat{\mathrm{LSOT}}_L^2=\sum_{ij}(P_L)_{ij}c_{ij}.$$

The original `MixSW`/`SMixW` function signatures remain usable. They still
return **projected** distances, not LSOT. The refactor exposes their scalar
projections and adds optional `bank`/`seed` keyword arguments. Matrix logs
use `torch.linalg.eigh`, so `geoopt` is no longer required for these functions.
Random realizations for an old global seed can change; the distribution does not.

## The same Tmean map for every plan

For the Gaussian pair map `T_ij(x)=M_ij x+b_ij`, with

$$M_{ij}=\Sigma_i^{-1/2}(\Sigma_i^{1/2}\Lambda_j\Sigma_i^{1/2})^{1/2}\Sigma_i^{-1/2},\qquad b_{ij}=n_j-M_{ij}m_i,$$

we use precisely the posterior-weighted map from the reference notebook:

$$T_P(x)=\sum_i\frac{\alpha_i g_i(x)}{\sum_k\alpha_k g_k(x)}\sum_j\frac{P_{ij}}{\alpha_i}T_{ij}(x).$$

This is a barycentric assignment, not a guarantee that `T_P#mu = nu`.
The implementation first collapses the pair maps into one affine map per
source component and then processes pixels in batches. It does not allocate
the reference notebook's `K0 x K1 x 3 x Npixels` tensor. This mathematically
equivalent implementation is shared by MW2, LSOT-Mix and LSOT-SMix.

Both the unfiltered output and the reference's optional guided-filter output
are saved. Filtering uses the **unclipped** displacement:
`source + guided_filter(mapped - source, source)`, independently per channel,
with radius 10 and epsilon 1e-4. Clipping to [0,1] happens for display/saving
and color evaluation. Stochastic `Trand`, hard-class `Tmax`, and separable
channel transfer are not part of this comparison; all methods use `Tmean`.

## Devices and timing

`--device auto` uses CUDA if available; `--device cpu` forces CPU;
`--device cuda:0` requires CUDA and fails explicitly if it is unavailable.
Default precision is float64. Install a CUDA-enabled PyTorch build to use GPU.

- **EM** uses scikit-learn on CPU and is timed once for the shared GMMs.
- **MW2 solver** is the original SciPy Gaussian cost matrix + POT CPU LP.
  Host/device transfers in its wrapper are included in transport time.
- **LSOT solver** and the common map run in PyTorch on the selected device.
  LSOT evaluates true Gaussian costs only on transported component pairs.
- **Transport runtime** includes parameter projection, sorting/lifting and
  true cost evaluation for LSOT; dense cost assembly and LP for MW2.
- **Map setup/application** use the same code for all methods. Pixel output
  transfer to CPU is included in map application. CUDA timings synchronize
  before and after calls. Each stage has warmups and recorded repetitions.
- **Banks** are sampled once before repeated timing; `bank_sampling_ms`
  records this maximum-bank preparation separately. It is not included in
  `transport_ms_mean` or `pipeline_ms`.
- **Pipeline runtime** is shared EM + input preparation + transport + map
  setup + pixel mapping + optional guided filtering. File I/O, downloading,
  plotting and evaluation are excluded. Guided-filter runtime is a single
  CPU measurement. Hardware/backend differences are recorded, not hidden.

No runtime superiority is assumed, particularly at K=10. EM or pixel mapping
can dominate application runtime even when constructing the plan is faster.

## Outputs and metrics

Each run writes the following under its configured output directory:

| Path | Contents |
| --- | --- |
| `metrics.csv` | One row per data seed, method and L |
| `paper_results.tsv` | Means and sample standard deviations across data seeds |
| `config.json`, `metadata.json` | Exact settings, versions, hardware, input hashes, metric conventions |
| `source.png`, `target.png` | Images actually used after any explicit resize |
| `seed_N/gmms.npz` | The two shared fitted GMMs |
| `seed_N/projection_bank.npz` | theta, psi and Mix matrices for the largest L |
| `seed_N/evaluation_bank.npz` | Independent RGB evaluation directions and sampled pixel indices |
| `seed_N/*_plan.npz` | Sparse rows, cols, mass, shape; no forced dense LSOT plan |
| `seed_N/*.png` | Individual transferred images and comparison grids |
| `seed_N/timings.json` | Individual timing repetitions |

`save_raw=true` also writes unfiltered, unclipped float output arrays.

- `cost_squared`: `sum(P_ij * c_ij)`, the unregularized Gaussian transport cost.
- `relative_cost_gap`: `(cost_squared - mw2_cost_squared) / mw2_cost_squared`;
  undefined/empty for effectively zero reference cost.
- `plan_rmse`: `sqrt(sum((P-P_MW2)^2)/(K0*K1))`. The reference is a component
  LP solution, which may be nonunique; this is not a full-distribution W2 plan.
- `color_sw2`: **root** empirical sliced W2 between clipped float output
  colors and target colors, using equal-size pixel samples and a separate
  shared bank of uniform RGB directions. Its value is computed before PNG
  quantization. It is not LSOT's Gaussian-parameter projected distance.
- `guided_color_sw2`: the same evaluation after optional guided filtering.
- `map_rmse_to_mw2`: RGB-coordinate RMSE against the raw MW2 output, not
  against the target image's unrelated spatial content.
- `marginal_l1_error`: maximum of row/column marginal L1 errors.

Within-seed timing std and across-seed std are different columns. Across-seed
std is empty when only one data seed is run. More projections need not improve
an average's cost monotonically. We save results without selecting a preferred
method or projection seed based on observed performance.

## Source layout and tests

| Location | Responsibility |
| --- | --- |
| `gmmot.py` | Unchanged MW2 reference routines |
| `param_proj/` | Original projected distances and reusable Mix/SMix banks |
| `lsot/gaussians.py` | Validated GMMs and sparse-pair Gaussian costs |
| `lsot/plans.py` | Monotone OT, proportional lifting, averaging, MW2 adapter |
| `lsot/maps.py` | Gaussian pair maps and posterior-weighted Tmean |
| `experiments/color_transfer/` | Images, EM, experiment, evaluation, configs |
| `notebooks/color_transfer.ipynb` | Runnable Colab/local walkthrough |
| `tests/` | Independent mathematical and offline end-to-end checks |

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Tests compare projections to independent SciPy formulas, lifts to a literal
dense fiber construction (unequal weights/counts and exact ties), costs/maps
to `gmmot.py`, averaged costs to per-projection costs, permuted self-transport
to identity, and Tmean to the original pixelwise formula. End-to-end tests
cover PNG normalization and images with different shapes. CUDA equivalence
tests run only when a CUDA device is available.

See [ATTRIBUTION.md](ATTRIBUTION.md) for provenance. Downloaded images,
environment files and generated experiment outputs are not committed.
