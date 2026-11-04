# 3DGS compression: MW2 and lifted sliced OT

This folder implements the experiment described in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
The default scene experiment uses Train and Truck, retention 10%, 1%, and 0.1%,
with image evaluation before and after a common fine-tuning schedule.

The code provides compression, rendering, fine-tuning, evaluation and reporting
entry points for local scene data and checkpoints.
CPU numerical and synthetic orchestration tests are recorded in [VALIDATION.md](VALIDATION.md).
**Real-scene CUDA compression, rendering and fine-tuning have not been run by the code author in this environment.**
Run the preflight and one-scene pilot before launching the full matrix. GPU quality,
convergence and speedup are experimental outcomes, not assumed properties.

## What is compared

| CLI name | Method |
|---|---|
| `mw2` | Exact balanced EMD with full Gaussian W2 costs, separately in each block |
| `min-Mix`, `min-SMix`, `min-B`, `min-B1D` | Minimum true lifted cost among the fixed projection bank |
| `avg-Mix`, `avg-SMix`, `avg-B`, `avg-B1D` | Average of the same lifted couplings |
| `ghap` | Pinned upstream GHAP sampler, including its native output-count policy |
| `hmw-local` | Reviewed HMW XYZ-Hilbert partition and native GHAP-cost Lloyd solver |
| `identity` | Uncompressed checkpoint, exported and evaluated through the same PLY route |

The default main table contains MW2, four min-LSOT methods, GHAP and HMW-local.
All four avg methods are implemented and can be selected from the command line.
There is no min-opt method in this experiment.

The controlled MW2/LSOT methods share source opacity weights, partition, exact
output budget, initialization, fixed target masses, Bures update, appearance
rule, export policy and fine-tuning. GHAP/HMW-local retain their own algorithms;
they are application baselines, not a claim of an otherwise identical solver comparison.
HMW-local here is **not** the full Cholesky-HMW distance. Blockwise MW2 is **not**
the exact global MW2 optimum of the complete scene.

## 1. Environment on A100 / H100

Use one NVIDIA GPU per job. A100/H100 are suitable for the float64 geometry;
the CUDA renderer and trainable 3DGS attributes use float32. No distributed
training or memory pooling across GPUs is required. Run independent jobs in
different output folders if using several GPUs.

Use a dedicated Python 3.10--3.12 environment, a C++ compiler, Eigen3 headers,
and a CUDA toolkit with `nvcc`. The documented installation choice is PyTorch
2.4.1 + torchvision 0.19.1 + CUDA 12.4. Keep the system toolkit and PyTorch CUDA
major versions compatible. PyTorch3D is pinned at `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba`.
This GPU installation recipe still needs validation on the actual server.

```bash
cd /path/to/LSOT_GMM
python3.11 -m venv .venv-3dgs
source .venv-3dgs/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Install Eigen headers using the server package manager, or set
# EIGEN3_INCLUDE_DIR to a directory containing Eigen/Core.
# Ubuntu example: sudo apt-get install build-essential libeigen3-dev

export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.0;9.0"
bash 3DGS/setup_env.sh --install-torch
```

Omit `--install-torch` to keep an existing compatible PyTorch/torchvision pair.
The setup script pins GHAP at `27b3a581c861069cc97e5629e216dc3d1f42a98a`, builds
only its required CUDA submodules, installs PyTorch3D, and builds the unchanged
HCP implementation already supplied under `HMW_code`. It rejects a modified GHAP checkout.
The OpenGL viewer is not needed on a headless server.

For numerical development without a GPU or renderer:

```bash
python -m pip install -e '.[test]' plyfile
python -m pytest 3DGS/tests -q
python 3DGS/preflight.py --config 3DGS/configs/smoke.json --device cpu --skip-hcp
```

## 2. Data and checkpoint preparation

Edit a copy of `3DGS/configs/hmw_protocol.json`, for example
`3DGS/local_hmw.json`. Paths are relative to the repository root unless absolute;
environment variables in paths are expanded. Local config files are git-ignored.

For each scene supply:

- A COLMAP scene directory with `images/` and `sparse/0/`.
- The corresponding trusted local 3DGS checkpoint (`.pth`), or a Gaussian PLY.
- For a PLY input, `spatial_lr_scale` from the original checkpoint if fine-tuning.
- Train: checkpoint iteration 30,000 and resolution `1`.
- Truck: checkpoint iteration 15,000 and resolution `-1`.
- Native SH degree 3, matching the source checkpoint.

No large scene or checkpoint is committed or silently downloaded. The historical
HMW checkpoints were not present in the uploaded repository. The original
[3DGS repository](https://github.com/graphdeco-inria/gaussian-splatting) documents
the dataset/COLMAP preparation. Do not replace a missing historical checkpoint
with a different backbone and call it an exact numerical reproduction.

```bash
python 3DGS/prepare_data.py --config 3DGS/local_hmw.json
python 3DGS/preflight.py --config 3DGS/local_hmw.json --renderer --scene train
```

`prepare_data.py` validates source arrays, counts, SH and checkpoint iteration,
then hashes the checkpoint, camera files and images. The renderer writes the
actual camera names, transforms and resolutions used by the pinned GHAP loader.
The orchestrator checks that all methods and both phases use the same split.

If a new backbone is necessary, train it explicitly:

```bash
python 3DGS/train_backbone.py --config 3DGS/local_hmw.json \
  --scene train --output-dir results/3dgs_backbones/train
```

Then point the scene config to the new `chkpnt30000.pth` (Truck: `chkpnt15000.pth`)
and set `backbone_provenance` to `newly_trained`. A separate provenance file
records that this is not the unavailable historical HMW checkpoint.

## 3. Pilot, then the full scene experiment

```bash
# First check compressed images before paying for fine-tuning.
python 3DGS/run.py --config 3DGS/local_hmw.json \
  --scenes train --retentions 0.1 --methods mw2 min-B1D ghap hmw-local \
  --no-finetune --output-dir results/3dgs_pilot_pre

# Same pilot with the common fine-tuning schedule.
python 3DGS/run.py --config 3DGS/local_hmw.json \
  --scenes train --retentions 0.1 --methods mw2 min-B1D ghap hmw-local \
  --output-dir results/3dgs_pilot_full

# Full configured Train/Truck x three retentions x all default methods.
python 3DGS/run.py --config 3DGS/local_hmw.json \
  --output-dir results/3dgs_full

# All eight LSOT variants plus MW2.
python 3DGS/run.py --config 3DGS/local_hmw.json \
  --methods mw2 min-Mix min-SMix min-B min-B1D avg-Mix avg-SMix avg-B avg-B1D \
  --output-dir results/3dgs_all9
```

`--compression-only` exports PLYs without requiring images, renderer or fine-tuning.
It is useful to debug transport but does not produce PSNR/SSIM/LPIPS.
The default orchestration stops on a failed job; `--continue-on-error` records
failures and proceeds, then exits nonzero if any configuration failed.

The HMW fine-tuning schedule is preserved: a fresh Adam state, checkpoint index
30,000 for both scenes, and loop indices 30,001 through 35,000. Geometry and
appearance remain trainable; densification and further compaction are disabled.
The unchanged GHAP loop skips the last optimizer update, hence 4,999 updates.
This is the reviewed HMW protocol, not GHAP's separate 15k+15k paper schedule.

The 10 outer updates and 64 Bures iterations are starting budgets, not claims
of convergence. Every block records `converged` or `iteration_cap`, and the
returned geometry minimizes the method's refreshed objective among the
evaluated iterates. Test-image metrics never select the compression iterate.

## 4. Solver scalability and ablations

Edit checkpoint paths in a copy of `configs/scaling.json`.

```bash
python 3DGS/benchmark.py --config 3DGS/local_scaling.json
python 3DGS/plot.py --results-dir results/3dgs_scaling
```

This grows **one full OT problem** on nested subsets of real scene Gaussians:
N = 1k, 2k, 5k, 10k, 20k, 50k, 100k and M = floor(0.1N), with L=100.
All methods use the same subset and initial target at each size. It does not
split larger problems into more fixed-size blocks.

`cold_solver_seconds` includes source geometry/projection preparation and one
solve. `warm_solver_seconds` reuses that source preparation and solves the same
target; it is reported separately. Common source extraction and target
initialization precede these solver timers. GPU cost computation, host transfers
and CPU EMD are included in MW2. CPU peak RSS is the configuration worker's
high-water mark; GPU allocated peak is measured for the cold trial.

`solver.max_dense_bytes` is a conservative allocation guard, not a measured
memory value. `timeout_seconds` bounds the entire configuration worker,
including its warmups and repeats. `skipped_budget`, `timeout`, `out_of_memory`
and unexpected `failed` cases stay in the CSV. A speedup is computed only where
both MW2 and the compared solver completed. Large MW2 also needs substantial
CPU RAM and CPU solve time even on an 80 GB GPU.

```bash
# Spatial-block-size ablation: same cap/partition for controlled methods.
python 3DGS/sweep.py --config 3DGS/local_hmw.json \
  --parameter block_cap --values 100 250 500 1000 \
  --methods mw2 min-Mix min-SMix min-B min-B1D

# Prefixes of the same fixed bank, not newly sampled banks at each L.
python 3DGS/sweep.py --config 3DGS/local_hmw.json \
  --parameter projections --values 25 50 100 200 \
  --methods mw2 min-Mix min-SMix min-B min-B1D
```

The native HMW-local baseline always uses cap=100. The common main setting
does not guarantee increasing speedup with total scene size: fixed local
blocks can keep individual OT problems small. The separate scaling experiment
is needed to test the intended claim.

## 5. Outputs and plots

Each scene/retention/method directory contains:

- `job.json`, `provenance.json`, `compression.json`, and `history.jsonl`.
- `timing/repeat_*/compression.json`: individual measurements; one warmup and
  three measurements by default. Each call rebuilds its source caches.
- `pre/point_cloud/iteration_1/point_cloud.ply`: measured repeat 0's model.
- `pre/test/pre/{renders,gt}/`: all held-out images, indexed identically across methods.
- `pre/results.json`, `per_view.json`, `evaluation.json`, `camera_manifest.json`.
- `post/start.pth`, `post/finetune.json`, `post/point_cloud/iteration_35000/point_cloud.ply`.
- Corresponding post-phase images and metric files.
- `complete.json` only after required artifacts and finite per-view metrics pass checks.

Root outputs include `summary.csv`, `quality.csv`, `runtime.csv`,
`timing_breakdown.csv`, input manifests and resolved configuration. The main
time is median compression wall time, excluding checkpoint disk IO, PLY disk
IO and fine-tuning; those costs are recorded separately. Native baseline
adapters include transfer/preparation in the compression call. These timings
must not be copied into the historical HMW table as if protocols were identical.

```bash
python 3DGS/summarize.py --results-dir results/3dgs_full
python 3DGS/plot.py --results-dir results/3dgs_full
python 3DGS/plot.py --results-dir results/3dgs_full \
  --scene train --retention 0.1 --phase post --views 0 5 10 \
  --methods identity ghap hmw-local mw2 min-B1D
```

Panels prepend GT and use the same view index and pixel dimensions for every
method. `--crop LEFT TOP RIGHT BOTTOM` applies an identical pixel crop.
Plots include pre/post quality, stage times, per-block objective histories,
solver time, memory, valid speedups and transport-cost gaps.

Resume requires matching data/config/code/environment fingerprints and checked
artifact hashes. If any changes, choose a new output directory. Timing repeats
are not independent quality seeds; run separate seed configurations to measure
quality variability. Numerical nonconvergence is recorded even if a valid PLY
and image metrics were produced.

## 6. Mathematical choices

Source OT masses are normalized activated opacity. Zero-opacity atoms are
excluded from controlled transport and recorded with their original indices.
Initial target masses are group opacity sums and remain fixed throughout
the controlled reduction. They are not renderer opacity parameters.

MW2 minimizes the true Gaussian W2 cost over balanced couplings. Min-LSOT
selects the smallest true lifted cost; avg-LSOT uses the mean plan. The four
projection laws and their sampling distributions come directly from the
existing `lsot`, `param_proj` and `distribution_proj` modules. With ties,
the lift distributes mass proportionally across each Cartesian fiber product.
That product is streamed and may be dense; the n+m-1 support bound applies only
without projected ties. Avg plans remain implicit sums, not dense matrices.

Conditional means and Gaussian Bures barycenters update geometry for a fixed
plan. The covariance update is neither an arithmetic covariance average nor
mixture moment matching. A changing LSOT ordering does not imply monotone
refreshed objective values. The code never silently substitutes an old plan
to force LSOT monotonicity.

Default appearance uses the geometry-generating plan: choose the minimum
GHAP-cost donor among incoming edges and copy its SH and raw opacity. For the
initial iterate, which was not generated by a transport update, use its current
plan and record that role. Alternatives are `appearance="full-block"` and
`appearance="mean"`; the latter averages activated opacity, then takes logit.
The default covariance export repairs only roundoff; `export_floor="ghap"`
is an explicit absolute-1e-6 ablation for controlled methods. Native GHAP keeps
its own donor and eigenvalue policy and may output fewer than the requested
Gaussians. Actual counts are always reported. If the requested GHAP budget is
smaller than its number of depth-10 KD leaves, the upstream allocation would
create a zero-target leaf. That configuration is recorded as
`unsupported_native_budget`; the runner does not silently change its KD depth.

## 7. Tiny end-to-end smoke test

```bash
python 3DGS/make_smoke_data.py
python 3DGS/run.py --config 3DGS/configs/smoke.json
python 3DGS/plot.py --results-dir results/3dgs_smoke
```

This exercises source IO, shared reduction, all nine methods, subprocesses,
repeat records, PLY export, summary tables and convergence plots on CPU. It
does not replace a real-scene renderer test. Re-running reuses completed results
only when their fingerprints still match.

See [SOURCE_REVIEW.md](SOURCE_REVIEW.md) for source attribution and distinctions
between the historical HMW branches. Existing interpolation, color-transfer
and barycenter implementations are not modified by this folder.
