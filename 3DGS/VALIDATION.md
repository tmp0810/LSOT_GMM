# Validation record

This records tests actually run during implementation. There were no CUDA GPUs,
Train/Truck camera datasets, or historical HMW checkpoints in the implementation
environment. The committed code includes the GPU pipeline, but no real-scene
quality, GPU speedup, or successful GPU installation is claimed here.

## Numerical and integration tests

Environment: Linux x86-64, Python 3.12.14, PyTorch 2.5.1+cpu, POT 0.9.7.post1,
plyfile 1.1.5. These CPU versions are distinct from the documented GPU installation
recipe (PyTorch 2.4.1 / torchvision 0.19.1 / CUDA 12.4).

```bash
python -m pytest 3DGS/tests tests/test_transport.py \
    tests/test_distribution_projections.py -q
```

Result: **63 passed, 14 skipped**. All skips were CUDA-only tests.

The checks cover:

- All eight new LSOT plan implementations against the existing reference code.
- Exact weighted marginals, proportional lifting of tied fibers, and
  MW2 <= min-LSOT <= avg-LSOT on the same input and projection bank.
- EMD primal/dual certificates and guarded dense allocation.
- Analytic commuting-covariance Bures barycenters and fixed-plan descent for
  noncommuting covariances, including thin rotated splats.
- Refreshed LSOT objectives, best-iterate selection and average-plan L1 bounds.
- PLY/checkpoint loading, quaternion/covariance round trips, SH ordering,
  opacity averaging, and exact output budgets.
- The HMW-local adapter against the unchanged uploaded native Lloyd solver.
- Warmups/repeated compression, export, resume hash checks and tamper detection.
- Missing MW2 measurements never generating a speedup, killed workers never
  accepting stale successful results, and resumed failures remaining failures.

The native HCP extension was built from the hash-verified source already in
`HMW_code`, using Eigen 3.4.0 and pybind11 3.1.0. Its permutation/determinism test
passed. No compiled binary is included in the commit.

## End-to-end CPU smoke runs

```bash
python 3DGS/make_smoke_data.py
python 3DGS/run.py --config 3DGS/configs/smoke.json \
    --output-dir results/3dgs_release
python 3DGS/run.py --config 3DGS/configs/smoke.json \
    --output-dir results/3dgs_release
python 3DGS/plot.py --results-dir results/3dgs_release
```

The synthetic source contained 40 Gaussians. Identity and all nine transport
methods completed; every compressed output contained exactly 8 Gaussians.
The second invocation reused all ten completed cases. Source extraction,
optimization, appearance transfer, PLY export, subprocess orchestration, CSV
aggregation and diagnostic plotting all ran. The deliberately tiny budget of
two outer iterations was recorded as capped when appropriate; this is not a
convergence or image-quality experiment.

A separate scaling smoke used the same synthetic PLY, the native Hilbert order,
N=20 and 40, retention 0.2, L=5, one warmup and two measurements. MW2, min-B1D
and avg-Mix all completed at both sizes. A subsequent invocation reused all six
cases. Cold/warm timings, objective values, memory records, CSVs and scaling
plots were produced. These tiny CPU timings are infrastructure checks, not
evidence for a paper's scalability claim.

## Static checks

```bash
python -m compileall -q 3DGS
bash -n 3DGS/setup_env.sh 3DGS/run_full.sh
```

Both passed. All seven Python code cells in `colab.ipynb` were compiled;
the notebook was not executed on Colab. The source tree outside `3DGS/` was
not modified by this implementation.

## Validation still required on the user's GPU

1. Install/build the pinned renderer, simple-knn, fused-ssim and PyTorch3D on
   the actual A100/H100 server. Run numerical CUDA preflight and CUDA tests.
2. Supply matching Train/Truck data and trusted source checkpoints. Run
   `prepare_data.py` and `preflight.py --renderer`.
3. Run one scene at retention 0.1 with MW2, min-B1D, GHAP and HMW-local, then
   inspect both rendered views and block convergence records.
4. Verify fine-tuning finishes with unchanged Gaussian counts, 4,999 optimizer
   updates, identical pre/post camera splits, and finite per-view metrics.
5. Run the full protocol and real-subset scaling experiments before making any
   quality or speed claim. Capped blocks, unsupported GHAP budgets, timeouts
   and memory limits must remain visible in the reported results.
