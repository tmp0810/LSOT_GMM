# Fast barycenter validation

CPU, float64, PyTorch 2.15.0+cpu, NumPy 2.3.5, POT 0.9.7.post1; one thread.

## Numerical tests

`100 passed, 13 skipped` on the documented targeted suite. All skips require CUDA.
Dense and sparse paths match the previous costs and gradients for all eight
projection/aggregation combinations. Tests include proportional fibers, exact
minimum ties, learned-mass dual gradients, checkpoint selection and reproducibility.

## Functional runs

- All-case 3x3 smoke: 76 method/node results, including MW2, MW2-variational, min-B1D and avg-B1D.
- Synthetic 7x7 smoke: 98 MW2/min-B1D results and complete grids/PDFs/GIFs. One Adam update; not a quality experiment.
- Completed center solve resumes without rerunning optimization.
- Scaling smoke verifies both a completed reference and explicit tuple-budget skipping.
- Synthetic center ablation ran all four min projections through 200 updates at L=100, with no early stopping.
- The final B1D diagnostic below uses the release solver, one seed, 200 updates, validation L=512 and independent test L=2048.

## Single-seed CPU diagnostic

| Method | Total ms | Common MW2 objective gap (%) |
| --- | ---: | ---: |
| MW2 | 11.61 | 0.00 |
| reference-fixed/min-LSOT-B1D | 1216.99 | 22.92 |
| fast-historical/min-LSOT-B1D | 893.17 | 23.72 |
| fast-fixed/min-LSOT-B1D | 892.04 | 20.67 |
| fast-fresh/min-LSOT-B1D | 991.97 | 11.29 |
| fast-partial/min-LSOT-B1D | 909.79 | 20.67 |
| MW2-variational | 686.29 | 18.58 |

These are local experimental results, not global convergence guarantees.
The MW2 multi-marginal baseline remains faster and has the lowest common
MW2 objective on this small problem. Fresh slices improve the B1D endpoint
in this run, but that does not establish a universal advantage.

## Pairwise timing diagnostic

| K | Method | Total ms | Cost / MW2 cost |
| ---: | --- | ---: | ---: |
| 10 | MW2 | 0.357 | 1.000 |
| 10 | min-LSOT-B1D | 0.859 | 1.172 |
| 100 | MW2 | 1.949 | 1.000 |
| 100 | min-LSOT-B1D | 4.816 | 5.515 |
| 500 | MW2 | 51.284 | 1.000 |
| 500 | min-LSOT-B1D | 26.007 | 17.838 |

Five repeated timings, one random-data seed, L=100. At larger K the total
min-B1D computation can be cheaper, but the large cost ratio shows that these
random-data measurements do not demonstrate equal-quality acceleration.
Cached-ground-cost timings and all raw values are retained in
[fast_solver_checks.json](validation/fast_solver_checks.json).

The complete 200-step, all-method, 7x7 image/synthetic study and GPU timing
were not run here. Use the supplied Colab notebook and benchmark commands.
