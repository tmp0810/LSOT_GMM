# Validation

Validated on CPU, float64, with NumPy 2.3.5, PyTorch 2.14.1+cpu,
POT 0.9.7.post1, and one computation thread.

```
python -m pytest -q experiments/barycenter/test_barycenter.py \
  tests/test_transport.py tests/test_distribution_projections.py \
  tests/test_linalg_batching.py
```

Result: **65 passed, 9 skipped**. Skips require a CUDA device.

Checks include:

- MW2 tuple costs, Gaussian centers/covariances, and LP objective agree with
  the original `create_cost_matrix_from_gmm` and `solveMMOT` at N=10.
- All eight sampled variational objectives agree with the existing avg/min
  LSOT solvers. Gradients for mixture weights, means, and covariance
  parameters agree with centered finite differences away from ordering and
  cumulative-mass boundaries; maximum observed absolute error was about
  `1.2e-11` in the sampled checks.
- The min objective chooses slices separately for each input pair.
- For single-Gaussian inputs, all eight L-BFGS variants agree with a
  100-iteration Gaussian barycenter reference to `1e-8` in objective value.
- Exact corner behavior, bilinear weights, image vertical flip/nonzero-blue
  preprocessing, component budgets, and positive-definite covariances.

The full pipeline smoke configuration ran all three cases and all nine
methods: **171 saved method/node results**, with no numerical stops. All
individual panels, method grids, PDFs, and perimeter GIFs were produced.
Rerunning the same command reused completed solves. The wheel contains the
bundled masks and both configs; mask recovery reproduces the bundled data.

Additional equal-weight center checks used the full default **L=100,
80-step budget**, original synthetic GMMs and K=12 image GMMs. All 16 LSOT
case/method combinations returned finite candidates with objective no larger
than initialization. They stopped early on the local `stalled` criterion;
this does not establish global optimality. The complete 7x7, 891-result run
was not performed during this validation.

Original PNGs are absent from the MW2 repository. Image tests use the
documented masks recovered from the saved notebook display. Redcross and
duck foreground masks match POT's corresponding 128x128 data exactly;
raw pixel identity for star and batman remains unverified.
