# Implementation validation

Validated on CPU with float64. CUDA is supported by the implementation,
but no CUDA device was available for this validation.

## Component-count sweep (2026-09-30)

- Full test suite: **46 passed, 9 skipped** (CUDA unavailable).
- Offline sweep at two K values, two data seeds and all eight LSOT variants:
  verifies grouped means, output table columns and exact equality of saved
  projection/evaluation banks across K. Multiple projection budgets are kept
  in separate columns; neither K nor L is pooled during aggregation.
- CLI smoke sweep completed K=10,20,50,100,200, MW2 plus min-Mix/SMix/B/B1D,
  L=8, CPU float64, seed 0. For this integration check only, the reference
  images were resized to maximum side 48 with a fitting budget of 1,500 pixels.
  All 25 configurations completed and the per-K output directories and root
  comparison tables were generated. This is not the full-resolution GPU
  benchmark; its timing and quality values should not be used as paper results.
- The existing reference JSON and its user-selected L/aggregation/seed values
  are preserved. The sweep command inherits these, overriding only K0=K1 and
  per-K output directories unless additional CLI overrides are supplied.

## Large CUDA eigensolver batch fix (2026-09-30)

A Colab run with K=100 and L=500 reported CUSOLVER_STATUS_INVALID_VALUE
inside the B projection's eigvalsh workspace query. The previous 65,536-pair
budget submitted all 50,000 matrices at once. Large small-matrix batches are
a known backend compatibility issue (see PyTorch issue #166004); the exact
threshold depends on the installed CUDA/PyTorch stack.

B projection, Gaussian pair costs and barycentric map setup now cap matrix
batches at 4,096, including when a larger budget is explicitly requested.
This changes partitioning only, not projections, candidate plans, precision
or the mathematical formulas. Nonfinite projection inputs raise an explicit
ValueError before entering the eigensolver.

Validation: **40 passed, 9 skipped**. A K=100,L=500 regression intercepts
every eigvalsh call, rejects oversized batches, and verifies the 13 resulting
batches against an independently partitioned result. A 10,000-pair test checks
the cost and map paths against closed-form equal-covariance results. Existing
formula and offline color-transfer tests pass. The corresponding CUDA
regression is included but skipped here: the original Colab failure cannot
be reproduced on this CPU-only host, so the GPU fix still needs a Colab rerun.

## B/B1D integration (2026-09-30)

- Editable installation with `python -m pip install -e '.[test]'`.
- `python -m pytest -q`: **37 passed, 8 skipped** (all skips require CUDA).
- B matches the general endpoint Busemann formula computed independently
  with SciPy matrix square roots, including noncommuting covariances.
  Tests check zero at the base, value `-t` along its unit-speed ray, and
  equivalence of chunked/un-chunked evaluation.
- B1D matches the mean/standard-deviation formula with full covariance.
  An explicit off-diagonal regression distinguishes `theta.T Sigma theta`
  from the upstream diagonal-only contraction.
- Both new families pass weighted, unequal-component-count, exact-tie,
  self-transport, independent dense lift/cost, minimum-selection and nested
  budget checks. Single-Gaussian/single-projection and float32 checks pass.
- Offline image test completes MW2 and all eight LSOT variants. Every LSOT
  plan is reconstructed from its saved bank; Mix/SMix seeded tensors are
  unchanged when B/B1D are enabled.
- `python -m experiments.color_transfer.run --config experiments/color_transfer/configs/smoke.json --device cpu`:
  all **17** method/budget configurations complete on Renoir -> Gauguin,
  K0=K1=10, L=4,8, maximum image side 96, at most 4,000 fitting pixels.
  Maximum marginal L1 error: **4.38e-16**. Selection archives agree with
  the argmin, and their mean candidate costs agree with averaged costs.
- Notebook code cells parse successfully. Full-resolution B/B1D color
  transfer was not benchmarked in this validation; the smoke run verifies
  integration, not quality or runtime superiority.

This check used CPU float64 with Python 3.12, PyTorch 2.14.0+cpu, POT
0.9.7.post1, SciPy 1.17.0, and scikit-learn 1.8.0. CUDA tests cover all four
families but were not executed on this machine. No additional runtime
dependencies were introduced by B/B1D.

## Previous Mix/SMix validation

The results below were recorded before adding B/B1D; they are retained as
the previous implementation's reference, not a new run of all four families.

- Editable installation with `python -m pip install -e '.[test]'`.
- `python -m pytest -q`: **20 passed, 4 skipped**. All skipped tests require CUDA.
- Offline integration test: completed for MW2 and both avg/min projection families.
- Notebook code cells parse successfully.
- Full `mw2_reference.json` run: completed on the two original 1024 x 768
  images, K0=K1=10, all pixels in EM, seed 0, L=10,50,100,
  including both avg/min variants (13 configurations). Both EM fits converged.
- Maximum marginal L1 error across that full run: below 4e-16.
- The original `gmmot.py` is unchanged from repository commit
  `d6e728a7f4c1aa5ae5c4d8b3f06c68a6fdb7c74e`.

Tests independently check the scalar projection formulas, exact-fiber lifting,
weighted marginals, projected optimality, Gaussian costs/maps, average-cost
identity, permuted self-transport, and the original posterior-weighted Tmean
formula. Minimum selection is checked against exhaustive dense fiber lifts
and independent SciPy costs, including collisions and exact cost ties. A
constructed example ensures selection uses Gaussian ground costs rather than
projected costs. Tests verify MW2 <= min <= avg for squared costs and
nonincreasing minimum cost along prefixes of a fixed bank. An offline integration test includes different source/target sizes.

## Full-image check: quality values

These are one reproducibility run, not aggregate paper results. `color_sw2`
uses the separate RGB evaluation bank and the unfiltered, clipped float output.

| Method | L | Gaussian cost squared | Relative cost gap | Color SW2 |
| --- | ---: | ---: | ---: | ---: |
| MW2 | — | 0.058179 | 0.000000 | 0.030790 |
| LSOT-Mix | 10 | 0.195172 | 2.354681 | 0.151192 |
| LSOT-SMix | 10 | 0.134068 | 1.304404 | 0.106883 |
| min-LSOT-Mix | 10 | 0.143435 | 1.465409 | 0.102259 |
| min-LSOT-SMix | 10 | 0.094664 | 0.627121 | 0.056332 |
| LSOT-Mix | 50 | 0.197729 | 2.398640 | 0.157386 |
| LSOT-SMix | 50 | 0.155755 | 1.677173 | 0.125206 |
| min-LSOT-Mix | 50 | 0.104501 | 0.796192 | 0.044838 |
| min-LSOT-SMix | 50 | 0.066457 | 0.142287 | 0.043035 |
| LSOT-Mix | 100 | 0.194122 | 2.336642 | 0.153663 |
| LSOT-SMix | 100 | 0.155740 | 1.676923 | 0.126832 |
| min-LSOT-Mix | 100 | 0.075036 | 0.289743 | 0.047265 |
| min-LSOT-SMix | 100 | 0.066457 | 0.142287 | 0.043035 |

The current averaged parameter-projection methods have worse color-distribution
scores than MW2 in this run. This is recorded as an experimental result;
projection scales, the reference images and seeds were not tuned to change it.
Minimum LSOT improves both the Gaussian cost and color SW2 over the average
in each tested configuration, while MW2 still has the lowest values. The
full-run selection archives were checked against the argmin, the mean
candidate cost against the averaged cost, and the minimum cost across nested
budgets. At L=100, the selected zero-based indices are 89 (Mix) and 12 (SMix).
The MW2 and averaged quality values match the previous implementation within
1e-13. This is one image pair/seed, not evidence of general superiority.

No projection rescaling or changes to the Mix/SMix sampling law were introduced.
Minimum transport timing includes the full finite-bank search. CUDA behavior
has corresponding tests but remains unverified on this CPU-only machine.

Rerunning the provided configurations writes all images, saved GMM parameters,
projection banks, sparse plans, detailed timings and CSV/TSV results locally.
Package versions and exact image hashes are saved in each run's metadata.json.
