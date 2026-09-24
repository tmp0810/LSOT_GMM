# Implementation validation

Validated on CPU with float64. CUDA is supported by the implementation,
but no CUDA device was available for this validation.

## Checks performed

- Editable installation with `python -m pip install -e '.[test]'`.
- `python -m pytest -q`: **11 passed, 2 skipped**. Both skipped tests require CUDA.
- Real-image smoke configuration: completed for MW2 and both LSOT variants.
- Full `mw2_reference.json` run: completed on the two original 1024 x 768
  images, K0=K1=10, all pixels in EM, seed 0, L=10,50,100. Both EM fits converged.
- Maximum marginal L1 error across that full run: below 4e-16.
- The original `gmmot.py` is unchanged from repository commit
  `d6e728a7f4c1aa5ae5c4d8b3f06c68a6fdb7c74e`.

Tests independently check the scalar projection formulas, exact-fiber lifting,
weighted marginals, projected optimality, Gaussian costs/maps, average-cost
identity, permuted self-transport, and the original posterior-weighted Tmean
formula. An offline integration test includes different source/target sizes.

## Full-image check: quality values

These are one reproducibility run, not aggregate paper results. `color_sw2`
uses the separate RGB evaluation bank and the unfiltered, clipped float output.

| Method | L | Gaussian cost squared | Relative cost gap | Color SW2 |
| --- | ---: | ---: | ---: | ---: |
| MW2 | — | 0.058179 | 0 | 0.030790 |
| LSOT-Mix | 10 | 0.195172 | 2.354681 | 0.151192 |
| LSOT-SMix | 10 | 0.134068 | 1.304404 | 0.106883 |
| LSOT-Mix | 50 | 0.197729 | 2.398640 | 0.157386 |
| LSOT-SMix | 50 | 0.155755 | 1.677173 | 0.125206 |
| LSOT-Mix | 100 | 0.194122 | 2.336642 | 0.153663 |
| LSOT-SMix | 100 | 0.155740 | 1.676923 | 0.126832 |

The current averaged parameter-projection methods have worse color-distribution
scores than MW2 in this run. This is recorded as an experimental result;
projection scales, the reference images and seeds were not tuned to change it.
No claim of superior application quality follows from the correctness tests.

Rerunning the provided configurations writes all images, saved GMM parameters,
projection banks, sparse plans, detailed timings and CSV/TSV results locally.
Package versions and exact image hashes are saved in each run's metadata.json.
