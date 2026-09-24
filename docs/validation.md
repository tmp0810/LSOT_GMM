# Implementation validation

Validated on CPU with float64. CUDA is supported by the implementation,
but no CUDA device was available for this validation.

## Checks performed

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
