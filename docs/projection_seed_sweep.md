# Projection seed sweeps

The existing JSON field accepts either an integer (backward compatible)
or a list. For example, change `"projection_seed": 42` to:

```json
"projection_seed": [0, 1, 2, 3, 4]
```

Alternatively, override it on the command line (both `run` and `sweep` support this):

```bash
python -m experiments.color_transfer.sweep --config experiments/color_transfer/configs/mw2_reference.json --component-counts 10 20 50 100 200 --projection-seeds 0 1 2 3 4 --output-dir results/color_transfer/k_projection_sweep
```

For each K and data seed, fit the GMM pair once, create the evaluation bank
once, and benchmark MW2 once. Only LSOT is rerun for each projection seed.
Within a bank, smaller L values use prefixes; Mix/SMix share the parameter
bank. Keep the maximum L fixed when comparing independent runs. To preserve
legacy results, the effective sampling seed is `projection_seed + seed`,
recorded as `effective_projection_seed` in `metrics.csv`.

For a list, LSOT images, plans and banks are stored under
`K_<K>/seed_<data>/projection_seed_<projection>/`; shared GMMs, evaluation
bank and MW2 outputs stay in `K_<K>/seed_<data>/`. Scalar configs keep their
original folder layout. `projection_seed_results.tsv` separates projection
seeds and lists MW2 once with an empty projection seed. The existing compact
and wide tables pool all LSOT runs; no best-seed selection is performed.

## Summary statistics

`paper_results.tsv` reports pooled LSOT means and descriptive sample standard
deviations across (data seed, projection seed) runs. `n_seeds` counts data
seeds, `n_projection_seeds` counts bank seeds, and `n_runs` counts runs.
Runs sharing a GMM are not independent data replicates. MW2 is never
duplicated for projection seeds. Use `projection_seed_results.tsv` to
inspect each projection seed separately; its statistics are across data seeds.
