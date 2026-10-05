# Mixing the cheapest lifted plans

Run from the repository root:

```bash
python -m experiments.color_transfer.run \
  --config experiments/color_transfer/configs/topk_trial.json
```

For each projection family and each data seed, the code samples one bank of
100 directions and constructs its 100 lifted plans. It ranks them by their
actual squared Gaussian component transport costs. `min` selects the cheapest
plan; `top2`, `top3`, and `top4` select the indicated number of cheapest plans
and take their equal-weight average before building the barycentric color map.
If costs tie, the bank order breaks the tie. GMMs, evaluation pixels, and
projection bank are shared across the four aggregations within a seed.

The reported `cost_squared` is the mean cost of the selected plans.
`color_sw2` and `color_w2` compare raw transferred RGB colors with the target;
the corresponding `guided_` columns evaluate images after guided filtering.
This trial uses 500 fixed evaluation projections for SW2 and 512 fixed sampled
pixels for exact empirical RGB W2. Transport runtime excludes color evaluation.
The recipe is a finite-bank experiment, not an optimization of the continuous
minimum LSOT objective.
