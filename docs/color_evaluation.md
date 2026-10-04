# Color transfer evaluation

The experiment computes two distances between the **transferred source RGB
pixels** and the original target RGB pixels. Both images are clipped to
`[0, 1]` after applying the learned color map, before PNG quantization.
They are pixel-level evaluation measures; neither is the Gaussian-mixture
component cost reported in `cost_squared`.

* `color_sw2` is the root sliced Wasserstein-2 distance on the common sampled
  pixels and `eval_projections` fixed evaluation directions.
* `color_w2` is the exact, unregularized empirical Wasserstein-2 distance
  using squared Euclidean RGB costs and POT `ot.emd2`, followed by a square
  root. It uses `min(eval_w2_samples, eval_samples, number of pixels in each
  image)` pixels, taken as a common prefix of the saved sample indices. The
  default `eval_w2_samples` is 512; exact OT on all 4096 SW2 evaluation
  pixels would be considerably more expensive.
* `guided_color_sw2` and `guided_color_w2` use the optional guided-filter
  output. When `guided_filter=false`, those columns are empty.

All methods at the same data seed use the same sampled pixel indices. The
indices are stored in `seed_<seed>/evaluation_bank.npz`, including the W2
subsample. `eval_seed`, `eval_samples`, `eval_w2_samples`, and
`eval_projections` are recorded in `config.json` under the output directory.
Pixel evaluation takes place outside the reported transport and pipeline
timers.

For a component sweep, `metrics.csv`, `paper_results.tsv`, `comparison.tsv`,
`color_w2.tsv`, and `guided_color_w2.tsv` include the new metric. To use a
different exact OT sample size on Kaggle, set `"eval_w2_samples": 512` in
the JSON config or add `--eval-w2-samples 512` to the run/sweep command.
When comparing color W2 across methods and component counts, keep the
evaluation images and pixel sample size fixed.
