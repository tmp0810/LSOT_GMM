# WikiArt: does increasing K improve color SW2?

This sweep uses **K0=K1 in {5, 10, 20, 50, 100, 200}**, fixed solver **L=100**,
and exactly nine methods by default:

- MW2 (one solve per pair/K/data seed).
- LSOT-Mix, LSOT-SMix, LSOT-B, LSOT-B1D (average plans).
- min-LSOT-Mix, min-LSOT-SMix, min-LSOT-B, min-LSOT-B1D.

**No min-opt runs.** The command overrides the older WikiArt config's
aggregations with `avg min`. Other settings are inherited from
`experiments/color_transfer/configs/wikiart_pairs.json`: RGB max side 256,
up to 10,000 fit pixels, full covariance, GMM seed 0, projection seed
20260909, 2,048 evaluation pixels and 128 SW2 evaluation directions.
The latter is independent of the solver's L=100. Exact empirical color W2
is also retained (256 evaluation pixels); the primary plots use color SW2.

## Colab cells

Update the existing checkout; no deletion of old experiment results is needed:

```python
%cd /content/LSOT_GMM
!git pull --ff-only origin main
!python -m pip install -q -e .
```

Run the six previously requested comparable pairs:

```python
!python -m experiments.color_transfer.wikiart_sweep \
    --config experiments/color_transfer/configs/wikiart_pairs.json \
    --pair-indices 2 5 8 13 14 16 \
    --component-counts 5 10 20 50 100 200 \
    --aggregations avg min --projections 100 --device cpu \
    --image-dir /content/LSOT_GMM/data/color_transfer/wikiart \
    --output-dir /content/LSOT_GMM/results/color_transfer/wikiart_k_sweep
```

CPU is explicit here to avoid the unresolved CUDA error in the earlier run;
this feature does not change GPU kernels. On a working CUDA environment,
`--device cuda` uses the existing GPU path. EM fitting and the MW2 reference
LP remain CPU operations. Increasing K=100/200 costs more; timing is not
claimed to increase by only a small amount. There are 324 method outputs for
six pairs, six K and one seed. Rerun the same command after interruption:
completed **pair/K** settings with matching config and image hashes are
reused; an incomplete pair/K is recomputed. Summaries are saved after each
completed K; per-K summaries also survive interruption during that K.
The sweep uses its own output directory, not the previous fixed-K results.

Replace `--pair-indices ...` with `--all-pairs` for all 19 pairs. These six
pairs were selected after inspecting earlier K=50 results, so their sweep is
exploratory, not evidence of a universal LSOT advantage. For robustness use
`--seeds 0 1 2` (three times the method runs). All methods share fitted GMMs
within each pair/K/seed. Identical random seeds preserve fit-pixel samples,
projection banks and evaluation banks across K; Gaussian fits themselves
are refitted and are not nested as K changes.

## Plotting (also runs automatically after the sweep)

Regenerate plots without refitting any GMM:

```python
!python -m experiments.color_transfer.plot_wikiart_sweep \
    --input results/color_transfer/wikiart_k_sweep/pair_k_summary.csv \
    --metric color_sw2
```

For outputs after guided filtering, run the same command with
`--metric guided_color_sw2`. Raw and guided results are never mixed. The plot
CLI also accepts `color_w2` or `guided_color_w2`. Missing K in interrupted
results break lines rather than fabricating measurements.

Display the six-pair grid in Colab:

```python
from IPython.display import Image, display
root = "/content/LSOT_GMM/results/color_transfer/wikiart_k_sweep"
display(Image(filename=f"{root}/plots/color_sw2/all_pairs.png"))
# One pair: all nine methods
display(Image(filename=f"{root}/plots/color_sw2/pair_02.png"))
# One method: one curve for each of the selected pairs
display(Image(filename=f"{root}/plots/color_sw2/method_min-LSOT-B1D_L100.png"))
```

The x axis shows the exact six K values on a log scale; y is **root empirical
SW2 on clipped float RGB**, before 8-bit PNG rounding, against the target
colors. Lower is better color-distribution matching; it does not guarantee
better visual appearance or preservation of structure. MW2 is a black curve,
not a constant baseline: its fitted mixtures also change with K. Minimum
variants use dashed lines, average variants solid lines. With multiple runs,
shading is mean ± one descriptive SD, not a confidence interval.

Increasing K makes the GMM richer but does **not** guarantee decreasing
pixel SW2: EM initialization, fitted covariances, component plans and the
posterior-weighted affine map all change. Compare the measured curves instead
of assuming monotonic improvement. The sampled evaluation is fixed across
methods/K, but still an estimate of the full pixel-distribution distance.

## Outputs

| Path under `wikiart_k_sweep/` | Meaning |
| --- | --- |
| `metrics.csv` | Individual runs, with pair, K, method, data/projection seeds and all quality/runtime metrics. |
| `pair_k_summary.csv` | Mean metrics per pair/K/method/L, descriptive run SD, and differences against same-K/seed MW2. |
| `method_k_summary.csv` | Equal-pair-weight mean per K/method; do not confuse it with a single pair. |
| `sweep_config.json` | Requested K, selected pairs, settings and completed K. |
| `K_50/pair_.../seed_0/` | Actual transferred PNGs, comparison grids, GMMs, evaluation banks and plans for that K/pair. |
| `plots/color_sw2/pair_02.png` | Nine curves for pair 02; analogous file per pair. |
| `plots/color_sw2/all_pairs.png` | Grid of all selected pairs with consistent method colors. |
| `plots/color_sw2/method_LSOT-Mix_L100.png` | One method across selected pairs; analogous file per method. |
| `plots/color_sw2/trends.csv` | Endpoint change, adjacent decrease/increase counts and best observed K for every pair/method. |

In `trends.csv`, `last_minus_first < 0` or `endpoint_reduction_pct > 0` means
an improvement between the smallest and largest **observed** K. It does not
imply monotonic improvement: check `nonincreasing_on_observed_K` and adjacent
increase counts. If only one K is available, endpoint-change fields are empty.
`best_observed_K` is descriptive, not held-out tuning. Read these tables in
Colab with:

```python
import pandas as pd
summary = pd.read_csv(f"{root}/pair_k_summary.csv")
display(summary.query("pair_index == 2")[[
    "K", "method", "color_sw2", "color_sw2_std", "delta_color_sw2_to_mw2"
]].sort_values(["method", "K"]))
display(pd.read_csv(f"{root}/plots/color_sw2/trends.csv"))
```
