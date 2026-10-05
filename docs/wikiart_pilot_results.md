# WikiArt pilot: five preregistered pairs, K=50, L=100

The [experiment guide](wikiart_color_transfer.md) describes the data preparation,
methods, commands and outputs. This report records one completed CPU run on
2026-10-05 with `wikiart_pairs.json`, five fixed pairs (0, 2, 3, 5, 14), GMM
seed 0 and projection seed 20260909. Images were fetched from the pinned
Sliced-Amortized-OT WikiArt URLs; the exact local RGB PNG hashes and sizes
are in [wikiart_pilot_data.json](wikiart_pilot_data.json). The original source
code for every result is in `experiments/color_transfer/`.

Each input was resized to longest side 256, a full-covariance 50-component RGB
GMM was fitted with up to 10,000 sampled pixels, and all methods reused the
same two GMMs. For evaluating *output colors*, exact empirical W2 used a fixed
256-pixel source/target subsample per pair. These are fixed-sample descriptive
results from five pairs and one seed, not a significance test.

| Method | Mean color W2 ↓ | Difference from MW2 ↓ | Pairs with lower W2 | Mean Gaussian cost gap ↑ |
| --- | ---: | ---: | ---: | ---: |
| MW2 | 0.05419 | 0 | — | 0% |
| avg-LSOT-Mix | 0.24571 | +0.19152 | 0/5 | +105.48% |
| avg-LSOT-SMix | 0.22756 | +0.17337 | 0/5 | +92.19% |
| avg-LSOT-B | 0.11540 | +0.06121 | 0/5 | +41.21% |
| avg-LSOT-B1D | 0.12827 | +0.07408 | 0/5 | +47.39% |
| min-LSOT-Mix | 0.06197 | +0.00778 | 2/5 | +18.31% |
| min-LSOT-SMix | 0.06059 | +0.00640 | 1/5 | +12.85% |
| min-LSOT-B | 0.05859 | +0.00440 | 0/5 | +6.44% |
| min-LSOT-B1D | 0.05903 | +0.00484 | 1/5 | +7.42% |
| min-opt-LSOT-Mix | 0.05963 | +0.00544 | 1/5 | +9.37% |
| min-opt-LSOT-SMix | 0.06032 | +0.00613 | 0/5 | +7.66% |
| min-opt-LSOT-B | 0.05854 | +0.00435 | 0/5 | +5.74% |
| min-opt-LSOT-B1D | **0.05624** | **+0.00205** | **2/5** | **+5.99%** |

The five pair averages still favor MW2 on raw color W2. The nearest LSOT
variant here is min-opt-LSOT-B1D. Its W2 is lower than MW2 on pair 2
(0.05956 versus 0.06019) and pair 14 (0.04518 versus 0.04730), but higher
on the other three. On pair 14, its sliced W2 is *higher* than MW2
(0.00841 versus 0.00708). The mean guided-filter W2 also needs to be compared
only to the guided MW2 score; filtering sometimes increases color W2 for both.
The output images should be inspected together with these measures.

The unmodified source images have empirical color W2 between 0.30484 and
0.50964 on these five pairs; MW2 reduces it to 0.04730–0.06019. Averaged
LSOT often makes colors too flat in the comparison images. Minimum and
optimized minimum variants retain substantially more contrast and are closer
to the MW2 images, but small numerical wins in a 256-pixel W2 sample are not
evidence of a general visual improvement. More image pairs and repeated GMM
and evaluation seeds would be needed for a stronger claim.

Full per-pair scores and all 13 method averages are saved as
[wikiart_pilot_pair_summary.csv](wikiart_pilot_pair_summary.csv) and
[wikiart_pilot_method_summary.csv](wikiart_pilot_method_summary.csv). The
per-pair run command saves complete per-method images, sparse plans, metadata,
raw metrics, guided comparisons and evaluation banks under
`results/color_transfer/wikiart/` (ignored by git because generated images
and third-party painting data are downloaded on demand).
