# WikiArt color transfer: all 19 upstream pairs, K=50, L=100

Completed 2026-10-05 on CPU with
[`wikiart_pairs.json`](../experiments/color_transfer/configs/wikiart_pairs.json),
`--all-pairs`, GMM seed 0 and projection seed 20260909. All 30 unique images
from the pinned source manifest were prepared; the original WikiArt page for
`cheerful-forms-1914` returned 404 and was replaced by an explicitly recorded
Wikimedia Commons image of the **same painting**. The full ordered pair list,
page URLs and preprocessed image hashes are in
[`wikiart_full_data.json`](wikiart_full_data.json).

**Setting.** Each image was resized to longest side 256, and a 50-component
full-covariance RGB GMM fitted using up to 10,000 pixels. MW2 and all twelve
LSOT variants reused the GMMs for each ordered pair. LSOT used a bank of
L=100 for Mix, SMix, B and B1D, with avg, min and min-opt aggregation. A
fixed 256-pixel evaluation sample per pair measured exact empirical RGB W2
after clipping the transferred colors; 2,048 pixels and 128 directions
measured RGB SW2. There are **19 × 13 = 247** individual method rows.

| Method | Mean color W2 ↓ | Δ versus MW2 ↓ | Pairs beating MW2 | Mean Gaussian cost gap |
| --- | ---: | ---: | ---: | ---: |
| MW2 | **0.05908** | — | — | 0% |
| LSOT-Mix (avg) | 0.25529 | +0.19620 | 0/19 | +137.2% |
| LSOT-SMix (avg) | 0.23698 | +0.17790 | 0/19 | +122.5% |
| LSOT-B (avg) | 0.13605 | +0.07697 | 0/19 | +55.2% |
| LSOT-B1D (avg) | 0.14708 | +0.08800 | 0/19 | +62.8% |
| min-LSOT-Mix | 0.07808 | +0.01900 | 2/19 | +26.9% |
| min-LSOT-SMix | 0.07306 | +0.01398 | 2/19 | +18.2% |
| min-LSOT-B | 0.07368 | +0.01460 | 0/19 | +11.2% |
| min-LSOT-B1D | 0.07301 | +0.01393 | 1/19 | +12.2% |
| min-opt-LSOT-Mix | 0.07274 | +0.01366 | 1/19 | +12.8% |
| min-opt-LSOT-SMix | 0.07152 | +0.01243 | 0/19 | +11.5% |
| min-opt-LSOT-B | 0.07175 | +0.01267 | 0/19 | +9.7% |
| min-opt-LSOT-B1D | **0.07113** | **+0.01204** | **2/19** | **+10.1%** |

MW2 has the lowest mean color W2. The closest LSOT variant here,
min-opt-LSOT-B1D, is about 20% higher than MW2 on this metric. Only **four
of 19 pairs** have any tested LSOT variant with a lower raw color W2 than
MW2: pair 2 (Fuji → Pyramids), pair 5 (Haystacks → Argenteuil), pair 7
(Céret → Monet irises), and pair 14 (Red Vineyards → The Seine). The largest
observed LSOT improvement is pair 14 with min-opt-LSOT-B1D, 0.04518 versus
MW2's 0.04730. Other metrics need not agree: on pair 14 this variant's RGB
SW2 is 0.00841 versus MW2's 0.00708. Full per-pair values are provided
below so favorable examples are not isolated from the other 15 pairs.

The source-to-target identity color W2 ranges from 0.19912 to 0.52810;
MW2's transferred images range from 0.03807 to 0.09010. Visual comparison
of the generated images for pairs 2 and 14 shows that averaged LSOT tends
to flatten contrast, while min and min-opt outputs generally resemble the
MW2 recoloring more closely. Input shapes, SHA-256 hashes, fitted GMMs,
projection banks, maps, comparison images and evaluation samples are saved
locally by the command documented in the
[experiment guide](wikiart_color_transfer.md).

- [All 247 per-pair method scores](wikiart_full_pair_summary.csv)
- [Thirteen equal-weight method averages and win counts](wikiart_full_method_summary.csv)
- [Data provenance and image hashes](wikiart_full_data.json)
- [Preselected five-pair pilot](wikiart_pilot_results.md)

The cost gap compares Gaussian **component** plans against MW2's optimal
component plan. The color W2/SW2 columns compare output **pixels** against
target pixels. A lower component cost does not guarantee better-looking
images. These are exploratory scores from one fitted-GMM seed and one fixed
pixel-evaluation sample per pair; a few differences of roughly (10^{-4})
to (10^{-3}) could change with another seed. No claim of systematic LSOT
superiority follows from them. Generated painting files and result PNGs
remain outside git; the checked-in manifests and CSV tables reproduce the
setting and report the measured values.
