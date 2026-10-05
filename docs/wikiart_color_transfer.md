# WikiArt color transfer: multiple fixed painting pairs

The source [Sliced-Amortized-OT repository](https://github.com/tmp0810/Sliced-Amortized-OT/tree/main/data_color_transfer)
contains `paintings/pairs.txt` and `wikiart-urls.txt`, **not the image files**.
This experiment pins both lists at upstream commit
`59234586542d905d951eae1af7b03566733e528f` and fetches paintings on
demand. The listed WikiArt page for `cheerful-forms-1914` currently returns
404. That one slug has an explicit fallback to the [same Franz Marc painting
on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Marc_-_Cheerful_Forms,_1914,_Hoberg,_Jansen_237.jpg).
`dataset.json` records the actual download page, image URL and hashes. All 30
unique paintings referenced by the 19 pairs were downloaded successfully in
the data-preparation check. No painting pixels are committed here.

The default, predetermined pilot runs pair indices **0, 2, 3, 5, 14** in the
original order. These include different styles and a same-artist pair; they
were chosen from the manifest before evaluating any result. `--all-pairs`
includes all **19** original pairs. Every completed pair is reported, even if
its LSOT output is worse than MW2. The manifest is ordered: swapping source
and target defines a different experiment.

## Colab

```bash
%cd /content/LSOT_GMM
!python -m pip install -e .
!python -m experiments.color_transfer.wikiart --prepare-only
!python -m experiments.color_transfer.wikiart --output-dir results/color_transfer/wikiart
```

If Colab reports `CUDA error: no kernel image is available for execution on
the device`, its installed PyTorch wheel does not contain kernels for that
GPU. Add `--device cpu` to the experiment command. Completed pairs are reused
and an incomplete pair is recomputed automatically.

The first preparation command downloads the selected paintings once into
`data/color_transfer/wikiart/`; the experiment command checks and reuses that
cache. The default config is
`experiments/color_transfer/configs/wikiart_pairs.json`: **K0=K1=50,
L=100**, all four projection families Mix/SMix/B/B1D, their avg/min/min-opt
variants and the MW2 reference. It uses data seed 0, projection seed 20260909,
RGB images with longest side 256, up to 10,000 sampled pixels for full-covariance
GMM fitting, and fixed 2,048-pixel SW2 / 256-pixel exact W2 evaluation
subsamples. The same fitted GMMs, pixel samples and evaluation directions are
shared by every method on a pair. The optional guided filter is applied after
each color map; raw and guided outputs are evaluated separately.

For a cheap check that the data downloader and end-to-end solver work:

```bash
!python -m experiments.color_transfer.wikiart \
    --config experiments/color_transfer/configs/wikiart_quick.json \
    --pair-indices 0 2 --output-dir results/color_transfer/wikiart_quick
```

Quick mode uses K=10 and L=8 and is **not comparable** to the K=50, L=100
experiment. For all manifest pairs, use `--all-pairs` with the main config.
To run a chosen subset, use `--pair-indices 1 4 7`; indices are zero based.
These runs may take substantially longer than the quick check.

If WikiArt rejects automated fetching, place an image named `<slug>.jpg`,
`<slug>.jpeg` or `<slug>.png` for each selected manifest slug under
`data/color_transfer/wikiart/`, then pass `--no-download`. The error identifies
the missing slug and its WikiArt page. `--force-download` refreshes an existing
file. `--prepare-only --all-pairs` checks all images without importing Torch.

## Files and interpretation

`results/color_transfer/wikiart/` contains:

| File | Meaning |
| --- | --- |
| `dataset.json` | Ordered selected pairs, source page URLs, local file paths, dimensions and SHA-256 hashes. |
| `metrics.csv` | All individual solver runs, with pair index, GMM seed and projection seed. |
| `pair_summary.csv` | Each method's mean for each pair; paired deltas against MW2 from the **same pair and GMM seed**. |
| `method_summary.csv` | Equal-weight averages across completed pairs, median deltas and counts of pairs with improved scores. |
| `pair_<index>_<source>__<target>/seed_0/comparison.png` | Source and target next to MW2 and every LSOT image for that pair. |
| `pair_.../seed_0/comparison_guided.png` | Corresponding images after optional guided filtering. |
| `pair_.../seed_0/gmms.npz`, `evaluation_bank.npz`, `*_plan.npz` | Shared fitted GMMs, evaluation pixels/directions and sparse component plans. |

`color_w2` and `color_sw2` are distances between the **transferred source RGB
pixels and the target RGB pixels**; lower is better for matching the target
color distribution. `identity_color_w2` and `identity_color_sw2` evaluate the
unmodified source with the same sampled pixels. `delta_color_w2_to_mw2` is
`LSOT color_w2 - MW2 color_w2`; a **negative** value favors LSOT.
`wins_delta_color_w2_to_mw2` counts the pairs with a strictly negative
per-pair mean delta. Guided metrics should be compared with guided metrics.

`cost_squared` and `relative_cost_gap` measure the transport cost between
fitted **Gaussian components**, not quality of the transferred pixels. A lower
Gaussian cost does not guarantee a lower color W2, so report both the paired
quality scores and the comparison images. Equal-weight pair averages are
descriptive; trying all 13 methods on a small pilot is exploratory, and the
best-looking pair or method should not be reported as a universal improvement.

After each completed pair the root CSV files are updated. An interrupted
command can be rerun with the same arguments: completed pairs with matching
image hashes and config are reused. `--no-resume` recomputes them. Each pair's
`config.json` and `metadata.json` record the exact preprocessing and input
hashes used by the original single-pair experiment.

An initial five-pair K=50, L=100 CPU run is documented in
[the pilot report](wikiart_pilot_results.md), with the numerical CSVs committed
beside it. Re-running in Colab produces its own image outputs and summaries.
The [full 19-pair run](wikiart_full_results.md) and its complete per-pair CSV
are also committed.

To test whether color SW2 improves as K increases, see
[the WikiArt K sweep](wikiart_k_sweep.md): K=5/10/20/50/100/200, L=100,
average/minimum LSOT across four projections and MW2, resumable pair/K runs
and per-pair/per-method plots. No min-opt runs are enabled in that sweep.
