# Provenance

- `gmmot.py` was supplied in this repository by its owner and is preserved
  unchanged. It credits Julie Delon. Its MW2 solver, Gaussian geometry and
  optional guided filter come from the
  [Delon--Desolneux GMM-OT implementation](https://github.com/judelo/gmmot).
- Reference application setting: [GMM_OT_color_transfer.ipynb](https://github.com/judelo/gmmot/blob/master/python/GMM_OT_color_transfer.ipynb),
  also supplied by the user as a Python export. The new notebook uses the same
  Renoir/Gauguin images, RGB scaling, full-covariance EM and posterior-weighted
  mean map; the experiment explicitly records its reproducibility additions.
- `param_proj/sot_gms.py` and `param_proj/sw.py` were supplied in this repository
  by its owner. The former is refactored to reuse named projection banks;
  the scalar formulas and the spectral Mix sampling law are retained.
- Reference paper: Julie Delon and Agnès Desolneux, *A Wasserstein-type distance
  in the space of Gaussian Mixture Models*, SIAM Journal on Imaging Sciences
  13(2), 936–970, 2020. https://arxiv.org/abs/1907.05254
- `distribution_proj/` is a Torch implementation of the Gaussian Busemann
  formulas and ray laws from Clément Bonet, Elsa Cazelles, Lucas Drumetz and
  Nicolas Courty, *Busemann Functions in the Wasserstein Space: Existence,
  Closed-Forms, and Applications to Slicing*, AISTATS 2026.
  https://arxiv.org/abs/2510.04579 (Eq. (18)-(19), Appendix B.2).
  We reviewed their [reference repository](https://github.com/clbonet/Busemann_Functions_in_the_Wasserstein_Space)
  at commit `5bb8a254f9c340a7a37af14218b7d6b06130e3d0`, specifically
  `lib_torch/sliced_busemann_gaussian.py` (`busemannGaussians` and
  `busemann_sliced_gaussian`), `lib_torch/utils_bw.py` (`exp_bw`), and
  `xp_gaussian_mixtures/sliced_busemann_gaussian1d.py`.
  Our implementation exposes scalar locations and reusable seeded banks
  instead of their sliced-distance functions, using the default base `eps=1`.
  B uses the equivalent `tr(sqrt(S Sigma S))` expression with bounded matrix
  batches. B1D corrects the upstream repeated-index einsum to include every
  covariance entry in `theta.T Sigma theta`; it retains the standard-deviation
  coordinate and the uniform `[-1,1]` mean-speed sampling. We did not copy the
  experiment notebooks or their unrelated torchdr/OTDD dependencies.
  The reviewed upstream tree has no top-level license file; no license for
  upstream material is inferred or added here.

Image assets are downloaded on demand from the upstream repository and
verified against the Git blob hashes reviewed for this implementation.
No new license is asserted for upstream or previously supplied material.

The optional WikiArt color-transfer experiment uses the painting URL list and
19 ordered pairs from the user's
[Sliced-Amortized-OT repository](https://github.com/tmp0810/Sliced-Amortized-OT/tree/main/data_color_transfer)
at revision `59234586542d905d951eae1af7b03566733e528f`. Those two text
manifests are copied under `experiments/color_transfer/wikiart/`; the paintings
themselves are downloaded on demand from their listed WikiArt pages, kept in
the ignored `data/` directory and retain their original page URLs and hashes
in the local result manifest. The paintings are not bundled with this repo.
One upstream WikiArt page (`cheerful-forms-1914`) now returns 404, so the
downloader falls back explicitly to a
[Wikimedia Commons file of the same Franz Marc artwork](https://commons.wikimedia.org/wiki/File:Marc_-_Cheerful_Forms,_1914,_Hoberg,_Jansen_237.jpg).
The source URL and converted image hash are recorded for that fallback.
