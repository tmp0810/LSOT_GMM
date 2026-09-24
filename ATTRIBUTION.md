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

Image assets are downloaded on demand from the upstream repository and
verified against the Git blob hashes reviewed for this implementation.
No new license is asserted for upstream or previously supplied material.
