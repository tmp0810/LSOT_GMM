# Intro GMM examples: MW2 and LSOT

This experiment reuses the 1D and 2D Gaussian mixtures in Julie Delon's
[`GMM_OT_introduction.ipynb`](https://github.com/judelo/gmmot/blob/master/python/GMM_OT_introduction.ipynb).
The notebook calls the mixture distance `GW2`; this repository calls the same
discrete component-OT quantity `MW2`. `cost_squared` is the squared distance.

Run from the repository root after `python -m pip install -e .`:

```bash
python -m experiments.intro.run --output-dir results/intro
```

The default is CPU float64, one seed, `L=100` directions per family, and
20 optimization updates with 8 perturbations per update. Use `--L 8
--opt-steps 2 --opt-samples 2 --grid-2d 20 --skip-grid-w2` for a quick check.
`--help` lists the other controls. Matplotlib uses a noninteractive backend;
no Jupyter widgets are needed.

## Fixed source and target GMMs

| Case | Source: weights; means; covariances | Target: weights; means; covariances |
| --- | --- | --- |
| 1D | `(0.3, 0.7)`; `(0.2, 0.4)`; `(0.0009, 0.0016)` | `(0.6, 0.4)`; `(0.6, 0.8)`; `(0.0036, 0.0049)` |
| 2D | `(0.5, 0.5)`; `(0.3, 0.3), (0.7, 0.4)`; both `0.01 I` | `(0.45, 0.55)`; `(0.5, 0.6), (0.4, 0.25)`; both `0.01 I` |

Each case has two source and two target components. `MW2` solves the exact
component LP using the existing `solve_mw2` implementation. For Mix, SMix, B,
and B1D, `avg-LSOT` averages all `L` lifted plans; `min-LSOT` picks the
lowest true Gaussian cost in that same bank; `min-opt-LSOT` starts at the bank
minimum and returns the cheapest actual lift it evaluates. All three call the
existing `lsot` implementations. There is no pixel color transfer in this
experiment.

## Results

- `metrics.csv`: 13 plans per case (MW2 plus three modes times four projection
  families), squared Gaussian transport cost, percentage gap to MW2, plan L1
  difference, plan size, and transport computation time. Time excludes bank
  sampling, figures, and file output. `min-opt` also records the initial
  minimum cost and best iteration.
- `plans/*.npz`: each sparse component coupling (`rows`, `cols`, `mass`,
  `shape`).
- `1d_components.png`, `2d_components.png`: source and target densities.
- `1d_couplings.png`, `2d_couplings.png`: all component plans on one color scale.
- `1d_interpolations.png`: Gaussian displacement interpolation at
  `t=0.2, 0.5, 0.8`; `2d_interpolations.png`: contours at `t=0.5`.
- `maps_1d/*.png`: every component's Gaussian map and the conditional-mean
  map, including MW2.
- `1d_grid_w2.png`, `2d_grid_w2.png`: the notebook's *separate* discrete-grid
  W2 comparison. The 1D plot includes its `ot.emd` grid coupling. Midpoint
  barycenters use POT's entropically regularized `bregman.barycenter`, with
  `reg=0.001`, on a 100-point 1D or 50-by-50 2D grid. They are not exact
  W2 barycenters of the continuous GMMs. Skip these figures with
  `--skip-grid-w2` if only component-OT methods are needed.

The original notebook normalizes its 1D grid histograms but passes
unnormalized 2D density samples into the POT barycenter. Here *both* are
normalized to unit mass before the grid W2 comparison; this is an intentional
correction. The Gaussian interpolation panels use continuous mixture densities
sampled on those grids. Changing the grid resolution affects the figures and
regularized grid baseline, never the component plans or their costs.

These two-component examples are useful for checking coupling and geodesic
construction. They do not measure scaling with `K`, statistical error, or
quality on real data.
