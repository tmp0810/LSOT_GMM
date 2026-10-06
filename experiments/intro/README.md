# 1D/2D interpolation panels for presentation

Run from the repository root in Colab:

```python
%cd /content/LSOT_GMM
!git pull --ff-only origin main
!python -m pip install -q -e .
!python -m experiments.intro.run --L 100 --output-dir results/intro
```

The original notebook's fixed two-component mixtures are used. There is no
EM fit or K sweep in this experiment. Exactly **eight LSOT methods** run:
average (`avg-LSOT-*`) and minimum (`min-LSOT-*`) across Mix, SMix, B and
B1D. MW2 is the ninth component-plan reference. The solver and interpolation
formulas are unchanged; no min-opt method runs.

By default, both dimensions export **t=0, 0.2, 0.4, 0.6, 0.8, 1**. For example,
under `results/intro/methods/min-LSOT-Mix_L100/`:

| Path | Contents |
| --- | --- |
| `1d/t_0.0.png` ... `1d/t_1.0.png` | Six individual 1D density panels. |
| `2d/t_0.0.png` ... `2d/t_1.0.png` | Six individual 2D contour panels. |
| `1d/interpolation_strip.png` / `.pdf` | Six-column strip for this method. |
| `2d/interpolation_strip.png` / `.pdf` | Corresponding 2D contour strip. |
| `1d/w2_vs_method.png` / `.pdf` | Two rows: regularized grid W2 above, this method below. |
| `2d/w2_vs_method.png` / `.pdf` | Corresponding 2D two-row figure. |
| `1d/sequence.npz`, `2d/sequence.npz` | Actual times, grid, points and normalized grid probabilities. |

Equivalent folders are generated for all eight LSOT methods and `MW2`.
`methods/grid-W2/` holds the six grid-W2 panels and its strips. The older
`1d_grid_w2.png`/`2d_grid_w2.png`, combined method figures, metrics, plans
and 1D maps are retained. The old combined previews still show three
overlaid 1D times / the 2D midpoint; the new method folders contain **all six
times in both dimensions**.

The two-row layout follows the supplied MW2 paper figure, but the upper row
is explicitly **a regularized discrete-grid W2 barycenter**, not an exact
continuous W2 geodesic. It uses POT with regularization 1e-3, normalized
input histograms and weights (1-t,t); 1D costs are scaled by their maximum,
as in the original implementation. This comparison is computed once per
case and reused across methods. `--skip-grid-w2` omits this baseline and
two-row comparisons, while still exporting all method panels and strips.

For each method's admissible component plan P, the lower row plots the
mixture of pairwise Gaussian displacement interpolants with weights P_ij.
Only the MW2 plan defines the MW2 geodesic; LSOT-plan paths are not claimed
to be globally optimal barycenters for a metric. t=0 is the source mixture
and t=1 the target; these endpoint panels agree for every method. Each
panel is normalized to sum to one on the same evaluation grid, with common
1D vertical limits / 2D contour levels across methods and times.

Use `--times 0 0.25 0.5 0.75 1` to choose a different increasing time list
in [0,1]. Times are recorded in `config.json` and each `sequence.npz`.
Separate output directories are recommended for different L/time settings.

Display the paper-style figures for every method in Colab:

```python
from pathlib import Path
from IPython.display import Image, display

root = Path("/content/LSOT_GMM/results/intro/methods")
for method in sorted(root.iterdir()):
    if method.name == "grid-W2":
        continue
    for case in ("1d", "2d"):
        image = method / case / "w2_vs_method.png"
        if image.exists():
            print(method.name, case)
            display(Image(filename=str(image)))
```

Display every individual t panel for one method:

```python
import numpy as np
folder = root / "min-LSOT-Mix_L100" / "2d"
for t in np.load(folder / "sequence.npz")["times"]:
    tag = f"{t:.12g}"
    if "." not in tag:
        tag += ".0"
    display(Image(filename=str(folder / f"t_{tag}.png")))
```
