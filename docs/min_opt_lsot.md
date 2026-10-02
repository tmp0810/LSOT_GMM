# Optimized minimum LSOT for color transfer

`avg`, `min` and `min-opt` use exactly the same component projections and
proportional lifting. For a direction $s$, let $\bar\pi^s$ be the lift
of the weighted one-dimensional optimal coupling. The objective optimized here
is its **Gaussian ground cost**

\[
H(s)=\sum_{ij}\bar\pi^s_{ij}
W_2^2\!\left(\mathcal N(m_i,\Sigma_i),
\mathcal N(n_j,\Lambda_j)\right).
\]

`avg` averages the $L$ lifted plans, and `min` selects the smallest
$H(s)$ among a common bank of $L$ directions. `min-opt` starts at that
same bank minimum and adapts the projection using the Gaussian-perturbation
Stein estimator in [Chapel, Tavenard and Vaiter (DGSWP)](https://github.com/rtavenar/dgswp/blob/main/dgswp/losses.py):

\[
\widehat{\nabla H_\varepsilon}(u)=
\frac{1}{N\varepsilon}\sum_{r=1}^{N}
\bigl[H(D(u+\varepsilon z_r))-H(D(u))\bigr]z_r,
\qquad z_r\sim\mathcal N(0,I).
\]

Here $u$ denotes unconstrained search coordinates and $D(u)$ decodes
them to a valid Mix, SMix, B or B1D projection. Unlike the uniform empirical
clouds in DGSWP, each $H$ evaluation computes **weighted** 1D OT, lifts its
coupling to Gaussian components, and evaluates Gaussian $W_2^2$ pair costs.
`min-opt` returns one actual lift with the lowest *unsmoothed* $H$ evaluated
during the search, including the starting bank minimum. Thus its cost cannot
exceed `min`'s cost for the same bank, although the optimization is heuristic:
it need not find the global minimum. The resulting barycentric color map is
computed by the same method as for the other two plans.

The `L` suffix in `min-opt-LSOT-Mix_L10`, for example, refers to the **initial
bank size**, not the number of optimization steps. The default run config
includes all three aggregations for all four projection families. Run a
smaller experiment first:

```bash
cd /content/LSOT_GMM
python -m pip install -e .
python -m experiments.color_transfer.run \
  --config experiments/color_transfer/configs/smoke.json
```

The main config is `experiments/color_transfer/configs/mw2_reference.json`.
Its optimization parameters are `opt_steps` (number of updates), `opt_samples`
(perturbations per update), `opt_epsilon` (perturbation scale),
`opt_learning_rate`, and `opt_max_gradient_norm`. Their first four values can
also be overridden by CLI flags, e.g. `--opt-steps 10 --opt-samples 4`.
All three methods share each sampled projection bank and evaluation data.
`transport_ms_mean` includes bank scoring and all optimization evaluations;
bank sampling time is reported separately. `*_selection.npz` for `min-opt`
stores initial bank scores, the best decoded direction, the retained cost
history, and its iteration for exact plan replay. `metrics.csv` records
`initial_projection` and `best_iteration`.
