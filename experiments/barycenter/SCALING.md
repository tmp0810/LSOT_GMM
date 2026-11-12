# Scaling barycenters with input mixture size

This experiment compares the existing full-tuple MW2 barycenter solver against
halving-based min-LSOT on four image GMMs. The main comparison is MW2 itself, not
MW2-variational. B, B1D, Mix and SMix, with min or avg, are supported.

## Run on Colab or a GPU server

```bash
python -m pip install -e .
python -m experiments.barycenter.scaling \
  --config experiments/barycenter/configs/scaling.json \
  --device cuda --output-dir results/barycenter_scaling
```

Start with a pilot if desired (use a separate output directory):

```bash
python -m experiments.barycenter.scaling \
  --config experiments/barycenter/configs/scaling.json \
  --component-counts 5 10 20 30 --seeds 0 \
  --device cuda --output-dir results/barycenter_scaling_pilot
```

`--device cpu` gives a same-device control. No changes to the older grid runner
or its presets are required. These commands run the center barycenter and plan
benchmarks, not a 7x7 grid. Add `--steps 200` in a new directory for the larger
optimization-budget control. Input fits are deterministic for a given seed.
Run only the solver or plans with `--task barycenters` or `--task plans`.
Original PNGs can be supplied with `--image-dir PATH`; otherwise the repository's
recovered notebook foreground masks are used, with their provenance recorded.

## Protocol

* Four shapes: redcross, duck, star, batman; 128x128 masks, equal mass per
  foreground pixel, vertical-axis flip as in the original notebook.
* Normalize all coordinates by 127 BEFORE EM; no image-specific rescaling,
  subsampling or component replication. GMM EM: full covariances, n_init=1,
  max_iter=300, tol=1e-3, reg_covar=1e-6 in normalized coordinates. Report
  convergence and fit time; fit time is excluded from solver time.
* K=5,10,20,30,50,100 components in EACH input; paired fit/optimizer seeds 0,1,2.
  These are three independent fitted instances, not three timing repeats of an
  identical instance. Timing spread also reflects data/optimization variability.
* Four equal barycenter weights. `--weights` accepts another positive quadruple
  summing to one. Do not include zero-weight edges in this K^4 scaling protocol.
* LSMT candidate budget Kb=4K-3. Four historical common-quantile starts. Adam
  lr=.003; all four to step25, two to step50, winner to step100: 200 total
  updates, retaining live optimizer states. Fresh L=100 directions by default.
* Fixed validation L=512, checked every10 updates and at stage/checkpoint ends.
  Independent test L=512; min-B and min-B1D test objectives are computed for
  EVERY successful method, including MW2. Returned candidate is best validated
  state. Halving can discard a late-improving start; no convergence claim is made.
* MW2 enumerates K^4 tuples and uses sparse marginal constraints with HiGHS.
  Fast exact 2D Gaussian algebra is enabled. Covariance iterations: at most100,
  relative change tolerance1e-8; achieved convergence and marginal residuals
  are reported. This is a numerical reference, not a symbolic exact solution.

## Timing and resources

Each barycenter method runs in a fresh subprocess. A tiny independent warmup
loads libraries (including one Adam step for LSMT) outside timing; warmup_ms is
reported. Input loading/transfers happen before timing. total_ms includes all
initialization, starts, optimization and validation. It excludes EM, subprocess
startup, saving files, density plots and post-hoc metrics. MW2 uses CPU;
LSMT uses the selected CPU/GPU. CPU threads, package versions, source hashes
and GPU name are recorded. Compare CPU-only runs separately if needed.

The MW2 phase_ms dictionary separates tuple construction, Gaussian costs,
constraint assembly and LP solve. Do not compare only LP time with LSMT total.
Checkpoint elapsed times include the discarded starts already processed;
early checkpoints are retrospective states of the eventual winner and are not
independent anytime-solutions before all selection stages complete.

MemoryMeter samples solver CPU RSS every10ms (short spikes may be missed), and
reports the absolute peak and extra above the resident-input baseline. CUDA
peaks are PyTorch allocated/reserved bytes, not total device usage. The parent
watchdog samples child RSS every50ms including process setup; its peak is a
separate field. A watchdog cannot guarantee avoiding host OOM on a very rapid
allocation; choose limits below available RAM, leaving space for the parent.

Defaults: timeout600s per solver job after warmup, absolute child RSS cap8GiB.
Both methods have the same limits. Preflight skips MW2 if a lower bound on its
materialized arrays already exceeds the RAM cap (at K=100 it exceeds8GiB).
This bound is NOT an estimate of all solver memory. Increase caps only on a
machine with sufficient RAM. CPU memory and GPU VRAM are different budgets.
`--max-tuples 0` disables the optional tuple-count skip; a positive cap is
available for short pilots. Failures do not prevent the other solvers running.

Statuses have distinct meanings:

* `ok`: solver finished; inspect optimizer/covariance convergence diagnostics.
* `timeout`: parent stopped the job after the configured duration.
* `memory_limit`: parent stopped it after observed RSS crossed the cap.
* `skipped_memory_estimate`: not attempted, materialized-array lower bound > cap.
* `skipped_tuple_budget`: not attempted because of a user-set tuple cap.
* `oom`: worker caught MemoryError or CUDA OutOfMemoryError.
* `killed_unknown`: OS killed the process without an identified cause; NOT
  labeled OOM automatically.
* `numerical_stop` / `solver_error` / `worker_error` / `startup_timeout`: inspect
  worker.log. Numerical-stop candidates can be evaluated but are excluded
  from success-only summary means.

Missing MW2 references yield blank relative gaps and reference distances,
never zero. Common MW2 objectives and independent sliced objectives are still
computed. Compare quality wherever BOTH methods succeed. Do not interpret
success-only mean times at partially failing K as uncensored performance.
Do not interpret this as a lower bound on all possible MW2 barycenter solvers.

## Outputs

* manifest.json: full settings, hardware, versions and code hash.
* metrics.csv: per-K, per-seed solver status, total time, memory and quality.
* summary.csv: success counts/statuses and mean/std over successful instances.
* paired_speedups.csv: paired MW2/min-LSOT timing ratios only where both succeed.
* checkpoints.csv: post-hoc quality of the surviving trajectory (when enabled).
* total_ms.png, cpu_peak_rss_bytes.png, cuda_peak_allocated_bytes.png,
  relative_mw2_gap.png: scaling curves. Failed/skipped runs are not plotted as
  finite runtimes; consult status_counts in summary.csv alongside plots.
* plans.csv, plans_time.png: six input pairs per fitted instance, exact MW2
  plans versus sliced plans; all ground-cost work and projection sampling are
  inside the per-call timer. Warmup1, measured repeats5; report median/quartiles,
  feasibility residuals and cost ratio. Exact LP runs on CPU/POT; sliced plans
  use the configured device. This isolates pairwise plan computation from the
  full-tuple barycenter algorithm.
* K_K/seed_S/inputs/: GMMs, masks, fit convergence, cache hashes and provenance.
* K_K/seed_S/METHOD/: spec, raw result, worker.log, barycenter, history,
  stage decisions and checkpoints. comparison.png shows inputs and results.

Resume uses existing raw solver results, including prior failures. `--no-resume`
reruns solver jobs and plans; EM fits remain cached. Different settings or source
versions require a new output directory to prevent mixing incomparable runs.
All prior experiment presets are retained unchanged.

## Synthetic K sweep (no image data)

```bash
python -m experiments.barycenter.scaling \
  --config experiments/barycenter/configs/scaling_synthetic.json \
  --device cuda --output-dir results/barycenter_scaling_synthetic
```

This preset runs MW2 and min-LSOT-B, K=5,10,20,30,50,100 and seeds0,1,2.
It uses the same halving100/fresh protocol, timing, resource limits, saved
checkpoints and pairwise-plan benchmarks. `--data synthetic` can also override
the data choice in other scaling configs. A different output folder is required.

The generator extends the four original 2D synthetic GMMs (3,4,4,3 parents),
not the four image masks. For every original component (alpha,m,Sigma), it
allocates n>=1 children, with counts summing to K in each GMM, apportioned
approximately by the original weights. Each child has mass alpha/n,
mean m+sqrt(tau)*chol(Sigma)*z, z~N(0,I), and covariance (1-tau)*Sigma.
The default tau=0.5 is controlled by `--synthetic-spread` (strictly between0 and1).
Means are distinct almost surely; this does not replicate identical atoms.
Each parent's expected child-mixture distribution equals the original Gaussian,
while finite realizations differ. Thus this is a representation-resolution
benchmark preserving the original shape family in expectation, not a claim
that increasing K necessarily creates intrinsically harder underlying shapes.
Seeds are deterministic per input/parent, sharing random prefixes across K.
No means or Gaussian tails are clipped. Plots retain the original [0,1]^2 window.

Generation time is reported separately as generation_ms; fit_ms is zero and EM
convergence is null (not applicable). Input NPZs, counts and template hashes
are cached and shared by all methods. Minimum K is4. `em_steps`, `em_tolerance`
and `image_dir` do not control synthetic generation; image_dir is rejected for
this mode to prevent ambiguity. test_kinds defaults to B in this preset.

This is distinct from `experiments.barycenter.run --cases synthetic`, which
still uses the untouched original 3,4,4,3 GMMs and supports the 7x7 grid. The
scaling script computes one four-input barycenter per K/seed, at equal weights
by default, not 49 grid cells.
