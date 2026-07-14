# Development Notes

This document tracks known issues, technical debt, and improvement priorities for surpyval. Issues are grouped by theme and ordered by severity within each section. Implemented items are removed; this reflects the state of the codebase as of 2026-07-13. The full test suite passes on Python 3.11+ with numpy 2.x / scipy 1.17 / pandas 3.x.

---

## 1. Type Hints — Finish the Package

The package ships `py.typed` and `mypy` runs in CI, so the typing
contract is live and must be honoured across the whole package, not
just the parts done so far.

**Still to do (in suggested order):**
- Public APIs of the remaining modules: univariate **competing risks**
  (Fine-Gray, cause-specific PH), the **recurrent** package
  (parametric intensity, renewal, regression, nonparametric MCF,
  competing risks), and the **experimental** forest.
- A second pass to annotate the *internal* helpers in every module
  (the `_cb_*` confidence-bound helpers, the `mle`/`mpp`/`mps`/`mse`/
  `mom` fitters, etc.) — public-API typing only annotates the surface.
- Once a module is fully typed (public **and** internal defs), turn on
  `disallow_untyped_defs` for it in `[tool.mypy]` so coverage ratchets
  and cannot regress. This cannot be enabled per-module until that
  module has zero unannotated defs.

Convention used so far: annotate array-like inputs as `npt.ArrayLike`
and array returns as `npt.NDArray`; declare dynamically-set instance
attributes as class-level annotations; leave the autograd-`numpy`
numeric kernels loosely typed (they buy little from precise typing).

---

## 2. Recurrent Module — Gaps

The former high-priority gaps are closed. Parameter uncertainty: every
likelihood-fitted recurrent model (parametric intensity,
proportional-intensity regression, renewal) exposes `covariance()`,
`standard_errors()` and `param_cb()` (Wald bounds on a transformed scale
respecting each parameter's support) via `LikelihoodInferenceMixin`, and the
parametric and regression models have a delta-method `cif_cb()` drawn as a
band by `plot()`. Model validation: `ParametricRecurrenceModel` has
`residuals()` (cumulative-hazard / PIT / per-item martingale, via the
time-rescaling theorem), `trend_test()` (runs the standalone Laplace /
MIL-HDBK-189C tests on the fitted data's windows) and `cramer_von_mises()`
(conditional-uniform CvM statistic with a parametric-bootstrap p-value).

### Missing capabilities — medium priority

**Diagnostics for the regression and renewal families**
`residuals()`, `trend_test()` and `cramer_von_mises()` exist only on
`ParametricRecurrenceModel`. The proportional-intensity extension is
mostly plumbing (scale each item's CIF by its `exp(Z'beta)` factor); the
renewal/virtual-age models need conditional-intensity residuals instead,
which is a genuinely different construction.

**Left-truncation support (partial)**
`handle_xicn` takes the surpyval `t`/`tl`/`tr` truncation fields, and the calendar-time NHPP models (`HPP`, `CrowAMSAA`, `Duane`, `CoxLewis`) integrate each item's likelihood from its entry time `tl`, so delayed-entry (warranty-from-first-sale) data is analysed correctly there. The virtual-age / history-dependent models (Kijima/G1/ARA/ARI) reject `tl > 0` with an explanatory error, since the virtual age at entry is undefined without the pre-entry history. Still to do: multi-window (gapped) observation per item.

**Input validation (done)**
`handle_xicn()` now validates its input up front with informative `ValueError`s: empty arrays, non-finite (NaN/inf) event times, invalid censoring codes (outside `{-1, 0, 1, 2}`), non-positive or non-finite counts, NaN truncation bounds, and non-finite covariates are all rejected. Event times are checked against each item's integration origin — its left-truncation bound when finite, otherwise the fallback origin `0` — so untruncated negative times are rejected while a genuine (possibly negative) left-truncation window still admits negative times.

---

### Missing capabilities — lower priority

**`handle_xicn` has no `e` (event-mark) parameter**
`CauseSpecificMCF.fit` bypasses it and can't take truncation; no `fit_from_df` for marks.

**General geometric-process estimator**
Both Doyen & Gaudoin imperfect-repair families (ARA, ARI) are implemented, along with their equivalences to Kijima-I/II and the Lam geometric process. The leftover virtual-age model worth adding is the general geometric-process estimator if a first-class `a` parameterisation is wanted.

**Competing failure modes (partial)**
The nonparametric scaffold is in place: `RecurrentEventData` carries an optional event-type (mark) column and `surpyval.recurrent.competing_risks.CauseSpecificMCF` produces per-cause MCF curves over a shared at-risk set. Still to do: parametric cause-specific intensity models (cause-specific NHPP) and proportional-intensity regression, plus `fit_from_df`/`handle_xicn` support for the mark column.

---

### Test coverage

The renewal, parametric and regression sub-packages have tests
(`tests/recurrent/test_counting.py`, `test_ara.py`, `test_ari.py`,
`test_cox_lewis.py`, `test_counting_process.py`, `test_parametric_inference.py`,
`test_hpp_proportional_intensity.py`, `test_regression_simulation.py`, …). The
remaining gaps for minimum viable coverage:

- CIF/IIF round-trips for each parametric model (HPP, CrowAMSAA, Duane,
  CoxLewis) are still thin (likelihood/AIC/BIC/SE are already covered).
- MCF and confidence bounds for `NonParametricCounting`.
- Predict round-trips for proportional intensity HPP and NHPP (fit/inference
  already covered).

---

## 3. Code Quality

### Docstring examples are not doctested
Most distribution docstring examples now execute correctly, but
scalar-returning examples (`mean`, `moment`) print pre-numpy-2 style
plain floats, so `pytest --doctest-modules` fails on them. Decide a
policy: either render with `np.float64(...)` reprs and run doctests in
CI, or keep the readable plain-float style and accept that examples
are unchecked.

### Duplicated simulation block
The same "simulate timelines to `T`" loop appears twice:

- `surpyval/recurrent/parametric/parametric_recurrence.py` (`time_terminated_simulation`)
- `surpyval/recurrent/regression/proportional_intensity.py` (`time_terminated_simulation`)

Extract a shared helper. The `count_terminated_simulation` methods are similarly duplicated.

---

## 4. Univariate Non-Parametric Module — Complete

The tracked engineering debt in `surpyval/univariate/nonparametric/` is
cleared: monotone (PCHIP) smooth interpolation, seedable `random()`, and
`to_dict`/`from_dict`/JSON serialization are done; the unjustified
`dist='t'` confidence-bound heuristic has been removed (`R_cb`/`cb` now
accept only `dist='z'`, raising an informative error for `'t'` that
points to `bootstrap_cb`/`band`); and the previously-untested `from_xrd`,
`set_lower_limit`, and non-step `plot()` paths now have direct tests.

Note (out of scope here, tracked under §2): `NonParametricCounting.mcf_cb`
in the recurrent module carries the same `dist='t'` heuristic and could be
removed for consistency.

---

## 5. Semi-Parametric Regression — Future Work

Additive hazards is implemented on both scales, completing the symmetry
with proportional hazards (semi-parametric `CoxPH` + parametric
`WeibullPH` etc.):

- **Semi-parametric** `AdditiveHazards` (Lin & Ying 1994) — the closed-form
  estimator `h(x|Z) = h₀(x) + β'Z` with an unspecified baseline, sandwich
  covariance, p-values, a Breslow-type baseline, `sf`/`ff`/`hf`/`Hf`/`df`
  prediction and `fit_from_df`.
- **Parametric** `AH(dist)` factory + pre-built `WeibullAH`,
  `ExponentialAH`, … — `h(x|Z) = h₀(x; θ) + β'Z` with a parametric
  baseline, fit by plain MLE. Positivity is not enforced: the fit finds
  the best positive-hazard solution and raises (pointing at PH) only if
  the optimum genuinely needs a negative hazard.

Three candidates remain, in priority order:

### Buckley-James (semi-parametric AFT) — High priority
The semi-parametric counterpart to Cox PH. Fits `log(T) = β'Z + ε` without assuming a parametric baseline distribution. Uses an iterative censoring-imputation (Buckley-James) algorithm. Completes the semi-parametric trio alongside `CoxPH`. Supported in R's `survival::survreg` with no baseline assumption.

### Semi-parametric Proportional Odds — Low priority
`O(x|Z) = O₀(x) · exp(β'Z)` with a non-parametric baseline odds step function. Requires joint NPMLE of (β, Λ₀) — profile likelihood with isotonic regression inner loop (Murphy, Rossini & van der Vaart, 1997). Significantly more complex than Cox PH to implement. Parametric PO (`WeibullPO`, `LogisticPO`, etc.) already covers most practical cases.

### Frailty Models — Low priority
`h(x|Z, u) = u · h₀(x) · exp(β'Z)` where `u` is a subject-level random effect (Gamma or log-normal frailty). Different problem class (clustered/recurrent data); significant scope increase. Defer until the other three are stable.

---

## 6. Degradation Analysis — Future Work

`surpyval.degradation` ships the classic pseudo-failure-time approach: per-unit
least-squares path fits (linear, quadratic, exponential, offset-exponential,
power, logarithmic, Lloyd-Lipow, Gompertz, Michaelis-Menten — plus
`path="best"` AICc selection across all of them), extrapolation to a
threshold, and a lifetime-distribution fit to the resulting pseudo failure
times (units whose path never reaches the threshold are right censored at
their last observation). It also has a two-stage noise-corrected population
path-parameter distribution (moments and REML variants) and
`DegradationModel.predict_rul(x, y)` for shrinkage RUL predictions with
credible intervals.

Natural extensions, roughly in priority order:

- **Uncertainty propagation for the life model.** The two-stage method treats
  pseudo failure times as exact, so life-model confidence bounds understate
  the true uncertainty. A bootstrap over units (refit paths + life model per
  resample) is the cheap fix; expose it as a `cb`/`bootstrap` option on the
  model.
- **Induced failure-time distribution.** With `(mu, Sigma, sigma^2)` fitted
  (especially by REML), the population failure-time distribution can be
  derived from the path model directly (Monte Carlo over `theta ~ MVN` through
  `inv_path`) instead of via noisy pseudo failures — the full Lu–Meeker
  program. Compare against the pseudo-failure Weibull as a diagnostic.
- **REML for nonlinear paths.** Exponential/power paths need FO/FOCE-style
  linearisation of the mixed model (or fitting the exponential path on the
  log scale, where it is linear). Uncertainty in `(mu, Sigma)` itself
  (hierarchical bands) is also unmodelled.
- **Stochastic-process degradation models.** Wiener process with drift (first
  passage → inverse Gaussian lifetimes) and gamma process (monotone
  degradation) — these are genuine models with proper likelihoods, and they
  handle measurement error and irregular sampling more honestly.
- **Destructive degradation** (one measurement per unit) — needs a different
  data model since per-unit path fits are impossible.

### Covariates / accelerated degradation testing (design)

Covariates (accelerating stresses such as temperature/voltage/humidity, or
observational factors such as lot/supplier/load) can enter the pipeline at two
different stages, and they are genuinely different features:

**Stage 1 — covariates on the life fit (classic ADT; build first).** Keep the
per-unit path fits unchanged; each unit also carries a covariate vector `Z_i`
(constant per unit — one row per unit, or a column in `fit_from_df`). Instead
of fitting a plain distribution to the pseudo failure times, feed
`(pseudo_failure_time_i, c_i, Z_i)` into the *existing* univariate regression
fitters (AFT / PH / parameter-substitution life-stress models, formulaic
formulas and all). API sketch: `DegradationAnalysis.fit(x, y, i, Z=...,
distribution=<regression fitter>)`, with `Z` forwarded through the delegated
lifetime functions (`model.sf(t, Z=use_stress)`) so life at use conditions is
one call. Censored (non-degrading) units flow through naturally since the
regression fitters already take `c`. Mostly plumbing — a small, separate PR.
Theoretical justification: for a linear path with stress acting on the rate,
`T = (y_t - a) / b(Z)`, so `log T = log(y_t - a) - log b(Z)` — exactly an AFT
model. For rate-acting stresses on threshold-crossing models the
pseudo-failure + AFT composition is the correct functional form, not just a
pragmatic one. Open design point: interaction with `path="best"` (default:
select on the pooled data as now, since the path stage stays per-unit either
way).

**Stage 2 — covariates on the path parameters (mechanistic).** Model the
degradation rate itself: `theta_i = Gamma z_i + u_i` (e.g. Arrhenius when a
rate is log-linear in `1/T`). This buys what Stage 1 cannot: pooling strength
across units (a unit with two points at high stress still informs `Gamma`, so
per-unit fits need not be possible), stress-conditional priors for
`predict_rul` (`theta ~ MVN(mu(Z), Sigma)` — the blend then knows a hot unit
should be degrading faster *before* seeing its data), and stresses that change
the path *shape* rather than just its scale. Implementation: for
linear-in-parameter paths with identity links this stays a linear mixed model
— `y_i = (z_i' ⊗ X_i) vec(Gamma) + X_i u_i + eps` — so the existing REML code
needs only a wider fixed-effects design matrix; the GLS profiling and
variance-parameter optimisation are untouched. Physically-motivated links
(log/Arrhenius, to keep rates positive) make the fixed effects nonlinear:
either fit on a transformed scale or wrap one Gauss-Newton layer around the
LMM.

**Stage 3 — time-varying stress (step-stress profiles).** Cumulative-exposure
/ cumulative-damage models where the rate changes with the stress profile
mid-test. Hardest by far (same reasons as time-varying covariates for the
regression module — see section 7); defer until Stages 1-2 are stable.

---

## 7. Time-Varying Covariates and Truncation (to be confirmed)

Full support for time-varying covariates (TVCs) and left/right truncation across all regression model families needs to be designed and confirmed before implementation. Key points established so far:

- **Left and right truncation** need to be added to `AFTFitter` and `ProportionalOddsFitter` — `ProportionalHazardsFitter` already handles both via `data.tl`/`data.tr`. Double truncation (observing only if `t_L < T < t_R`) is handled by dividing the likelihood by `S(t_L|Z) - S(t_R|Z)`.
- **TVCs via start-stop format** `(t_start, t_stop, event, Z)` — requires truncation support as a prerequisite, since each interval is a left-truncated observation.
- **Difficulty varies by model family:**
  - Cox PH and parametric PH: medium — cumulative hazard is additive over intervals, segments compose cleanly
  - Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds` accumulates linearly; interval contributions are just `β'·Z·Δt`
  - Parametric AFT: harder — must track cumulative "accelerated age" `φ(Z₁)·t₁ + φ(Z₂)·(t₂-t₁) + ...` across prior intervals
  - Parametric PO: not practical — no additive hazard structure; would require numerical integration per interval

---

## 8. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU support. JAX is the spiritual successor and a near-drop-in replacement for `autograd.numpy` patterns. The interim steps (inlining the `autograd_gamma` gradients into `surpyval/utils/autograd_gamma_compat.py` and upgrading to `autograd` 1.8 for numpy 2.x compatibility) are done, so there is no urgency. A JAX migration can be revisited once the library is otherwise stable — it is a multi-week effort touching every gradient computation.

---

## 9. Discrete Lifetime Distributions

Tier 1 is implemented: `Geometric`, `DiscreteWeibull` (Nakagawa–Osaki Type
I) and `NegativeBinomial`, all discrete lifetimes on `{1, 2, 3, ...}` with
zero-inflation living at 0. They flow through the standard MLE machinery
(censoring, truncation, counts, `zi`/`lfp`) by overriding `log_df`/`log_sf`
and the function set, with `supports_mpp = False`. Validated against
`scipy.stats` and by parameter recovery under censoring/truncation/ZI.

Natural extensions:

- **Tier 2:** a standalone `Poisson` distribution (count data; distinct
  from the recurrent Poisson *processes*); `BetaGeometric` (discrete-time
  frailty — Geometric with Beta-mixed `p`); and a general "discretize any
  continuous distribution" wrapper (`P(k) = F(k+1) − F(k)`) to cheaply
  yield discrete Gamma/Lognormal/Normal.
- **Tier 3 (niche / heavy-tailed):** Zeta / discrete Pareto / Yule–Simon,
  the logarithmic (log-series) distribution, and discrete
  Rayleigh/Gompertz hazard shapes.
- **Probability plotting** for the discrete families (step-aware plotting
  positions) so they are not MLE-only; and a discrete-aware `plot()`.
