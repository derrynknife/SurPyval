# Development Notes

This document tracks **outstanding work** for surpyval — known gaps, technical
debt, and improvement priorities. It is forward-looking only: completed work is
not recorded here (see `docs/changelog.rst` for release history). Items are
grouped by theme and ordered by priority within each section.

**Package state (2026-07-14).** Version `0.11.1`. The full test suite (994
tests, 1 documented skip) passes on Python 3.11–3.13 with numpy 2.x / scipy
1.13+ / pandas 2.2+. The parametric, nonparametric, regression (PH / AFT / PO /
additive-hazards / accelerated-life), recurrent-event, degradation, copula and
(experimental) survival-forest modules are all implemented and tested. The
correctness and release items in §1–§2 are the highest-value next steps.

---

## 1. Release — cut 0.12.0

Two genuine blockers, both process rather than code:

- **Rewrite the changelog.** `docs/changelog.rst` tops out at `v0.11.0
  (planned)`, whose bullets are a TODO wishlist rather than a record; there is
  no `0.11.1` entry and nothing for the large body of work since (parametric
  additive hazards, recurrent parameter-uncertainty + diagnostics, discrete
  lifetime distributions, `handle_xicn` validation, copula and degradation
  features, the simulation/`dist='t'` cleanups, the Fine-Gray regression
  implementation, the Efron-Hessian fix, and delta-method confidence bounds
  (`cb`/`param_cb`/`covariance`) on the parametric regression models). Write
  real per-version entries before tagging.
- **Add a publish workflow.** `.github/workflows/actions.yml` is CI-only (lint +
  pytest matrix + coverage, `on: [push]`). There is no tag-triggered
  `pypa/gh-action-pypi-publish` step, so releasing to PyPI is fully manual. Add
  a release workflow and tag `v0.12.0`.

Polish (non-blocking): add a `docs` optional-dependency group (docs deps
currently live only in `docs/requirements.txt`); add API doc pages for the
newer surfaces (multivariate copulas, degradation RUL, the experimental
forest).

---

## 2. Correctness — broken or fragile public API (high priority)

- **`CompetingRisksProportionalHazards.fit_from_df` is unimplemented** (`raise
  NotImplementedError("Not yet...")`). Add the DataFrame entry point for both
  the Cox and Fine-Gray competing-risks paths.
- **Competing-risks regression needs test coverage and Cox-path review.** The
  Fine-Gray path is now implemented (IPCW subdistribution model, with tests);
  the cause-specific Cox path (`CompetingRisksProportionalHazards.fit(
  how="Cox")`) fits and predicts but is thinly tested, and its shared Breslow
  `baseline()` (all-cause event counts with a Fine-Gray-style risk set) should
  be reviewed for the cause-specific CIF before it is relied on.
- **Cox PH exact / Kalbfleisch-Prentice tie methods.** `cox_ph.py` now supports
  Breslow and Efron ties, both with analytic-Hessian standard errors and
  p-values (the Efron Hessian, previously computed with an inner product in
  place of the outer product `a a'` and then discarded, is fixed and validated
  against finite differences). Two lower-value tie methods remain unimplemented
  (`# TODO: Cox-Exact` / `# TODO: K&P` at `cox_ph.py:460-461`):
  - **Exact partial likelihood** — averages over all orderings of tied events;
    combinatorially expensive and only meaningfully different from Efron under
    heavy ties. Niche.
  - **Kalbfleisch-Prentice** — the discrete-time (grouped) proportional-hazards
    likelihood for tied data. Also niche; Breslow + Efron already match what
    R's `survival` and lifelines offer by default.

  Neither is a common need, so both are deferred. Also: only right-censoring
  and left-truncation (`tl`) are wired; right/interval truncation is missing
  (see §6).
- **`Beta` MPP guard.** `Beta` does not set `supports_mpp = False`, so
  `Beta.fit(how="MPP")` passes the support check and hits a raw
  `NotImplementedError` (`beta.py:404`) instead of the clean `ValueError` other
  MPP-unsupported distributions raise. One-line consistency fix.

---

## 3. Type Hints — finish and ratchet

The package ships `py.typed` and `mypy` runs in CI, but no module yet enables
`disallow_untyped_defs`, so coverage cannot be enforced. Finish the contract:

- Public APIs of the remaining modules: univariate **competing risks**
  (Fine-Gray, cause-specific PH), the **recurrent** package (parametric
  intensity, renewal, regression, nonparametric MCF, competing risks), and the
  **experimental** forest.
- A second pass over the *internal* helpers in every module (the `_cb_*`
  confidence-bound helpers, the `mle`/`mpp`/`mps`/`mse`/`mom` fitters, etc.).
- Once a module has zero unannotated defs, enable `disallow_untyped_defs` for it
  in `[tool.mypy]` so coverage ratchets and cannot regress.

Convention: annotate array-like inputs as `npt.ArrayLike` and array returns as
`npt.NDArray`; declare dynamically-set instance attributes as class-level
annotations; leave the autograd-`numpy` numeric kernels loosely typed.

---

## 4. Recurrent Module — remaining gaps

The high-priority work (parameter uncertainty, `cif_cb`, residuals/trend/CvM
diagnostics on `ParametricRecurrenceModel`, input validation) is done. What
remains:

**Medium priority**

- **Diagnostics for the regression and renewal families.** `residuals()`,
  `trend_test()` and `cramer_von_mises()` exist only on
  `ParametricRecurrenceModel`. The proportional-intensity extension is mostly
  plumbing (scale each item's CIF by its `exp(Z'β)` factor); the
  renewal/virtual-age models need conditional-intensity residuals instead, which
  is a genuinely different construction.
- **Multi-window (gapped) observation per item.** `handle_xicn` supports a
  single `[tl, tr]` window per item; genuinely gapped observation is not yet
  modelled.

**Lower priority**

- **Event marks (`e`).** `handle_xicn` has no event-mark parameter;
  `CauseSpecificMCF.fit` bypasses it and cannot take truncation, and there is no
  `fit_from_df` for marks. Beyond the nonparametric MCF scaffold, parametric
  cause-specific intensity models (cause-specific NHPP) and proportional-
  intensity regression are unbuilt.
- **General geometric-process estimator.** ARA/ARI (and their Kijima-I/II and
  Lam geometric-process equivalences) are implemented; a first-class general
  geometric-process `a` parameterisation is the remaining virtual-age model.

**Test coverage gaps**

- CIF/IIF round-trips for each parametric model (HPP, CrowAMSAA, Duane,
  CoxLewis) are thin (likelihood/AIC/BIC/SE are covered).
- MCF and confidence bounds for `NonParametricCounting`.
- Predict round-trips for proportional-intensity HPP and NHPP.

---

## 5. Semi-Parametric Regression — future work

Proportional hazards (`CoxPH` + parametric `WeibullPH` …) and additive hazards
(semi-parametric `AdditiveHazards` + parametric `AH(dist)`) are both complete.
Three candidates remain, in priority order:

- **Buckley-James (semi-parametric AFT) — high.** The semi-parametric
  counterpart to Cox PH: fits `log(T) = β'Z + ε` with no parametric baseline via
  iterative censoring imputation. Completes the semi-parametric trio.
- **Semi-parametric proportional odds — low.** `O(x|Z) = O₀(x)·exp(β'Z)` with a
  non-parametric baseline odds step function; needs joint NPMLE of `(β, Λ₀)`
  (profile likelihood with an isotonic inner loop). Much harder than Cox PH;
  parametric PO already covers most practical cases.
- **Frailty models — low.** `h(x|Z,u) = u·h₀(x)·exp(β'Z)` with a subject-level
  random effect (Gamma / log-normal). Different problem class (clustered data);
  defer until the above are stable.

---

## 6. Truncation and Time-Varying Covariates

Current state to build on (correcting an earlier note): `AFTFitter` and
`ProportionalOddsFitter` **do** accept a truncation argument `t` and thread it
into `SurpyvalData`, and `ProportionalHazardsFitter` handles left-truncation via
`tl`. The outstanding work is:

- **Confirm and test truncated-likelihood correctness.** Verify that the
  `S(t_L|Z) − S(t_R|Z)` normalisation is actually applied (not merely accepted)
  across AFT / PO / PH, and add regression tests for double truncation. Add
  right/interval truncation to Cox PH (only `tl` is wired today).
- **Time-varying covariates via start-stop format** `(t_start, t_stop, event,
  Z)` — each interval is a left-truncated observation, so truncation support is
  the prerequisite. Difficulty varies by family:
  - Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds`
    accumulates linearly; interval contributions are just `β'·Z·Δt`.
  - Cox PH and parametric PH: medium — cumulative hazard is additive over
    intervals, so segments compose cleanly.
  - Parametric AFT: harder — must track cumulative "accelerated age"
    `φ(Z₁)·t₁ + φ(Z₂)·(t₂−t₁) + …` across prior intervals.
  - Parametric PO: not practical — no additive structure; needs numerical
    integration per interval.

---

## 7. Degradation Analysis — future work

`surpyval.degradation` ships the pseudo-failure-time approach (per-unit path
fits over 9 forms with `path="best"` AICc selection, extrapolation to a
threshold, lifetime fit to the pseudo failure times), a two-stage
noise-corrected population path-parameter distribution (moment and REML
variants), and `predict_rul` shrinkage RUL predictions. Natural extensions,
roughly in priority order:

- **Uncertainty propagation for the life model.** The two-stage method treats
  pseudo failure times as exact, so life-model confidence bounds understate the
  true uncertainty. A bootstrap over units (refit paths + life model per
  resample) is the cheap fix; expose it as a `cb`/`bootstrap` option.
- **Induced failure-time distribution.** With `(μ, Σ, σ²)` fitted (especially by
  REML), derive the population failure-time distribution from the path model
  directly (Monte Carlo over `θ ~ MVN` through `inv_path`) instead of via noisy
  pseudo failures — the full Lu–Meeker program. Compare against the
  pseudo-failure Weibull as a diagnostic.
- **REML for nonlinear paths.** Exponential/power paths need FO/FOCE-style
  linearisation (or fitting the exponential path on the log scale, where it is
  linear). Hierarchical uncertainty in `(μ, Σ)` itself is also unmodelled. Note
  the moment `path_param_cov` is genuinely unreliable with few units (it warns
  and PSD-clips) — the bootstrap above is the more robust route.
- **Stochastic-process degradation models.** Wiener process with drift (first
  passage → inverse-Gaussian lifetimes) and gamma process (monotone
  degradation) — genuine models with proper likelihoods that handle measurement
  error and irregular sampling more honestly.
- **Destructive degradation** (one measurement per unit) — needs a different
  data model since per-unit path fits are impossible.

### Covariates / accelerated degradation testing (design)

Covariates (accelerating stresses such as temperature/voltage/humidity, or
observational factors such as lot/supplier/load) can enter at two different
stages, and they are genuinely different features:

**Stage 1 — covariates on the life fit (classic ADT; build first).** Keep the
per-unit path fits unchanged; each unit carries a constant covariate vector
`Z_i`. Instead of fitting a plain distribution to the pseudo failure times, feed
`(pseudo_failure_time_i, c_i, Z_i)` into the *existing* univariate regression
fitters (AFT / PH / parameter-substitution life-stress models). API sketch:
`DegradationAnalysis.fit(x, y, i, Z=..., distribution=<regression fitter>)`, with
`Z` forwarded through the delegated lifetime functions so life at use conditions
is one call. Censored units flow through since the regression fitters take `c`.
Mostly plumbing. Theoretical justification: for a linear path with stress acting
on the rate, `T = (y_t − a)/b(Z)`, so `log T = log(y_t − a) − log b(Z)` — exactly
an AFT model. Open design point: interaction with `path="best"` (default: select
on pooled data, since the path stage stays per-unit).

**Stage 2 — covariates on the path parameters (mechanistic).** Model the
degradation rate itself: `θ_i = Γ z_i + u_i` (e.g. Arrhenius when a rate is
log-linear in `1/T`). This buys what Stage 1 cannot: pooling strength across
units, stress-conditional priors for `predict_rul`, and stresses that change the
path *shape* rather than just its scale. For linear-in-parameter paths with
identity links this stays a linear mixed model — `y_i = (z_i' ⊗ X_i) vec(Γ) +
X_i u_i + ε` — so the existing REML code needs only a wider fixed-effects design
matrix. Physically-motivated links (log/Arrhenius) make the fixed effects
nonlinear: fit on a transformed scale or wrap one Gauss-Newton layer around the
LMM.

**Stage 3 — time-varying stress (step-stress profiles).** Cumulative-exposure /
cumulative-damage models where the rate changes with the stress profile
mid-test. Hardest by far (same reasons as TVCs in §6); defer until Stages 1–2
are stable.

---

## 8. Discrete Lifetime Distributions — Tier 2/3

Tier 1 (`Geometric`, `DiscreteWeibull`, `NegativeBinomial` on `{1, 2, 3, …}`
with zero-inflation at 0) is done. Extensions:

- **Tier 2:** a standalone `Poisson` distribution (count data; distinct from the
  recurrent Poisson *processes*); `BetaGeometric` (discrete-time frailty —
  Geometric with Beta-mixed `p`); and a general "discretize any continuous
  distribution" wrapper (`P(k) = F(k+1) − F(k)`) to cheaply yield discrete
  Gamma/Lognormal/Normal.
- **Tier 3 (niche / heavy-tailed):** Zeta / discrete Pareto / Yule–Simon, the
  logarithmic (log-series) distribution, and discrete Rayleigh/Gompertz hazard
  shapes.
- **Probability plotting** for the discrete families (step-aware plotting
  positions) so they are not MLE-only, plus a discrete-aware `plot()`.

---

## 9. Multivariate Copulas — future work

Five bivariate families (`Independence`, `Clayton`, `Gumbel`, `Frank`,
`Gaussian`) are implemented with censoring/truncation-aware likelihoods, IFM +
MLE fitting, dependence measures and conditional-inversion sampling. Outstanding:

- **More families:** Student-t (tail dependence), Joe, AMH.
- **Beyond bivariate:** `Copula.fit` currently rejects dimension ≠ 2. Nested or
  vine constructions are needed for genuine multivariate (D > 2) dependence.

---

## 10. Code Quality and Smaller Gaps

- **Doctest policy.** Most distribution docstring examples execute, but
  scalar-returning examples (`mean`, `moment`) print pre-numpy-2 plain floats, so
  `pytest --doctest-modules` fails on them. Decide: render with `np.float64(...)`
  reprs and run doctests in CI, or keep the readable style and accept that
  examples are unchecked.
- **Document the MPP/Gamma limitation.** MPP shape estimation for `Gamma` (and
  especially `Gamma` with an offset) is unreliable — the probability plot's own
  shape depends on the unknown shape parameter, so errors compound. This is a
  fundamental limitation of probability-plotting the family, not a bug
  (`test_fit.py:352` skips it deliberately). Worth a clear docstring/warning
  pointing users to MLE.
- **Niche unimplemented paths (deferred, genuine but low-value):** defective
  (LFP) `qf`/`moment`/`entropy`; MSE estimation with truncation;
  `ExactEventTime` for interval-censored data. Each raises an explicit
  `NotImplementedError` today; implement only if a use case appears.

---

## 11. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU
support; JAX is the spiritual successor and a near-drop-in replacement for
`autograd.numpy` patterns. The interim compatibility steps (inlined
`autograd_gamma` gradients, autograd 1.8 for numpy 2.x) are done, so there is no
urgency. Revisit once the library is otherwise stable — it is a multi-week effort
touching every gradient computation.
