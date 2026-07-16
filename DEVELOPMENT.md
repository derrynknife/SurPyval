# Development Notes

This document tracks **outstanding work** for surpyval — known gaps, technical
debt, and improvement priorities. It is forward-looking only: completed work is
not recorded here (see `docs/changelog.rst` for release history). Items are
grouped by theme and ordered by priority within each section.

**Package state (2026-07-15).** Version `0.12.0` is released to PyPI; `0.13.0`
is in progress on `master` (unreleased — see `docs/changelog.rst`, which adds
the Tier-2 discrete distributions, Cox time-varying covariates, the Cox
delayed-entry baseline fix, a parametric competing-risks model, and a broad
documentation pass). The full test suite passes on Python 3.11–3.13 with numpy
2.x / scipy 1.13+ / pandas 2.2+. The parametric, nonparametric, regression
(PH / AFT / PO / additive-hazards / accelerated-life), recurrent-event,
degradation, copula and (experimental) survival-forest modules are all
implemented and tested. The highest-value remaining threads are the
degradation extensions (§7), the recurrent-module gaps (§4), and the
type-hint ratchet (§3).

---

## 1. Release & docs polish

Release tooling is in place: `.github/workflows/publish.yml` builds the
sdist/wheel and uploads on a `v*` tag via PyPI Trusted Publishing (OIDC), with
a guard that the tag matches the packaged version, and `docs/changelog.rst`
carries a real per-version history. The narrative guides now have executed,
worked examples across the model families (regression, competing risks,
degradation, copulas, data-input flexibility) and the estimation-theory
sections are complete. Remaining polish (non-blocking):

- **Cut 0.13.0** when ready: bump `pyproject.toml` / `surpyval/__init__.py` to
  `0.13.0`, finalise the changelog entry's date, tag `v0.13.0`.
- Add a `docs` optional-dependency group (docs deps currently live only in
  `docs/requirements.txt`).
- Add API (autodoc) reference pages for the newer surfaces (multivariate
  copulas, degradation RUL / ADT covariates, the experimental forest); only
  narrative examples exist for these today.

---

## 2. Correctness — broken or fragile public API (high priority)

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

  Neither is a common need, so both are deferred. Cox wires right-censoring and
  left-truncation (`tl`, including delayed-entry / start-stop time-varying
  covariates); right/interval truncation is intentionally rejected with a clear
  error, as the forward partial likelihood cannot express it.

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

- **Diagnostics for the regression and renewal families.** `residuals()` and
  `trend_test()` now also exist on the proportional-intensity regression models
  (each item's time-rescaling residuals use its own `exp(Z'β)`-scaled CIF).
  Still missing: `cramer_von_mises()` for the regression family (its parametric
  bootstrap needs a covariate-aware refit — the model does not yet hold a
  reference to its fitter), and all three diagnostics for the renewal/virtual-
  age models, which need conditional-intensity residuals instead — a genuinely
  different construction.
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

The semi-parametric trio is complete: proportional hazards (`CoxPH`), additive
hazards (`AdditiveHazards`), and now accelerated failure time (`BuckleyJames` —
`log T = β'Z + ε` with an unspecified error distribution, fit by the
Buckley-James imputation iteration with an Efron-corrected residual KM,
two-cycle handling, bootstrap CIs, and `fit_from_df`; coefficients follow the
package's accelerated-failure sign convention). Two lower-priority candidates
remain:

- **Semi-parametric proportional odds — low.** `O(x|Z) = O₀(x)·exp(β'Z)` with a
  non-parametric baseline odds step function; needs joint NPMLE of `(β, Λ₀)`
  (profile likelihood with an isotonic inner loop). Much harder than Cox PH;
  parametric PO already covers most practical cases.
- **Frailty models — low.** `h(x|Z,u) = u·h₀(x)·exp(β'Z)` with a subject-level
  random effect (Gamma / log-normal). Different problem class (clustered data);
  defer until the above are stable.

---

## 6. Time-Varying Covariates for the parametric families

Truncation and Cox time-varying covariates are done (0.13.0): the parametric
AFT / PO / PH truncation correction is verified with active covariates, and
`CoxPH.fit_tvc` / `fit_tvc_from_df` / `predict_tvc` fit counting-process
(start-stop) data with a delayed-entry-correct baseline. What remains is
time-varying covariates for the *parametric* families. Each interval is a
left-truncated observation; difficulty varies by family:

- Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds`
  accumulates linearly; interval contributions are just `β'·Z·Δt`.
- Parametric PH: medium — cumulative hazard is additive over intervals, so
  segments compose cleanly (as in the Cox case).
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
variants), `predict_rul` shrinkage RUL predictions, and `DegradationModel.cb`
two-stage confidence bounds on the reliability (an analytic delta-method /
generated-regressor correction that folds the first-stage path-fit and
extrapolation uncertainty into the life-model covariance, plus a `method=
"bootstrap"` cross-check). Natural extensions, roughly in priority order:

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

### Covariates / accelerated degradation testing

Stage-1 accelerated degradation testing (covariates on the life fit) is done:
`DegradationAnalysis.fit(x, y, i, Z=..., distribution=...)` feeds
`(pseudo_failure_time_i, c_i, Z_i)` into the univariate regression fitters, and
the prediction methods take the stress vector `Z` at which to evaluate life.
Two open points remain there: two-stage confidence bounds for the covariate
life fit (`DegradationModel.cb` / `life_parameter_covariance` currently raise
`NotImplementedError` for covariate models — the generated-regressor correction
is not yet derived for the regression life fit), and the interaction with
`path="best"`. The larger remaining features are Stages 2 and 3:

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

## 8. Discrete Lifetime Distributions — Tier 3

Tier 1 (`Geometric`, `DiscreteWeibull`, `NegativeBinomial`, with zero-inflation)
and Tier 2 (`Poisson`, `BetaGeometric`, and the `Discretize(distribution)`
wrapper) are done. Remaining extensions:

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
- **Niche unimplemented paths (deferred, genuine but low-value):** MSE
  estimation with truncation and `ExactEventTime` for interval-censored data
  each raise an explicit `NotImplementedError` today; implement only if a use
  case appears. (Defective/offset/zero-inflated `qf` and `moment` are now
  implemented; `entropy` is defined where it exists — no probability atom —
  and raises a clear error for limited-failure / zero-inflated models, whose
  mixed discrete-continuous law has no single differential entropy.)

---

## 11. Long-term: Replace `autograd` with JAX (deferred)

`autograd` (HIPS/autograd) is in low-activity maintenance mode with no GPU
support; JAX is the spiritual successor and a near-drop-in replacement for
`autograd.numpy` patterns. The interim compatibility steps (inlined
`autograd_gamma` gradients, autograd 1.8 for numpy 2.x) are done, so there is no
urgency. Revisit once the library is otherwise stable — it is a multi-week effort
touching every gradient computation.
