# Development Notes

This document tracks **outstanding work** for surpyval — known gaps, technical
debt, and improvement priorities. It is forward-looking only: completed work is
not recorded here (see `docs/changelog.rst` for release history). Items are
grouped by theme and ordered by priority within each section.

**Package state (2026-07-15).** Version `0.12.0` (released to PyPI). The full
test suite (1078 tests, 1 documented skip) passes on Python 3.11–3.13 with numpy
2.x / scipy 1.13+ / pandas 2.2+. The parametric, nonparametric, regression (PH /
AFT / PO / additive-hazards / accelerated-life), recurrent-event, degradation,
copula and (experimental) survival-forest modules are all implemented and
tested. With 0.12.0 shipped, the degradation extensions (§7) and correctness
hardening (§2, §6) are the highest-value next steps.

---

## 1. Release & docs polish

0.12.0 is released to PyPI. The changelog (`docs/changelog.rst`) carries a real
per-version history, and `.github/workflows/publish.yml` builds the sdist/wheel
and uploads on a `v*` tag via PyPI Trusted Publishing (OIDC), with a guard that
the tag matches the packaged version. Remaining polish (non-blocking):

- Add a `docs` optional-dependency group (docs deps currently live only in
  `docs/requirements.txt`).
- Add API doc pages for the newer surfaces (multivariate copulas, degradation
  RUL / ADT covariates, the experimental forest).

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

  Neither is a common need, so both are deferred. Also: only right-censoring
  and left-truncation (`tl`) are wired; right/interval truncation is missing
  (see §6).

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

## 6. Truncation and Time-Varying Covariates

Current state to build on: the parametric `AFTFitter`, `ProportionalOddsFitter`
and `ProportionalHazardsFitter` accept a truncation argument `t` and thread it
into `SurpyvalData`, and the `regression_neg_ll` truncation correction
(`− log[F(t_R|Z) − F(t_L|Z)]` per row, each with its own covariates) is now
*verified* correct: a covariate-truncation recovery test confirms the
coefficient and scale are recovered from left-, right-, interval- and
partially-truncated data, while the naive (ignore-truncation) fit is biased.
Semi-parametric `CoxPH` handles left-truncation (delayed entry) via `tl`; its
partial likelihood matches a brute-force reference on staggered-entry data and
now satisfies the episode-splitting identity (the `fit` optimiser gained a
minimisation fallback so staggered-risk-set data converges). Right/interval
truncation for Cox is rejected with a clear error (a 2-D `tl`), since the
forward partial likelihood cannot express it. The outstanding work is:

- **Time-varying covariates for Cox — done.** `CoxPH.fit_tvc` /
  `fit_tvc_from_df` take counting-process (start-stop) data `(ident, start,
  stop, event, Z)`, one row per interval on which the covariates are constant.
  `handle_tvc` validates the structure (non-overlapping intervals, at most one
  terminal event on a subject's last interval) and maps each interval to a
  delayed-entry observation `(x = stop, tl = start, c = 1 − event)`, which the
  partial likelihood fits exactly. The Breslow baseline was fixed to respect
  `tl` (and to weight by `n`), so `H0` is now correct for delayed-entry /
  start-stop data. `SemiParametricRegressionModel.predict_tvc` gives the
  survival of a subject along a supplied covariate path,
  `H(t) = Σ_{u_j ≤ t} h0(u_j) exp(β'Z(u_j))`, reducing to `sf(t, Z)` for a
  constant covariate.
- **Time-varying covariates for the parametric families — future work.** Each
  interval is a left-truncated observation; difficulty varies by family:
  - Additive hazards (Lin-Ying): easiest — `H(t) = H₀(t) + β'·∫Z(s)ds`
    accumulates linearly; interval contributions are just `β'·Z·Δt`.
  - Parametric PH: medium — cumulative hazard is additive over intervals, so
    segments compose cleanly (as in the Cox case above).
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

### Covariates / accelerated degradation testing (design)

Covariates (accelerating stresses such as temperature/voltage/humidity, or
observational factors such as lot/supplier/load) can enter at two different
stages, and they are genuinely different features:

**Stage 1 — covariates on the life fit (classic ADT).** *Shipped.* The
per-unit path fits are unchanged; each unit carries a constant covariate vector
`Z_i`. Passing `Z` to `DegradationAnalysis.fit(x, y, i, Z=..., distribution=...)`
feeds `(pseudo_failure_time_i, c_i, Z_i)` into the univariate regression fitters
instead of a plain distribution — a plain `distribution` (e.g. `Weibull`) is
auto-wrapped in `AFT(distribution)`, while an explicit regression fitter (e.g.
`AFT(LogNormal)`, `WeibullPH`, `CoxPH`) is used directly. `Z` is aligned to the
measurement arrays and validated constant within each unit, then reduced to one
row per unit. The delegated lifetime functions (`sf`/`ff`/`df`/`hf`/`Hf`/`qf`/
`mean`/`random`) take the stress vector `Z` at which to evaluate life, so life at
use conditions is one call; `qf`/`mean` come from inverting/integrating the
regression survival function (the fitters expose no closed `qf`/`mean`), and
`random` uses inverse-transform sampling. First-stage regression confidence
bounds are available via `life_model.cb(x, Z, ...)`; the two-stage
`DegradationModel.cb` / `life_parameter_covariance` raise `NotImplementedError`
for covariate models (the generated-regressor correction is not yet derived for
the regression life fit). `fit_from_df` gains `Z_cols`. Theoretical
justification: for a linear path with stress acting on the rate, `T = (y_t −
a)/b(Z)`, so `log T = log(y_t − a) − log b(Z)` — exactly an AFT model, and the
fitted AFT coefficient recovers the simulated stress coefficient. Remaining open
point: two-stage bounds for the covariate life fit, and interaction with
`path="best"` (path stage stays per-unit, selected on pooled data).

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
with zero-inflation at 0) and Tier 2 are done. Extensions:

- **Tier 2 — done.** A standalone `Poisson` distribution on `{0, 1, 2, …}`
  (count data; distinct from the recurrent Poisson *processes*);
  `BetaGeometric` (discrete-time frailty — Geometric with Beta-mixed `p`, whose
  marginal hazard decreases with time); and a general `Discretize(distribution)`
  wrapper that turns any non-negative continuous distribution into its
  integer-binned counterpart (`K = ⌈T⌉`, so `P(K=k) = F(k) − F(k−1)` and
  `R_K(k) = R(k)`), yielding a discrete Gamma / Log-Normal / Weibull etc. fit by
  MLE on the underlying parameters.
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
