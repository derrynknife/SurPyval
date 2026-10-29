# Decisions

Design decisions made while working on the open issues, with the reason for each and where it stands. Newest first.

**Who decided:**
- **maintainer:** chosen in review.
- **principle:** settled by a numbered rule in `docs/Design Principles.rst`.
- **proposed:** made during implementation and not yet confirmed.

**Before asking the maintainer, check the principles.** If one settles the question, apply it and record it here.

---

## 2026-10-29

### Bayesian fitting of the parametric distributions

| Decision | By | Reason | Status |
|---|---|---|---|
| SurPyval fits by Bayes' rule with engines of its own: closed forms where they exist (Exponential with a Gamma prior, a Normal's mean with its spread known, Bernoulli with a Beta prior), else a self-checking grid for one to three free parameters, with a normal (Laplace) approximation as an option. No probabilistic-programming library is a dependency, required or optional, and `fit_bayes` has no method that calls one; a sampler of SurPyval's own, for regression and pooling, is later work. | maintainer | Deterministic answers with no seed, no new dependency, and fast enough to run inline. | done |
| The entry points are `Distribution.fit_bayes(...)`, `fit_bayes(data)` with a `SurpyvalData`, and `fit_bayes_from_df(...)`; `fit` is not changed. All the code is in `surpyval/bayes/`; the distributions' base class has two delegating methods. | maintainer | Additive: existing fits are unaffected. | done |
| A Bayesian model's `sf`, `ff`, `df`, `hf`, `Hf`, `qf`, `mean` and `random` are the posterior predictive (a new unit's life with the parameter uncertainty averaged in); `params` are the posterior medians, and `at_params()` is the plug-in model there. | maintainer | A proper distribution, so every identity holds and anything that takes a model takes it. | done |
| A parameter with no prior is flat on its unconstrained scale (log for a positive parameter, logit for a proportion), allowed only where the data alone determine it: refused with no failure information; with every parameter flat, refused unless the maximum-likelihood fit's maximum is verified; and refused whenever the grid finds the posterior does not fall away (it has no finite mass). | maintainer | Never invents information; a reference posterior in one line. | done |
| Credible intervals are `cri(x, on=...)`, `quantile_cri(p)`, `mean_cri()` and `param_cri(name)`, with `alpha_ci` and `bound=` as everywhere; `cb`, `quantile_cb`, `mean_cb` and `param_cb` refuse and point to them. | maintainer | Different questions, different names. | done |
| Priors given on a parameter (`around`, `between`, any SurPyval distribution), on several (`MultivariateNormal`, `from_model`), on quantities (`on_quantities`: B-lives, `R_at`, the mean, the median, parameters by name) and from old data (`from_data`, a power prior). `around(m, between=(a, b))` is normal on the unconstrained scale with its ends at the 5% and 95% points (90% by `mass=`); its median is the interval's middle on that scale, and the summary says what was built. A prior's support must lie in its parameter's range (refused, or `truncate=True`). | maintainer (the worked examples) | — | done |
| `on_quantities` takes exactly one statement per free parameter and carries the quantities' density to the parameters by the change of variables (the Jacobian by central differences at every grid point). More statements than parameters (matching them as well as possible) is not built. | proposed | Exact where it is defined; matching over-determined statements is a separate piece of elicitation. | done |
| `from_model(model)` of a maximum-likelihood fit is a normal on the unconstrained scale with the fit's covariance carried there, divided by `weight` and widened by `widen` squared; refused where the fit's maximum is not verified. `from_model` of a Bayesian fit is its priors times its data's likelihood raised to `weight` (exact); with `param=` or `widen` it is a normal fitted to the posterior, and says so. | proposed | Exact where the data are at hand. | done |
| `update(x, c, ...)` refits with the original prior and the old and new data together (exact); `replace=True` takes the new records in place of the old. | maintainer | Updating in steps equals fitting at once. | done |
| `compare_to(other, on="qf", p=0.10)` is the posterior probability that this model's quantity exceeds the other's: two Bayesian fits as independent posteriors; a maximum-likelihood fit's estimate as normal on the log scale (natural scale for a probability) with its delta-method standard error, independent of this one. The result states which. | maintainer (the worked example) | — | done |
| The grid: on the unconstrained scale along the curvature's axes at the peak, stretched (`z = 3 sinh(u / 3)`) so cells are fine near the peak and coarse in long tails, widened until every face is below 1e-8 of the peak, and a second grid offset by half a cell. A summary's quantiles spread each point's mass over its cell and remove the spread's own bias by extrapolation from a two-cell spread. The error estimate is half the difference of the two grids' 2.5%, 50% and 97.5% points of every free parameter and the 10% quantile, as a fraction of the 95% interval's width; it must be at most 0.1% with one or two free parameters and 0.5% with three (refined up to 241 and 61 points a dimension; past that the fit refuses with the numbers). | proposed | 0.1% at three parameters needs grids too large to stay near a few seconds; the error is always reported. | done |
| Where the data disagree with a proper prior on one parameter (the posterior median outside the prior's 99% interval), the fit warns with both intervals and sides with neither. | proposed | Design principle 22. | done |
| `mean()` is infinite, with a warning, where parameter values with an infinite mean carry more than 1e-12 of the posterior; below that they are below what the grid resolves. | proposed | — | done |
| Conformance: registered as `Weibull[bayes]` (the grid) and `Exponential[bayes]` (the closed form, with its prior's time scale rescaled with the data). Not applicable, with reasons in the registry: `maximum` (not a maximisation), `convergence` (no optimiser to starve; the grid's own refusal is tested), and containment of the predictive hazard in `cri(on="hf")` (the predictive hazard is not the posterior mean of the hazard; checked against the posterior median instead). The intervals are not Wald (they close onto the posterior median). | proposed | Principle 18 as it applies to a posterior. | done |
| Validation (rules fixed before the first run): the vectorised likelihood equals the fitters' to 1e-10 for every continuous distribution and kind of data; the closed forms are checked in exact arithmetic and the grid matches them to 1e-4; update in steps and a power prior at weight 1 equal the fit of all the data (to the grid's tolerance where the start differs); with flat priors and 1,000 complete Weibull values the posterior median is within 0.02 standard errors of the maximum and the 95% interval within 2% of the likelihood-ratio interval; on a hard posterior (2 failures, 98 suspensions) the reported error is within a factor of 2 of the error against a 4x finer reference. | maintainer (the design) | — | done |
| Results of those rules, as found: the likelihood identity holds to 1e-16; the closed forms are exact; the grid matches them to 1e-4; update and the power prior agree; the peak is the maximum to 1e-6 and the 95% intervals are within 2% of the likelihood-ratio intervals; the reported error on the hard posterior is 2.7e-4 against an actual 4.7e-4 (within the factor of 2). One rule fails and is kept as a strict expected failure: beta's posterior median is 0.037 standard errors from the maximum at n = 1000 (the rule asks 0.02); the difference is the posterior's skewness, which shrinks as 1 / sqrt(n), so the rule was too tight for 1,000 values. | proposed | Reported, not loosened. | done |
| Simulation-based calibration (Talts et al. 2018), fixed before its first run: 500 replicates per scenario; the true parameters drawn from the prior and the data simulated with seed 8100 + r, the 999 posterior draws with seed 8600 + r; the rank of the truth among the draws for each parameter and the B10 life, in 20 bins, must pass a chi-square test at the 0.001 level (a Bonferroni note: about 30 tests), and the 95% interval's coverage must be within 0.02 of 0.95; a replicate whose fit refuses fails its scenario. Scenarios: Weibull with 20 complete values (weakly informative and informative priors), with 30% and 80% Type I censoring, interval inspections, left truncation and about 3 failures in 40; LogNormal, Gamma and LogLogistic with 30% censoring; Exponential with a Gamma prior and 30% censoring. | proposed | Principle 17's coverage studies, for a posterior. | in the validation pull request |

---

## 2026-10-17

### Joint models: separate covariates, interval data, dynamic prediction (#891)

| Decision | By | Reason | Status |
|---|---|---|---|
| Separate covariates for the hazard and the path are chosen by name from the one covariate set the records carry: `fit(..., hazard_covariates=[...], path_covariates=[...] or {path parameter: [...]})`, default every covariate in both (as v1). Not a second covariate array (`path_Z`, as the closed #966 had). This revisits v1's "no new words such as `path_covariates`": the capability was asked for, and these two arguments say what they do; the alternative was one mapping, `covariates={"site": "hazard", "load": ["b"]}`. | maintainer | An `ItemLifetime` has one `covariates`, so a second array would need a second place on the record; naming the columns keeps `fit(data)`, arrays and `fit_from_df` alike. The maintainer kept the names `hazard_covariates` and `path_covariates` (2026-10-17). | done |
| A covariate in neither part is refused; one in only some parts has no parameters elsewhere (no `nan`, no 0 entries): `b_load` without `a_load` or `load`. | proposed | A coefficient that is not estimated is not a parameter. | done |
| Saved models write `path_covariates` and `hazard_covariates` (`None` where every covariate is in both); a dict without them loads as every covariate in both. | principle 21 | Saved models keep working. | done |
| Interval and left censoring: `S(l) - S(u)` over the item's paths, the hazard integrated over `(l, u]` directly (not a difference of two cumulative hazards); a left censored item's interval starts at its entry. Right truncation divides by the population's `S(tl) - S(tr)`, as left truncation already divided by `S(tl)`. | proposed | The same conditioning v1 chose for late entry; the direct integral keeps a narrow late interval accurate. | done |
| One item's dynamic prediction is an object, `JointModel.predict(...)` -> `DynamicPrediction`, beside the array prediction `sf/ff/cb(x, Z, markers=, given=)`, which it uses and which is unchanged. Its `cb` is the model's (Wald on `log(-log sf)`); its `qf` brackets on a grid and refines by Newton steps on `log sf` (derivative `-hf`). | proposed | The object answers the one-asset questions (hazard, remaining life, smoothed marker, draws) without a second implementation. | done |
| `DynamicPrediction.to_dict(x=None, alpha_ci=0.05)` is strict JSON for a report or chart: the readings as `[age, value]` pairs, a grid from `age` to the 0.99 quantile (0.9, or ten scales on, where the survival stops short of it), `sf`/`ff`/`hf` with the two-sided Wald bounds on `sf`, remaining-life quantiles at 0.1, 0.5 and 0.9, and the marker's path with its interval from the first reading; non-finite values are `None` with the package's `"non_finite"` record. | proposed | The same convention as every other `to_dict`, so a reader can restore the values. | done |
| The calibration study's seeds (7300 + r, 7600 + r) and its rules (`check_bias` defaults, coverage slack 0.02) were fixed before its first run. | principle 17 | — | done |

## 2026-10-16

### Data classes: usability (#931–#940, #945; tracking #953)

| Decision | By | Reason | Status |
|---|---|---|---|
| `from_dataframe` names its columns `x_col`, `c_col`, `n_col`, `xl_col`, `xr_col`, `tl_col`, `tr_col`, `i_col`, `e_col`, `Z_cols` (the `fit` argument with `_col`), not `id=`, `time=`, `event=` as #936 sketched; a status column (1 = failed) is `event_col`. | principle 21 | Every DataFrame entry point names a column argument after the fit argument it fills. | done |
| `print(data)` is the repr (the size and a table of the first rows), as for an `EventHistoryData` or a DataFrame; `summary()` stays a method, not `__str__`. | proposed | `print(data)` shows the data; the docs (Conventions) rely on it. #934 suggested `__str__ = summary()`. | done |
| The likelihoods' split arrays (`x_o`, `mask_i`, `Z_t`, ...) and the recurrent likelihood pieces keep their names and stay public, left out of `dir()`; no underscore rename with deprecated aliases. | proposed | Some 50 names used across the fitters and the diagnostics; principle 21's one-release deprecation for each is more churn than the clutter costs. #934 suggested the rename. | done |
| `summary()` counts items (rows weighted by `n`) everywhere, and the heading reads "4 items, 3 rows" for both datasets. | proposed | #934: the counts disagreed between lines. | done |
| `for_event_type(["seal", "bearing"])` takes the types as one mode rather than refusing a list. | proposed | #932 allowed either; merging is what the list means. | done |
| `censor_at` / `enter_at` refuse a cut-off at or before 0 only where every time and truncation bound of the data is at or above 0; data on the real line (a Normal baseline) takes a negative cut-off. | proposed | A negative age is a silent error for ages from 0, and a valid time on the real line. | done |
| `select(ids=)` on data without ids is refused (rows are sorted and grouped, so their numbers would not be stable); `item(k)` by row number stays; `SurpyvalData(..., ids=)` gives rows ids. | proposed | #932's recommendation. | done |
| `fit(data)` takes the data's unit; `fit(data, unit=other)` is refused, as are records in different units in one `from_items`; nothing is converted (`to_unit` left for later). | #945 as written | — | done |
| `groupby(name)` refuses items without the attribute (naming them); `select(where={name: value})` leaves them out. | proposed | A group of "none" is a silent default; a selection by value is not. | done |
| A `SurpyvalData` or `EventHistoryData` dict is tagged `"class"` and read by `surpyval.from_dict` / `from_json` by that tag, as a model is by `"model"`. | #935 as written | — | done |
| `predict(..., given="age")` conditions on each running item's age by the ratio `S(t) / S(age)` from the model's own functions, for every lifetime model alike; a failed item gets `nan`; `qf` and `mean` are not conditioned. | proposed | With fixed covariates the ratio is the conditional survival under every model (the AFT subtlety of #962 is for covariates that change). | done |
| `EventHistoryData.from_dataframe` builds each item's `ItemHistory` (rather than arrays), so every check names the item; an item in `ends=` with no row cannot carry covariates and is refused with `Z_cols`. | proposed | — | done |
| `fit_from_surpyval_data` keeps accepting covariates it does not fit. | proposed | The regression fitters fit through it with the data's `Z`; refusing broke them. `fit(data)` refuses, as before. | done |

### Branches left behind

| Decision | By | Reason | Status |
|---|---|---|---|
| Test planning (#862–#865, #880), built on branch `claude/happy-einstein-9bpmln` on 14 Oct but never opened as a PR, is merged. | maintainer | Test planning is statistics, so it belongs in SurPyval. | merged (#965) |
| Leave `claude/roadmap-live-npv` (a ROADMAP section) and PR #858 (Reliafy links) unmerged. | maintainer | No code in either. | left as is |

### Diagnostics: `diagnose()` (#867–#875, #881)

| Decision | By | Reason | Status |
|---|---|---|---|
| The API is as designed in #867: `table`, `advice`, `plot()`, `ok`, `raise_if_failed()`, `to_dict()`, one registry of families. It can change later if needed. | maintainer | — | merged with the diagnostics PR |
| `check_ph()` computes the exact `cox.zph` score test (R's survival 3.0 and later). The old approximation is removed, and every place in the package that runs, quotes or tests the PH test agrees. | maintainer | "Use the better version. Make sure the package is consistent." The old values matched lifelines and pre-3.0 R, not current R. | merged with the diagnostics PR |
| The signature is `diagnose(alpha, columns, *, n_boot, random_state, **family_options)`. | principle 19, 21, 24 | 19: every method that draws takes `random_state`. 21: `n_boot` is the standard name. 24: family options (`use_stress=`, `t0=`) default to the simple behaviour. | done; recurrent events takes them too |
| Rows that choose between models, where the test rejecting is the good outcome, are screens: ok or warn, never fail. These are the recurrent trend and repair rows and the frailty variance row. | principle (framework rule: screens set ok/warn) | They are model-choice screens, not tests of an assumption. | done (Verdicts section) |
| Competing risks runs the Cox checks per cause, but not calibration. | proposed | A cause's exp(−H_k) is not the probability of being free of that cause when other causes compete, so there is nothing to calibrate it against. | merged with the diagnostics PR |
| Docs data typed from memory (Nelson's insulating fluid) must be checked against a published source, or replaced with simulated data. | principle 16, 23 | Quoted data and numbers must be checked. | done: checked against R's `Sleuth3::case0802`; one typo fixed (100.58 → 100.59) |
| The accelerated-life advice to "plan more stress levels" names `surpyval.test_planning`'s accelerated life test planning (#864), not "simulate a plan with `model.random`". | proposed | Test planning is now in SurPyval. | done |

### Partial-effects plots (#876)

| Decision | By | Reason | Status |
|---|---|---|---|
| Hold the other covariates at their means, including 0/1 indicators; a formula's categorical variable at its reference level; `reference=` overrides. | maintainer | Consistent with how the regressions centre their covariates. | merged with the partial-effects PR |

### Joint models (#891 v1)

| Decision | By | Reason | Status |
|---|---|---|---|
| Use the data classes' language: `markers={age: value}` on `ItemLifetime`, `fit(data)`, and prediction from a record, `sf(x, item)`. No new words such as "readings", `link=` or `path_covariates`. | maintainer | — | merged (joint-model PR) |
| `JointPH(dist, path_model)` takes any degradation `PathModel`. `WeibullJointPH = JointPH(Weibull, LinearPath)`. Parameter names come from the parts: `alpha`, `beta`, `marker_effect`; the path's own `a`, `b`, `c`; `a_<cov>`; `sd_a`, `corr_a_b`; `sd_noise`. | maintainer | The marker function has its own names, as a life model does. | merged (joint-model PR) |
| v1 records hold one plain value per age, and only on `ItemLifetime`. | maintainer | — | merged (joint-model PR) |
| Keep: `marker_effect` as the link's name; the array fit route; `unit=`; reshapes trim readings to the observation window; exact conditioning on entry for late entry; Wald-only `cb`; fixed quadrature settings; joining the shared conformance checks. | maintainer | — | merged (joint-model PR) |
| Centre the marker at the mean reading by default; `marker_reference=` sets a round value; `summary()` says which value `alpha` is at. | maintainer | Only `alpha`'s meaning changes. | merged (joint-model PR) |
| Array prediction `sf(x, Z=, markers=, given=)`. `sf(x, item)` builds exactly those arguments. With no readings and no `given`, it gives the new-unit (population) prediction. | maintainer | — | merged (joint-model PR) |
| Every path parameter varies between units by default; `varying=` chooses which. | maintainer | — | merged (joint-model PR) |
| Units differ on each parameter's own scale by default; `LinkedPathModel` (e.g. `{"b": "log"}`) gives a log scale. | principle 15, 24 | The same default as the degradation module; the alternative is an option. | done; two `LinkedPathModel` bugs fixed on the way |
| Keep `fit_from_df`. | principle 14, 21 | Every fitter has one, and the entry points agree. | merged (joint-model PR) |
| Run a 200-fleet recovery study per path model before merging. | principle 17 | Intervals must achieve their stated coverage; 24 fleets was too few to tell. | done: `scripts/studies/joint_recovery` |

### `fit_from_recurrent_data` refusals (#959)

| Decision | By | Reason | Status |
|---|---|---|---|
| Refuse what `fit(data)` refuses, in one shared check. What a fitter takes is read from its `fit` signature. | maintainer | One source of truth, so the two routes cannot disagree. | merged (#964) |

### Survival given an age (#962)

| Decision | By | Reason | Status |
|---|---|---|---|
| Under PH, PO, additive hazards and Cox, `given=g` needs the covariates from `g` only. AFT and accelerated life still need them from 0. | maintainer (option A) | The part before `g` divides out of S(x)/S(g) under these models, but not under AFT, whose age at `g` depends on it. This matches what fitting already allows for late entry. | merged (#964) |

---

## 2026-10-15

### Working method

| Decision | By |
|---|---|
| One issue at a time. Explain it with a minimal example, then the maintainer decides, then build. | maintainer |
| Before merging, show before/after results on the real code, and list anything done beyond what was agreed. | maintainer |
| A long plan is a map, not a commitment; re-plan after each issue. | maintainer |

### AFT on baselines defined below 0 (Normal, Gumbel, Logistic)

| Decision | By | Reason | Status |
|---|---|---|---|
| Keep these AFT models. AFT means S(t \| X) = S0(φ(X)·t), which rescales time about 0. No warning and no deprecation. | maintainer | By that definition everything follows: dependence on where 0 is, the sign flip before 0, the clock anchored at 0. | done |
| AFT and ALT are distinct. A model whose covariates shift the location is an accelerated life model, e.g. `AcceleratedLife(Normal, life_models.Linear)`. R's gaussian `survreg` "AFT" is that model. | maintainer | SurPyval's naming follows the textbook definition, not R's. | done (guide note) |
| A `GeneralLinear` life model (μ = a + b'X). | maintainer | So the location model can take several covariates. | #963 open |
| A time before 0 needs the covariate from that time to 0. A faster-ageing unit looks safer before 0: "it is what it is". | maintainer | It follows from the anchoring at 0. | done |

### Covariates must cover every time the answer depends on (#930, #952, #957)

| Decision | By | Reason | Status |
|---|---|---|---|
| Refuse a schedule or path that starts after the time origin, rather than holding its first value back. This is a breaking change. | maintainer | "This is a silent error." | merged (#951) |
| On baselines defined below 0, PH and PO need the covariates from −∞. AFT and AH are anchored at 0. In `fit_tvc`, a first `xl = 0` means the unit entered at 0, and `-inf` means followed since −∞. | maintainer | The model's maths depends on the whole stretch. | merged (#956) |
| Integrate a `CovariatePath` below 0, and from −∞ for PH and PO. | maintainer | — | merged (#958) |
