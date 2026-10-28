# Decisions

Design decisions made while working on the open issues, with the reason for each and where it stands. Newest first.

**Who decided:**
- **maintainer:** chosen in review.
- **principle:** settled by a numbered rule in `docs/Design Principles.rst`.
- **proposed:** made during implementation and not yet confirmed.

**Before asking the maintainer, check the principles.** If one settles the question, apply it and record it here.

---

## 2026-10-29

### Robust standard errors on the parametric regressions

| Decision | By | Reason | Status |
|---|---|---|---|
| The names are the Cox model's: `robust_covariance(cluster=None)`, `robust_standard_errors(cluster=None)` and `summary(robust=True, cluster=...)`; not a `robust=` argument to `fit` (lifelines' form). | principle 21 | The same option has the same name everywhere, and the fit is the same either way. | proposed |
| HC0, no small-sample factor; counts are frequency weights (a row with `n = 3` is three units, each its own cluster without `cluster`, three members of its cluster with it). | principle 5, 16 | R's `sandwich` and lifelines use HC0; counts must equal repetition, as `CoxPH.robust_covariance` already reads them. | proposed |
| `cluster` is one label per row given to the fit, in that order; rows dropped for a missing covariate are matched and left out (the fit records which given row each fitted row is). | principle 3 | A label belongs to its row; the fit sorts and drops rows. | proposed |
| Only the standard errors and the summary change; `cb`, `param_cb` and `p_values` keep the model-based covariance. A `fit_tvc` model refuses (its likelihood is over histories, not rows), naming the bootstrap and likelihood-ratio bounds. | principle 24 | Simple by default; the robust form is an option beside it. | proposed |

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
