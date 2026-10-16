# Decisions

Design decisions made while working on the open issues, with the reason for each and where it stands. Newest first.

**Who decided:**
- **maintainer:** chosen in review.
- **principle:** settled by a numbered rule in `docs/Design Principles.rst`.
- **proposed:** made during implementation and not yet confirmed.

**Before asking the maintainer, check the principles.** If one settles the question, apply it and record it here.

---

## 2026-10-16

### Diagnostics: `diagnose()` (#867–#875, #881)

| Decision | By | Reason | Status |
|---|---|---|---|
| The API is as designed in #867: `table`, `advice`, `plot()`, `ok`, `raise_if_failed()`, `to_dict()`, one registry of families. It can change later if needed. | maintainer | — | on branch `diag-base-wave2` |
| `check_ph()` computes the exact `cox.zph` score test (R's survival 3.0 and later). The old approximation is removed, and every place in the package that runs, quotes or tests the PH test agrees. | maintainer | "Use the better version. Make sure the package is consistent." The old values matched lifelines and pre-3.0 R, not current R. | on branch |
| The signature is `diagnose(alpha, columns, *, n_boot, random_state, **family_options)`. | principle 19, 21, 24 | 19: every method that draws takes `random_state`. 21: `n_boot` is the standard name. 24: family options (`use_stress=`, `t0=`) default to the simple behaviour. | on branch; recurrent events' fixed 99 refits with seed 0 must switch to these |
| Rows that choose between models, where the test rejecting is the good outcome, are screens: ok or warn, never fail. These are the recurrent trend and repair rows and the frailty variance row. | principle (framework rule: screens set ok/warn) | They are model-choice screens, not tests of an assumption. | to be documented once in the Verdicts section |
| Competing risks runs the Cox checks per cause, but not calibration. | proposed | A cause's exp(−H_k) is not the probability of being free of that cause when other causes compete, so there is nothing to calibrate it against. | on branch |
| Docs data typed from memory (Nelson's insulating fluid) must be checked against a published source, or replaced with simulated data. | principle 16, 23 | Quoted data and numbers must be checked. | to do |

### Partial-effects plots (#876)

| Decision | By | Reason | Status |
|---|---|---|---|
| Hold the other covariates at their means, including 0/1 indicators; a formula's categorical variable at its reference level; `reference=` overrides. | maintainer | Consistent with how the regressions centre their covariates. | worktree branch, not merged |

### Joint models (#891 v1)

| Decision | By | Reason | Status |
|---|---|---|---|
| Use the data classes' language: `markers={age: value}` on `ItemLifetime`, `fit(data)`, and prediction from a record, `sf(x, item)`. No new words such as "readings", `link=` or `path_covariates`. | maintainer | — | worktree branch |
| `JointPH(dist, path_model)` takes any degradation `PathModel`. `WeibullJointPH = JointPH(Weibull, LinearPath)`. Parameter names come from the parts: `alpha`, `beta`, `marker_effect`; the path's own `a`, `b`, `c`; `a_<cov>`; `sd_a`, `corr_a_b`; `sd_noise`. | maintainer | The marker function has its own names, as a life model does. | worktree branch |
| v1 records hold one plain value per age, and only on `ItemLifetime`. | maintainer | — | worktree branch |
| Keep: `marker_effect` as the link's name; the array fit route; `unit=`; reshapes trim readings to the observation window; exact conditioning on entry for late entry; Wald-only `cb`; fixed quadrature settings; joining the shared conformance checks. | maintainer | — | worktree branch |
| Centre the marker at the mean reading by default; `marker_reference=` sets a round value; `summary()` says which value `alpha` is at. | maintainer | Only `alpha`'s meaning changes. | worktree branch |
| Array prediction `sf(x, Z=, markers=, given=)`. `sf(x, item)` builds exactly those arguments. With no readings and no `given`, it gives the new-unit (population) prediction. | maintainer | — | worktree branch |
| Every path parameter varies between units by default; `varying=` chooses which. | maintainer | — | worktree branch |
| Units differ on each parameter's own scale by default; `LinkedPathModel` (e.g. `{"b": "log"}`) gives a log scale. | principle 15, 24 | The same default as the degradation module; the alternative is an option. | to check that `LinkedPathModel` works as a path |
| Keep `fit_from_df`. | principle 14, 21 | Every fitter has one, and the entry points agree. | worktree branch |
| Run a 200-fleet recovery study per path model before merging. | principle 17 | Intervals must achieve their stated coverage; 24 fleets was too few to tell. | to do |

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
