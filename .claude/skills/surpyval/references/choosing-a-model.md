# Choosing a model: the yes/no flow

Follow these questions in order, one at a time. Each has exactly two
answers; each answer says where to go next or which call to make. Ask the
user when the data does not answer a question; do not guess a branch. The
same flow, for people, is the docs page `docs/Choosing a Model.rst`.

SurPyval's conventions apply throughout: `c = 0` observed, `c = 1` right
censored (the opposite of R's `status`, so data from R needs `c = 1 -
status`), `c = -1` left censored, `c = 2` interval censored with `x = [lower,
upper]`; `n` counts identical rows; `tl` / `tr` truncate.

## Stage 1: the shape of the problem

1. **Can one unit have the event more than once?** (repaired and back in
   service; readmitted)
   - Yes → **recurrent events**: rows carry the unit id `i`, `x` is the age at
     each event and the unit's last row (`c = 1`) is the end of watching.
     `surpyval.recurrent`: `NonParametricCounting` (MCF), `CrowAMSAA` (NHPP),
     `GeneralizedRenewal` / `GeneralizedOneRenewal` (repair quality). See
     "The xicn data model" in `SKILL.md`. Stop.
   - No → question 2.
2. **Can the event be one of several different kinds?**
   - Yes → question 3.
   - No → question 4.
3. **Does the first event end the unit's record?**
   - Yes → **competing risks**: `sp.CompetingRisks.fit(x, e=e, c=c)` (cumulative
     incidence), `sp.FineGray.fit(x, Z, e, c=c)` (regression). Never a
     Kaplan-Meier of one kind with the others censored. Stop.
   - No (units move between states) → **multi-state**: `sp.MultiState.fit`.
     Stop.
4. **Are there condition readings over time?** (wear, crack length,
   vibration, a biomarker)
   - Yes → question 5.
   - No → one event per unit: stage 2 (question 6).
5. **Are there also failure times, which the readings may drive?**
   - Yes → **joint model**: `sp.WeibullJointPH.fit(...)`. Stop.
   - No → **degradation**: `surpyval.degradation.DegradationAnalysis.fit(x, y,
     i, threshold=...)`. Stop.

## Stage 2: writing each unit's record

6. **Did the unit have the event while it was watched?**
   - Yes → question 7.
   - No → right censored: `c = 1`, `x` = last time seen working. Question 9.
7. **Is the exact time of the event known?**
   - Yes → observed: `c = 0`, `x` = that time. Question 9.
   - No → question 8.
8. **Did the event happen before the first look?**
   - Yes → left censored: `c = -1`, `x` = time of the first look. Question 9.
   - No → interval censored: `c = 2`, `x = [last seen working, found
     failed]`. Question 9.
9. **Do several units have exactly the same record?**
   - Yes → one row with `n` = how many. Question 10.
   - No → question 10 (`n` defaults to 1).
10. **Is the unit in the data only because it had already lasted to some age?**
    (installed before monitoring; enrolled after diagnosis)
    - Yes → left truncated: `tl` = that age. Question 11.
    - No → question 11.
11. **Is the unit in the data only because it failed by some time?**
    (warranty returns only)
    - Yes → right truncated: `tr` = that time. Record done.
    - No → record done.

When every unit has a record, go to stage 3.

## Stage 3: the question

12. **Is this planning a test before there is data?**
    - Yes → `surpyval.test_planning` (e.g. `demonstration_sample_size`,
      `demonstration_plan`, `life_test_sample_size`). Stop.
    - No → question 13.
13. **Do the units differ in something that may change their life?**
    (treatment, stress, design)
    - Yes → regression with covariates `Z` (one row per row of `x`).
      Question 14.
    - No → question 17.
14. **Is the covariate a stress that was set (temperature, voltage), to be
    extrapolated to use conditions?**
    - Yes → **accelerated life**: `sp.AcceleratedLife(sp.Weibull,
      sp.life_models.Arrhenius).fit(x, Z, c)` (temperature in kelvin; also
      `Eyring`, `Power`, ... in `sp.life_models`). Stop.
    - No → question 15.
15. **Should a life distribution be assumed?**
    - Yes → question 16.
    - No → **Cox**: `sp.CoxPH.fit(x, Z, c=c)`. Stop.
16. **Does the covariate stretch or shrink time, rather than scale the risk?**
    - Yes → **AFT**: `sp.WeibullAFT.fit(x, Z, c)`. Stop.
    - No → **PH**: `sp.WeibullPH.fit(x, Z, c)`. Stop.
17. **Are answers needed beyond the longest observed time?** (extrapolation)
    - Yes → parametric. Question 20.
    - No → question 18.
18. **Is a mean life, a B-life below the first failure, or a smooth curve to
    predict from needed?**
    - Yes → parametric. Question 20.
    - No → non-parametric. Question 19.
19. **Is any record interval or left censored, or any unit right truncated?**
    - Yes → `sp.Turnbull.fit(x, c=c, n=n, tl=tl, tr=tr)`. Stop.
    - No → `sp.KaplanMeier.fit(x, c=c, n=n, tl=tl)` (or `sp.NelsonAalen`).
      Stop.
20. **Has at least one unit failed?**
    - Yes → fit `sp.Weibull.fit(x, c, n, ...)` first. Question 21.
    - No → no fit exists; give a bound: `sp.weibayes(x, c, beta=...)` with an
      assumed Weibull shape, or `sp.success_run(n)` with none. Stop.
21. **Will some units never fail?**
    - Yes → add `lfp=True`. Question 22.
    - No → question 22.
22. **Do some units fail at time zero?**
    - Yes → add `zi=True`. Question 23.
    - No → question 23.
23. **Is there an age before which nothing fails?**
    - Yes → add `offset=True`. Question 24.
    - No → question 24.
24. **Do the failures look like two modes mixed together?**
    - Yes → `sp.MixtureModel.fit(x, c=c, dist=sp.Weibull, m=2)`. Then check.
    - No → `sp.fit_best(x, c, ...)` (plain candidates, lowest AIC); with
      `lfp`/`zi`/`offset`, fit each candidate with the same options and
      compare `aic()`. Then check.

## Parametric or non-parametric?

- Non-parametric (`KaplanMeier`, `NelsonAalen`, `Turnbull`) describes only
  what was seen and stops at the longest observation; its mean is only the
  area up to that point. `mean()` gives that area and warns when units are
  still at risk there; report it as `rmst(tau)`, the restricted mean.
- Parametric assumes a shape, and so can extrapolate (a mean life, a B1 life,
  survival at a time no unit reached).
- Check the shape against the non-parametric estimate with `diagnose()`.
- The further the extrapolation, the more the answer rests on the assumed
  shape. SurPyval does not warn about it, by design: say so to the user when
  an answer is an extrapolation, and how far.

```python
import numpy as np
import surpyval as sp

x = np.array([17.88, 28.92, 33, 41.52, 42.12, 45.6, 48.4, 51.84, 51.96,
              54.12, 55.56, 67.8, 68.64, 68.64, 68.88, 84.12, 93.12, 98.64,
              105.12, 105.84, 127.92, 128.04, 173.4])
c = (x > 100).astype(int)          # 1 = still running at 100
x = np.minimum(x, 100)

km = sp.KaplanMeier.fit(x, c)      # question 19, No
weibull = sp.Weibull.fit(x, c)     # question 20, Yes
km.rmst(100)["rmst"], weibull.rmst(100)["rmst"]   # agree inside the data
weibull.sf(150)                    # an extrapolation: only the Weibull answers
```

## Check the answer

- `model.diagnose(random_state=...)` on every parametric and regression
  model: goodness of fit against the non-parametric estimate, and the
  alternatives it suggests (another distribution, an offset, an lfp).
- Bounds on whatever is reported: `cb` (a function of time, `on="sf"` by
  default), `quantile_cb` (a B-life), `param_cb` (a parameter). The defaults
  are two-sided 95% Wald bounds (`alpha_ci=0.05`, `bound="two-sided"`,
  `method="wald"`). **With few failures (under about 20), use
  `method="lr"`** (advice; the default stays Wald) and treat a choice between
  distributions as weak. `bound=` is the side (`"two-sided"`, `"upper"`,
  `"lower"`), not the method.
- Compare models only on the same data.

```python
report = weibull.diagnose(random_state=1)
report.ok
weibull.cb(50, method="lr")
weibull.quantile_cb(0.1, method="lr")

# question 20, No: ten units ran 500 hours with no failure, shape 2 assumed
bound = sp.weibayes([500] * 10, c=[1] * 10, beta=2)
assert round(bound.sf(500), 3) == round(sp.success_run(10), 3)

# question 12: units to demonstrate R = 0.9 with no failures
from surpyval import test_planning
test_planning.demonstration_sample_size(0.9)   # 29
```
