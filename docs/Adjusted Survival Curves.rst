Adjusted Survival Curves
========================

Two groups are compared: two suppliers, two designs, treated and untreated
patients. Nobody assigned the groups at random, so they differ in other
ways as well: one supplier's parts went mostly to the heavy-duty sites, the
treated patients were older. The plain Kaplan-Meier curve of each group
then shows two things at once, the effect of the group and the effect of
everything else that differs between the groups, and cannot tell them
apart.

An **adjusted survival curve** answers a sharper question:

    *What would the survival curve be if every unit in the sample had been
    in this group?*

Each group's adjusted curve is about the same set of units, with the same
mix of duty, age and so on. A difference between two adjusted curves is
then a difference between the groups, and not between the units that
happened to be in them. From the curves come the two summaries people ask
for: the difference in the chance of surviving to a given time, and the
difference in the restricted mean survival time (the average time lived
within a horizon), each with an interval.

This page shows the two standard ways of getting adjusted curves, what to
check, and when the two disagree. The functions are documented in
:doc:`comparison_and_validation`.

Two ways to adjust
------------------

**Weighting.** Give each unit a weight of one over the probability of
being in its own group, given its covariates (a *propensity* weight). A
light-duty part from the supplier that rarely got light duty counts for
several parts; a heavy-duty part from the same supplier counts for about
one. In the weighted sample each group has the whole sample's mix of
covariates, and the weighted Kaplan-Meier curve of each group is its
adjusted curve. The Kaplan-Meier estimator takes the weights as ``w`` (see
:doc:`Weighted Data`); ``sf_diff`` and ``rmst_diff`` compare two weighted
curves with robust variances; and ``weight_diagnostics`` checks the
weights. SurPyval does not fit the propensity model: the weights are
yours.

**Direct adjustment** (also called standardisation, or g-computation). Fit
a regression model for the lifetime with the group and the other
covariates in it. Then, for each group, predict every unit's survival curve
*as if it were in that group*, leaving its other covariates as they are,
and average the predictions over the sample:

.. math::

    \bar S_g(t) = \frac{1}{N} \sum_{i=1}^{N} \hat S(t \mid g, Z_i).

``direct_adjustment`` does this for a fitted parametric regression
(``WeibullAFT``, ``LogNormalAFT``, ``WeibullPH`` and the rest) or ``CoxPH``.

Both need the same thing from you: the covariates that drive *both* the
group a unit ended up in and how long it lasts (the confounders) must be
measured and used. Neither can adjust for something that was not recorded.

An engineering example: two suppliers, different duty
-----------------------------------------------------

A plant buys a seal from suppliers A and B. Supplier B's seals went mostly
to the light-duty pumps, supplier A's mostly to the heavy-duty ones. The
data below are simulated so that the truth is known: heavy duty halves the
life, and supplier B's seals last about 16% longer on the same duty. Units
still running at 1,500 hours are censored.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import surpyval as sp

    rng = np.random.default_rng(42)
    n = 600
    heavy = rng.binomial(1, 0.5, n)
    is_b = rng.binomial(1, np.where(heavy == 1, 0.25, 0.75))
    life = 1000 * np.exp(-0.7 * heavy + 0.15 * is_b) * rng.weibull(2.0, n)
    df = pd.DataFrame(
        {
            "hours": np.minimum(life, 1500),
            "c": (life > 1500).astype(int),
            "supplier": np.where(is_b == 1, "B", "A"),
            "heavy": heavy,
        }
    )
    pd.crosstab(df["supplier"], df["heavy"])

The plain comparison, a Kaplan-Meier curve per supplier, says that 34
percentage points more of supplier B's seals are still running at 800
hours:

.. jupyter-execute::

    a, b = df[df["supplier"] == "A"], df[df["supplier"] == "B"]
    km_a = sp.KaplanMeier.fit(a["hours"], c=a["c"])
    km_b = sp.KaplanMeier.fit(b["hours"], c=b["c"])
    naive = sp.sf_diff(km_b, km_a, 800, transform="linear")
    print(round(naive["difference"], 3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(naive["difference"], 2) == 0.34, naive["difference"]

Most of that is duty, not the supplier. On the same duty mix the true
difference at 800 hours is 8 points.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    def _S(sup):
        scale = 1000 * np.exp(-0.7 * np.array([0, 1]) + 0.15 * sup)
        return np.mean(np.exp(-((800 / scale) ** 2)))

    _truth = _S(1) - _S(0)
    assert round(_truth, 2) == 0.08, _truth

Direct adjustment
~~~~~~~~~~~~~~~~~

Fit a regression with the supplier and the duty, and hand it to
``direct_adjustment`` with the name of the group. A model fitted with a
formula needs the data frame as ``Z``, so that the supplier can be set in
it and the formula's columns rebuilt:

.. jupyter-execute::

    model = sp.WeibullAFT.fit_from_df(
        df, x_col="hours", c_col="c", formula="C(supplier) + heavy"
    )
    adjusted = sp.direct_adjustment(model, group="supplier", Z=df)
    adjusted

.. jupyter-execute::

    res = adjusted.sf_diff(800, "B", "A")
    print(
        "difference at 800 h: {:.3f} ({:.3f} to {:.3f})".format(
            res["difference"], res["lower"], res["upper"]
        )
    )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(res["difference"], 2) == 0.13, res
    assert res["lower"] < _truth < res["upper"], (res, _truth)
    assert naive["lower"] > res["upper"], (naive, res)

The adjusted difference is 13 points, with an interval that holds the true
8 and lies well below the plain comparison's 34. The restricted mean says
the same in hours: on the plant's duty mix, a seal from B runs for more of
the first 1,500 hours than one from A.

.. jupyter-execute::

    rm = adjusted.rmst_diff(1500, "B", "A")
    print(
        "extra hours within 1,500: {:.0f} ({:.0f} to {:.0f})".format(
            rm["difference"], rm["lower"], rm["upper"]
        )
    )

.. jupyter-execute::

    ax = adjusted.plot()

``adjusted.sf(x, level)``, ``ff`` and ``cb`` give one group's adjusted
curve and its confidence bounds, and ``rmst(tau, level)`` its restricted
mean.

By default the curves are averaged over the units the model was fitted to
(with their counts and weights). Pass ``Z`` to average over another
population, for example next year's planned duty mix, and ``w`` to weight
its rows.

Weighting
~~~~~~~~~

For weighting, each seal needs the probability that a seal on its duty
came from its supplier. With one yes/no covariate that is just a
proportion within each duty class; with several covariates it would come
from a logistic regression fitted elsewhere.

.. jupyter-execute::

    share_b = df.groupby("heavy")["supplier"].transform(
        lambda s: (s == "B").mean()
    )
    df["w"] = np.where(df["supplier"] == "B", 1 / share_b, 1 / (1 - share_b))

Check the weights before using them:

.. jupyter-execute::

    check = sp.weight_diagnostics(df["w"], df["supplier"], df[["heavy"]])
    check["groups"].round(3)

.. jupyter-execute::

    check["balance"].round(3)

.. jupyter-execute::

    check["flags"]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert check["flags"] == []
    _g = check["groups"]
    assert all(_g["ess_ratio"].round(2) > 0.75), _g
    _bal = check["balance"].iloc[0]
    assert round(_bal["smd_unweighted"], 1) == 1.0, _bal
    assert abs(_bal["smd_weighted"]) < 1e-9, _bal

Three things to read:

- **Effective sample size** (``ess``): the number of equally weighted
  units that would carry the same information. Here each supplier keeps
  nearly 80% of its units. A group left with a small fraction has uneven
  weights, and wide intervals to match.
- **Extreme weights** (``max_share``, ``n_extreme``): the largest weight's
  share of its group's total, and the number of weights more than ten
  times the group's mean. One unit carrying a tenth of its group means the
  adjusted curve is close to that unit's own history.
- **Balance**: the standardised mean difference of each covariate between
  the groups, before and after weighting. Before, duty differs between the
  suppliers by a full standard deviation; after, by nothing. Below 0.1
  after weighting is the usual rule.

``flags`` lists, in plain sentences, anything outside those rules; here it
is empty. The weighted comparison:

.. jupyter-execute::

    a, b = df[df["supplier"] == "A"], df[df["supplier"] == "B"]
    kw_a = sp.KaplanMeier.fit(a["hours"], c=a["c"], w=a["w"])
    kw_b = sp.KaplanMeier.fit(b["hours"], c=b["c"], w=b["w"])
    weighted = sp.sf_diff(kw_b, kw_a, 800, transform="linear")
    print(
        "difference at 800 h: {:.3f} ({:.3f} to {:.3f})".format(
            weighted["difference"], weighted["lower"], weighted["upper"]
        )
    )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(weighted["difference"], 2) == 0.11, weighted
    assert weighted["lower"] < _truth < weighted["upper"]
    assert weighted["se"] > res["se"]

The two routes agree: 11 points by weighting, 13 by direct adjustment, and
both intervals hold the truth. The weighted interval is wider. That is the
usual trade: weighting assumes nothing about the shape of the lifetime
distribution and pays for it in precision. ``sp.rmst_diff(kw_b, kw_a,
1500)`` is the weighted counterpart of the restricted-mean difference.

When the two disagree
---------------------

Each route leans on a different model, and each fails in its own way.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Direct adjustment
     - Weighting
   * - What must be right
     - The model for the *lifetime* given the group and the covariates
     - The model for the *group* given the covariates (the propensity)
   * - Typical failure
     - A missing interaction or curvature; proportional hazards assumed
       where they do not hold; predictions for covariate values a group
       never had (extrapolation, which the model does without complaint)
     - A wrong propensity model, seen as imbalance after weighting; groups
       that barely overlap, seen as extreme weights and a small effective
       sample size
   * - Precision
     - Higher: the model lends strength
     - Lower, and much lower with extreme weights
   * - How to check
     - The model's own diagnostics (``model.diagnose()``, residuals, an
       interaction term, a different family)
     - ``weight_diagnostics``

When the two answers agree, as above, each supports the other. When they
do not, something specific is wrong, and the checks say where to look:

- Balance is poor after weighting: the propensity model is wrong. Fix the
  weights before reading anything into the difference.
- The weights are extreme: the groups hardly overlap on some covariate.
  There are units in one group with no counterpart in the other.
  Weighting shows it (through the diagnostics); direct adjustment hides
  it, by extrapolating the regression into the gap. Neither answer is
  trustworthy there. Restrict the comparison to where the groups overlap
  and say so.
- Balance is good and the weights are mild, yet the answers differ: look
  at the lifetime model. Add the interaction of the group with the
  covariates, try another family, and compare the model's curve for each
  group with the weighted Kaplan-Meier curve.

A medical example: lung cancer survival by sex
----------------------------------------------

In the North Central Cancer Treatment Group lung cancer data (``lung`` in
R's ``survival``), women live longer than men. Women in the study were also
slightly younger and in slightly better condition (ECOG performance
score). How much of the difference is left once age and condition are
held equal?

.. jupyter-execute::

    from surpyval.datasets import load_lung

    lung = load_lung().dropna(subset=["ph.ecog"])
    Z = lung[["sex", "age", "ph.ecog"]].to_numpy(dtype=float)
    cox = sp.CoxPH.fit(lung["time"].to_numpy(), Z, c=1 - lung["status"])
    by_sex = sp.direct_adjustment(cox, group=0)
    by_sex.levels

The group is column 0 of ``Z`` (sex, 1 for men and 2 for women). The
adjusted chance of living a year, and the difference:

.. jupyter-execute::

    res = by_sex.sf_diff(365, 2, 1)
    print("men {:.3f}, women {:.3f}".format(res["sf_b"], res["sf_a"]))
    print(
        "difference: {:.3f} ({:.3f} to {:.3f}), p = {:.4f}".format(
            res["difference"], res["lower"], res["upper"], res["p_value"]
        )
    )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(res["sf_b"], 2) == 0.34, res
    assert round(res["sf_a"], 2) == 0.52, res
    assert res["lower"] > 0, res

If every patient were a woman, 52% would be alive at one year; if every
patient were a man, 34%: with age and condition held equal, the gap
remains. Over the first 700 days:

.. jupyter-execute::

    rm = by_sex.rmst_diff(700, 2, 1)
    print(
        "extra days within 700: {:.0f} ({:.0f} to {:.0f})".format(
            rm["difference"], rm["lower"], rm["upper"]
        )
    )

.. jupyter-execute::

    ax = by_sex.plot()

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(rm["difference"]) == 99, rm

A Cox model's baseline ends at the last observed time, so its restricted
mean refuses a horizon beyond it; a parametric regression extrapolates and
takes any horizon.

This is an adjusted *association*. Sex was not assigned, and the
adjustment covers age and ECOG score only; whatever else differs between
the men and women in the study is still in the difference.

How the uncertainty is computed
-------------------------------

An adjusted curve is uncertain for two reasons.

1. **The model was estimated.** For a parametric regression the standard
   error comes from the delta method: the gradient :math:`g` of the
   averaged quantity in the model's parameters and their covariance
   :math:`\Sigma` give :math:`g^\top \Sigma\, g`. For a Cox model it is
   the variance of the baseline hazard plus the coefficients' part, the
   same two parts as the bounds of its own predicted curve
   (``CoxPH.cb``), carried through the average.
2. **The units are a sample.** The average is over the units that happened
   to be studied. Another sample would have other covariates, so the
   average of even a perfectly known model would move. This adds the
   spread of the units' own predictions,
   :math:`\sum_i (v_i - \bar v)^2 / N^2`, where :math:`v_i` is unit
   :math:`i`'s predicted survival (or difference, or restricted mean).

``covariates="sampled"``, the default, includes both. Use
``covariates="fixed"`` when the rows *are* the population of interest (a
reference duty mix, or exactly this fleet): only the first part is then
counted.

The two curves being compared come from one model and one set of units,
so they are not independent. ``sf_diff`` and ``rmst_diff`` take the
standard error of the difference itself, not the sum of two separate
variances. Intervals for a curve are formed on the logit scale, so they
stay between 0 and 1; intervals for a difference are the difference plus
or minus a normal quantile times its standard error. ``robust=True`` uses
the model's robust (sandwich) covariance in the first part.

The curves are checked against R: a Cox model's adjusted curves equal the
row means of ``survival::survfit`` with ``newdata``, and a Weibull model's
curves, differences, restricted means and delta-method standard errors
match ``survival::survreg`` worked by hand. The Cox curves also match
``riskRegression::ate`` (the G-formula behind ``adjustedCurves``' direct
method). Its standard errors can differ from the ones here, by 4% to 13%
on the ``lung`` data, and the reason is the estimator, not the averaging.
Both count the sampling of the covariates the same way (that part agrees
exactly). For the model's own uncertainty, ``ate`` uses the empirical
(sandwich) influence function of the Cox fit, and SurPyval uses the
model-based variance that ``survfit`` reports for a single curve. The two
agree when the proportional hazards model holds, and drift apart when it
fits less well. On ``lung`` the same gap shows for a single patient:
``riskRegression``'s standard error of one curve is 10% below
``survfit``'s for one sex and 8% above it for the other.

For the weighted route, the variance is the weighted Kaplan-Meier's robust
one (see :doc:`Weighted Data`). It treats the weights as known. Weights
estimated from the same data usually make the true variance a little
smaller, so those intervals are, if anything, conservative.

What is not built
-----------------

- **Fitting the propensity model.** Bring the weights; any logistic
  regression will produce them.
- **Bounds for other model families.** ``direct_adjustment`` averages the
  predictions of any regression model with ``sf(x, Z)``, but the variance
  is built for the parametric regression families and ``CoxPH``. For
  another family, ``cb``, ``sf_diff``, ``rmst`` and ``rmst_diff`` refuse
  and say so; refit the model to resampled rows for a bootstrap.
- **Stratified Cox models** are refused: put the stratifying variable in
  the model as a covariate, or use a parametric regression.
- **Doubly robust estimators** (which combine the two routes) and
  adjusted cumulative incidence under competing risks.

References
----------

- Makuch, R. W. (1982). Adjusted survival curve estimation using
  covariates. *Journal of Chronic Diseases* 35, 437-443.
- Cole, S. R. and Hernán, M. A. (2004). Adjusted survival curves with
  inverse probability weights. *Computer Methods and Programs in
  Biomedicine* 75, 45-49.
- Austin, P. C. (2014). The use of propensity score methods with survival
  or time-to-event outcomes: reporting measures of effect similar to those
  used in randomized experiments. *Statistics in Medicine* 33, 1242-1258.
- Austin, P. C. and Stuart, E. A. (2015). Moving towards best practice
  when using inverse probability of treatment weighting (IPTW) using the
  propensity score to estimate causal treatment effects in observational
  studies. *Statistics in Medicine* 34, 3661-3679.
- Hernán, M. A. and Robins, J. M. (2020). *Causal Inference: What If*.
  Chapman & Hall/CRC.
