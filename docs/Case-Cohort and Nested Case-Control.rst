Case-Cohort and Nested Case-Control
===================================

Sometimes the follow-up of a large cohort is cheap, but measuring the
covariates is not: a blood sample has to be assayed, old records read by
hand, a part sent to a laboratory. When the event is rare, almost all of
that cost would go on units that never have it, and they carry little of
the information about a hazard ratio. Two designs measure the covariates
on every case (each unit with the event) and only a sample of the
others.

- **Case-cohort.** At the start, a random *subcohort* is drawn from the
  whole cohort. The covariates are measured on the subcohort and on
  every case, whether in the subcohort or not. One subcohort serves for
  every outcome studied in the cohort.
- **Nested case-control.** Each time a unit has the event, a few
  *controls* are drawn at random from the units still at risk at that
  moment. The case and its controls form a *matched set*, and the
  covariates are measured on them.

Fitting an ordinary Cox model to such a sample would be wrong: the
cases are far more common in it than in the cohort, and the risk sets
are not the cohort's. SurPyval fits the estimators that correct for the
sampling.

In plain words
--------------

- **What is estimated.** The same hazard ratios as a Cox fit to the
  whole cohort would give (``exp(params)``), with standard errors and
  Wald intervals in ``summary()``.
- **Case-cohort.** At each event time the case is compared with the
  subcohort members still at risk, who stand in for everyone at risk.
  The estimators differ only in whether the case itself is in that risk
  set and how the members are weighted; in practice they agree closely.
  The standard errors have two parts: the usual one, from the events, and
  one for having sampled the subcohort, which vanishes when the subcohort
  is the whole cohort.
- **Nested case-control.** Each case is compared with its own matched
  set: given that one member of the set had the event then, how likely
  is it that it was this one, given everyone's covariates? This is the
  conditional logistic likelihood, the Cox partial likelihood on a
  sampled risk set.
- **Absolute risks.** A case-cohort sample knows the sampling fraction
  (the subcohort's size against the cohort's), so it can estimate the
  baseline hazard, and ``sf``, ``Hf`` and the other curves are given. A
  nested case-control sample records nothing about how many were at risk,
  so it has no baseline: its curves refuse and say so.

A case-cohort fit
-----------------

The data are one row per unit sampled: its time ``x``, ``c`` (0 a case,
1 right censored), its covariates, and ``subcohort``, True for the
subcohort's members. A unit outside the subcohort must be a case.
``cohort_size`` is the size of the cohort the subcohort was drawn from.

.. code:: python

    import numpy as np
    from surpyval import CoxPH

    rng = np.random.default_rng(1)
    z = rng.normal(size=2000)                      # the whole cohort
    t = rng.exponential(1 / (0.02 * np.exp(0.7 * z)))
    x, c = np.minimum(t, 10.0), (t > 10.0).astype(int)
    sub = rng.random(2000) < 0.15                  # a 15% subcohort
    keep = sub | (c == 0)                          # and every case

    model = CoxPH.fit_case_cohort(
        x[keep], z[keep], subcohort=sub[keep], cohort_size=2000,
        c=c[keep],
    )
    model.params, model.standard_errors()
    # (array([0.7095]), array([0.0892]))

Here the whole cohort's fit gives 0.711 with a standard error of
0.048: the sample of 664 units recovers the hazard ratio, with the wider
interval its smaller size implies.

``method`` chooses the estimator, with the names of R's
``survival::cch``:

``"prentice"`` (the default)
    The risk set at a case's time is the subcohort at risk, plus the case
    if it is from outside the subcohort (Prentice 1986).
``"self-prentice"``
    The subcohort at risk alone (Self and Prentice 1988). It has the
    same large-sample variance as Prentice's.
``"lin-ying"``
    Every case while at risk, and each non-case of the subcohort
    weighted up to stand for the cohort's non-cases (Lin and Ying 1993).
    ``robust=True`` gives its design-based variance.
``"barlow"``
    A weighted Cox fit: a subcohort member counts as the inverse of the
    sampling fraction, a case counts once at its event, with the robust
    variance (Barlow 1994).
``"borgan-i"``, ``"borgan-ii"``
    For a subcohort drawn separately within strata (say, more from a
    rare exposure group): Self-Prentice and Lin-Ying with each stratum's
    own sampling fraction (Borgan et al. 2000). Give the stratum of each
    unit as ``sampling_strata`` and ``cohort_size`` as a dict of each
    stratum's size in the cohort. The model itself is not stratified.

Delayed entry (``tl``) is taken by every method. Each estimator's
estimating equation and variance, with its source, is in the API
reference of :func:`~surpyval.univariate.regression.proportional_hazards.case_cohort.fit_case_cohort`.

What a case-cohort fit refuses
------------------------------

The estimators maximise a *pseudo*-likelihood, whose value is not the
likelihood of the data. So ``log_likelihood``, ``aic()``, ``bic()`` and
``likelihood_ratio_test()`` refuse; compare models by the Wald tests of
``summary()``. The residuals and ``diagnose()`` refuse too (the sample's
rows are not the cohort's units), as do pointwise bounds on the curves
(``cb``), whose variance under the sampling is not built here. Each
refusal says why and what to use instead.

The baseline is Breslow's, with each risk set weighted by the inverse of
each member's chance of being sampled: a subcohort member counts as the
cohort's size over the subcohort's (within its stratum for Borgan I), a
case counts once at its own event; for Lin-Ying and Borgan II, their own
weights.

A nested case-control fit
-------------------------

The data are one row per sampled unit: its covariates, ``case`` (True for
the case) and ``sets``, its matched set. No times are needed.

.. code:: python

    model = CoxPH.fit_nested_case_control(Z, case, sets)
    model.summary()

The default ``tie_method="exact"`` is the exact conditional likelihood
(R's ``clogit``); where several cases share a set (tied event times)
``"efron"`` and ``"breslow"`` are its approximations, and with one case
per set the three agree. A set with no case or no control says nothing
about the hazard ratios and is left out, with a warning. The conditional
likelihood is a true likelihood, so ``log_likelihood``,
``likelihood_ratio_test()`` and the information criteria are given.

Checked against
---------------

Every case-cohort estimator matches R's ``survival`` 3.8.12 (``cch``,
and ``coxph`` with Barlow's weights) on the National Wilms Tumor Study
example of ``?cch``, with and without delayed entry, to about 1e-9 in
the coefficients and the standard errors; the nested case-control fit
matches ``clogit`` on ``infert`` and on a nested case-control sample of
the same cohort, exact and Breslow. With the whole cohort taken as the
subcohort every estimator is the ordinary ``CoxPH.fit``. The script is
``scripts/reference/reference_r_case_cohort.R``.

References
----------

- Prentice, R. L. (1986). A case-cohort design for epidemiologic cohort
  studies and disease prevention trials. *Biometrika* 73, 1-11.
- Self, S. G. and Prentice, R. L. (1988). Asymptotic distribution
  theory and efficiency results for case-cohort studies. *Annals of
  Statistics* 16, 64-81.
- Lin, D. Y. and Ying, Z. (1993). Cox regression with incomplete
  covariate measurements. *JASA* 88, 1341-1349.
- Barlow, W. E. (1994). Robust variance estimation for the case-cohort
  design. *Biometrics* 50, 1064-1072.
- Therneau, T. M. and Li, H. (1999). Computing the Cox model for
  case-cohort designs. *Lifetime Data Analysis* 5, 99-112.
- Borgan, Ø., Langholz, B., Samuelsen, S. O., Goldstein, L. and Pogoda,
  J. (2000). Exposure stratified case-cohort designs. *Lifetime Data
  Analysis* 6, 39-58.
- Goldstein, L. and Langholz, B. (1992). Asymptotic theory for nested
  case-control sampling in the Cox regression model. *Annals of
  Statistics* 20, 1903-1928.
