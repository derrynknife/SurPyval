Design Principles
=================

These are the rules every part of SurPyval keeps, whichever model you use.
They exist so that what you learn about one model holds for the others, and
so that a contributor or reviewer can check a change against a fixed list
rather than against memory. :doc:`Conventions` gives the details of several
of them (the data format, missing values, seeds, saving and loading).

A principle is only as good as its check, so each one names the tests that
enforce it. Most are **properties**: general statements checked for every
registered model by the conformance suite (``surpyval/tests/conformance``,
see :doc:`Contributing`), or on generated data by the property-based tests
(``surpyval/tests/properties``). Where a model is known to break a
principle, the test marks it as a strict expected failure naming the issue
that tracks the fix, so the suite stays green and turns red the day the fix
lands.

Each principle is marked **checked** (every model it applies to is tested),
**partly checked** (tested for some models or some cases; the gap is named),
or **judgement** (a matter of review that no test can fully decide).

Inputs
------

1. **One data model.** Every fitter takes ``x``, ``c``, ``n``, ``t``,
   ``tl`` and ``tr`` with the same meanings: censoring flag 0 observed,
   1 right, -1 left, 2 interval; intervals are :math:`(l, r]` and truncation
   windows :math:`(t_l, t_r]`.

   *Partly checked.* The property tests generate data in every form (mixed
   censoring, ties, counts, truncation, tiny samples) for the univariate,
   regression and recurrent fitters, and the reference tests check the
   conventions against R; the other families are checked through their
   conformance fixtures only.

2. **Invalid input raises a** ``ValueError`` **that names the argument and
   says how to fix it** -- never a ``TypeError``, an ``IndexError`` or a
   silently wrong answer.

   *Partly checked* by ``properties/test_validation.py``, for the
   univariate fitters.

3. **Missing values.** ``nan`` in, ``nan`` out at prediction. At fit, a row
   with a missing value is dropped with one warning where rows are
   independent, and refused where a row is part of one unit (a recurrent
   item, a degradation path). A probability outside [0, 1] given to any
   ``qf`` gives ``nan`` there, with one warning.

   *Checked* by ``conformance/test_missing.py`` (``qf_outside`` for
   ``qf``, #611).

4. **Order doesn't matter.** The order of the data rows never changes a fit,
   and permuting a query permutes the result.

   *Checked* by ``conformance/test_metamorphic.py`` and
   ``conformance/test_vectorisation.py``, and on generated data by the
   property tests.

5. **Counts equal repetition.** ``n = k`` gives the same answer as ``k``
   identical rows.

   *Checked* by ``conformance/test_metamorphic.py``.

6. **Units don't matter.** Rescaling time rescales the answer and nothing
   else. Nor does a covariate's origin: with ``center=True`` every
   regression gives the same model when a constant is added to a
   covariate, and so do the defaults of Cox, Fine-Gray and the families
   whose baseline maps exactly between origins. Nor does a covariate's
   scale: multiplying a column by a constant divides its coefficient by it
   and leaves the maximised likelihood and every prediction as they were,
   whatever the covariate's scale (``1/T`` in kelvin is about 0.003).

   *Checked* by ``conformance/test_metamorphic.py``
   (``test_covariate_origin*``, and ``test_covariate_scale*`` with a
   column multiplied by 1/731, 731 and 1e-6, by ``fit`` and by
   ``fit_tvc``) and ``test_maximum.py``'s small- and large-scale (x1e4)
   fits (#577, #612). One family is excepted:
   the Beta4's likelihood is unbounded, so its maximum-likelihood fit can
   depend on the units, and warns when it does; its MPS fit does not
   (#385).

Outputs
-------

7. **Shape in, shape out.** A scalar query gives a numpy scalar, a 1-D or
   2-D query a result of its shape, and an empty query an empty result of
   its shape; a two-sided confidence bound adds a last ``[lower, upper]``
   axis. With covariates the shape is that of the times, rows and times
   paired: one row for every time, or one time for every row; other
   counts raise. ``grid=True`` on the Cox and parametric regression
   functions, and on the survival tree and forest, gives the row-by-time
   grid, ``(n_rows,) + x.shape``.
   ``surpyval.utils.shapes`` applies the rule at every model's public
   methods.

   *Checked* by ``conformance/test_vectorisation.py`` and ``cb_shape`` in
   ``conformance/test_options.py``, for every registered model, and for
   the time-varying ``sf_tvc`` by ``conformance/test_tvc.py``.

8. **The functions of a model agree with each other.**
   :math:`S + F = 1`, :math:`H = -\log S`, :math:`f = h S`, ``qf`` inverts
   ``ff``, and the causes' cumulative incidences sum to :math:`1 - S`.

   *Checked* by ``conformance/test_identities.py``.

9. **Valid and accurate values.** Survival stays in :math:`[0, 1]` and
   never increases; cumulative quantities never decrease. The documented
   exception is the parametric additive hazards models, whose survival can
   exceed 1 where :math:`h_0 + \beta'Z < 0` (#376); the Lin-Ying model
   predicts with the running maximum of its estimate. A distribution's
   functions are accurate to double
   precision wherever the value is representable, in the tails and at
   extreme parameters too.

   *Checked* by ``conformance/test_bounds.py``, and for accuracy by
   ``reference/test_tails.py`` against 50-digit mpmath values, for every
   distribution with a closed form.

10. **Covariate rows are independent.** Evaluating rows together gives the
    same as evaluating them one at a time.

    *Checked* by ``conformance/test_vectorisation.py``.

11. **Behaviour outside the data is defined and documented.** A parametric
    model is defined everywhere by its formula. An estimate with no formula
    for its shape -- a step estimate, a semi-parametric baseline -- starts
    at its initial value before the first time (survival 1, everything
    cumulative 0; at time 0 for the additive hazards model, whose
    covariate effect acts from time 0), and after the last time either
    holds its last value
    (the single-event estimates) or is ``nan`` (the recurrent mean
    cumulative functions), the same for all of a model's functions.
    ``set_support(lower, upper)`` gives a non-parametric estimate an explicit
    support: its start value from ``lower`` to the first time, its last
    value held up to ``upper``, and ``nan`` outside, for every function,
    ``interp`` and confidence bound.

    *Checked* by ``conformance/test_outside_data.py``, which also sets
    bounds on every non-parametric estimate, requires each to have
    ``set_support``, and requires every bound to be ``nan`` outside the
    data when none is set; and for time-varying covariates by
    ``conformance/test_tvc.py``, for a path that starts before 0, at 0 or
    later, and for queries at 0 and below.

Estimation
----------

12. **A fit returns what it claims:** the optimum of its stated estimator.
    If the estimator has no optimum on the data, the fit refuses or warns;
    it never returns a silent degenerate answer. Where the data do not
    determine a coefficient -- a covariate column that is constant where
    the model has an intercept, or a combination of the others -- the fit
    says so: the coefficient is ``nan`` and listed in ``aliased``, with
    one warning naming the column, rather than an arbitrary value.

    *Partly checked.* The property tests check that parametric fits, on
    generated data with every kind of censoring and truncation, are local
    optima of a likelihood the test computes itself from the fitted
    model's functions (``properties/test_parametric.py``), and the
    reference tests compare fits with R, lifelines and scikit-survival;
    ``calibration/test_refit_registry.py`` refits every registered model
    to data drawn from itself (on demand, ``--run-calibration``). Where the likelihood has no
    finite maximum, univariate MLE refuses and the regression, frailty,
    Fine-Gray, copula, mixture and degradation fits warn "No finite
    maximum" (#392); known gap: abutting intervals such as (1, 3] and (3, 5], whose likelihood has a flat
    ridge. ``conformance/test_aliasing.py`` refits every registered model
    that has coefficients with a repeated covariate column, and with a
    constant one where it has an intercept, and requires the aliasing and
    otherwise the fit without the column; the time-varying fits are
    checked in ``univariate/regression/test_aliasing.py``. The
    derivatives a fit takes -- the gradient and Hessian with which it
    searches, verifies its maximum and computes its covariance -- agree
    with finite differences at the fit, for every registered model that
    takes them (``conformance/test_derivatives.py``, #562).

13. **Failure is never silent.** An optimiser that does not converge warns,
    and a fit never quietly returns its starting values.

    *Checked* by ``conformance/test_convergence.py``: every iterative fit
    is starved (an iteration limit of 1, a start a million times the
    answer, or data with no maximum) and must warn, raise, or still reach
    the maximum; a closed-form or exact estimator is excluded, with the
    reason. A fit accepts an optimiser's answer only when it is a
    verified maximum (zero gradient, positive-definite Hessian), and a fit
    given ``init`` is also started from the default start. A likelihood
    with no finite maximum warns so (#392), whatever the model. And by
    ``conformance/test_maximum.py``: every maximum-likelihood fit in the
    registry -- the univariate distributions, mixtures, the parametric and
    semi-parametric regressions, frailty, competing-risks, recurrence and
    copula models, and the degradation process and destructive models --
    records what it reached as its model's ``maximum``
    (``"verified"``, ``"unverified"`` or ``"no finite maximum"``), warns
    exactly when that is not a verified maximum, its fixture's fit, its
    starved fit and its time-varying-covariate fit alike; and a verified
    maximum passes an independent check at the reported parameters (the
    gradient of the model's own likelihood ~0 and its Hessian positive
    definite, a parameter on a boundary of its space held out where the
    likelihood does not rise off it, and a refit from off it finding
    nothing higher, #579).

14. **Entry points agree.** ``fit``, ``fit_from_df``, a formula,
    ``from_params`` and ``fit_tvc`` give the same model for the same data.

    *Checked* by ``conformance/test_fit_paths.py``, and for a regression
    model's attributes (every builder, ``from_dict`` included, gives the
    same declared attributes) by ``conformance/test_attributes.py``.

15. **Defaults are the statistically best standard choice, and the same
    everywhere.** For example, every Cox fit defaults to Efron's tie
    handling, which is far less biased than Breslow's.

    The first half is *judgement*, informed by the calibration studies and
    the literature. The second half is *checked* by
    ``conformance/test_defaults.py``: every entry point of a fitter gives
    a shared argument the same default, and the tie-handling default is
    the same across the Cox-based models.

16. **Conventions follow the established references** -- R's
    ``survival``, ``cmprsk`` and ``pec``, lifelines, scikit-survival --
    unless there is a documented reason to differ.

    *Checked* by ``surpyval/tests/reference``, which compares results with
    values those packages computed (regenerated by
    ``scripts/reference/regenerate.sh``).

Uncertainty
-----------

17. **Intervals achieve their stated coverage, and tests their stated
    size.**

    *Partly checked* by the calibration studies
    (``surpyval/tests/calibration``, run on demand), which cover the main
    parametric, non-parametric, Cox, regression, degradation and recurrent
    bounds and the hypothesis tests, not every model.

18. **Intervals behave consistently.** Bounds contain the estimate and stay
    in the valid range; a one-sided bound is the matching end of the
    two-sided bound; a higher confidence level gives a wider interval.

    *Checked* by ``conformance/test_options.py``, over every uncertainty
    method, confidence level and option of every registered model.

Behaviour and API
-----------------

19. **One seed rule.** Every method that draws takes ``random_state``.
    ``None`` draws from numpy's global generator, so ``np.random.seed``
    reproduces it; an explicit seed gets its own stream and leaves the
    global one alone.

    *Checked* by ``conformance/test_seeds.py``, for every registered model
    that draws.

20. **Saving and loading.** Every model round-trips through strict JSON with
    identical predictions, stamped with the oldest schema version that can
    read it, and through ``pickle``, which process pools and the packages
    built on SurPyval use to move a fitted model between processes.

    *Checked* by ``conformance/test_serialisation.py``,
    ``conformance/test_pickle.py`` (#573) and
    ``properties/test_serialisation.py``, and for tuple and mixed
    ``str`` / ``int`` cause labels by ``conformance/test_labels.py``.

21. **Consistent names.** The same option has the same name, meaning and
    default everywhere (``alpha_ci``, ``bound``, ``on``, ``interp``,
    ``Z``, ``random_state``, ``n_boot``, ``tie_method``, ``event``, and
    ``x`` for the times and ``p`` for a quantile's probability), and so
    does the same attribute: every model's fitted values are ``params``,
    named entry by entry by the attribute ``parameter_names``, and every
    model with ``covariance()`` has ``standard_errors()``, the square roots
    of its diagonal; a regression coefficient is named by its covariate's
    column, else ``coef_j``, and the limited-failure proportion is
    ``lfp_p``. Every
    DataFrame entry point (``fit_from_df``, ``fit_tvc_from_df``,
    ``fit_tvc_timeline_from_df``) names a column argument after the ``fit``
    argument it fills with a ``_col`` suffix, ``_cols`` for a list of
    columns: ``x_col``, ``c_col``, ``n_col``, ``xl_col``, ``xr_col``,
    ``tl_col``, ``tr_col``, ``i_col``, ``e_col``, ``y_col``, ``Z_cols``.
    When a name changes, the old one keeps working for one release with a
    ``DeprecationWarning`` naming the new one.

    The quantities a model comparison reads -- ``aic``, ``bic``,
    ``neg_ll``, ``log_likelihood``, ``covariance`` -- are the same kind
    (a method or a value) on every model, and every full-likelihood fit
    offers ``aic`` and ``bic``.

    *Checked* by ``conformance/test_options.py``,
    ``conformance/test_params.py``, ``conformance/test_comparison.py``
    (standard errors, #613), ``conformance/test_repr.py`` (every fitter
    prints what it is, #614), for the column names,
    ``conformance/test_fit_paths.py``, and for the comparison quantities
    ``conformance/test_comparison.py`` and ``conformance/test_surface.py``
    (#572; the degradation process models, ``DestructiveDegradation``
    and ``CauseSpecificNHPP`` have them since #711).

22. **Warnings and errors.** One warning per problem, with counts, saying
    what happened and what to do about it. No raw numpy warning escapes
    from package code.

    *Checked* by ``conformance/test_warnings.py``, which fails for any raw
    numerical warning raised inside SurPyval while fitting or predicting
    with a registered model.

23. **Documentation.** Every public item is documented with a runnable
    example, and every number quoted in the prose is checked.

    *Checked* for what exists: the documentation build runs every example
    and the hidden checks of the quoted numbers (see :doc:`Contributing`),
    and the docstring examples run as tests. *Checked* for completeness
    by ``conformance/test_documentation.py``: every public item has a
    docstring with an example, and a new one without fails.

24. **Simple by default; more as an option.** When a method reaches its
    limit -- data it cannot handle, an approximation that breaks down, a
    question that needs a heavier computation -- the new approach is added
    as an option beside it, not put in its place. The default stays the
    simple, standard method that serves the usual case, so a plain call
    stays fast and easy to explain, and its results do not move. A
    default changes only when it is wrong for the usual case (principle
    15; for example a band that did not hold its level, #390), not because
    a better method exists for a harder one. For example, trees split
    greedily by default and take conditional inference with
    ``selection="ctree"``; ``cb`` gives Wald bounds and ``bootstrap_cb``
    resamples.

    *Judgement*, applied in review: a change to a default says which
    principle the old default broke.

Adding to the list
------------------

When a bug is fixed, ask which principle it broke. If the principle's check
did not catch it, extend the check -- a new property in the conformance
suite covers every model at once. If no principle covers it, it may be a new
one: add it here with its check.
