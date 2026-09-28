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
   item, a degradation path).

   *Checked* by ``conformance/test_missing.py``; known gaps #382 (a
   missing query time) and #388 (a missing frailty group).

4. **Order doesn't matter.** The order of the data rows never changes a fit,
   and permuting a query permutes the result.

   *Checked* by ``conformance/test_metamorphic.py`` and
   ``conformance/test_vectorisation.py``, and on generated data by the
   property tests.

5. **Counts equal repetition.** ``n = k`` gives the same answer as ``k``
   identical rows.

   *Checked* by ``conformance/test_metamorphic.py``.

6. **Units don't matter.** Rescaling time rescales the answer and nothing
   else.

   *Checked* by ``conformance/test_metamorphic.py``; known gaps #385 and
   #393.

Outputs
-------

7. **Shapes.** A scalar, 1-D or 2-D query gives a result of the same shape,
   and an empty query an empty result.

   *Checked* by ``conformance/test_vectorisation.py``; known gap #381.

8. **The functions of a model agree with each other.**
   :math:`S + F = 1`, :math:`H = -\log S`, :math:`f = h S`, ``qf`` inverts
   ``ff``, and the causes' cumulative incidences sum to :math:`1 - S`.

   *Checked* by ``conformance/test_identities.py``; known gaps #383 and
   #384.

9. **Valid values.** Survival stays in :math:`[0, 1]` and never increases;
   cumulative quantities never decrease. The documented exception is the
   additive hazards model, whose estimate need not be monotone (#376).

   *Checked* by ``conformance/test_bounds.py``.

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
    ``set_bounds(lower, upper)`` gives a non-parametric estimate an explicit
    support: its start value from ``lower`` to the first time, its last
    value held up to ``upper``, and ``nan`` outside, for every function,
    ``interp`` and confidence bound.

    *Checked* by ``conformance/test_outside_data.py``, which also sets
    bounds on every non-parametric estimate and requires each to have
    ``set_bounds``.

Estimation
----------

12. **A fit returns what it claims:** the optimum of its stated estimator.
    If the estimator has no optimum on the data, the fit refuses or warns;
    it never returns a silent degenerate answer.

    *Partly checked.* The property tests check that parametric fits, on
    generated data with every kind of censoring and truncation, are local
    optima of a likelihood the test computes itself from the fitted
    model's functions (``properties/test_parametric.py``), and the
    reference tests compare fits with R, lifelines and scikit-survival;
    other families rely on the reference tests; known gap #392.

13. **Failure is never silent.** An optimiser that does not converge warns,
    and a fit never quietly returns its starting values.

    *Checked* by ``conformance/test_convergence.py``: every iterative fit
    is starved (an iteration limit of 1, a start a million times the
    answer, or data with no maximum) and must warn, raise, or still reach
    the maximum; a closed-form or exact estimator is excluded, with the
    reason. Known gaps: the univariate (#427), accelerated-life (#428)
    and recurrent (#429) fits stop far from a distant start silently, and
    the regression, Fine-Gray, copula, mixture and degradation fits
    return a finite answer silently where the likelihood has no maximum
    (#392).

14. **Entry points agree.** ``fit``, ``fit_from_df``, a formula,
    ``from_params`` and ``fit_tvc`` give the same model for the same data.

    *Checked* by ``conformance/test_fit_paths.py``.

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
    ``scripts/reference/regenerate.sh``); known gap #380.

Uncertainty
-----------

17. **Intervals achieve their stated coverage, and tests their stated
    size.**

    *Partly checked* by the calibration studies
    (``surpyval/tests/calibration``, run nightly), which cover the main
    parametric, non-parametric, Cox, regression, degradation and recurrent
    bounds and the hypothesis tests, not every model; known gap #390.

18. **Intervals behave consistently.** Bounds contain the estimate and stay
    in the valid range; a one-sided bound is the matching end of the
    two-sided bound; a higher confidence level gives a wider interval.

    *Checked* by ``conformance/test_options.py``, over every uncertainty
    method, confidence level and option of every registered model.

Behaviour and API
-----------------

19. **One seed rule.** ``seed=None`` (or ``random_state=None``) draws from
    numpy's global generator, so ``np.random.seed`` reproduces it; an
    explicit seed gets its own stream and leaves the global one alone.

    *Checked* by ``conformance/test_seeds.py``; known gap #389.

20. **Saving and loading.** Every model round-trips through strict JSON with
    identical predictions, stamped with the oldest schema version that can
    read it.

    *Checked* by ``conformance/test_serialisation.py`` and
    ``properties/test_serialisation.py``.

21. **Consistent names.** The same option has the same name, meaning and
    default everywhere (``alpha_ci``, ``bound``, ``on``, ``interp``,
    ``Z``, ``seed``).

    *Checked* by ``conformance/test_options.py``.

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

Adding to the list
------------------

When a bug is fixed, ask which principle it broke. If the principle's check
did not catch it, extend the check -- a new property in the conformance
suite covers every model at once. If no principle covers it, it may be a new
one: add it here with its check.
