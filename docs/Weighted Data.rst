Weighted Data
=============

Some data come with a number on each row that says how much that row
*counts*: a **weight**. SurPyval takes it as ``w``, beside the counts ``n``,
in the Kaplan-Meier, Nelson-Aalen and Fleming-Harrington estimators, the
parametric distributions, the parametric regressions, Cox regression,
Aalen's additive hazards model, the non-parametric competing-risks
estimate (``CompetingRisks``) and the competing-risks regressions
(cause-specific Cox and Fine-Gray).
This page says what weights are, when to use ``w`` rather than ``n``, why a
weighted fit's intervals are computed differently, and what a weighted fit
will not do.

What a weight is
----------------

A count ``n`` says how many *identical units* a row records: ``n = 3`` is
three units, each seen, with the same time and the same censoring. Writing
the row out three times gives the same answer.

A weight ``w`` says how many units of the *population* each unit stands
for, or how much it should be up- or down-weighted to undo a bias. Nobody
saw ``w`` units. Three common sources:

- **Sampling weights.** You inspected one pump in every 40 at the small
  sites and one in every 5 at the big site. Each pump inspected at a small
  site stands for 40 pumps, each at the big site for 5. Without the weights
  the big site dominates the answer; with them the answer describes the
  whole fleet. Survey and registry data work the same way.
- **Inverse probability of treatment weights.** In observational data the
  units that got a treatment (an overhaul, a new supplier, a drug) differ
  from those that did not. Weighting each unit by one over the probability
  that it got what it got makes the groups comparable, as in a randomised
  trial.
- **Inverse probability of censoring weights.** If units drop out of
  follow-up for reasons tied to their risk, each unit still followed is
  weighted by one over its probability of still being followed.

A weight is any positive number (0.37, 12.5); a count is a whole number. A
row may have both: ``n = 3`` identical units, each standing for ``w = 12.5``.

When to use ``w`` and when ``n``
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * -
     - Count ``n``
     - Weight ``w``
   * - Meaning
     - ``n`` identical units were seen
     - one unit seen, standing for ``w`` units
   * - Values
     - 1, 2, 3, ...
     - any positive number
   * - Every value multiplied by 10
     - ten times the data: the intervals shrink
     - the same data: **nothing changes**
   * - Variance
     - the classical formulas
     - a robust (sandwich) variance

Use ``n`` when the numbers count units you saw. Use ``w`` when they say
how much each unit stands for, **even if they are whole numbers**: survey
weights of exactly 40 put in ``n`` would be read as forty units seen, and
the intervals would be more than six times too narrow. SurPyval refuses a
decimal ``n`` and says to use ``w``, but it cannot tell whole-number weights
from counts.

Why the intervals change
------------------------

The estimate itself is easy: each unit's contribution is multiplied by its
weight (to the numbers at risk and of failures, or to the likelihood).

The uncertainty is not. The classical formulas -- Greenwood's for a
Kaplan-Meier curve, the inverse of the information for a fitted Weibull --
assume that the information in the data grows with the number of units, and
with weights they would take the *total of the weights* for that number.
Multiply every weight by 10 and they would claim ten times the information
from the same data. The honest amount of information depends on how uneven
the weights are: a few units with huge weights give a noisy answer however
large the total. A useful summary is the **effective sample size**,
:math:`(\sum n w)^2 / \sum n w^2`, which equals the number of units when
the weights are equal and is smaller the more uneven they are. A weighted
fit prints it, and warns when it is below 20.

So every weighted fit uses a **robust** variance, one that is valid
whatever the weights' scale, and there is no option of the classical one.
The intervals are then built from it exactly as before (log(-log) bounds
for a survival curve, Wald bounds on each parameter's scale), so ``cb``,
``param_cb``, ``quantile_cb`` and the rest work as usual.

A worked example
----------------

Suppose a study samples long-lived units more often than short-lived ones:
of a population of lifetimes, a unit above the median is kept with
probability 0.8 and one below it with probability 0.2. The sample then
overstates how long units last. Each kept unit stands for one over its
probability of being kept:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    rng = np.random.default_rng(1)
    truth = surv.Weibull.from_params([10.0, 1.5])
    t = truth.random(2000, random_state=rng)
    censor = rng.uniform(0, 30, 2000)
    keep_probability = np.where(t > truth.qf(0.5), 0.8, 0.2)
    kept = rng.uniform(size=2000) < keep_probability
    x = np.minimum(t, censor)[kept]
    c = (censor < t)[kept].astype(int)
    w = 1 / keep_probability[kept]

    print(surv.Weibull.fit(x, c).params)          # biased: too long-lived
    model = surv.Weibull.fit(x, c, w=w)
    print(model.params)                           # close to (10, 1.5)
    print(model)

The weighted fit recovers the population's Weibull, and its bounds use the
robust covariance:

.. jupyter-execute::

    print(model.param_cb("alpha"))
    print(model.cb([5, 10, 15]))
    print(model.covariance())

The same weights scaled by any constant give the same model and the same
intervals:

.. jupyter-execute::

    scaled = surv.Weibull.fit(x, c, w=10 * w)
    print(np.allclose(scaled.cb([5, 10, 15]), model.cb([5, 10, 15])))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(scaled.cb([5, 10, 15]), model.cb([5, 10, 15]), rtol=1e-6)
    assert abs(model.params[0] - 10) < 1 and abs(model.params[1] - 1.5) < 0.2
    assert surv.Weibull.fit(x, c).params[0] > 11

A Kaplan-Meier curve and a Cox model take ``w`` the same way:

.. jupyter-execute::

    km = surv.KaplanMeier.fit(x, c, w=w)
    print(km.sf([5, 10, 15]))
    print(km.cb([5, 10, 15]))

    z = rng.integers(0, 2, kept.sum())
    cox = surv.CoxPH.fit(x, z, c=c, w=w)
    print(cox.params, cox.standard_errors())

The at-risk table of ``plot_nonparametric`` and ``at_risk_table`` counts
the units themselves, not their weights. ``sf_diff`` and ``rmst_diff`` of
weighted estimates use their robust variances.

Bootstrap bounds of a weighted fit
----------------------------------

``method="bootstrap"`` on ``cb``, ``param_cb`` and ``quantile_cb`` (and
``mrl_cb`` of a distribution) of a weighted parametric fit or a weighted
parametric regression is the *case* bootstrap: each resample draws the
units with replacement, each keeping its own weight, covariates, time,
censoring and entry time, and refits the weighted model. The bounds are
the percentiles of the refits, corrected for bias as for unweighted data
(and, for the regressions, for skewness: the BCa interval). They do not
lean on the normal approximation the Wald bounds make, so they are the
ones to compare with when the sample is small or the weights are uneven.
The parametric bootstrap of unweighted data simulates from the fitted
model, which knows nothing of how the units were sampled; resampling the
units does. A seed gives the same bounds at any scale of the weights.

.. jupyter-execute::

    print(model.param_cb("beta"))
    print(model.param_cb("beta", method="bootstrap", n_boot=200, random_state=1))

Since nothing is simulated, left- and interval-censored data are
resampled as they are. A limited-failure or zero-inflated fit with
weights has no bootstrap (without weights it has, with BCa intervals; see
:doc:`Parametric SurPyval Modelling`); discrete fits and time-varying
covariate fits have none, weighted or not.

Interval censored data: the weighted Turnbull estimate
------------------------------------------------------

``Turnbull.fit`` takes ``w`` too, for data the Kaplan-Meier family cannot
handle: inspection intervals, left censoring and right truncation. The
estimate is the one that best explains the population the weights
describe: each unit counts as ``w`` units when the estimate shares out
the chance of failing among the intervals the unit could have failed in.
Multiplying every weight by the same number changes nothing, whole-number
weights give the same curve as counting each row that many times
(``n``), and on right censored data its Kaplan-Meier option is the
weighted Kaplan-Meier.

What it does not have is a formula for its uncertainty. Greenwood's
formula would read the weights as units seen, and the robust
(infinitesimal jackknife) variance of the weighted Turnbull estimate is
not built. So ``cb``, ``quantile_cb``, the bounds of ``rmst`` and
``sf_diff`` refuse it, and its bounds come from ``bootstrap_cb``: the
units (a row with ``n = k`` is ``k`` units) are drawn with replacement,
each keeping its weight, the estimate is refitted, and the bounds are
the percentiles of the refits. Give a seed (``random_state``) and the
same bounds come back, at any scale of the weights. ``plot`` draws the
curve without bounds.

.. jupyter-execute::

    # Inspected every 2 time units: a failure is known to lie in (2k, 2k + 2]
    left = np.where(c == 1, x, np.floor(x / 2) * 2)
    right = np.where(c == 1, np.inf, left + 2)
    tb = surv.Turnbull.fit(xl=left, xr=right, w=w)
    print(tb.sf([4, 8]))
    print(tb.bootstrap_cb([4, 8], n_boot=200, random_state=1))

A probability plot of weighted left or interval censored or truncated
data uses these weighted Turnbull positions (``heuristic="Turnbull"``).

In symbols: with :math:`\alpha_{ij} = 1` when piece :math:`j` lies in
row :math:`i`'s interval, the self-consistency (EM) step is

.. math::

    p_j \leftarrow \frac{1}{\sum_i n_i w_i}
        \sum_i n_i w_i \frac{\alpha_{ij} p_j}{\sum_k \alpha_{ik} p_k},

the unweighted step with :math:`n_i w_i` in place of :math:`n_i`; delayed
entry and right truncation add their unseen "ghost" units in the same
way, each carrying its row's :math:`n_i w_i`. Without truncation the
default algorithm is the EM-ICM, as R's ``icenReg::ic_np``, whose
``weights`` give the same estimate (checked to 1e-8). The
Fleming-Harrington and Nelson-Aalen options are applied to the weighted
expected numbers at risk and of failures, with Fleming-Harrington's tied
failures at their mean weight as in the weighted ``FlemingHarrington``.
The coverage of the 95% bootstrap bounds under informative sampling was
0.946 and 0.930 (Monte Carlo error 0.01) at two times in 500 simulated
samples (``scripts/studies/weighted_turnbull/coverage.py``).

Comparing weighted groups
-------------------------

``logrank`` and ``pairwise_logrank`` take ``w``. The numerator compares
the weighted observed and expected events in each group; the variance is
the design-based one, the sum of the units' squared weighted score
residuals, so the test is R's robust score test,
``coxph(Surv(x, d) ~ group, weights = w, robust = TRUE, ties =
"breslow")$rscore``. Every weighting, ``strata``, ``tl`` and ``t0`` take
weights, and the result does not depend on their scale. A weighted test
answers the population question the weights describe; an unweighted test
of weighted data answers a question about the sample, and a test that
weighted the counts without this variance would be too confident.

.. jupyter-execute::

    group = (rng.uniform(size=kept.sum()) < 0.5).astype(int)
    print(surv.logrank(x, group, c=c, w=w))

``SurpyvalData`` holds the weights too (``SurpyvalData(x, c, n, w=w)``,
``SurpyvalData.from_dataframe(df, ..., w_col="weight")``), and
``fit_from_df`` takes ``w_col`` (Cox's too: a row dropped for a missing
covariate drops its weight with it).

A Cox model with time-varying covariates takes weights as well,
``CoxPH.fit_tvc(i, xl, xr, c, Z, w=w)`` (and ``fit_tvc_from_df(...,
w_col=)``), one weight per interval row: give each of a subject's rows
the subject's weight. Its robust covariance clusters the rows by subject,
as R's ``coxph(Surv(start, stop, event) ~ z, weights = w, robust = TRUE,
id = id)``, so splitting a subject's follow-up into episodes on which
nothing changes gives the same fit and the same standard errors.

Competing-risks regressions
---------------------------

Both competing-risks regressions take ``w``:
``CompetingRisksProportionalHazards.fit(x, Z, e, w=w)`` (``model="Cox"``
or ``"Fine-Gray"``), ``FineGray.fit(x, Z, e, event=..., w=w)`` and their
``fit_from_df(..., w_col=)``.

- **Cause-specific Cox** (``model="Cox"``) is one weighted Cox model per
  cause, the other causes censored, exactly as ``CoxPH.fit(..., w=w)``:
  R's ``coxph(Surv(x, e == k) ~ z, weights = w, robust = TRUE)`` for each
  cause ``k``, with R's tie convention and the robust (Lin-Wei) standard
  errors.
- **Fine-Gray** keeps a unit that failed from another cause in the risk
  set of the cause of interest, down-weighted by how likely it is still
  to be followed (``G(t-) / G(x-)``, ``G`` the probability of not yet
  being censored). With weights, each unit counts ``w`` times that. ``G``
  is estimated from the weighted censoring times, so a whole-number
  weight gives exactly the fit of the row repeated that many times. The
  standard errors are robust, each unit's contributions over its time at
  risk summed (clustered by unit).

R's ``survival::finegray(..., weights = w)`` multiplies the weights into
its censoring weights the same way but estimates ``G`` without the
weights; on the reference data the two differ in the third decimal of
the coefficients. ``cmprsk::crr`` takes no weights. The weighted
Fine-Gray standard errors, like those of ``coxph`` on ``finegray``'s
data, hold ``G`` fixed: they leave out the part that comes from
estimating it. The fit without weights includes it (Fine and Gray's own
sandwich, its default), so a fit with ``w = 1`` and a fit without ``w``
give the same coefficients and slightly different standard errors unless
nothing is censored. A weighted fit has no model-based covariance:
``covariance(robust=False)`` is refused.

The information criteria, the diagnostics (``diagnose()``, the
proportional subdistribution hazards test) and the penalised fit refuse
weighted fits; ``summary()`` gives robust Wald tests and intervals, and
the printout the effective sample size.

.. jupyter-execute::

    from surpyval.univariate.competing_risks import FineGray

    # Two causes of failure, and the group of the log-rank example above
    kind = rng.uniform(size=x.size) < 0.6
    cause = np.where(c == 1, None, np.where(kind, "wear", "shock"))
    fg = FineGray.fit(x, group, cause, event="wear", w=w)
    print(fg)

In formulas: the weighted Fine-Gray estimate solves

.. math::

    \sum_i n_i w_i \delta_i \left(Z_i - \bar Z(x_i)\right) = 0,
    \qquad
    \bar Z(t) = \frac{\sum_j n_j w_j W_j(t) e^{\beta' Z_j} Z_j}
                      {\sum_j n_j w_j W_j(t) e^{\beta' Z_j}},

with ``W_j(t)`` 1 while unit ``j`` is under observation,
``G(t-)/G(x_j-)`` after it failed from another cause and 0 otherwise, and
``G`` the Kaplan-Meier of the censoring with each unit counted ``n w``.
The covariance is ``A^-1 (sum_i n_i w_i^2 L_i L_i') A^-1``, ``A`` the
weighted information and ``L_i`` unit ``i``'s score residual over its
whole time at risk, as for Cox.

Distributional and Royston-Parmar regression with weights
---------------------------------------------------------

Distributional regression (``WeibullDistributional.fit(..., w=w)``, any
``Distributional(...)`` fitter) and the Royston-Parmar spline regression
(``RoystonParmarPH.fit(..., w=w)``, ``RoystonParmarPO``,
``RoystonParmarRegression(...)``) take weights as the other parametric
regressions do: the estimates maximise the weighted log-likelihood, every
row's term counted ``n * w`` times, and the covariance is the robust
(sandwich) one. ``fit_from_df`` takes ``w_col``; a row dropped for a
missing covariate drops its weight with it. The Wald bounds
(``cb``, ``param_cb``, ``quantile_cb`` and a distributional model's
``parameter_cb``) use the robust covariance, ``method="bootstrap"`` is
the case bootstrap, and the likelihood-ratio bounds, the
``likelihood_ratio_test()`` and the information criteria refuse.

A Royston-Parmar fit places its knots on the event times counted by
``n``, *not* weighted, so the spline's shape is the same whatever the
weights and their scale, and ``w = 1`` gives the unweighted fit exactly.
A weight of 2 is therefore not quite a count of 2 here: a count moves the
quantiles the knots sit at and a weight does not. Pass ``knots`` to fix
them; with the same knots, whole-number weights give the fit of the rows
counted that many times. R's ``flexsurvspline(..., weights = w)`` places
its default knots at weighted quantiles found by an approximate expansion
of the weights (each rounded to a multiple of a hundredth of a unit), so
pass the knots to it too when comparing. A left truncation at 0 (the
usual "no late entry") is read as none.

.. jupyter-execute::

    load = rng.uniform(0, 1, kept.sum())  # no effect: coefficients near 0
    model = surv.WeibullDistributional.fit(
        x, load, c=c, w=w, covariates={"alpha": [0], "beta": [0]}
    )
    print(model.params.round(3), model.standard_errors().round(3))

Checked against R: ``flexsurvreg(..., anc = list(shape = ~ z1), weights =
w)`` and ``flexsurvspline(..., weights = w)`` on both scales for the
estimates (to 1e-5) and the log-likelihood, and a sandwich computed by
hand from numDeriv's per-row scores and Hessian for the robust standard
errors (to 1e-4 relative); flexsurv's own standard errors with weights are
model based.

Competing risks: weighted cumulative incidence
----------------------------------------------

``CompetingRisks.fit(x, e, c, n, w=w)`` (and ``fit_from_df(...,
w_col=)``) gives the Aalen-Johansen cumulative incidence of each cause in
the population the weights describe. Each unit counts ``n * w`` in the
numbers at risk and of failures from each cause, so whole-number weights
give exactly the estimate of the same rows counted that many times, and
``w = 1`` gives the unweighted estimate. The bounds of ``cb`` (on the
cumulative incidence, and on ``sf``, ``ff`` and ``Hf``) use the robust,
infinitesimal-jackknife variance, the one R's ``survfit(Surv(x,
factor(e)) ~ 1, weights = w)`` reports as ``std.err``, so they do not
change when every weight is multiplied by the same number. With ``w = 1``
they are the robust bounds, close to but not the same as those without
``w`` (which use Aalen's variance). The fit warns if the effective sample
size is below 20, and its printout shows it.

In symbols, with :math:`\hat S` the weighted all-cause Kaplan-Meier
survival, :math:`h_{jk} = d_{jk} / r_j` cause :math:`k`'s weighted hazard
increment at :math:`t_j` and :math:`V_i` unit :math:`i`'s influence on
:math:`\log \hat S` (the formula below), unit :math:`i`'s influence on
cause :math:`k`'s cumulative incidence is

.. math::

    U_{ik}(t) = \sum_{t_j \le t} \hat S(t_{j-1}) \left[
        \frac{dN_{ik}(t_j) - Y_i(t_j)\, h_{jk}}{r_j}
        + h_{jk}\, V_i(t_{j-1}) \right],

and the variance is :math:`\sum_i n_i w_i^2 U_{ik}(t)^2`. The estimate
takes no delayed entry, weighted or not; Gray's test refuses weights (see
below).

Penalised fits with weights
---------------------------

``CoxPH.fit_penalised``, ``CoxPH.fit_tvc_penalised`` and the parametric
families' ``fit_penalised`` (``WeibullPH``, ``WeibullAFT``, ``WeibullPO``
and the others) take ``w``. A penalised fit divides the log-likelihood by
the number of units before adding the penalty; with weights that number is
the weighted one, :math:`N_w = \sum n w`, and the covariates are
standardised with the weighted mean and standard deviation (divisor
:math:`N_w`). Both scale with the weights, so the fit depends only on their
proportions, and glmnet's Cox fit with ``weights`` is the same objective
(checked to 1e-6). Penalised estimates have no standard errors, weighted or
not, and the information criteria stay refused. The details are in
:doc:`Regression Modelling with SurPyval` (Penalised regression). The other
models' penalised fits (competing risks, recurrent events, Royston-Parmar,
distributional) refuse ``w``.

What a weighted fit will not do
-------------------------------

Some bounds and methods are wrong with weights, and refuse with a message
saying what to use instead:

- **Likelihood-ratio bounds** (``method="lr"``): the likelihood-ratio
  statistic of a weighted likelihood is not chi-squared. Use the Wald
  bounds (the default) or the case bootstrap (``method="bootstrap"``).
- **AIC, BIC, AICc and** ``fit_best``: a weighted fit maximises a
  *pseudo*-likelihood, which does not compare models.
- ``likelihood_ratio_test()`` of a regression or Cox model: the ratio of
  weighted pseudo-likelihoods is not chi-squared (it grows with the scale
  of the weights). Use the Wald tests of ``summary()``, which are on the
  robust covariance.
- ``life_table`` of weighted data or a weighted estimate: use
  ``KaplanMeier.fit(data)``, or the arrays without the weights for the
  table of the units themselves.
- **Bayesian fits**: ``fit_bayes``, ``update``, ``prior.from_data``,
  ``log_likelihood_fn``, ``BayesianParametric.from_draws`` and the PyMC
  helpers (``log_likelihood``, ``censored_likelihood``, ``to_surpyval``)
  refuse weighted data, since a posterior from the weighted likelihood
  would treat the weights as units seen. Fit by maximum likelihood with
  the weights instead.
- **Simultaneous bands and** ``bootstrap_cb`` of a Kaplan-Meier,
  Nelson-Aalen or Fleming-Harrington estimate: use ``cb``. A weighted
  Turnbull estimate is the other way round: ``cb`` refuses and
  ``bootstrap_cb`` gives its bounds (above).
- **Rank-based plotting positions** (Blom, median ranks, ...): a weighted
  model's probability plot uses the weighted Kaplan-Meier, Nelson-Aalen,
  Fleming-Harrington or Turnbull positions.
- **Methods other than maximum likelihood** (``how="MPP"``, ``"MOM"``,
  ``"MSE"``, ``"MPS"``).
- The Cox model's residuals and ``diagnose()``, strata and
  exact ties. These may come later.
- **Gray's test** (``gray_test``) of weighted data: R's ``cmprsk`` has no
  weighted version to check one against. Use ``logrank(..., w=w)`` on the
  cause-specific hazards, or compare the groups' weighted cumulative
  incidences by their bounds.
- **Aalen's** ``constant_effects_test()``: its null distribution is
  simulated for an unweighted fit, and a weighted version has not been
  built or checked. Judge a constant effect from the robust bands of
  ``plot()`` or ``cumulative_coefficients_cb()`` (a straight line through
  0 inside the band), or test it on the data without ``w``.
- **The fitters that do not take** ``w``: the parametric competing-risks
  models, recurrent events, degradation, mixtures and the frailty models,
  among others, which refuse weighted data (the Bayesian fits and the
  life table are above).
- **Reshaping weighted data** with ``censor_at``, ``select`` and the other
  record-based methods: index the rows, ``data[rows]``, which keeps the
  weights, or build a new ``SurpyvalData`` from the arrays.

Two cautions:

- **Estimated weights.** When the weights are themselves estimated (inverse
  probability of treatment weights from a logistic regression), the robust
  variance ignores that estimation and is usually a little conservative:
  intervals slightly too wide. This is what R and lifelines do too.
- **Complex survey designs.** With ``w`` alone, units are independent draws
  with unequal weights. Strata, clusters of units sampled together and
  finite-population corrections are out of scope; for a Cox model or a
  parametric regression, ``robust_covariance(cluster)`` (and
  ``summary(robust=True, cluster=...)``) gives the variance for units
  sampled in clusters, and R's ``survey`` package handles full designs.

The formulas
------------

Kaplan-Meier, Nelson-Aalen and Fleming-Harrington use weighted numbers at
risk and of deaths, :math:`r_k = \sum_i n_i w_i Y_i(t_k)` and
:math:`d_k = \sum_i n_i w_i dN_i(t_k)`. Their variance is the
infinitesimal jackknife (R's ``survfit(..., weights = w, robust = TRUE)``):
with :math:`U_i(t)` the derivative of :math:`\log \hat S(t)` in unit
:math:`i`'s weight,

.. math::

    \widehat{Var}\,\log \hat S(t) = \sum_i n_i w_i^2 U_i(t)^2,
    \qquad
    U_i^{KM}(t) = -\sum_{t_k \le t}
        \frac{dN_i(t_k) - Y_i(t_k)\, d_k / r_k}{r_k - d_k}

(with :math:`r_k` in place of :math:`r_k - d_k` for Nelson-Aalen).
Fleming-Harrington's increment with :math:`m_k` tied deaths takes each at
the deaths' mean weight, :math:`\sum_{l < m_k} a_k / (r_k - l a_k)`,
:math:`a_k = d_k / m_k`, as R's ``ctype = 2`` does. The restricted mean's
variance is the same jackknife of the area under the curve.

A parametric or regression fit maximises the pseudo-likelihood
:math:`\sum_i n_i w_i \ell_i(\theta)`. With :math:`H` the Hessian of its
negative and :math:`s_i` the score of one unit of row :math:`i`, the robust
covariance is the sandwich (Huber, 1967; Binder, 1983)

.. math::

    V = H^{-1} \Big( \sum_i n_i w_i^2 s_i s_i^\top \Big) H^{-1},

as R's ``survreg(..., weights = w, robust = TRUE)`` computes it, and every
Wald bound follows from it by the delta method. ``covariance()`` holds it,
and so a regression's ``robust_covariance()`` and ``summary(robust=True)``
are the same. A parametric model's ``hess_inv`` is :math:`H^{-1}` itself,
the inverse Hessian of the weighted log-likelihood with the weights as given
(R's ``naive.var``): it is not the variance of the estimate, and nothing in
SurPyval uses it for one, so read ``covariance()`` or ``standard_errors()``
for a variance; with
``cluster`` the meat sums each cluster's :math:`n_i w_i s_i` first, as
``survreg`` with ``cluster()`` does.

A saved weighted model (``to_dict``, ``to_json``) keeps each matrix under
its own name: ``"hess_inv"`` is :math:`H^{-1}` and ``"covariance"`` is
:math:`V`. It is stamped schema 4, which SurPyval 0.26 refuses with a
request to upgrade (0.26 saved :math:`V` as ``"hess_inv"``, and would read
the smaller :math:`H^{-1}` as the variance). A weighted model saved by
0.26 still loads, with its robust bounds and no ``hess_inv``, which it did
not save. Weighted data (``SurpyvalData`` with ``w``) is schema 3.

The Cox model maximises the weighted partial likelihood, with Efron's tie
correction taking the :math:`m` tied deaths at their mean weight (R's
``coxph`` convention), and its covariance is the Lin-Wei sandwich
:math:`\sum_i n_i w_i^2 D_i^\top D_i`, :math:`D_i` the dfbeta of unit
:math:`i` (R's ``coxph(..., weights = w, robust = TRUE)``); within clusters
the weighted dfbetas are summed first. R's ``survey::svycoxph`` gives the
same coefficients and standard errors larger by
:math:`\sqrt{n / (n - 1)}`, a finite-sample factor SurPyval, as R's
``survival``, does not apply.

Aalen's additive hazards model with weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In plain words: at each event time Aalen's model fits a straight-line
(least-squares) regression of "who failed now" on the covariates of
everyone still at risk. With weights it is a *weighted* least-squares
fit, each unit counting as much as its weight says, and the running sum
of the fitted slopes is the cumulative effect :math:`B(t)`. Its
uncertainty is measured by how much each unit moves :math:`B(t)`, unit by
unit, so it does not depend on how the weights are scaled.

At each event time, with :math:`X` the rows at risk (a leading 1 for the
baseline), :math:`W` the diagonal of :math:`n_i w_i` over them, :math:`P`
the ridge penalty of ``penalizer`` (0 by default) and :math:`dN` the
events,

.. math::

    dB(t) = (X^\top W X + P)^{-1} X^\top W\, dN(t),

and the robust variance of :math:`B(t)` sums each unit's squared
influence, its martingale residuals mapped as the increments are:

.. math::

    \psi_i(t) = \sum_{s \le t} (X^\top W X + P)^{-1} X_i w_i
        \bigl(dN_i(s) - Y_i(s) X_i^\top dB(s)\bigr), \qquad
    \widehat{Var}\, B(t) = \sum_i n_i \psi_i(t) \psi_i(t)^\top.

This is R's ``timereg::aalen(..., weights = w, robust = 1)``
(``robvar.cum``), checked to about :math:`10^{-15}`. Every bound of the
model (``cumulative_coefficients_cb``, ``summary`` and ``cb`` on ``sf``,
``ff`` and ``Hf``) takes it. With a penalty, :math:`P` is ``penalizer``
times each covariate's weighted sum of squares about its weighted mean,
over :math:`\sum_i n_i - 1`, so that it grows with the weights as
:math:`X^\top W X` does and the fit does not depend on their scale; the
price is that whole-number weights then differ from counts (without a
penalty they are the same, to the last digit).

The robust formula: HC0, with counts as frequency weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every robust (sandwich) standard error in SurPyval, with weights or from
``robust_covariance()`` and ``summary(robust=True)`` without them, is the
plain sandwich above, known as **HC0**: no small-sample correction is
applied. R's ``survival`` (``survreg`` and ``coxph`` with ``robust =
TRUE``), R's ``sandwich`` package by default and lifelines give the same.
R's ``survey`` package multiplies each standard error by
:math:`\sqrt{n / (n - 1)}`, :math:`n` the number of units (or of clusters
with ``cluster``). The difference is small unless the sample is: about
0.5% at :math:`n = 100` (:math:`\sqrt{100/99} = 1.005`) and about 5% at
:math:`n = 10` (:math:`\sqrt{10/9} = 1.054`). To match ``survey``, multiply
SurPyval's standard errors by that factor.

The counts ``n`` are **frequency weights**: a row with ``n = 3`` is three
units, in the sandwich as in the fit. Without ``cluster`` each of the three
is a unit (a group) of its own; with ``cluster`` the three are members of
the row's cluster. Either way the result is that of the data written out
one row per unit, so counting does not change a robust standard error.

Weights and counts with tied failure times (Cox, Efron)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a weighted Cox model with tied failure times, Efron's method gives a
different answer for one subject with weight 3 than for three identical
copies of that subject (or one row with ``n = 3``). The copies are three
deaths tied at one time, and Efron's correction shrinks the risk set as
each tied death is processed; a single weighted death is one death, and
there is nothing to shrink. R's ``coxph`` does the same (SurPyval follows
it).

A worked example: two subjects, A and B, at equal risk; A dies, B is still
at risk. Each death adds a term :math:`\log(\text{its risk} /
\text{the risk set's total})` to the partial log-likelihood, here with
every risk 1.

- **A with weight 3** (one row, ``w = 3``): one death of weight 3 against
  a risk set of weight :math:`3 + 1 = 4`, so the death term is
  :math:`-3 \log 4 = -4.16`.
- **Three tied copies of A** (three rows, or ``n = 3``): three deaths, the
  risk set 4, then 3, then 2 as Efron processes them, so the term is
  :math:`-\log 4 - \log 3 - \log 2 = -3.18`.
- **With Breslow's ties** (``tie_method="breslow"``) both are
  :math:`-3 \log 4`: Breslow does not shrink the risk set within a tie.

So use the counts ``n`` for repeated identical units (several pumps that
failed at the same inspection) and the weights ``w`` for survey and
importance weights, where one unit stands for others that were not seen.
The test ``surpyval/tests/univariate/regression/test_weighted_cox_ties.py``
checks these numbers against SurPyval's fits and R's ``coxph``.

Weights and counts in the log-rank test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The log-rank test does **not** have the tie problem of the Cox model above.
Its weighted observed-minus-expected numerator treats a tied death of
weight 3 and three tied copies the same, for every weighting (log-rank,
Fleming-Harrington, Gehan, Tarone-Ware, Peto-Peto): there is no
shrinking risk set. What differs is how sure the test says it is.

- **Counts** ``n`` give exactly the answer of writing each row out ``n``
  times, statistic, degrees of freedom and p-value alike (checked to
  rounding, ``1e-10``, with no ties, heavy ties and late entry,
  ``pairwise_logrank`` too). The classical (hypergeometric) variance treats
  them as that many units seen, which is right for counts.
- **Weights** ``w`` give the same numerator but a larger variance, so a
  smaller statistic and a larger p-value than the copies. The variance is
  the robust one, which counts the *units*, not the weights: ten copies of
  a unit are ten independent pieces of evidence, but one unit with weight
  10 is still one. Roughly, the statistic is the copies' times the
  effective sample size over the total weight,
  :math:`(\sum w)^2 / \sum w^2 \,/\, \sum w`, so it is lower the more
  uneven the weights (about a third of the copies' statistic for weights
  spread evenly over 1 to 4). Multiplying the weights by a constant changes
  nothing, as it should.

Use ``n`` for repeated identical units and ``w`` for sampling or inverse
probability weights, where a unit stands for others that were not seen:
treating those as copies would claim evidence nobody collected. Two smaller
differences: a test with all weights equal to 1 is the robust test, which
differs a little from the classical test without ``w`` (more with tied
times, as the classical variance has a tie correction the robust one has
not), and Gehan's, Tarone-Ware's and Peto-Peto's weights use the numbers at
risk scaled to a mean weight of 1 per unit, so they too do not match the
copies. Gray's test for competing risks takes counts ``n`` only, and they
equal copies.

Translating from lifelines and R
--------------------------------

- lifelines ``KaplanMeierFitter().fit(T, E, weights=w)`` is
  ``KaplanMeier.fit(T, 1 - E, w=w)``; ``CoxPHFitter().fit(df, "T", "E",
  weights_col="w", robust=True)`` is ``CoxPH.fit_from_df(df, "T", Z_cols,
  c_col=..., w_col="w")``, with the flag column holding ``1 - E``.
- R ``survfit(Surv(T, E) ~ 1, weights = w, robust = TRUE)`` is
  ``KaplanMeier.fit(T, 1 - E, w=w)``; ``survreg(Surv(T, E) ~ z, weights =
  w, robust = TRUE)`` is ``WeibullAFT.fit(T, z, c=1 - E, w=w)`` (on
  SurPyval's parameterisation); ``coxph(Surv(T, E) ~ z, weights = w, robust
  = TRUE)`` is ``CoxPH.fit(T, z, c=1 - E, w=w)``; per cause, ``coxph(Surv(T,
  cause == k) ~ z, weights = w, robust = TRUE)`` is
  ``CompetingRisksProportionalHazards.fit(T, z, e, w=w)``;
  ``timereg::aalen(Surv(T, E) ~ z, weights = w, robust = 1)`` is
  ``AalenAdditive.fit(T, z, c=1 - E, w=w)`` (pass ``weights = df$w``:
  ``aalen`` looks for a bare ``w`` outside the data).

The reference tests (``surpyval/tests/reference/test_weights.py``) compare
these with R ``survival`` and ``survey`` and with lifelines, and the
calibration study (``surpyval/tests/calibration/test_coverage_weights.py``)
checks the intervals' coverage under informative sampling, at any scale of
the weights, and with estimated treatment weights.
