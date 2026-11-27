Bayesian Fitting with SurPyval
==============================

With plenty of failures, the data say what a life distribution is. With
three failures they say much less, but the engineer, the clinician or the
manufacturer often knows more than the data show: "wear-out, a Weibull
shape of about 2", "a B10 of around 5,000 hours", "it is the same design
as last year's fleet". A Bayesian fit puts that knowledge, the **prior**,
together with the data, by Bayes' rule, and gives back a model that is
used like any other SurPyval model.

``Weibull.fit_bayes(x, c, priors=...)`` takes the data exactly as
``Weibull.fit`` does (any censoring, truncation, counts and intervals) and
returns a :class:`~surpyval.bayes.BayesianParametric`:

- its ``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``, ``mean`` and
  ``random`` describe **the life of the next unit**, with the uncertainty
  about the parameters averaged in (the *posterior predictive*);
- ``cri``, ``quantile_cri``, ``mean_cri`` and ``param_cri`` say **how sure
  we are**: credible intervals, read as "there is a 95% chance the B10
  lies between these two values, given the prior and the data".

A credible interval is not a confidence interval. A confidence interval is
a statement about the method (95% of such intervals contain the truth); a
credible interval is a statement about this parameter, given what was
believed and what was seen. A Bayesian model's ``cb`` refuses and points to
``cri``.

Few failures and a belief about the shape
-----------------------------------------

Forty pump seals: three have failed, at 2,100, 3,400 and 4,050 hours, and
37 are still running. The seals wear out, and from the manufacturer's data
and experience the Weibull shape is "about 2, somewhere between 1.5 and
3".

.. jupyter-execute::

    import numpy as np
    import surpyval as sp

    x = np.r_[2100, 3400, 4050, np.linspace(1500, 6000, 37)]
    c = np.r_[0, 0, 0, np.ones(37)]

    mle = sp.Weibull.fit(x, c)
    print(mle.param_cb("beta").round(2))

With three failures, the maximum-likelihood fit's 95% confidence interval
on the shape runs from about 1 to 6.5: the data alone can hardly tell
early failures from wear-out.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _cb = mle.param_cb("beta")
    assert round(_cb[0]) == 1 and round(_cb[1], 1) == 6.5, _cb

Say what is believed, and fit:

.. jupyter-execute::

    belief = {"beta": sp.prior.around(2, between=(1.5, 3))}
    model = sp.Weibull.fit_bayes(x, c, priors=belief, unit="hours")
    model

The summary reads from the top: the data, the priors as they were built
("about 2, 90% between 1.5 and 3" is a LogNormal whose 5% and 95% points
are 1.5 and 3), the posterior's medians and 95% credible intervals, the
B10 life and the mean life with theirs, and what the prior did: it
narrowed the shape's interval by about 70% against the maximum-likelihood
interval, while the scale, which was given no prior, came from the data
alone. The last line says how it was computed (below).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lo, _hi = model.param_cri("beta")
    _mlo, _mhi = mle.param_cb("beta")
    assert round(100 * (1 - (_hi - _lo) / (_mhi - _mlo)), -1) == 70

The model answers the usual questions. The chance that a new seal
survives 5,000 hours, with the 95% credible interval on that reliability:

.. jupyter-execute::

    print(model.sf(5000).round(3), model.cri(5000).round(3))

The B10 life comes in two forms, and they answer different questions:

.. jupyter-execute::

    print(model.qf(0.1).round(0))               # 10% of new seals fail by
    print(model.quantile_cri(0.1, point=True).round(0))  # our best B10
    print(model.quantile_cri(0.1).round(0))    # its 95% credible interval

``qf(0.1)`` is the age by which 10% of new seals will have failed, all
things considered: the uncertainty in the parameters is averaged in, which
spreads the distribution and pulls its early quantiles in. It is what a
spares forecast or a reliability block diagram needs. ``quantile_cri``
is about the population's B10 itself: its posterior median, the single
best guess, and the interval it lies in. With few failures the two differ
(here about 4,700 against 4,900 hours).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(float(model.qf(0.1)), -2) == 4700, model.qf(0.1)
    _b10 = float(model.quantile_cri(0.1, point=True))
    assert round(_b10, -2) == 4900, _b10

``model.plot()`` draws the data on Weibull paper with the predictive
curve, its credible band and, dashed, the maximum-likelihood fit, so what
the prior did is visible at a glance; ``model.plot_posterior()`` draws
each parameter's prior and posterior, and their joint contours.

.. jupyter-execute::

    ax = model.plot()

.. jupyter-execute::

    fig = model.plot_posterior()

A belief about a B-life
-----------------------

People seldom think in shapes and scales. "Our B10 is around 5,000 hours;
we'd be surprised if it were under 3,000 or over 8,000. And it wears out."
``prior.on_quantities`` takes the belief as said, one statement for each
free parameter:

.. jupyter-execute::

    belief = sp.prior.on_quantities(
        sp.Weibull,
        B10=sp.prior.around(5000, between=(3000, 8000)),
        beta=sp.prior.between(1.2, 4),
    )
    sp.prior.check(sp.Weibull, belief)

``prior.check`` shows what the prior says before any data, in the terms
people use. Here its B10 is between 3,000 and 8,000 hours with 90%
probability, as said, and so the prior is exactly the belief: SurPyval
carries the density of the B10 and the shape to the Weibull's parameters
by the change of variables, with nothing approximated. Run it on any
prior: a "vague" prior often says something nobody believes (a LogNormal
with a spread of 10 on alpha puts B10 anywhere from a nanosecond to
geological time).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = sp.prior.check(sp.Weibull, belief).table["B10"]
    assert round(_t[1], -2) == 3000 and round(_t[2], -2) == 8000, _t

.. jupyter-execute::

    sp.Weibull.fit_bayes(x, c, priors=belief, unit="hours")

The same works with the reliability at a time (``R_at={2000:
sp.prior.between(0.9, 0.99)}``), with the mean or median, or with any
quantile (``quantile={0.5: ...}``).

Last year's fleet as the prior for a new design
-----------------------------------------------

Last year's fleet of 60 units gave a good Weibull fit. This year's units
are a new design, so far with two failures among 20:

.. jupyter-execute::

    last_year = sp.Weibull.fit(sp.Weibull.random(60, 9000, 1.8,
                                                 random_state=2025))
    x_new = np.r_[3900, 6100, np.linspace(1000, 5000, 18)]
    c_new = np.r_[0, 0, np.ones(18)]
    print(sp.Weibull.fit(x_new, c_new).param_cb("beta").round(1))

Two failures say little: the maximum-likelihood shape's interval runs
from about 3.6 to 20. The new design is similar, but not the same, so the
old fleet is counted as half as much evidence:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _cb = sp.Weibull.fit(x_new, c_new).param_cb("beta")
    assert round(_cb[0], 1) == 3.6 and round(_cb[1]) == 20, _cb

.. jupyter-execute::

    prior = sp.prior.from_model(last_year, weight=0.5)
    new = sp.Weibull.fit_bayes(x_new, c_new, priors=prior)
    new

``from_model`` takes the old fit's estimates and their covariance (how
sure they were and how they move together), so a fit from 60 failures
counts for a lot and one from five for little; ``weight`` discounts it
(0.5 counts it as half the evidence) and ``widen`` loosens it further; and
``param="beta"`` takes only one parameter's prior from it. The weight is a
judgement: try two and compare. Where the old data are at hand,
``prior.from_data(x_old, c_old, weight=0.5)`` uses their likelihood itself
(a *power prior*), which is exact, and better with few old failures. A
model known only by its numbers (a handbook, a data sheet) is
``prior.from_params(sp.Weibull, [9000, 1.8], within=0.2)``: each within
about 20%.

Is the new design's B10 longer than the old one's?

.. jupyter-execute::

    new.compare_to(last_year, on="qf", p=0.10)

The answer is a probability, with the assumption it rests on.

As data arrive
--------------

``update`` adds data to a Bayesian fit. It keeps the original prior and
all the data, so updating in steps gives exactly the posterior of fitting
everything at once:

.. jupyter-execute::

    first = sp.Weibull.fit_bayes(x[:20], c[:20], priors={
        "beta": sp.prior.around(2, between=(1.5, 3))})
    both = first.update(x[20:], c[20:])
    print(np.allclose(both.params, model.params))

Units that were running at the last update and are still running (or have
since failed) are the same units: pass the current records with
``replace=True``, so that no unit counts twice.

Comparing fits: WAIC and leave-one-out
--------------------------------------

Which belief predicts the seals better: wear-out (the shape about 2),
random failures (an Exponential), or early failures (a shape below 1)?
AIC compares maximum-likelihood fits and ignores the priors. For Bayesian
fits the question is how well each predicts a seal it has not seen,
averaged over its posterior: the *expected log predictive density*
(``elpd``), estimated by leaving out each unit in turn (``loo()``) or by
WAIC (``waic()``). Higher is better; ``ic`` is ``-2 elpd``, on AIC's
scale.

.. jupyter-execute::

    random = sp.Exponential.fit_bayes(
        x, c, priors={"failure_rate": sp.prior.between(1e-6, 1e-3)})
    early = sp.Weibull.fit_bayes(
        x, c, priors={"beta": sp.prior.around(0.8, between=(0.5, 1.2))})
    comparison = sp.bayes.compare(
        {"wear-out": model, "random": random, "early": early})
    comparison

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = comparison.table
    assert comparison.best == "wear-out"
    assert round(_t["random"]["elpd_diff"], 1) == -0.9, _t
    assert round(_t["random"]["se_diff"], 1) == 0.7, _t
    assert round(_t["early"]["elpd_diff"], 1) == -1.5, _t

Wear-out comes first, but by less than one and a half standard errors of
the difference from random failures: three failures cannot tell the two
apart. Read ``se_diff``, not ``se``: each fit's own standard error is
large (the three failures and 37 survivors predict very differently),
but the fits are scored on the same seals, and the paired difference
cancels most of it.

Every row counts through its own likelihood term (a survivor's
probability of surviving, an interval's probability, a truncated row's
term over the probability of its window), and a row with count ``n`` is
``n`` units. Each fit is scored from its one posterior, without
refitting, by :math:`p(y_i \mid y_{-i}) = 1 / E[1 / p(y_i \mid \theta)]`:

- a conjugate fit in closed form, exactly;
- a grid fit by the grid's quadrature. ``edge`` is each row's
  share of its leave-one-out posterior in the grid's outermost cells, and
  ``loo()`` warns above 0.001: without that row the data say much less,
  and the grid, laid for the full posterior, may not hold it;
- a Laplace fit's draws, and draws from another sampler
  (``from_draws``), by Pareto-smoothed
  importance sampling (Vehtari, Gelman and Gabry, 2017; Vehtari, Simpson,
  Gelman, Yao and Gabry, 2024), as R's ``loo`` and ArviZ do, taking the
  draws as independent. ``pareto_k`` is each row's diagnostic, and
  ``loo()`` warns, counting the rows, above
  :math:`\min(1 - 1/\log_{10} S, 0.7)` for :math:`S` draws.

WAIC is :math:`\mathrm{lppd} - p_\mathrm{WAIC}`, with
:math:`p_\mathrm{WAIC}` the posterior variance of each row's
log-likelihood; it warns when a row's term is above 0.4, where it is
unreliable and ``loo()`` is the better estimate. The standard errors are
:math:`\sqrt{N}` times the standard deviation of the units' values (with
divisor :math:`N - 1`, as R's ``loo``).

Many parameters: the Laplace approximation
------------------------------------------

The grid computes the posterior exactly, but only for one to three free
parameters: a grid in more dimensions needs too many points. Past three
(an offset and a limited-failure proportion on a Weibull, say), and
whenever ``method="laplace"`` is asked for, SurPyval uses the *Laplace
approximation*.

**What it is, in words.** Find the single most probable set of parameter
values (the peak of the posterior), measure how quickly the posterior
falls away from that peak in every direction, and take the posterior to
be the bell-shaped (normal) distribution with that centre and that
spread. It is done where posteriors are closest to bell-shaped: on the
log of a positive parameter and the logit of a proportion. Every
interval and prediction is then carried back to the parameters' own
scales, so a shape's interval is still above zero and a proportion's is
still between 0 and 1.

**When it is good.** With a moderate or large amount of data, a
posterior is close to normal on that scale, and the approximation is
then close to the exact answer: in SurPyval's tests, 100 Weibull lives
(30% censored) give means and 90% interval ends within 0.2 standard
deviations of the grid's exact ones, and an Exponential's interval moves
to within 1% of the exact (conjugate) one by 1,000 lives.

**When it is not.** With very few failures, or a parameter pressed
against a bound (an offset near the first failure, a proportion near 1),
the posterior is skewed, and a symmetric bell misses one of its tails.
Then use the grid (up to three free parameters, exact), or sample the
exact posterior with PyMC through :mod:`surpyval.pymc`.

**How it checks itself.** Every Laplace fit compares the exact posterior
with its normal at 4,096 points drawn from the normal: where the ratio
of the two has a heavy tail, the normal is missing mass. The heaviness of
that tail is the *Pareto k* (``model.diagnostics["pareto_k"]``, also in
the summary and in ``diagnose()``): below 0.5 the approximation is good,
up to 0.7 usable, and above 0.7 poor, and the fit warns. A skewed
posterior can still give a moderate k, so there is a second check: how
far reweighting the draws towards the exact posterior (below) would move
each parameter's 2.5%, 50% and 97.5% points. Past 10% of the 95%
interval's width (``diagnostics["shift"]``) the fit warns and names the
point. The draws come from a scrambled Sobol sequence with a fixed seed,
so a fit gives the same numbers on every run.

.. jupyter-execute::

    x4 = sp.Weibull.random(150, 100, 2, random_state=5) + 50
    c4 = (x4 > 170).astype(int)
    four = sp.Weibull.fit_bayes(
        np.minimum(x4, 170), c4, offset=True, lfp=True,
        priors={"beta": sp.prior.between(1, 4),
                "lfp_p": sp.prior.between(0.5, 0.99)})
    print(four.method, round(four.diagnostics["pareto_k"], 2))
    print(four.param_cri("gamma").round(1))

**A correction.** ``importance=True`` reweights the normal's draws
towards the exact posterior (Pareto-smoothed importance sampling: each
draw weighted by the ratio of the exact density to the normal's, the
largest ratios smoothed). It costs nothing more than the check, and in
the tests it brings a two-parameter fit's means and interval ends to
within 0.01 standard deviations of the grid's. It is used only where the
Pareto k is at most 0.7; above that the weights cannot be trusted
either, and the fit says so.

**The formulas.** With :math:`\phi` the free parameters on the
unconstrained scale and :math:`\ell(\phi) = \log p(\phi) + \log
L(\phi)` the log posterior there (the prior including the change of
scale), the approximation is

.. math::

    p(\phi \mid D) \approx \mathcal{N}\left(\hat\phi,\,
    \left[-\nabla^2 \ell(\hat\phi)\right]^{-1}\right),
    \qquad \hat\phi = \arg\max \ell(\phi),

the Hessian by central differences. A plain Laplace fit's parameter
quantiles are the normal's own, exactly; functions of the parameters
(``sf``, a B-life) are averaged over the draws. The log evidence is
:math:`\ell(\hat\phi) + \tfrac{d}{2}\log 2\pi + \tfrac12 \log
|\Sigma|`, or, reweighted, the importance-sampling estimate
:math:`\log \tfrac1S \sum_s e^{\ell(\phi_s)} / q(\phi_s)`. The
Pareto k is that of Yao, Vehtari, Simpson and Gelman (2018), "Yes, but
did it work?: evaluating variational inference".

Flexible families: a prior on the shape
---------------------------------------

The ``GeneralizedGamma`` and the ``GeneralizedF`` hold the Weibull, the
lognormal and the gamma (and the ``GeneralizedF`` the log-logistic) as
members, so their extra shapes say *which kind* of distribution the data
look like. That is exactly what a moderate sample hardly pins down: far
out along ``Q`` (or as ``P`` grows) the likelihood flattens towards a
limit of the family instead of falling away, and with every parameter
flat the posterior has no finite mass. ``fit_bayes`` refuses such a fit
and says so, rather than return an interval that only reflects where
the grid stopped.

The remedy is a weakly informative prior on the shape alone: one that
says "somewhere near the familiar members" without choosing between
them. For the ``GeneralizedGamma``, ``Q`` from a ``Normal(0, 1)`` puts
the lognormal (``Q = 0``), the Weibull and the gamma (``Q = 1``, ``Q =
sigma``) well inside its range and anything past ``|Q| = 2`` in its
tails; ``mu`` and ``sigma`` can stay flat, as the data fix them.

.. code-block:: python

    import numpy as np
    import surpyval as sp

    rng = np.random.default_rng(746)
    true = sp.GeneralizedGamma.from_params([2.0, 0.5, 0.7])
    t = true.qf(rng.uniform(size=60))
    cens = rng.uniform(0, 12, 60)
    x, c = np.minimum(t, cens), (cens < t).astype(int)

    model = sp.GeneralizedGamma.fit_bayes(
        x, c=c, priors={"Q": sp.Normal.from_params([0.0, 1.0])})

On these 60 lives (52% censored) ``Q`` has a 95% credible interval of
-0.009 to 1.56, against the maximum-likelihood -0.015 to 1.44: the
prior only stops the tails, and the summary reports that ``Q`` "came
mostly from the data". The grid is exact here (three parameters) and
takes about ten seconds.

For the ``GeneralizedF`` give ``P`` a prior as well, as a range on its
own scale, for example ``sp.prior.between(0.05, 5)``. On 100 lives from
``GeneralizedF(2, 0.5, 0.5, 1)`` (27% censored, drawn as above with
censoring times up to 25) and ``Q`` from ``Normal(0, 1)``, a flat ``P``
gets an interval of 0.006 to 26 (maximum likelihood: 1e-4 to 530),
because the likelihood is nearly as high at the generalised gamma edge
(``P`` near 0) as anywhere; with the range it is 0.053 to 4.1. Four
parameters are past the grid, so the fit is the Laplace approximation,
and its Pareto k says how far to trust it: 0.86 with ``P`` flat (poor),
0.56 with the range. (``surpyval.pymc`` has no ``GeneralizedF`` yet: it
needs an incomplete beta function, which torch, whose families are kept
the same as PyMC's, does not have.)

Regression: a prior on an effect
--------------------------------

Small accelerated tests are where a prior on an effect matters most. A
power-law life model says the life falls as the voltage to a power
``n``; past tests of the same insulation put ``n`` near -2, give or take
0.5. Fifteen units at three voltages, and the life at the use voltage of
10 with its credible interval:

.. code-block:: python

    from surpyval.life_models import Power

    fitter = sp.AcceleratedLife(sp.Weibull, Power)
    model = fitter.fit_bayes(
        x, Z=voltage,
        priors={"n": sp.Normal.from_params([-2.0, 0.5]),
                "beta": sp.prior.around(2, between=(1.2, 4))},
    )
    model.param_cri("n")                 # the exponent's credible interval
    model.sf(100.0, [[10.0]])            # reliability at use, predictive
    model.quantile_cri(0.1, [[10.0]])    # the B10 at use, with its interval

The proportional hazards and accelerated failure time fitters take the
same: ``sp.WeibullPH.fit_bayes(x, Z, c, priors={"coef_0": ...})``, a
coefficient by its name (``coef_0``, or its column's with
``fit_bayes_from_df``). The result is a
:class:`~surpyval.bayes.BayesianRegression`: ``sf``, ``ff``, ``df``,
``hf``, ``Hf`` and ``qf`` at covariates are the posterior predictive
(the uncertainty in the parameters averaged in), ``cri(x, Z, on=...)``,
``quantile_cri`` and ``param_cri`` the credible intervals, and
``compare()`` sets each prior, the maximum-likelihood estimate and the
posterior side by side.

A regression has at least three parameters (the baseline's and one per
covariate), so it is computed by the Laplace approximation of the
section above, with the same checks (``diagnostics["pareto_k"]`` and
``["shift"]``, a warning where it is poor) and the same
``importance=True`` correction. With a few dozen failures it is close to
exact: in the tests, an exponential model's reweighted posterior is
within 0.6% of the exact interval's width, and with flat priors and 400
units the posterior medians are the maximum-likelihood estimates and the
posterior spreads their standard errors (within 0.01%). For the exact
posterior of a small or awkward regression, sample it with PyMC through
:mod:`surpyval.pymc`.

A coefficient of ``exp(beta'Z)`` that is given no prior gets
``Normal(0, 2.5 / sd)``, ``sd`` its covariate's standard deviation in the
data: a change of 2.5 in the log hazard (or the log life) for one
standard deviation of the covariate is one prior standard deviation,
which is weak, and the same whatever the covariate's unit. The summary
and ``model.default_priors`` say which priors were chosen this way. Any
other parameter with no prior is flat on its unconstrained scale, as for
a distribution, and refused where the data do not determine it. Not
covered, and refused with what to use: weights, penalties (a prior on a
coefficient is the Bayesian penalty), time-varying covariates, the
semi-parametric Cox model and the other regression families.


Cured fractions and mixtures
----------------------------

**A cured fraction with covariates.** In a mixture-cure regression some
units never fail (patients cured, parts without the defect), and the
covariates move both how many fail at all (the *incidence*, ``lfp_p``
at the reference row, with its coefficients on the logit scale) and when
the rest fail (the *latency*). ``MixtureCure(dist).fit_bayes`` puts
priors on all of them; a coefficient given none gets ``Normal(0, 2.5 /
sd)``, ``sd`` its covariate's standard deviation, as the other
regressions' do (on the logit scale, a change of 2.5 in the log odds of
ever failing for one standard deviation of the covariate). The posterior
is the Laplace approximation, with its checks (``importance=True`` to
correct it).

.. code-block:: python

    import surpyval as sp

    # 150 units followed to 40 hours; z = 1 for the new supplier
    model = sp.MixtureCure(sp.Weibull).fit_bayes(
        x, Z, c,
        priors={"lfp_p_coef_0": sp.Normal.from_params([-0.5, 0.5])})
    model.cure_probability([[1.0]])   # 0.33: never fail, new supplier
    model.cure_cri([[1.0]])           # 0.24 to 0.43
    model.cri(20.0, [[1.0]])          # survival to 20 hours: 0.25 to 0.44

A cured fraction is only known from a *plateau*: units still under
observation, and not failing, long after the failures have thinned out.
With too short a follow-up the data cannot tell a cured unit from one
that has not failed yet. The fit then warns, saying how it knows
(Maller and Zhou's check, ``model.mle.follow_up()``), and the cure
fraction's posterior rests on its prior. Where the data show no cured
fraction at all (the maximum-likelihood ``lfp_p`` runs to 1) it warns as
the maximum-likelihood fit does, and with no prior on ``lfp_p`` it
refuses: that posterior has no finite mass. Give ``lfp_p`` a prior from
what is known (``sp.prior.between(0.6, 0.95)``: "60% to 95% of units at
the reference row fail eventually"), or follow the units for longer.

**Mixtures of lives.** ``MixtureModel.fit_bayes`` refuses. A mixture's
likelihood is the same whichever component is called the first, so its
posterior has a peak for every numbering of the components (*label
switching*), and neither a single normal nor the grid describes it.
Sample it with PyMC through ``surpyval.pymc.mixture_likelihood``, which
numbers the components in increasing order of one parameter so that the
labels cannot switch (see "Mixtures" in :doc:`PyMC with SurPyval`).

Checked against: the maximum-likelihood fit (with flat priors, the
posterior's peak) and simulation-based calibration of the coverage of
the intervals (a reduced run of 60 repetitions, 200 units: the 90%
intervals held the truth in 0.87 to 0.92 of them for the five
parameters; the 50% intervals in 0.38 to 0.58, the latency coefficient's
0.38 below the band of 0.41 to 0.59 the full study of 300 repetitions
is held to, which runs with the calibration studies).

An activation energy from the literature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A temperature-accelerated test at three temperatures seldom pins down
the activation energy on its own, and it is the number the extrapolation
to the use temperature hangs on. The failure mechanism's activation
energy is often known roughly from the literature, say about 0.7 eV,
fairly surely between 0.6 and 0.8.

**Check the unit before writing the prior.** SurPyval's ``Arrhenius``
life model is :math:`L(T) = b\,e^{a / T}` with ``T`` in kelvin, so its
parameter ``a`` is the activation energy *divided by Boltzmann's
constant*, in kelvin: 0.7 eV is ``a`` = 8,123 K, not 0.7. The prior is
written on ``a``, so the belief in eV is divided by the constant:

.. code-block:: python

    import numpy as np
    import surpyval as sp
    from surpyval.life_models import Arrhenius

    K_B = 8.617333262e-5                     # Boltzmann's constant, eV / K

    # 36 units, 12 at each of 85, 105 and 125 C; the test stops at 6,000 h
    T = np.repeat(np.array([85.0, 105.0, 125.0]) + 273.15, 12)
    scale = 2e-6 * np.exp((0.7 / K_B) / T)
    x = sp.Weibull.random(36, 1.0, 2.5, random_state=3) * scale
    c = (x > 6000.0).astype(int)
    x = np.minimum(x, 6000.0)

    fitter = sp.AcceleratedLife(sp.Weibull, Arrhenius)
    model = fitter.fit_bayes(
        x, T[:, None], c,
        priors={"a": sp.prior.around(0.7 / K_B,
                                     between=(0.6 / K_B, 0.8 / K_B))},
        importance=True,
    )
    model.param_cri("a") * K_B           # the activation energy, in eV
    use = [[55.0 + 273.15]]
    model.quantile_cri(0.1, use)         # the B10 life at 55 C

With the data alone (no prior on ``a``) the activation energy's 95%
credible interval is 0.53 to 0.92 eV; with the prior it is 0.60 to 0.80
eV, and the B10 life at 55 C, an extrapolation 30 C below the lowest
test temperature, narrows from 17,000-125,600 hours to 23,300-79,200.
A prior mistakenly written in eV (``sp.prior.between(0.6, 0.8)`` on
``a``) is caught: the fit warns that the data disagree with it, since
their own estimate of ``a`` is in the thousands.

Beliefs about a life, earlier fits, and checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A belief is often about a life at some condition rather than about a
parameter: "the B10 at the use stress is between 20,000 and 40,000
hours". ``sp.prior.on_function`` states it on any function of the
regression model, and ``sp.prior.on_quantities`` carries it to the
parameters; ``instead_of`` names the parameter whose prior the
statement stands in place of (here the scale), and the others keep
theirs:

.. code-block:: python

    belief = sp.prior.on_quantities(
        fitter,
        sp.prior.on_function(lambda m: m.qf(0.1, use),
                             sp.prior.between(20_000, 40_000),
                             name="B10 at 55 C"),
        instead_of="b",
    )
    model = fitter.fit_bayes(x, T[:, None], c, priors=belief)

An earlier regression is a prior in two ways.
``sp.prior.from_model(earlier)`` takes its estimates and their
covariance as a joint normal (``weight=`` counts the old evidence for
less; an earlier *Bayesian* regression is taken exactly, its priors and
its data). ``sp.prior.from_fit(earlier, weight=)`` takes its data
themselves as a power prior: at ``weight=1`` it is the fit of the old
and new rows together (to 2e-5 in the tests).

``model.loo()`` and ``model.waic()`` estimate how well the regression
predicts a row it has not seen (a row is the unit), and
``sp.bayes.compare`` ranks regressions fitted to the same rows by them:
a model with a covariate against the same model without it, say. In the
tests the leave-one-out values match leaving each row out exactly (an
exponential model in two groups, where that has a closed form) within
0.02 a row.

``model.diagnose()`` gathers the checks in one report: the Laplace
approximation's, any disagreement between the data and the priors, and
posterior predictive checks on the number of failures, the total time
on test and the shortest time, each simulated for the same units at the
same covariates and censored as the data were.

Reliability growth and repairable systems: Crow-AMSAA and the HPP
-----------------------------------------------------------------

A reliability growth test has few failures, and the question asked of it
is rarely "what is ``beta``?". It is "what is the MTBF now, what will it
be after another 400 hours, how many failures should we plan for, and
have we met the requirement?". Each has a plain Bayesian answer.

In words: the failures say how the failure intensity is changing (for
the Crow-AMSAA power law, a shape ``beta`` below 1 is growth). A prior
says what was believed before the test: about the growth rate, about the
MTBF the system should have reached, or the previous test phase's
result. The posterior combines them, and every answer above is read from
it.

.. code-block:: python

    import surpyval as sp
    from surpyval.recurrent import CrowAMSAA

    # one prototype, tested to 400 hours: 11 failures
    x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    c = [0] * 11 + [1]

    model = CrowAMSAA.fit_bayes(
        x, c=c,
        priors={"beta": sp.prior.around(0.7, between=(0.4, 1.2))},
    )
    model.mtbf_cri()                    # the MTBF now: 25 to 113 hours
    model.mtbf_cri(800.0)               # projected to 800 hours: 24 to 154
    model.count(duration=100.0)         # failures in the next 100 hours
    model.mtbf_probability(30.0)        # P(MTBF now >= 30 hours) = 0.92

The posterior median of the MTBF now is 51 hours (the maximum-likelihood
estimate is 41), with a 95% credible interval of 25 to 113; with no
prior at all it is 22 to 125. The next 100 hours are expected to bring
2.0 failures, and with 90% probability between 0 and 5:
``model.count`` is the *predictive* distribution, a Poisson count whose
mean is itself uncertain, so it is wider than a Poisson at the fitted
values. The probability that the MTBF now is at least 30 hours is 0.92.

The posterior of a Crow-AMSAA fit has two parameters, so it is computed
on the grid: exact, with a stated integration error, and no
approximation to check. (The grid is laid on the expected number of
failures by the end of the test and the shape, not on ``alpha`` and
``beta`` directly, whose posterior is a curved ridge with few failures.)

**A belief about the MTBF, not a parameter.** ``alpha`` means little to
most people; the MTBF at the end of the test means a lot. State the
belief there, in place of a prior on ``alpha``:

.. code-block:: python

    belief = sp.prior.on_quantities(
        CrowAMSAA,
        mtbf_at={400.0: sp.prior.between(40, 80)},
        instead_of="alpha",
    )
    model = CrowAMSAA.fit_bayes(
        x, c=c,
        priors=[belief,
                {"beta": sp.prior.around(0.7, between=(0.4, 1.2))}],
    )

``intensity_at`` and ``cif_at`` (the expected number of failures by a
time) are stated the same way, and ``sp.prior.on_function`` takes any
other function of the model.

**The previous test phase as the prior.** With the earlier phase's
fitted model ``phase1`` at hand:

.. code-block:: python

    model = CrowAMSAA.fit_bayes(
        x2, c=c2, priors=sp.prior.from_fit(phase1, weight=0.5))

``from_fit`` uses the earlier phase's data themselves, their likelihood
raised to ``weight``: 1 says the design did not change between the
phases (the fit is then that of both phases' systems together), and a
lower weight discounts the old evidence because fixes were made. The
weight is a judgement; try two and compare.
``sp.prior.from_model(phase1)`` uses the earlier estimates and their
covariance instead, and takes an earlier Bayesian fit exactly.

**A constant rate, and no failures.** ``HPP.fit_bayes`` with a Gamma
prior on the rate is the gamma-Poisson pair, in closed form: a prior
``Gamma(a, b)`` (worth ``a`` failures in ``b`` hours) and ``r`` failures
in ``T`` hours give ``Gamma(a + r, b + T)``. It answers the question a
maximum-likelihood fit cannot: after 100 hours with **no** failures,

.. code-block:: python

    from surpyval.recurrent import HPP

    model = HPP.fit_bayes(
        [100.0], c=[1],
        priors={"lambda": sp.Gamma.from_params([2.0, 100.0])})
    model.mtbf_probability(50.0)        # 0.908

the rate is ``Gamma(2, 200)``, and the probability that the MTBF is at
least 50 hours is 0.908.

``model.loo()`` and ``model.waic()`` take **one system as the unit**
(its whole history is one observation of the process), so
``sp.bayes.compare`` between a Crow-AMSAA and an HPP fit asks which
predicts a new system better; with one system there is nothing to leave
out. ``model.diagnose()`` runs posterior predictive checks on the number
of events, the share of them in the later half of each system's
observation (which a wrong trend gets wrong) and the most events on one
system (which a fleet of unlike systems gets wrong).

Checked against: the closed form for the shape (with no priors and a
common end ``T``, ``beta`` is Gamma(N - 1, rate ``sum log(T / t)``), to
0.2%), the maximum-likelihood estimate (the posterior's peak with no
priors), PyMC (two frozen references: every posterior point within 2.3
of the sampler's own Monte Carlo standard errors), and simulation-based
calibration (300 repetitions each: the 90% intervals held the truth in
0.93 and 0.91 of them for ``alpha`` and ``beta``, 0.86 and 0.89 for the
HPP's rate, all inside three standard errors of 0.90; 0.90 and 0.91 for
a proportional intensity's rate and coefficient on the grid, and 0.87 to
0.92 for one of four parameters by the Laplace approximation).

Other intensities, covariates and failure causes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every NHPP intensity has ``fit_bayes``, with the same priors and
results: ``Duane``, ``CoxLewis`` and the software reliability growth
models. Each has two or three parameters, so each posterior is computed
on the grid. A model with a finite number of faults (``GoelOkumoto`` and
the like) fitted to data that show no levelling off has no posterior
without priors, and says so: give its parameters priors.

**Covariates.** ``ProportionalIntensityHPP.fit_bayes(x, Z, i, c)`` and
``ProportionalIntensityNHPP.fit_bayes(x, Z, i, c, baseline=CrowAMSAA)``
put priors on the baseline's parameters and on the covariates'
coefficients. A coefficient with no prior gets the regressions' default,
``Normal(0, 2.5 / sd)`` with ``sd`` its covariate's standard deviation
over the systems, stated in ``default_priors`` and the summary. Results
are given for a covariate set (one kind of system):

.. code-block:: python

    from surpyval.recurrent import CrowAMSAA, ProportionalIntensityNHPP

    model = ProportionalIntensityNHPP.fit_bayes(
        x, Z, i, c, baseline=CrowAMSAA)
    harsh = [1.0]
    model.mtbf_cri(harsh)                     # the MTBF now, harsh duty
    model.count(harsh, duration=100.0)        # failures in the next 100
    model.mtbf_probability(40.0, harsh)       # P(MTBF >= 40)

With up to three free parameters (a baseline with one covariate) the
posterior is on the grid; beyond, it is the Laplace approximation, with
its checks in ``diagnostics`` and ``importance=True`` to correct it. In
the tests the Laplace intervals are within 3% to 6% of the exact (grid)
interval's width on twelve systems, and within 0.2% once reweighted.

**Failure causes.** ``CauseSpecificNHPP.fit_bayes(x, i, c, e=causes)``
fits one process a cause. The causes' likelihoods are separate, so each
cause has its own posterior (``model.models[cause]``, the process's own
Bayesian fit of that cause's events). Priors are one set for every
cause or a set a cause (``priors={"seal": {"beta": ...}}``). Each result
takes the cause, or ``event=None`` for all causes together, whose
intervals combine the causes' posteriors by 4,000 joint draws with a
fixed seed:

.. code-block:: python

    from surpyval.recurrent import CauseSpecificNHPP

    model = CauseSpecificNHPP.fit_bayes(x, i, c, e=causes)
    model.cri(400.0, "seal", on="mtbf")       # counting seal failures
    model.cri(400.0, on="mtbf")               # counting every failure
    model.mtbf_probability(30.0)              # the system requirement

Degradation processes: Wiener and gamma
---------------------------------------

Five units are read every 10 hours to 100 hours, and the wear rises by
about 0.5 an hour with noise; a unit fails when its wear reaches 100.
``WienerProcess.fit_bayes`` (or ``GammaProcess.fit_bayes``, for wear that
never goes down) takes the readings as ``fit`` does and puts priors on
the drift ``mu`` and the diffusion ``sigma`` (and on any stress
coefficients). With two or three free parameters the posterior is on
the grid, exact to its stated error.

.. code-block:: python

    from surpyval.degradation import WienerProcess

    model = WienerProcess.fit_bayes(t, y, i, threshold=100)
    model.param_cri("mu")            # 0.409 to 0.569 an hour
    model.quantile_cri(0.1)          # B10 life: 148.4 to 202.0 hours
    model.quantile_cri(0.1, point=True)  # its best guess, 172.4 hours
    model.qf(0.1)                    # a new unit's B10: 167.8 hours
    model.mean_cri()                 # mean life: 175.9 to 244.7 hours

The life is the time a unit's wear first reaches the threshold (its
*first passage*). As for the distributions, ``sf``, ``ff`` and ``qf``
are for a new unit with all the uncertainty included (so its B10 is
earlier than the B10 at the best-guess parameters), and ``cri`` and
``quantile_cri`` are the population's credible intervals. ``mean`` is
the posterior median of the mean life, since a slow process that is
still credible can make the posterior mean very large.

**Remaining life.** A unit at a wear of 60 has 40 to go, and the
increments are independent, so its remaining life is a fresh passage
over 40:

.. code-block:: python

    rul = model.predict_rul(60.0)
    rul.median        # 80.2 hours: an even chance either side
    rul.interval      # 52.7 to 125.2: where this unit's life lies (95%)
    rul.median_cri    # 69.0 to 95.4: the population's median, credible
    rul.cri([50.0, 100.0])   # bands on the chance of lasting that long

The interval is for this one unit; the credible interval of the median
is the uncertainty in the parameters alone, and narrows with more
readings, while the unit's own spread does not.

**A drift from an earlier fleet.** Two new units with only their own
readings put the B10 between 150 and 245 hours. The earlier fleet's
drift, counted at half weight, narrows it to 155 to 216:

.. code-block:: python

    import surpyval as sp

    old = WienerProcess.fit(t, y, i, threshold=100)
    drift = sp.prior.from_model(old, param="mu", weight=0.5)
    model = WienerProcess.fit_bayes(
        t_new, y_new, i_new, threshold=100, priors={"mu": drift})

``sp.prior.from_fit(earlier_bayesian_fit, weight=...)`` takes the
earlier fleet's readings themselves (their likelihood raised to the
weight); at weight 1 it is the same as fitting both fleets' readings
together.

**A belief about the life.** "The B10 is about 170 hours, between 140
and 200" stands in place of a prior on the drift:

.. code-block:: python

    belief = sp.prior.on_quantities(
        WienerProcess,
        B10=sp.prior.around(170, between=(140, 200)),
        instead_of=["mu"],
    )
    model = WienerProcess.fit_bayes(
        t_new, y_new, i_new, threshold=100, priors=[belief])

``on_quantities`` takes the first-passage life's B-lives, ``R_at`` and
``mean`` for either process, at the reference stress, from the fit's
``y0``.

The starting level ``y0`` is the maximum-likelihood fit's (given, or
estimated from the readings) and held: it is not in the likelihood of
the increments.

**Readings off a coarse gauge.** A gauge that reads wear to the nearest
2, when an interval adds only about 1 of spread, turns every increment
into a whole number of steps, some of them zero. Taken as exact, those
readings look more scattered than the wear is, and the posterior puts
the shape rate ``alpha`` too low. Give the step as ``gauge`` (with
``rounding``, ``exact_start`` and ``gauge_method`` as for ``fit``) and
the posterior is built on the likelihood ``GammaProcess.fit(...,
gauge=)`` maximises: the probability that each unit's true path passes
through the bins of its readings. For five units whose wear is a gamma
process with ``alpha`` 2 and ``beta`` 4, read every 10 hours to 100 (the
``GammaProcess.fit_bayes`` example):

.. code-block:: python

    y_read = np.round(y / 2.0) * 2.0
    model = GammaProcess.fit_bayes(t, y_read, i, 100, gauge=2.0)
    model.param_cri("alpha")    # 1.12 to 3.65: holds the true 2
    GammaProcess.fit_bayes(t, y_read, i, 100).param_cri("alpha")
    # 0.68 to 1.50 with the readings taken as exact

With no priors the posterior's peak is that fit's estimate; as the
step shrinks the posterior becomes the one of exact readings; and in
200 simulated fleets read off such a gauge the gauge fit's intervals
hold the truth at their level while the fit that ignores the gauge
falls well short (``test_sbc_process_bayes_gauge.py``). The exact
method carries each unit's path through 16 cells of each bin, so it
costs about ten times the ``"independent"`` approximation: a few
seconds to a minute on the grid with two parameters, minutes with a
stress coefficient.

**A stress profile.** A fit with stress takes one stress row ``Z``, or
a stress profile (a ``StepSchedule``), in ``sf``, ``ff``, ``qf``,
``cri``, ``quantile_cri`` and ``predict_rul``. Under a profile a unit
ages at the rate of the stress in force, so its clock is piecewise
linear in time; each posterior point runs its own clock at its own
stress coefficients, and the results are those of the fitted model at
every point at once:

.. code-block:: python

    from surpyval import StepSchedule

    profile = StepSchedule.from_changepoints([0, 60, 120], [1.0, 2.0, 1.5])
    model.sf([90.0, 150.0], profile)    # a new unit, all uncertainty in
    model.cri([90.0, 150.0], profile)   # the population's band
    model.predict_rul(60.0, Z=profile)  # time zero of the profile is now

For ``predict_rul`` the profile starts now. ``model.mean(profile)`` and
``model.mean_cri(profile)`` give the mean life under the profile and its
credible interval, at every posterior point at once. In words: through
each step of the profile the unit's clock runs at that step's pace, so
the time it spends alive in the step is the time its clock spends
alive over the step's stretch of clock, divided by the pace. For the
Wiener process that stretch has a closed form (the Inverse Gaussian's
partial mean); for the gamma process it is one integral, taken on a
fixed rule. With :math:`I(u) = \int_0^u S_0(\tau)\,d\tau` the
reference-clock survival's integral and :math:`AF_k` the pace of step
:math:`k`, which takes the clock from :math:`\tau_k` to
:math:`\tau_{k+1}`,

.. math::

    E[T] = \sum_k \frac{I(\tau_{k+1}) - I(\tau_k)}{AF_k},

the last step running on for ever (a cyclic profile is laid out until
what is left is below 1e-12 of the whole). A step at a stress so low
that the clock all but stops (storage, or an idle spell) is integrated
in calendar time instead, since its stretch of clock is too short for
the difference of :math:`I`: its calendar length then counts in full.
It agrees with the fitted model's mean under the profile, point by
point, to about 1e-9, and with a direct integration of the calendar
survival through an idle step to 1e-9.

With no priors the Wiener posterior is the textbook one: flat on ``mu``
and on ``log sigma`` is :math:`p(\mu, \sigma^2) \propto 1/\sigma^2`, so
:math:`\sigma^2` is :math:`S / \chi^2_{n-1}` and :math:`\mu` a Student
:math:`t_{n-1}` about :math:`\hat\mu = \sum \Delta y / \sum \Delta t`
with scale :math:`\sqrt{S / ((n-1) \sum \Delta t)}`, where
:math:`S = \sum (\Delta y - \hat\mu \Delta t)^2 / \Delta t` over the
:math:`n` increments. The tests hold the grid to it to 0.2% of the 95%
intervals' widths.

Competing risks and multi-state models
--------------------------------------

A part that can fail in two ways, wear and cracking, has a life
distribution for each, and the first to happen ends its life. The
likelihood of such data splits into one factor a cause: a part that
cracked contributes the crack's density and the chance it had not yet
worn out, and a part still running contributes the chance it has done
neither. So with priors stated a cause at a time, what is learned about
one cause does not depend on what is believed about the other, and each
cause's posterior is simply the ordinary Bayesian fit of its
distribution, with the other cause's failures treated as parts still
running when they were removed. ``ParametricCompetingRisks.fit_bayes``
does exactly that, and combines the fits:

.. code-block:: python

    import surpyval as sp
    from surpyval.univariate.competing_risks import ParametricCompetingRisks

    x = [12, 25, 31, 40, 44, 52, 60, 71, 80, 80]
    e = ["wear", "crack", "wear", "wear", "crack", "wear", "crack",
         "wear", None, None]
    # three cracks: a constant crack rate; wear-out with a shape of about 3
    model = ParametricCompetingRisks.fit_bayes(
        x, e, dist={"wear": sp.Weibull, "crack": sp.Exponential},
        priors={"wear": {"beta": sp.prior.around(3, between=(2, 5))}})

    model.models["wear"]          # wear's own BayesianParametric fit
    model.cif(50.0, "wear")       # P(a new part has worn out by 50)
    model.cri(50.0, "wear")       # the 95% credible interval on it
    model.quantile_cri(0.1)       # the B10 life, failing from any cause
    model.probability_of_cause("crack")

Priors are one set for every cause or a set a cause, as for the
cause-specific intensities. The predictions for a new part (``sf``,
``ff``, ``cif``, ``qf``) are the posterior predictive, and exact: since
the causes' posteriors are independent, the chance a new part survives
every cause is the product of its chances of surviving each, and its
chance of wearing out by ``t`` is the integral of wear's predictive
density against the other causes' predictive survival. The credible
intervals come from 4,000 joint draws (one from each cause's posterior)
with a fixed seed, so the numbers repeat.

**Failures found at an inspection.** A part found cracked at the
40-hour inspection, having been working at 20 hours, failed somewhere
in between, and it was the crack that got it first. The chance of that
is not the crack's own chance of happening between 20 and 40 hours: wear
was at work too, and a part that wore out first in that window would
have been found worn, not cracked. So the part's contribution to the
likelihood is the chance that its first failure of any kind came in the
window *and* was the crack: at each moment in the window, the chance the
part is still whole times the crack's hazard, added up over the window,

.. math::

    \int_{x_l}^{x_r} h_{\text{crack}}(u)\, S(u)\, du,
    \qquad S(u) = S_{\text{crack}}(u)\, S_{\text{wear}}(u),

the crack's cumulative incidence over the window
(:math:`\mathrm{CIF}_{\text{crack}}(x_r) - \mathrm{CIF}_{\text{crack}}(x_l)`),
not :math:`F_{\text{crack}}(x_r) - F_{\text{crack}}(x_l)`. A part found
failed at its first inspection (``c = -1``) has the window from its
entry (or from the start). This is the likelihood ``fit`` maximises for
such rows, and because it holds both causes' parameters the posterior
no longer splits: ``fit_bayes`` then fits every cause's parameters
together, on the grid for up to three free parameters in all (two
Exponential causes, or a Weibull and an Exponential) and by the Laplace
engine beyond. ``model.joint`` says which was done, ``model.posterior``
is the joint posterior (its parameters named ``wear.beta``,
``crack.failure_rate``), and each of ``model.models`` is its cause's
marginal. Priors are still one set for every cause or a set a cause:
a parameter at a time, or a cause's joint prior
(``prior.on_quantities`` with a belief about a cause's B10 life,
``prior.from_model`` with an earlier fit of that cause,
``prior.MultivariateNormal``), which is put on that cause's parameters
(``wear.alpha`` and ``wear.beta``) and means exactly what it means in the
cause's own fit. The window's integral is taken at every
posterior point on the cause's own probability scale, by a 48-point
Gauss-Legendre rule (to about 1e-9 against adaptive quadrature in the
tests). With flat priors the posterior's peak is the maximum-likelihood
estimate, and with no windows in the data the joint fit gives the same
posterior as the fit a cause at a time.

A part found failed whose cause was not recorded (``e`` missing, ``c``
of 2, or -1 for a window from its entry) is taken too, by ``fit`` and
``fit_bayes`` alike: all it says is that the part's first failure, of
whichever kind, came in the window, so its contribution is the all-cause
survival over the window, :math:`S(x_l) - S(x_r)` (the sum of the
causes' incidences over it). It tells the fit mostly about how long
parts last; about which cause ended them it says only a little, through
the shape of the all-cause survival. This assumes that whether a cause
goes unrecorded does not depend on which cause it was (an inspection
that cannot tell wear from a crack is fine; one that records cracks and
loses the cause of worn parts is not, and would bias the split). WAIC and leave-one-out
(``model.result.waic()``, ``loo()``) take left truncation (``tl``) on
this path: each row's log-likelihood carries its own entry term.

The same holds for a **multi-state model**: units moving between
healthy, degraded and failed, say, with a distribution for the hazard of
each move. Its likelihood is a factor a transition, so
``ParametricMultiState.fit_bayes`` fits each transition's distribution
by Bayes' rule on the spells in its starting state, and the
probabilities of being in each state follow from joint draws through
the model's own equations:

.. code-block:: python

    from surpyval.multistate import ParametricMultiState

    model = ParametricMultiState.fit_bayes(x, e, state, tl=tl)
    model.occupancy(240.0, "failed")       # P(a new unit is failed)
    model.occupancy_cri(240.0, "failed")   # its 95% credible interval
    model.length_of_stay(240.0, "degraded")

A state occupation probability for a new unit is the mean over 1,000
joint draws (``model.draws``; fixed seed), so it carries a Monte Carlo
error of about its spread over the draws divided by 31; the transitions'
posteriors themselves are exact. A model of two states (working, then
failed) is the distribution's own Bayesian fit, posterior for posterior.

**With covariates.** The regression forms split the same way: each
cause's (or transition's) hazard is a regression on the covariates, its
likelihood a factor of its own, so each posterior is that regression's
own ``fit_bayes`` on the cause-specific data (the Laplace engine, as for
every regression). The results are for one set of covariates at a time:

.. code-block:: python

    from surpyval import ParametricCompetingRisksRegression
    from surpyval.multistate import ParametricMultiStateRegression

    model = ParametricCompetingRisksRegression.fit_bayes(x, Z, e)
    unit = model.at([1.0])            # a unit with Z = 1
    unit.cif(10.0, "wear")            # P(it has worn out by 10)
    unit.cri(10.0, "wear")            # the 95% credible interval
    unit.quantile_cri(0.1)            # its all-cause B10 life
    model.cif(10.0, [1.0], "wear")    # the same, in one call

    ms = ParametricMultiStateRegression.fit_bayes(
        x, Z, e, state, tl=tl, i=unit_id)
    ms.occupancy(120.0, [1], "failed")
    ms.occupancy_cri(120.0, [1], "failed")

Priors are one set for every cause (transition) or a set each, named as
the regression names its parameters (``coef_0``, or a covariate's
column; ``beta``); a coefficient given none has the regressions' default,
:math:`\mathrm{Normal}(0, 2.5 / \mathrm{sd})`. A unit's predictions at
its covariates are the posterior predictive, as without covariates; the
competing-risks intervals come from 1,024 joint draws, each draw's
regressions evaluated at the covariates, and the multi-state
probabilities from 1,000. A model of one cause, or of two states, is the
regression's own Bayesian fit.

What every Bayesian fit reports
-------------------------------

The distributions' fits, the regressions' and the other families' are the
same computation: priors on a model's parameters times the likelihood the
family's own ``fit`` uses. One rule chooses how the posterior is
computed, and every result says which was used:

1. **in closed form** where the prior and the data are conjugate (a Gamma
   prior on a constant rate, for one);
2. **on the grid** for one to three free parameters: the exact
   posterior, to a stated integration error;
3. **by the Laplace approximation** beyond three, with its two checks
   against the exact posterior (the Pareto k and the shift of the
   interval's ends) and ``importance=True`` to correct it.

There is no sampler in SurPyval. For the exact posterior of a model with
more than three free parameters, sample it with PyMC through
:mod:`surpyval.pymc`.

Beside its intervals, each result gives:

- ``summary()``: the data, each prior (those SurPyval chose are marked
  "default" and listed in ``default_priors``), the posterior beside the
  maximum-likelihood estimate, and the engine with its checks;
- a note, and a warning at the fit, where **the data disagree with a
  prior**: a parameter whose maximum-likelihood estimate (what the data
  say alone) lies outside the central 99% of the prior given for it
  (``diagnostics["prior_conflict"]``). Neither is taken as right; the
  note says to look;
- ``function_cri(fn)`` and ``probability(condition)``: the credible
  interval of any function of the fitted model, and the posterior
  probability that a requirement is met;
- ``waic()`` and ``loo()``, per *unit* of the data (a row of lives; one
  system's whole history for a recurrent-event process);
- ``ppc()``, a **posterior predictive check**. In words: simulate a few
  hundred data sets from the fitted model, each from one posterior draw
  and under the observed design (the same units, covariates and
  observation windows), and see whether a summary of the real data, such
  as the number of failures, sits among the simulated ones. If the real
  value is in the extreme 2.5% at either end, the model does not
  reproduce that feature of the data. ``diagnose()`` gathers this with
  the engine's checks and the prior conflicts in one report.

A belief about a quantity the model implies, rather than about a
parameter, is stated with ``sp.prior.on_function``: any function of the
fitted model (a B10 life at a use stress, the MTBF at the end of a test).
``sp.prior.on_quantities`` carries it to the parameters by the change of
variables; ``instead_of=`` names the parameters the statements replace
the priors of, and the others keep their own. An earlier fit of any
family is a prior through ``sp.prior.from_model`` (its estimates and
covariance, or exactly for an earlier Bayesian fit) or
``sp.prior.from_fit`` (its data, as a power prior with a ``weight``).

Priors: the helpers, and their dangers
--------------------------------------

A prior on a parameter is any SurPyval distribution
(``sp.Gamma.from_params([2, 1])``, ``sp.Beta.from_params([...])`` for a
proportion) or one of the helpers:

- ``prior.around(m, between=(a, b), mass=0.9)``: "about ``m``, almost
  surely between ``a`` and ``b``"; a LogNormal for a positive parameter, a
  logit-normal for a proportion, a Normal on the real line;
- ``prior.between(a, b)``: "somewhere between ``a`` and ``b``";
- ``prior.uniform(a, b)``: "anything from ``a`` to ``b`` equally", which is
  not "no opinion": it rules out everything else, and it is not uniform on
  any other scale;
- ``prior.flat()``: no opinion, flat on the log scale for a positive
  parameter. It is the default for a parameter given no prior, and it is
  allowed only where the data determine the parameter (old data given as
  a prior with ``prior.from_data`` or ``as_prior()`` count with the new):
  with no failures, a flat prior on the scale is refused, with what to do
  instead, and so is a fit in which every parameter is flat where the
  maximum-likelihood fit has no verified maximum;
- ``prior.fixed(v)``: the value is known, as ``fixed=`` in ``fit``.

A prior that puts mass where the parameter cannot be (a Normal on a
shape, which must be positive) is refused. Where the data contradict a
prior, the fit warns, with the prior's interval and the posterior median,
and takes neither side: check both.

The formulas
------------

Bayes' rule: the posterior density of the parameters :math:`\theta` given
the data :math:`D` is

.. math::

    p(\theta \mid D) = \frac{p(\theta) \, L(\theta; D)}{\int p(\theta)
    \, L(\theta; D) \, d\theta},

where :math:`p(\theta)` is the prior and :math:`L` the likelihood: the
same one ``fit`` maximises, a product of :math:`f(x)` for each failure,
:math:`S(x)` for each unit still running, :math:`F(x)` for each left
censored unit, :math:`F(r) - F(l)` for each interval, each divided by
:math:`S(t_l) - S(t_r)` for a truncated one. The denominator is the
*evidence* (``model.log_evidence``, for Bayes factors, where every prior
is proper).

The predictive survival function averages the parameters out,

.. math::

    S_{\text{pred}}(t) = \int S(t \mid \theta) \, p(\theta \mid D) \,
    d\theta,

and is a proper distribution: :math:`F = 1 - S`, :math:`f = -S'`,
:math:`h = f / S`, and ``qf`` inverts ``ff``. The credible interval on
:math:`S(t)` is the 2.5% and 97.5% points of :math:`S(t \mid \theta)` over
the posterior.

**How it is computed.** For the textbook cases the posterior is in closed
form: an Exponential with a Gamma(:math:`a, b`) prior on its rate and
data that are exact or right censored (with any late entry) has the
Gamma(:math:`a + r, b + T`) posterior, :math:`r` the failures and
:math:`T` the total time on test, and a Lomax predictive; a Normal's mean
with its spread known, and a Bernoulli with a Beta prior, are the others.
Otherwise, for one to three free parameters, SurPyval integrates on a grid:
on the parameters' unconstrained scale (logs of positive parameters,
logits of proportions), along the posterior's own axes at its peak,
widened until the density at every edge is below :math:`10^{-8}` of the
peak, and laid twice, offset by half a cell. Half the difference between
the two grids' summaries is the integration error the summary reports; it
must be below 0.1% of the 95% interval's width (0.5% with three free
parameters), or the grid is refined, and past its budget the fit refuses
with the numbers. No random numbers are involved: the same data and
prior give the same answer on every machine. Past three free parameters,
or with ``method="laplace"``, the Laplace approximation is used instead
(next section).

**Very small and very large probabilities.** A chance of failure of one
in a million million is as much a number as one of a half, and it is
where a safety case or a rare adverse event lives. The predictive
``ff`` is therefore the posterior average of each parameter value's own
failure function, :math:`F_{\text{pred}}(t) = \sum_i w_i F(t \mid
\theta_i)`, and not :math:`1 - S_{\text{pred}}(t)`: a double next to 1
holds about sixteen digits, so the subtraction would leave eight digits
of a probability of :math:`10^{-8}` and none of one of :math:`10^{-16}`.
``sf`` is the same average of the survival functions, so it keeps its
digits far past the data (whichever of the two is larger is one minus the
other, so they always sum to 1); ``Hf`` is :math:`-\log(1 - F)` (as ``log1p``)
while less than half have failed and :math:`-\log S` after, and stays
finite where ``sf`` itself underflows. The credible intervals from
``cri`` are built the same way, and ``qf`` solves on the log of ``ff`` up
to the median and on the log of ``sf`` above it, so
``model.qf(1e-12)`` and ``model.qf(1 - 1e-12)`` are both found (a limited
failure population's quantile is infinite above the proportion that ever
fails).

**Many points: a table.** A grid of two or three free parameters has tens
of thousands of points, and each value of the predictive is a sum over
all of them, too slow for a calculation that asks for tens of thousands
of values (an optimal replacement age, an integral). Where the posterior
has more than 2,000 points, the first predictive value builds a table:
the predictive cumulative hazard :math:`H(t)` is summed exactly at a few
hundred times and read between them by a cubic in :math:`\log H` against
:math:`\log t` (against :math:`t` for a distribution on the whole line),
with the exact slope at each time, so ``sf``, ``ff``, ``Hf``, ``hf`` and
``df`` come from one curve and stay a proper distribution. The times are
placed adaptively: each stretch is checked against the full sum at two
points and split until ``sf`` and ``ff`` (relative to itself, so a small
failure probability keeps its digits) are within :math:`3 \times
10^{-7}` and ``hf`` within ten times that. A stretch that cannot be
brought within that (near the posterior points' own offsets, say), and
the far tails (where the points have a failure or survival probability
below :math:`10^{-10}`), use the full sum, so nothing is extrapolated:
``sf``, ``ff`` and ``Hf`` are within :math:`10^{-6}` of the full sum,
relative, and ``hf`` and ``df`` within :math:`10^{-5}`. Building it
takes under a second for 37,500 points; after it a value takes
microseconds. Set ``model.predictive_table = False`` for the full sum
everywhere.

**Random lives.** ``model.random(size, random_state)`` takes one uniform
number per life. With a closed-form predictive the life is its quantile
at that number. Otherwise the number picks a parameter value by where it
falls among the cumulative weights :math:`w_i`, and its place within that
value's share, rescaled to (0, 1), is the probability at which that
value's own quantile function gives the life. The draws follow the
distribution that ``sf``, ``ff`` and ``qf`` describe, and the same seed
gives the same lives.

The posterior itself is ``model.posterior``: ``draws(k, random_state)``,
``to_dataframe()``, ``density(points)`` (exact for the grid), and
``marginal(name)``; ``model.draw_models(k)`` gives ordinary models, one
per draw. :meth:`~surpyval.bayes.BayesianParametric.from_draws` builds a
Bayesian model from another sampler's draws, and
:func:`surpyval.bayes.log_likelihood_fn` gives SurPyval's likelihood, at
many parameter points at once, to any sampler.

References: Gelman et al., *Bayesian Data Analysis* (3rd ed., 2013);
Hamada, Wilson, Reese and Martz, *Bayesian Reliability* (2008); Meeker,
Escobar and Pascual, *Statistical Methods for Reliability Data* (2nd ed.,
2021), chapter 16; Ibrahim and Chen, "Power prior distributions for
regression models" (2000); Talts et al., "Validating Bayesian inference
algorithms with simulation-based calibration" (2018).
