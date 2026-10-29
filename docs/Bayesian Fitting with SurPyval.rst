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
prior give the same answer on every machine. ``method="laplace"`` is a
normal approximation at the peak instead, labelled approximate.

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
