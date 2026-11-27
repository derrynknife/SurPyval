PyMC with SurPyval
==================

If you already build Bayesian models in `PyMC <https://www.pymc.io>`_,
``surpyval.pymc`` lets you use SurPyval's handling of lifetime data inside
them. You write the model as usual -- your priors, your hierarchy, your
sampler settings -- and one call adds the likelihood of your data: every
failure, survivor, inspection interval and truncation window, counted
exactly as SurPyval's ``fit`` counts them.

In plain words: a censored or truncated lifetime is not an ordinary data
point, so it cannot go into ``observed=`` of a PyMC distribution. A unit
still running at 3,000 hours tells you only that its life is longer than
3,000 hours; a unit found failed at an inspection tells you only that it
failed between two inspections; a unit you only heard about because it
survived to 1,000 hours is from a population cut off at 1,000 hours. Each
needs its own term in the likelihood. ``censored_likelihood`` writes those
terms for you.

SurPyval does not install PyMC and does not depend on it. ``import
surpyval`` never imports it. The helpers import it when you call them, and
if it is not installed they stop with a message saying to install it
(``pip install pymc``).

The helpers are:

- :func:`~surpyval.pymc.censored_likelihood`: add the likelihood of a data
  set to the open model, as a ``pm.Potential``;
- :func:`~surpyval.pymc.log_likelihood`: the same term as a pytensor
  expression, to combine yourself;
- :func:`~surpyval.pymc.prior_from_model`: a prior from a SurPyval fit
  (last year's data, another fleet), keeping its uncertainty;
- :func:`~surpyval.pymc.prior_from_spec`: priors from a plain description
  that can be stored as JSON;
- :func:`~surpyval.pymc.to_surpyval`: the posterior draws back into a
  SurPyval model, with its functions, credible intervals and plots;
- :func:`~surpyval.pymc.pooled_likelihood` and
  :func:`~surpyval.pymc.group_models`: groups (sites, fleets, batches)
  sharing one population distribution, and a SurPyval model per group
  and for a new group.

They support the Weibull, Exponential, LogNormal, Gamma, LogLogistic,
Normal, Gumbel, Logistic, Rayleigh, Gompertz, InverseGaussian,
GeneralizedGamma and BirnbaumSaunders distributions
(``surpyval.pymc.SUPPORTED``); any other is refused with that list. (The
generalised gamma's likelihood, with Temme's expansion for small ``Q``,
is a larger graph than the others': pytensor takes about a minute to
compile it, once per model.) The
likelihood helpers also take the proportional hazards, accelerated failure
time and accelerated life fitters on those distributions
(``surpyval.pymc.REGRESSION``), with covariates.

From an old fleet to a new design
---------------------------------

A fleet of pumps of an old design has years of records, and a new design
has a handful. The question is the new design's B10 life. With six
records, the new design's shape parameter is barely known; the old
fleet's is. A hierarchical model lets the two designs share the shape and
pull their scales towards a common value, so the new design borrows
strength from the old without being forced to be the same.

The old fleet's records are of every kind: failures, pumps still running,
pumps found failed at an inspection (an interval), and pumps that were
already in service when records began at 2,000 hours (left truncated:
pumps that failed before then never made it into the records).

.. code:: python

    import numpy as np
    import pymc as pm
    import surpyval as sp
    from surpyval.pymc import censored_likelihood

    # The old design: x is a time, or [lower, upper] for an interval;
    # c is 0 failed, 1 still running, 2 failed between the two times;
    # n counts identical rows; tl is when each pump entered the records.
    old = sp.SurpyvalData(
        x=[12_400, 15_900, [17_000, 21_000], 19_800, 22_000, 24_500,
           [9_000, 13_000], 26_000, 8_700, 30_000],
        c=[0, 0, 2, 0, 1, 0, 2, 1, 0, 1],
        n=[1, 2, 3, 1, 9, 1, 2, 6, 1, 14],
        tl=[2_000, 2_000, 2_000, 0, 0, 2_000, 0, 2_000, 0, 2_000],
    )

    # The new design: three failures and three pumps still running.
    x_new = np.array([6_100, 9_800, 11_500, 8_000, 10_000, 12_000])
    c_new = np.array([0, 0, 0, 1, 1, 1])

    with pm.Model() as model:
        beta = pm.LogNormal("beta", mu=0.8, sigma=0.3)  # shared shape
        mu = pm.Normal("mu", mu=9.8, sigma=1.0)         # typical log scale
        tau = pm.HalfNormal("tau", sigma=0.5)           # how far designs differ
        z = pm.Normal("z", mu=0.0, sigma=1.0, shape=2)  # non-centred
        log_alpha = pm.Deterministic("log_alpha", mu + tau * z)

        censored_likelihood(sp.Weibull, old, name="old",
                            alpha=pm.math.exp(log_alpha[0]), beta=beta)
        censored_likelihood(sp.Weibull, x=x_new, c=c_new, name="new",
                            alpha=pm.math.exp(log_alpha[1]), beta=beta)

        trace = pm.sample(target_accept=0.95, random_seed=1)

Everything else is ordinary PyMC and ArviZ: ``az.summary(trace)``,
``az.plot_trace(trace)``, ``pm.find_MAP()``. (The scales are written
non-centred, ``mu + tau * z``, as usual for a hierarchy with few groups:
the centred form gives divergences here.) The B10 of the new design, for each posterior draw, is
the Weibull quantile at 0.1:

.. code:: python

    alpha_new = np.exp(trace.posterior["log_alpha"].values[..., 1]).ravel()
    beta_draws = trace.posterior["beta"].values.ravel()
    b10 = alpha_new * (-np.log(0.9)) ** (1 / beta_draws)
    np.percentile(b10, [5, 50, 95])   # a 90% credible interval and median

In one run with PyMC 6.3 this gave a median B10 of about 5,900 hours,
with a 90% credible interval of about 3,300 to 11,600 hours: wide, as six
records of a new design should leave it, with the shape borrowed from the
old fleet. (These numbers are illustrative: PyMC is not installed when
the documentation is built, so the code on this page is not run, and
another sampler version or seed gives slightly different draws.)

To use the posterior as a SurPyval model, turn the trace back into one with
:func:`~surpyval.pymc.to_surpyval`, naming the variable that holds each
parameter (with an index for an element of a vector) and how to get from
it to the parameter:

.. code:: python

    from surpyval.pymc import to_surpyval

    new_design = to_surpyval(trace, sp.Weibull,
                             alpha="log_alpha[1]", transform={"alpha": "exp"},
                             beta="beta", data=(x_new, c_new))
    new_design.quantile_cri(0.10)   # the B10's credible interval
    new_design.sf(10_000)           # the chance a new pump survives 10,000 h
    new_design.plot()               # its data, the curve and its band

The result is a :class:`~surpyval.bayes.BayesianParametric`, the same
model ``fit_bayes`` returns: its ``sf``, ``ff``, ``qf`` and ``mean`` are
those of a new unit with the uncertainty in the parameters averaged in,
and ``cri``, ``quantile_cri``, ``mean_cri`` and ``param_cri`` give
credible intervals. A parameter left out is the variable of its own name;
a transform is ``"exp"``, ``"log"``, ``"inverse"`` or a function of the
draws. Draws outside a parameter's bounds (a log scale passed without
``transform={"alpha": "exp"}``) are refused.

The data can be a :class:`~surpyval.SurpyvalData` (from arrays,
``from_items``, ``from_dataframe`` ...) or the arrays themselves, in any of
the forms ``fit`` takes: ``x``, ``c``, ``n`` with ``t`` or ``tl`` and
``tr``, or ``xl`` and ``xr`` for intervals. They are checked exactly as
``fit`` checks them, so a mistake (a censoring flag of 3, a time outside
its truncation window, a negative time for a Weibull) is refused with the
same message ``fit`` gives.

What each row adds
------------------

Each row adds one term to the log-likelihood, multiplied by its count
``n``. Writing :math:`f` for the density, :math:`F` for the CDF and
:math:`S = 1 - F` for the survival function:

==================================  ==========================================
row                                 term
==================================  ==========================================
failure at ``x`` (``c=0``)          :math:`\log f(x)`
running at ``x`` (``c=1``)          :math:`\log S(x)`
failed before ``x`` (``c=-1``)      :math:`\log F(x)`
failed in ``[xl, xr]`` (``c=2``)    :math:`\log[F(x_r) - F(x_l)]`
seen only in ``(tl, tr]``           minus :math:`\log[F(t_r) - F(t_l)]`
==================================  ==========================================

A truncation window is the condition for a row to be in the data at all,
so its probability divides every row it applies to. A censored row with a
truncation bound on its censored side becomes an interval ending at that
bound (a pump running at 9,000 hours in a study that ends at 10,000 can
only have failed between the two), as in ``fit``.

The terms are written in the stable forms SurPyval uses itself, so they
stay accurate far into the tails: :math:`\log F` of a Weibull is
:math:`\log(1 - e^{-H})` computed without cancellation, a window whose
lower end is in the upper tail is taken from the survival function
(:math:`\log S(t_l) + \log(1 - S(t_r) / S(t_l))`), and the normal tails
use the scaled complementary error function. At fixed parameters the term
equals SurPyval's own log-likelihood to 1e-10 or better, for every family,
row type and truncation pattern, and its gradient, which PyMC's samplers
use, matches a numerical derivative of SurPyval's (both are tested).

The term is a ``pm.Potential``. It counts in ``pm.sample`` and
``pm.find_MAP`` as an observed variable would, but, like any potential, it
is not drawn by the prior or posterior predictive samplers and is not in
the ``log_likelihood`` group that LOO and WAIC read.

Covariates
----------

A parameter can be one value for every row, or one value per row. Per-row
values give each row its own distribution, which is how covariates enter:

.. code:: python

    Z = np.column_stack([load, temperature])   # one row per data row
    with pm.Model():
        b0 = pm.Normal("b0", mu=9.0, sigma=2.0)
        b = pm.Normal("b", mu=0.0, sigma=1.0, shape=2)
        beta = pm.LogNormal("beta", mu=0.5, sigma=0.5)
        # A Weibull accelerated-failure-time model: log(alpha) linear in Z
        censored_likelihood(sp.Weibull, x=x, c=c, tl=tl,
                            alpha=pm.math.exp(b0 + Z @ b), beta=beta)

The rules:

- a parameter is a number, a PyMC variable or a pytensor expression, of
  shape ``()`` (every row) or ``(len(data),)`` (one per row); anything
  else is refused;
- per-row values follow the data's rows in order. Arrays passed as ``x``,
  ``c``, ... keep the order you give them. A ``SurpyvalData`` built from
  arrays without covariates merges identical rows and sorts them, so for
  per-row values pass the arrays, or build the ``SurpyvalData`` with
  ``Z=`` (which keeps the order) and use ``data.Z``;
- a length that is only known when the model runs is checked then.

Regression models
-----------------

The likelihood helpers take a regression fitter in place of a
distribution, with the covariates as ``Z``: the term is then exactly the
likelihood that fitter's ``fit`` maximises, its covariates entered as
SurPyval enters them. This is the way to the exact posterior of a
regression when SurPyval's own ``fit_bayes`` (a normal approximation, see
:doc:`Bayesian Fitting with SurPyval`) is not good enough, such as with
very few failures.

- **Proportional hazards** (``surpyval.WeibullPH``, ``LogNormalPH``, ...):
  the hazard is the baseline's times :math:`e^{\beta' Z}`. Give the
  baseline's parameters and the coefficients, by name (``coef_0=...``)
  or as one vector (``coefs=b``).
- **Accelerated failure time** (``surpyval.WeibullAFT``,
  ``surpyval.AFT(surpyval.Gamma)``, ...): the covariates speed up time,
  :math:`S(x \mid Z) = S_0(e^{\beta' Z} x)`; the same parameters.
- **Accelerated life** (``surpyval.AcceleratedLife(surpyval.Weibull,
  Power)``, ...): the life model's :math:`L(Z)` takes the place of the
  baseline's life parameter. Give the baseline's other parameters, and
  the life model's in a dict, ``life={"a": a, "n": n}`` (a dict, because
  their names, such as ``n`` and ``c``, can be the data's).

.. code:: python

    with pm.Model():
        alpha = pm.LogNormal("alpha", mu=np.log(100.0), sigma=1.0)
        beta = pm.LogNormal("beta", mu=np.log(1.5), sigma=0.5)
        b = pm.Normal("b", mu=0.0, sigma=1.0, shape=2)
        censored_likelihood(sp.WeibullPH, x=x, c=c, Z=Z,
                            alpha=alpha, beta=beta, coefs=b)
        trace = pm.sample(random_seed=1)

    from surpyval.life_models import Power
    with pm.Model():
        beta = pm.LogNormal("beta", mu=np.log(2.0), sigma=0.5)
        a = pm.LogNormal("a", mu=np.log(5e3), sigma=2.0)
        n = pm.Normal("n", mu=-2.0, sigma=0.5)   # past tests: n near -2
        censored_likelihood(sp.AcceleratedLife(sp.Weibull, Power),
                            x=x, c=c, Z=voltage, beta=beta,
                            life={"a": a, "n": n})

The tests check each kind against the fitter's own likelihood to
:math:`10^{-10}` for every row type and truncation, and its gradient
against a central difference; a Weibull proportional hazards model
sampled this way agrees with ``fit_bayes(..., importance=True)`` within
0.11 posterior standard deviations at the 5%, 50% and 95% points. A
custom proportional hazards link, and a life model that cannot be written
as a pytensor expression, are refused; give the baseline's parameter per
row yourself then (see Covariates).

Groups sharing a population
---------------------------

Eight sites run the same pumps. Most have thirty records, two have four.
Fitting each site alone gives the small sites wild answers; fitting all
of them as one hides real differences between the sites. The middle way
is a *hierarchical* model: each site has its own scale, but the sites'
scales are drawn from one population distribution, whose centre and
spread are estimated with them. A site with little data is then pulled
towards the population (it *borrows strength*), and a site with plenty
keeps its own value. How hard the pull is depends on how different the
sites turn out to be, which the data decide.

``pooled_likelihood`` builds this inside a PyMC model. For each pooled
parameter (here the Weibull scale ``alpha``), on its log scale,

.. math::

    \log \alpha_g = \mu + \tau z_g, \qquad z_g \sim N(0, 1),

with a Normal prior on the population's centre :math:`\mu` (``pool``
gives its centre and spread) and a half-normal prior on the spread between
sites :math:`\tau` (``tau_scale``). This is the non-centred form, which
samples well when groups have little data. The other parameters are
shared (here the shape, ``beta``), and the likelihood is SurPyval's,
every row at its own site's parameters.

.. code:: python

    from surpyval.pymc import group_models, pooled_likelihood

    with pm.Model():
        beta = pm.LogNormal("beta", mu=np.log(2.0), sigma=0.5)
        pooling = pooled_likelihood(
            sp.Weibull, x=x, c=c, group=site,
            pool={"alpha": (np.log(1000.0), 1.0)},   # a typical alpha ~1000
            shared={"beta": beta},
            tau_scale=0.5,
        )
        trace = pm.sample(random_seed=1, target_accept=0.95)

    models = group_models(trace, pooling, random_state=1)
    models["site 6"].param_cri("alpha")      # a small site, pulled in
    models["new group"].quantile_cri(0.1)    # a site not yet seen

``group_models`` gives a :class:`~surpyval.bayes.BayesianParametric` for
each site, and one for a ``"new group"``: each draw's population with a
fresh :math:`z`, so its uncertainty includes how much sites differ. In
the tests (eight simulated sites whose scales differ by a spread of 0.3 on
the log scale), the spread's 95% interval is 0.24 to 0.77, the two small
sites' scales move from 669 and 416 (each fitted alone) to 747 and 543,
towards the population's 967, and the new site's 90% interval for its
scale (446 to 2,036) is wider than any observed site's. A group with no
failures is kept, with a warning: its survivors still inform the
population.

``shrinkage(trace, pooling)`` says how far each group was pulled, as a
table: each group's own maximum-likelihood estimate (its data alone),
its pooled posterior median, the population's median, and ``pulled``,
the share of the way from its own estimate to the population's that
pooling moved it, on the log scale (0 is no pooling, 1 complete
pooling). The group's own fit frees every parameter, shared ones too, so
with a shared parameter ``pulled`` can fall a little outside 0 to 1.

When every group has plenty of data the non-centred form's :math:`z_g`
and :math:`\tau` are tied together and it samples worse; ``centred=True``
draws :math:`\log \alpha_g \sim N(\mu, \tau)` directly (one variable
``groups_alpha``) instead. ``group_models`` and ``shrinkage`` read
either.

Repairable systems: a fleet's growth rates
------------------------------------------

The power-law (Crow-AMSAA) process expects :math:`(t / \alpha)^\beta`
events by time :math:`t`: :math:`\beta` below 1 is reliability growth.
``nhpp_likelihood`` adds SurPyval's exact likelihood of systems' event
histories under it (the one ``CrowAMSAA.fit`` maximises: event times,
each system's end of observation, counts in windows, truncation) to a
PyMC model, and ``nhpp_log_likelihood`` gives it as an expression.

Systems of one design, run by different crews, improve at different
rates, but the rates are alike. ``pooled_likelihood(CrowAMSAA, ...)``
gives each system (or each group of systems, ``group=``) its own
:math:`\beta` drawn from a population, as above:

.. code:: python

    from surpyval.recurrent import CrowAMSAA

    with pm.Model():
        alpha = pm.LogNormal("alpha", mu=np.log(20.0), sigma=1.0)
        pooling = pooled_likelihood(
            CrowAMSAA, x=x, i=system, c=c,
            pool={"beta": (np.log(0.8), 0.5)},
            shared={"alpha": alpha},
            tau_scale=0.5,
        )
        trace = pm.sample(random_seed=1)

    models = group_models(trace, pooling, random_state=1)
    models["unit 7"].mtbf_cri()          # its MTBF now
    models["new group"].param_cri("beta")  # a system not yet seen
    shrinkage(trace, pooling)

Here ``group_models`` gives a
:class:`~surpyval.recurrent.bayes.BayesianRecurrence` a system, with its
MTBF, expected events and their intervals; the new system's is read at
the latest time any system was observed to. In the tests (eight systems,
growth rates log-normal about 0.8 with a spread of 0.3; two watched only
a short time, with four events each) the spread's 95% interval holds
0.3, the two short systems' rates move from 0.52 and 0.26 (each fitted
alone) to 0.62 and 0.53, towards the population's 0.74, and the new
system's interval on its rate is wider than any observed system's.

Mixtures
--------

A mixture of lives (two failure modes, a weak and a strong
subpopulation) has a likelihood that does not change when its
components are renumbered. Its posterior then has a copy of every peak
for each numbering, and a sampler's draws of "component 0" are a blend
of the components (*label switching*), so SurPyval's own engines refuse
it (``MixtureModel.fit_bayes``). ``mixture_likelihood`` adds the
mixture's exact likelihood, every row type and truncation as
``MixtureModel.fit`` takes them, and keeps one numbering: the
components are ordered by one parameter, which must increase from
component 0 to the last.

**Which parameter is ordered, and what it means.** By default the
family's first parameter: a Weibull's (or a Gamma's) ``alpha``, a
LogNormal's or Normal's ``mu``, an Exponential's rate. For a Weibull
mixture component 0 is then, in every draw, the component with the
shortest characteristic life (the "infant mortality" mode), and
component 1 the longer one; its weight ``w[0]`` is the share of units in
the short-lived subpopulation. Give that parameter PyMC's ordered
transform, so that every draw is ordered:

.. code:: python

    import pymc as pm
    import surpyval as sp
    from surpyval.pymc import mixture_likelihood, to_surpyval

    with pm.Model():
        w = pm.Dirichlet("w", a=[1.0, 1.0])
        log_alpha = pm.Normal(
            "log_alpha", mu=np.log([5.0, 50.0]), sigma=1.5, shape=2,
            transform=pm.distributions.transforms.ordered)
        beta = pm.LogNormal("beta", mu=0.3, sigma=0.5, shape=2)
        mixture_likelihood(
            sp.Weibull, x=x, c=c, weights=w,
            alpha=pm.math.exp(log_alpha), beta=beta)
        trace = pm.sample(random_seed=1)

A second Potential, ``likelihood_order``, is ``-inf`` outside the
ordered region (it is 0 for every draw of the ordered transform), so the
order also holds for parameters built some other way. ``order="beta"``
orders by another parameter (better where the components share a scale
and differ in shape), and ``order=None`` leaves the labels free.
Ordering does not help where two components have nearly the same value
of the ordered parameter: they can still trade places between draws, and
the parameter they differ in should be the one ordered.

``mixture_log_likelihood`` gives the same term as an expression. In the
tests it equals ``MixtureModel``'s own log-likelihood at given
parameters to 1e-10, with exact, right-, left- and interval-censored
rows, counts and truncation, for Weibull and LogNormal mixtures of two
and three components.

Priors from SurPyval
--------------------

``prior_from_model`` turns a fitted SurPyval model into a joint prior: a
multivariate normal on the log of each positive parameter (a real-line
parameter as it is), centred on the fit's estimates with its covariance.
It keeps how sure the old fit was, and how its parameters moved together.
``weight`` counts the old fit as a share of its information: ``weight=0.5``
doubles its covariance, for data only partly like the new.

.. code:: python

    from surpyval.pymc import prior_from_model, prior_from_spec

    last_year = sp.Weibull.fit(x_2025, c_2025)
    with pm.Model():
        prior = prior_from_model(last_year, name="fleet", weight=0.5)
        censored_likelihood(sp.Weibull, x=x_2026, c=c_2026, **prior)
        trace = pm.sample(random_seed=1)

The variables are ``fleet_z`` (the unconstrained vector) and
``fleet_alpha``, ``fleet_beta``. A model with no covariance (one built
with ``from_params``) is refused; say how sure you are with
``prior_from_spec`` instead.

``prior_from_spec`` builds independent priors from a description of each
parameter's prior: a SurPyval distribution, or the same as a tuple or a
dict, which can be stored as JSON. The parameters are SurPyval's, in
SurPyval's order, whatever PyMC calls them (a Weibull prior is ``(scale,
shape)``, a Gamma prior's second parameter is its rate):

.. code:: python

    with pm.Model():
        prior = prior_from_spec(
            {"alpha": ("lognormal", 9.0, 1.0),
             "beta": sp.LogNormal.from_params([0.8, 0.3])},
            dist=sp.Weibull,
        )
        censored_likelihood(sp.Weibull, x=x, c=c, **prior)

With ``dist`` given, every name must be one of its parameters, and a
prior that puts mass below zero on a positive parameter (a Normal on a
Weibull shape) is refused. The priors it builds are Normal, LogNormal,
Gamma, Weibull, Exponential, Uniform, Beta and Logistic.

API
---

.. autofunction:: surpyval.pymc.censored_likelihood

.. autofunction:: surpyval.pymc.log_likelihood

.. autofunction:: surpyval.pymc.prior_from_model

.. autofunction:: surpyval.pymc.prior_from_spec

.. autofunction:: surpyval.pymc.to_surpyval

.. autofunction:: surpyval.pymc.pooled_likelihood

.. autofunction:: surpyval.pymc.group_models

.. autofunction:: surpyval.pymc.shrinkage

.. autofunction:: surpyval.pymc.nhpp_likelihood

.. autofunction:: surpyval.pymc.nhpp_log_likelihood

.. autofunction:: surpyval.pymc.mixture_likelihood

.. autofunction:: surpyval.pymc.mixture_log_likelihood
