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
  that can be stored as JSON.

They support the Weibull, Exponential, LogNormal, Gamma, LogLogistic,
Normal, Gumbel, Logistic and Rayleigh distributions
(``surpyval.pymc.SUPPORTED``); any other is refused with that list.

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

With PyMC 6.3 this gives a median B10 of about 5,900 hours, with a 90%
credible interval of about 3,300 to 11,600 hours: wide, as six records of
a new design should leave it, with the shape borrowed from the old fleet.

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
