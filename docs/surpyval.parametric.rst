Parametric
==========

Lifetime distributions with a fixed functional form, fitted by estimating
a few parameters. Each distribution is exported as a ready-made instance
(``surpyval.Weibull``, ``surpyval.LogNormal``, ...). Its ``fit`` accepts
any mix of observed, censored and truncated data and returns a
:doc:`Parametric <univariate/parametric_class>` model; its
``from_params`` builds the same model from known parameters. The
distribution's own functions (``sf``, ``ff``, ``df``, ``hf``, ``Hf``,
``qf``, ``mean``, ``moment``, ``random``, ...) can also be called directly
with the parameters as extra arguments, for example
``Weibull.sf(x, alpha, beta)``.

``fit`` supports five estimation methods (``how='MLE'``, ``'MPP'``,
``'MOM'``, ``'MSE'``, ``'MPS'``), fixed parameters (``fixed``), an offset
(``offset=True``), a limited failure population (``lfp=True``) and zero
inflation (``zi=True``); :func:`~surpyval.fit_best.fit_best` fits every
candidate distribution and keeps the best (see
:doc:`comparison_and_validation`). The theory is in
:doc:`Parametric Estimation` and worked examples are in
:doc:`Parametric SurPyval Modelling`.

Parametric Class
----------------

The fitted model every distribution's ``fit`` and ``from_params``
returns.

.. toctree::
   :maxdepth: 1

   univariate/parametric_class


Distribution Classes
--------------------

Continuous lifetime distributions.

.. toctree::
   :maxdepth: 1

   univariate/exponential
   univariate/hypoexponential
   univariate/weibull
   univariate/expo_weibull
   univariate/gumbel
   univariate/gumbel_lev
   univariate/gamma
   univariate/normal
   univariate/lognormal
   univariate/logistic
   univariate/loglogistic
   univariate/uniform
   univariate/rayleigh
   univariate/beta
   univariate/beta4

Discrete Distribution Classes
-----------------------------

Discrete lifetimes on the positive integers, for cycle- or
demand-counted data. All accept the same censoring and truncation
formats as the continuous distributions.

.. toctree::
   :maxdepth: 1

   univariate/geometric
   univariate/poisson
   univariate/binomial
   univariate/negative_binomial
   univariate/beta_geometric
   univariate/discrete_weibull
   univariate/discretize

Special Distributions
---------------------

Per-demand and degenerate models with no (or a fixed) time dimension,
for composing into mixtures, competing risks and demand studies.

.. toctree::
   :maxdepth: 1

   univariate/bernoulli
   univariate/fixed_event_probability
   univariate/exact_event_time
   univariate/degenerate

Flexible Parametric (Royston-Parmar)
------------------------------------

A spline model for data whose hazard no standard distribution fits.

.. toctree::
   :maxdepth: 1

   univariate/royston_parmar

Custom Distributions
--------------------

Define a new distribution from its cumulative hazard function alone.

.. toctree::
   :maxdepth: 1

   univariate/custom

Mixture Modelling
-----------------

A population made of several sub-populations, each with its own
distribution of the same family.

.. toctree::
   :maxdepth: 1

   univariate/mixture
