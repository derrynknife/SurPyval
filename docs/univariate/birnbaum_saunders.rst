Birnbaum-Saunders Distribution
==============================

**In plain words.** The Birnbaum-Saunders distribution describes a
positive, right-skewed time, such as a time to failure, to recovery or to
an event in a study. Half of all times fall below its scale ``beta`` (the
median), and its shape ``alpha`` says how spread out they are: for a
small shape the times cluster tightly around the median and their
logarithm is almost normal, so it looks much like a lognormal; for a
larger shape the spread is wider and the upper tail longer. Its hazard
rises to a peak and then falls towards a constant, like the lognormal's
and the inverse Gaussian's, so it is one of the families worth comparing
when a hazard does not keep rising. (It is also called the fatigue-life
distribution: Birnbaum and Saunders derived it in 1969 as the time for
damage built up in small, random steps to reach a critical level. Here it
is simply a two-parameter lifetime distribution.)

``BirnbaumSaunders`` is an instance of the class below: call its ``fit``
to estimate the parameters from data (exact, censored and truncated, with
an offset if wanted), or ``from_params`` for known parameters; both
return a :doc:`Parametric <parametric_class>` model.

Parameters
----------

- ``alpha`` (:math:`\alpha`), positive: the shape;
- ``beta`` (:math:`\beta`), positive: the scale, which is the median.

With

.. math::

    \xi(x) = \frac{1}{\alpha}\left(\sqrt{\frac{x}{\beta}} -
    \sqrt{\frac{\beta}{x}}\right),

the distribution function is the standard normal's at :math:`\xi`:

.. math::

    F(x) = \Phi(\xi(x)), \qquad
    f(x) = \frac{x + \beta}{2\alpha\sqrt{\beta}\,x^{3/2}}\,\phi(\xi(x)),

with :math:`\phi` the standard normal density. The quantile has a closed
form, :math:`q(u) = \tfrac{\beta}{4}\left(\alpha z + \sqrt{\alpha^2 z^2 +
4}\right)^2` with :math:`z = \Phi^{-1}(u)`; the mean is
:math:`\beta(1 + \alpha^2/2)` and the variance :math:`(\alpha\beta)^2(1 +
5\alpha^2/4)`. The hazard tends to :math:`1/(2\alpha^2\beta)` as
:math:`x` grows.

This is the common statistical parameterisation, the one of
``scipy.stats.fatiguelife(alpha, scale=beta)`` and R's ``VGAM``
(``dbisa(x, scale = beta, shape = alpha)``). Note that the names are the
other way round from SurPyval's Weibull, whose ``alpha`` is the scale and
``beta`` the shape: here they follow the literature on this distribution.

Limits
------

As :math:`\alpha` falls to 0 the distribution concentrates on
:math:`\beta`, and :math:`\ln x` tends to a normal with mean
:math:`\ln\beta` and standard deviation :math:`\alpha`: the lognormal.

How it is computed
------------------

:math:`\xi` is computed as :math:`(x - \beta) / (\alpha\sqrt{x}\sqrt{\beta})`,
which does not lose digits near the median, and the logs of both tails
from the normal's log tail functions, so they hold far into either tail.
The quantile below the median is written as :math:`\beta / (\sqrt{w^2 +
1} - w)^2`, :math:`w = \alpha z / 2`, where the sum in the closed form
cancels; ``isf`` works from the survival probability itself, so that it
holds deep in the upper tail. The raw moments have a closed form:
:math:`E[X^m]` is :math:`\beta^m` times a polynomial in :math:`\alpha^2`
with integer coefficients (from the Chebyshev polynomial :math:`T_{2m}`).

Validation
----------

The density, distribution and survival functions and the quantiles agree
with ``scipy.stats.fatiguelife`` to 1e-10 from a thousandth to a thousand
times the median, for shapes from 0.05 to 8; the far tails (out to
:math:`10^{\pm 6}` times the median) and quantiles down to
:math:`10^{-300}` agree with 60-digit mpmath to 1e-10 or better. The mean,
variance and moments agree with their closed forms and with quadrature.
The maximum likelihood fits to right-censored, left-censored,
interval-censored and left-truncated data, and to survival's ``lung``
data, agree with a likelihood written independently and maximised by
Nelder-Mead to 1e-8 in log-likelihood, 1e-6 in the estimates and 1e-3 in
the standard errors (``surpyval/tests/reference/test_birnbaum_saunders.py``).
A check against R's ``VGAM`` is committed
(``scripts/reference/reference_r_birnbaum_saunders.R``) and runs once its
output is generated.

Covariates
----------

``AFT(BirnbaumSaunders)`` rescales time, which multiplies ``beta``;
``Distributional(BirnbaumSaunders)`` puts covariates on either parameter
(raising ``beta`` makes every quantile later; ``alpha`` has no direction:
it moves the quantiles below the median earlier and those above it later).

What it does not do
-------------------

**Probability-plot fitting** (``how="MPP"``) is refused: the straight
line, :math:`x = \beta + \alpha\sqrt{\beta}\,z\sqrt{x}`, mixes the time
with the probability, so no axis of the time alone and one of the
probability alone makes it straight. ``plot()`` draws on the fitted
shape. ``PH(BirnbaumSaunders)`` fits a proportional hazards model on
this baseline, but there the covariates do not keep the distribution in
the family (a multiple of its hazard is not a Birnbaum-Saunders hazard),
so it has no map between the baseline at the covariates' means and at 0
(as the AFT model has); centre the covariates, or use ``center=True``,
if they are far from 0.

.. autoclass:: surpyval.univariate.parametric.distributions.birnbaum_saunders.BirnbaumSaunders_
   :members:
   :inherited-members:
