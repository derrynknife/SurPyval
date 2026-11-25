Inverse Gaussian Distribution
=============================

**In plain words.** Picture a quantity that drifts steadily towards a
limit but wanders randomly as it goes: wear accumulating with noise, a
patient's condition improving unevenly, a price drifting to a barrier.
The time it first reaches the limit has the inverse Gaussian
distribution. That makes it the natural lifetime of a unit that fails
when noisy, accumulated damage crosses a threshold, and a common model
for durations in medicine and economics. Its hazard rises to a peak and
then falls to a constant, so a unit that has survived well past the
typical time goes on failing at a steady rate.

``InverseGaussian`` is an instance of the class below: call its ``fit``
to estimate the parameters from data (exact, censored and truncated, with
an offset if wanted), or ``from_params`` for known parameters; both
return a :doc:`Parametric <parametric_class>` model.

Parameters
----------

The parameterisation is that of R's ``statmod`` (``mean``, ``shape``) and
``SuppDists`` (``nu``, ``lambda``):

- ``mu``, positive: the mean;
- ``lam`` (:math:`\lambda`), positive: the shape; the variance is
  :math:`\mu^3/\lambda`, so a larger :math:`\lambda` is a tighter
  distribution.

With :math:`a = \sqrt{\lambda/x}\,(x/\mu - 1)` and :math:`b =
\sqrt{\lambda/x}\,(x/\mu + 1)`,

.. math::

    f(x) = \sqrt{\frac{\lambda}{2\pi x^3}}
    \exp\left(-\frac{\lambda (x - \mu)^2}{2\mu^2 x}\right), \qquad
    S(x) = \Phi(-a) - e^{2\lambda/\mu}\,\Phi(-b).

The hazard tends to :math:`\lambda / (2\mu^2)` as :math:`x` grows.

Limits
------

As :math:`\lambda/\mu` grows the distribution tends to a normal with
mean :math:`\mu` and variance :math:`\mu^3/\lambda`. The first passage
time of a Wiener process with drift :math:`\nu` and diffusion
:math:`\sigma` to a level :math:`h` is ``InverseGaussian(h / nu, h**2 /
sigma**2)``, the lifetime ``WienerProcess`` models implies (see
:doc:`../Degradation Modelling with SurPyval`).

How it is computed
------------------

Written as it stands, :math:`e^{2\lambda/\mu}` overflows (from
:math:`\lambda/\mu = 355`) and the difference in :math:`S` loses its
digits above the mean. SurPyval computes :math:`F` as the sum of its two
positive terms in logs, and above the mean :math:`S = \phi(a)\,(M(a) -
M(b))`, with :math:`M` Mills' ratio; far above the mean, where
:math:`M(a)` and :math:`M(b)` agree to many digits, their difference is
the integral of :math:`-M'` between them. The hazard is
:math:`\sqrt{\lambda/x^3}/(M(a) - M(b))`, with :math:`\phi(a)`, which
underflows, cancelled. The quantile has no closed form: it is solved by
Newton's method on the log of the smaller tail.

Exact, untruncated data have a closed-form maximum likelihood estimate,
:math:`\hat\mu = \bar x` and :math:`1/\hat\lambda = \overline{1/x} -
1/\bar x`, which ``fit`` uses.

Validation
----------

Against statmod 1.5.2 the density, distribution function, log survival
function and quantiles agree to 1e-10 for :math:`\lambda/\mu` from 0.15
to 500, and against SuppDists the density and distribution function
(except where SuppDists overflows, at :math:`\lambda/\mu = 500`). The far
upper tail agrees with mpmath to 1e-13 up to :math:`x = 10^8 \mu`. The
fits to survival's ``lung`` data (right censored, and left truncated)
agree with a likelihood maximised in R to 1e-6 in log-likelihood and a
hundredth of a standard error in the estimates
(``surpyval/tests/reference/test_inverse_gaussian.py``).

Covariates
----------

``AFT(InverseGaussian)`` rescales time, which multiplies both ``mu`` and
``lam``; ``Distributional(InverseGaussian)`` puts covariates on either
(raising ``mu`` makes every quantile later; ``lam`` has no direction).

What it does not do
-------------------

**Probability-plot fitting** (``how="MPP"``) is refused: the straight-line
axis is the quantile of the inverse Gaussian of mean 1 and shape
:math:`\lambda/\mu`, which needs the answer. ``plot()`` draws on the
fitted shape.

.. autoclass:: surpyval.univariate.parametric.distributions.inverse_gaussian.InverseGaussian_
   :members:
   :inherited-members:
