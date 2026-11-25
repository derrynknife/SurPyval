Gompertz Distribution
=====================

**In plain words.** The Gompertz distribution describes a risk that grows
by a fixed percentage for every unit of age: if the hazard rises 8% a
year, it doubles about every nine years. Benjamin Gompertz found this
pattern in human mortality in 1825, and it still describes adult
mortality, ageing in laboratory animals and many wear-out processes well.
It differs from the Weibull, whose hazard grows as a power of age: the
Gompertz hazard eventually outruns any power.

The shape may also be negative. Then the hazard *falls* by a fixed
percentage per unit of age, the cumulative hazard levels off, and a
fraction of units never fails: the survival curve has a floor, as in a
cure model. SurPyval allows this, as R's ``flexsurv`` does, and says so
in the results (an infinite mean, infinite quantiles above the fraction
that fails, ``inf`` in random draws).

``Gompertz`` is an instance of the class below: call its ``fit`` to
estimate the parameters from data (exact, censored and truncated, with an
offset if wanted), or ``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

Parameters
----------

The parameterisation is R's ``flexsurv`` (``gompertz``) and ``eha``:

- ``shape`` :math:`a`, any real number: the hazard's growth rate per unit
  of time (its logarithm rises by :math:`a` per unit);
- ``rate`` :math:`b`, positive: the hazard at time 0.

.. math::

    h(x) = b e^{a x}, \qquad
    H(x) = \frac{b}{a}\left(e^{a x} - 1\right), \qquad
    S(x) = e^{-H(x)},

with :math:`H(x) = b x` at :math:`a = 0`, the exponential. For
:math:`a < 0`, :math:`S(\infty) = e^{b/a} > 0`.

Special cases and limits
------------------------

- ``shape = 0`` is the exponential with the same rate
  (``Exponential(rate)``); the functions are exact through 0, where
  :math:`H` is computed as :math:`b x (1 + a x/2 + (a x)^2/6 + \cdots)`,
  and so are their derivatives in the shape.
- A negative shape is a defective (cure-like) distribution with a
  never-failing fraction :math:`e^{b/a}`.

Mean and moments
----------------

For :math:`a > 0` the mean is :math:`e^{b/a} E_1(b/a) / a`, with
:math:`E_1` the exponential integral (computed so it holds for a shape
near 0, where it tends to :math:`1/b`); higher moments are computed by
quadrature. For :math:`a < 0` they are infinite.

Covariates
----------

``PH(Gompertz)`` multiplies the rate by :math:`e^{\beta' Z}`, the model
``flexsurvreg(..., dist = "gompertz")`` fits with covariates; the shape is
shared. ``AFT(Gompertz)`` rescales time, which multiplies both parameters
by :math:`e^{-\beta' Z}`. Both are exact reparameterisations under a
change of the covariates' origin, so ``center=True`` and the default agree.

Validation
----------

Against flexsurv 2.3.2 the density, distribution function, log survival
function and quantiles agree to 1e-10 for positive, zero, tiny and
negative shapes, and the fits to survival's ``lung`` data (in years:
right censored, left truncated, and proportional hazards on ``sex``)
agree in log-likelihood to 1e-6 and in the estimates to a hundredth of a
standard error (``surpyval/tests/reference/test_gompertz.py``).

What it does not do
-------------------

**Probability-plot fitting** (``how="MPP"``) is refused: the straight-line
axis, :math:`\ln(1 - (a/b)\ln(1 - F))`, needs the ratio of the two
parameters being estimated. ``plot()`` draws on the fitted parameters; fit
by ``"MLE"`` (the default), ``"MPS"``, ``"MSE"`` or ``"MOM"``.

.. autoclass:: surpyval.univariate.parametric.distributions.gompertz.Gompertz_
   :members:
   :inherited-members:
