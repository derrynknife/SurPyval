Generalised F Distribution
==========================

**In plain words.** The generalised F adds one more shape parameter,
``P``, to the generalised gamma. Where the generalised gamma already
holds the Weibull, the lognormal and the gamma, the generalised F also
holds the log-logistic, allows a heavier (power-law) upper tail, and lets
the hazard take more shapes. It is the most flexible of the standard
parametric survival families, which is also its weakness: four
parameters ask a lot of the data. Fit the generalised gamma first and
compare the two by likelihood ratio before relying on ``P``.

``GeneralizedF`` is an instance of the class below: call its ``fit`` to
estimate the parameters from data (exact, censored and truncated, with an
offset if wanted), or ``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

Parameters
----------

The parameterisation is Prentice's (1975), the one R's ``flexsurv`` calls
``genf``, so results compare one to one with ``flexsurvreg``:

- ``mu``, any real number: the location of the log time;
- ``sigma``, positive: the scale of the log time;
- ``Q``, any real number, and ``P``, positive: the shapes.

With :math:`\delta = \sqrt{Q^2 + 2P}`, :math:`s_1 = 2/(Q^2 + 2P +
Q\delta)`, :math:`s_2 = 2/(Q^2 + 2P - Q\delta)` and :math:`z = \delta(\ln
x - \mu)/\sigma`, the variable :math:`e^{z}` has an F distribution on
:math:`2s_1` and :math:`2s_2` degrees of freedom, so with :math:`B = s_1
e^{z}/(s_1 e^{z} + s_2)` (a beta variable of shapes :math:`s_1`,
:math:`s_2`)

.. math::

    S(x) = 1 - I_B(s_1, s_2), \qquad
    f(x) = \frac{\delta\,(s_1/s_2)^{s_1} e^{s_1 z}}{\sigma x\, B(s_1, s_2)
    \left(1 + s_1 e^{z}/s_2\right)^{s_1 + s_2}},

with :math:`I` the regularised incomplete beta function.

Special cases
-------------

- ``P -> 0`` is the generalised gamma with the same ``mu``, ``sigma`` and
  ``Q`` (a limit: ``P`` must be positive). On data the generalised gamma
  describes, the likelihood is highest at that edge: the fit stops near
  ``P = 0`` and warns that it has not reached a verified maximum. Fit
  ``GeneralizedGamma`` there, and compare the two by likelihood ratio.
- ``Q = 0, P = 1`` is the log-logistic with scale :math:`e^{\mu}` and
  shape :math:`\sqrt 2/\sigma`.

Moments
-------

:math:`E[X^m] = e^{m\mu} (s_2/s_1)^{k}\, B(s_1 + k, s_2 - k)/B(s_1, s_2)`
with :math:`k = m\sigma/\delta`. The upper tail falls as a power of
:math:`x`, so only the moments with :math:`s_2 > k` exist; the others are
``inf``.

How it is computed
------------------

Each tail comes from whichever of :math:`B` and :math:`1 - B` is at most
1/2, both computed from logs (:math:`\ln B = -\operatorname{softplus}(-(z
+ \ln(s_1/s_2)))`), and the other tail is its complement: the incomplete
beta near an argument of 1 works from its rounded complement (``F`` at
0.999 was 7e-6 off where :math:`s_2 = 0.2`). The shape that cancels in
:math:`Q^2 + 2P - Q\delta` when :math:`P \ll Q^2` is taken from
:math:`s_1 s_2 = 2/(P(Q^2 + 2P))`. The quantile is the inverse incomplete
beta, refined by Newton's method on the log of the smaller tail.

Validation
----------

Against flexsurv 2.3.2 the density, distribution function and log survival
function agree to 1e-10 at parameter sets with both signs of ``Q``, ``Q
= 0`` and ``P`` from 1e-4 to 2; against mpmath to 5e-13 down to ``P =
1e-6``. flexsurv's ``qgenf`` (through R's ``qf``) is itself off by up to
6e-8 of the probability (and fails at some parameters), so the quantiles
are checked by inverting ``ff`` to 1e-14 and against ``qgenf`` where
``qgenf`` inverts ``pgenf``. The fits to survival's ``lung`` data (right
censored, left truncated, and AFT on ``sex``) agree in log-likelihood to
1e-6 and in the estimates to a hundredth of a standard error
(``surpyval/tests/reference/test_generalized_f.py``).

Bounds
------

Wald bounds are immediate; likelihood-ratio bounds take tens of seconds
(38 s for ``sf`` at two times on 40 rows), since each point profiles four
parameters; the bootstrap refits the four-parameter model each time and
takes many minutes.

**Which bound to trust at moderate sample sizes.** With a few hundred
units the data often cannot tell ``P`` apart from one of its two edges,
and then the Wald bounds on ``mu``, ``sigma`` and the mean are too
narrow. Prefer ``method="lr"`` (or the bootstrap) there. In a study of
n = 200 units from ``GeneralizedF(2, 0.5, 0.5, 1)``, 30% censored, the
95% Wald bounds covered ``mu``, ``sigma`` and the mean only 91%, 87% and
84% of the time, while the likelihood-ratio bounds on all four
parameters covered 94-96%; the Wald bounds reached 95% by n = 500 to
1000 (``surpyval/tests/calibration/test_coverage_new_families.py``).
The bounds on ``sf`` and on quantiles such as B10 were fine at n = 200.

The two edges are these. As ``P`` falls to 0 the model becomes the
generalised gamma, and the fit warns when it stops there: fit
``GeneralizedGamma`` instead. As ``P`` grows while ``sigma`` shrinks
(:math:`\sigma^2 P` and :math:`\sigma Q` held), the log time tends to
an asymmetric Laplace distribution about ``mu`` (the time a power law
either side of :math:`e^\mu`), which no member reaches. At n = 200
about 3 fits in 10 were at one edge or the other.

On some samples that limit fits better than any member, and the fit
cannot reach it: the search stops at a local maximum (on one sample of
200, at ``P = 1.8``, 0.29 below the limit in log-likelihood) or out on
the ridge towards it with ``P`` in the thousands. The fit then warns "No
finite maximum", naming the limit's ``mu`` and the rates ``a`` below and
``b`` above it at which the log time falls away; the reported ``sigma``
and ``P`` and the Wald bounds mean nothing there, and the
likelihood-ratio bounds allow for the limit. Where both edges are at
least as likely as the fit, the warning names this one, the better.

The likelihood-ratio bounds allow for both. Where a fit is at the
generalised gamma edge, the profiles are also searched from ``P = 1``,
since the likelihood is flat in ``log P`` there. Where the asymmetric
Laplace limit fits the data better than the fit, the deviances are
measured from the limit. Where a profile runs along the ridge to that
limit and the limit is inside the region, the bound is the edge of the
parameter's space: 0 for ``sigma``'s lower side, ``inf`` for ``P``'s
upper side, and plus or minus ``inf`` for ``Q``, whichever way the
limit's asymmetry points.

Covariates
----------

``AFT(GeneralizedF)`` puts covariates on ``mu`` (flexsurv's covariates on
``genf``), and ``Distributional(GeneralizedF)`` on any parameter.

What it does not do
-------------------

- **Probability-plot fitting** (``how="MPP"``) is refused: the axis
  depends on ``Q`` and ``P``. ``plot()`` draws on the fitted shapes.
- **The method of moments** (``how="MOM"``) is refused: the upper tail
  falls as a power of :math:`x`, so the fourth moment it would match may
  not exist (on 400 heavy-tailed quantiles a moments fit put ``sf(1)`` at
  0.55 where the maximum likelihood fit has 0.92). Fit by ``"MLE"`` (the
  default), ``"MPS"`` or ``"MSE"``.
- **The PyMC and torch helpers** do not take it yet: torch has no
  incomplete beta function, and the two helpers take the same families.

.. autoclass:: surpyval.univariate.parametric.distributions.generalized_f.GeneralizedF_
   :members:
   :inherited-members:
