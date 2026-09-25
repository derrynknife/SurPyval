Parametric Regression Models
=============================

A fitted parametric baseline distribution combined with a covariate
function. SurPyval provides four families in which the covariates act on
the baseline in a fixed way -- proportional hazards (PH), accelerated
failure time (AFT), proportional odds (PO) and parametric additive hazards
(AH) -- plus the Accelerated Life (AL) family for physics-motivated
stress relationships. The theory is in :doc:`../regression analysis` and
worked examples are in :doc:`../Regression Modelling with SurPyval`.

Each of the four covariate families is available as:

1. **Pre-built instances** -- ready to use with any number of covariates
   (``WeibullPH``, ``LogNormalAFT``, ...);
2. **Factory functions** -- compose any surpyval distribution on the fly
   (``PH(dist)``, ``AFT(dist)``, ``PO(dist)``, ``AH(dist)``);
3. **Low-level fitter classes** -- documented below, and the class of the
   instances.

PH, AFT and PO use the log-linear covariate function
:math:`\phi(Z) = e^{\beta'Z}`; AH adds :math:`\beta'Z` to the hazard;
Accelerated Life models use a stress function chosen from the life
models below. Every ``fit`` (and ``fit_from_df``, which takes a DataFrame
and column names or a formula) returns a
:class:`~surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel`
(documented at the end of this page),
whose ``params`` are the distribution parameters followed by the
covariate coefficients.

**Signs.** In PH, AFT and AH a positive coefficient shortens life; in PO
a positive coefficient *lengthens* it (it multiplies the survival odds).

**Time-varying covariates.** Where the cumulative hazard is additive
over disjoint intervals, the proportional-hazards and additive-hazards
families (``WeibullPH`` / ``PH(dist)`` and ``WeibullAH`` / ``AH(dist)``)
also *fit* start-stop time-varying-covariate data with ``fit_tvc`` /
``fit_tvc_timeline``, reusing the ordinary maximum-likelihood fit; the
AFT family fits it with its own accumulated-age likelihood. A fitted PH,
AH or AFT model can then be *evaluated* along a piecewise-constant
covariate path with ``sf_tvc`` / ``Hf_tvc``, describing the path as a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`. See
:ref:`tvc-parametric` in the how-to guide.


Proportional Hazards (PH)
--------------------------

.. math::

    h(x \mid Z) = h_0(x) \cdot \phi(Z)

Pre-built instances: ``ExponentialPH``, ``NormalPH``, ``WeibullPH``,
``GumbelPH``, ``LogisticPH``, ``LogNormalPH``, ``GammaPH``.

.. autofunction:: surpyval.univariate.regression.proportional_hazards.PH

.. autoclass:: surpyval.univariate.regression.proportional_hazards.proportional_hazards_fitter.ProportionalHazardsFitter
    :members: fit, fit_from_df, fit_tvc, fit_tvc_from_df, fit_tvc_timeline, fit_tvc_timeline_from_df, Hf, hf, sf, ff, df


Accelerated Failure Time (AFT)
--------------------------------

.. math::

    H(x \mid Z) = H_0\!\left(e^{\beta'Z} \cdot x\right)

Pre-built instances: ``ExponentialAFT``, ``NormalAFT``, ``WeibullAFT``,
``GumbelAFT``, ``LogisticAFT``, ``LogNormalAFT``, ``GammaAFT``.

.. autofunction:: surpyval.univariate.regression.accelerated_failure_time.aft_fitter.AFT

.. autoclass:: surpyval.univariate.regression.accelerated_failure_time.aft_fitter.AFTFitter
    :members: fit, fit_from_df, fit_tvc, fit_tvc_from_df, fit_tvc_timeline, Hf, hf, sf, ff, df


Proportional Odds (PO)
-----------------------

.. math::

    \frac{S(x \mid Z)}{F(x \mid Z)} = \frac{S_0(x)}{F_0(x)} \cdot e^{\beta'Z}

Pre-built instances: ``ExponentialPO``, ``NormalPO``, ``WeibullPO``,
``GumbelPO``, ``LogisticPO``, ``LogNormalPO``, ``GammaPO``.

.. autofunction:: surpyval.univariate.regression.proportional_odds.proportional_odds_fitter.PO

.. autoclass:: surpyval.univariate.regression.proportional_odds.proportional_odds_fitter.ProportionalOddsFitter
    :members: fit, fit_from_df, Hf, hf, sf, ff, df


Additive Hazards (AH)
---------------------

.. math::

    h(x \mid Z) = h_0(x) + \beta'Z, \qquad H(x \mid Z) = H_0(x) + x\,\beta'Z

The fully parametric counterpart of the semi-parametric Lin-Ying model
(:doc:`additive_hazards`). Nothing keeps the hazard positive, so a fit
whose hazard would be non-positive at an observed event fails rather
than return an invalid model.

Pre-built instances: ``ExponentialAH``, ``NormalAH``, ``WeibullAH``,
``GumbelAH``, ``LogisticAH``, ``LogNormalAH``, ``GammaAH``.

.. autofunction:: surpyval.univariate.regression.additive_hazards.AH

.. autoclass:: surpyval.univariate.regression.additive_hazards.additive_hazards_fitter.AdditiveHazardsFitter
    :members: fit, fit_from_df, fit_tvc, fit_tvc_from_df, fit_tvc_timeline, fit_tvc_timeline_from_df, Hf, hf, sf, ff, df


Accelerated Life (AL)
----------------------

AL models substitute the life parameter of a distribution with a
physics-motivated stress function :math:`L(Z)`. They are designed for a
few discrete, controlled stress levels (e.g. temperature, voltage).
Supported distributions, and the parameter the life replaces:
``Weibull`` (:math:`\alpha = L`), ``Normal``, ``Gumbel`` and ``Logistic``
(:math:`\mu = L`), ``LogNormal`` (:math:`\mu = \ln L`), ``Exponential``
and ``Gamma`` (the rate is :math:`1/L`).

Factory::

    from surpyval import Weibull
    from surpyval import AcceleratedLife, Power, Eyring
    model = AcceleratedLife(Weibull, Power).fit(x, Z=stress, c=c)

.. autofunction:: surpyval.univariate.regression.accelerated_life.accelerated_life.AcceleratedLife

The available life models, importable from ``surpyval``, with :math:`Z`
the stress (:math:`Z_1, Z_2` for the two-stress models) and the
parameter names as the fitted model reports them:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Life model
     - :math:`L(Z)`
     - Parameters
   * - ``Power``
     - :math:`a Z^{n}`
     - ``a`` (> 0), ``n``
   * - ``InversePower``
     - :math:`1 / (a Z^{n})`
     - ``a`` (> 0), ``n``
   * - ``ExponentialLifeModel``
     - :math:`b\, e^{a / Z}`
     - ``a``, ``b`` (> 0)
   * - ``InverseExponential``
     - :math:`1 / (b\, e^{a / Z})`
     - ``a``, ``b`` (> 0)
   * - ``Eyring``
     - :math:`\frac{1}{Z} e^{-(b - a / Z)}`
     - ``a``, ``b``
   * - ``InverseEyring``
     - :math:`Z\, e^{c - a / Z}`
     - ``a``, ``c``
   * - ``Linear``
     - :math:`a + b Z`
     - ``a``, ``b``
   * - ``DualExponential``
     - :math:`c\, e^{a / Z_1} e^{b / Z_2}`
     - ``a``, ``b``, ``c`` (> 0)
   * - ``DualPower``
     - :math:`c\, Z_1^{m} Z_2^{n}`
     - ``c`` (> 0), ``m``, ``n``
   * - ``PowerExponential``
     - :math:`c\, e^{a / Z_1} Z_2^{n}`
     - ``c`` (> 0), ``a``, ``n``

Custom life models can be created by subclassing ``LifeModel``::

    from surpyval import LifeModel, AcceleratedLife
    import autograd.numpy as anp

    class MyStressModel(LifeModel):
        def __init__(self):
            super().__init__(
                name="MyStressModel",
                phi_param_map={"a": 0, "b": 1},
                phi_bounds=((None, None), (None, None)),
            )

        def phi(self, Z, *params):
            a, b = params
            return anp.exp(a + b * Z)

        def phi_init(self, life, Z):
            b, a = anp.polyfit(Z.flatten(), anp.log(life), 1)
            return [float(a), float(b)]

    model = AcceleratedLife(Weibull, MyStressModel()).fit(x, Z=stress, c=c)

.. autoclass:: surpyval.univariate.regression.accelerated_life.parameter_substitution.ParameterSubstitutionFitter
    :members: fit, fit_from_df, sf, ff, df

.. autoclass:: surpyval.univariate.regression.accelerated_life.lifemodel.LifeModel
    :members:


Time-varying covariate schedules
--------------------------------

A ``StepSchedule`` describes a piecewise-constant covariate path ``Z(t)`` for
``sf_tvc`` / ``Hf_tvc`` evaluation. Build one structurally
(``from_changepoints`` / ``from_intervals`` / ``cyclic`` / ``constant``) or from
a step-valued expression string in ``t`` (``from_expression``); see
:ref:`tvc-parametric` for worked examples. ``StepValuedError`` is raised
when an expression is not provably step-valued.

.. autoclass:: surpyval.univariate.regression.tvc_schedule.StepSchedule
    :members:

.. autoclass:: surpyval.univariate.regression.tvc_schedule.StepValuedError
    :exclude-members: add_note, with_traceback


The fitted model
----------------

Every parametric regression ``fit`` returns a
``ParametricRegressionModel`` carrying the fitted distribution and
covariate parameters together with prediction, plotting and
serialisation methods.

.. autoclass:: surpyval.univariate.regression.parametric_regression_model.ParametricRegressionModel
    :members:
