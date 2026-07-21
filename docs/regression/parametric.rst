Parametric Regression Models
=============================

SurPyval provides three families of parametric regression model plus the
Accelerated Life family for physics-motivated stress relationships. Each family
is available as:

1. **Pre-built instances** — ready to use with any number of covariates
2. **Factory functions** — compose any surpyval distribution on the fly
3. **Low-level fitter classes** — for custom phi functions or life models

All parametric regression models use :math:`\phi(Z) = e^{\beta'Z}` (log-linear
covariates) except Accelerated Life models, which use domain-specific stress
functions.

**Time-varying covariates.** Where the cumulative hazard is additive over
disjoint intervals, the proportional-hazards and additive-hazards families
(``WeibullPH`` / ``PH(dist)`` and ``WeibullAH`` / ``AH(dist)``) also *fit*
start-stop time-varying-covariate data with ``fit_tvc`` / ``fit_tvc_timeline``,
reusing the ordinary maximum-likelihood fit. A fitted PH, AH or AFT model can
then be *evaluated* along a piecewise-constant covariate path with ``sf_tvc`` /
``Hf_tvc``, describing the path as a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`. See
:ref:`tvc-parametric` in the how-to guide.


Proportional Hazards (PH)
--------------------------

.. math::

    h(x \mid Z) = h_0(x) \cdot \phi(Z)

Pre-built instances: ``ExponentialPH``, ``NormalPH``, ``WeibullPH``,
``GumbelPH``, ``LogisticPH``, ``LogNormalPH``, ``GammaPH``.

Factory::

    from surpyval import LogNormal
    from surpyval import PH
    model = PH(LogNormal).fit(x, Z=Z, c=c)

.. autoclass:: surpyval.univariate.regression.proportional_hazards.proportional_hazards_fitter.ProportionalHazardsFitter
    :members: fit, Hf, hf, sf, ff, df


Accelerated Failure Time (AFT)
--------------------------------

.. math::

    H(x \mid Z) = H_0\!\left(e^{\beta'Z} \cdot x\right)

Pre-built instances: ``ExponentialAFT``, ``NormalAFT``, ``WeibullAFT``,
``GumbelAFT``, ``LogisticAFT``, ``LogNormalAFT``, ``GammaAFT``.

Factory::

    from surpyval import Gamma
    from surpyval import AFT
    model = AFT(Gamma).fit(x, Z=Z, c=c)

.. autoclass:: surpyval.univariate.regression.accelerated_failure_time.aft_fitter.AFTFitter
    :members: fit, Hf, hf, sf, ff, df


Proportional Odds (PO)
-----------------------

.. math::

    \frac{S(x \mid Z)}{F(x \mid Z)} = \frac{S_0(x)}{F_0(x)} \cdot e^{\beta'Z}

Pre-built instances: ``ExponentialPO``, ``NormalPO``, ``WeibullPO``,
``GumbelPO``, ``LogisticPO``, ``LogNormalPO``, ``GammaPO``.

Factory::

    from surpyval import Logistic
    from surpyval import PO
    model = PO(Logistic).fit(x, Z=Z, c=c)

.. autoclass:: surpyval.univariate.regression.proportional_odds.proportional_odds_fitter.ProportionalOddsFitter
    :members: fit, Hf, hf, sf, ff, df


Accelerated Life (AL)
----------------------

AL models substitute the life parameter of a distribution with a
physics-motivated stress function. Designed for discrete, controlled stress
levels (e.g. temperature, voltage).

Factory::

    from surpyval import Weibull
    from surpyval import AcceleratedLife, Power, Eyring
    model = AcceleratedLife(Weibull, Power).fit(x, Z=stress, c=c)

Available life models: ``Power``, ``InversePower``, ``Eyring``,
``InverseEyring``, ``ExponentialLifeModel``, ``InverseExponential``,
``Linear``, ``DualExponential``, ``DualPower``, ``PowerExponential``.

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
    :members: fit

.. autoclass:: surpyval.univariate.regression.accelerated_life.lifemodel.LifeModel
    :members:


Time-varying covariate schedules
--------------------------------

A ``StepSchedule`` describes a piecewise-constant covariate path ``Z(t)`` for
``sf_tvc`` / ``Hf_tvc`` evaluation. Build one structurally
(``from_changepoints`` / ``from_intervals`` / ``cyclic`` / ``constant``) or from
a step-valued expression string in ``t`` (``from_expression``); see
:ref:`tvc-parametric` for worked examples.

.. autoclass:: surpyval.univariate.regression.tvc_schedule.StepSchedule
    :members:
