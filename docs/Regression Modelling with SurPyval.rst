
Regression Modelling with SurPyval
==================================

Regression analysis lets us capture the effect of covariates on survival times.
For the rest of this page we assume the following imports:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt


SurPyval supports four families of regression model, each with pre-built
convenience instances for every standard distribution, plus factory functions
for custom combinations:

+-------------------+--------------------------------------+----------------------------------------+
| Family            | Effect of covariates                 | Ready-to-use examples                  |
+===================+======================================+========================================+
| Proportional      | Multiplies the hazard rate           | ``WeibullPH``, ``ExponentialPH``, …    |
| Hazards (PH)      | :math:`h(x|Z) = h_0(x)\,\phi(Z)`    |                                        |
+-------------------+--------------------------------------+----------------------------------------+
| Accelerated       | Scales the time axis                 | ``WeibullAFT``, ``LogNormalAFT``, …    |
| Failure Time      | :math:`H(x|Z) = H_0(x\,\phi(Z))`   |                                        |
| (AFT)             |                                      |                                        |
+-------------------+--------------------------------------+----------------------------------------+
| Proportional      | Scales the survival odds             | ``WeibullPO``, ``LogisticPO``, …       |
| Odds (PO)         | :math:`O(x|Z) = O_0(x)\,\phi(Z)`    |                                        |
+-------------------+--------------------------------------+----------------------------------------+
| Accelerated Life  | Substitutes the life parameter       | ``AcceleratedLife(Weibull, Power)``,   |
| (AL)              | with a physics-motivated function    | ``AcceleratedLife(Weibull, Eyring)``,… |
+-------------------+--------------------------------------+----------------------------------------+

For PH, AFT, and PO the covariate function is always the log-linear form:

.. math::

    \phi(Z) = e^{\beta_1 z_1 + \beta_2 z_2 + \cdots}

For AL models the function is a domain-specific stress relationship (Arrhenius,
Eyring, Power Law, etc.) — see the :ref:`accelerated-life` section below.


Semi-Parametric — Cox Proportional Hazards
------------------------------------------

The Cox PH model is the most widely used survival regression model. It estimates
β without assuming any parametric form for the baseline hazard.

.. jupyter-execute::

    from surpyval.datasets import load_tires_data
    from surpyval import CoxPH

    tires = load_tires_data()
    x = tires['Survival']
    c = tires['Censoring']
    Z = tires[['Wedge gauge', 'Interbelt gauge', 'Peel force',
               'Wedge gauge×peel force']]
    model = CoxPH.fit(x=x, Z=Z, c=c)
    model

Check which coefficients are significant:

.. jupyter-execute::

    print(model.p_values)

Plot survival curves at the mean covariate values, ±10 %:

.. jupyter-execute::

    Z_mean = Z.mean().values
    plot_x = np.linspace(x.min(), x.max())
    for f in [0.9, 1.0, 1.1]:
        plt.step(plot_x, model.sf(plot_x, Z=Z_mean * f), label=f'{f:.0%}')
    plt.legend(title='Covariate scale')
    plt.xlabel('Survival time')
    plt.ylabel('S(x)')


Parametric Proportional Hazards (PH)
-------------------------------------

Parametric PH models assume a fully parametric baseline — allowing extrapolation
beyond the observed time range. Every standard distribution has a pre-built
instance; all accept any number of covariates.

Using a pre-built instance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import WeibullPH

    model = WeibullPH.fit(x=x, Z=Z, c=c)
    model

Using the factory for any distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import LogNormal
    from surpyval.regression import PH

    model = PH(LogNormal).fit(x=x, Z=Z, c=c)
    model


Accelerated Failure Time (AFT)
--------------------------------

An AFT model accelerates (or decelerates) the time axis by :math:`\phi(Z)`.
A positive β coefficient shortens life (higher stress), negative extends it.

.. math::

    H(x \mid Z) = H_0\!\left(e^{\beta'Z} \cdot x\right)

Using a pre-built instance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import WeibullAFT

    model = WeibullAFT.fit(x=x, Z=Z, c=c)
    model

Using the factory for any distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import Gamma
    from surpyval.regression import AFT

    model = AFT(Gamma).fit(x=x, Z=Z, c=c)
    model

Comparing survival curves across covariate values:

.. jupyter-execute::

    from surpyval import WeibullAFT

    model = WeibullAFT.fit(x=x, Z=Z, c=c)
    Z_mean = Z.mean().values
    plot_x = np.linspace(x.min(), x.max())
    for f in [0.9, 1.0, 1.1]:
        plt.plot(plot_x, model.sf(plot_x, Z=Z_mean * f), label=f'{f:.0%}')
    plt.legend(title='Covariate scale')
    plt.xlabel('Survival time')
    plt.ylabel('S(x)')


Proportional Odds (PO)
-----------------------

A PO model scales the baseline survival odds by :math:`\phi(Z)`. The covariate
effect attenuates at long follow-up times, making PO more realistic than PH
in many settings. It is the natural companion to the Logistic and Log-Logistic
distributions.

.. math::

    \frac{S(x \mid Z)}{F(x \mid Z)} = \frac{S_0(x)}{F_0(x)} \cdot e^{\beta'Z}

Using a pre-built instance
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import LogisticPO

    model = LogisticPO.fit(x=x, Z=Z, c=c)
    model

Using the factory for any distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval.regression import PO

    model = PO(Weibull).fit(x=x, Z=Z, c=c)
    model


.. _accelerated-life:

Accelerated Life (AL)
----------------------

AL models substitute the *life parameter* of a distribution with a
physics-motivated stress function. They are the standard approach in
accelerated life testing (ALT) where stress levels are discrete and
controlled (e.g. temperature, voltage).

Available life models
~~~~~~~~~~~~~~~~~~~~~~

+------------------------+----------------------------------------+
| Life model             | Formula :math:`\phi(Z)`                |
+========================+========================================+
| ``Power``              | :math:`a \cdot Z^n`                    |
+------------------------+----------------------------------------+
| ``InversePower``       | :math:`1 / (a \cdot Z^n)`              |
+------------------------+----------------------------------------+
| ``Eyring``             | :math:`Z^{-1} e^{-(c - a/Z)}`         |
+------------------------+----------------------------------------+
| ``InverseEyring``      | Reciprocal of Eyring                   |
+------------------------+----------------------------------------+
| ``ExponentialLifeModel``| :math:`b \cdot e^{a/Z}` (Arrhenius)  |
+------------------------+----------------------------------------+
| ``InverseExponential`` | Reciprocal of Arrhenius                |
+------------------------+----------------------------------------+
| ``Linear``             | :math:`a + b \cdot Z`                  |
+------------------------+----------------------------------------+
| ``DualExponential``    | :math:`c \cdot e^{a/Z_1} e^{b/Z_2}`  |
+------------------------+----------------------------------------+
| ``DualPower``          | :math:`c \cdot Z_1^m Z_2^n`           |
+------------------------+----------------------------------------+
| ``PowerExponential``   | :math:`c \cdot e^{a/Z_1} Z_2^n`      |
+------------------------+----------------------------------------+

Using the factory
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval.regression import AcceleratedLife, Power, Eyring

    # Discrete stress levels (required for AL models)
    stress = np.array([100]*20 + [150]*20 + [200]*20, dtype=float)
    np.random.seed(42)
    # Simulate failure times decreasing with stress
    scale = 1000.0 / stress
    x_al = np.random.weibull(2, 60) * scale * 100
    c_al = np.ones(60, dtype=int)
    c_al[np.random.choice(60, 10, replace=False)] = 1  # some right-censored

    Z_al = stress.reshape(-1, 1)

    model = AcceleratedLife(Weibull, Power).fit(x_al, Z=Z_al)
    model

.. jupyter-execute::

    # Eyring model — common for temperature-accelerated testing
    model_eyring = AcceleratedLife(Weibull, Eyring).fit(x_al, Z=Z_al)
    model_eyring

Creating a custom life model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can pass any callable as the life model by subclassing ``LifeModel``.
For example, a simple log-linear single-stress relationship:

.. jupyter-execute::

    from surpyval.regression import LifeModel, AcceleratedLife, ParameterSubstitutionFitter
    from surpyval import Weibull
    import autograd.numpy as anp

    class LogLinearSingle(LifeModel):
        def __init__(self):
            super().__init__(
                name="LogLinearSingle",
                phi_param_map={"a": 0, "b": 1},
                phi_bounds=((None, None), (None, None)),
            )

        def phi(self, Z, *params):
            a, b = params
            return anp.exp(a + b * Z)

        def phi_init(self, life, Z):
            Z = Z.flatten()
            b, a = anp.polyfit(Z, anp.log(life), 1)
            return [float(a), float(b)]

    model_custom = AcceleratedLife(Weibull, LogLinearSingle()).fit(x_al, Z=Z_al)
    model_custom


Model Selection
---------------

All parametric regression models expose AIC and BIC for comparison:

.. jupyter-execute::

    from surpyval import WeibullAFT, WeibullPH
    from surpyval.regression import PO
    from surpyval import Weibull

    models = {
        'WeibullPH':  WeibullPH.fit(x=x, Z=Z, c=c),
        'WeibullAFT': WeibullAFT.fit(x=x, Z=Z, c=c),
        'WeibullPO':  PO(Weibull).fit(x=x, Z=Z, c=c),
    }

    for name, m in models.items():
        print(f'{name:12s}  AIC={m.aic():.2f}  BIC={m.bic():.2f}')
