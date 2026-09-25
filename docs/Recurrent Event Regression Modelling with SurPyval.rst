Recurrent Event Regression Modelling with SurPyval
===================================================

Modelling recurrent events where we have covariates is simple with surpyval.
Using the same API as regular survival regression, all we need to do is select
a model and fit it to our data. The two proportional-intensity models are
``ProportionalIntensityHPP`` (a constant baseline rate) and
``ProportionalIntensityNHPP`` (a time-varying baseline, e.g. Duane or
Crow-AMSAA). Both scale a baseline intensity by a covariate factor
:math:`e^{Z\beta}`, so a coefficient :math:`\beta_k` multiplies the event rate
by :math:`e^{\beta_k}` for each unit of covariate :math:`k`. The theory,
assumptions and references are on the :doc:`Recurrent Event Regression
Analysis` page; for the models without covariates see
:doc:`Recurrent Event Modelling with SurPyval`.

Both fitters take the recurrent ``x`` / ``i`` / ``c`` / ``n`` arrays exactly as
the models without covariates do (including ``tl`` / ``tr`` truncation and
interval-counted rows), plus the covariates ``Z`` as the **second** argument:
``fit(x, Z, i=..., c=...)``. ``Z`` is either a 2-D array with one row per row
of ``x`` (repeat an item's covariates on each of its rows; a 1-D array is one
covariate), or a dictionary mapping each item id to its covariate values (a
list, or a plain number for a single covariate). The covariates describe
the item and should be constant within it.

Proportional-Intensity HPP
--------------------------

The data are the usual recurrent ``x`` / ``i`` / ``c`` arrays plus a covariate
matrix ``Z`` with one row per observation. Here three items are each observed
until a right-censoring row, and each carries a single covariate (say, a duty
cycle) that is constant for the item.

.. jupyter-execute::

    from surpyval.recurrent import ProportionalIntensityHPP
    import numpy as np

    x = np.array([5.0, 8.0, 6.0, 10.0, 7.0, 9.0])
    i = np.array([1, 1, 2, 2, 3, 3])
    c = np.array([0, 1, 0, 1, 0, 1])
    Z = np.array([[0.1], [0.1], [0.5], [0.5], [0.9], [0.9]])

    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    model

The fit reports the baseline rate parameter and a covariate coefficient. The
coefficient enters multiplicatively through :math:`e^{Z\beta}`, so it reads as
the log of the rate ratio: a one-unit increase in the covariate multiplies the
event rate by :math:`e^{\beta}`. (With three events this is only a
demonstration of the data format; the worked example below has enough data to
learn from.) The same fit with the covariates given per item:

.. jupyter-execute::

    Z_by_item = {1: [0.1], 2: [0.5], 3: [0.9]}
    ProportionalIntensityHPP.fit(x, Z_by_item, i=i, c=c).coeffs

Predicting at a covariate setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To predict the expected number of events for an item, pass its covariates to
``cif``. The instantaneous rate at a covariate setting comes from ``iif``:

.. jupyter-execute::

    Z_item = np.array([0.5])
    print("expected events by t=12 :", round(float(model.cif(12.0, Z_item)), 3))
    print("event rate at t=12      :", round(float(model.iif(12.0, Z_item)), 3))

Because the covariate factor multiplies the whole cumulative intensity, the
ratio of expected counts between two covariate settings is constant in time and
equals :math:`e^{(Z_2 - Z_1)\beta}`:

.. jupyter-execute::

    z1, z2 = np.array([0.2]), np.array([0.7])
    ratio = model.cif(10.0, z2) / model.cif(10.0, z1)
    print("count ratio  :", round(float(ratio), 3))
    print("exp((z2-z1)b):", round(float(np.exp((z2 - z1) @ model.coeffs)), 3))

A worked example with known effects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see what the models can recover, let's simulate a fleet of 30 electric
motors whose failure rate depends on two covariates: whether the motor works
in a humid environment (0 or 1) and its duty cycle (between 0 and 1). The true
model is a proportional-intensity power law,

.. math::

    \Lambda(t \mid Z) = \left(\frac{t}{25}\right)^{1.5}
    e^{0.7\,\text{humid} + 1.0\,\text{duty}},

so the motors wear out (the shape 1.5 is above one), a humid environment
multiplies the failure rate by :math:`e^{0.7} \approx 2`, and full duty
multiplies it by :math:`e^{1.0} \approx 2.7`. The motors entered service at
different times, so each is observed for between 30 and 60 months. For a
Poisson process the simulation is simple: draw the number of failures from a
Poisson distribution with mean :math:`\Lambda(T \mid Z)`, then place them
independently with distribution function :math:`\Lambda(t)/\Lambda(T)`,
which for a power law means :math:`t = T\,U^{1/1.5}` for uniform :math:`U`:

.. jupyter-execute::

    rng = np.random.default_rng(1)
    x, i, c, Z = [], [], [], []
    for motor in range(1, 31):
        T = rng.uniform(30, 60)                               # months observed
        z = np.array([rng.integers(0, 2), rng.uniform(0, 1)])  # humid, duty
        expected = (T / 25) ** 1.5 * np.exp(z @ [0.7, 1.0])
        k = rng.poisson(expected)
        times = np.sort(T * rng.uniform(size=k) ** (1 / 1.5))
        x += [*times, T]                     # the failures, then the censoring row
        i += [motor] * (k + 1)
        c += [0] * k + [1]
        Z += [z] * (k + 1)                   # the motor's covariates on every row
    x, i, c, Z = map(np.array, (x, i, c, Z))
    print(f"{(c == 0).sum()} failures on {len(np.unique(i))} motors")

First a constant-rate model:

.. jupyter-execute::

    fleet_hpp = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    fleet_hpp

Rate ratios and their uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitted model carries the usual likelihood inference. ``parameter_names``
lists the baseline parameters followed by the coefficients, ``standard_errors``
gives a standard error for each, and ``param_cb`` a confidence interval.
Exponentiating a coefficient, and the ends of its interval, gives the rate
ratio and its confidence interval:

.. jupyter-execute::

    print("parameters :", fleet_hpp.parameter_names)
    print("std errors :", fleet_hpp.standard_errors().round(3))
    for k, name in enumerate(["humid", "duty"]):
        lower, upper = np.exp(fleet_hpp.param_cb(f"beta_{k}"))
        print(f"{name:6s} rate ratio {np.exp(fleet_hpp.coeffs[k]):.2f}"
              f"  (95% CI {lower:.2f} to {upper:.2f})")

Both intervals contain the true rate ratios (about 2.0 and 2.7). The
coefficients are estimated reasonably well even by this constant-rate model;
but, as the next section shows, the HPP is the wrong model for these motors.

Proportional-Intensity NHPP
---------------------------

When the baseline rate itself varies with time — reliability growth, wear-out —
use ``ProportionalIntensityNHPP`` with a counting-process baseline. The default
baseline is the Duane model; any NHPP baseline (``CrowAMSAA``, ``CoxLewis``)
can be supplied via ``dist``.

.. jupyter-execute::

    from surpyval.recurrent import ProportionalIntensityNHPP, Duane
    import numpy as np

    x_toy = np.array([2.0, 5.0, 3.0, 7.0, 1.0, 4.0, 2.0, 6.0])
    i_toy = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    c_toy = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    Z_toy = np.array([[0.2], [0.2], [0.5], [0.5], [0.8], [0.8], [0.3], [0.3]])

    toy = ProportionalIntensityNHPP.fit(x_toy, Z_toy, i=i_toy, c=c_toy, dist=Duane)
    toy

Confidence bounds on the fitted cumulative intensity at a covariate setting are
available from ``cif_cb`` (delta method, computed on the log scale so they stay
positive), as ``[lower, upper]`` columns:

.. jupyter-execute::

    ts = np.array([2.0, 4.0, 6.0])
    toy.cif_cb(ts, np.array([0.5]))

Choosing the baseline
~~~~~~~~~~~~~~~~~~~~~

Back to the motors. A power-law baseline can capture their wear-out, and the
information criteria show how much better it describes the data than the
constant rate:

.. jupyter-execute::

    from surpyval.recurrent import CrowAMSAA

    fleet = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    fleet

.. jupyter-execute::

    print("PI-HPP  AIC:", round(fleet_hpp.aic, 2))
    print("PI-NHPP AIC:", round(fleet.aic, 2))
    print("rate ratios:", np.exp(fleet.coeffs).round(2))

The Crow-AMSAA baseline recovers the wear-out shape (about 1.5) and scale
(about 23, against the true 25), and the rate ratios, about 1.9 and 2.3, are
within the precision that thirty motors allow of the true 2.0 and 2.7.

.. note::

    The default ``Duane`` baseline describes the same power-law model in a
    different parameterisation, so it reaches the same maximum likelihood.
    Its scale parameter ``b`` is usually a very small number, which a fixed
    starting point would sit orders of magnitude away from; the fit
    therefore starts from the baseline fitted *without* covariates, with
    every coefficient at zero. Pass ``init`` (baseline parameters followed
    by one value per coefficient) to start somewhere else.

.. jupyter-execute::

    duane = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c)

    print("Duane      AIC:", round(duane.aic, 2))
    print("Crow-AMSAA AIC:", round(fleet.aic, 2))

Prediction and simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Every prediction takes the covariates of the item being predicted as its
second argument: ``cif`` (the expected number of events), ``iif`` (the event
rate), ``inv_cif`` (the time by which a given number of events is expected)
and ``cif_cb`` (confidence bounds on the expected number). For a humid motor
at half duty:

.. jupyter-execute::

    z = np.array([1.0, 0.5])
    print("expected failures by 40 months :", fleet.cif(40, z).round(2))
    print("failure rate at 40 months      :", fleet.iif(40, z).round(3))
    print("time of the 5th failure        :", fleet.inv_cif(5, z).round(1))
    print("95% bounds on the expected count at 20 and 40 months:")
    print(fleet.cif_cb([20, 40], z).round(2))

``plot`` draws the fitted cumulative intensity, with its confidence band, at
the *average* of the covariate rows, over the non-parametric MCF of all the
data (which ignores the covariates):

.. jupyter-execute::

    fleet.plot()

The model can also simulate items with given covariates. ``mcf(x, Z)`` estimates
the MCF by simulation (which, for these Poisson models, reproduces ``cif`` up
to simulation noise), and ``time_terminated_simulation(T, Z, items)``,
``count_terminated_simulation(events, Z, items)`` and their ``..._data``
versions work as for the models without covariates:

.. jupyter-execute::

    print("simulated MCF   :", fleet.mcf([20, 40], z, items=500, seed=1))
    print("closed-form cif :", fleet.cif(np.array([20, 40]), z).round(3))

    sims = fleet.time_terminated_simulation_data(40, z, items=5, seed=2)
    print("failures on five simulated motors:",
          [int((sims.c[sims.i == k] == 0).sum()) for k in range(1, 6)])

As for any Poisson process, the actual number of failures in a future period
is Poisson distributed about the expected number, so a plug-in prediction
interval for one motor's failures between 40 and 50 months follows from the
fitted ``cif``:

.. jupyter-execute::

    from scipy.stats import poisson

    expected = float(fleet.cif(50, z) - fleet.cif(40, z))
    print(f"expected {expected:.2f}; 90% prediction interval",
          poisson.ppf([0.05, 0.95], expected))

Saving and loading a regression model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like the other recurrence models, a fitted proportional-intensity model can be
saved with ``to_dict`` / ``to_json`` and restored with ``surpyval.from_dict``
/ ``surpyval.from_json``. The baseline, its parameters and the coefficients
are stored, so predictions are reproduced exactly; the fitted data and
likelihood are not, so re-fit if you need confidence bounds or diagnostics.

.. jupyter-execute::

    import surpyval

    restored = surpyval.from_dict(fleet.to_dict())
    print(np.allclose(restored.cif(40, z), fleet.cif(40, z)))

Model checking
--------------

The regression models carry the same diagnostics as the unconditional
intensity models, applied per item with each item's intensity scaled by its
covariate factor:

.. jupyter-execute::

    print("residual mean :", round(fleet.residuals().mean(), 3))
    print("trend         :", fleet.trend_test().trend)

The residuals are the time-rescaling residuals pooled across items (i.i.d.
Exp(1) under a well-specified model); the trend test checks whether a
time-varying baseline was warranted at all. Here the residual mean is below
one although the model is correct: each motor's final, censored gap is left
out of the residuals, and with about six failures per motor that selection is
noticeable (see :doc:`Recurrent Event Regression Analysis`). The trend test,
unsurprisingly, points to an increasing rate:

.. jupyter-execute::

    fleet.trend_test()

The martingale residuals — observed minus expected failures, one per motor —
check the covariate model. If the effect of duty cycle were not log-linear,
their average would drift with duty cycle; here there is no obvious pattern:

.. jupyter-execute::

    martingale = fleet.residuals(kind="martingale")
    duty = np.array([Z[i == k][0, 1] for k in np.unique(i)])
    for label, mask in [("duty < 0.5 ", duty < 0.5), ("duty >= 0.5", duty >= 0.5)]:
        print(label, "mean martingale residual:", martingale[mask].mean().round(2))

A Cramér–von Mises goodness-of-fit test is also available via
``cramer_von_mises``. It is a parametric bootstrap that refits the whole
regression for each replicate, so keep ``n_boot`` modest while exploring. It
tells the two models of the motors apart:

.. jupyter-execute::

    print("PI-HPP  p-value:", round(fleet_hpp.cramer_von_mises(n_boot=30, seed=1).p_value, 3))
    print("PI-NHPP p-value:", round(fleet.cramer_von_mises(n_boot=30, seed=1).p_value, 3))

The constant-rate model is rejected at the 5% level, while the power-law model
is consistent with the data.

See :doc:`Recurrent Event Regression Analysis` for the theory and references
behind these models.
