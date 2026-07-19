Recurrent Event Regression Modelling with SurPyval
===================================================

Modelling recurrent events where we have covariates is simple with surpyval.
Using the same API as regular survival regression, all we need to do is select
a model and fit it to our data. The two proportional-intensity models are
``ProportionalIntensityHPP`` (a constant baseline rate) and
``ProportionalIntensityNHPP`` (a time-varying baseline, e.g. Duane or
Crow-AMSAA). Both scale a baseline intensity by a covariate factor
:math:`e^{Z\beta}`.

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
event rate by :math:`e^{\beta}`.

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

Proportional-Intensity NHPP
---------------------------

When the baseline rate itself varies with time — reliability growth, wear-out —
use ``ProportionalIntensityNHPP`` with a counting-process baseline. The default
baseline is the Duane model; any NHPP baseline (``CrowAMSAA``, ``CoxLewis``)
can be supplied via ``dist``.

.. jupyter-execute::

    from surpyval.recurrent import ProportionalIntensityNHPP, Duane
    import numpy as np

    x = np.array([2.0, 5.0, 3.0, 7.0, 1.0, 4.0, 2.0, 6.0])
    i = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    c = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    Z = np.array([[0.2], [0.2], [0.5], [0.5], [0.8], [0.8], [0.3], [0.3]])

    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=Duane)
    model

Confidence bounds on the fitted cumulative intensity at a covariate setting are
available from ``cif_cb`` (delta method, computed on the log scale so they stay
positive):

.. jupyter-execute::

    ts = np.array([2.0, 4.0, 6.0])
    model.cif_cb(ts, np.array([0.5]))

Model checking
--------------

The regression models carry the same diagnostics as the unconditional
intensity models, applied per item with each item's intensity scaled by its
covariate factor:

.. jupyter-execute::

    print("residual mean :", round(model.residuals().mean(), 3))
    print("trend         :", model.trend_test().trend)

The residuals are the time-rescaling residuals pooled across items (i.i.d.
Exp(1) under a well-specified model); the trend test checks whether a
time-varying baseline was warranted at all. A Cramér–von Mises goodness-of-fit
test is also available via ``cramer_von_mises`` (a parametric bootstrap, so
keep ``n_boot`` modest while exploring).

See :doc:`Recurrent Event Regression Analysis` for the theory and references
behind these models.
