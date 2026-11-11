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
the item and must be constant within it: values that change between an
item's rows raise a ``ValueError`` (for a load, an environment or an
overhaul state that changes during an item's life, see
`Covariates that change over time`_ below). Both fitters also take ``init``,
the starting values for the search: the baseline parameters on their natural
scale followed by one value per coefficient. Gapped observation
(``windows``) is not available with covariates. A covariate column the data
cannot determine -- a linear combination of the others, or a constant column,
whose effect the baseline's scale already carries (the HPP's rate, Duane's
``b``, Crow-AMSAA's and Cox-Lewis's ``alpha``) -- is aliased, as in every
SurPyval regression: its coefficient is ``nan`` and listed in
``model.aliased``, one warning names it, and the other parameters and the
predictions are those of the fit without it.

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
    :hide-code:
    :hide-output:

    assert (c == 0).sum() == 3

.. jupyter-execute::

    Z_by_item = {1: [0.1], 2: [0.5], 3: [0.9]}
    ProportionalIntensityHPP.fit(x, Z_by_item, i=i, c=c).coeffs

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(ProportionalIntensityHPP.fit(x, Z_by_item, i=i, c=c)
                       .coeffs, model.coeffs)

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

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.isclose(float(ratio), float(np.exp((z2 - z1) @ model.coeffs)))

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
        lower, upper = np.exp(fleet_hpp.param_cb(f"coef_{k}"))
        print(f"{name:6s} rate ratio {np.exp(fleet_hpp.coeffs[k]):.2f}"
              f"  (95% CI {lower:.2f} to {upper:.2f})")

Both intervals contain the true rate ratios (about 2.0 and 2.7). The
coefficients are estimated reasonably well even by this constant-rate model;
but, as the next section shows, the HPP is the wrong model for these motors.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    for _k, _true in enumerate([np.exp(0.7), np.exp(1.0)]):
        _lo, _hi = np.exp(fleet_hpp.param_cb(f"coef_{_k}"))
        assert _lo < _true < _hi, (_k, _lo, _hi)

Proportional-Intensity NHPP
---------------------------

When the baseline rate itself varies with time — reliability growth, wear-out —
use ``ProportionalIntensityNHPP`` with a counting-process baseline. The default
baseline is the Duane model; any NHPP baseline (``CrowAMSAA``, ``CoxLewis``)
can be supplied via ``baseline``.

.. jupyter-execute::

    from surpyval.recurrent import ProportionalIntensityNHPP, Duane
    import numpy as np

    x_toy = np.array([2.0, 5.0, 3.0, 7.0, 1.0, 4.0, 2.0, 6.0])
    i_toy = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    c_toy = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    Z_toy = np.array([[0.2], [0.2], [0.5], [0.5], [0.8], [0.8], [0.3], [0.3]])

    toy = ProportionalIntensityNHPP.fit(x_toy, Z_toy, i=i_toy, c=c_toy, baseline=Duane)
    toy

Confidence bounds on the fitted cumulative intensity at a covariate setting are
available from ``cif_cb`` (delta method, computed on the log scale so they stay
positive), as ``[lower, upper]`` columns. As for the models without
covariates, the level is set by the total tail probability ``alpha_ci``
(default 0.05) and ``bound="lower"``/``"upper"`` gives one-sided bounds:

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

    fleet = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, baseline=CrowAMSAA)
    fleet

.. jupyter-execute::

    print("PI-HPP  AIC:", round(fleet_hpp.aic(), 2))
    print("PI-NHPP AIC:", round(fleet.aic(), 2))
    print("rate ratios:", np.exp(fleet.coeffs).round(2))

The Crow-AMSAA baseline recovers the wear-out shape (about 1.5) and scale
(about 23, against the true 25), and the rate ratios, about 1.9 and 2.3, are
within the precision that thirty motors allow of the true 2.0 and 2.7.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert fleet.aic() < fleet_hpp.aic()
    _alpha, _beta = fleet.params          # Crow-AMSAA scale and shape
    assert round(_beta, 1) == 1.5 and round(_alpha) == 23, fleet.params
    assert np.exp(fleet.coeffs).round(1).tolist() == [1.9, 2.3]

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

    print("Duane      AIC:", round(duane.aic(), 2))
    print("Crow-AMSAA AIC:", round(fleet.aic(), 2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.isclose(duane.aic(), fleet.aic(), atol=0.01)

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
the *average* of the covariate rows (an average over rows, so items with more
events weigh more), over the non-parametric MCF of all the data (which ignores
the covariates). It is a rough visual check only: the curve at the average
covariates is not the average of the items' curves. ``plot_bounds`` and
``alpha_ci`` work as for the other models:

.. jupyter-execute::

    fleet.plot()

The model can also simulate items with given covariates. ``mcf(x, Z)`` estimates
the MCF by simulation (which, for these Poisson models, reproduces ``cif`` up
to simulation noise), and ``time_terminated_simulation(T, Z, items)``,
``count_terminated_simulation(events, Z, items)`` and their ``..._data``
versions work as for the models without covariates. Their ``Z`` is one item's
covariate vector, so a missing (``nan``) value in it raises a ``ValueError``
(see :ref:`missing-values`):

.. jupyter-execute::

    print("simulated MCF   :", fleet.mcf([20, 40], z, items=500, random_state=1))
    print("closed-form cif :", fleet.cif(np.array([20, 40]), z).round(3))

    sims = fleet.time_terminated_simulation_data(40, z, items=5, random_state=2)
    print("failures on five simulated motors:",
          [int((sims.c[sims.i == k] == 0).sum()) for k in range(1, 6)])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _sim = fleet.mcf([20, 40], z, items=500, random_state=1)
    _cif = fleet.cif(np.array([20, 40]), z)
    assert np.all(np.abs(_sim / _cif - 1) < 0.05), (_sim, _cif)

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
    :hide-code:
    :hide-output:

    assert fleet.residuals().mean() < 1
    assert round((c == 0).sum() / len(np.unique(i))) == 6
    assert fleet.trend_test().trend == "increasing"

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

    p_hpp = fleet_hpp.cramer_von_mises(n_boot=30, random_state=1).p_value
    p_nhpp = fleet.cramer_von_mises(n_boot=30, random_state=1).p_value
    print("PI-HPP  p-value:", round(p_hpp, 3))
    print("PI-NHPP p-value:", round(p_nhpp, 3))

The constant-rate model is rejected at the 5% level, while the power-law model
is consistent with the data.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert p_hpp < 0.05 < p_nhpp, (p_hpp, p_nhpp)

Covariates that change over time
--------------------------------

Everything above describes an item by covariates that never change: this
motor is in a humid room, that one runs at half duty. Many of the things
that drive failures are not like that. A pump runs at high load in some
months and low load in others. A machine is overhauled part of the way
through its life. A vehicle moves from a sealed road to a gravel one. The
covariate belongs to a *period* of the item's life, not to the item.

There are two tempting shortcuts, and both give the wrong answer:

- **Use the first (or the last) value.** A pump that started on low load and
  spent most of its life on high load is then filed under "low load", and
  the failures that the high load caused are blamed on low load. The effect
  of the covariate is diluted towards zero, or even reversed.
- **Use the average.** A pump that was on high load for half of its life is
  given a load of 0.5. That throws away *when* the failures happened. If
  eight of its ten failures fell in the high-load months, that is strong
  evidence about load; the average cannot see it. An average is also the
  wrong arithmetic: the model multiplies the failure rate by a factor, and
  the average of a factor over time is not the factor at the average. And
  when the failure rate itself changes with age, a spell of high load late
  in life costs more failures than the same spell early in life, which no
  single number per pump can express.

The model for a changing covariate is a small step from the one above. It
assumes that **the value the covariate has now multiplies the failure rate
now**: while the pump is on high load its failure rate is the baseline rate
at its present age times :math:`e^{\beta}`, and the moment it returns to low
load the rate returns to the baseline. There is no memory: the model does
not carry damage from past high-load spells into later low-load ones. (If
you believe that past exposure matters, put it in the data as a covariate
of its own, such as the cumulative hours at high load so far.) The
covariates must be *piecewise constant*: each holds one value over an
interval of time and then steps to another. The theory is on the
:doc:`Recurrent Event Regression Analysis` page.

Fitting uses the same names and data shapes as the time-varying fits of the
life models (``fit_tvc``, ``fit_tvc_timeline`` and their ``_from_df``
forms; see :doc:`Regression Modelling with SurPyval`), so what you know
from one carries over to the other.

The data: a log of what happened and when
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To have a known answer to recover, we simulate 40 pumps, each watched for
three to five years. Two things change over a pump's life. Its load
switches between low and high a few times, and high load multiplies the
failure rate by :math:`e^{0.8} \approx 2.2`. Most pumps are overhauled
once, somewhere between months 12 and 48, and from then on fail at
:math:`e^{-0.6} \approx 0.55` of the rate they otherwise would. Underneath,
every pump wears out: with both covariates at zero the expected number of
failures by month :math:`t` is :math:`(t / 20)^{1.4}`.

The natural record of such a history is a *timeline*: one row each time
something happens to a pump, saying when it happened, whether it was a
failure (``c = 0``) or not (``c = 1``), and what the covariates are *from
that moment on*:

.. jupyter-execute::

    import pandas as pd

    rng = np.random.default_rng(1)
    true_coefficients = np.array([0.8, -0.6])       # high load, overhauled

    def baseline(t):                 # expected failures by month t at Z = 0
        return (t / 20.0) ** 1.4

    def baseline_inverse(m):
        return 20.0 * m ** (1 / 1.4)

    rows = []
    for pump in range(1, 41):
        end = rng.uniform(36, 60)                   # months watched
        switches = rng.uniform(0, end, rng.integers(1, 5))
        overhaul = rng.uniform(12, 48) if rng.uniform() < 0.8 else np.inf
        edges = np.unique(np.r_[0.0, switches, min(overhaul, end), end])
        high = bool(rng.integers(0, 2))             # the load it starts on
        for a, b in zip(edges[:-1], edges[1:]):
            z = np.array([float(high), float(a >= overhaul)])
            expected = (baseline(b) - baseline(a)) * np.exp(z @ true_coefficients)
            fails = baseline_inverse(
                rng.uniform(baseline(a), baseline(b), rng.poisson(expected))
            )
            rows.append((pump, a, 1, *z))           # the covariates from a on
            rows += [(pump, t, 0, *z) for t in fails]
            if b in switches:
                high = not high                     # the load changes at b
        rows.append((pump, end, 1, *z))             # the end of the record
    pump_log = pd.DataFrame(
        rows, columns=["pump", "month", "c", "high_load", "overhauled"]
    ).sort_values(["pump", "month"], ignore_index=True)
    print(f"{(pump_log.c == 0).sum()} failures on {pump_log.pump.nunique()} pumps")
    pump_log[pump_log.pump == 1].round(2)

Read the first pump's rows from the top: its record starts at month 0 (the
first row of an item is always its entry, never a failure); each ``c = 0``
row is a failure; each other ``c = 1`` row is a change of load or the
overhaul; and the last row is the end of the record, whose covariate values
are not used (nothing follows it).

Fitting
~~~~~~~

``fit_tvc_timeline_from_df`` reads such a log directly. As with the fixed
covariates above, ``baseline`` picks the shared trend:

.. jupyter-execute::

    pumps = ProportionalIntensityNHPP.fit_tvc_timeline_from_df(
        pump_log,
        i_col="pump",
        x_col="month",
        Z_cols=["high_load", "overhauled"],
        c_col="c",
        baseline=CrowAMSAA,
    )
    pumps

.. jupyter-execute::

    pumps.summary().round(3)

The coefficients are named by their columns. High load is estimated to
multiply the failure rate by :math:`e^{0.86} \approx 2.4` and an overhaul
by :math:`e^{-0.51} \approx 0.60`; each interval contains the value the data
were simulated with (0.8 and -0.6), as do those of the baseline's scale
``alpha`` (20) and shape ``beta`` (1.4).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _summary = pumps.summary()
    assert _summary.index.tolist() == ["alpha", "beta", "high_load", "overhauled"]
    assert _summary["estimate"].round(2).tolist()[2:] == [0.86, -0.51], _summary
    for _name, _true in zip(_summary.index, [20.0, 1.4, 0.8, -0.6]):
        _lo, _hi = _summary.loc[_name, ["lower 95%", "upper 95%"]]
        assert _lo < _true < _hi, (_name, _lo, _hi)
    assert round(np.exp(0.86), 1) == 2.4 and round(np.exp(-0.51), 2) == 0.60

The other data shape is *start-stop* rows, the form ``fit_tvc`` and
``fit_tvc_from_df`` take: one row per interval ``(start, stop]`` of an
item's life, with the covariates that held throughout it and ``c = 0`` if a
failure ended it (``c = 1`` if it ended for any other reason). A timeline
becomes start-stop rows by pairing each row with the one before it:

.. jupyter-execute::

    before = pump_log.groupby("pump").shift()       # each pump's previous row
    intervals = pd.DataFrame({
        "pump": pump_log.pump,
        "start": before.month,
        "stop": pump_log.month,
        "c": pump_log.c,
        "high_load": before.high_load,              # what held over the interval
        "overhauled": before.overhauled,
    }).dropna(ignore_index=True)
    intervals[intervals.pump == 1].round(2)

.. jupyter-execute::

    same = ProportionalIntensityNHPP.fit_tvc_from_df(
        intervals,
        i_col="pump",
        xl_col="start",
        xr_col="stop",
        c_col="c",
        Z_cols=["high_load", "overhauled"],
        baseline=CrowAMSAA,
    )
    print(np.allclose(same.coeffs, pumps.coeffs))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(same.coeffs, pumps.coeffs)
    assert np.allclose(same.params, pumps.params)

A few rules of the start-stop form, all shared with the life models:

- An item's intervals must not overlap. They may leave gaps: the item is
  simply not counted as at risk during a gap (it was out of service, or
  nobody was recording). A first interval that starts after zero is delayed
  entry: the item is only watched from then on.
- Cutting an interval in two at any point, with the same covariates on both
  pieces, changes nothing. So there is no harm in cutting at every calendar
  month if that is how the data arrive.
- Where failures are only counted per period (so many failures between two
  inspections) rather than timed, give the period as one row with ``c = 2``
  and the count in ``n``. This is the same meaning ``c = 2`` and ``n`` have
  for the fixed-covariate fit.
- A missing covariate value is refused: the row is part of an item's
  history, and dropping it would silently remove the item from risk for
  that interval (see :ref:`missing-values`).

Left-censored counts (``c = -1``) and right truncation (``tr``) are not
part of the start-stop form: a count goes in as a ``c = 2`` row over its
interval, and a fit that needs ``tr`` is refused rather than approximated.

What the shortcuts would have said
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fixed-covariate fit needs one value per pump. Here are the two
shortcuts from the top of this section, on the same failures: each pump's
*first* values, and its *time-averaged* ones. Every pump starts out not yet
overhauled, so the first-value shortcut has no overhaul column to offer at
all, and is fitted on the load alone:

.. jupyter-execute::

    is_event_or_end = (pump_log.c == 0) | ~pump_log.pump.duplicated(keep="last")
    events = pump_log[is_event_or_end]

    first = pump_log.groupby("pump")[["high_load"]].first()
    months_at = intervals.stop - intervals.start
    average = (
        intervals[["high_load", "overhauled"]].mul(months_at, axis=0)
        .groupby(intervals.pump).sum()
        .div(months_at.groupby(intervals.pump).sum(), axis=0)
    )

    comparison = pd.DataFrame(
        {"truth": true_coefficients, "time-varying": pumps.coeffs},
        index=["high_load", "overhauled"],
    )
    for label, per_pump in [("first value", first), ("time average", average)]:
        shortcut = ProportionalIntensityNHPP.fit(
            events.month.values,
            per_pump.loc[events.pump],          # one row of values per pump
            i=events.pump.values,
            c=events.c.values,
            baseline=CrowAMSAA,
        )
        comparison[label] = pd.Series(shortcut.coeffs, index=per_pump.columns)
        comparison.loc["AIC", label] = shortcut.aic()
    comparison.loc["AIC", "time-varying"] = pumps.aic()
    comparison.round(2)

The first-value fit finds no effect of load at all (-0.05 against a true
0.8): which load a pump happened to start on says little about the load it
was on when it failed. It cannot say anything about the overhaul. The
time-average fit is closer, because it at least knows how long each pump
spent in each state, but it has thrown away *when* the failures fell, and
it shows: its estimates are further from the truth than the time-varying
fit's on both covariates, and its interval for the overhaul, printed below,
now includes zero, so it could not tell you the overhaul did anything. The
AIC row says the same from the data's side: the same failures are described
best by the model that follows the covariates through time.

.. jupyter-execute::

    print("time average :", shortcut.param_cb("overhauled").round(2))
    print("time-varying :", pumps.param_cb("overhauled").round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert comparison.loc["high_load", "first value"].round(2) == -0.05
    assert np.isnan(comparison.loc["overhauled", "first value"])
    _truth = comparison["truth"].iloc[:2]
    _off = comparison.iloc[:2].sub(_truth, axis=0).abs()
    assert (_off["time-varying"] < _off["time average"]).all(), _off
    _lo, _hi = shortcut.param_cb("overhauled")
    assert _lo < 0 < _hi, (_lo, _hi)
    assert pumps.param_cb("overhauled")[1] < 0
    _aic = comparison.loc["AIC"]
    assert _aic["time-varying"] < _aic["time average"] < _aic["first value"]

Predicting along a planned schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A prediction now needs to know the covariates over the whole period it
covers, not one value: a *schedule*. A
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` holds
one, exactly as for the life models. Suppose a new pump is planned to run on
low load for two years, on high load from month 24, and to be overhauled at
month 36:

.. jupyter-execute::

    from surpyval import StepSchedule

    plan = StepSchedule.from_changepoints(
        [0, 24, 36],                    # the month each row takes effect
        [[0, 0], [1, 0], [1, 1]],       # high_load, overhauled
    )
    months = np.array([12.0, 24.0, 36.0, 48.0])
    print("expected failures :", pumps.cif_tvc(months, plan).round(2))
    print("failure rate      :", pumps.iif_tvc(months, plan).round(3))

``cif_tvc`` is the expected number of failures from month 0 to each time
*if the pump follows the schedule*: it adds up, period by period, the
baseline's expected count over the period times the factor for the
covariates in force. ``iif_tvc`` is the failure rate (failures per month)
at each time, with the covariates in force at that time; at a time where
the schedule steps, it is the rate just after the step. Read the expected
count as you would any ``cif``: it is a mean. The actual number of failures
is Poisson distributed around it, as in the prediction interval above.

``cb_tvc`` gives the confidence bounds, on the expected count by default
and on the rate with ``on="iif"``. They are computed as ``cif_cb`` and
``iif_cb`` are (the delta method on the log scale), so a schedule that never
changes gives exactly the fixed-covariate bounds:

.. jupyter-execute::

    print(pumps.cb_tvc(months, plan).round(2))

    always_low = StepSchedule.constant([0, 0])
    print(np.allclose(pumps.cif_tvc(months, always_low), pumps.cif(months, [0, 0])))
    print(np.allclose(pumps.cb_tvc(months, always_low), pumps.cif_cb(months, [0, 0])))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(pumps.cif_tvc(months, always_low), pumps.cif(months, [0, 0]))
    assert np.allclose(pumps.cb_tvc(months, always_low), pumps.cif_cb(months, [0, 0]))
    assert np.allclose(
        pumps.cb_tvc(months, always_low, on="iif"), pumps.iif_cb(months, [0, 0])
    )

The usual question in service is about a pump that already has some age:
how many failures should we expect *from here on*? ``given`` counts from an
age instead of from zero, and then the schedule only needs to say what
happens from that age. For a pump that is 30 months old, on high load, and
due for its overhaul at month 36:

.. jupyter-execute::

    from_now = StepSchedule.from_changepoints([30, 36], [[1, 0], [1, 1]])
    no_overhaul = StepSchedule.from_changepoints([30], [[1, 0]])
    print("next 12 months, overhaul at 36 :",
          pumps.cif_tvc(42, from_now, given=30).round(2),
          pumps.cb_tvc(42, from_now, given=30).round(2))
    print("next 12 months, no overhaul    :",
          pumps.cif_tvc(42, no_overhaul, given=30).round(2),
          pumps.cb_tvc(42, no_overhaul, given=30).round(2))

Comparing schedules like this is how the model answers "what is the
overhaul worth?" for one pump's actual plan, not for an average pump. The
schedule can also be written as an array of covariate rows with ``xl``
giving the time each takes effect (``pumps.cif_tvc(42, [[1, 0], [1, 1]],
xl=[30, 36], given=30)``), or built with ``StepSchedule.cyclic`` for a
pattern that repeats, such as a high-load season each year.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.isclose(
        pumps.cif_tvc(42, [[1, 0], [1, 1]], xl=[30, 36], given=30),
        pumps.cif_tvc(42, from_now, given=30),
    )
    assert pumps.cif_tvc(42, from_now, given=30) < pumps.cif_tvc(
        42, no_overhaul, given=30
    )

What carries over, and what does not yet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A model fitted this way is an ordinary proportional-intensity model. The
fixed-covariate predictions (``cif``, ``iif``, ``inv_cif``, ``cif_cb``,
``iif_cb``) answer for an item held at one covariate setting for its whole
life; ``summary``, ``standard_errors``, ``param_cb``, ``covariance``,
``aic`` and ``bic`` work as before; simulation and ``mcf`` simulate an item
at one fixed covariate vector; and ``to_dict`` / ``from_dict`` keep every
prediction, the ones along a schedule included.

Some things are not available for a model fitted to changing covariates,
and each refuses with a message that says so, instead of quietly treating
the covariates as fixed:

- ``residuals``, ``cramer_von_mises``, ``trend_test``, ``diagnose`` and
  ``plot`` read each item's single covariate row, which such a fit does not
  have.
- Simulation along a schedule: the simulation methods and ``mcf`` take one
  covariate vector, and refuse a schedule. Use ``cif_tvc`` for the expected
  count along a schedule.
- A covariate that changes continuously (a
  :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`) is
  refused by the three ``_tvc`` methods; approximate it by steps.
- ``fit_penalised`` has no time-varying form, and the ``_from_df`` forms
  take ``Z_cols`` only (no ``formula``), as this model's ``fit_from_df``
  does.

See :doc:`Recurrent Event Regression Analysis` for the theory and references
behind these models.
