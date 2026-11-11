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
:doc:`Recurrent Event Modelling with SurPyval`. When the failures come in
several types (failure modes) and a covariate may act on one type and not
another, see `Several failure modes, each with its own covariate effects`_.
To estimate covariate effects without choosing a baseline shape at all (the
Andersen-Gill, Prentice-Williams-Peterson and Wei-Lin-Weissfeld models, with
robust standard errors), see `Without a baseline shape: Andersen-Gill,
Prentice-Williams-Peterson and Wei-Lin-Weissfeld`_ at the end of this page.

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

Several failure modes, each with its own covariate effects
-----------------------------------------------------------

A repairable item rarely fails in only one way. A pump has seal failures and
electrical trips; a vehicle has brake, tyre and engine repairs; a patient is
readmitted for different reasons. :doc:`Recurrent Event Modelling with
SurPyval` shows how ``CauseSpecificNHPP`` fits one failure-rate curve per
**failure mode** (event type). This section adds covariates to that picture.

The question it answers is: *which covariate drives which kind of failure?*
Abrasive fluid wears seals; there is no reason for it to trip the
motor's protection. A weak power supply causes trips and does nothing to the
seal. The
model gives every failure mode

- its own **baseline**: how often that kind of failure happens, and how that
  changes with age, on an item whose covariates are all zero; and
- its own **coefficients**: one per covariate, saying by what factor the
  covariate multiplies the rate of *that* kind of failure.

So "abrasive service triples seal failures and leaves the trips alone"
is something the model can say, and can say with confidence bounds on each
part.

**Reading a rate ratio per mode.** The exponential of a coefficient is a
*rate ratio*: the factor by which one more unit of the covariate multiplies
the rate of that mode's failures, at every age. A rate ratio of 3 for
``abrasive`` on the seal mode means that, of two pumps of the same age, the
one in abrasive service is having seal failures three times as often. A rate
ratio of 1 means the covariate does nothing to that mode; below 1, it
protects against it. Each mode has its own ratio for each covariate, with its
own interval. An interval that contains 1 says the data do not show an
effect on that mode. It does not say there is none: a mode with few failures
has wide intervals.

**Why not fit all the failures together?** Because a covariate that matters a
great deal for one mode can look unimportant, or vanish, in the total. If
seal failures are a small share of all failures, tripling them moves the
total rate only a little, and a model of all failures reports a small rate
ratio that may not even be distinguishable from 1. The conclusion "abrasive
service hardly matters" would be wrong where it counts: for the seals, for
the spare seals to hold, and for the decision whether a better seal is worth
buying. A covariate that raises one mode and lowers another can cancel
altogether. Fitting the modes apart keeps each effect at full strength, and
adding the modes back up (the *totals* below) still gives the answer for
failures of any kind.

**What the model assumes.** The failures of each mode arrive as a Poisson
process (a repair leaves the item as it was just before the failure, as in
the models above), and the modes do not interfere with each other: a seal
failure neither brings a trip forward nor postpones it. Under that
assumption the modes can be fitted one at a time, each to its own failures
over the whole time every item was watched, with the other modes' failures
simply left out. That is what ``CauseSpecificProportionalIntensity`` does:
each mode is an ordinary proportional-intensity fit (the models at the top
of this page), and the fitted models are in ``models[event]``. The formulas
are on the :doc:`Recurrent Event Regression Analysis` page.

The data
~~~~~~~~

To have a known answer, we simulate 60 pumps watched for 48 months. Half
pump an abrasive slurry. Seal failures are the rarer mode; they become more
frequent as a pump ages, and abrasive service triples them (a coefficient of
1.1). Electrical trips are the common nuisance: about one every four months,
at a constant rate, whatever the fluid (a coefficient of 0). The event log has one row
per failure, with its mode, and one row per pump for the end of its record
(``c = 1``, with no mode):

.. jupyter-execute::

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2026)
    months = 48.0
    # mode: (alpha, beta) of a power-law baseline, and the abrasive coefficient
    truth = {"seal": (70.0, 1.6, 1.1), "trip": (4.0, 1.0, 0.0)}

    rows = []
    for pump in range(60):
        abrasive = float(pump % 2)
        for mode, (alpha, beta, coef) in truth.items():
            expected = (months / alpha) ** beta * np.exp(coef * abrasive)
            times = months * rng.uniform(size=rng.poisson(expected)) ** (1 / beta)
            rows += [(pump, t, 0, mode, abrasive) for t in np.sort(times)]
        rows.append((pump, months, 1, None, abrasive))
    log = pd.DataFrame(rows, columns=["pump", "month", "c", "mode", "abrasive"])

    print(log.head(4))
    print(log.groupby(["mode", "abrasive"]).size().unstack())

The counts already tell the story: seal failures are several times as common
on the abrasive pumps, trips about the same on both.

Fitting, and the coefficient table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fit takes what the single-mode regressions take, plus the mode of each
failure: ``e`` in ``fit`` (``None`` on the end-of-record rows), ``e_col`` in
``fit_from_df``. ``baseline`` is the baseline of every mode: ``Duane`` (the
default), ``CrowAMSAA`` or ``CoxLewis``, or ``HPP`` for constant rates.

.. jupyter-execute::

    from surpyval.recurrent import CauseSpecificProportionalIntensity, CrowAMSAA

    modes = CauseSpecificProportionalIntensity.fit_from_df(
        log, x_col="month", Z_cols="abrasive", e_col="mode",
        i_col="pump", c_col="c", baseline=CrowAMSAA,
    )
    print(modes)
    table = modes.coefficients()
    print(table[["coef", "exp(coef)", "exp(coef) lower 95%",
                 "exp(coef) upper 95%", "p"]].round(3).to_string())

``coefficients()`` has one row per mode and covariate. ``exp(coef)`` is the
rate ratio, with its 95% interval beside it, and ``p`` is the p-value of the
test that the covariate does nothing to that mode. Abrasive service
multiplies the seal failure rate by about three, and the whole interval is
well above 1. For the trips the ratio is close to 1 and its interval
contains 1: no sign of an effect, which is the truth here.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lo = table["exp(coef) lower 95%"]
    _hi = table["exp(coef) upper 95%"]
    assert _lo["seal", "abrasive"] > 1.0
    assert _lo["seal", "abrasive"] < np.exp(1.1) < _hi["seal", "abrasive"]
    assert _lo["trip", "abrasive"] < 1.0 < _hi["trip", "abrasive"]

``summary()`` lists every fitted parameter of every mode, the baseline's
first, in the shape of ``CauseSpecificNHPP.summary``. The seal's ``beta`` is
above 1 (its failures become more frequent with age) and the trips' is near
1 (a constant rate):

.. jupyter-execute::

    print(modes.summary().round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _s = modes.summary()
    for _mode, (_alpha, _beta, _coef) in truth.items():
        for _name, _true in (("alpha", _alpha), ("beta", _beta),
                             ("abrasive", _coef)):
            _row = _s.loc[(_mode, _name)]
            assert _row["lower 95%"] < _true < _row["upper 95%"], (_mode, _name)
    assert np.allclose(
        modes.summary().loc["seal"].to_numpy(),
        modes.models["seal"].summary().to_numpy(),
    )

What fitting all the failures together would have said
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is the single-mode regression fitted to the same log with the modes
ignored, every failure counted alike:

.. jupyter-execute::

    from surpyval.recurrent import ProportionalIntensityNHPP

    together = ProportionalIntensityNHPP.fit_from_df(
        log, x_col="month", Z_cols="abrasive", i_col="pump", c_col="c",
        baseline=CrowAMSAA,
    )
    ratio = np.exp(together.coeffs[0])
    lower, upper = np.exp(together.param_cb("abrasive"))
    print(f"all failures together: rate ratio {ratio:.2f} "
          f"(95% CI {lower:.2f} to {upper:.2f})")
    seal = table.loc[("seal", "abrasive")]
    print(f"seal failures alone  : rate ratio {seal['exp(coef)']:.2f} "
          f"(95% CI {seal['exp(coef) lower 95%']:.2f} to "
          f"{seal['exp(coef) upper 95%']:.2f})")

Fitted together, abrasive service appears to do nothing: a rate ratio close
to 1, with an interval that contains 1. That is a fair description of
*failures of any kind*, because nine failures in ten are trips, which the
fluid does not affect. It is the wrong answer to "does abrasive service
matter?": it triples the seal failures, and the fit by mode shows it with an
interval that is nowhere near 1. The pooled fit also has one baseline for
everything, so it cannot show that seal failures grow with age while trips
do not.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert ratio < 0.6 * seal["exp(coef)"]
    assert lower < 1.0 < upper
    assert seal["exp(coef) lower 95%"] > 1.5

Predictions for each mode, with bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prediction methods are those of the single-mode model with the mode as
``event`` after the covariates: ``cif(x, Z, event)`` is the expected number
of failures of that mode by age ``x`` on an item with covariates ``Z``,
``iif(x, Z, event)`` the rate of that mode's failures at age ``x``, and
``cif_cb`` and ``iif_cb`` their confidence bounds. Each is the mode's own
model's, so ``modes.cif(x, Z, "seal")`` is ``modes.models["seal"].cif(x,
Z)``. The expected number of failures of each mode in four years, on clean
and on abrasive service:

.. jupyter-execute::

    for mode in modes.event_types:
        for label, Z in (("clean", [0]), ("abrasive", [1])):
            lo, hi = modes.cif_cb(months, Z, mode)
            print(f"{mode:5s} {label:8s}: {modes.cif(months, Z, mode):5.2f} "
                  f"(95% CI {lo:.2f} to {hi:.2f})")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(modes.cif(months, [1], "seal"),
                       modes.models["seal"].cif(months, [1]))
    assert np.allclose(modes.cif_cb(months, [1], "seal"),
                       modes.models["seal"].cif_cb(months, [1]))
    for _mode, (_alpha, _beta, _coef) in truth.items():
        for _z in (0.0, 1.0):
            _lo, _hi = modes.cif_cb(months, [_z], _mode, alpha_ci=0.01)
            _true = (months / _alpha) ** _beta * np.exp(_coef * _z)
            assert _lo < _true < _hi, (_mode, _z, _lo, _true, _hi)

All modes together: the totals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An owner also needs the answer for failures of any kind: how many call-outs
to expect, whatever broke. The modes are independent, so their expected
counts and their rates simply add: ``total_cif(x, Z)`` and ``total_iif(x,
Z)`` are the sums over the modes at a covariate row. Their uncertainties add
too. Each mode was fitted to its own failures, so the estimates are
independent and the variance of the sum is the sum of the modes' variances;
``total_cif_cb`` and ``total_iif_cb`` form the bounds from it on the log
scale, exactly as ``CauseSpecificNHPP`` does for its totals. A model with a
single mode therefore has total bounds equal to that mode's bounds.

.. jupyter-execute::

    for label, Z in (("clean", [0]), ("abrasive", [1])):
        lo, hi = modes.total_cif_cb(months, Z)
        print(f"{label:8s}: {modes.total_cif(months, Z):5.2f} failures of any "
              f"kind in four years (95% CI {lo:.2f} to {hi:.2f})")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(
        modes.total_cif(months, [1]),
        sum(modes.cif(months, [1], k) for k in modes.event_types),
    )
    assert np.allclose(
        modes.total_iif(months, [1]),
        sum(modes.iif(months, [1], k) for k in modes.event_types),
    )

Unlike the pooled fit above, the total here is built from the right model
for each mode: it knows that the extra failures on abrasive service are seal
failures, and that they grow with age.

Failure modes with covariates that change over time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a covariate belongs to a period of an item's life (see `Covariates that
change over time`_ above), the fits are ``fit_tvc`` (start-stop rows) and
``fit_tvc_timeline`` (a log of what happened and when), with their
``_from_df`` forms. They take exactly the rows the single-mode fits take,
plus the mode of each failure. Suppose the pumps were all on clean water at
first and half of them were switched to the slurry at month 20. A pump's
timeline then has a row for its entry, one for each failure, one for the
switch and one for the end of its record; ``abrasive`` on a row is the value
in force from that row on:

.. jupyter-execute::

    rng = np.random.default_rng(2027)
    switch = 20.0

    def failures(alpha, beta, factor, start, end):
        """Failure times of one mode on (start, end] at a constant factor."""
        low, high = (start / alpha) ** beta, (end / alpha) ** beta
        count = rng.poisson((high - low) * factor)
        return alpha * rng.uniform(low, high, size=count) ** (1 / beta)

    rows = []
    for pump in range(60):
        periods = [(0.0, months, 0.0)]
        if pump % 2:
            periods = [(0.0, switch, 0.0), (switch, months, 1.0)]
        for start, end, abrasive in periods:
            rows.append((pump, start, 1, None, abrasive))
            for mode, (alpha, beta, coef) in truth.items():
                rows += [
                    (pump, t, 0, mode, abrasive)
                    for t in failures(alpha, beta, np.exp(coef * abrasive),
                                      start, end)
                ]
        rows.append((pump, months, 1, None, abrasive))
    timeline = pd.DataFrame(
        rows, columns=["pump", "month", "c", "mode", "abrasive"]
    ).sort_values(["pump", "month"], kind="stable")

    switched = CauseSpecificProportionalIntensity.fit_tvc_timeline_from_df(
        timeline, "pump", "month", "abrasive", "c", "mode", baseline=CrowAMSAA
    )
    print(switched.coefficients()[["exp(coef)", "exp(coef) lower 95%",
                                   "exp(coef) upper 95%"]].round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = switched.coefficients()
    for _mode, (_alpha, _beta, _coef) in truth.items():
        _row = _t.loc[(_mode, "abrasive")]
        assert _row["coef lower 95%"] < _coef < _row["coef upper 95%"], _mode

Predictions along a planned schedule take the mode in the same place:
``cif_tvc(x, Z, event, xl, given)``, ``iif_tvc`` and ``cb_tvc``, and
``total_cif_tvc``, ``total_iif_tvc`` and ``total_cb_tvc`` for failures of
any kind. For a pump that is 24 months old and about to go onto the slurry,
the next two years:

.. jupyter-execute::

    from surpyval import StepSchedule

    slurry = StepSchedule.from_changepoints([24], [[1.0]])
    stay_clean = StepSchedule.from_changepoints([24], [[0.0]])
    for label, plan in (("onto the slurry", slurry), ("stays clean", stay_clean)):
        seals = switched.cif_tvc(48, plan, "seal", given=24)
        every = switched.total_cif_tvc(48, plan, given=24)
        lo, hi = switched.total_cb_tvc(48, plan, given=24)
        print(f"{label:15s}: {seals:.2f} seal failures, {every:.2f} of any "
              f"kind (95% CI {lo:.2f} to {hi:.2f})")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.isclose(
        switched.total_cif_tvc(48, slurry, given=24),
        sum(switched.cif_tvc(48, slurry, k, given=24)
            for k in switched.event_types),
    )
    assert np.allclose(
        switched.total_cb_tvc([24.0, 48.0], StepSchedule.constant([1.0])),
        switched.total_cif_cb([24.0, 48.0], [1.0]),
    )

Comparing, saving, and what is not available yet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``neg_ll``, ``aic``, ``bic`` and ``aic_c`` are those of the whole model: the
likelihood is the product of the modes', so the log-likelihoods add (and BIC
counts the failures of every mode), as for ``CauseSpecificNHPP``. Use them to
compare baselines or sets of covariates. ``to_dict`` / ``from_dict`` (and
``to_json`` / ``from_json``) keep every prediction, and each mode's
parameter covariance, so the restored model gives the same bounds and
tables:

.. jupyter-execute::

    hpp = CauseSpecificProportionalIntensity.fit_from_df(
        log, x_col="month", Z_cols="abrasive", e_col="mode",
        i_col="pump", c_col="c", baseline=surpyval.recurrent.HPP,
    )
    print("AIC, constant rates   :", round(hpp.aic(), 1))
    print("AIC, power-law trends :", round(modes.aic(), 1))

    restored = CauseSpecificProportionalIntensity.from_dict(modes.to_dict())
    print(restored.total_cif_cb(months, [1]).round(2))
    print(modes.total_cif_cb(months, [1]).round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert modes.aic() < hpp.aic()
    assert np.allclose(restored.total_cif_cb(months, [1]),
                       modes.total_cif_cb(months, [1]))
    assert np.allclose(restored.coefficients().to_numpy(),
                       modes.coefficients().to_numpy())

Some things the single-mode models and ``CauseSpecificNHPP`` offer are not
carried over to this model yet. Each of these refuses with a message that
says what to use instead:

- **Diagnostics** (``diagnose``, ``residuals``, ``cramer_von_mises``,
  ``trend_test``) and ``plot``: call them on one mode's own model,
  ``modes.models["seal"].diagnose()``. Each mode is a Poisson process of its
  own, so its diagnostics are valid there (for a fixed-covariate fit; a mode
  fitted to changing covariates refuses them, as above).
- **Simulation** and the simulated ``mcf``: simulate each mode from its own
  model at a covariate row; the modes are independent, so their simulated
  failures can be pooled. The expected counts need no simulation (``cif``,
  ``total_cif``).
- **Forecasts** (``forecast`` and ``surpyval.forecast``): the expected
  number of failures over a time ahead is ``total_cif_tvc(age + horizon, Z,
  xl=[age], given=age)``, with bounds from ``total_cb_tvc``; one mode's
  forecast with its prediction interval is ``surpyval.forecast(
  modes.models["seal"], age, horizon, Z=Z)``.
- ``growth``, and the MTBF methods (``mtbf``, ``total_mtbf`` and their
  bounds): read each mode's trend parameter from ``summary()``, and take the
  reciprocal of ``iif`` / ``total_iif`` and of their bounds.
- **Counts of failures** between inspections (``c = 2``) and right
  truncation: a count carries no mode per failure. Fit each mode's own counts
  with the single-mode regression.
- Only the Wald (delta-method) bounds are offered, as on the single-mode
  models; there is no penalised fit, and the ``_from_df`` forms take
  ``Z_cols`` only (no ``formula``).

Without a baseline shape: Andersen-Gill, Prentice-Williams-Peterson and Wei-Lin-Weissfeld
------------------------------------------------------------------------------------------

Every model above gives the baseline intensity a formula. Three fitters
leave it unspecified, as ``CoxPH`` does for a single event, and estimate
only how the covariates change the rate: ``AndersenGill``,
``PrenticeWilliamsPeterson`` and ``WeiLinWeissfeld``. They differ in what
they assume a repeat event is; :doc:`Recurrent Event Regression Analysis`
sets the four models side by side, in words. In one line each:

- ``AndersenGill``: an event does not change the item; one baseline for all
  events. The coefficient is a log *rate ratio*.
- ``PrenticeWilliamsPeterson``: each event number has its own baseline, and
  an item is at risk of its second event only after its first.
  ``time_scale="total"`` (the default) measures time as the item's age,
  ``time_scale="gap"`` as the time since its last event.
- ``WeiLinWeissfeld``: the time from the start to the first event, to the
  second, and so on, each modelled on its own.

All three take the event log the other recurrent models take: ``fit(x, Z,
i=..., c=...)``, with ``c = 0`` for an event and ``c = 1`` for the end of an
item's observation, or ``fit_from_df`` with the column names. All three
return the same kind of model, and all three report **robust standard
errors**, with each item a cluster: an item's events are not independent
pieces of evidence, and the robust errors do not pretend they are.

The data are laid out for the model (the risk interval of each row, and its
event number) and handed to ``CoxPH``, so tied times, a coefficient the
data cannot determine and a likelihood with no maximum are handled, and
reported, as ``CoxPH`` does. The results equal R's ``coxph(... +
cluster(id))`` on the matching layout.

An engineering example: trucks, and what the log does not record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Eighty haul trucks are watched for 36 months; half work a dusty pit. To
have a known answer the breakdowns are simulated: dust multiplies the
breakdown rate by :math:`e^{0.5} \approx 1.65`, and, as in any real fleet,
some trucks are simply worse than others for reasons the log does not
record (a factor ``hidden`` that the analysis never sees).

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    from surpyval.recurrent import AndersenGill

    rng = np.random.default_rng(2026)
    months = 36.0
    rows = []
    for truck in range(80):
        dusty = float(truck % 2)
        hidden = rng.gamma(1 / 0.6, 0.6)   # not in the log
        expected = (months / 9.0) ** 1.3 * hidden * np.exp(0.5 * dusty)
        times = months * rng.uniform(size=rng.poisson(expected)) ** (1 / 1.3)
        rows += [(truck, t, 0, dusty) for t in np.sort(times)]
        rows.append((truck, months, 1, dusty))
    fleet_log = pd.DataFrame(rows, columns=["truck", "month", "c", "dusty"])

    ag = AndersenGill.fit_from_df(
        fleet_log, x_col="month", Z_cols="dusty", i_col="truck", c_col="c"
    )
    ag

The estimate is a rate ratio of about 1.5, and its 95% interval contains
the true 1.65. ``summary()`` shows the two standard errors side by side:

.. jupyter-execute::

    ag.summary()[["coef", "exp(coef)", "se(coef)", "naive se(coef)"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = ag.summary()
    assert round(float(_t["exp(coef)"].iloc[0]), 1) == 1.5, _t
    assert _t["exp(coef) lower 95%"].iloc[0] < np.exp(0.5), _t
    assert np.exp(0.5) < _t["exp(coef) upper 95%"].iloc[0], _t
    _ratio = float(_t["se(coef)"].iloc[0] / _t["naive se(coef)"].iloc[0])
    assert 2.0 < _ratio < 2.5, _ratio

The robust standard error is more than twice the model-based ("naive") one.
The naive error counts 656 breakdowns as 656 independent observations of
the effect of dust; the robust one sees that they come from 80 trucks, and
that trucks differ. Had the naive error been used, the interval would have
been less than half as wide as the data justify. ``covariance()`` and
``standard_errors()`` are the robust ones; ``naive_covariance()`` is the
other.

``cif(x, Z)`` is the expected number of breakdowns by age ``x``, the
cumulative baseline scaled by the covariates. It agrees closely with the
parametric model fitted to the same log, without having been told the shape
of the trend:

.. jupyter-execute::

    from surpyval.recurrent import CrowAMSAA, ProportionalIntensityNHPP

    parametric = ProportionalIntensityNHPP.fit_from_df(
        fleet_log, x_col="month", Z_cols="dusty", i_col="truck", c_col="c",
        baseline=CrowAMSAA,
    )
    ages = [12, 24, 36]
    print("Andersen-Gill, clean pit :", ag.cif(ages, [0]).round(2))
    print("Crow-AMSAA,    clean pit :", parametric.cif(ages, [0]).round(2))
    print("Andersen-Gill, dusty pit :", ag.cif(ages, [1]).round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(ag.cif(ages, [0]), parametric.cif(ages, [0]), rtol=0.05)

What it cannot do is forecast: the baseline is a step function that stops
at the last breakdown, so for month 48 use the parametric model. Bounds on
the curve itself are not offered either (``cif_cb`` says so and points to
the alternatives); the robust interval of the rate ratio is in
``summary()``.

A medical example: one trial, three questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three hundred patients are randomised to a treatment or not, and followed
for 24 months for recurrent infections. In this simulation an infection
changes the patient: the first comes slowly (on average after 30 months
untreated), and after one, the next comes much sooner (8 months). The
treatment stretches every one of these waiting times by the same factor,
:math:`e^{0.5}`, a log hazard ratio of :math:`-0.5` for each wait.

.. jupyter-execute::

    from surpyval.recurrent import PrenticeWilliamsPeterson, WeiLinWeissfeld

    rng = np.random.default_rng(1989)
    follow_up = 24.0
    rows = []
    for patient in range(300):
        treated = float(patient % 2)
        age, number = 0.0, 1
        while True:
            mean_wait = (30.0 if number == 1 else 8.0) * np.exp(0.5 * treated)
            age += rng.exponential(mean_wait)
            if age >= follow_up:
                break
            rows.append((patient, age, 0, treated))
            number += 1
        rows.append((patient, follow_up, 1, treated))
    trial = pd.DataFrame(rows, columns=["patient", "month", "c", "treated"])

    columns = dict(x_col="month", Z_cols="treated", i_col="patient", c_col="c")
    fits = {
        "Andersen-Gill": AndersenGill.fit_from_df(trial, **columns),
        "PWP, gap time": PrenticeWilliamsPeterson.fit_from_df(
            trial, time_scale="gap", max_events=3, **columns
        ),
        "PWP, total time": PrenticeWilliamsPeterson.fit_from_df(
            trial, max_events=3, **columns
        ),
        "Wei-Lin-Weissfeld": WeiLinWeissfeld.fit_from_df(
            trial, max_events=3, **columns
        ),
    }
    pd.DataFrame(
        {
            name: {
                "coef": fit.params[0],
                "se(coef)": fit.standard_errors()[0],
                "naive se(coef)": np.sqrt(fit.naive_covariance()[0, 0]),
            }
            for name, fit in fits.items()
        }
    ).T.round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _gap = fits["PWP, gap time"]
    assert abs(_gap.params[0] + 0.5) < 2 * _gap.standard_errors()[0]
    assert all(fit.params[0] < 0 for fit in fits.values())
    _r = {k: f.standard_errors()[0] / np.sqrt(f.naive_covariance()[0, 0])
          for k, f in fits.items()}
    assert 0.9 < _r["PWP, gap time"] < 1.1, _r
    assert _r["Andersen-Gill"] > 1.25 and _r["Wei-Lin-Weissfeld"] > 1.25, _r
    assert fits["Wei-Lin-Weissfeld"].params[0] < _gap.params[0]

All four find that the treatment helps, and they are answers to different
questions:

- The **gap-time** model matches how these data arise, and recovers the
  effect on each wait (the true value is :math:`-0.5`). Its robust and
  naive standard errors agree: once each infection number has its own
  baseline, nothing is left over to make a patient's waits alike.
- The **Andersen-Gill** coefficient is the effect on the overall infection
  rate. It is larger in size, because the treatment helps twice: it delays
  each infection, and by delaying the first it keeps patients longer in the
  state where infections are rare. Its robust error is well above its naive
  one: with a single baseline, the dependence the model leaves out (a
  patient who has had one infection is on the fast clock) shows up as
  patients differing.
- The **Wei-Lin-Weissfeld** coefficient is the effect on the time from
  randomisation to each infection number. It too carries the knock-on
  effect of delaying earlier infections, and is the one that compares the
  groups as randomised.

``max_events=3`` models the first three infections only: few patients have
a fourth, and a baseline estimated from a handful of them is poorly
determined. For the conditional models, what follows a patient's third
infection is left out; for the marginal model it is the number of event
numbers modelled.

With a baseline per event number, ``cif`` takes the event number as
``stratum``, and for the gap-time and marginal models ``sf`` turns it into
a probability: of waiting longer than ``x`` for that infection, or of not
having had it by ``x`` months after randomisation.

.. jupyter-execute::

    gap = fits["PWP, gap time"]
    print("P(first wait > 12 months), untreated :",
          gap.sf(12, [0], stratum=1).round(2))
    print("P(second wait > 12 months), untreated:",
          gap.sf(12, [0], stratum=2).round(2))

    wlw = fits["Wei-Lin-Weissfeld"]
    print("P(no second infection by month 24), untreated:",
          wlw.sf(24, [0], stratum=2).round(2))
    print("P(no second infection by month 24), treated  :",
          wlw.sf(24, [1], stratum=2).round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert gap.sf(12, [0], stratum=2) < gap.sf(12, [0], stratum=1)
    assert abs(gap.sf(12, [0], stratum=1) - np.exp(-12 / 30)) < 0.08
    assert abs(gap.sf(12, [0], stratum=2) - np.exp(-12 / 8)) < 0.08
    assert wlw.sf(24, [1], stratum=2) > wlw.sf(24, [0], stratum=2)

The second wait is much the shorter, as simulated (the true values are
0.67 and 0.22).

Start-stop rows, saving, and what is not available
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In an event log, the covariates on a row are those in force up to that
row's time, so they may change from one of an item's rows to the next (the
marginal model, which starts every clock at 0, needs them constant within
an item and refuses otherwise). For an item that enters late, a break in
its observation, or a covariate that changes between events,
``AndersenGill`` and ``PrenticeWilliamsPeterson`` take start-stop rows,
``fit_tvc(i, xl, xr, c, Z)`` or ``fit_tvc_from_df``: each row an interval
``(xl, xr]`` of one item, ending in an event (``c = 0``) or not. This is
the layout of R's ``Surv(start, stop, event)``:

.. jupyter-execute::

    rows = pd.DataFrame({
        "id":    [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        "start": [2, 5, 9, 0, 6, 0, 3, 8, 1, 4],
        "stop":  [5, 8, 12, 6, 11, 3, 7, 12, 4, 10],
        "c":     [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
        "load":  [0.2, 0.9, 0.9, 1.5, 1.1, 0.4, 0.1, 0.8, 1.9, 1.3],
    })
    small = AndersenGill.fit_tvc_from_df(
        rows, i_col="id", xl_col="start", xr_col="stop", c_col="c",
        Z_cols="load",
    )
    small.params.round(3)

A model saves and loads like any other, keeping both covariances and every
prediction:

.. jupyter-execute::

    import surpyval

    restored = surpyval.from_dict(ag.to_dict())
    print(restored.standard_errors().round(4), ag.standard_errors().round(4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.array_equal(restored.standard_errors(), ag.standard_errors())
    assert np.array_equal(restored.cif(ages, [1]), ag.cif(ages, [1]))

``model.cox`` is the Cox model fitted to the laid-out rows, for anything
this model does not wrap: ``model.diagnose()`` is its ``diagnose()`` (the
proportional-hazards tests, read here as a screen, since they take the rows
as independent), and its residuals are row by row. Not available yet, each
a refusal or an absent method rather than a wrong answer:

- **Bounds on the curve** (``cif_cb``): the robust variance is of the
  coefficients only. ``model.cox.cb(x, Z, on="Hf")`` gives the model-based
  bound.
- **Simulation and forecasts**: the baseline has no shape beyond the data.
  Use the parametric models above.
- **Counted failures** (``c = 2``) and left-censored rows: the partial
  likelihood needs each event's time. ``ProportionalIntensityNHPP`` takes
  them.
- **Weights** and **penalised** fits.
- **Several failure modes**: fit one model per mode, to that mode's events
  with the other modes' rows flagged ``c = 1`` (as
  ``CauseSpecificProportionalIntensity`` does for the parametric models).

See :doc:`Recurrent Event Regression Analysis` for the theory and references
behind these models.
