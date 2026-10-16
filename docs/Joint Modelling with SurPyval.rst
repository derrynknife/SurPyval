Joint Modelling with SurPyval
=============================

Many units carry a measurement of their condition that is read again and
again: a pump's vibration, a gearbox's oil-debris count, a tyre's tread
depth, a patient's biomarker. The measurement -- the *marker* -- drives
the risk of failure, but it is read with noise, only now and then, and
the readings stop when the unit fails. A *joint model* describes the
readings and the failure together, and gives what an engineer wants from
them: each unit's chance of surviving the next few months **given its own
readings so far**, updated as each new reading arrives. This is the
*dynamic prediction*, a remaining-useful-life estimate per asset.

Import the model from ``surpyval.joint``. The API reference is
:doc:`surpyval.joint`.

Why not a regression on the readings?
-------------------------------------

The obvious analyses are biased, and in known directions:

- **A covariate taken at the start** (the first reading) ignores how the
  marker moves afterwards, and its effect is much too small.
- **A time-varying covariate** (the last reading carried forward, in
  ``CoxPH.fit_tvc``) treats each noisy reading as the truth. Measurement
  error pulls its effect towards zero, and the hazard reacts to the noise.
- **A model of the readings alone** (``DegradationAnalysis``) ignores that
  the units whose marker rose fastest leave the data first, at failure;
  and a threshold model of failure (a unit fails when its marker crosses a
  level) cannot describe failures that happen *below* the threshold, at a
  higher rate the higher the marker.

The joint model shares each unit's random effects between its path and its
hazard, so the hazard depends on the unit's true (noise-free) marker, and
the likelihood of the readings and of the failure are maximised together.
It is the hazard, not a threshold, that ends a unit's life: the right model
when the marker raises the risk of failure rather than defining it.
Degradation to a threshold remains :doc:`Degradation Modelling with
SurPyval`'s.

The model
---------

For unit :math:`i` with readings :math:`y_{ij}` at ages :math:`t_{ij}`,
hazard covariates :math:`z_i` and path covariates :math:`w_i`:

.. math::

    y_{ij} &= m_i(t_{ij}) + \varepsilon_{ij}, \qquad
        \varepsilon_{ij} \sim N(0, \sigma^2) \\
    m_i(t) &= (\beta_0 + \delta_0' w_i + b_{0i})
        + (\beta_1 + \delta_1' w_i + b_{1i})\, t, \qquad
        b_i \sim N(0, D) \\
    h_i(t \mid b_i) &= \frac{k}{\alpha}\left(\frac{t}{\alpha}\right)^{k - 1}
        \exp\big(\gamma' z_i + a\,(m_i(t) - c)\big)

The longitudinal part is a linear mixed model -- a straight path with a
random intercept and slope per unit -- and the survival part a Weibull
proportional hazards model whose covariate is the path's current value.
The parameters are named, in ``params``:

- ``alpha`` and ``beta``: the Weibull scale and shape of the baseline,
  for a unit whose marker stays at the *marker centre* :math:`c`
  (``marker_centre``, the mean reading unless given) and whose covariates
  are 0;
- the hazard coefficients :math:`\gamma`, by their covariates' names;
- ``association`` (:math:`a`): the log hazard ratio per unit of the
  marker's current value;
- ``path_intercept`` and ``path_slope`` (:math:`\beta_0, \beta_1`), and
  for each path covariate ``w``, ``path_intercept:w`` and ``path_slope:w``;
- ``sd_intercept``, ``sd_slope`` and ``corr``: the random effects'
  standard deviations and correlation (:math:`D`);
- ``sigma``: the noise of a reading.

The random effects are integrated out of the likelihood by adaptive
Gauss-Hermite quadrature, ``n_quad`` points per dimension at each unit's
posterior mode (Rizopoulos 2012), and the likelihood is maximised over
all the parameters at once. The fit starts from the two-stage estimate:
the readings fitted alone as a linear mixed model (by REML), then the
survival part with each unit's predicted path.

A fleet of pumps
----------------

``simulate`` draws a fleet from a model with known parameters, which is
how we can see the model recover them. Here 300 pumps' vibration (mm/s)
starts near 2.8 and rises by about 1.6 mm/s a year, each pump at its own
rate; every month a route reading is taken, with noise of 0.6 mm/s; each
1 mm/s of true vibration multiplies the hazard by :math:`e^{0.9} = 2.5`,
and heavy-duty pumps fail at :math:`e^{0.5} = 1.65` times the rate. The
pumps are watched for two years.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from surpyval.joint import JointModel

    truth = JointModel.from_params(
        {
            "alpha": 900.0, "beta": 1.4, "heavy": 0.5, "association": 0.9,
            "path_intercept": 2.8, "path_slope": 1.6 / 365,
            "sd_intercept": 0.5, "sd_slope": 0.7 / 365, "corr": 0.0,
            "sigma": 0.6,
        },
        marker_centre=3.5,
    )
    heavy = np.tile([0.0, 1.0, 0.0, 0.0, 1.0], 60)   # 40% heavy duty
    fleet = truth.simulate(300, 30.0, 730.0, Z=heavy, random_state=1)
    fleet.head()

The data are in the "long" layout: one row per reading (``marker_x``, the
age in days, and ``marker_y``, the vibration), with its pump's id, failure
or censoring time ``x``, censoring flag ``c`` and covariates repeated on
each row. This is how readings are usually stored, and how R's ``JM``
takes them.

.. jupyter-execute::

    units = fleet.groupby("i").first()
    print(f"{len(units)} pumps, {len(fleet)} readings, "
          f"{int((units['c'] == 0).sum())} failures")

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, g in list(fleet.groupby("i"))[:25]:
        ax.plot(g["marker_x"], g["marker_y"], lw=0.8,
                color="C3" if g["c"].iat[0] == 0 else "C0")
    ax.set_xlabel("Age (days)")
    ax.set_ylabel("Vibration (mm/s)")
    ax.set_title("25 pumps: red failed, blue still running at the end");

Fitting
-------

``fit_from_df`` takes the long DataFrame; ``fit`` takes the same data as
arrays, one row per unit (``x``, ``c``, ``Z``, ``i``) and one per reading
(``marker_x``, ``marker_y``, ``marker_i``).

.. jupyter-execute::

    model = JointModel.fit_from_df(
        fleet, x_col="x", c_col="c", i_col="i",
        marker_x_col="marker_x", marker_y_col="marker_y",
        Z_cols=["heavy"], marker_centre=3.5,
    )
    model

.. jupyter-execute::

    model.summary()[["coef", "se(coef)", "coef lower 95%", "coef upper 95%"]]

The association is about 0.96: each mm/s of true vibration multiplies a
pump's hazard by about 2.6, and the 95% interval covers the true 0.9.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(model.association, 2) == 0.96, model.association
    _lo, _hi = model.param_cb("association")
    assert _lo < 0.9 < _hi and round(np.exp(model.association), 1) == 2.6

Compare a Weibull regression on each pump's *first* reading, the
covariate-at-the-start analysis: its effect per mm/s is much smaller,
because the first reading says little about where the vibration went
afterwards, and a noisy reading less still.

.. jupyter-execute::

    import surpyval as sp

    first = fleet.groupby("i").first()
    naive = sp.WeibullPH.fit_from_df(
        first, x_col="x", c_col="c", Z_cols=["marker_y", "heavy"]
    )
    print(f"first reading: {naive.params[2]:.2f} per mm/s; "
          f"joint model: {model.association:.2f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert naive.params[2] < 0.5 * model.association, naive.params

A new pump
----------

For a pump that has no readings yet, ``sf`` is the survival function of
the fleet: each pump's survival averaged over the spread of the paths.
Wald bounds come from the fit's covariance (``cb``).

.. jupyter-execute::

    x = np.array([180.0, 365.0, 730.0])
    print(model.sf(x, Z=[0.0]).round(3), model.sf(x, Z=[1.0]).round(3))
    model.cb(x, Z=[1.0]).round(3)

Dynamic prediction: one pump's remaining life
---------------------------------------------

Now the question the model exists for. A heavy-duty pump has been read
five times; its vibration has risen from 3.3 to 5.0 mm/s in 120 days,
faster than most. ``predict`` gives its failure-age distribution given
those readings and that it is running today, at 120 days.

.. jupyter-execute::

    ages = [0.0, 30.0, 60.0, 90.0, 120.0]
    readings = [3.3, 3.9, 4.2, 4.6, 5.0]
    pump = model.predict(ages, readings, Z=[1.0])
    pump

Its ``sf(x)`` is the probability it is still running at age ``x`` (1 up to
today), and ``remaining_life`` the time from today it outlives with a given
probability: here the median, and a 90% prediction interval.

.. jupyter-execute::

    horizons = 120.0 + np.array([30.0, 90.0, 180.0])
    print("survival 30, 90 and 180 days from now:", pump.sf(horizons).round(3))
    print("remaining life (5%, 50%, 95%):",
          pump.remaining_life([0.05, 0.5, 0.95]).round(0))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _median = float(pump.remaining_life(0.5))
    assert 60 < _median < 200, _median

``cb`` bounds that survival for the uncertainty in the fitted parameters
(the posterior of the pump's own random effects is recomputed at each
parameter value, by the delta method). The ``remaining_life`` interval is
the spread of the pump's own life, which no amount of data removes.

.. jupyter-execute::

    pump.cb(horizons).round(3)

The prediction follows the pump. A calm pump of the same age can expect
nearly twice the remaining life, and a new reading moves the prediction: a
sixth reading of 5.6 mm/s at 150 days lowers it again.

.. jupyter-execute::

    calm = model.predict(ages, [3.0, 3.0, 3.1, 3.2, 3.1], Z=[1.0])
    later = model.predict(ages + [150.0], readings + [5.6], Z=[1.0])
    print("median remaining life, calm pump:", calm.remaining_life().round(0))
    print("P(running at 300 days): now {:.3f}, after the new reading {:.3f}"
          .format(pump.sf(300.0), later.sf(300.0)))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _ratio = calm.remaining_life() / pump.remaining_life()
    assert 1.7 < _ratio < 2.0, _ratio
    assert later.sf(300.0) < pump.sf(300.0)

A picture of the prediction: the pump's readings, its smoothed path and
where it is heading (``marker`` and ``marker_interval``), and its survival
from today with the parameter band.

.. jupyter-execute::

    t = np.linspace(0.0, 480.0, 97)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    band = pump.marker_interval(t)
    left.fill_between(t, band[:, 0], band[:, 1], alpha=0.25)
    left.plot(t, pump.marker(t), label="true vibration (posterior mean)")
    left.plot(ages, readings, "ko", label="readings")
    left.axvline(pump.age, color="grey", ls=":")
    left.set_xlabel("Age (days)"); left.set_ylabel("Vibration (mm/s)")
    left.legend()
    future = np.linspace(pump.age, 480.0, 60)
    cb = pump.cb(future)
    right.fill_between(future, cb[:, 0], cb[:, 1], alpha=0.25)
    right.plot(future, pump.sf(future))
    right.set_xlabel("Age (days)"); right.set_ylabel("P(still running)")
    right.set_title("Survival from today");

Each reading is a unit's ``ItemHistory`` too: an item's ``markers`` are its
readings, ``{age: value}``, and ``JointModel.fit_from_items`` fits a list
of items (or an ``EventHistoryData`` of them); ``predict`` takes one item.

.. jupyter-execute::

    from surpyval import ItemHistory

    item = ItemHistory("P-17", right_censored=120.0, covariates=[1.0],
                       markers=dict(zip(ages, readings)))
    float(model.predict(item).sf(300.0))

Censoring, truncation and covariates
------------------------------------

Every kind of censoring of the failure is taken, as in every SurPyval
model: ``c = 1`` for a unit still running at ``x``, ``-1`` for one found
failed at ``x``, and ``2`` with ``x = [lower, upper]`` for a failure found
at an inspection. Left truncation (``tl``) is a unit that came under
observation at an age, already running -- a pump installed before the
monitoring started; such a unit is in the data because it survived to
``tl``, and its likelihood is divided by the probability of that, averaged
over the random effects, so that ``D`` stays the spread of the paths of
new pumps. Readings must be taken while a unit is watched (at or before
its time ``x``).

Hazard covariates are ``Z`` (one row per unit); covariates of the path are
``path_Z``, and each shifts both the path's intercept and its slope. Pass
the same columns to both for a covariate that acts on both. A column the
data cannot determine is aliased (``nan``, with one warning), as in every
regression.

The time unit and the marker's unit do not matter: rescaling either
rescales the parameters in their units and changes no prediction. A model
saves to and loads from a dict or JSON (``to_dict``, ``surpyval.from_dict``)
with its covariance, so a loaded model predicts, bounds and updates as the
fitted one did.

A real data set: primary biliary cirrhosis
------------------------------------------

The Mayo Clinic trial (:func:`~surpyval.datasets.load_pbc2`) recorded serum
bilirubin at each visit, and deaths, with transplantation censored. On the
log scale, bilirubin is close to a straight line per patient.

.. jupyter-execute::

    from surpyval.datasets import load_pbc2

    pbc = load_pbc2()
    pbc["log_bilirubin"] = np.log(pbc["serBilir"])
    pbc["censored"] = 1 - pbc["status2"]
    liver = JointModel.fit_from_df(
        pbc, x_col="years", c_col="censored", i_col="id",
        marker_x_col="year", marker_y_col="log_bilirubin",
    )
    print(f"association {liver.association:.2f}: each unit of log "
          f"bilirubin multiplies the hazard of death by "
          f"{np.exp(liver.association):.1f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(liver.association, 2) == 1.24, liver.association

How good are the estimates?
---------------------------

On fleets like the one above (200 pumps, monthly readings, two years,
15% of the pumps withdrawn early), 1,000 simulated fleets gave every fit a
verified maximum; average estimates within 2.2% of the truth for every
parameter (the association 1.7% high, a fifth of its standard deviation);
standard errors within 5% of the estimates' actual spread; and 95% Wald
intervals that covered the truth 94.0 to 95.1% of the time. The 95% band
of a held-out pump's dynamic prediction covered its true conditional
survival 94.1 to 94.4% of the time at 60, 180 and 365 days ahead. A
shorter run of the same study (200 fleets) is
``surpyval/tests/calibration/test_coverage_joint.py``, and the
predictions themselves are checked against Monte Carlo in the test suite.

Not yet
-------

The first version is deliberately one model; the API leaves room for
each of these without a break:

- **AFT and accelerated-life survival parts**, through cumulative exposure,
  :math:`S(t \mid b) = S_0\big(\int_0^t e^{a\, m(s)} ds\big)`
  (``survival=``);
- **links to the path's slope or its cumulative value** (``link=``);
- **spline paths** for markers that rise late in life (``path=``): a
  straight path fitted to a curved one biases the joint model too;
- **several markers**, each with its own path or as noisy views of one
  latent health index (an ``ItemHistory`` reading may already be a row);
- **the two-stage estimate** as an option, to compare with.

References
----------

Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event
Data, with Applications in R*. Chapman & Hall/CRC.

Rizopoulos, D. (2010). JM: An R package for the joint modelling of
longitudinal and time-to-event data. *Journal of Statistical Software*,
35(9), 1-33.

Rizopoulos, D. (2011). Dynamic predictions and prospective accuracy in
joint models for longitudinal and time-to-event data. *Biometrics*, 67(3),
819-829.
