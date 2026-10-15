Joint Models
============

A measurement read again and again on each item -- a pump's vibration, a
gearbox's oil debris, a tyre's tread depth, a patient's biomarker -- often
says how close the item is to failing. The item's risk rises as the
measurement does. But the measurement is noisy, it is read only at
inspections, and it stops when the item fails. A **joint model** describes
the measurement (the *marker*) and the lifetime together, so that each is
read in the light of the other. In reliability terms it gives each asset's
remaining life given its readings so far, updated as new readings arrive.

SurPyval's joint model (``WeibullJointPH``, #891) is the standard one
(Rizopoulos, 2012):

- **the marker** is a straight line in age with an intercept and a slope of
  the item's own (drawn, across items, from a bivariate normal), read with
  normal noise;
- **the hazard** is Weibull proportional hazards in the line's *current,
  noise-free* value, and the item's covariates, which also shift its line.

Why not the obvious approaches?
-------------------------------

Two shortcuts are common, and both are biased:

- **The last reading as a time-varying covariate** (a time-varying Cox or
  Weibull PH fit, the reading carried forward to the next). It takes a noisy
  reading as the truth, and between readings a value that is out of date.
  Both pull the marker's estimated effect towards zero -- regression
  dilution -- however many items there are.
- **The marker alone** (a degradation model). The readings stop when an item
  fails, and the items that fail are the ones whose marker was high: fitting
  the readings without the failures under-states how markers rise. And a
  threshold on the marker cannot describe failures that come at any level,
  with a risk that only *grows* with it.

The joint likelihood fixes both: each item's line is integrated over what
its readings leave uncertain, and its readings and its lifetime inform the
same line.

The model
---------

Item :math:`i` has readings :math:`y_{ij}` at ages :math:`t_{ij}`,
covariates :math:`z_i`, and fails or is censored at :math:`T_i`:

.. math::

    m_i(t) = (\beta_0 + b_{i0}) + (\beta_1 + b_{i1})\, t + \delta' z_i,
    \qquad y_{ij} = m_i(t_{ij}) + \varepsilon_{ij},

.. math::

    h_i(t) = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{\beta-1}
    \exp\left(\gamma' z_i + \zeta\,\bigl(m_i(t) - c\bigr)\right),

with :math:`b_i \sim N(0, D)` and :math:`\varepsilon_{ij} \sim
N(0, \sigma^2)`. :math:`\zeta` (``marker_effect``) is the change in the log
hazard per unit of the marker: ``exp(marker_effect)`` is the hazard ratio of
one unit. The marker enters centred at :math:`c` (``marker_center``, the mean
reading of the data), so the Weibull :math:`\alpha` and :math:`\beta` are the
baseline of an item with its marker at :math:`c` and its covariates at 0.

Given its line, an item's readings and lifetime are independent, so its
likelihood is

.. math::

    L_i = \int p(y_i \mid b)\, h_i(T_i \mid b)^{d_i}\, S_i(T_i \mid b)\,
    \phi(b; 0, D)\, db,

:math:`d_i` 1 for a failure and 0 for a censoring. The fit maximises the sum
of the :math:`\log L_i`, each two-dimensional integral by adaptive
Gauss--Hermite quadrature (7 points a dimension, the grid centred at each
item's most likely line and scaled by how sure its readings make it); the
nodes are held fixed while the parameters are searched (with autograd's
gradient), then placed again at the new parameters, until the
log-likelihood stops moving. The cumulative hazard along a line is by
15-point Gauss--Legendre quadrature. The fit starts from the marker's mixed
model alone (by REML, the longitudinal part of ``DegradationAnalysis``) and
a survival fit with each item's predicted line plugged in -- a two-stage
estimate, which is biased, and only a start. The covariance is the inverse
of the Hessian of the negative log-likelihood, by autograd.

An item that came under observation at age :math:`L_i > 0`
(``left_truncation``) is in the data because it reached :math:`L_i`: its
likelihood is divided by the probability of that, :math:`\int S(L_i \mid b)
\phi(b; 0, D)\, db`, over the population's lines.

The data: readings on the item records
--------------------------------------

The readings are the ``markers`` of each item's
:class:`~surpyval.ItemLifetime`, ``{age: value}``, taken while it was
observed (from its ``left_truncation``, else 0, to its failure or
censoring). Here is a fleet of 300 pumps followed for two years and read
monthly, simulated from the model: vibration starts near 3 mm/s and rises
about 1 mm/s a year, each pump at its own level and rate, read with noise
of 0.7 mm/s; each mm/s multiplies the hazard by :math:`e^{0.9} = 2.5`.

.. jupyter-execute::

    import numpy as np
    from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH

    rng = np.random.default_rng(42)


    def pump(k):
        start = rng.normal(3.0, 0.6)             # vibration when new, mm/s
        rate = rng.normal(1.0, 0.5) / 365        # its rise, mm/s a day
        # hazard: Weibull (800 days, shape 2) at 4 mm/s, x e^0.9 per mm/s
        t = np.linspace(0, 3000, 30001)
        h = (2 / 800) * (t / 800) * np.exp(0.9 * (start + rate * t - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(rng.exponential(), H, t)
        end = min(life, 730.0)                    # followed for two years
        ages = np.arange(0.0, end, 30.0)          # read monthly
        readings = start + rate * ages + rng.normal(0, 0.7, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(
            f"P{k}",
            **{outcome: round(end, 1)},
            markers=dict(zip(ages, readings.round(2))),
        )


    fleet = SurpyvalData.from_items([pump(k) for k in range(300)])
    print(fleet.summary())

Each record holds its readings, and checks them: a reading after the
item's failure or censoring, or before it came under observation, is
refused, naming the item. A new reading of an item still running is added
with ``update_markers`` (after ``extend``-ing it to that age), which returns a
new record:

.. jupyter-execute::

    p = fleet.item("P7")
    print(p.outcome, p.value, len(p.markers), "readings")

    pump_now = ItemLifetime("P900", right_censored=60, markers={0: 3.1, 30: 3.4})
    pump_now.extend(90).update_markers(90, 3.9)

.. jupyter-execute::
    :raises: ValueError

    pump_now.update_markers(120, 4.2)

The dataset carries the readings through every reshape:
``fleet.censor_at(365)`` keeps those up to 365 days, ``enter_at`` those from
the new start, ``select`` and slicing those of the items kept, and
``to_json`` writes them. Every other fit refuses data that holds markers --
it would otherwise ignore them -- and says how to leave them out:

.. jupyter-execute::
    :raises: ValueError

    from surpyval import Weibull

    Weibull.fit(fleet)

.. jupyter-execute::

    Weibull.fit(fleet.without("markers")).params

Fitting
-------

``WeibullJointPH.fit`` takes the dataset, as every fitter does:

.. jupyter-execute::

    model = WeibullJointPH.fit(fleet, unit="days")
    model

The marker part recovers the simulation (a line from 3 mm/s, rising 1/365
= 0.00274 mm/s a day, noise 0.7), and the survival part a marker effect of
about 0.9. (The data can also be given as arrays, as for every fitter:
``WeibullJointPH.fit(x, c=c, Z=Z, markers=[{age: value, ...}, ...])``, one
``{age: value}`` per item.) The model has ``params`` (named by ``parameter_names``),
``covariance()``, ``standard_errors()``, ``neg_ll()``, ``aic()``, ``bic()``
and ``summary()``, as every model does:

.. jupyter-execute::

    model.summary()[["coef", "se(coef)", "exp(coef)"]].round(4)

``neg_ll()`` is the negative log-likelihood of the readings and the lifetimes
together, so ``aic()`` and ``bic()`` compare joint models of the same data;
they are not comparable with a model of the lifetimes alone.

The time-varying covariate fit, by contrast, with each pump's last reading
carried forward (each reading a step of its ``time_varying_covariates``):

.. jupyter-execute::

    from surpyval import WeibullPH

    centre = model.marker_center
    last_reading = SurpyvalData.from_items(
        [
            item.clone(
                markers=None,
                time_varying_covariates={
                    age: [value - centre] for age, value in item.markers.items()
                },
            )
            for item in fleet.to_items()
        ]
    )
    naive = WeibullPH.fit(last_reading)
    print(f"last reading carried forward: {naive.params[-1]:.2f}")
    print(f"joint model:                  {model.params[-1]:.2f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(naive.params[-1], 2) == 0.55
    assert round(model.params[-1], 2) == 0.98
    assert round(model.standard_errors()[-1], 2) == 0.10
    assert abs(model.params[-1] - 0.9) < model.standard_errors()[-1]
    assert model.maximum == "verified"
    est = dict(zip(model.parameter_names, model.params))
    assert round(est["marker_intercept"], 1) == 3.0
    assert round(est["marker_slope"], 5) == 0.00273
    assert round(est["sd_noise"], 1) == 0.7

The last-reading fit puts the marker's effect at 0.55, 39% below the 0.9
it was simulated with; the joint model's 0.98 is within its standard error
(0.10) of it. The issue's simulation study (#891) found the same over 300
such fleets: the last reading carried forward was biased by -39%, its 95%
intervals covering the truth 0.3% of the time; the joint model by +1%,
covering it 96% of the time.

Covariates and late entry
~~~~~~~~~~~~~~~~~~~~~~~~~

A record's ``covariates`` enter both parts: the marker's line
(``marker_<name>``, a shift of its level) and the hazard (``<name>``, a log
hazard ratio at a given marker). Named covariates (``covariates={"site":
1}``) name the parameters. Items that came under observation late -- a
fleet whose records start when monitoring was installed -- are
``left_truncation``-ed, and the fit accounts for having seen only the ones
that reached that age.

Prediction: survival given the readings so far
----------------------------------------------

The model predicts for an item, given as a record: still running at its
``right_censored`` age, with its readings so far and its covariates.
``model.sf(x, item)`` is its survival to each age ``x``, given its readings
and that it was running at that age (1 at or before it), over the lines its
readings leave possible; ``ff`` is its failure probability, and ``cb`` bounds
either from the parameters' uncertainty (the delta method on
:math:`\log(-\log S)`, with the parameters' covariance). Two pumps a year
old, one whose vibration has stayed low and one whose has risen:

.. jupyter-execute::

    quiet = ItemLifetime(
        "Q", right_censored=365, markers={0: 3.0, 120: 3.1, 240: 3.0, 360: 3.2}
    )
    rising = ItemLifetime(
        "R", right_censored=365, markers={0: 3.0, 120: 3.6, 240: 4.3, 360: 4.9}
    )
    for item in (quiet, rising):
        print(item.id, model.ff([425, 545], item).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert model.ff(545, rising) > 2 * model.ff(545, quiet)

The rising pump is more than twice as likely to fail in the next six months.
Bounds, and a new reading as it arrives (the prediction is updated by
asking again with the updated record):

.. jupyter-execute::

    print(model.cb([425, 545], rising, on="ff").round(3))
    rising_later = rising.extend(395).update_markers(395, 5.6)
    print(model.ff(545, rising_later).round(3))

.. jupyter-execute::

    import matplotlib.pyplot as plt

    x = np.linspace(365, 900, 100)
    fig, ax = plt.subplots()
    for item in (quiet, rising):
        band = model.cb(x, item)
        ax.plot(x, model.sf(x, item), label=f"pump {item.id}")
        ax.fill_between(x, band[:, 0], band[:, 1], alpha=0.2)
    ax.set_xlabel("age (days)")
    ax.set_ylabel("survival, given the readings to day 365")
    ax.legend();

``sf`` keeps the package's shapes (a scalar ``x`` gives a scalar, an array an
array of its shape, and a two-sided ``cb`` adds a last ``[lower, upper]``
axis). Far beyond the data the survival can level off above 0: the lines
are straight, and one whose slope is negative takes the hazard towards 0, as
the model says (a reason to read predictions far beyond the data with care).

Saving and loading
------------------

The model saves to strict JSON and loads back, predicting and bounding as
before:

.. jupyter-execute::

    import surpyval

    restored = surpyval.from_json(model.to_json())
    restored.sf(545, rising) == model.sf(545, rising)

What v1 does and does not do
----------------------------

The joint model of v1 (#891) is the standard one. It has:

- one marker, a straight line in age with a random intercept and slope;
- a Weibull proportional hazards survival part in the line's current value
  (``JointPH(Weibull)``, which is ``WeibullJointPH``; other baselines are
  refused);
- covariates in both parts;
- failures and right censoring, with left truncation; left or interval
  censored items and right truncation are refused;
- predictions for one item given its readings, with Wald (delta method)
  bounds.

Not yet (see #891's roadmap): curved (spline) paths, which real late-life
rises often need -- a straight line fitted to a curve biases the joint
model too; the slope or the cumulative marker as the link; other survival
parts (AFT, splines); several markers; ``fit_from_df``.

References
----------

- Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event
  Data, with Applications in R*. Chapman & Hall/CRC.
- Wulfsohn, M. S. and Tsiatis, A. A. (1997). A joint model for survival and
  longitudinal data measured with error. *Biometrics*, 53, 330--339.
- Tsiatis, A. A. and Davidian, M. (2004). Joint modeling of longitudinal and
  time-to-event data: an overview. *Statistica Sinica*, 14, 809--834.

API reference
-------------

``WeibullJointPH`` is ``JointPH(Weibull)``, an instance of the fitter class
below; import everything from ``surpyval`` (or
``surpyval.univariate.joint``).

.. autofunction:: surpyval.univariate.joint.joint_ph.JointPH

.. autoclass:: surpyval.univariate.joint.joint_ph.JointPHFitter
   :members: fit

.. autoclass:: surpyval.univariate.joint.joint_model.JointModel
   :members: sf, ff, cb, summary, covariance, standard_errors, to_dict, from_dict
