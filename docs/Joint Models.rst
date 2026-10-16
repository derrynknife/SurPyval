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

- **the marker** follows a degradation path model -- a straight line in
  age for ``WeibullJointPH``, or any of the degradation module's paths
  (``QuadraticPath``, ``ExponentialPath``, ...) or one of your own -- with
  parameters of the item's own (drawn, across items, from a multivariate
  normal), read with normal noise;
- **the hazard** is Weibull proportional hazards in the path's *current,
  noise-free* value, and the item's covariates, which also shift its path.

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
covariates :math:`z_i`, and fails or is censored at :math:`T_i`. Its marker
follows the path model :math:`m(t; \theta)` (``path(t, *theta)``; for
``WeibullJointPH`` the line :math:`a + b t`) with parameters
:math:`\theta_i` of its own, read with noise:

.. math::

    y_{ij} = m(t_{ij}; \theta_i) + \varepsilon_{ij}, \qquad
    \theta_{iV} \sim N(\mu_V + \Delta_V z_i, D), \qquad
    \theta_{iC} = \mu_C + \Delta_C z_i,

the parameters that vary between items (:math:`V`, all of them by default)
multivariate normal about means the covariates shift, the others
(:math:`C`) common to every item, and :math:`\varepsilon_{ij} \sim N(0,
\sigma^2)`. The hazard is

.. math::

    h_i(t) = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{\beta-1}
    \exp\left(\gamma' z_i + \zeta\,\bigl(m(t; \theta_i) - r\bigr)\right).

:math:`\zeta` (``marker_effect``) is the change in the log hazard per unit of
the marker: ``exp(marker_effect)`` is the hazard ratio of one unit. The
marker enters relative to :math:`r` (``marker_reference``: the mean reading
of the data, unless the fit is given another), so the Weibull :math:`\alpha`
and :math:`\beta` are the baseline of an item with its marker at
:math:`r` and its covariates at 0.

Given its path, an item's readings and lifetime are independent, so its
likelihood is the integral over its varying parameters

.. math::

    L_i = \int p(y_i \mid \theta)\, h_i(T_i \mid \theta)^{d_i}\,
    S_i(T_i \mid \theta)\, \phi(\theta_V; \mu_V + \Delta_V z_i, D)\,
    d\theta_V,

:math:`d_i` 1 for a failure and 0 for a censoring. The fit maximises the sum
of the :math:`\log L_i`, each integral by adaptive Gauss--Hermite
quadrature: a product grid of 7 points a dimension (49 for two varying
parameters, 343 for three, the most that may vary), centred at each item's
most likely parameters and scaled by how sure its readings make them (the
Gauss--Newton curvature, from the path's Jacobian: exact for a path linear
in its parameters). The nodes are held fixed while the parameters are
searched (with autograd's gradient, in the parameters' standard errors),
and a step is taken only as far as the likelihood with the nodes placed
again at the new point improves; this repeats until the log-likelihood
stops moving. A common path parameter is searched through the path model
itself (its Jacobian, and differences of it for the Hessian). The
cumulative hazard along a path is by 15-point Gauss--Legendre quadrature.
The fit starts from the path fitted to each item alone (the path model's
own ``fit``; items with too few readings for it are left out of this
step), the spread of those fits about the covariates' means, and a
survival fit with each item's predicted path plugged in -- a two-stage
estimate, which is biased, and only a start. The covariance is the inverse
of the Hessian of the negative log-likelihood, by autograd.

An item that came under observation at age :math:`L_i > 0`
(``left_truncation``) is in the data because it reached :math:`L_i`: its
likelihood is divided by the probability of that, :math:`\int S(L_i \mid
\theta) \phi(\theta_V)\, d\theta_V`, over the population's paths.

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

The path recovers the simulation (a line from ``a`` = 3 mm/s, rising ``b``
= 1/365 = 0.00274 mm/s a day, the pumps' levels spread by ``sd_a`` = 0.6 and
rates by ``sd_b`` = 0.5/365, noise ``sd_noise`` = 0.7), and the survival part
a marker effect of about 0.9. The names come from the parts: the path
model's own parameter names (``a``, ``b``), ``sd_<name>`` and
``corr_<name>_<name>`` for their spread between items, ``sd_noise``, the
Weibull ``alpha`` and ``beta`` (the heading says at which marker value
``alpha`` is), each covariate's coefficient in the hazard (``<cov>``) and
shifts of the path means (``a_<cov>``, ``b_<cov>``), and
``marker_effect``. The data can also be given as arrays, as for every
fitter -- ``WeibullJointPH.fit(x, c=c, Z=Z, markers=[{age: value, ...},
...])``, one ``{age: value}`` per item -- or as a DataFrame of one row per
item, ``fit_from_df(df, x_col=..., markers_col=..., c_col=...,
Z_cols=[...])``, its ``markers_col`` holding each item's ``{age: value}``.
The model has ``params`` (named by ``parameter_names``), ``covariance()``,
``standard_errors()``, ``neg_ll()``, ``aic()``, ``bic()`` and
``summary()``, as every model does:

.. jupyter-execute::

    model.summary()[["coef", "se(coef)", "exp(coef)"]].round(4)

``neg_ll()`` is the negative log-likelihood of the readings and the lifetimes
together, so ``aic()`` and ``bic()`` compare joint models of the same data;
they are not comparable with a model of the lifetimes alone.

The time-varying covariate fit, by contrast, with each pump's last reading
carried forward (each reading a step of its ``time_varying_covariates``):

.. jupyter-execute::

    from surpyval import WeibullPH

    centre = model.marker_reference
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
    assert round(est["a"], 1) == 3.0
    assert round(est["b"], 5) == 0.00273
    assert round(est["sd_noise"], 1) == 0.7
    assert round(est["sd_a"], 1) == 0.6

The last-reading fit puts the marker's effect at 0.55, 39% below the 0.9
it was simulated with; the joint model's 0.98 is within its standard error
(0.10) of it. The issue's simulation study (#891) found the same over 300
such fleets: the last reading carried forward was biased by -39%, its 95%
intervals covering the truth 0.3% of the time; the joint model by +1%,
covering it 96% of the time.

Covariates and late entry
~~~~~~~~~~~~~~~~~~~~~~~~~

A record's ``covariates`` enter both parts: each path parameter's mean
(``a_<name>``, ``b_<name>``: a shift of the level, of the rate) and the
hazard (``<name>``, a log hazard ratio at a given marker). Named covariates
(``covariates={"site": 1}``) name the parameters; a covariate the data
cannot determine (constant, or a combination of others) is aliased, its
coefficients ``nan``, with one warning. Items that came under observation
late -- a fleet whose records start when monitoring was installed -- are
``left_truncation``-ed, and the fit accounts for having seen only the ones
that reached that age.

The marker reference
~~~~~~~~~~~~~~~~~~~~

``alpha`` is the baseline of an item whose marker is at
``marker_reference``, by default the mean reading (3.66 mm/s here). A
round value reads more easily: ``fit(..., marker_reference=4.0)`` reports
``alpha`` for an item at 4 mm/s. Nothing else moves -- the other
parameters, the likelihood and every prediction are the same:

.. jupyter-execute::

    at_four = WeibullJointPH.fit(fleet, unit="days", marker_reference=4.0)
    print(round(model.marker_reference, 3), round(model.params[6], 1))
    print(at_four.marker_reference, round(at_four.params[6], 1))
    at_four.summary().index[6]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(np.delete(at_four.params, 6), np.delete(model.params, 6))
    assert at_four.parameter_names[6] == "alpha"

Other paths
~~~~~~~~~~~

``JointPH(Weibull, path_model)`` takes any of the degradation module's path
models, or a ``PathModel`` of your own; ``varying=`` names the path
parameters that vary between items (the others are common to all). A wear
signal growing exponentially, ``a exp(b t)``, from a level and at a rate of
each item's own:

.. jupyter-execute::

    from surpyval import JointPH, Weibull
    from surpyval.degradation import ExponentialPath, QuadraticPath


    def wearing(k):
        a, b = rng.normal(3.0, 0.4), rng.normal(0.0008, 0.0003)
        t = np.linspace(0, 3000, 30001)
        h = (2 / 800) * (t / 800) * np.exp(0.9 * (a * np.exp(b * t) - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(rng.exponential(), H, t)
        end = min(life, 730.0)
        ages = np.arange(0.0, end, 30.0)
        y = a * np.exp(b * ages) + rng.normal(0, 0.3, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, y)))


    wear = SurpyvalData.from_items([wearing(k) for k in range(200)])
    exponential = JointPH(Weibull, ExponentialPath).fit(wear, marker_reference=4.0)
    exponential.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert exponential.maximum == "verified"
    z = (exponential.params[-1] - 0.9) / exponential.standard_errors()[-1]
    assert abs(z) < 3

A path's parameters can be held common: ``JointPH(Weibull, QuadraticPath,
varying=("a", "b"))`` gives each item its own level and rate, and every item
the same curvature ``c``. At most three may vary (the quadrature's grid is
7 points in each). A path model fitted to a curve it cannot follow biases
the joint model, as it does a degradation model: choose it as you would
there (see :doc:`Degradation Analysis`).

Prediction: survival given the readings so far
----------------------------------------------

The model predicts for one item: ``model.sf(x, Z, markers=..., given=...)``
is its survival to each age ``x`` given its covariates ``Z``, its readings
so far ``markers`` (``{age: value}``) and that it was still running at age
``given`` (1 at or before it), over the paths its readings leave possible;
``ff`` is its failure probability, and ``cb`` bounds either from the
parameters' uncertainty (the delta method on :math:`\log(-\log S)`, with
the parameters' covariance). With no readings and no ``given`` it is a new
item's survival, averaged over the population's paths. An item record
gives all three at once: ``model.sf(x, item)`` reads its covariates, its
``markers`` and its ``right_censored`` age as ``given``, so the two forms
are the same prediction. Two pumps a year old, one whose vibration has
stayed low and one whose has risen:

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
    assert model.ff(545, rising) == model.ff(
        545, markers=rising.markers, given=365
    )

The same without the records, and for a new pump:

.. jupyter-execute::

    print(model.ff([425, 545], markers=rising.markers, given=365).round(3))
    print(model.ff([365, 730]).round(3))  # a new pump, no readings yet

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
the model says (a reason to read predictions far beyond the data with care;
past about ten times the data's ages the levelled-off value moves by the
quadrature's error, about 1e-5 of itself).

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

- one marker, following any degradation path model (a straight line by
  default), with up to three of its parameters varying between items
  (multivariate normal on their natural scale) and the rest common;
- a Weibull proportional hazards survival part in the path's current value
  (``JointPH(Weibull, path_model)``; ``WeibullJointPH`` is the line; other
  baselines are refused);
- covariates in both parts (every path parameter's mean, and the hazard);
- failures and right censoring, with left truncation; left or interval
  censored items and right truncation are refused;
- predictions for one item given its readings, with Wald (delta method)
  bounds.

Not yet (see #891's roadmap): spline paths; a log scale for a positive
path parameter's spread (the degradation module's ``links``); the slope or
the cumulative marker as the link; other survival parts (AFT, splines);
several markers.

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

``WeibullJointPH`` is ``JointPH(Weibull, LinearPath)``, an instance of the
fitter class below; import everything from ``surpyval`` (or
``surpyval.univariate.joint``).

.. autofunction:: surpyval.univariate.joint.joint_ph.JointPH

.. autoclass:: surpyval.univariate.joint.joint_ph.JointPHFitter
   :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.joint.joint_model.JointModel
   :members: sf, ff, cb, summary, covariance, standard_errors, to_dict, from_dict
