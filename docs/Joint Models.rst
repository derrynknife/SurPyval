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

The hazard need not follow the marker's level. It can follow **how fast the
marker is changing** (its slope), or **the exposure accumulated so far**
(the area under its path), or more than one of these: see `What the hazard
follows: the link`_. And the marker need not multiply the hazard at all:
in ``WeibullJointAFT`` it makes the item **age faster**, the life used up
at a high reading not given back when the reading falls (see `How the
reading acts: a higher failure rate, or faster ageing`_).

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
same line. (A middle way, fitting each item's line first and the lifetimes
on those lines second, is quick but still biased: see `A quick first look:
the two-stage fit`_.) (A third approach, which models no path at all, is
:doc:`landmarking <Landmark Analysis>`: a model of the units still at risk
at a time, by their last readings then.)

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
\sigma^2)`. By default every covariate shifts every path parameter's mean
and enters the hazard; a fit can put each covariate in one part or both
(:math:`\Delta` and :math:`\gamma` are then 0 where a covariate is left
out: see `Covariates in each part`_). The hazard is

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
parameters, 343 for three; see :ref:`how the integral is done
<joint-integration>` for four and five, the most that may vary), centred
at each item's
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
estimate, which is biased (`A quick first look: the two-stage fit`_
returns it as a model of its own). The covariance is the inverse
of the Hessian of the negative log-likelihood, by autograd.

An item found failed at an inspection is known only to have failed between
two ages, :math:`l_i < T_i \le u_i` (``interval_censored``; ``left_censored``
is the interval from its entry, or 0): :math:`h_i(T_i)^{d_i} S_i(T_i)` is
then :math:`S_i(l_i \mid \theta) - S_i(u_i \mid \theta)`, the hazard
integrated over the interval directly (so a narrow interval late in life
loses no precision). An item that came under observation at age
:math:`L_i > 0` (``left_truncation``) is in the data because it reached
:math:`L_i`, and one right truncated at :math:`R_i` (``right_truncation``)
because it failed by then: its likelihood is divided by the probability
of that, :math:`\int (S(L_i \mid \theta) - S(R_i \mid \theta))
\phi(\theta_V)\, d\theta_V`, over the population's paths. The
likelihood of every kind of item agrees with an independent dense-grid
integral over the random effects to within 1e-5 an item
(``tests/univariate/joint/test_joint_extensions.py``).

.. _joint-integration:

How the integral is done
~~~~~~~~~~~~~~~~~~~~~~~~

Each item's likelihood is an average over the path parameters it could
have (its own level, its own rate, ...), and that average is a sum over a
grid of 7 values of each, placed where that item's readings and lifetime
say its parameters are. The grid grows quickly with the number of varying
parameters: 7 nodes for one, 49 for two, 343 for three, 2,401 for four and
16,807 for five, for every item at every step of the search. Up to five
may vary.

``integration=`` on ``fit`` (and ``fit_from_df``) chooses the nodes:

- ``"grid"``: the full grid. The default, whatever the number of varying
  parameters: the more accurate choice.
- ``"pruned"``: the grid without the nodes whose weight is negligible --
  its far corners, where several parameters are extreme at once. The rule
  is fixed: the lowest-weight nodes are left out, as many as hold less
  than 1e-9 of the grid's weight between them (nodes of equal weight
  together), whatever the data. That is none of the nodes for up to three
  parameters, 80 of 2,401 for four and 1,392 of 16,807 for five. An
  option, never the default: it trades a little accuracy for a little
  speed, and in practice saves little (below).

The model says which it used (``model.integration``, and the
``Integration`` line of its printout), keeps it when saved, and predicts
on the same kind of grid. A model saved before the choice existed reads as
``"grid"``.

.. code:: python

    from surpyval import JointPH, Weibull

    fitter = JointPH(Weibull, cubic)            # a, b, c and d all vary
    model = fitter.fit(data)                    # the full grid: 2,401 nodes an item
    model.integration                           # 'grid'
    pruned = fitter.fit(data, integration="pruned")

The pruned grid is a small saving that costs no accuracy: 3% fewer nodes
with four varying parameters and 8% fewer with five (one evaluation of the
likelihood and its gradient is 1% and 6% quicker). Where the integrand is
exactly normal (no marker effect, a path linear in its parameters) both
grids reproduce the closed form in four and five dimensions, the pruned one
short by the weight it leaves out, 3e-10 and 8e-10 of an item's likelihood.
With a marker effect the two log-likelihoods differ by that much again on
items read often, and by up to 7e-7 an item in the hardest case tried (five
varying parameters, a reading every 0.7 years with noise five times the
usual, a marker effect of 2); a fleet of 60 items fitted on each grid gave
estimates within 1e-6 of a standard error of each other
(``tests/univariate/joint/test_joint_integration.py``).

Two things to expect with four or five varying parameters. The fit takes
minutes where two or three take seconds -- the fleet of 60 items took two
to five minutes -- and most of that is the search over the ten or fifteen
variances and correlations of the items' spread, not the grid. And the
7 points themselves are less exact where items have few, noisy readings
and the marker effect is strong: against 11 points a dimension the
log-likelihood of such a fleet was off by up to 5e-5 an item (it was
within 4e-10 an item on a fleet read often). That is the grid's own
error, the same on the full grid and the pruned one. Fewer correlations
(next) make that search smaller.

Which path parameters are correlated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default every pair of varying parameters is correlated between items:
an item that starts high may rise faster, or slower, and the fit
estimates by how much (``corr_a_b``). That is one correlation for two
varying parameters, three for three, six for four and ten for five, and a
fleet of tens of items cannot pin down ten. ``random_effects=`` on the
fitter says which are correlated:

- ``"full"``: every pair (the default; the model as it always was).
- ``"diagonal"``: none. Each varying parameter varies between items
  independently of the others.
- blocks of names, such as ``[("a", "b"), ("c",)]``: those in a block are
  correlated with each other and with none outside it. Every varying
  parameter is in exactly one block; a name that is not a varying
  parameter, or one named twice or left out, is refused, saying which.

.. code:: python

    from surpyval import JointPH, Weibull
    from surpyval.degradation import QuadraticPath

    independent = JointPH(Weibull, random_effects="diagonal").fit(data)
    independent.parameter_names[:4]       # ['a', 'b', 'sd_a', 'sd_b']
    blocks = JointPH(Weibull, QuadraticPath, random_effects=[("a", "b"), ("c",)])

A correlation left out is 0, not estimated: it has no entry in
``params``, in ``summary()`` or in the saved model, and ``aic()`` counts
the parameters that are estimated. The structures nest -- the full one
holds every other with some correlations at 0 -- so they compare on the
same data by ``aic()`` or by the likelihood ratio. The model keeps the
structure as ``random_effects_structure`` (``"full"``, ``"diagonal"``,
or the blocks as a tuple of tuples of names), and its printout and the
summary's heading of the spread say it. A model with a structure other
than the full one saves as schema 5, which SurPyval 0.28 refuses rather
than read its parameters as a full one's; a model saved before the
choice existed reads as ``"full"``.

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

From measurements
~~~~~~~~~~~~~~~~~

The readings and the lifetimes are two things, and they can be kept as
two: the life data, one row per pump, and the readings as a
:class:`~surpyval.Measurements` (see :doc:`Conventions`), built from a
long table of one row per reading. The fit matches the two by the pump's
id, never by position, so the tables need not be in the same order:

.. jupyter-execute::

    import pandas as pd
    from surpyval import Measurements

    readings = pd.DataFrame(
        [
            (pump_id, age, value)
            for pump_id, held in fleet.markers.items()
            for age, value in held.items()
        ],
        columns=["pump", "day", "vibration"],
    ).sample(frac=1.0, random_state=0)
    vibration = Measurements.from_dataframe(
        readings, i_col="pump", x_col="day", m_col="vibration"
    )
    life = fleet.without("markers")
    separate = WeibullJointPH.fit(life, m=vibration, unit="days")
    separate.params[-1], model.params[-1]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.array_equal(separate.params, model.params)

``m`` is the other name of ``markers``; a Measurements may also be given
beside arrays, with each row's unit as ``i``,
``WeibullJointPH.fit(life.x, c=life.c, i=life.i, m=vibration)``, or in a
DataFrame fit, ``fit_from_df(df, x_col=..., c_col=..., i_col="pump",
m=vibration)``. A pump with no readings is fitted on its lifetime alone,
as an item with ``{}`` is; readings of a pump with no life row are warned
of, naming it, and not used. The model takes one marker: readings of
several measurements are refused until one is chosen with
``vibration.with_measurement(name)``.

Covariates in each part
~~~~~~~~~~~~~~~~~~~~~~~

A record's ``covariates`` enter both parts by default: each path
parameter's mean (``a_<name>``, ``b_<name>``: a shift of the level, of the
rate) and the hazard (``<name>``, a log hazard ratio at a given marker).
Named covariates (``covariates={"site": 1}``) name the parameters; a
covariate the data cannot determine (constant, or a combination of others)
is aliased, its coefficients ``nan``, with one warning.

A condition often acts on one part only. A heavy load can make a pump's
vibration rise faster without making it any more likely to fail *at a
given vibration*: it acts through the marker, on the path. A different
site -- another maintenance crew, another duty -- can change the risk at a
given vibration without changing how vibration rises: it acts on the
hazard. ``hazard_covariates=[...]`` names the covariates in the hazard and
``path_covariates=`` those that shift the path, a list (every path
parameter) or ``{path parameter: [...]}`` (here only the rate ``b``); each
defaults to every covariate, ``[]`` is none, and a covariate must be in at
least one. Putting a covariate where it does not act wastes the data on
coefficients that are 0, and putting it only where it does not act biases
the others. A fleet with ``site`` acting on the hazard only and ``load``
on the rate only:

.. jupyter-execute::

    gen = np.random.default_rng(11)


    def monitored(k):
        site, load = gen.integers(0, 2, 2)
        start = gen.normal(3.0, 0.6)
        rate = (gen.normal(1.0, 0.5) + 0.6 * load) / 365  # load: faster rise
        t = np.linspace(0, 3000, 30001)
        h = (2 / 800) * (t / 800) * np.exp(
            0.5 * site + 0.9 * (start + rate * t - 4.0)  # site: more risk
        )
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(gen.exponential(), H, t)
        end = min(life, 730.0)
        ages = np.arange(0.0, end, 30.0)
        readings = start + rate * ages + gen.normal(0, 0.7, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(
            f"M{k}",
            **{outcome: round(end, 1)},
            covariates={"site": float(site), "load": float(load)},
            markers=dict(zip(ages, readings.round(2))),
        ), life


    monitored_fleet = [monitored(k) for k in range(300)]
    duty = SurpyvalData.from_items([item for item, _ in monitored_fleet])
    split = WeibullJointPH.fit(
        duty,
        hazard_covariates=["site"],
        path_covariates={"b": ["load"]},
        marker_reference=4.0,
    )
    split.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert split.maximum == "verified"
    assert split.parameter_names == [
        "a", "b", "b_load", "sd_a", "sd_b", "corr_a_b", "sd_noise",
        "alpha", "beta", "site", "marker_effect",
    ]
    est = dict(zip(split.parameter_names, split.params))
    se = dict(zip(split.parameter_names, split.standard_errors()))
    for name, truth in [("site", 0.5), ("b_load", 0.6 / 365),
                        ("marker_effect", 0.9)]:
        assert abs(est[name] - truth) < 3 * se[name], name

``b_load`` is the extra rise a day under load (0.6/365 = 0.0016 simulated)
and ``site`` the log hazard ratio of the second site at a given vibration
(0.5); there is no ``a_load``, ``b_site``, ``a_site`` or ``load``. Every
covariate is still needed to predict (``Z`` names them all), and the model
saves which part each is in. A model fitted, or saved, before the parts
existed has every covariate in both.

Late entry
~~~~~~~~~~

Items that came under observation late -- a fleet whose records start when
monitoring was installed -- are ``left_truncation``-ed, and the fit
accounts for having seen only the ones that reached that age; items in
the data only because they failed by an age (a recall of failed units, say)
are ``right_truncation``-ed, likewise.

Found failed at an inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where a failure is only found at the next inspection, the item failed
between its last two inspections: an ``interval_censored`` record
``[last seen running, found failed]``, or ``left_censored`` if it was found
failed at the first. The same pumps, inspected (and read) every 90 days:

.. jupyter-execute::

    def inspected(item, life, every=90.0):
        stops = np.arange(every, 730.0 + 1e-9, every)
        found = stops[stops >= life]
        if not found.size:
            return item.clone(right_censored=stops[-1], markers={
                a: v for a, v in item.markers.items() if a <= stops[-1]
            })
        last = found[0] - every
        readings = {a: v for a, v in item.markers.items() if a <= last}
        if last <= 0:
            return item.clone(left_censored=found[0], markers=readings)
        return item.clone(interval_censored=[last, found[0]], markers=readings)


    found = SurpyvalData.from_items([inspected(*pair) for pair in monitored_fleet])
    print((found.c == 2).sum(), "interval and", (found.c == -1).sum(), "left censored")
    from_inspections = WeibullJointPH.fit(
        found,
        hazard_covariates=["site"],
        path_covariates={"b": ["load"]},
        marker_reference=4.0,
    )
    from_inspections.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert from_inspections.maximum == "verified"
    est = dict(zip(from_inspections.parameter_names, from_inspections.params))
    se = dict(
        zip(from_inspections.parameter_names, from_inspections.standard_errors())
    )
    for name, truth in [("site", 0.5), ("b_load", 0.6 / 365),
                        ("marker_effect", 0.9)]:
        assert abs(est[name] - truth) < 3 * se[name], name

The estimates are close to the exact data's, the standard errors of the
survival part a little wider: the inspections lose when, within 90 days,
each pump failed. With arrays, an interval is a ``[lower, upper]`` row of
``x`` with ``c = 2`` (and ``fit_from_df`` takes ``xl_col`` and ``xr_col``).

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

Each parameter is on its own scale by default: the rate ``b`` above is
normal between items, so a fleet whose rates are near 0 would give some
items a falling path. A rate that must stay positive can go on a log
scale with the degradation module's ``LinkedPathModel``: then ``log(b)``
is normal between items (``b`` log-normal), and the names say so --
``log(b)``, its spread ``sd_log(b)`` and correlation ``corr_a_log(b)``
are on the log scale, and a covariate's shift ``log(b)_<cov>`` is a log
ratio of rates:

.. jupyter-execute::

    from surpyval.degradation import LinkedPathModel

    log_rate = JointPH(Weibull, LinkedPathModel(ExponentialPath, {"b": "log"}))
    log_rate = log_rate.fit(wear, marker_reference=4.0)
    log_rate.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert log_rate.maximum == "verified"
    assert log_rate.parameter_names[:5] == [
        "a", "log(b)", "sd_a", "sd_log(b)", "corr_a_log(b)"
    ]
    z = (log_rate.params[-1] - 0.9) / log_rate.standard_errors()[-1]
    assert abs(z) < 3

``varying=`` then names ``log(b)``. The two fits describe the same fleet
with a different spread of rates, and predict alike where the data are.

What the hazard follows: the link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far the risk has followed the marker's *level*: a pump at 5 mm/s is at
more risk than one at 3 mm/s, however each got there. That is one of three
things a marker can say about risk, and ``JointPH(..., link=...)`` chooses
between them:

- ``link="value"`` (the default): **risk follows the level.** What matters
  is where the marker is now. A vibration level that is a symptom of the
  present damage; a biomarker that measures how ill the patient is today.
- ``link="slope"``: **risk follows how fast it is changing.** What matters
  is the marker's rate of rise, not where it stands. Bearings that settle
  at different vibration levels by their mounting, where a level that is
  *climbing* is the sign of trouble; a tumour marker whose doubling time,
  not its level, says how aggressive the disease is.
- ``link="cumulative"``: **risk follows the accumulated exposure.** What
  matters is the total so far, the area under the marker's path from age
  0: every hour at a high reading adds to the damage, and it does not go
  away when the reading falls. A load, temperature or contamination
  reading where the wear done is the dose received; a patient's cumulative
  exposure to a raised blood pressure or blood sugar.

A list gives the hazard more than one, each with an effect of its own:
``link=["value", "slope"]`` asks whether, at a given level, a marker that
is rising is at more risk than one that is steady.

In formulas, with :math:`m_i(t) = m(t; \theta_i)` the item's noise-free
path, the hazard is

.. math::

    h_i(t) = h_0(t) \exp\left(\gamma' z_i
    + \zeta_v \bigl(m_i(t) - r\bigr)
    + \zeta_s\, m_i'(t)
    + \zeta_c \int_0^t m_i(s)\, ds\right),

with only the terms of the links chosen: ``marker_effect``
(:math:`\zeta_v`) per unit of the marker, ``slope_effect``
(:math:`\zeta_s`) per unit of the marker *a unit of time*, and
``cumulative_effect`` (:math:`\zeta_c`) per unit of the marker *times a
unit of time*. The slope is the path model's own derivative (``slope``;
exact for the built-in paths, central differences for a path of one's
own without one), and the area is by 15-point Gauss--Legendre quadrature
(exact for a line or a quadratic). Everything else -- the data, the
quadrature over each item's path, the predictions, the bounds, saving --
is as before.

Three things to know:

- **The baseline.** ``alpha`` and ``beta`` are the Weibull of an item whose
  slope is 0 and whose accumulated area is 0 (the printout says so:
  ``alpha at slope = 0``). ``marker_reference`` is the value's reference
  only, and is refused where the hazard does not follow the value. The
  area is of the marker itself, from age 0: for the exposure *above* a
  level -- the hours above a rated temperature, say -- subtract that level
  from the readings first.
- **Paths whose slope starts infinite.** The cumulative hazard runs from
  age 0, so the slope link needs a path whose slope is finite there. A
  logarithmic path (slope :math:`b / t`) and a Lloyd-Lipow path (slope
  :math:`b / t^2`) are refused with the slope link when the fitter is
  built, and a power path :math:`a t^b` is refused by the fit when its
  average item's power is below 1 (slope :math:`a b t^{b - 1}`): near age
  0 the failure rate would be infinite for every item whose slope has the
  sign of its effect. Each refusal names the links the path can take
  instead, and paths whose slope is finite at 0 (a line, a quadratic, an
  exponential).
- **The units.** A slope's and an area's effects depend on the unit of
  age as well as the marker's: a ``slope_effect`` per mm/s a *day* is 365
  times one per mm/s a *year*, and a ``cumulative_effect`` per
  (mm/s)·day is 365 times one per (mm/s)·year. Ages in years (or a rate
  per year) keep them readable. Give both units -- ``unit=`` for the ages
  and ``marker_unit=`` for the readings (a ``Measurements`` read with
  ``units=`` gives its own) -- and the model says what each effect is per,
  in ``effect_units`` and on its printout's ``Effect units`` line. They
  are labels only: nothing is converted.
- **What the data can tell apart.** For a straight line the slope is the
  item's rate ``b``, constant in age: its effect is found from how
  lifetimes differ *between* items with different rates, so it needs
  rates that vary between items (``b`` in ``varying``, or shifted by a
  covariate). The links are compared by ``aic()``, the models having the
  same data, or all at once by ``compare_links`` (see `Comparing the links`_).

A fleet of gearboxes, their oil debris read monthly for two years (ages
in years). Each starts near 3 ppm and rises about 1 ppm a year, at a level
and rate of its own; what wears a gearbox out here is how fast the debris
is rising, each ppm a year multiplying the hazard by :math:`e^{1.0} = 2.7`:

.. jupyter-execute::

    gen = np.random.default_rng(5)


    def gearboxes(n, alpha, beta, slope=0.0, cumulative=0.0):
        spread = np.array([[0.36, 0.06], [0.06, 0.25]])  # sds 0.6, 0.5
        lines = gen.multivariate_normal([3.0, 1.0], spread, size=n)
        target = gen.exponential(size=n)
        t = np.linspace(0.0, 40.0, 400001)
        items = []
        for k, (level, rate) in enumerate(lines):
            log_ratio = slope * rate + cumulative * (level * t + rate * t**2 / 2)
            h = beta / alpha * (t / alpha) ** (beta - 1) * np.exp(log_ratio)
            H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
            life = np.interp(target[k], H, t)
            end = min(life, 2.0)                      # followed for two years
            ages = np.arange(0.0, end + 1e-9, 1 / 12)  # read monthly
            readings = level + rate * ages + gen.normal(0, 0.7, ages.size)
            outcome = "observed" if life <= 2.0 else "right_censored"
            items.append(
                ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, readings)))
            )
        return SurpyvalData.from_items(items)


    rising = gearboxes(300, alpha=3.0, beta=2.0, slope=1.0)
    by_slope = JointPH(Weibull, link="slope").fit(rising, unit="years")
    by_slope.summary()[["coef", "se(coef)", "exp(coef)"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_slope.maximum == "verified"
    assert by_slope.parameter_names[-1] == "slope_effect"
    z = (by_slope.params[-1] - 1.0) / by_slope.standard_errors()[-1]
    assert abs(z) < 3
    assert round(by_slope.params[-1], 2) == 0.96

The ``slope_effect`` recovers the 1.0 it was simulated with. The model
that follows the level instead fits the same fleet worse:

.. jupyter-execute::

    by_level = JointPH(Weibull).fit(rising, unit="years")
    print(f"AIC, slope link: {by_slope.aic():.1f}")
    print(f"AIC, value link: {by_level.aic():.1f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_slope.aic() < by_level.aic()

Two gearboxes at the same reading now, one steady and one climbing, are
then at different risk:

.. jupyter-execute::

    steady = {0.0: 3.5, 0.3: 3.5, 0.6: 3.6, 0.9: 3.5}
    climbing = {0.0: 2.6, 0.3: 3.0, 0.6: 3.3, 0.9: 3.6}
    for name, readings in [("steady", steady), ("climbing", climbing)]:
        print(name, by_slope.ff(1.5, markers=readings, given=0.9).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_slope.ff(1.5, markers=climbing, given=0.9) > by_slope.ff(
        1.5, markers=steady, given=0.9
    )

The same fleet with the damage accumulating instead -- every ppm-year of
debris so far multiplying the hazard by :math:`e^{0.3}`:

.. jupyter-execute::

    worn = gearboxes(300, alpha=4.0, beta=1.5, cumulative=0.3)
    by_area = JointPH(Weibull, link="cumulative").fit(
        worn, unit="years", marker_unit="ppm"
    )
    by_area.summary()[["coef", "se(coef)", "exp(coef)"]].round(3).tail(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_area.maximum == "verified"
    z = (by_area.params[-1] - 0.3) / by_area.standard_errors()[-1]
    assert abs(z) < 3
    assert by_area.effect_units == {"cumulative_effect": "ppm·years"}

With the units given, the model says what the effect is per: each
ppm·year of debris so far multiplies the hazard by ``exp(coef)``.

.. jupyter-execute::

    by_area.effect_units

A medical example: the Mayo Clinic trial in primary biliary cirrhosis
(``load_pbc2``), 312 patients whose serum bilirubin was measured at each
visit, followed to death (a transplant or the end of follow-up is a
censoring). Does a bilirubin that is *rising* say more than its level
does? The log of the bilirubin as a line in years, the hazard following
both its level and its slope:

.. jupyter-execute::

    from surpyval.datasets import load_pbc2

    pbc = load_pbc2()
    pbc["log_bilirubin"] = np.log(pbc["serBilir"])
    patients = pbc.groupby("id").agg(
        years=("years", "first"), died=("status2", "first")
    )
    patients["censored"] = 1 - patients["died"]
    patients["readings"] = [
        dict(zip(visits["year"], visits["log_bilirubin"]))
        for _, visits in pbc.groupby("id")
    ]
    level_and_slope = JointPH(Weibull, link=["value", "slope"]).fit_from_df(
        patients,
        x_col="years",
        c_col="censored",
        markers_col="readings",
        marker_reference=0.0,
        unit="years",
    )
    level_and_slope.summary()[["coef", "se(coef)", "exp(coef)", "p"]].round(3).tail(4)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert level_and_slope.maximum == "verified"
    est = dict(zip(level_and_slope.parameter_names, level_and_slope.params))
    se = dict(
        zip(level_and_slope.parameter_names, level_and_slope.standard_errors())
    )
    assert round(est["marker_effect"], 2) == 1.03
    assert round(est["slope_effect"], 1) == 3.0
    assert est["slope_effect"] > 2.5 * se["slope_effect"]

Both matter. A patient whose log bilirubin is one unit higher is at about
2.8 times the hazard, and at the same level one whose log bilirubin is
rising 0.1 a year faster is at :math:`e^{0.30} = 1.3` times the hazard.
R's ``JM`` (``jointModel`` with ``parameterization = "both"``) gives the
same estimates on these data, as it does for the value, the slope and the
cumulative links alone: the comparison, with the treatment in the hazard,
is ``tests/reference/test_joint_links.py``.

Comparing the links
~~~~~~~~~~~~~~~~~~~

Which link a marker's risk follows is a question about each dataset, and
the data answer it by comparing fits. A fitter's ``compare_links`` fits
the same model -- its baseline, path model and varying parameters -- with
each link to the same data, in one call. By default it fits five: the
value, the slope and the area alone, and the slope and the area each beside
the value (``links=[...]`` chooses others, each a link or a list of them).
On the gearboxes whose risk follows how fast the debris rises:

.. jupyter-execute::

    links = JointPH(Weibull).compare_links(rising, unit="years")
    links.table[["k", "log_likelihood", "aic", "delta", "weight"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert links.best_link == ("slope",)
    assert list(links.table.index[:2]) == ["slope", "value + slope"]
    assert links.table["weight"].iloc[0] > 0.6

The table is ranked by AIC (``metric="bic"`` or ``"aic_c"`` for another
criterion), best first. AIC is :math:`2k - 2 \log L`, with :math:`L` the
likelihood of the readings and the lifetimes together: the links' AICs
compare with each other, not with that of a model of the lifetimes alone.
``delta`` is each link's AIC less the best's and ``weight`` its Akaike
weight, its share of :math:`e^{-\Delta/2}`: the slope link has about two
thirds and the value with the slope the rest, the links without the slope
almost none. ``links.best`` is the best fit, and
``links.models["value + slope"]`` any other.

Two links less than 2 apart are no real preference, and the simpler of
them is the best: the one with fewer parameters, and on a tie the level
before the slope before the area. So a link one effect richer than another
is the best only where it is more than 2 lower, which, by AIC, is where the
likelihood-ratio test of the extra effect rejects it at the 4.55% level; a
``delta`` can be below 0 for a richer link the data do not prefer. The
comparison says the same in words, row by row (``reads_as``: "as good: no
real preference" within 2, "worse" from 2 to 10, "clearly worse" beyond,
after Burnham and Anderson), and as a whole:

.. jupyter-execute::

    print(links.verdict)
    links.table[["delta", "lr_against", "lr_p_value", "reads_as"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert links.verdict.startswith(
        "Risk follows how fast the marker is changing. Adding the level "
        "does not improve the fit."
    )

``lr_against`` and ``lr_p_value`` give, for a link with another nested in
it, the likelihood-ratio test of its extra effects against that one (the
best, where the best is nested in it). ``links.to_dict()`` holds the
verdict, the table and the tests as strict JSON; save a fitted model with
its own ``to_dict``, as ``links.best.to_dict()``.

A link whose terms are all in another's is that model with the extra
effects at 0, so the two are also compared by a likelihood-ratio test:
twice the gain in log-likelihood, on one degree of freedom an extra effect.

.. jupyter-execute::

    links.likelihood_ratio_tests().round(4)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    tests = links.likelihood_ratio_tests().set_index(["simpler", "richer"])
    assert tests.loc[("value", "value + slope"), "p_value"] < 0.01
    assert tests.loc[("slope", "value + slope"), "p_value"] > 0.05

The slope adds to the value (p = 0.0002) and the value adds nothing to the
slope (p = 0.37): risk here follows how fast the debris is rising, not
where it stands. Links that are not nested, such as the value and the
slope, are compared by the table only.

Each candidate is fitted as its own ``fit`` would fit it, to the same
maximum. The value link is fitted first, and the others start from its
estimates of the path (the path changes little from link to link), which
saves about a third of the time; a fit from that start that does not reach
a verified maximum is made again from its own start. A fit that is not a verified maximum is shown in the table as
``set aside`` and is neither ranked nor tested, and a link that cannot be
fitted is listed with the reason (``links.failed``); each is said in one
warning. A path that cannot take some of the default links is compared
without them, saying which: a logarithmic path without the slope, and a
Lloyd-Lipow path, which also has no finite area from age 0, can take only
the value, so there is nothing to compare and it is refused.

A reading that acts later: a lag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every link so far acts at once: the hazard at age :math:`t` follows the
path at :math:`t`. Some mechanisms have a lead time -- debris in the oil
now says what the bearing will do some hundreds of hours on. ``lag=``
shifts every link by a fixed time, in the lifetimes' unit:
``JointPH(Weibull, link="value", lag=200.0)`` makes the hazard at age
:math:`t` follow the path at :math:`\max(t - 200, 0)`. Until the lag
has passed the path is held at its age-0 value: the value link follows
the value at age 0, and the slope and the area are 0 there. The
cumulative hazard is integrated on each side of the lag (where the
lagged path has a kink, and the slope link a jump), so the likelihood
keeps its accuracy. ``JointAFT`` takes ``lag=`` the same way. With
``lag=0`` (the default) the model is the one without a lag, number for
number. The fitted model keeps its ``lag``, prints it, saves it and
predicts with it.

The lag is fixed, not estimated: the likelihood has a kink in it at every
item's age, so it is not smooth in the lag. Choose it from what you know
of the mechanism, or fit the lags you have in mind and compare them by
AIC (the same data and the same number of parameters):

.. code-block:: python

    fits = {lag: JointPH(Weibull, lag=lag).fit(data) for lag in (0.0, 0.25, 0.5)}
    {lag: model.aic() for lag, model in fits.items()}

Expect the data to say little about a lag when the marker is close to a
straight line: the line ``L`` earlier is the line now less ``b L``, which
the marker's effect and the baseline largely absorb. On 300 simulated
items whose hazard followed the line half a year earlier, the AICs of
lags 0, 0.25 and 0.5 were within 0.6 of each other. A path that is
infinite at age 0 (logarithmic, Lloyd-Lipow) cannot lag its value, and is
refused.

A path's parameters can be held common: ``JointPH(Weibull, QuadraticPath,
varying=("a", "b"))`` gives each item its own level and rate, and every item
the same curvature ``c``. At most five may vary (the quadrature's grid is
7 points in each, the full grid by default: see :ref:`how the
integral is done <joint-integration>`). A path model fitted to a curve it cannot follow biases
the joint model, as it does a degradation model: choose it as you would
there (see :doc:`Degradation Analysis`).

How the reading acts: a higher failure rate, or faster ageing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model so far is a proportional hazards model (``JointPH``): a high
reading multiplies the failure rate for as long as it is high, and when the
reading comes back down the rate comes down with it, as though the high
stretch had never happened. ``JointAFT`` (``WeibullJointAFT`` for a
line) is the other way a reading can act: it makes the item **age
faster**. A month at a high reading uses up more than a month of life, and
the life used up is not given back when the reading falls. It is the
cumulative-exposure model of step-stress accelerated life tests, with the
measured reading as the stress, so its effects mean what they mean in an
accelerated test of the same product. It suits a reading of a stress that
consumes life -- a load, a temperature, a speed -- and a biomarker that
hastens the course of a disease; an effect reads as "a life so many times
shorter", which many find easier than a hazard ratio. (It is a statistical model of the readings and the
lifetimes, not a model of the physics of the damage.)

By age :math:`t` the item has used up the *effective age*

.. math::

    A_i(t) = \int_0^t \exp\left(\gamma' z_i
    + \zeta_v \bigl(m_i(s) - r\bigr) + \zeta_s\, m_i'(s)\right) ds,

and its survival and hazard are the baseline Weibull's at that age:

.. math::

    S_i(t) = \exp\left(-\left(\frac{A_i(t)}{\alpha}\right)^{\beta}
    \right), \qquad
    h_i(t) = \frac{\beta}{\alpha}\left(\frac{A_i(t)}{\alpha}
    \right)^{\beta - 1} A_i'(t).

A worked example, with made-up numbers. Say each ppm of oil debris above 3
ppm makes a gearbox age :math:`e^{0.3}` times as fast (``marker_effect``
0.3, ``marker_reference`` 3), and the baseline is a Weibull of scale 6
years and shape 2. At 3 ppm a gearbox ages a year a year; at 5 ppm it ages
:math:`e^{0.3 \times 2} = 1.82` years a year. One that ran two years at 3
ppm and then a year at 5 ppm has an effective age of :math:`2 + 1.82 =
3.82` years at the age of three, and a survival of :math:`\exp(-(3.82 /
6)^2) = 0.67`, against 0.78 for one that stayed at 3 ppm. If its debris
then falls back to 3 ppm, the proportional hazards model returns it to the
failure rate of a three-year-old (0.17 a year); the accelerated model keeps
it a 3.82-year-old (0.21 a year).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _A = 2 + np.exp(0.3 * 2)
    assert round(np.exp(0.6), 2) == 1.82 and round(_A, 2) == 3.82
    assert round(float(Weibull.sf(_A, 6.0, 2.0)), 2) == 0.67
    assert round(float(Weibull.sf(3.0, 6.0, 2.0)), 2) == 0.78
    assert round(float(Weibull.hf(3.0, 6.0, 2.0)), 2) == 0.17
    assert round(float(Weibull.hf(_A, 6.0, 2.0)), 2) == 0.21

The fitter is built and used as ``JointPH``'s is: the same data, path
models, ``varying``, ``hazard_covariates`` and ``path_covariates``,
``marker_reference``, ``fit``, ``fit_from_df``, predictions and
saving, and the same parameter names. A fleet of 300 gearboxes whose debris
is a line in years, each ageing :math:`e^{0.6}` times as fast for each ppm
above 4:

.. jupyter-execute::

    from surpyval import WeibullJointAFT

    gen_aft = np.random.default_rng(29)


    def loaded(n, alpha=2.5, beta=2.0, effect=0.6, centre=4.0):
        spread = np.array([[0.36, 0.06], [0.06, 0.25]])  # sds 0.6, 0.5
        lines = gen_aft.multivariate_normal([3.0, 1.0], spread, size=n)
        items = []
        for k, (level, rate) in enumerate(lines):
            used = alpha * gen_aft.weibull(beta)  # the effective age it reaches
            # a line's effective age: e^(effect (level - centre)) (e^(k t) - 1) / k
            start, speed = np.exp(effect * (level - centre)), effect * rate
            grow = speed * used / start
            life = np.log1p(grow) / speed if grow > -1 else np.inf
            end = min(life, 2.0)                       # followed for two years
            ages = np.arange(0.0, end + 1e-9, 1 / 12)  # read monthly
            readings = level + rate * ages + gen_aft.normal(0, 0.7, ages.size)
            outcome = "observed" if life <= 2.0 else "right_censored"
            items.append(
                ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, readings)))
            )
        return SurpyvalData.from_items(items)


    gearboxes_aft = loaded(300)
    aged = WeibullJointAFT.fit(gearboxes_aft, marker_reference=4.0, unit="years")
    aged.summary()[["coef", "se(coef)", "exp(coef)"]].round(3).tail(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert aged.maximum == "verified" and aged.survival == "AFT"
    _p = dict(zip(aged.parameter_names, aged.params))
    _se = dict(zip(aged.parameter_names, aged.standard_errors()))
    for _name, _truth in [("marker_effect", 0.6), ("alpha", 2.5), ("beta", 2.0)]:
        assert abs(_p[_name] - _truth) < 3 * _se[_name], _name

``exp(coef)`` of ``marker_effect`` is how many times as fast a gearbox
ages for each ppm more (0.62 against the 0.6 simulated); ``alpha`` and
``beta`` are the Weibull of a gearbox held at the reference, 4 ppm. The
two models of the same fleet compare by ``aic()``, as the links do:

.. jupyter-execute::

    multiplied = WeibullJointPH.fit(gearboxes_aft, marker_reference=4.0, unit="years")
    print(f"AIC, faster ageing:        {aged.aic():.1f}")
    print(f"AIC, higher failure rate:  {multiplied.aic():.1f}")
    print(f"marker_effect: {aged.params[-1]:.2f} (ageing), "
          f"{multiplied.params[-1]:.2f} (hazard)")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(aged.params[-1], 2) == 0.62
    assert round(multiplied.params[-1], 2) == 1.13
    assert round(aged.aic() - multiplied.aic()) == 4

Here the data cannot tell the two apart: the proportional hazards model
even has the lower AIC, by 4, though the fleet was simulated from the
accelerated one. A line that only rises gives both models much the same
shape of risk, the hazard model's effect (1.13) taking the place of
roughly :math:`\beta` times the ageing one (:math:`2.05 \times 0.62 =
1.27`).
The two part company where a reading rises and falls, or where the
baseline is not a Weibull; with paths that only rise, choose the model by
what the reading is rather than by a few units of AIC.

Four things to know:

- **The links.** ``link="value"`` (the default) and ``"slope"`` (or both)
  set the ageing rate; ``"cumulative"`` is refused, because the
  effective age already accumulates the reading over the item's life.
- **When the two models agree.** With a reading that never changes and a
  Weibull baseline the accelerated model is the proportional hazards one,
  each effect multiplied by ``beta`` (an ageing rate :math:`e^{\zeta}`
  is a hazard ratio :math:`e^{\beta\zeta}`): the two fits of such a fleet
  have the same likelihood and parameters to 1e-4
  (``tests/univariate/joint/test_joint_aft.py``). Once the reading
  moves they differ.
- **The history counts.** The effective age at ``given`` depends on the
  whole path from age 0, so ages are ages since new, and the readings
  should start early in the item's life: before the first reading the
  item's path is the path model's, extended back from its readings.
- **Which model.** By what the reading is: a symptom of the present
  damage (proportional hazards), or a stress that uses up life
  (accelerated); and by ``aic()`` where the readings move enough for
  the data to tell.

The fitted model says which survival part it has (``aged.survival`` is
``"AFT"``; the printout's survival line and the summary's heading say
so), saves it with ``to_dict`` (a file that SurPyval 0.28 and older
refuse rather than misread as proportional hazards) and predicts as before:

.. jupyter-execute::

    now = aged.predict(markers={0.0: 3.4, 0.5: 3.9, 1.0: 4.5}, given=1.0)
    print("life left, 10/50/90%:", now.remaining_life([0.1, 0.5, 0.9]).round(2))
    print(aged.cb([1.5, 2.0], markers=now.markers, given=1.0).round(3))

The likelihood is a second one beside the proportional hazards one, with
the same quadrature over each item's path: the effective age by the
cumulative hazard's 15-point Gauss--Legendre rule, and an interval's
cumulative hazard from the effective age at its start and the age used up
over it, integrated directly. It agrees with an independent dense-grid
integral for every kind of item to within 1e-5 an item, for the value and
the slope links and with covariates in each part.

A recovery study of the accelerated model (run on 2026-11-18 with
``scripts/studies/joint_recovery/study.py``'s ``aft_`` paths; this page
does not re-run it) fitted 200 fleets for each of the four paths of the
study below -- the same pumps, readings and paths, each pump ageing at
the rate :math:`e^{0.45 (m - 4)}` against a Weibull of 800 days and shape
2 in effective age -- with ``JointAFT`` and ``marker_reference=4.0``.
All 800 fits reached a verified maximum. The marker effect's bias was
+2.6% (Monte Carlo se 1.3%) for the line, +2.5% (1.1%) for the
exponential, +2.3% (1.2%) with ``b`` on a log link and +1.1% (1.5%) for
the quadratic, its 95% interval covering the truth 94.5%, 94.0%, 93.0%
and 95.5% of the time. Of the 37 parameters, 36 intervals covered the
truth between 92% and 98% of the time; the exception was the quadratic's
``sd_a``, 0.905 (0.021), whose estimate was 0.8% (0.3%) low. The spread of
the estimates was 0.91 to 1.07 times their mean standard error, except the
quadratic's ``alpha`` at 1.21, from two fleets with long tails (1.01
without them). Biases of more than two Monte Carlo standard errors, all
small: the marker effect for the line and the exponential (above),
``alpha`` +0.7% (0.4%) for the line and +2.1% (0.9%) for the quadratic,
and the exponential's ``corr_a_b``, +0.011 (0.005). The per-parameter
table is ``scripts/studies/joint_recovery/table.txt``.

How well it recovers the truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simulation study (run on 2026-10-16 with the script
``scripts/studies/joint_recovery/study.py``, whose summaries are beside it; this page does not re-run it) fitted 200
simulated fleets for each of four paths. Each fleet had 300 pumps read
monthly for up to two years, a Weibull hazard (``alpha`` 800 days, ``beta``
2 at a marker of 4) and a marker effect of 0.9, and was fitted with
``marker_reference=4.0``. The paths were:

- the line, ``a`` 3 (sd 0.6) and ``b`` 1/365 (sd 0.5/365), noise 0.7;
- the exponential ``a exp(b t)``, ``a`` 3 (sd 0.4) and ``b`` 0.0008
  (sd 0.0003), noise 0.3;
- the same with ``b`` on a log link, ``log(b)`` normal about log(0.0008)
  with sd 0.3;
- the quadratic, ``a`` 3 (sd 0.5), ``b`` 0.5/365 (sd 0.3/365) and ``c``
  1/365² common to all, noise 0.4.

All 800 fits reached a verified maximum except two of the log-link ones
(fitted, but unverified). For the marker effect:

- line: bias +0.4% (Monte Carlo se 0.8%), coverage of the 95% interval
  0.930 (0.018);
- exponential: +0.9% (0.7%), 0.980 (0.010);
- log link: +2.0% (0.7%), 0.955 (0.015);
- quadratic: +0.5% (0.7%), 0.960 (0.014).

Over all 37 parameters of the four models, 36 intervals covered the truth
between 92% and 98% of the time. The exception was the log-link model's
``sd_a`` at 0.900 (0.021). The spread of the estimates was 0.86 to 1.14
times their mean standard error.

Some biases are small but more than two Monte Carlo standard errors from
0:

- ``beta``: +1.4% (0.6%) for the line and +2.2% (0.9%) for the
  quadratic. The Weibull shape's maximum likelihood estimate is biased
  upward in samples this size; a Weibull fitted alone to 300
  lifetimes censored at 730 days (2000 samples, ``scripts/studies/joint_recovery/beta_bias.py``)
  is +0.7% (0.1%).
- the exponential's ``sd_b``: -0.9% (0.3%).
- the log-link model's ``alpha``, ``marker_effect`` and
  ``corr_a_log(b)``: +1.3% (0.4%), +2.0% (0.7%) and +0.011 (0.005).

These are small-sample biases of maximum likelihood, not the quadrature.
Refitting the same fleets with 11 points a dimension instead of 7 moved
no estimate by more than 0.0104 of its standard error. With four times the
pumps (100 fleets of 1200), the log-link model's biases fell to +0.4%
(0.25%), +0.2% (0.5%) and +0.005 (0.0035), and every interval covered the
truth between 93% and 98% of the time.

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

One item's dynamic prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``model.predict(item)`` (or ``predict(Z, markers=..., given=...)``) holds one
item's prediction as an object, for the questions asked of one asset: its
survival, failure probability and hazard from now on given its readings
(``sf``, ``ff``, ``hf``, ``df``, ``Hf``), its remaining life
(``remaining_life(p)``, the life left that it outlives with probability
``1 - p``, and ``qf`` the age), a Wald band from the parameters'
uncertainty (``cb``, the same as ``model.cb``), its marker smoothed through
the readings and projected beyond them (``marker`` and ``marker_interval``,
the noise-free path's mean and interval given the readings and that it
reached its age now), and draws of its failure age (``random``). Its
``sf``, ``ff`` and ``cb`` are the model's for the same item; the hazard is
the item's at that age given its readings and that it reached it.

.. jupyter-execute::

    now = model.predict(rising)
    print(now)
    print("life left, 10/50/90%:", now.remaining_life([0.1, 0.5, 0.9]).round(0))
    print("hazard a day at 425 and 545:", now.hf([425, 545]).round(5))
    print("vibration at 365 and 545:", now.marker([365, 545]).round(2))
    print(now.marker_interval(545).round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert now.sf(545) == model.sf(545, rising)
    assert np.allclose(now.ff(now.qf([0.1, 0.5, 0.9])), [0.1, 0.5, 0.9])
    assert now.remaining_life(0.5) < model.predict(quiet).remaining_life(0.5)

The 90% range of its life left is wide -- a prediction of one pump's life,
not a confidence interval -- and the band of :meth:`cb` adds the
uncertainty of the fitted parameters to the survival. A calibration study
(``tests/calibration/test_coverage_joint.py``, 200 simulated fleets of 200
pumps with the covariates of `Covariates in each part`_, with exact failure
times and with 90-day inspections) found the 95% band of a held-out pump's
conditional survival 60, 180 and 365 days ahead covering the true
conditional survival 93.0%, 89.5% and 93.0% of the time (exact times) and
93.5%, 94.5% and 94.0% (inspections); each parameter's 95% interval
covered its truth 93.0% to 96.5% of the time (exact) and 91.5% to 97.5%
(inspections), the standard errors 0.91 to 1.09 times the estimates'
spread. The Wald band can run a little narrow for this sample size, most
at 180 days ahead with exact times.

How good are the predictions
----------------------------

A joint model gives each running pump a chance of failing in the next six
months. Before acting on those chances, check how good they were on pumps
whose outcome is known. Three questions, at an age ``s`` (standing at one
year old, say) and over a window ``w`` (the next 180 days), among the
pumps still running at ``s``:

- **Do the pumps that failed get the higher chances?** The dynamic AUC
  (:func:`~surpyval.metrics.dynamic.dynamic_auc`): take one pump that
  failed in the window and one that did not; the AUC is how often the
  model gave the one that failed the higher chance. 0.5 is a coin toss, 1
  is perfect.
- **How far are the chances from what happened?** The dynamic Brier score
  (:func:`~surpyval.metrics.dynamic.dynamic_brier`): the mean squared
  difference between each pump's chance and 1 if it failed in the
  window, 0 if not. Lower is better.
- **Does "30%" mean about 30%?** Calibration
  (:func:`~surpyval.metrics.dynamic.dynamic_calibration`): group the pumps
  by their chance, and set each group's mean chance beside the share that
  failed.

Each pump's chance is made from what was known at ``s``: its readings up to
age ``s``, never those after. And each question is worth asking against a
simpler model, to see whether the readings earn their cost: here a Weibull
fitted to the ages alone, which gives every pump of the same age the same
chance. A dict of models scores each on the same pumps and compares them,
with the difference's interval and p-value:

.. jupyter-execute::

    from surpyval.metrics import dynamic_auc, dynamic_brier, dynamic_calibration

    models = {"age only": Weibull.fit(fleet.without("markers")), "joint": model}
    auc = dynamic_auc(models, fleet, at=[180, 365], horizon=180)
    auc

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert (auc.estimate("age only") == 0.5).all()
    assert (auc.estimate("joint") > 0.6).all()
    assert auc.estimate("joint")[1] > auc.estimate("joint")[0]

The age alone ranks the pumps no better than a coin toss: at one age it
gives them all the same chance. The readings rank them well, and better at
a year old than at six months, when each pump's readings say more about its
line. The Brier score tells the same story in the error of the chances:

.. jupyter-execute::

    brier = dynamic_brier(models, fleet, at=[180, 365], horizon=180)
    brier.comparisons

.. jupyter-execute::
    :hide-code:
    :hide-output:

    p = brier.comparisons["p_value"].to_numpy()
    assert p[0] > 0.05 and p[1] < 0.01
    assert (brier.estimate("joint") < brier.estimate("age only")).all()

At six months the readings do not yet cut the error by more than chance; at
a year they clearly do. Calibration, in groups of predicted chance:

.. jupyter-execute::

    cal = dynamic_calibration(
        model, fleet, at=365, horizon=180, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.6, 1]
    )
    columns = ["risk_from", "risk_to", "n", "predicted", "observed", "lower", "upper"]
    cal.table[columns].round(3)

A pump censored inside the window (still running when last seen) is not
known to have failed or not. It is accounted for, as in Rizopoulos (2011)
and Blanche et al. (2015): the AUC and the Brier score weight the pumps
whose outcome is known by the inverse of the chance of being seen to the end
of the window (the reverse Kaplan-Meier of the pumps at risk at ``s``), and
calibration's share that failed is each group's Kaplan-Meier estimate. With
no censoring inside the window they are the plain AUC, mean squared error
and share. A calibration study (``tests/calibration/test_metric_bias.py``:
200 simulated data sets of 300 units, 30% of them censored, scored with the
true model's predictions) found the mean estimates within Monte Carlo
error of the true values at two ages.

The pumps scored above are the ones the model was fitted to, which flatters
it. ``cross_validate=k`` gives honest numbers: the pumps are split at random
into ``k`` groups, the model is refitted ``k`` times without one group, and
each pump is predicted by the fit that never saw it. Give the fitters in
place of the fitted models (``LandmarkCox`` and functions of the data work
too, for a like-for-like comparison with landmarking):

.. jupyter-execute::

    honest = dynamic_brier(
        {"age only": Weibull, "joint": WeibullJointPH},
        fleet, at=[180, 365], horizon=180, cross_validate=5, random_state=1,
    )
    honest.scores[["model", "at", "estimate", "se"]]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(honest.estimate("joint"), brier.estimate("joint"), atol=0.01)

With 300 pumps the model barely flatters itself here; with fewer units or
a richer model the difference grows. The intervals are two-sided 95% by
default (``alpha_ci=0.1`` for 90%), and the result says so.

A quick first look: the two-stage fit
-------------------------------------

There is a quicker way to put the readings and the lifetimes together, in
two steps. First, fit each pump's line to its readings alone, pulling a
pump with few readings towards the fleet's average line (the less its own
readings say, the further it is pulled; statisticians call the result a
best linear unbiased predictor). Then fit the lifetimes by Weibull
proportional hazards with each pump's line as a time-varying covariate, as
if the lines were known. It takes a fraction of a second, against seconds
or minutes for the joint fit, and it is the method most published analyses
used before joint models. It is ``fit(..., how="two-stage")``, and it is
exactly where the joint fit starts its search:

.. jupyter-execute::

    quick = WeibullJointPH.fit(fleet, unit="days", how="two-stage")
    quick

In formulas, step one fits :math:`y_{ij} = m(t_{ij}; \theta_i) +
\varepsilon_{ij}`, :math:`\theta_i \sim N(\mu + \Delta z_i, D)`, to the
readings alone (the path fitted to each item, then a moment estimate of
the means and of :math:`D`), and takes each item's most likely path given
its readings, :math:`\hat\theta_i`. Step two maximises the likelihood of
the lifetimes with the hazard :math:`h_0(t) \exp(\gamma' z_i + \zeta\,
m(t; \hat\theta_i))` (or the slope or the area: the ``link``), the path
taken as a known covariate. ``JointAFT`` fitters take it too, with the
accelerated survival part in step two.

The result is a joint model like any other -- the same parameters,
predictions (``sf``, ``cb``, ``predict``), accuracy measures and saving
-- and it says what it is (``quick.how`` is ``"two-stage"``). It is
approximate in three ways, which the printout and ``summary()`` state:

- **Its estimates are biased.** Step two forgets that each line is an
  estimate, and that a pump's readings stop when it fails, so the pumps
  that failed early have the least certain lines.
- **Its standard errors are too small.** Those of the survival part take
  each line as known, so they and the bands of its predictions (``cb``)
  are too narrow; the path part (step one's moment estimates) has none
  (``nan``).
- **It has no likelihood.** The two steps fit different things, so there
  is no likelihood of the whole model: ``neg_ll()``, ``aic()``,
  ``aic_c()``, ``bic()`` and ``compare_links`` refuse it.

.. jupyter-execute::
    :raises: ValueError

    quick.aic()

With a reading every month, as here, the two fits are close:

.. jupyter-execute::

    def effect(m):
        return f"{m.params[-1]:.2f} (se {m.standard_errors()[-1]:.2f})"


    print("two-stage:  ", effect(quick))
    print("joint model:", effect(model))

With a reading every six months they part:

.. jupyter-execute::

    every_six_months = SurpyvalData.from_items(
        [
            item.clone(
                markers={a: v for a, v in item.markers.items() if a % 180 == 0}
            )
            for item in fleet.to_items()
        ]
    )
    sparse_quick = WeibullJointPH.fit(every_six_months, how="two-stage")
    sparse_joint = WeibullJointPH.fit(every_six_months)
    print("two-stage:  ", effect(sparse_quick))
    print("joint model:", effect(sparse_joint))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert quick.how == "two-stage" and quick.maximum == "not applicable"
    assert round(quick.params[-1], 2) == 0.99
    assert round(sparse_quick.params[-1], 2) == 0.86
    assert round(sparse_joint.params[-1], 2) == 1.04
    assert sparse_quick.standard_errors()[-1] < sparse_joint.standard_errors()[-1]
    assert np.isnan(quick.standard_errors()[0])

The two-stage effect is the lower, and claims to be the more certain (a
standard error of 0.10 against 0.15).

Its predictions are compared with the joint model's as any two models'
are, by their accuracy (see `How good are the predictions`_); give the
two-stage fit as a function of the data to cross-validate it:

.. jupyter-execute::

    two_ways = dynamic_brier(
        {
            "two-stage": lambda data: WeibullJointPH.fit(data, how="two-stage"),
            "joint": WeibullJointPH,
        },
        fleet, at=[180, 365], horizon=180, cross_validate=5, random_state=1,
    )
    two_ways.comparisons

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert (two_ways.estimate("joint") <= two_ways.estimate("two-stage") + 0.002).all()
    p = two_ways.comparisons["p_value"].to_numpy()
    assert p[0] > 0.5 and p[1] < 0.05

Predicting each pump from a fit that never saw it, the two are alike at
six months; at a year the joint model's chances are a little better (a
Brier score lower by 0.006, p = 0.03). Here the quick fit would have done
for a first look; the study below says where it does not.

How much the two-stage fit loses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simulation study (run on 2026-11-20 with
``scripts/studies/joint_recovery/study.py`` and ``HOW=two-stage``; this
page does not re-run it) fitted the two-stage estimate to the same 200
fleets of each path as the joint model's studies (`How well it recovers the
truth`_, and the accelerated model's), and to 200 fleets of the line read
every 180 days instead of every 30, fitted both ways. For the marker
effect, the bias (Monte Carlo standard error) and how often the 95%
interval covered the truth:

.. list-table::
   :header-rows: 1

   * - fleets
     - joint: bias
     - joint: coverage
     - two-stage: bias
     - two-stage: coverage
   * - line
     - +0.4% (0.8%)
     - 0.930
     - +0.3% (0.8%)
     - 0.900
   * - exponential
     - +0.9% (0.7%)
     - 0.980
     - +0.7% (0.7%)
     - 0.965
   * - exponential, log link
     - +2.0% (0.7%)
     - 0.955
     - +4.8% (0.7%)
     - 0.935
   * - quadratic
     - +0.5% (0.7%)
     - 0.960
     - -4.8% (0.9%)
     - 0.830
   * - line, accelerated
     - +2.6% (1.3%)
     - 0.945
     - -0.1% (1.3%)
     - 0.920
   * - exponential, accelerated
     - +2.5% (1.1%)
     - 0.940
     - +1.5% (1.1%)
     - 0.950
   * - log link, accelerated
     - +2.3% (1.2%)
     - 0.930
     - -1.9% (1.2%)
     - 0.930
   * - quadratic, accelerated
     - +1.1% (1.5%)
     - 0.955
     - -1.0% (2.4%)
     - 0.800
   * - line, read every 180 days
     - +2.4% (1.1%)
     - 0.920
     - -8.0% (0.9%)
     - 0.805

With a reading every month and a path that does not curve much, the
two-stage effect is about as good as the joint model's; what it loses is
in its intervals, which cover the truth less often than they say (90% of
the time for the line, the estimates' spread 1.11 times their standard
error). The loss grows where the path curves (the quadratic: intervals
covering 80% to 83% of the time, the spread 1.4 to 1.7 times the standard
error) and where readings are few (every 180 days: 8% low, covering 80.5%
of the time, where the joint fit is within 2.4% and covers 92%). Step one's
spread between items is poor: the line's ``sd_b`` came out 14% low, the
log link's ``sd_log(b)`` 68% low and the quadratic's ``sd_b`` five times
the truth (+403%), where the joint fits' were within 1%; so a two-stage
model's predictions for items with few readings, which lean on that
spread, deserve the least trust. Each two-stage fit took about a tenth of
a second. The per-parameter tables are in
``scripts/studies/joint_recovery/table.txt``.

Use it for a first look at a new dataset, and to see how far the joint
model moves the answer; report the joint fit.

Saving and loading
------------------

The model saves to strict JSON and loads back, predicting and bounding as
before:

.. jupyter-execute::

    import surpyval

    restored = surpyval.from_json(model.to_json())
    restored.sf(545, rising) == model.sf(545, rising)

What it does and does not do
----------------------------

The joint model (#891) is the standard one. It has:

- one marker, following any degradation path model (a straight line by
  default), with up to five of its parameters varying between items
  (multivariate normal on their natural scale, or on a log scale with a
  ``LinkedPathModel``; every pair correlated, none, or blocks of them,
  ``random_effects=``) and the rest common;
- a Weibull proportional hazards survival part (``JointPH(Weibull,
  path_model)``; ``WeibullJointPH`` is the line; other baselines are
  refused) in the path's current value, its slope, the area under it, or
  more than one of them (``link=``);
- or an accelerated (AFT) survival part on the Weibull (``JointAFT(Weibull,
  path_model)``; ``WeibullJointAFT`` is the line), the path's value, its
  slope or both setting how fast the item ages;
- covariates in both parts (every path parameter's mean, and the hazard)
  by default, or each in the parts named by ``hazard_covariates`` and
  ``path_covariates``;
- failures, right, left and interval censoring, with left and right
  truncation;
- predictions for one item given its readings, with Wald (delta method)
  bounds, as arrays (``sf``, ``ff``, ``cb``) or as one item's dynamic
  prediction (``predict``: the hazard, remaining life, the smoothed and
  projected marker, and draws);
- the accuracy of those predictions at an age over a window (dynamic AUC,
  Brier score and calibration in ``surpyval.metrics``, with
  cross-validation by unit);
- the quick, approximate two-stage fit (``how="two-stage"``) for a first
  look and for comparison.

Not yet (see #891's roadmap): spline paths; other baselines (a
log-normal or log-logistic one for the accelerated part, whose risk can
rise and then fall, and spline baselines); several markers. The
accelerated part has not yet been compared with R's ``JM``
(``method = "weibull-AFT-aGH"``, the same model) on reference data, nor
had its prediction band put through a calibration study. The two-stage fit
has not yet been compared with R's two-step equivalent (``nlme::lme``,
then ``survival`` with the fitted paths) on reference data, and its path
part has no standard errors.

References
----------

- Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event
  Data, with Applications in R*. Chapman & Hall/CRC.
- Rizopoulos, D. (2011). Dynamic predictions and prospective accuracy in
  joint models for longitudinal and time-to-event data. *Biometrics*, 67,
  819--829.
- Blanche, P., Proust-Lima, C., Loubère, L., Berr, C., Dartigues, J.-F. and
  Jacqmin-Gadda, H. (2015). Quantifying and comparing dynamic predictive
  accuracy of joint models for longitudinal marker and time-to-event in
  presence of censoring and competing risks. *Biometrics*, 71, 102--113.
- Wulfsohn, M. S. and Tsiatis, A. A. (1997). A joint model for survival and
  longitudinal data measured with error. *Biometrics*, 53, 330--339.
- Tsiatis, A. A. and Davidian, M. (2004). Joint modeling of longitudinal and
  time-to-event data: an overview. *Statistica Sinica*, 14, 809--834.

API reference
-------------

``WeibullJointPH`` is ``JointPH(Weibull, LinearPath)`` and
``WeibullJointAFT`` is ``JointAFT(Weibull, LinearPath)``, instances of the
fitter classes below; import everything from ``surpyval`` (or
``surpyval.univariate.joint``).

.. autofunction:: surpyval.univariate.joint.joint_ph.JointPH

.. autoclass:: surpyval.univariate.joint.joint_ph.JointPHFitter
   :members: fit, fit_from_df, compare_links

.. autofunction:: surpyval.univariate.joint.joint_aft.JointAFT

.. autoclass:: surpyval.univariate.joint.joint_aft.JointAFTFitter

.. autoclass:: surpyval.univariate.joint.joint_model.JointModel
   :members: sf, ff, cb, predict, summary, covariance, standard_errors, to_dict, from_dict

.. autoclass:: surpyval.univariate.joint.compare.LinkComparison
   :members: table, likelihood_ratio_tests, best_link, best_label

.. autoclass:: surpyval.univariate.joint.dynamic.DynamicPrediction
   :members: sf, ff, hf, df, Hf, qf, remaining_life, cb, marker, marker_interval, random
