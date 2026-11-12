Landmark Analysis
=================

A **dynamic prediction** answers a question asked part way through
follow-up: *this unit has lasted to time* :math:`s` *and this is what we
know about it now; how likely is it to have its event in the next*
:math:`w`? A pump that has run 2 years and whose vibration was last read as
high; a patient alive 6 months after treatment whose blood counts have
recovered. The answer changes as time passes and as what is known changes,
so it is asked again at each **landmark** time :math:`s`.

**Landmarking** (van Houwelingen, 2007; van Houwelingen and Putter, 2011)
answers it directly:

1. at the landmark :math:`s`, keep the units still at risk -- entered by
   :math:`s` and event-free past it;
2. give each its covariates *as they were known at* :math:`s` -- the fixed
   ones, and the last value at or before :math:`s` of anything that
   changes over time (a time-varying covariate, a measurement);
3. stop following them at the horizon :math:`s + w`, censoring any still
   running then (or failing later);
4. fit an ordinary model, such as a Cox model, to that data set.

Why it avoids immortal-time bias
--------------------------------

A tempting shortcut is to group units by something that happens *during*
follow-up -- whether a repair was made, whether a treatment response was
seen -- and compare the groups from time 0. It is biased: a unit can only
join the "it happened" group by surviving until it happens, so that group
is credited with event-free time it was guaranteed (*immortal* time), and
looks better than it is. A landmark data set cannot do this: it groups
units by what was known *at* :math:`s`, and counts only what happens to
them *after* :math:`s`. Nothing from the future is used.

When a joint model is better
----------------------------

Landmarking takes the last reading as the truth until :math:`s`. If the
covariate is a noisy measurement read at irregular times, a
:doc:`joint model <Joint Models>` uses every reading, allows for the
noise and describes how the measurement moves between readings, and gives
the dynamic prediction from the same model at any time. It is the better
model when its assumptions about the measurement's path hold; landmarking
asks for no model of the covariate at all, which makes it simpler and
robust to a wrong one, at the cost of using only the latest value.

The landmark data
-----------------

:func:`~surpyval.landmark` derives the landmark data from a
:class:`~surpyval.SurpyvalData` of the units' lifetimes (one row per unit,
with its time, censoring, counts, entry, id and covariates; time-varying
covariates and measurements as :meth:`SurpyvalData.from_items
<surpyval.SurpyvalData.from_items>` carries them). The result is another
``SurpyvalData``, which any model fits:

.. code:: python

    import numpy as np
    import surpyval as sp

    # A unit's covariate switches from 0 to 1 at some time (a recovery, a
    # change of duty); z1 is fixed.
    rng = np.random.default_rng(1)
    items = []
    for unit in range(300):
        switch = rng.exponential(3.0)
        z1 = float(rng.uniform() < 0.5)
        t = rng.exponential(1 / (0.15 * np.exp(0.5 * z1)))
        if t > switch:  # the hazard doubles after the switch
            t = switch + rng.exponential(1 / (0.3 * np.exp(0.5 * z1)))
        end = rng.uniform(4, 10)
        outcome = {"observed": t} if t <= end else {"right_censored": end}
        items.append(sp.ItemLifetime(
            unit, **outcome, covariates={"z1": z1},
            time_varying_covariates=sp.StepSchedule.from_changepoints(
                [0, switch], [0, 1]),
        ))
    life = sp.SurpyvalData.from_items(items)

    data = sp.landmark(life, at=2, horizon=3)
    data.covariate_names     # ['z1', 'time_varying_0'], as known at 2
    data.tl[:3], data.x[:3]  # entered at 2, followed at most to 5

At one landmark the rows are the units at risk then, entered at the
landmark (``tl = s``), each with its covariates as known then. With
several landmarks, ``at=[0, 1, 2, 3]``, the landmark data sets are stacked
into one, the *super landmark data set*: a unit appears once for each
landmark it is at risk at, the landmark is a covariate column ``L``, and
``interactions=True`` adds each covariate times ``L``. Each row's unit and
landmark are in its ``attributes`` (provisional: once ``SurpyvalData`` can
hold a unit's id on several rows, the unit will be its ``i``). A
measurement with no reading at or before a landmark leaves that unit out
of that landmark, with a warning that counts them: nothing is known about
it there. (The form in which ``SurpyvalData`` carries measurements,
``data.m``, is provisional too, and will follow its measurements class.)

Landmark models
---------------

:class:`~surpyval.LandmarkCox` fits a Cox model to the landmark data and
predicts from it. With one landmark it is the Cox model of that landmark
data set. With several it is a **supermodel**: the effect of each covariate
may change smoothly with the landmark,

.. math::

    h(t \mid Z(s), s) = h_0(t \mid s)\,\exp\big(\beta(s)' Z(s)\big),
    \qquad s \le t \le s + w, \qquad
    \beta(s) = \beta_0 + \beta_1 u + \dots + \beta_d u^d,

with :math:`u = (s - s_1) / (s_L - s_1)` running from 0 at the first
landmark to 1 at the last (``degree=d``; or any basis of :math:`s` you
give, a spline for example, as ``basis=``; ``varying=`` names the
covariates whose effect changes). The baseline hazard is either

- **stratified** (``baseline="stratified"``, the default): a baseline of
  its own at each landmark, which predicts at the landmarks fitted; or
- **pooled** (``baseline="pooled"``): one baseline :math:`h_0(t)
  e^{\theta(s)}`, :math:`\theta(s) = \theta_1 u + \dots + \theta_d u^d`
  (the "proportional baselines" model), which predicts at any :math:`s`
  between the first and last landmark.

.. code:: python

    model = sp.LandmarkCox.fit(life, at=[0, 1, 2, 3, 4], horizon=3)
    model.summary()                  # coefficients, robust standard errors
    model.beta([0, 2, 4])            # each covariate's effect at s
    model.predict(2, [[0, 0], [1, 1]])   # P(T <= 5 | T > 2, Z(2))
    model.table([[0, 0], [0, 1], [1, 0], [1, 1]])  # landmarks x profiles

``predict(s, Z)`` is the dynamic prediction :math:`P(T \le s + w \mid T >
s, Z(s))`; ``sf``, ``ff`` and ``Hf(x, Z, s=...)`` give it for any time
:math:`x` from :math:`s` to :math:`s + w`. ``table`` lays out the dynamic
prediction table, a row for each landmark and covariate profile with the
probability and its bounds.

Standard errors and bounds
--------------------------

In the stacked data a unit appears several times, so its rows are not
independent and the model-based standard errors of the Cox fit are too
small. The standard errors are therefore the **robust** (sandwich) ones,
clustered on the unit, as R's ``coxph(... + cluster(id))`` reports;
``covariance(robust=False)`` gives the model-based one for comparison.
The bounds on a prediction (``cb``, and the ``table``) carry the same
robust variance to the prediction: each unit's influence on the baseline
over :math:`(s, x]` and on the coefficients is added up over its rows,
and the influences are squared and summed over the units (the
infinitesimal jackknife). The bound is normal on the log cumulative
hazard by default (``bound_type="log-log"``), which keeps it inside (0, 1).
In simulations of a correctly specified model with four landmarks per
unit, the 95% robust intervals covered the coefficients 93%-96% of the
time and a window probability 95%.

Only these Wald bounds are offered: the stacked rows have no likelihood of
their own for a likelihood-ratio bound, and no bootstrap is built (resample
the units, refit, and take quantiles, for one).

Survival from a landmark by group
---------------------------------

The simplest landmark analysis needs no model:
:func:`~surpyval.landmark_kaplan_meier` gives the Kaplan-Meier estimate of
survival from the landmark, :math:`P(T > x \mid T > s)`, for each group
of units by their covariates known at :math:`s`, with its usual bounds
(each unit counted once):

.. code:: python

    km = sp.landmark_kaplan_meier(life, at=2, horizon=3, by="time_varying_0")
    km[1.0].sf([3, 4, 5])      # switched by time 2
    km[0.0].sf([3, 4, 5])      # not yet switched at 2

Checked against R
-----------------

The landmark data sets match ``dynpred``'s ``cutLM`` (0.1.2) row for row;
the Cox model at one landmark and the stratified and pooled supermodels
match ``survival``'s ``coxph(... + cluster(id))`` (3.8.12): coefficients,
robust and model-based standard errors, and the window probabilities from
``survfit``, to about 1e-10, with Efron's and Breslow's ties.

One convention differs from ``cutLM``'s long format, whose default
(``right = TRUE``) does not use a change of covariate recorded exactly at
the landmark: here a value recorded at :math:`s` is known at :math:`s`
(``cutLM``'s ``right = FALSE``, and its wide format).

References
----------

- van Houwelingen, H. C. (2007). Dynamic prediction by landmarking in
  event history analysis. *Scandinavian Journal of Statistics*, 34,
  70--85.
- van Houwelingen, H. C. and Putter, H. (2011). *Dynamic Prediction in
  Clinical Survival Analysis*. Chapman & Hall/CRC.
- Lin, D. Y. and Wei, L. J. (1989). The robust inference for the Cox
  proportional hazards model. *Journal of the American Statistical
  Association*, 84, 1074--1078.

API reference
-------------

Import everything from ``surpyval`` (or ``surpyval.univariate.landmark``).

.. autofunction:: surpyval.univariate.landmark.landmark

.. autofunction:: surpyval.univariate.landmark.landmark_kaplan_meier

.. autoclass:: surpyval.univariate.landmark.LandmarkCox
   :members: fit, fit_from_df, predict, table, sf, ff, Hf, cb, beta, beta_cb, summary, covariance, standard_errors, aliased, to_dict, from_dict
