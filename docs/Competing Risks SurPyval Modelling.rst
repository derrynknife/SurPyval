Competing Risks SurPyval Modelling
===================================

This page shows how to use SurPyval's competing risks classes. For the
theoretical background see :doc:`Competing Risks Analysis`; for the complete
list of arguments and methods see the :doc:`surpyval.competing_risks` API
reference.

.. note::

    Every example on this page is executed when the documentation is built, so
    the outputs shown are produced by the installed version of surpyval.

SurPyval's competing-risks tools, and the question each one answers:

.. list-table::
   :header-rows: 1

   * - Tool
     - Answers
   * - ``CompetingRisks``
     - Non-parametric (Aalen-Johansen) cumulative incidence of each cause.
   * - ``ParametricCompetingRisks``
     - One distribution per cause; smooth CIFs, extrapolation, simulation.
   * - ``gray_test``
     - Does the cumulative incidence of a cause differ between groups?
   * - ``FineGray``
     - How do covariates change the cumulative incidence of one cause?
   * - ``CompetingRisksProportionalHazards``
     - How do covariates change each cause-specific hazard (``model="Cox"``),
       or the incidence of every cause (``model="Fine-Gray"``)?
   * - ``ParametricCompetingRisksRegression``
     - A parametric regression per cause; smooth CIFs at any covariates,
       beyond the last observation.

The classes live in ``surpyval.univariate.competing_risks``; ``gray_test`` is
also available at the top level as ``surpyval.gray_test``. All of them accept
right-censored data, and ``ParametricCompetingRisks`` also failures found at
inspections (left- and interval-censored rows). For repeated events with several failure modes see
the cause-specific MCF and NHPP models in
:doc:`Recurrent Event Modelling with SurPyval`.

Standard imports used throughout this page:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt


Fitting a Competing Risks Model
-------------------------------

The ``CompetingRisks`` class estimates a non-parametric cumulative incidence
function (CIF) for each failure cause with the Aalen-Johansen estimator. Pass
the observed times, a cause indicator, and optional censoring flags.

Competing-risks data format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every competing-risks fit takes the same core arrays:

- ``x`` -- the observed time of each row (failure or censoring time);
- ``e`` -- the cause of each row. Any hashable labels work (integers,
  strings, ...). A censored row has no cause: use ``None`` (or ``NaN``, or a
  blank cell in a DataFrame);
- ``c`` -- optional censoring flags, ``0`` observed and ``1`` right-censored
  (and, for ``ParametricCompetingRisks``, ``-1`` left- and ``2``
  interval-censored).
  When ``c`` is omitted it is derived from ``e``: a missing cause means
  censored. If you do pass ``c``, every row with ``c == 1`` must have a missing
  cause and every other row must have one, otherwise a ``ValueError`` explains
  the mismatch;
- ``n`` -- optional counts, when each row stands for several identical units;
- ``tl`` -- for ``CompetingRisks``, ``ParametricCompetingRisks`` and the
  cause-specific ``CompetingRisksProportionalHazards``, the optional age at
  which each row came under observation (left truncation, or delayed entry;
  see `Delayed entry`_, `Assembling a model from separately fitted causes`_
  and `Delayed entry in the cause-specific model`_).

Left- and interval-censored rows (``c`` of ``-1`` or ``2``) are rejected by
every fit but ``ParametricCompetingRisks`` (see `Failures found at
inspections`_), whose error says so.

lifelines, scikit-survival and R's ``cmprsk`` code the causes as one
integer column with ``0`` for a censored row. Here ``0`` is a cause like any
other (you may number causes from 0), so convert such data with
``e = np.where(np.asarray(e) == 0, None, e)``. Because the two codings
cannot be told apart, a fit whose labels are numbers including 0, with none
missing and no ``c``, warns and says so; passing ``c`` (``np.zeros(len(e))``
when nothing is censored) silences it.

The smallest possible example is the six-unit data set worked by hand on the
:doc:`Competing Risks Analysis` page. The fourth unit is censored:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import CompetingRisks

    x = [1, 2, 3, 4, 5, 6]
    e = ["A", "B", "A", None, "B", "A"]

    small = CompetingRisks.fit(x, e)
    print("CIF of A at t=6:", small.cif(6, "A"), "(7/12 = %.4f)" % (7 / 12))
    print("CIF of B at t=6:", small.cif(6, "B"), "(5/12 = %.4f)" % (5 / 12))

and "one minus Kaplan-Meier with the other cause censored" gives the
misleading answers discussed on the theory page -- 1 for A and 0.6 for B, a
total "probability" of 1.6:

.. jupyter-execute::

    for k in ["A", "B"]:
        c_k = [0 if ei == k else 1 for ei in e]   # other cause -> censored
        naive = surv.KaplanMeier.fit(x, c=c_k)
        print(k, "naive 1 - KM at t=6:", naive.ff(6))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    naive_ff = [
        surv.KaplanMeier.fit(x, c=[0 if ei == k else 1 for ei in e]).ff(6)
        for k in ["A", "B"]
    ]
    assert np.allclose(np.ravel(naive_ff), [1.0, 0.6]), naive_ff
    assert np.isclose(small.cif(6, "A"), 7 / 12)
    assert np.isclose(small.cif(6, "B"), 5 / 12)

When each row stands for several identical units, give the counts in ``n``
rather than repeating rows. Doubling every row of the six-unit data gives the
same CIFs, because the estimator only uses the proportions
:math:`d_{k,j}/r_j`. The ``iif`` method returns the individual increments
:math:`\hat{S}(x_{j-1})\,d_{k,j}/r_j`, here the three terms
:math:`\tfrac{1}{6}, \tfrac{1}{6}, \tfrac{1}{4}` of the hand calculation on the
theory page:

.. jupyter-execute::

    doubled = CompetingRisks.fit(x, e, n=[2] * 6)
    print("CIF of A at t=3, 6     :", doubled.cif([3, 6], "A"))
    print("increments at t=1, 3, 6:", doubled.iif([1, 3, 6], "A"))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(doubled.cif([3, 6], "A"), small.cif([3, 6], "A"))
    assert np.allclose(doubled.iif([1, 3, 6], "A"), [1 / 6, 1 / 6, 1 / 4])

Non-parametric cumulative incidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A more realistic example: components fail either by wear-out (a Weibull
latent life) or by random shocks (an exponential latent life), whichever comes
first, and some are removed from test (censored) before failing. Because we
simulate the data we know the true CIFs and can check the estimates:

.. jupyter-execute::

    rng = np.random.default_rng(0)
    N = 300
    t_wear = 100 * rng.weibull(3.0, N)       # Weibull(alpha=100, beta=3)
    t_shock = rng.exponential(150, N)        # Exponential, mean 150
    t_cens = rng.uniform(0, 200, N)          # removal from test

    x = np.minimum.reduce([t_wear, t_shock, t_cens])
    e = np.where(x == t_cens, None,
                 np.where(t_wear < t_shock, "wear", "shock")).astype(object)

    model = CompetingRisks.fit(x, e)         # c is derived from e
    print(model)
    print("event index map:", model.event_idx_map)

Causes are sorted and mapped to internal indices (``event_idx_map``); you
always refer to a cause by its own label. Once fitted, you can query the CIF
for each cause:

.. jupyter-execute::

    t_plot = np.linspace(0, 200, 400)
    for k in ["wear", "shock"]:
        plt.step(t_plot, model.cif(t_plot, event=k), where="post", label=k)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.legend()
    plt.title('Competing Risks CIF by Cause')

The CIFs of all causes add up to the all-cause failure probability. The
Aalen-Johansen increments are weighted with the Kaplan-Meier all-cause
survival, so the sum equals one minus the ordinary Kaplan-Meier curve of the
data with every cause treated as a failure -- exactly, not approximately:

.. jupyter-execute::

    t = np.array([25.0, 50.0, 100.0, 150.0])
    total = model.cif(t, "wear") + model.cif(t, "shock")
    all_cause_km = surv.KaplanMeier.fit(x, c=(e == None).astype(int))
    print("sum of CIFs   :", np.round(total, 4))
    print("1 - KM (all)  :", np.round(all_cause_km.ff(t), 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(total, all_cause_km.ff(t), rtol=0, atol=1e-12)

The model's own all-cause ``ff`` is a different estimate unless you ask for
this one. ``sf``, ``ff`` and ``Hf`` use the estimator given by ``how``, and
the default is Nelson-Aalen, ``ff = 1 - exp(-H)``. That is never above the
sum of the CIFs, and is below it from the first event on; the gap is small
while few units have failed and grows as the risk sets shrink. Fit with ``how="Kaplan-Meier"`` when ``ff`` should match
the CIFs:

.. jupyter-execute::

    km_fit = CompetingRisks.fit(x, e, how="Kaplan-Meier")
    print("ff, Nelson-Aalen :", np.round(model.ff(t), 4))
    print("ff, Kaplan-Meier :", np.round(km_fit.ff(t), 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(km_fit.ff(t), total, rtol=0, atol=1e-12)
    assert np.all(model.ff(t) <= total + 1e-12)

Now compare the CIF of wear-out with the naive "1 - Kaplan-Meier with shocks
censored" curve, and with the truth. For this simulation the true CIF of wear
is :math:`\int_0^t f_{\text{wear}}(u)\, S_{\text{shock}}(u)\, du`:

.. jupyter-execute::

    from scipy.integrate import quad

    def true_cif_wear(t):
        dens = lambda u: surv.Weibull.df(u, 100, 3.0) * np.exp(-u / 150)
        return quad(dens, 0, t)[0]

    naive_wear = surv.KaplanMeier.fit(x, c=(e != "wear").astype(int))

    plt.step(t_plot, model.cif(t_plot, "wear"), where="post",
             label="Aalen-Johansen CIF")
    plt.step(t_plot, naive_wear.ff(t_plot), where="post",
             label="naive 1 - KM (shocks censored)")
    plt.plot(t_plot, [true_cif_wear(ti) for ti in t_plot], "k--",
             label="true CIF")
    plt.xlabel("Time")
    plt.ylabel("Probability of wear-out failure")
    plt.legend()

The naive curve climbs towards one because it pretends that a component
destroyed by a shock could still have worn out later. The Aalen-Johansen
estimate tracks the true incidence, which levels off at the probability that
wear-out, rather than a shock, is what ends a component's life.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # naive curve well above the truth, Aalen-Johansen close to it
    assert naive_wear.ff(200.0) > 0.95, naive_wear.ff(200.0)
    _err = [model.cif(ti, "wear") - true_cif_wear(ti) for ti in t]
    assert np.max(np.abs(_err)) < 0.08, _err

Delayed entry
^^^^^^^^^^^^^

Often a unit comes under observation only some time after it started: a
patient joins a study at diagnosis but the clock is their age, or a component
enters the records when they began, long after it was installed. A unit that
failed before it would have entered is never seen, so the units you do see
are the ones that survived to their entry. Pass each unit's entry time as
``tl``: the unit then counts as at risk only between its entry and its exit
(``tl < t <= x``), and the estimates are those of the units still running at
each time. They are conditional on surviving to the earliest entry, as a
``KaplanMeier`` fit with ``tl`` is. Here half the components were watched
from new and the rest only came under observation at a random age; the fit
with the entry ages comes close to the fit of every component from new (the
``model`` above), while the fit that ignores them counts too many at risk
early on and understates the shocks:

.. jupyter-execute::

    rng_entry = np.random.default_rng(1)
    entry = np.where(rng_entry.uniform(size=N) < 0.5, 0.0,
                     rng_entry.uniform(0, 80, N))
    seen = x > entry                         # failed before entry: never seen
    late = CompetingRisks.fit(x[seen], e[seen], tl=entry[seen])
    ignored = CompetingRisks.fit(x[seen], e[seen])
    print("with entry     :", np.round(late.cif(t, "shock"), 4))
    print("ignoring entry :", np.round(ignored.cif(t, "shock"), 4))
    print("all from new   :", np.round(model.cif(t, "shock"), 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _full = model.cif(t, "shock")
    _late = np.abs(late.cif(t, "shock") - _full)
    _ignored = np.abs(ignored.cif(t, "shock") - _full)
    assert np.all(_late < 0.03), _late
    assert np.all(_late < _ignored), (_late, _ignored)

``fit_from_df`` takes the column as ``tl_col``, and ``fit(data)`` passes a
``SurpyvalData``'s entry times through. The bounds (``cb``) work as before:
the variance is a function of the numbers at risk and of failures at each
time, which already count only the units that have entered. This is R's
``survfit(Surv(entry, x, factor(e)) ~ 1)``; right truncation is not
supported.

What the fitted model returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All methods take the query times first and a cause label as ``event``. The
CIF methods need a cause; the others treat ``event=None`` (the default) as
"all causes combined". Every function is a step function that is zero (or
one, for ``sf``) before the first observed time.

.. list-table::
   :header-rows: 1

   * - Method
     - Returns
   * - ``cif(x, event)``
     - Aalen-Johansen cumulative incidence :math:`\hat{F}_k(x)` -- the
       real-world probability of having failed from ``event`` by ``x``.
   * - ``iif(x, event)``
     - The CIF jump at the most recent observed time at or before ``x``
       (zero if no failure from ``event`` happened at that time).
   * - ``hf(x, event=None)``
     - The Nelson-Aalen hazard increment :math:`d_{k,j}/r_j` at the most
       recent observed time at or before ``x`` (a jump size, not a rate).
   * - ``Hf(x, event=None)``
     - Cumulative (cause-specific) hazard :math:`\hat{H}_k(x)`.
   * - ``sf(x, event=None)`` / ``ff(x, event=None)``
     - The survival and ``1 - sf``, by the ``method`` the model was fitted
       with: :math:`e^{-\hat{H}_k(x)}` (Nelson-Aalen, the default) or the
       product limit :math:`\prod (1 - d_{k,j}/r_j)` (Kaplan-Meier). With an
       ``event`` these are the *net* quantities (the cause acting alone);
       with ``event=None`` they are all-cause.
   * - ``df(x, event=None)``
     - ``hf * sf``.

.. jupyter-execute::

    print("CIF wear         :", np.round(model.cif(t, "wear"), 4))
    print("net ff wear      :", np.round(model.ff(t, event="wear"), 4))
    print("cum. hazard wear :", np.round(model.Hf(t, event="wear"), 4))
    print("all-cause sf     :", np.round(model.sf(t), 4))

.. warning::

    ``ff(x, event=k)`` is the net failure probability: "one minus
    Kaplan-Meier with the other causes censored" (exactly that with
    ``method="Kaplan-Meier"``, its Nelson-Aalen version by default). It
    answers "what if cause ``k`` were the only cause?", which is only
    meaningful if the causes act independently. For "how likely is a
    failure from cause ``k``?" always use ``cif``.

The ``method`` argument (``"Nelson-Aalen"``, the default, or
``"Kaplan-Meier"``) selects the survival estimator that ``sf``, ``ff`` and
``Hf`` report (``Hf`` is ``-log sf``, so the three stay consistent); the
all-cause estimate is also stored as ``model.S`` at the distinct observed
times ``model.x``. It does not change ``cif``: the Aalen-Johansen weights are
always the product-limit (Kaplan-Meier) survival, the only one for which the
CIFs sum to the all-cause failure probability.

.. jupyter-execute::

    km_model = CompetingRisks.fit(x, e, how="Kaplan-Meier")
    print("same CIFs          :", np.allclose(km_model.cif(t, "wear"), model.cif(t, "wear")))
    print("sf (Kaplan-Meier)  :", np.round(km_model.sf(t), 4))
    print("sf (Nelson-Aalen)  :", np.round(model.sf(t), 4))
    print("CIFs sum to 1 - KM :", np.allclose(
        km_model.cif(t, "wear") + km_model.cif(t, "shock"), km_model.ff(t)))

Like the single-cause step estimates, every function starts at its initial
value before the first observed time (``sf`` 1, the others 0) and holds its
last value after the last one, however far from the data. To give the
estimate an explicit range instead, call ``set_support(lower, upper)``: the
functions keep that convention only within ``[lower, upper]`` and are ``nan``
outside it. The bounds must contain the observed times; ``lower`` may be
negative and either bound infinite. ``set_support`` returns the model, and
``to_dict`` saves the bounds with it.

.. jupyter-execute::

    bounded = CompetingRisks.fit(x, e).set_support(0, 2 * x.max())
    q = [-1, 0, 1.5 * x.max(), 3 * x.max()]
    print("bounded CIF wear :", np.round(bounded.cif(q, "wear"), 4))
    print("CIF wear         :", np.round(model.cif(q, "wear"), 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _b = bounded.cif(q, "wear")
    assert np.isnan(_b[[0, 3]]).all() and _b[1] == 0
    assert _b[2] == model.cif(x.max(), "wear")

Confidence bounds
~~~~~~~~~~~~~~~~~

``cb(x, event)`` gives pointwise confidence bounds on a cause's cumulative
incidence: two-sided ``[lower, upper]`` rows by default, or one side with
``bound="lower"`` or ``"upper"``, at the level ``alpha_ci`` (0.05, a 95%
interval). The variance is Aalen's, the ``var`` that R's ``cmprsk::cuminc``
reports, and the interval is formed on the log(-log) scale so that it stays in
[0, 1] (``bound_type="normal"`` gives the plain estimate :math:`\pm` z
standard errors); see :doc:`Competing Risks Analysis` for the formula. With
``on="sf"``, ``"ff"`` or ``"Hf"`` it bounds the all-cause functions
(``event=None``) or a cause's net ones instead, as ``KaplanMeier`` or
``NelsonAalen`` (by ``how``) would for the same data.

.. jupyter-execute::

    t = np.array([25.0, 50.0, 100.0, 150.0])
    bounds = model.cb(t, "wear")
    for ti, f, (lo, hi) in zip(t, model.cif(t, "wear"), bounds):
        print(f"CIF wear at {ti:5.0f}: {f:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("all-cause sf bounds at 100:", np.round(model.cb(100.0, on="sf"), 3))

    plt.step(t_plot, model.cif(t_plot, "wear"), where="post", label="CIF wear")
    b = model.cb(t_plot[t_plot <= x.max()], "wear")
    plt.fill_between(t_plot[t_plot <= x.max()], b[:, 0], b[:, 1], step="post",
                     alpha=0.3, label="95% bounds")
    plt.xlabel("Time")
    plt.ylabel("Cumulative incidence")
    plt.legend()

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _f = model.cif(t, "wear")
    assert np.all((bounds[:, 0] <= _f) & (_f <= bounds[:, 1]))
    assert np.all((bounds >= 0) & (bounds <= 1))

Like the single-cause bounds, they are exactly the estimate's start (0, or 1
on ``sf``) before the first observed time, and NaN, with a warning, after the
last, where the estimate only holds its last value; with ``set_support`` they
are carried from the last time to ``upper`` instead.

``plot`` draws each cause's cumulative incidence as its own curve, with its
bounds as a pale band of its colour, as the single-event estimates' plots do
(``plot_bounds=False`` leaves them out; ``alpha_ci``, ``bound`` and
``bound_type`` are passed to ``cb``). ``stacked=True`` stacks the causes
instead, so that the top is the all-cause failure probability; a cause's
bounds do not bound its layer, so the bounds drawn are the top's, as two thin
grey lines:

.. jupyter-execute::

    model.plot()

Data held in a pandas DataFrame can be passed with ``fit_from_df``, naming the
time and cause columns (and optionally ``c_col`` and ``n_col``). The frame is
kept on the model as ``source_df``:

.. jupyter-execute::

    import pandas as pd

    df = pd.DataFrame({"time": x, "mode": e})
    model_df = CompetingRisks.fit_from_df(df, x_col="time", e_col="mode")
    np.allclose(model_df.cif(t, "shock"), model.cif(t, "shock"))

Parametric cumulative incidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ParametricCompetingRisks`` fits one parametric distribution to each cause
(the fit separates by cause, as explained on the theory page). The default is
a Weibull for every cause; pass a ``{cause: distribution}`` dict to choose per
cause. Here we use the distributions the data were simulated from:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import ParametricCompetingRisks

    pmodel = ParametricCompetingRisks.fit(
        x, e, dist={"wear": surv.Weibull, "shock": surv.Exponential}
    )
    print(pmodel)
    for k in pmodel.causes:
        print(k, pmodel.models[k].params)

The fitted per-cause models are ordinary SurPyval models in ``pmodel.models``
(true values: Weibull :math:`\alpha = 100, \beta = 3` and an exponential rate
of :math:`1/150 \approx 0.0067`; the sample has only 83 wear-out failures, most
of them below 100, so the Weibull scale carries the most sampling error).
The smooth parametric CIFs sit on top of the non-parametric steps:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    n_wear = int(np.sum(e == "wear"))
    assert n_wear == 83, n_wear
    assert np.mean(x[e == "wear"] < 100) > 0.5
    _rel = np.abs(np.r_[pmodel.models["wear"].params / [100, 3],
                        pmodel.models["shock"].params * 150] - 1)
    assert np.argmax(_rel) == 0, _rel      # the Weibull scale

.. jupyter-execute::

    for k, colour in [("wear", "C0"), ("shock", "C1")]:
        plt.step(t_plot, model.cif(t_plot, k), where="post", color=colour,
                 alpha=0.5, label=k + " (Aalen-Johansen)")
        plt.plot(t_plot, pmodel.cif(t_plot, k), color=colour,
                 label=k + " (parametric)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative incidence")
    plt.legend()

Unlike the step function, the parametric model can be evaluated beyond the
last observation. ``probability_of_cause`` gives :math:`F_k(\infty)`, the
long-run share of failures from each cause (these sum to one unless a cause
has a cure fraction):

.. jupyter-execute::

    for k in pmodel.causes:
        print(k, "eventual probability: %.3f" % pmodel.probability_of_cause(k))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.isclose(sum(pmodel.probability_of_cause(k)
                          for k in pmodel.causes), 1.0)

The other methods follow the same conventions as ``CompetingRisks``:
``hf``/``Hf`` take an optional ``event`` (cause-specific) and otherwise sum over
causes; ``sf`` and ``ff`` are the all-cause survival and failure probability,
and with ``event=k`` the net survival and failure probability of cause ``k``
alone (its own model, as if the other causes were removed);
``iif(x, event)`` is the sub-distribution density; and ``cif(x)`` without an
``event`` is the all-cause incidence ``1 - sf``. ``cif`` returns a float for a
scalar time:

.. jupyter-execute::

    print("CIF wear at 100    :", pmodel.cif(100.0, "wear"))
    print("all-cause ff at 100:", pmodel.ff(100.0))
    print("hazard of shock    :", pmodel.hf(100.0, event="shock"))

Because the joint likelihood is the product of the per-cause likelihoods,
``neg_ll`` and ``aic`` are sums over causes. ``bic`` is the joint criterion
``2 neg_ll + K ln(n)``, with ``K`` the parameters of every cause and ``n`` the
observed failures of any cause (here 189: 83 wear and 106 shock), not the sum
of the causes' own BICs. Either can be used to compare candidate
distributions -- for example, whether the shock mode needs a Weibull rather
than an exponential:

.. jupyter-execute::

    all_weibull = ParametricCompetingRisks.fit(x, e)   # Weibull for both causes
    print("Weibull + Exponential AIC: %.1f, BIC: %.1f"
          % (pmodel.aic(), pmodel.bic()))
    print("Weibull + Weibull     AIC: %.1f, BIC: %.1f"
          % (all_weibull.aic(), all_weibull.bic()))
    print("fitted shock shape:", all_weibull.models["shock"].params[1])

The shock mode's fitted Weibull shape is close to one (an exponential), and
the Weibull + Exponential model has the lower AIC and BIC: the extra shape
parameter is not worth its cost. ``ParametricCompetingRisks`` also has
a ``fit_from_df(df, x_col, e_col, c_col=None, n_col=None, dist=Weibull,
how="MLE")``; ``how`` is passed to each cause's distribution fit.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert abs(all_weibull.models["shock"].params[1] - 1) < 0.05
    assert pmodel.aic() < all_weibull.aic()
    assert pmodel.bic() < all_weibull.bic()
    assert sum(e == "wear") == 83 and sum(e == "shock") == 106
    assert np.isclose(pmodel.bic(), 2 * pmodel.neg_ll() + 3 * np.log(189))

Lifetime of the whole unit: MTTF and B-lives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A unit's life ends at its first failure from any cause, so the questions
"how long does a unit last on average?" (the mean time to failure, MTTF) and
"by when have 10% failed?" (the B10 life) are about the all-cause
lifetime. The fitted model answers them with ``mean()`` and ``qf(p)``, and
``df(x)`` gives the density of that lifetime.

In formulas: the all-cause survival is the product of the causes' survivals,
:math:`S(t) = \prod_k S_k(t)`, its density is
:math:`f(t) = S(t)\sum_k h_k(t)`, and

.. math::
    \mathrm{MTTF} = \int_0^\infty S(t)\,dt, \qquad
    B_{10} = F^{-1}(0.1) = H^{-1}\!\left(-\log 0.9\right),

with :math:`H = \sum_k H_k`. Neither has a closed form in general, so the
mean is integrated numerically and the quantile found by root finding on the
cumulative hazard (which keeps its precision far into the upper tail).

.. jupyter-execute::

    print("MTTF          : %.2f" % pmodel.mean())
    print("B10 life      : %.2f" % pmodel.qf(0.1))
    print("median life   : %.2f" % pmodel.qf(0.5))
    print("net B10, wear : %.2f" % pmodel.qf(0.1, event="wear"))

The net B10 of wear-out alone, ``qf(0.1, event="wear")``, is much later: it is
the B10 a unit would have if shocks could not happen. The all-cause figures are
the ones that describe the units actually in service.

If a cause has a cure (limited-failure) fraction, some units never fail, so
the MTTF is infinite and a quantile above the share that ever fails is
infinite too; ``mean(defective=True)`` gives :math:`\int t\,dF(t)` over the
units that fail, as ``Parametric.mean`` does.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _b10 = pmodel.qf(0.1)
    assert np.isclose(pmodel.ff(_b10), 0.1)
    assert pmodel.qf(0.1, event="wear") > _b10
    from scipy.integrate import quad
    _area = quad(lambda t: float(pmodel.sf(t)), 0, np.inf, limit=200)[0]
    assert np.isclose(_area, pmodel.mean(), rtol=1e-8)

How sure are these numbers? Bands and parameter uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every number above comes from parameters estimated from a sample, so each
has sampling error. ``cb`` gives a confidence band: with 95% confidence the
true cumulative incidence of a cause, or the all-cause survival, lies
between its two curves at each time. A narrow band says the data pin the
curve down; a wide one says more failures are needed.

.. jupyter-execute::

    for k, colour in [("wear", "C0"), ("shock", "C1")]:
        band = pmodel.cb(t_plot, k)          # on="cif" is the default
        plt.plot(t_plot, pmodel.cif(t_plot, k), color=colour, label=k)
        plt.fill_between(t_plot, band[:, 0], band[:, 1], color=colour,
                         alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Cumulative incidence (95% band)")
    plt.legend()

.. jupyter-execute::

    print("CIF wear at 100   : %.3f, 95%% band %s"
          % (pmodel.cif(100.0, "wear"), pmodel.cb(100.0, "wear").round(3)))
    print("survival at 100   : %.3f, 95%% band %s"
          % (pmodel.sf(100.0), pmodel.cb(100.0, on="sf").round(3)))

How the band is made: the fit's parameter covariance, ``covariance()``,
holds every cause's parameters, named by ``covariance_names``
(``"wear: alpha"``, ...). It is block diagonal, because each cause is
estimated from its own part of the likelihood. The delta method carries it
to the curve on a scale that keeps the band in range: the logit of the
cumulative incidence, and the log of the all-cause cumulative hazard for
``sf``, ``ff`` and ``Hf``. ``cb(x, event=k, on="sf")`` is cause ``k``'s net
survival, whose band is its own model's (likelihood-ratio bands too, with
``method="lr"``). A simulation study,
``scripts/studies/parametric_competing_risks_cb_coverage.py``, checks that
the 95% bands cover about 95% of the time.

For a number with no ``cb`` of its own, such as the MTTF or the B10 life,
draw parameter vectors from the covariance and rebuild the model at each
with ``with_params``; the spread of the number over the draws shows its
uncertainty. This rests on the same normal approximation as the bands:

.. jupyter-execute::

    rng = np.random.default_rng(1)
    draws = rng.multivariate_normal(pmodel._estimates(), pmodel.covariance(),
                                    size=300)
    mttf, b10 = [], []
    for d in draws:
        other = pmodel.with_params(d)
        mttf.append(other.mean())
        b10.append(other.qf(0.1))
    print("MTTF %.1f, 95%% range of the draws %s"
          % (pmodel.mean(), np.percentile(mttf, [2.5, 97.5]).round(1)))
    print("B10  %.1f, 95%% range of the draws %s"
          % (pmodel.qf(0.1), np.percentile(b10, [2.5, 97.5]).round(1)))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _band = pmodel.cb(100.0, "wear")
    assert _band[0] < pmodel.cif(100.0, "wear") < _band[1]
    assert np.percentile(mttf, 2.5) < pmodel.mean() < np.percentile(mttf, 97.5)
    assert np.percentile(b10, 2.5) < pmodel.qf(0.1) < np.percentile(b10, 97.5)
    assert pmodel.covariance_names[0] == "shock: failure_rate"

Assembling a model from separately fitted causes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ParametricCompetingRisks.from_fitted`` combines already-fitted single-cause
models -- a ``{cause: model}`` dict, or a list whose causes become ``0, 1,
...`` -- into one competing-risks model. Each model can use any SurPyval option
(a different family, an offset, a limited-failure or zero-inflated fit, ...).
The one rule: fit each model to the *cause-specific view* of the data, with
that cause's failures observed and every other row right-censored.

**Delayed entry (left truncation).** Suppose the components were only put
under observation at a random age ``entry`` (when the maintenance records
start, say), so that units which failed before entering were never seen.
Ignoring the entry ages biases the fit; ``ParametricCompetingRisks.fit`` takes
them as ``tl`` (and ``fit_from_df`` as ``tl_col``), which is the exact
maximum-likelihood fit: the truncated likelihood factorises by cause too, so it
is the same as fitting each cause with ``tl=entry`` and assembling the result
with ``from_fitted``. (Right truncation does not factorise by cause, and
``fit`` does not take it; interval censoring does not either, and ``fit``
takes it by fitting the causes together, below.)

.. jupyter-execute::

    truth = ParametricCompetingRisks.from_fitted({
        "wear": surv.Weibull.from_params([100.0, 3.0]),
        "shock": surv.Exponential.from_params([1 / 150]),
    })
    sim = truth.random(2000, random_state=4)
    entry = np.random.default_rng(5).uniform(0, 80, 2000)
    seen = sim["x"] > entry                      # only survivors to entry
    x_t, e_t, tl = sim["x"][seen], sim["e"][seen], entry[seen]

    naive = ParametricCompetingRisks.fit(
        x_t, e_t, dist={"wear": surv.Weibull, "shock": surv.Exponential}
    )

    adjusted = ParametricCompetingRisks.fit(
        x_t, e_t, dist={"wear": surv.Weibull, "shock": surv.Exponential},
        tl=tl,
    )

    print("true shock rate    : %.5f" % (1 / 150))
    print("ignoring entry     : %.5f" % naive.models["shock"].params[0])
    print("with truncation    : %.5f" % adjusted.models["shock"].params[0])
    print("P(wear) true / naive / adjusted: %.3f / %.3f / %.3f" % (
        truth.probability_of_cause("wear"),
        naive.probability_of_cause("wear"),
        adjusted.probability_of_cause("wear"),
    ))

Early failures -- which are disproportionately shocks -- were never observed,
so the naive fit underestimates the shock rate and overstates the share of
wear-out. The truncation-aware fit recovers both.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _rate, _pw = 1 / 150, truth.probability_of_cause("wear")
    assert naive.models["shock"].params[0] < 0.8 * _rate
    assert naive.probability_of_cause("wear") > _pw + 0.05
    assert abs(adjusted.models["shock"].params[0] / _rate - 1) < 0.05
    assert abs(adjusted.probability_of_cause("wear") - _pw) < 0.02
    # The same as fitting each cause's view with the entry ages.
    for _k, _dist in [("wear", surv.Weibull), ("shock", surv.Exponential)]:
        _alone = _dist.fit(x_t, c=np.where(e_t == _k, 0, 1), tl=tl)
        assert np.allclose(_alone.params, adjusted.models[_k].params)

Failures found at inspections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often a failure is not seen when it happens but found at the next
inspection: the unit was working at one inspection and found failed, with
its cause, at the next. Its failure time is known only to lie between the
two (interval censored, ``c = 2``, with ``x`` the row ``[xl, xr]``); one
found failed at the first inspection is left censored (``c = -1``, ``x`` the
inspection time). ``ParametricCompetingRisks.fit`` takes such rows. A row
like that counts the chance that the unit failed from its cause between
the two inspections, and that chance depends on every cause -- the unit had
to survive all of them until it failed -- so the causes can no longer be
fitted one at a time: with any such row, ``fit`` fits them together, by
maximum likelihood over all of their parameters at once. Without such rows
nothing changes.

.. jupyter-execute::

    rng = np.random.default_rng(6)
    sim = truth.random(400, random_state=7)
    end = rng.uniform(40, 300, 400)              # end of each unit's follow-up
    last = 20 * np.floor(end / 20)               # inspected every 20 hours
    found = sim["x"] <= last
    hi = 20 * np.ceil(sim["x"] / 20)             # the inspection that found it
    lo = hi - 20
    x_i = np.column_stack([np.where(found & (lo > 0), lo, np.where(found, hi, last)),
                           np.where(found, hi, last)])
    c_i = np.where(found, np.where(lo > 0, 2, -1), 1)
    e_i = np.where(found, sim["e"], None)

    inspected = ParametricCompetingRisks.fit(
        x_i, e_i, c=c_i, dist={"wear": surv.Weibull, "shock": surv.Exponential}
    )
    print(inspected.parameter_names)
    print(inspected.params.round(4))
    print("P(wear) true / fitted: %.3f / %.3f" % (
        truth.probability_of_cause("wear"),
        inspected.probability_of_cause("wear"),
    ))
    print(inspected.cb([50.0, 100.0], "wear").round(3))

Because the causes are fitted together, their estimates are correlated:
:meth:`~surpyval.univariate.competing_risks.ParametricCompetingRisks.covariance`
is the inverse of the joint Hessian, with terms between causes, and every
bound (``cb`` on ``cif``, ``sf``, ``ff`` and ``Hf``, ``standard_errors``) uses
it. Each cause's model in ``models`` carries its parameters and its block of
that covariance, so its own Wald bounds hold; it has no data of its own, so
its likelihood-ratio and bootstrap bounds, ``neg_ll`` and ``aic`` are the
model's, not the cause's. Only ``how="MLE"`` and continuous distributions
are supported, and a row in a window needs its cause. ``diagnose()`` does
not yet check such a fit (its checks need each failure's exact time) and
says so; compare distributions by their ``aic()``, and the all-cause
survival ``sf`` with ``surpyval.Turnbull.fit`` of the same windows. Saved
with ``to_dict``, such a fit keeps the covariance between causes and is
stamped schema 4, which SurPyval 0.26 and older refuse, asking for an
upgrade, rather than load it without those terms.

In formulas: a unit that entered at :math:`t_l` and was found failed from
cause :math:`k` in :math:`(x_l, x_r]` contributes

.. math::
    \frac{\mathrm{CIF}_k(x_r) - \mathrm{CIF}_k(x_l)}{S(t_l)}
    = \frac{1}{S(t_l)} \int_{x_l}^{x_r} f_k(u) \prod_{j \neq k} S_j(u)\,du,

a left-censored one the same from :math:`t_l` (or the start) to :math:`x`.
The integral is taken by Gauss-Legendre quadrature on the cause's own
probability scale, refined until it holds to about :math:`10^{-12}`.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert inspected.maximum == "verified"
    assert abs(inspected.probability_of_cause("wear")
               - truth.probability_of_cause("wear")) < 0.05
    assert abs(inspected.covariance()[0, 2]) > 0

Simulating competing-risks data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous example already used ``random``. It draws a latent time from each
cause's distribution (by inverse-transform sampling through its ``qf``) and
keeps the earliest, returning a structured array with fields ``x`` (time) and
``e`` (cause). A unit whose latent times are all infinite (possible with cure
models) never fails and is returned with ``x = inf`` and cause ``None``.
Censoring is not part of the model, so add it yourself when simulating a
study:

.. jupyter-execute::

    draws = truth.random(5, random_state=1)
    print(draws)
    print(draws["x"], draws["e"])

A small Monte-Carlo study: how well does a 150-unit, censored test estimate
the long-run share of wear-out failures?

.. jupyter-execute::

    rng = np.random.default_rng(2)
    estimates = []
    for _ in range(30):
        s = truth.random(150, random_state=rng.integers(1_000_000))
        cens = rng.uniform(0, 200, 150)
        x_sim = np.minimum(s["x"], cens)
        e_sim = np.where(s["x"] <= cens, s["e"], None)
        fit = ParametricCompetingRisks.fit(
            x_sim, e_sim, dist={"wear": surv.Weibull, "shock": surv.Exponential}
        )
        estimates.append(fit.probability_of_cause("wear"))
    print("true P(wear) = %.3f;  estimates: mean %.3f, sd %.3f" % (
        truth.probability_of_cause("wear"), np.mean(estimates), np.std(estimates)))

Saving and loading competing-risks models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every competing-risks model -- ``CompetingRisks``,
``ParametricCompetingRisks``, the Fine-Gray model returned by ``FineGray.fit``
and ``CompetingRisksProportionalHazards`` -- can be serialised with
``to_dict``/``to_json`` and restored with the class's ``from_dict``/``from_json``
or with the package-level ``surpyval.from_dict``/``surpyval.from_json``, which
work out the class from the dictionary. The restored model reproduces every
prediction exactly:

.. jupyter-execute::

    import json

    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    print(type(restored).__name__,
          np.allclose(restored.cif(t, "wear"), model.cif(t, "wear")))

    restored_p = surv.from_dict(pmodel.to_dict())
    print(type(restored_p).__name__, restored_p.cif(100.0, "wear"))


Comparing incidence across groups: Gray's test
----------------------------------------------

Having estimated a cumulative incidence function for each group,
``surpyval.gray_test`` tests whether the CIFs *differ* for a chosen cause. It is
the
competing-risks analogue of the log-rank test, but with an important
distinction: where a cause-specific log-rank compares the instantaneous
*hazards* of a cause, Gray's test compares the *incidence* — the CIFs directly.
It does this by keeping competing-cause failures in the subdistribution risk
set, rather than removing them, with each group's risk set estimated from that
group's own data (and so its own censoring distribution).
Reach for it when the clinical or engineering question is "how many fail of this
cause", not "how fast".

Pass the observed times ``x``, the per-observation cause label ``e``, the group
label, and the ``cause`` of interest; optional ``n`` gives row counts.
Censored rows follow the same rules as for the model classes (see
`Competing-risks data format`_): a missing cause (``None`` or ``NaN``) marks a
censored row, and an optional ``c`` must agree with it.

Here two groups have genuinely different cause-1 incidence:

.. jupyter-execute::

    from surpyval import gray_test

    rng = np.random.default_rng(7)

    def simulate(n, p_cause1):
        is1 = rng.random(n) < p_cause1
        t = rng.exponential(6.0, n)
        return t, np.where(is1, 1, 2)         # causes labelled 1 and 2

    x_a, e_a = simulate(300, 0.35)
    x_b, e_b = simulate(300, 0.60)            # higher cause-1 incidence
    x = np.concatenate([x_a, x_b])
    e = np.concatenate([e_a, e_b])
    group = np.array([0] * 300 + [1] * 300)

    result = gray_test(x, e, group, event=1)
    print('statistic = %.2f   df = %d   p = %.3g'
          % (result.statistic, result.df, result.p_value))

The tiny ``p``-value correctly flags the difference in cause-1 incidence. The
result is a named tuple ``(statistic, df, p_value, cause, groups)``; ``df`` is
the number of groups minus one, so more than two groups are compared in one
test. The statistic is Gray's (1988): an observed-minus-expected count of
cause-1 failures on the subdistribution risk sets, with Gray's asymptotic
variance (the :doc:`Competing Risks Analysis` page gives the formulas).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert result.p_value < 1e-6, result.p_value
    assert result.df == 1

Calibration
~~~~~~~~~~~

On data where the groups share the same incidence the test is calibrated,
returning ``p``-values spread over ``[0, 1]`` — including under censoring,
and when the groups are censored differently. A quick check: simulate many
pairs of groups with identical cause-specific hazards and independent
censoring, and count how often ``p < 0.05``. The second pair of groups is
censored very differently, exponentially with means 2 and 50:

.. jupyter-execute::

    rng = np.random.default_rng(3)

    def simulate_cr(n, h1, h2, cens_mean):
        t1 = rng.exponential(1 / h1, n)       # latent cause-1 time
        t2 = rng.exponential(1 / h2, n)       # latent cause-2 time
        cz = rng.exponential(cens_mean, n)    # censoring time
        x = np.minimum.reduce([t1, t2, cz])
        e = np.where(x == cz, None, np.where(t1 < t2, 1, 2)).astype(object)
        return x, e

    def rejection_rate(cens_mean_0, cens_mean_1, n=100, reps=200):
        p_values = []
        for _ in range(reps):
            x0, e0 = simulate_cr(n, 0.1, 0.2, cens_mean_0)
            x1, e1 = simulate_cr(n, 0.1, 0.2, cens_mean_1)
            res = gray_test(np.concatenate([x0, x1]),
                            np.concatenate([e0, e1]),
                            np.repeat([0, 1], n), event=1)
            p_values.append(res.p_value)
        return np.mean(np.array(p_values) < 0.05)

    same = rejection_rate(10.0, 10.0)
    different = rejection_rate(2.0, 50.0)
    print("same censoring:      rejection rate at 5%%: %.3f" % same)
    print("different censoring: rejection rate at 5%%: %.3f" % different)

Both rejection rates are close to the nominal 5%, as they should be when the
null hypothesis is true.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert abs(same - 0.05) < 0.03 and abs(different - 0.05) < 0.03

Gray's test versus a cause-specific log-rank
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The difference between "how fast" and "how many" is easiest to see in an
example. Give two groups the *same* cause-1 hazard, but a much higher cause-2
hazard in group B. Cause 1 strikes at the same rate among the survivors in
both groups, but in group B most units are removed by cause 2 before cause 1
has a chance, so far fewer of them ever fail from cause 1. A cause-specific
log-rank (``surpyval.logrank`` with cause 2 treated as censored) compares the
rates and finds nothing; Gray's test compares the incidence and finds the
difference:

.. jupyter-execute::

    from surpyval import logrank

    x_a, e_a = simulate_cr(300, 0.1, 0.05, 20.0)    # group A
    x_b, e_b = simulate_cr(300, 0.1, 0.30, 20.0)    # group B: more cause 2
    x = np.concatenate([x_a, x_b])
    e = np.concatenate([e_a, e_b])
    group = np.repeat(["A", "B"], 300)

    cs = logrank(x, group, c=np.where(e == 1, 0, 1))
    gray = gray_test(x, e, group, event=1)
    print("cause-specific log-rank p = %.3f" % cs.p_value)
    print("Gray's test             p = %.2g" % gray.p_value)

    t_grid = np.linspace(0, 40, 400)
    for g, xs, es in [("A", x_a, e_a), ("B", x_b, e_b)]:
        cr = CompetingRisks.fit(xs, es)
        plt.step(t_grid, cr.cif(t_grid, 1), where="post", label="group " + g)
    plt.xlabel("Time")
    plt.ylabel("Cumulative incidence of cause 1")
    plt.legend()

Neither answer is wrong. If the question is whether the groups differ in the
mechanism behind cause 1, the log-rank is the relevant test; if it is whether
they differ in how many units end up failing from cause 1, it is Gray's.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert cs.p_value > 0.1 and gray.p_value < 1e-4

The ``rho`` argument weights each event time by
:math:`\{1 - \hat{F}^0(t^-)\}^{\rho}`, where :math:`\hat{F}^0` is Gray's
pooled CIF of the cause. The default ``rho=0`` is the standard test; ``rho > 0`` emphasises
differences in *early* incidence:

.. jupyter-execute::

    print("rho = 1: p = %.2g" % gray_test(x, e, group, event=1, rho=1.0).p_value)


Time lost to a cause: the restricted mean time lost
---------------------------------------------------

The cumulative incidence answers "what fraction has failed from this cause
by time t". A single number that summarises the whole curve up to a
horizon is the **restricted mean time lost** (RMTL): of the first
:math:`\tau` units of time, how much, on average, a unit loses to that
cause. A unit that fails from the cause at time :math:`t < \tau` loses
:math:`\tau - t`; a unit that survives to :math:`\tau`, or fails from
another cause, loses nothing to it. It is the area under the cause's
cumulative incidence,

.. math::
    \text{RMTL}_k(\tau) = \int_0^\tau F_k(t)\,dt,

so it is in units of time and needs no model for how the causes act
(Andersen, 2013). The times lost to all the causes add up to
:math:`\tau` less the restricted mean survival time (``rmst``), so with
one cause the time lost is just :math:`\tau - \text{RMST}`.

``rmtl(event, tau)`` gives it with a standard error and a 95% interval;
``tau`` defaults to the largest observed time:

.. jupyter-execute::

    import numpy as np
    import surpyval as sp

    rng = np.random.default_rng(3)

    def simulate(n, shock_mean):
        wear = 100.0 * rng.weibull(3.0, n)
        shock = rng.exponential(shock_mean, n)
        censor = rng.uniform(0.0, 250.0, n)
        x = np.minimum.reduce([wear, shock, censor])
        e = np.where(x == censor, None, np.where(wear < shock, "wear", "shock"))
        return x, e.astype(object)

    old = sp.CompetingRisks.fit(*simulate(200, 150.0))
    new = sp.CompetingRisks.fit(*simulate(200, 600.0))
    res = old.rmtl("shock", tau=100)
    print("time lost to shocks in the first 100 hours: %.1f (%.1f to %.1f)"
          % (res["rmtl"], res["lower"], res["upper"]))

``surpyval.rmtl_diff`` compares the time lost to a cause between two
groups, as ``rmst_diff`` compares survival: the difference with its interval
and a z-test of no difference, and the ratio with its interval. Here a
design change that makes shocks four times rarer saves many hours in the first 100:

.. jupyter-execute::

    cmp = sp.rmtl_diff(old, new, "shock", tau=100)
    print("difference %.1f hours (%.1f to %.1f), p = %.2g"
          % (cmp["difference"], cmp["lower"], cmp["upper"], cmp["p_value"]))
    print("ratio %.2f (%.2f to %.2f)"
          % (cmp["ratio"], cmp["ratio_lower"], cmp["ratio_upper"]))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert 0 < res["lower"] < res["rmtl"] < res["upper"] < 100
    assert cmp["difference"] > 0 and cmp["p_value"] < 0.05

The standard error is the delta method's on the hazard increments at each
event time (a multinomial at each time, as Greenwood's formula takes a
binomial); with one cause it is exactly the Greenwood-based standard error of
``KaplanMeier.rmst``. It is the standard error of R's ``GFDrmtl`` package,
and within 1% of R ``survival``'s ``summary(fit, rmean = tau)`` (an
infinitesimal-jackknife standard error) on the data checked. The interval is ``rmtl +- 1.96 se``; in simulation (n = 200) it
covers the true time lost about 95% of the time, and ``rmtl_diff``'s test
rejects about 5% of the time when the groups do not differ. A ``tau`` past
the last observed time is refused while some units are still event-free
there, as for ``rmst``. A model fitted with weights refuses ``rmtl``: its
robust variance is not built.

``ParametricCompetingRisks.rmtl`` gives the same quantity from the fitted
distributions, by numerical integration of the cumulative incidence, with
a delta-method standard error over every cause's parameters and an interval
on the logit of ``rmtl / tau`` (so within 0 and ``tau``); a ``tau`` past
the data is allowed, the fit extrapolating. ``rmtl_diff`` accepts either
kind of model in either position.

Fine-Gray Sub-distribution Hazards
----------------------------------

The Fine-Gray model estimates the effect of covariates directly on the
cumulative incidence function of a chosen cause, using an
inverse-probability-of-censoring-weighted subdistribution risk set. ``e`` is
the per-observation cause label (``None`` for a censored row) and ``cause``
selects the cause of interest. ``Z`` is the covariate matrix, one row per
observation.

Here we simulate two-cause data whose cause-1 incidence follows a Fine-Gray
model with coefficients :math:`(0.7, -0.4)`, apply right-censoring, and recover
the coefficients. (This is the simulation design of Fine and Gray's paper: the
true CIF of cause 1 is :math:`F_1(t \mid Z) = 1 - \{1 - p(1 - e^{-t})\}^{\exp(Z\beta)}`
with :math:`p = 0.5`.)

.. jupyter-execute::

    from surpyval.univariate.competing_risks import FineGray

    rng = np.random.default_rng(1)
    N, beta, p = 800, np.array([0.7, -0.4]), 0.5
    Z = rng.uniform(-1, 1, size=(N, 2))
    phi = np.exp(Z @ beta)
    p1 = 1 - (1 - p) ** phi                    # P(cause = 1 | Z)
    is1 = rng.uniform(size=N) < p1

    x = np.empty(N)
    e = np.empty(N, dtype=object)
    v = rng.uniform(size=N)
    w = 1 - (1 - v * p1) ** (1 / phi)          # invert the cause-1 CIF
    x[is1] = (-np.log(np.clip(1 - w / p, 1e-12, 1.0)))[is1]
    e[is1] = 1
    x[~is1] = rng.exponential(1.0, size=N)[~is1]   # cause 2 mops up the rest
    e[~is1] = 2

    cens = rng.exponential(3.0, size=N)        # independent right-censoring
    c = (x > cens).astype(int)
    x = np.minimum(x, cens)
    e[c == 1] = None

    model = FineGray.fit(x, Z, e, c=c, event=1)
    model

The IPCW correction is what lets the coefficients come back near their true
values *under* censoring — a naive unweighted subdistribution risk set would be
biased. The fitted model stores the estimates as arrays, one entry per column
of ``Z``; ``np.exp(model.beta)`` gives the sub-distribution hazard ratios:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    # "come back near their true values": within two standard errors
    assert np.all(np.abs(model.beta - beta) < 2 * model.standard_errors()), model.beta

.. jupyter-execute::

    print("beta     :", np.round(model.beta, 3))     # also model.coefficients
    print("se       :", np.round(model.standard_errors(), 3))
    print("p-values :", model.p_values)
    print("SHR      :", np.round(np.exp(model.beta), 3))
    print("cov      :\n", np.round(model.covariance(), 4))

The standard errors, and the p-values and ``summary()`` built on them, are
the robust ones Fine and Gray derived (a sandwich variance that allows for
the censoring weights being estimated from the same data), which is what
R's ``cmprsk::crr`` prints. The model-based ones, from the curvature of the
weighted partial likelihood alone, are there on request, with
``robust=False``:

.. jupyter-execute::

    print("robust se      :", np.round(model.standard_errors(), 4))
    print("model-based se :", np.round(model.standard_errors(robust=False), 4))
    model.summary(robust=False)[["coef", "se(coef)", "p"]]

``covariance()`` and ``covariance(robust=False)`` are the two matrices; see
:doc:`Competing Risks Analysis` for the formulas. ``log_likelihood`` is
the maximised weighted partial log-likelihood (``cmprsk::crr``'s ``loglik``),
whose ``aic()`` and ``bic()`` (with the events of the cause as BIC's sample
size) compare Fine-Gray models of the same cause on the same data; a
``CompetingRisksProportionalHazards`` fitted with ``model="Fine-Gray"`` has
none, as its causes' partial likelihoods are not parts of one likelihood,
while one fitted with ``model="Cox"`` has the sum of its cause-specific
partial likelihoods, R's multi-state ``coxph``'s. Because the model targets the incidence directly, ``cif``
reads off the cumulative incidence of the cause at any covariate value (one
covariate vector per call). The dashed lines are the true CIFs:

.. jupyter-execute::

    t = np.linspace(0, 3, 200)
    for z1, label, colour in [(-1.0, 'Z1 = -1', 'C0'), (1.0, 'Z1 = +1', 'C1')]:
        z = np.array([z1, 0.0])
        plt.plot(t, model.cif(t, Z=z), color=colour, label=label)
        plt.plot(t, 1 - (1 - p * (1 - np.exp(-t))) ** np.exp(z @ beta),
                 '--', color=colour)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Cumulative incidence of cause 1')

A positive :math:`\beta_0` raises the cause-1 incidence, so the ``Z1 = +1``
curve sits above ``Z1 = -1``. The fitted CIF is a step function built on the
observed cause-1 event times, so it is flat after the last of them (about
:math:`t = 5.6` in this sample) rather than extrapolating. ``sf(x, Z)`` returns
``1 - cif(x, Z)`` and ``phi(Z)`` the multiplier :math:`e^{Z\beta}`. As the
Cox fit does, the Fine-Gray fit centres the covariates on their means, which
leaves the coefficients and every prediction unchanged but keeps
:math:`e^{Z\beta}` from overflowing on a covariate far from zero; the
baseline is then reported at :math:`Z = 0`, or, with ``center=True``, at the
means (``model.center``, and ``phi`` is relative to them).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.all(model.cif(t[1:], Z=np.array([1.0, 0.0]))
                  > model.cif(t[1:], Z=np.array([-1.0, 0.0])))
    _last = np.max(x[(e == 1) & (c == 0)])
    assert round(_last, 1) == 5.6, _last

Things to watch:

- ``cause`` must be given when the data contain more than one cause, and it
  must be one that is observed; with a single cause it may be omitted.
- The Fine-Gray model is fitted for one cause at a time. To model every cause,
  use ``CompetingRisksProportionalHazards`` with ``model="Fine-Gray"`` (below);
  the separate fits are not constrained to be mutually consistent.
- Censoring times tied with event times (common when times are recorded in
  whole days or months) follow R's ``cmprsk::crr``: the event is taken to
  come first. A unit that failed from a competing cause at :math:`x_i` keeps
  the weight :math:`\hat{G}(t^-)/\hat{G}(x_i^-)` at a later event time
  :math:`t`, with :math:`\hat{G}` the Kaplan-Meier estimate of the censoring
  survival taken just before each time, so censorings at :math:`t` or at
  :math:`x_i` do not count against the events there. These are ``crr``'s
  weights; without such ties the left limits are just :math:`\hat{G}(t)`
  and :math:`\hat{G}(x_i)`. See
  :doc:`Competing Risks Analysis` for the details.

The fitted model serialises like the other competing-risks models; the
optimiser result (``model.res``) is not stored:

.. jupyter-execute::

    reloaded = surv.from_dict(model.to_dict())
    np.allclose(reloaded.cif([0.5, 1.0], Z=[1.0, 0.0]),
                model.cif([0.5, 1.0], Z=[1.0, 0.0]))

Strata and clusters
~~~~~~~~~~~~~~~~~~~

Two common departures from the plain model have their own options.

**Strata.** When the units come from a few groups (sites, centres,
production lines) whose cause-1 incidence has a different shape over time,
or whose censoring differs, ``strata=`` gives each group its own baseline
subdistribution hazard and its own censoring distribution, while the
covariates' effects are shared (Zhou, Latouche, Rocha and Fine, 2011; R's
``crrSC::crrs``). This is the competing-risks counterpart of a stratified
Cox model: use it for a factor you need to allow for but do not need an
effect for. Predictions then name the stratum:

.. jupyter-execute::

    site = np.where(np.arange(N) % 3 == 0, "east", "west")
    by_site = FineGray.fit(x, Z, e, c=c, event=1, strata=site)
    print(by_site.beta.round(3), by_site.standard_errors().round(3))
    print(by_site.cif([0.5, 1.0], Z=[1.0, 0.0], stratum="east").round(3))

With many small strata (a stratum per centre with a handful of units each)
a censoring distribution per stratum is poorly estimated; then
``strata_censoring="common"`` estimates one for all the strata, each still
with its own baseline (``crrs``'s ``ctype = 2``).

**Clusters.** When units are not independent of each other within a group
(the two eyes or kidneys of a patient, the pumps of one station, the
components of one machine), ``cluster=`` keeps the estimates and makes the
standard errors allow for the dependence: the robust covariance adds up
the influence of a cluster's units before squaring it (Zhou, Fine,
Latouche and Labopin, 2012; R's ``crrSC::crrc``). Without it, a covariate
shared by the units of a cluster has standard errors that are too small.

.. jupyter-execute::

    pair = np.arange(N) // 2
    paired = FineGray.fit(x, Z, e, c=c, event=1, cluster=pair)
    print(paired.beta.round(3), paired.standard_errors().round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(paired.beta, model.beta)
    assert by_site.cif(1.0, Z=[1.0, 0.0], stratum="east") > 0

Both combine with each other and with weights ``w``, and ``fit_from_df``
takes them as ``strata_col`` and ``cluster_col``. R's ``crrs`` with
``ctype = 2`` treats each stratum as a cluster too: its standard errors are
those of ``strata_censoring="common", cluster=<the strata>``. The
model-based covariance (``covariance(robust=False)``) ignores the clusters,
and the diagnostics (``diagnose()``, the proportionality tests and the
residuals) refuse a fit with strata or clusters, as their tests take one
baseline and independent units: check proportionality on the fit without
them, or within each stratum.

Cause-Specific Proportional Hazards
-----------------------------------

``CompetingRisksProportionalHazards`` fits a proportional-hazards model per
cause. With ``model="Cox"`` (the default) each cause is a Cox model with the
other causes treated as censored; ``model="Fine-Gray"`` fits the subdistribution
model above for every cause. It reuses the simulated data from the previous
section:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    csph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, model="Cox")
    # cumulative incidence of cause 1 at a covariate vector
    csph.cif(np.array([0.5, 1.0, 2.0]), Z=[0.5, -0.5], event=1)

The cause-specific model answers "what drives the rate of this cause among those
still at risk", while Fine-Gray answers "what drives the eventual incidence of
this cause"; the two coincide only when the competing causes are unaffected by
the covariates.

Coefficients and predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitted coefficients are in ``betas``, one row per cause. Look the row up
through ``event_idx_map`` rather than assuming an order:

.. jupyter-execute::

    for cause, row in csph.event_idx_map.items():
        print("cause", cause, "cause-specific log hazard ratios:",
              np.round(csph.betas[row], 3))

In this simulation :math:`Z_1` raises the cause-1 *incidence* (the Fine-Gray
coefficient is +0.7), and it does so by raising the cause-1 hazard and
lowering the cause-2 hazard: the two sets of cause-specific coefficients
together produce the incidence effect.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _b = {k: csph.betas[r] for k, r in csph.event_idx_map.items()}
    assert _b[1][0] > 0 and _b[2][0] < 0, _b

The causes are sorted, so the row order of ``betas`` is reproducible.
``phi_e(Z, event)`` is a cause's hazard multiplier :math:`e^{Z\hat\beta_k}`
(relative to a unit at the covariate means, ``center``, for a fit with
``center=True``, which keeps the baselines there), the cause given by its
label as for ``cif``; and ``results`` holds each cause's optimiser result.
The model also has ``phi``, kept for backward compatibility, and ``beta``,
deprecated: ``beta`` is the *sum* of the rows of ``betas``, which is not a
quantity of the model, and no prediction uses it. Read the coefficients from
``betas``, or from ``params`` (flattened cause by cause).

.. jupyter-execute::

    print("cause-1 hazard multiplier at z = [0.5, -0.5]:",
          np.round(csph.phi_e(np.array([0.5, -0.5]), 1), 3))

``summary()`` is the coefficient table, one row per cause and covariate,
named ``"<cause>: <covariate>"`` (``parameter_names``), with the standard
errors, Wald intervals and p-values of each cause's fit; ``covariance()`` is
block-diagonal across the causes, as their fits are separate:

.. jupyter-execute::

    csph.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(4)

``tie_method`` chooses how the Cox fits handle tied failure times (see
:doc:`regression/cox_ph`): ``"efron"`` (the default, as for ``CoxPH``), ``"breslow"``, ``"exact"`` or
``"kalbfleisch-prentice"`` (``"kp"``). The simulated times are continuous, so
there are no ties and every method gives the same fit. Rounding the times up
to the next 0.25 creates heavy ties (the largest is 107 cause-1 failures at a
single time):

.. jupyter-execute::

    x_tied = np.ceil(x * 4) / 4
    tied = {}
    for tm in ["breslow", "efron", "exact", "kp"]:
        fit_t = CompetingRisksProportionalHazards.fit(x_tied, Z, e, c=c,
                                                      tie_method=tm)
        tied[tm] = fit_t.betas[row]
        print("%-8s cause 1: %s" % (tm, np.round(tied[tm], 3)))

``"exact"`` averages the partial likelihood over every order in which the
tied failures could have happened, which is the right treatment when the ties
come from rounding a continuous time, as here. Efron's approximation is very
close to it; Breslow's pulls the coefficients towards zero. ``"kp"`` fits a
different model, one in which time is genuinely discrete, so its coefficients
are log *odds* ratios rather than log hazard ratios and come out larger. Every
method is fast, even with ties this heavy.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = x_tied[(e == 1) & (c == 0)]
    _largest = np.max(np.unique(_t, return_counts=True)[1])
    assert _largest == 107, _largest
    assert np.all(np.abs(tied["efron"] - tied["exact"]) < 0.01)
    assert np.all(np.abs(tied["breslow"]) < np.abs(tied["exact"]))
    assert np.all(np.abs(tied["kp"]) > np.abs(tied["exact"]))

With ``model="Cox"`` the model has the usual functions, each taking the times,
one covariate vector ``Z`` and an optional ``event``:

- ``cif(x, Z, event)`` -- the cumulative incidence of ``event`` at ``Z``:
  over each step :math:`x_j` the unit, event-free with probability
  :math:`\hat{S}(x_{j-1} \mid Z) = e^{-H}`, fails with probability
  :math:`1 - e^{-\Delta H}`, and cause :math:`k` takes the share
  :math:`\Delta H_k / \Delta H` of it (the matrix-exponential form R's
  ``survival`` uses for a multi-state ``coxph``);
- ``Hf``/``hf`` -- the cause-specific cumulative hazard and its increment at
  the most recent event time; with ``event=None`` they are summed over causes,
  each cause with its own coefficients;
- ``sf``/``ff``/``df`` -- :math:`e^{-H}`, :math:`1 - e^{-H}` and ``hf * sf``
  of those hazards. ``sf(x, Z)`` with no ``event`` is the all-cause survival;
  with an ``event`` it is the net (cause-alone) survival, not a probability of
  the real-world outcome.

.. jupyter-execute::

    times = np.array([0.5, 1.0, 2.0])
    z = [0.5, -0.5]
    cif1 = csph.cif(times, Z=z, event=1)
    cif2 = csph.cif(times, Z=z, event=2)
    print("CIF cause 1          :", np.round(cif1, 4))
    print("CIF cause 2          :", np.round(cif2, 4))
    print("sum of CIFs          :", np.round(cif1 + cif2, 4))
    print("1 - all-cause sf     :", np.round(1 - csph.sf(times, Z=z), 4))

The CIFs add up to the all-cause failure probability ``1 - sf``, exactly, so
their total never exceeds one.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _gap = (cif1 + cif2) - (1 - csph.sf(times, Z=z))
    assert np.all(np.abs(_gap) < 1e-12), _gap

With ``model="Fine-Gray"``, ``cif``, ``sf`` (``1 - cif``), ``ff`` and ``Hf`` need
an ``event`` and come from each cause's Fine-Gray model; ``hf`` and ``df``
raise a ``ValueError`` because the step baseline has no pointwise density.
Comparing the two fits on the same data:

.. jupyter-execute::

    fg_all = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, model="Fine-Gray")
    for cause, row in fg_all.event_idx_map.items():
        print("cause", cause, "Fine-Gray coefficients:",
              np.round(fg_all.betas[row], 3))
    print("Fine-Gray CIF of cause 1:",
          np.round(fg_all.cif(times, Z=z, event=1), 4))

The Fine-Gray coefficients for cause 1 match ``FineGray.fit`` above. For cause
2 they have the opposite sign to cause 1, even though the covariates were only
built into the cause-1 incidence: whatever raises the incidence of one cause
necessarily lowers the incidence of the other.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _fg = {k: fg_all.betas[r] for k, r in fg_all.event_idx_map.items()}
    assert np.allclose(_fg[1], model.beta)
    assert np.all(np.sign(_fg[2]) == -np.sign(_fg[1])), _fg

Both kinds of fit can be saved and restored like the other competing-risks
models. The dictionary holds the per-cause coefficients and baselines (and,
for ``model="Fine-Gray"``, each cause's Fine-Gray model); the optimiser results
in ``results`` are not stored, so a restored model has ``results = None``:

.. jupyter-execute::

    for fitted in [csph, fg_all]:
        reloaded = surv.from_dict(json.loads(json.dumps(fitted.to_dict())))
        print(fitted.model, type(reloaded).__name__,
              np.array_equal(reloaded.cif(times, Z=z, event=1),
                             fitted.cif(times, Z=z, event=1)))

Delayed entry in the cause-specific model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Records often start after the units did: patients enter a registry at
diagnosis, machines enter the maintenance system when it goes live. A unit
that failed before it entered is never seen, and a unit that entered at age
3 was not at risk, as far as the data know, before age 3. Pass each row's
entry age as ``tl`` (``tl_col`` for ``fit_from_df``) and the cause-specific
model counts a unit in a risk set only once it has entered, ``tl < t <= x``,
as ``CoxPH`` does; every cause's coefficients and baseline, and so the
incidences, use those risk sets. Here the units come under observation at a
uniform age between 0 and 4, and only those still working then are seen:

.. jupyter-execute::

    rng = np.random.default_rng(11)
    z_u = rng.binomial(1, 0.5, 3000).astype(float)
    t_wear = rng.exponential(1 / (0.2 * np.exp(0.8 * z_u)))  # log HR 0.8
    t_shock = rng.exponential(1 / 0.1, 3000)
    x_u = np.minimum(t_wear, t_shock)
    e_u = np.where(t_wear < t_shock, "wear", "shock")
    entry = rng.uniform(0, 4, 3000)
    seen = x_u > entry                       # only units working at entry
    x_s, z_s, e_s, tl = x_u[seen], z_u[seen, None], e_u[seen], entry[seen]

    naive = CompetingRisksProportionalHazards.fit(x_s, z_s, e_s)
    entered = CompetingRisksProportionalHazards.fit(x_s, z_s, e_s, tl=tl)
    true_cif = 0.2 * np.exp(0.8) / (0.2 * np.exp(0.8) + 0.1) * (
        1 - np.exp(-2 * (0.2 * np.exp(0.8) + 0.1)))
    print("wear log HR, ignoring entry / with tl: %.3f / %.3f" % (
        naive.betas[naive.event_idx_map["wear"], 0],
        entered.betas[entered.event_idx_map["wear"], 0]))
    print("P(wear by 2 | z = 1), true / ignoring entry / with tl: "
          "%.3f / %.3f / %.3f" % (true_cif, naive.cif(2, [1], "wear"),
                                  entered.cif(2, [1], "wear")))

Ignoring the entry ages counts the late entrants at risk from age 0, when
they could not have been seen to fail, so the baseline hazards come out far
too low and the incidence of wear by age 2 is about half the truth. With
``tl`` it is recovered. The estimates are conditional on being event-free at
the earliest entry, as for ``KaplanMeier`` with ``tl``.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert naive.cif(2, [1], "wear") < 0.6 * true_cif
    assert abs(entered.cif(2, [1], "wear") - true_cif) < 0.06
    assert abs(entered.betas[entered.event_idx_map["wear"], 0] - 0.8) < 0.1

The Fine-Gray model (``model="Fine-Gray"``, ``FineGray``) and Gray's test
refuse ``tl`` where an entry is after the first time, with a message naming
the cause-specific model and the Aalen-Johansen estimate: their
subdistribution risk set keeps the units that already failed from another
cause, and with late entry it needs truncation weights, which SurPyval does
not build (see :doc:`Competing Risks Analysis`).

Fitting from a DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~

``fit_from_df`` takes the time and cause column names, and the covariates
either as ``Z_cols`` (a column name or list of names) or as a ``formula``;
``c_col``, ``n_col``, ``model`` and ``tie_method`` (default ``"efron"``, passed to
each cause's Cox fit) are optional. A blank/``NaN`` cause marks a censored row,
and rows with a missing (or infinite) covariate are dropped, with a warning
giving the count -- as are rows of ``Z`` containing ``NaN`` in ``fit``, for
both the ``Cox`` and ``Fine-Gray`` fits and for ``FineGray.fit``.
The fitted model predicts from a DataFrame of the covariate columns, read by
name (their order and any other columns do not matter), or still from an
array ``Z`` in the fitted column order:

.. jupyter-execute::

    frame = pd.DataFrame({"time": x, "cause": e, "z1": Z[:, 0], "z2": Z[:, 1]})
    csph_df = CompetingRisksProportionalHazards.fit_from_df(
        frame, x_col="time", e_col="cause", Z_cols=["z1", "z2"]
    )
    print(csph_df.feature_names)
    new = pd.DataFrame({"z2": [z[1]], "z1": [z[0]]})
    print(np.allclose(csph_df.cif(times, Z=new, event=1), cif1),
          np.allclose(csph_df.cif(times, Z=z, event=1), cif1))

With a ``formula`` the prediction DataFrame holds the *raw* covariates and the
fitted formula expands them -- categorical levels coded against the fitted
reference level, transforms with their fitted statistics -- exactly as
``CoxPH`` does, before and after ``to_dict`` / ``from_dict``. A categorical
level the model was not fitted with raises a ``ValueError`` naming the column
and the level (there is no coefficient for it), and a row with a missing
covariate -- or a missing (``NaN``) time -- predicts ``nan``, leaving the
other rows as they are:

.. jupyter-execute::

    import textwrap

    frame["band"] = np.where(Z[:, 1] > 0, "high", "low")
    csph_f = CompetingRisksProportionalHazards.fit_from_df(
        frame, x_col="time", e_col="cause", formula="z1 + band"
    )
    print(csph_f.feature_names)
    rows = pd.DataFrame({"z1": [0.5, 0.5], "band": ["low", "high"]})
    print(csph_f.cif([2.0, 2.0], rows, event=1).round(4))
    try:
        csph_f.cif([2.0], pd.DataFrame({"z1": [0.5], "band": ["mid"]}), 1)
    except ValueError as err:
        print(textwrap.fill(str(err), 79))

Parametric Competing-Risks Regression
-------------------------------------

The cause-specific Cox model above tells you how a covariate changes each
cause's hazard, but its cumulative incidences are step functions that stop
at the last event: it cannot say what happens after the study ends.
``ParametricCompetingRisksRegression`` answers the same question with a
parametric model for every cause, so each cause's cumulative incidence is a
smooth curve, at any covariate values, that carries on past the data:

- an engineer with two years of field returns asks what share of pumps
  running hot will have had a seal failure by the end of a five-year
  warranty, when bearings fail too;
- a clinician asks for a patient's ten-year probability of dying of the
  disease, given age and stage, from a study with six years of follow-up,
  when patients also die of other causes.

Each cause gets its own regression of its hazard on the covariates: a
Weibull proportional hazards model by default (``model=WeibullPH``), or any
of the parametric regressions (``ExponentialPH``, ``WeibullAFT``,
``LogNormalAFT``, ``LogLogisticPO`` and the rest), the same for every cause
or one per cause in a dict. Each cause is fitted with the failures of the
other causes treated as censored; that is the maximum-likelihood fit of the
whole model, and it does not assume the causes are independent (the theory
page, :doc:`Competing Risks Analysis`, explains why).

Here 600 pumps run at a temperature between 40 and 80 degrees. Seals wear
out, faster when hot; bearings fail at a rate that temperature does not
change. The test stops at 8 (thousand hours), so the data end there:

.. jupyter-execute::

    from surpyval import (
        ExponentialPH,
        ParametricCompetingRisksRegression,
        WeibullPH,
    )

    pump_rng = np.random.default_rng(7)
    n_pumps = 600
    pump_temp = pump_rng.uniform(40, 80, n_pumps)
    pump_hot = (pump_temp - 60) / 10                  # per 10 degrees above 60
    pump_seal = 12 * pump_rng.weibull(2.5, n_pumps) * np.exp(-0.5 * pump_hot / 2.5)
    pump_bearing = pump_rng.exponential(25, n_pumps)
    pump_first = np.minimum(pump_seal, pump_bearing)
    pump_x = np.minimum(pump_first, 8.0)
    pump_e = np.where(pump_seal < pump_bearing, "seal", "bearing").astype(object)
    pump_e[pump_first > 8.0] = None            # still running at the end
    pump_Z = pump_hot[:, None]

    pcr = ParametricCompetingRisksRegression.fit(pump_x, pump_Z, pump_e)
    print(pcr)
    pcr.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(3)

The table stacks each cause's own regression table. The seal's coefficient
is its log hazard ratio per 10 degrees (the data were simulated with 0.5),
and the bearing's is indistinguishable from zero, as simulated:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _table = pcr.summary()
    _seal = _table.loc[("seal", "coefficients", "coef_0")]
    _bearing = _table.loc[("bearing", "coefficients", "coef_0")]
    assert abs(_seal["coef"] - 0.5) < 2 * _seal["se(coef)"], _seal
    assert _bearing["p"] > 0.05, _bearing

The fitted per-cause models are ordinary regression models, in
``pcr.models["seal"]`` and ``pcr.models["bearing"]``, with everything those
offer. The cumulative incidence of a cause is ``cif(x, Z, event)``. The
covariate rows ``Z`` are paired with the times as in every regression: one
row used at every time, one row per time, or several rows at one time.
Below, the seal's cumulative incidence at 50 and at 75 degrees, with the
cause-specific Cox model's step estimate for comparison. The test ended at
8; the parametric curves carry on to 20:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    pump_cox = CompetingRisksProportionalHazards.fit(pump_x, pump_Z, pump_e)
    t_pump = np.linspace(0, 20, 201)
    for degrees, colour in [(50, "C0"), (75, "C3")]:
        row = [(degrees - 60) / 10]
        plt.plot(t_pump, pcr.cif(t_pump, row, "seal"), color=colour,
                 label=f"{degrees} degrees (parametric)")
        seen = t_pump <= 8
        plt.step(t_pump[seen], pump_cox.cif(t_pump[seen], row, "seal"),
                 where="post", color=colour, alpha=0.5,
                 label=f"{degrees} degrees (Cox)")
    plt.axvline(8, color="grey", linestyle=":")
    plt.xlabel("Time (thousand hours)")
    plt.ylabel("Cumulative incidence of a seal failure")
    plt.legend()

Inside the data the two agree; past it only the parametric model has an
answer, and that answer rests on the assumed shape of each hazard, which
the data cannot check beyond 8. Use the cause-specific Cox model when you
only need the covariates' effects or predictions inside the follow-up, and
this one when you need a smooth curve or a prediction beyond it, after
checking the fit of each cause's model (``pcr.models[cause].diagnose()``).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = np.array([2.0, 4.0, 6.0, 7.5])
    for _row in ([-1.0], [1.5]):
        _gap = np.abs(pcr.cif(_t, _row, "seal") - pump_cox.cif(_t, _row, "seal"))
        assert _gap.max() < 0.05, _gap

``probability_of_cause(Z, event)`` is the eventual share of each cause, the
cumulative incidence at infinity, and ``cb`` gives confidence bounds on the
cumulative incidence (or, with ``on=``, on the all-cause survival, failure
probability or cumulative hazard):

.. jupyter-execute::

    pump_rows = [[-1.0], [1.5]]                  # 50 and 75 degrees
    print("eventual share of seal failures:",
          pcr.probability_of_cause(pump_rows, "seal").round(3))
    print("by 15, at 75 degrees           :",
          round(float(pcr.cif(15, [1.5], "seal")), 3),
          pcr.cb(15, [1.5], "seal").round(3))

The hotter pumps lose more of their units to the seal, because the seal's
hazard rises with temperature and the bearing's does not. The causes'
eventual shares sum to one:

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _share = pcr.probability_of_cause(pump_rows, "seal")
    assert _share[1] > _share[0], _share
    assert np.allclose(
        _share + pcr.probability_of_cause(pump_rows, "bearing"), 1.0
    )
    _lo, _hi = pcr.cb(15, [1.5], "seal")
    assert _lo < pcr.cif(15, [1.5], "seal") < _hi

The other functions follow ``ParametricCompetingRisks``, with ``Z`` after
the times. Without ``event``, ``sf``, ``ff``, ``hf``, ``Hf`` and ``df`` are
of the first failure of any cause; with ``event=k`` they are cause ``k``'s
net ones, its own model's. ``qf(p, Z)`` and ``mean(Z)`` are the B-lives
and the mean life of the whole unit, and ``random(size, Z)`` simulates
units at each covariate row:

.. jupyter-execute::

    print("survival to 8 at 75 degrees:", round(float(pcr.sf(8, [1.5])), 3))
    print("B10 life at 50 and 75      :", pcr.qf(0.1, pump_rows).round(2))
    print("mean life at 50 and 75     :", pcr.mean(pump_rows).round(2))
    pump_draws, pump_drawn_at = pcr.random(4, [[1.5]], random_state=0)
    print(pump_draws["x"].round(2), list(pump_draws["e"]))

A different model per cause is a dict. The bearing's hazard is constant in
the simulation, so an exponential model for it has one parameter fewer;
``aic`` and ``bic`` are those of the whole model (the log-likelihood sums
over the causes, and BIC's sample size is the failures of any cause), so
they compare the two:

.. jupyter-execute::

    simpler = ParametricCompetingRisksRegression.fit(
        pump_x, pump_Z, pump_e,
        model={"seal": WeibullPH, "bearing": ExponentialPH},
    )
    print("Weibull + Weibull     AIC: %.1f" % pcr.aic())
    print("Weibull + Exponential AIC: %.1f" % simpler.aic())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert simpler.aic() < pcr.aic(), (simpler.aic(), pcr.aic())

The simpler model has the lower AIC. ``fit_from_df`` takes the column
names, and the covariates as ``Z_cols`` or as a ``formula``; the fitted
model then predicts from a DataFrame of the raw covariates. Entry times
(``tl``, ``tl_col``) are taken as in the single-cause regressions. The
model saves and loads with ``to_dict`` / ``from_dict`` like every other.
Left- and interval-censored rows are refused (without covariates,
``ParametricCompetingRisks`` takes them), and so are weights:

.. jupyter-execute::

    pumps = pd.DataFrame({
        "hours": pump_x, "mode": pump_e, "temp": pump_temp,
        "site": np.where(pump_rng.uniform(size=n_pumps) < 0.5, "coast", "inland"),
    })
    by_site = ParametricCompetingRisksRegression.fit_from_df(
        pumps, x_col="hours", e_col="mode", formula="temp + site"
    )
    print(by_site.feature_names)
    new_pumps = pd.DataFrame({"temp": [75, 75], "site": ["coast", "inland"]})
    print(by_site.cif([15, 15], new_pumps, "seal").round(3))

Scoring predictions of one cause: Brier score and AUC
-----------------------------------------------------

Once a model predicts each subject's cumulative incidence of a cause, two
questions follow: are the predicted probabilities close to what happened
(the *Brier score*: lower is better), and do the subjects who went on to have
that cause by a time get higher predictions than those who did not (the
*time-dependent AUC*: 0.5 is chance, 1 is perfect)? The prediction metrics of
:mod:`surpyval.metrics` answer both for one cause at a time: pass the event
types as ``e`` and the cause scored as ``cause``, and the predicted cumulative
incidences ``F_k(t | Z)`` of that cause in place of predicted survival.

Censored subjects are handled as for the ordinary metrics, by weighting the
others by the inverse of the probability of being still under observation
(IPCW, with the Kaplan-Meier estimate of the censoring of every cause). What
is new is the subject whose first event is of *another* cause: they can never
have the scored cause, so

* in the Brier score they count, with their weight, as subjects whose
  outcome is "no event of this cause", and
* in the AUC they count as controls, alongside the subjects still free of
  every event (``controls="cause_free"``, the default, as R's
  ``riskRegression::Score`` and ``timeROC``'s ``AUC_2``);
  ``controls="event_free"`` leaves them out and compares the cases with the
  event-free subjects only (``timeROC``'s ``AUC_1``).

Here two cause-specific Cox models are fitted to the first 500 subjects of
the Fine-Gray simulation above, one with both covariates and one with
:math:`Z_2` only, and scored for cause 1 on the other 300. A model's ``cif``
pairs each time with a row of ``Z``, so a column of the matrix is one horizon
for every subject:

.. jupyter-execute::

    from surpyval.metrics import score_predictions

    train = np.arange(N) < 500
    test = ~train
    both = CompetingRisksProportionalHazards.fit(
        x[train], Z[train], e[train], c=c[train]
    )
    z2_only = CompetingRisksProportionalHazards.fit(
        x[train], Z[train][:, [1]], e[train], c=c[train]
    )
    horizons = [0.25, 0.5, 1.0]


    def incidence(model, Z_new, cause):
        n = len(Z_new)
        return np.column_stack(
            [model.cif(np.full(n, t), Z_new, event=cause) for t in horizons]
        )


    scores = score_predictions(
        x[test],
        c[test],
        {
            "Z2 only": incidence(z2_only, Z[test][:, [1]], 1),
            "Z1 and Z2": incidence(both, Z[test], 1),
        },
        horizons,
        e=e[test],
        cause=1,
    )
    scores.scores.round(4)

The model with both covariates ranks the subjects who had cause 1 clearly
better from half a time unit on, while its Brier scores are only a little
lower; the paired comparisons say how sure that is:

.. jupyter-execute::

    scores.comparisons.round(4)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _auc = scores.comparisons.query("metric == 'auc'")
    assert (_auc.difference > 0).all() and _auc.p_value.iloc[-1] < 0.05

:func:`~surpyval.metrics.validation.brier_score`,
:func:`~surpyval.metrics.validation.integrated_brier_score` and
:func:`~surpyval.metrics.validation.auc_td` take the same ``e`` and ``cause``
and return the bare estimates. A ``cause`` that is not among the event types
is refused, as is ``cause`` without ``e``.

In formulas, with :math:`\hat G` the censoring Kaplan-Meier, :math:`\delta_i`
the event indicator and :math:`\epsilon_i` the cause of subject :math:`i`'s
event, the Brier score of cause :math:`k` at :math:`t` is

.. math::
    BS_k(t) = \frac1n \sum_i \Big[
        \frac{(I(\epsilon_i = k) - F_k(t\mid Z_i))^2\,
              I(x_i \le t,\ \delta_i=1)}{\hat G(x_i-)}
      + \frac{F_k(t\mid Z_i)^2\, I(x_i > t)}{\hat G(t)} \Big],

and the AUC is the weighted proportion of (case, control) pairs the
predictions order correctly, ties counting half, with the cases
(:math:`x_i \le t`, :math:`\epsilon_i = k`) weighted by
:math:`1/\hat G(x_i-)`, the event-free controls by :math:`1/\hat G(t)` and the
competing-event controls by :math:`1/\hat G(x_j-)` (Blanche, Dartigues and
Jacqmin-Gadda 2013). The standard errors come from the same influence
functions as for one cause, and agree with ``riskRegression::Score(cause = k,
se.fit = TRUE)`` to rounding: Brier score, AUC, the integrated Brier score
and the comparisons, with and without tied times. For a cause other than the
first one, ``Score`` (version 2026.3.11) gives AUC standard errors that change
when the causes are relabelled; SurPyval's match those it gives with the cause
coded 1, which also agree with ``timeROC``.
