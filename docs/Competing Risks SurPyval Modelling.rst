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
     - How do covariates change each cause-specific hazard (``how="Cox"``),
       or the incidence of every cause (``how="Fine-Gray"``)?

The classes live in ``surpyval.univariate.competing_risks``; ``gray_test`` is
also available at the top level as ``surpyval.gray_test``. All of them accept
right-censored data only. For repeated events with several failure modes see
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
- ``c`` -- optional censoring flags, ``0`` observed and ``1`` right-censored.
  When ``c`` is omitted it is derived from ``e``: a missing cause means
  censored. If you do pass ``c``, every row with ``c == 1`` must have a missing
  cause and every other row must have one, otherwise a ``ValueError`` explains
  the mismatch;
- ``n`` -- optional counts, when each row stands for several identical units.

Left- and interval-censored rows (``c`` of ``-1`` or ``2``) are rejected.

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

    ``ff(x, event=k)`` is the net failure probability, the Nelson-Aalen
    version of "one minus Kaplan-Meier with the other causes censored". It
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

    km_model = CompetingRisks.fit(x, e, method="Kaplan-Meier")
    print("same CIFs          :", np.allclose(km_model.cif(t, "wear"), model.cif(t, "wear")))
    print("sf (Kaplan-Meier)  :", np.round(km_model.sf(t), 4))
    print("sf (Nelson-Aalen)  :", np.round(model.sf(t), 4))
    print("CIFs sum to 1 - KM :", np.allclose(
        km_model.cif(t, "wear") + km_model.cif(t, "shock"), km_model.ff(t)))

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

The other methods follow the same conventions as ``CompetingRisks``:
``hf``/``Hf`` take an optional ``event`` (cause-specific) and otherwise sum over
causes; ``sf`` and ``ff`` are the all-cause survival and failure probability;
``iif(x, event)`` is the sub-distribution density; and ``cif(x)`` without an
``event`` is the all-cause incidence ``1 - sf``. ``cif`` returns a float for a
scalar time:

.. jupyter-execute::

    print("CIF wear at 100    :", pmodel.cif(100.0, "wear"))
    print("all-cause ff at 100:", pmodel.ff(100.0))
    print("hazard of shock    :", pmodel.hf(100.0, event="shock"))

Because the joint likelihood is the product of the per-cause likelihoods,
``neg_ll``, ``aic`` and ``bic`` are sums over causes and can be used to compare
candidate distributions -- for example, whether the shock mode needs a Weibull
rather than an exponential:

.. jupyter-execute::

    all_weibull = ParametricCompetingRisks.fit(x, e)   # Weibull for both causes
    print("Weibull + Exponential AIC: %.1f" % pmodel.aic())
    print("Weibull + Weibull     AIC: %.1f" % all_weibull.aic())
    print("fitted shock shape:", all_weibull.models["shock"].params[1])

The shock mode's fitted Weibull shape is close to one (an exponential), and
the Weibull + Exponential model has the lower AIC: the extra shape parameter is
not worth its cost. ``ParametricCompetingRisks`` also has
a ``fit_from_df(df, x_col, e_col, c_col=None, n_col=None, dist=Weibull,
how="MLE")``; ``how`` is passed to each cause's distribution fit.

Assembling a model from separately fitted causes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ParametricCompetingRisks.from_fitted`` combines already-fitted single-cause
models -- a ``{cause: model}`` dict, or a list whose causes become ``0, 1,
...`` -- into one competing-risks model. Each model can use any SurPyval option
(a different family, an offset, a limited-failure or zero-inflated fit, ...).
The one rule: fit each model to the *cause-specific view* of the data, with
that cause's failures observed and every other row right-censored.

This is also how to handle **delayed entry (left truncation)**, which the
competing-risks ``fit`` methods do not take as an argument. Suppose the
components were only put under observation at a random age ``entry``, so that
units which failed before entering were never seen. Ignoring the entry ages
biases the fit; fitting each cause with ``tl=entry`` and assembling the result
is the exact maximum-likelihood fit (the truncated likelihood factorises by
cause too):

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

    per_cause = {}
    for k, dist in [("wear", surv.Weibull), ("shock", surv.Exponential)]:
        c_k = np.where(e_t == k, 0, 1)           # cause-specific view
        per_cause[k] = dist.fit(x_t, c=c_k, tl=tl)
    adjusted = ParametricCompetingRisks.from_fitted(per_cause)

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

``CompetingRisks``, ``ParametricCompetingRisks`` and the Fine-Gray model
returned by ``FineGray.fit`` can be serialised with ``to_dict``/``to_json`` and
restored with the class's ``from_dict``/``from_json`` or with the package-level
``surpyval.from_dict``/``surpyval.from_json``, which work out the class from
the dictionary. The restored model reproduces every prediction exactly
(``CompetingRisksProportionalHazards`` is not serialisable):

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
set with an inverse-probability-of-censoring weight, rather than removing them.
Reach for it when the clinical or engineering question is "how many fail of this
cause", not "how fast".

Pass the observed times ``x``, the per-observation cause label ``e``, the group
label, and the ``cause`` of interest (use ``c`` for censored rows, or mark them
with a ``None`` cause). Here two groups have genuinely different cause-1
incidence:

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

    result = gray_test(x, e, group, cause=1)
    print('statistic = %.2f   df = %d   p = %.3g'
          % (result.statistic, result.df, result.p_value))

The tiny ``p``-value correctly flags the difference in cause-1 incidence. The
result is a named tuple ``(statistic, df, p_value, cause, groups)``; ``df`` is
the number of groups minus one, so more than two groups are compared in one
test.

Calibration
~~~~~~~~~~~

On data where the groups share the same incidence the test is calibrated,
returning ``p``-values spread over ``[0, 1]`` — including under censoring,
which is where the inverse-probability weighting earns its keep. A quick
check: simulate many pairs of groups with identical cause-specific hazards and
independent censoring, and count how often ``p < 0.05``:

.. jupyter-execute::

    rng = np.random.default_rng(3)

    def simulate_cr(n, h1, h2, cens_mean):
        t1 = rng.exponential(1 / h1, n)       # latent cause-1 time
        t2 = rng.exponential(1 / h2, n)       # latent cause-2 time
        cz = rng.exponential(cens_mean, n)    # censoring time
        x = np.minimum.reduce([t1, t2, cz])
        e = np.where(x == cz, None, np.where(t1 < t2, 1, 2)).astype(object)
        return x, e

    p_values = []
    for _ in range(200):
        x0, e0 = simulate_cr(100, 0.1, 0.2, 10.0)
        x1, e1 = simulate_cr(100, 0.1, 0.2, 10.0)
        res = gray_test(np.concatenate([x0, x1]), np.concatenate([e0, e1]),
                        np.repeat([0, 1], 100), cause=1)
        p_values.append(res.p_value)
    print("rejection rate at 5%%: %.3f" % np.mean(np.array(p_values) < 0.05))

The rejection rate is close to the nominal 5%, as it should be when the null
hypothesis is true.

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
    gray = gray_test(x, e, group, cause=1)
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

The ``rho`` argument weights each event time by
:math:`\{1 - \hat{F}(t^-)\}^{\rho}`, where :math:`\hat{F}` is the pooled CIF of
the cause. The default ``rho=0`` is the standard test; ``rho > 0`` emphasises
differences in *early* incidence:

.. jupyter-execute::

    print("rho = 1: p = %.2g" % gray_test(x, e, group, cause=1, rho=1.0).p_value)


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

    model = FineGray.fit(x, Z, e, c=c, cause=1)
    model

The IPCW correction is what lets the coefficients come back near their true
values *under* censoring — a naive unweighted subdistribution risk set would be
biased. The fitted model stores the estimates as arrays, one entry per column
of ``Z``; ``np.exp(model.beta)`` gives the sub-distribution hazard ratios:

.. jupyter-execute::

    print("beta     :", np.round(model.beta, 3))     # also model.coefficients
    print("se       :", np.round(model.se, 3))
    print("p-values :", model.p_values)
    print("SHR      :", np.round(np.exp(model.beta), 3))
    print("cov      :\n", np.round(model.cov, 4))

The standard errors come from the inverse Hessian of the weighted partial
likelihood (the robust variance of Fine and Gray is not implemented), so treat
them as approximate. Because the model targets the incidence directly, ``cif``
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
``1 - cif(x, Z)`` and ``phi(Z)`` the multiplier :math:`e^{Z\beta}`.

Things to watch:

- ``cause`` must be given when the data contain more than one cause, and it
  must be one that is observed; with a single cause it may be omitted.
- The Fine-Gray model is fitted for one cause at a time. To model every cause,
  use ``CompetingRisksProportionalHazards`` with ``how="Fine-Gray"`` (below);
  the separate fits are not constrained to be mutually consistent.

The fitted model serialises like the other competing-risks models; the
optimiser result (``model.res``) is not stored:

.. jupyter-execute::

    reloaded = surv.from_dict(model.to_dict())
    np.allclose(reloaded.cif([0.5, 1.0], Z=[1.0, 0.0]),
                model.cif([0.5, 1.0], Z=[1.0, 0.0]))

Cause-Specific Proportional Hazards
-----------------------------------

``CompetingRisksProportionalHazards`` fits a proportional-hazards model per
cause. With ``how="Cox"`` (the default) each cause is a Cox model with the
other causes treated as censored; ``how="Fine-Gray"`` fits the subdistribution
model above for every cause. It reuses the simulated data from the previous
section:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    csph = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Cox")
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

With ``how="Cox"`` the model has the usual functions, each taking the times,
one covariate vector ``Z`` and an optional ``event``:

- ``cif(x, Z, event)`` -- the cumulative incidence of ``event`` at ``Z``,
  :math:`\sum_{x_j \le x} \Delta\hat{\Lambda}_{k,0}(x_j) e^{Z\hat\beta_k}
  \hat{S}(x_{j-1} \mid Z)` with :math:`\hat{S}` the product-limit all-cause
  survival at ``Z``;
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

The CIFs add up to the all-cause failure probability. Exactly, in fact, for
the product-limit survival the CIFs are built on (so their total never exceeds
one); ``sf`` reports the Cox survival :math:`e^{-H}`, which is very slightly
higher, so ``1 - sf`` sits just below the sum.

With ``how="Fine-Gray"``, ``cif``, ``sf`` (``1 - cif``), ``ff`` and ``Hf`` need
an ``event`` and come from each cause's Fine-Gray model; ``hf`` and ``df``
raise a ``ValueError`` because the step baseline has no pointwise density.
Comparing the two fits on the same data:

.. jupyter-execute::

    fg_all = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Fine-Gray")
    for cause, row in fg_all.event_idx_map.items():
        print("cause", cause, "Fine-Gray coefficients:",
              np.round(fg_all.betas[row], 3))
    print("Fine-Gray CIF of cause 1:",
          np.round(fg_all.cif(times, Z=z, event=1), 4))

The Fine-Gray coefficients for cause 1 match ``FineGray.fit`` above. For cause
2 they have the opposite sign to cause 1, even though the covariates were only
built into the cause-1 incidence: whatever raises the incidence of one cause
necessarily lowers the incidence of the other.

Fitting from a DataFrame
~~~~~~~~~~~~~~~~~~~~~~~~

``fit_from_df`` takes the time and cause column names, and the covariates
either as ``Z_cols`` (a column name or list of names) or as a ``formula``;
``c_col``, ``n_col``, ``how`` and ``tie_method`` (default ``"efron"``, passed to
each cause's Cox fit) are optional. A blank/``NaN`` cause marks a censored row.
Predictions still take a covariate array ``Z`` in column order:

.. jupyter-execute::

    frame = pd.DataFrame({"time": x, "cause": e, "z1": Z[:, 0], "z2": Z[:, 1]})
    csph_df = CompetingRisksProportionalHazards.fit_from_df(
        frame, x_col="time", e_col="cause", Z_cols=["z1", "z2"]
    )
    print(csph_df.feature_names)
    print(np.allclose(csph_df.cif(times, Z=z, event=1), cif1))
