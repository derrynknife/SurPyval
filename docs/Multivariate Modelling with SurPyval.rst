Multivariate Modelling with SurPyval
====================================

The ``surpyval.multivariate`` module models several *correlated* event-time
series jointly. The dependence between the series is specified with a
**copula** while the marginal behaviour of each series is any existing SurPyval
distribution — so the margins and the dependence are chosen independently. For
the concepts (Sklar's theorem, the copula families and their tail behaviour,
and the estimation strategies) see the :doc:`Multivariate Analysis` page; this
page is the how-to. The full API is on the :doc:`surpyval.multivariate`
reference page.

SurPyval provides the ``Independence``, ``Clayton``, ``Gumbel``, ``Frank`` and
``Gaussian`` copulas. Each is a ready-made object (like ``surpyval.Weibull``)
with two ways to create a model: ``fit`` to data, or ``from_params`` for a
known parameter. Both return a
:class:`~surpyval.multivariate.parametric.copula.copula_model.CopulaModel`.
Models are bivariate: exactly two series.

.. note::

    ``surpyval.multivariate.Gumbel`` is the Gumbel *copula*; the univariate
    Gumbel distribution is ``surpyval.Gumbel``. Import the copulas from
    ``surpyval.multivariate`` to keep the two apart.

Fitting a copula
----------------

The copulas live in their own package (like ``surpyval.recurrent``) and are
not imported into the top-level namespace. Here we simulate correlated
lifetimes from a known Clayton copula (see the last section) and check the fit
recovers it:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv
    from matplotlib import pyplot as plt
    from surpyval.multivariate import Clayton

    truth = Clayton.from_params(
        2.0,
        margins=[surv.Weibull.from_params([10.0, 2.0]),
                 surv.LogNormal.from_params([2.5, 0.5])],
    )
    data = truth.random(3000, random_state=1)
    x1, x2 = data[:, 0], data[:, 1]

    # margins are SurPyval distributions, fitted along with the copula
    model = Clayton.fit(
        [x1, x2],
        margins=[surv.Weibull, surv.LogNormal],
        how="IFM",
    )
    print("theta       :", model.params)
    print("Kendall's tau:", round(model.kendall_tau(), 3))
    model.margins         # the two fitted univariate models

The true parameter is :math:`\theta = 2` (Kendall's :math:`\tau = 0.5`), and
the margins come back close to Weibull(10, 2) and LogNormal(2.5, 0.5). The
model's ``repr`` summarises it:

.. jupyter-execute::

    print(model)

Laying out the data
~~~~~~~~~~~~~~~~~~~

A joint observation is a *row*: the two lifetimes of one shaft's bearings, one
patient's two complications. ``x`` can be given in either of two layouts:

- a **list (or tuple) of columns**, one array per series: ``[x1, x2]``, as
  above;
- a **2-D numpy array** of shape ``(N, 2)``, one row per joint observation.

``c`` (censoring) and ``xl``/``xr`` (interval bounds) follow the same two
layouts, ``n`` is one count per row (shape ``(N,)``), and ``t`` holds a
truncation window per row *and* series (shape ``(N, 2, 2)``). Internally the
inputs are normalised by
:class:`~surpyval.multivariate.parametric.data.MultivariateSurpyvalData`,
which you can also build directly to check your shapes:

.. jupyter-execute::

    from surpyval.multivariate import MultivariateSurpyvalData

    md = MultivariateSurpyvalData([x1, x2])        # list of columns
    print(md.N, "rows x", md.D, "series; x:", md.x.shape, "c:", md.c.shape,
          "t:", md.t.shape)
    same = MultivariateSurpyvalData(data)          # (N, 2) array
    print(np.array_equal(md.x, same.x))

.. warning::

    A *list* is always read as a list of columns. A list of rows such as
    ``[[3.1, 5.0], [4.2, 6.3], [2.2, 7.7]]`` would be read as three series of
    two observations each, one series per inner list. Convert rows to a numpy
    array first: ``np.asarray(rows)``.

Two shortcuts save typing. A single row of two codes, such as ``c=[0, 1]``,
applies to every row (here: series 1 always observed, series 2 always right
censored). And rows that repeat can be given once with a count in ``n``. The
fit is the same as with the rows written out, because each row's
log-likelihood is simply multiplied by its count. ``dimension(d)`` returns one
series' arrays ``(x, c, xl, xr, tl, tr)``:

.. jupyter-execute::

    shared = MultivariateSurpyvalData(data[:4], c=[0, 1])
    print(shared.c)
    print(shared.dimension(1)[:2])          # series 2: values and codes

    rows = np.ceil(data[:200])              # rounding up creates repeats
    uniq, counts = np.unique(rows, axis=0, return_counts=True)
    by_count = Clayton.fit(uniq, n=counts, margins=[surv.Weibull, surv.LogNormal])
    by_row = Clayton.fit(rows, margins=[surv.Weibull, surv.LogNormal])
    print("%d distinct rows; theta %.3f (counts) vs %.3f (rows)" % (
        len(uniq), by_count.params[0], by_row.params[0]))

Interval-censored entries need their bounds: a ``c`` of ``2`` without ``xl``
and ``xr`` raises a ``ValueError``, as does any array of the wrong shape.

IFM and MLE
~~~~~~~~~~~

Two estimation strategies are available via ``how``:

* ``"IFM"`` (*Inference Functions for Margins*, the default) fits each margin
  independently and then fits the single copula parameter holding the margins
  fixed. Fast, and correct unless the truncation or censoring of one series
  depends on the other.
* ``"MLE"`` jointly optimises the copula parameter together with all margin
  parameters, starting from the IFM solution.

(The ``Independence`` copula has no parameter, so its ``fit`` only fits the
margins, whichever ``how`` is given.)

On well-behaved data the two agree closely; MLE takes longer because it
searches over every parameter at once:

.. jupyter-execute::

    import time

    small = data[:800]
    for how in ["IFM", "MLE"]:
        start = time.perf_counter()
        fit = Clayton.fit(small, margins=[surv.Weibull, surv.LogNormal], how=how)
        print("%s: theta = %.3f, Weibull = %s, LogNormal = %s  (%.2f s)" % (
            how, fit.params[0], np.round(fit.margins[0].params, 3),
            np.round(fit.margins[1].params, 3), time.perf_counter() - start))

The IFM first stage fits each margin with everything that belongs to it: its
values and censoring codes, the row counts ``n`` and that series' own
truncation window. What it cannot see is how truncation or censoring of *one*
series changes what is seen of the *other*; for that, use ``"MLE"`` (see
`Truncated observation`_ and `Censoring that depends on the other series`_
below). The fitted model records how it was obtained in ``method``
(``"IFM"``, ``"MLE"``, or ``"given"`` for ``from_params``) and keeps the
normalised data in ``data`` (``None`` for ``from_params``); ``params`` holds the copula parameter and
``margins`` the two margin models.

Margins can also be passed **already fitted**. With ``how="IFM"`` they are
used as they are and only the copula parameter is estimated. This is useful
when a margin has been fitted with options the copula fit does not pass on,
or reused from an earlier analysis. (With ``how="MLE"`` a fitted margin
supplies the starting values and is re-estimated jointly with the copula
with the same configuration: an offset, limited-failure or zero-inflated
option is kept, and so are its ``fixed`` parameters. A non-parametric
margin, which has no parameters to re-estimate, needs IFM.)

.. jupyter-execute::

    m1 = surv.Weibull.fit(x1)
    m2 = surv.LogNormal.fit(x2)
    prefit = Clayton.fit([x1, x2], margins=[m1, m2])
    print(prefit.params, prefit.margins[0] is m1)

Choosing a copula family
~~~~~~~~~~~~~~~~~~~~~~~~

A fitted model reports the maximised joint log-likelihood of its data as
``log_likelihood`` (``neg_ll()`` is its negative), and the information
criteria ``aic()`` and ``bic()``. The likelihood is the full joint one the fit
maximised, with every row's censoring, truncation and count, so it serves for
censored data too (with complete data it is the sum of the log joint density
``pdf`` over the rows). ``k``, the number of estimated parameters, counts the
copula parameter and the margin parameters the fit estimated, so every model
below has ``k = 5`` except the Independence copula (``k = 4``); AIC charges the
extra parameter:

.. jupyter-execute::

    from surpyval.multivariate import Independence, Gumbel, Frank, Gaussian

    fits = {}
    for fam in [Independence, Clayton, Gumbel, Frank, Gaussian]:
        fits[fam.name] = fam.fit(data, margins=[surv.Weibull, surv.LogNormal])

    for name, m in fits.items():
        print("%-12s params=%-22s tau=%.3f  tails=%s  loglik=%.1f  AIC=%.1f" % (
            name, np.round(m.params, 3), m.kendall_tau(),
            np.round(m.tail_dependence(), 3), m.log_likelihood, m.aic()))

The Clayton copula, which generated the data, has the highest likelihood (and
lowest AIC) by a wide margin, even though Gaussian and Frank reach a similar Kendall's tau: the
data carry strong *lower-tail* dependence (joint early failures) that only
Clayton can express. A picture tells the same story. Transforming each series
to ranks in :math:`(0, 1)` (pseudo-observations) removes the margins and shows
the copula itself; compare the data with samples from two fitted families:

.. jupyter-execute::

    from scipy.stats import rankdata

    def pseudo_obs(xy):
        return np.column_stack([rankdata(col) / (len(col) + 1) for col in xy.T])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    panels = [("data", data),
              ("Clayton fit", fits["Clayton"].random(3000, random_state=2)),
              ("Gaussian fit", fits["Gaussian"].random(3000, random_state=2))]
    for ax, (title, xy) in zip(axes, panels):
        u = pseudo_obs(xy)
        ax.scatter(u[:, 0], u[:, 1], s=2, alpha=0.4)
        ax.set_title(title)
        ax.set_xlabel("rank of series 1")
    axes[0].set_ylabel("rank of series 2")

The tight cluster in the bottom-left corner of the data (both series failing
early) is reproduced by Clayton and missing from the Gaussian copula.

.. warning::

    Clayton (:math:`\theta > 0`) and Gumbel (:math:`\theta \geq 1`) can only
    express positive dependence. If the empirical Kendall's tau of your data is
    negative, use Frank or Gaussian: a Clayton or Gumbel fit is pushed to its
    independence boundary (:math:`\theta \to 0` or :math:`1`), where it *is*
    the independence copula, with the same likelihood.

Here is that failure on purpose, with data simulated from a Frank copula with
:math:`\theta = -5` (Kendall's :math:`\tau \approx -0.46`):

.. jupyter-execute::

    from scipy.stats import kendalltau

    neg = Frank.from_params(-5.0, margins=truth.margins).random(1000, random_state=4)
    print("empirical tau: %.3f" % kendalltau(neg[:, 0], neg[:, 1]).statistic)
    for fam in [Independence, Clayton, Gumbel, Frank, Gaussian]:
        m = fam.fit(neg, margins=[surv.Weibull, surv.LogNormal])
        print("%-12s params=%-24s loglik=%.1f" % (
            fam.name, np.round(m.params, 3), m.log_likelihood))

Clayton and Gumbel collapse onto independence, with exactly its
log-likelihood; Frank recovers :math:`\theta` and fits far better, with the
Gaussian copula second.

Censoring and truncation
------------------------

The differentiator of the SurPyval copula implementation is that the joint
likelihood supports the **full** censoring and truncation matrix, per
dimension, using the same convention as the univariate models
(``c`` of ``0`` observed, ``1`` right, ``-1`` left, ``2`` interval; ``t`` for
a truncation window). Each series of a joint observation carries its own
censoring code — pass one censoring column per series. Here each series is
right-censored at its own fixed threshold (censoring that is unrelated to the
lifetimes, so the default IFM fit is appropriate), and the fit still recovers
the copula parameter:

.. jupyter-execute::

    thresholds = np.array([14.0, 16.0])
    c = (data > thresholds).astype(int)          # per-series right-censoring
    x_obs = np.minimum(data, thresholds)

    model_c = Clayton.fit(
        [x_obs[:, 0], x_obs[:, 1]],
        c=[c[:, 0], c[:, 1]],                    # one column per series
        margins=[surv.Weibull, surv.LogNormal],
        how="IFM",
    )
    print("censored fraction:", round(c.mean(), 2))
    print("theta (censored) :", model_c.params)

What does a censored row contribute to the likelihood? The
:doc:`Multivariate Analysis` page derives the rule: each row is the
probability of a rectangle, built from the copula CDF ``C``, its partial
derivatives (the h-functions) and its density, evaluated at the
margin-transformed values :math:`u_j = F_j(x_j)`. The copula objects expose
these building blocks directly, as functions of ``(u, v, theta)``: ``cdf``,
``du`` (:math:`\partial C/\partial u`), ``dv`` and ``pdf`` (the copula
density). Here is one row, :math:`(x_1, x_2) = (10, 18)`, under four
censoring patterns, each checked against the joint distribution of the fitted
model:

.. jupyter-execute::

    th = model.params[0]
    F1, F2 = model.margins
    u, v = F1.ff(10.0), F2.ff(18.0)

    def show(label, by_hand, from_joint):
        print("%-24s %.6f   %.6f" % (label, np.ravel(by_hand)[0],
                                     np.ravel(from_joint)[0]))

    print("%-24s %-11s  %s" % ("row pattern", "by hand", "from the joint"))
    # both observed: copula density times the two marginal densities
    show("both observed", Clayton.pdf(u, v, th) * F1.df(10.0) * F2.df(18.0),
         model.pdf([[10, 18]]))

    # 1 observed, 2 right censored: f1 * (1 - dC/du). Check: the derivative
    # in x1 of P(X1 <= x1, X2 > 18) = F1(x1) - H(x1, 18), by differencing
    h = 1e-4
    joint = lambda a: F1.ff(a) - model.cdf([[a, 18.0]])
    show("1 observed, 2 right", F1.df(10.0) * (1 - Clayton.du(u, v, th)),
         (joint(10 + h) - joint(10 - h)) / (2 * h))

    # both right censored: the joint survival function
    show("both right", 1 - u - v + Clayton.cdf(u, v, th), model.sf([[10, 18]]))

    # 1 left censored, 2 interval censored in (15, 20]
    show("1 left, 2 in (15, 20]",
         Clayton.cdf(u, F2.ff(20.0), th) - Clayton.cdf(u, F2.ff(15.0), th),
         model.cdf([[10, 20]]) - model.cdf([[10, 15]]))

Each pair agrees. The fit applies exactly these expressions, row by row, and
the same four building blocks cover all sixteen combinations of codes.

Interval and left censoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose series 2 is only checked at inspections every 5 time units, so each
of its failures is known to lie in an interval, while series 1 cannot be
resolved below 4 units (a left-censored "failed before 4"). Interval-censored
entries have ``c == 2`` and take their bounds from ``xl`` and ``xr`` (same
layout as ``x``; the value in ``x`` is ignored for those entries);
left-censored entries have ``c == -1`` with the bound in ``x``:

.. jupyter-execute::

    x_mix = data.copy()
    c_mix = np.zeros_like(data, dtype=int)

    # series 1: left censored below 4
    early = data[:, 0] < 4.0
    c_mix[early, 0] = -1
    x_mix[early, 0] = 4.0

    # series 2: interval censored between inspections every 5 units
    xl = data.copy()
    xr = data.copy()
    xl[:, 1] = np.floor(data[:, 1] / 5.0) * 5.0
    xr[:, 1] = xl[:, 1] + 5.0
    c_mix[:, 1] = 2

    model_mix = Clayton.fit(x_mix, c=c_mix, xl=xl, xr=xr,
                            margins=[surv.Weibull, surv.LogNormal])
    print("left-censored fraction of series 1:", round(early.mean(), 3))
    print("theta:", model_mix.params)
    print("margins:", [np.round(m.params, 3) for m in model_mix.margins])

Even with every series-2 time reduced to a 5-unit interval, the copula
parameter and both margins are recovered.

Truncated observation
~~~~~~~~~~~~~~~~~~~~~

Truncation means some joint observations could never have been seen. Suppose
only shafts whose first bearing survived a 3-unit burn-in reach the field, so
the field data contain no rows with :math:`X_1 \leq 3`. The truncation window
is given per row and per series as ``t[i, j] = [lower, upper]``, with
``-np.inf``/``np.inf`` for "no limit":

.. jupyter-execute::

    field = data[data[:, 0] > 3.0]
    t = np.empty((len(field), 2, 2))
    t[..., 0], t[..., 1] = -np.inf, np.inf       # no truncation by default
    t[:, 0, 0] = 3.0                             # series 1 left-truncated at 3

    print("truth: theta = 2, Weibull = [10, 2], LogNormal = [2.5, 0.5]")
    for how in ["IFM", "MLE"]:
        fit = Clayton.fit(field, t=t, margins=[surv.Weibull, surv.LogNormal],
                          how=how)
        print("%s:   theta = %.3f, Weibull = %s, LogNormal = %s" % (
            how, fit.params[0], np.round(fit.margins[0].params, 3),
            np.round(fit.margins[1].params, 3)))

Read the IFM line margin by margin. The Weibull margin of series 1 was fitted
with its left truncation at 3, which is exactly the selection series 1 went
through, and it comes back close to the truth. The LogNormal margin of series
2 does not: :math:`\mu` is too large and :math:`\sigma` too small. Series 2 was
never truncated itself, but the burn-in selected its rows too. With positive
dependence a unit whose bearing 1 lasted past 3 tends to have a long-lived
bearing 2, so the field sample under-represents short series-2 lives, and the
IFM margin, fitted to that sample as if it were the population, is shifted to
longer and less variable lives. The copula stage then has to explain the data
with that distorted margin, and it settles on far too little dependence.

The ``how="MLE"`` fit gets all three right. Its correction does *not* come
from the truncation divisor, which for a burn-in on series 1 alone is just
:math:`P(X_1 > 3)`, free of the copula. It comes from fitting margin 2 jointly
with the copula: each :math:`x_2` enters the likelihood through the copula
density, paired with its :math:`x_1`, so the model knows which series-2 values
the burn-in favours. The :doc:`Multivariate Analysis` page gives the formula
for this selection. Use ``how="MLE"`` whenever truncation of one series
selects the rows of another; with every series truncated, all the margins are
affected.

Censoring that depends on the other series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same issue arises without any truncation when the *censoring* of one
series is set by the other. Suppose a shaft is retired 3 time units after its
first bearing fails, so bearing 2 is right censored at :math:`x_1 + 3` unless
it has already failed. Each row is still in the sample, and the joint
likelihood is valid (the censoring time depends only on the observed
:math:`x_1`). But series 2 on its own is now informatively censored: the
bearings that are censored early are the partners of early bearing-1
failures. Through the dependence, these are not a typical sample of the
bearings still running at that age, and a univariate fit assumes they are.

.. jupyter-execute::

    retire = data[:, 0] + 3.0
    c_dep = np.column_stack([np.zeros(len(data), dtype=int),
                             (data[:, 1] > retire).astype(int)])
    x_dep = np.column_stack([data[:, 0], np.minimum(data[:, 1], retire)])
    print("series 2 censored fraction: %.2f" % c_dep[:, 1].mean())

    for how in ["IFM", "MLE"]:
        fit = Clayton.fit(x_dep, c=c_dep, margins=[surv.Weibull, surv.LogNormal],
                          how=how)
        print("%s:   theta = %.3f, LogNormal = %s" % (
            how, fit.params[0], np.round(fit.margins[1].params, 3)))

The IFM LogNormal margin is fitted as if the censoring were independent of
bearing 2's life, which it is not, and comes out too long and too variable
(:math:`\mu` of 2.70 against 2.5). The copula stage, handed that margin, finds much too little
dependence (:math:`\theta` of 1.13 against 2). The joint fit recovers both. When IFM and MLE
disagree like this, look for truncation or censoring of one series that is
driven by the other; when they agree, the faster IFM fit is fine.

Working with a fitted model
---------------------------

The fitted model exposes the joint distribution functions, the dependence
measures, and a correlated sampler. Points are given as rows ``[x1, x2]``:

.. jupyter-execute::

    print("joint cdf  :", model.cdf([[10, 18], [5, 25]]))  # P(X1<=x1, X2<=x2)
    print("joint sf   :", model.sf([[10, 18]]))            # P(X1>x1, X2>x2)
    print("joint pdf  :", model.pdf([[10, 18]]))
    print("cond. cdf  :", model.conditional_cdf(np.array([[10, 18]]),
                                                 given_dim=0))  # h-function
    print("Kendall tau:", round(model.kendall_tau(), 3))
    print("Spearman   :", round(model.spearman_rho(), 3))
    lower, upper = model.tail_dependence()
    print("tail dep.  : lower %.3f, upper %.3f" % (lower, upper))

    model.random(3, random_state=0)     # correlated (N, 2) samples

``ff`` is an alias of ``cdf``. For Clayton the lower tail-dependence
coefficient is :math:`2^{-1/\theta} \approx 0.71` and the upper one is zero.
Spearman's rho is estimated by simulation for the Clayton and Gumbel copulas
(closed forms are used for Frank, Gaussian and Independence), so for those
two treat its third decimal place with caution.

Conditional probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~

``conditional_cdf(x, given_dim=0)`` is :math:`P(X_2 \leq x_2 \mid X_1 =
x_1)` — the probability that the second series has failed by :math:`x_2`
*given that the first failed at exactly* :math:`x_1`; ``given_dim=1`` swaps
the roles. It shows how knowledge of one failure updates the other:

.. jupyter-execute::

    x2_query = 12.0
    print("P(X2 <= 12) unconditionally: %.3f" % model.margins[1].ff(x2_query))
    for x1_seen in [2.0, 10.0, 20.0]:
        p = model.conditional_cdf([[x1_seen, x2_query]], given_dim=0)[0]
        print("P(X2 <= 12 | X1 = %4.1f)    : %.3f" % (x1_seen, p))

An early failure of the first bearing makes an early failure of the second
much more likely; a late one makes it less likely. Conditioning on *survival*
rather than on an exact failure time uses the joint survival function: by the
definition of conditional probability,
:math:`P(X_2 > x_2 \mid X_1 > x_1) = S(x_1, x_2) / S_1(x_1)`:

.. jupyter-execute::

    def p_survive_given_survived(m, x1, x2):
        return m.sf([[x1, x2]])[0] / m.margins[0].sf(x1)

    print("P(X2 > 12)            : %.3f" % model.margins[1].sf(x2_query))
    print("P(X2 > 12 | X1 > 5)   : %.3f" % p_survive_given_survived(model, 5.0, 12.0))

System reliability
~~~~~~~~~~~~~~~~~~

The joint functions answer system-level questions directly. A **series**
system (it needs both parts) survives to :math:`t` with probability
:math:`S(t, t)` = ``sf([[t, t]])``; a **parallel** (redundant) system fails by
:math:`t` only if both parts have, with probability :math:`H(t, t)` =
``cdf([[t, t]])``. Comparing with the same margins joined by the
``Independence`` copula shows what ignoring the dependence would cost:

.. jupyter-execute::

    indep = Independence.from_params([], margins=model.margins)

    t_grid = np.array([[3.0, 3.0], [5.0, 5.0], [8.0, 8.0]])
    print("P(parallel pair failed by t): dependent vs independent")
    for row, dep_p, ind_p in zip(t_grid, model.cdf(t_grid), indep.cdf(t_grid)):
        print("  t = %.0f: %.5f vs %.5f  (ratio %.1f)" % (row[0], dep_p, ind_p,
                                                       dep_p / ind_p))

At short times the redundant pair is many times more likely to have failed
than the independence assumption suggests, because of Clayton's lower-tail
dependence: redundancy buys much less protection against common-cause early
failure than the margins alone would imply.

Plotting and simulation
~~~~~~~~~~~~~~~~~~~~~~~

``random(size, random_state=None)`` returns an ``(size, 2)`` array of
correlated lifetimes: it samples the copula (by inverting the h-function, or
directly for the Gaussian copula) and maps the uniforms through each margin's
quantile function. There is no built-in plot method, but simulated samples and
the joint functions plot directly with matplotlib; here a sample is drawn over
contours of the joint survival function:

.. jupyter-execute::

    sample = model.random(1000, random_state=3)
    g1, g2 = np.meshgrid(np.linspace(0.5, 25, 60), np.linspace(2, 40, 60))
    joint_sf = model.sf(np.column_stack([g1.ravel(), g2.ravel()])).reshape(g1.shape)

    plt.scatter(sample[:, 0], sample[:, 1], s=4, alpha=0.4)
    cs = plt.contour(g1, g2, joint_sf, levels=[0.1, 0.25, 0.5, 0.75],
                     colors="k")
    plt.clabel(cs, fmt="S=%.2f")
    plt.xlabel("series 1 (Weibull margin)")
    plt.ylabel("series 2 (LogNormal margin)")

Saving and loading a copula model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``CopulaModel`` serialises to a plain dictionary (``to_dict``) or JSON file
(``to_json``) holding the family, its parameter, how it was fitted and each
margin's own serialisation. Restore it with ``CopulaModel.from_dict`` /
``CopulaModel.from_json`` or with the package-level ``surpyval.from_dict`` /
``surpyval.from_json``. The fitting data are not stored, and every margin must
itself be serialisable (the built-in distributions are):

.. jupyter-execute::

    import json
    from surpyval.multivariate import CopulaModel

    d = model.to_dict()
    print(json.dumps(d)[:120], "...")
    restored = surv.from_dict(json.loads(json.dumps(d)))
    print(type(restored).__name__, restored.copula.name, restored.params)
    print(np.allclose(restored.cdf([[10, 18]]), model.cdf([[10, 18]])))
    print(CopulaModel.from_dict(d).margins[1].params)   # class-level reader

Building a model from known parameters
--------------------------------------

As with the univariate distributions, a model can be created directly from
parameters and pre-built margins -- useful for Monte-Carlo simulation (the
data at the top of this page were generated this way):

.. jupyter-execute::

    sim = Clayton.from_params(
        2.0,
        margins=[surv.Weibull.from_params([10, 2]),
                 surv.LogNormal.from_params([3, 0.4])],
    )
    sim.random(5, random_state=0)

The margins must be models (they need ``ff``, ``df`` and ``qf``), such as
those returned by ``from_params`` or ``fit``; the Independence copula takes an
empty parameter list.

Same correlation, different tails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common way to set up a simulation is to choose the strength of dependence
as a Kendall's tau and convert it to each family's parameter with the
relations on the :doc:`Multivariate Analysis` page. Holding :math:`\tau = 0.5`
fixed and the margins fixed, the families still disagree sharply about joint
extremes:

.. jupyter-execute::

    from scipy.optimize import brentq

    tau = 0.5
    params = {
        "Clayton": 2 * tau / (1 - tau),
        "Gumbel": 1 / (1 - tau),
        "Gaussian": np.sin(np.pi * tau / 2),
        "Frank": brentq(lambda th: Frank.kendall_tau(th) - tau, 0.1, 50),
    }
    margins = [surv.Weibull.from_params([10.0, 2.0]),
               surv.Weibull.from_params([10.0, 2.0])]
    q_lo = margins[0].qf(0.05)                   # 5% quantile of each margin
    q_hi = margins[0].qf(0.95)                   # 95% quantile
    families = {"Clayton": Clayton, "Gumbel": Gumbel,
                "Gaussian": Gaussian, "Frank": Frank}

    print("family    param   tau    P(both < 5% q)  P(both > 95% q)")
    for name, fam in families.items():
        m = fam.from_params(params[name], margins=margins)
        both_early = m.cdf([[q_lo, q_lo]])[0]
        both_late = m.sf([[q_hi, q_hi]])[0]
        print("%-9s %5.3f   %.3f   %.4f          %.4f" % (
            name, params[name], m.kendall_tau(), both_early, both_late))
    print("independent               %.4f          %.4f" % (0.05**2, 0.05**2))

All four have the same Kendall's tau, but Clayton makes a joint failure below
the 5% quantile far more likely than the others, and Gumbel a joint survival
beyond the 95% quantile. Frank and Gaussian treat the two tails alike (their
two probabilities are equal), with Frank, whose dependence is weakest in the
tails, below Gaussian in both. Clayton, for its part, gives the lowest
probability of joint survival beyond the 95% quantile. When the quantity you care about is a joint extreme — both
redundant units failing early, both components outliving a warranty — the
choice of family matters as much as the strength of dependence.

Defining your own copula family
-------------------------------

The five families are instances of classes derived from
:class:`~surpyval.multivariate.parametric.copula.copula.Copula`, and a new
family can be added the same way. The one thing a subclass must supply is the
copula CDF ``cdf(u, v, theta)``, plus a ``name``, the parameter ``bounds``
(in the same ``(low, high)`` form as the univariate fitters, ``None`` for
unbounded) and ``param_names``. Everything else is derived from the CDF:
``du``, ``dv`` and ``pdf`` by automatic differentiation (so write the CDF with
arithmetic operators and the functions of ``surpyval.np``, autograd's numpy),
sampling by inverting ``du``, and Kendall's tau and Spearman's rho by
simulation. As an example, the Ali-Mikhail-Haq copula
:math:`C(u, v) = uv / \{1 - \theta(1 - u)(1 - v)\}`,
:math:`-1 \leq \theta < 1`, which only allows weak dependence
(:math:`-0.18 < \tau < 1/3`):

.. jupyter-execute::

    from surpyval.multivariate import Copula

    class AliMikhailHaq(Copula):
        name = "Ali-Mikhail-Haq"
        bounds = ((-1, 1),)
        param_names = ("theta",)

        def cdf(self, u, v, theta):
            return u * v / (1 - theta * (1 - u) * (1 - v))

    AMH = AliMikhailHaq()
    amh_truth = AMH.from_params(0.6, margins=truth.margins)
    amh_data = amh_truth.random(1000, random_state=0)
    # init (optional) starts the search at a value strictly inside the bounds
    amh_fit = AMH.fit(amh_data, margins=[surv.Weibull, surv.LogNormal], init=0.5)
    print(amh_fit)
    print("Kendall's tau (simulated): %.3f" % amh_fit.kendall_tau())

The estimate, 0.55 against a true 0.6, is typical for this weakly dependent
family, whose parameter is hard to pin down with 1,000 rows. The simulated
tau agrees with the family's closed form,
:math:`1 - 2\{\theta + (1 - \theta)^2 \ln(1 - \theta)\}/(3\theta^2) = 0.145`
at the fitted :math:`\theta`, to within the simulation error.

Two notes. Without ``init`` the search starts from a point strictly inside
the ``bounds``: :math:`\theta = 1` when that is inside them, otherwise the
midpoint of a finite range (0 here) or one unit inside a one-sided bound. The
built-in families start from the value matching the data's Kendall's tau; pass
``init`` (one value per parameter, strictly inside the bounds, or a
``ValueError`` explains the problem) when you have a better guess, as above.
And a model of a custom family cannot be restored with ``from_dict``, which
rebuilds a copula from its name and so only knows the five built-in families.

