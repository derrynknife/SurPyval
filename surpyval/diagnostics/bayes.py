"""``diagnose()`` for a Bayesian fit (#867).

A :class:`~surpyval.bayes.model.BayesianParametric` model --
``Weibull.fit_bayes(x, c, n, t, priors=...)`` by the grid, the Laplace
approximation or a closed form, or ``BayesianParametric.from_draws`` with
its data -- is checked for whether the distribution reproduces the data,
whether the prior and the data disagree, and whether the posterior was
computed accurately. The checks are listed in :data:`BAYES` at the
bottom.

The checks
----------
``posterior_predictive``
    Posterior predictive checks (Gelman, Meng and Stern, 1996). For each
    of ``n_boot`` draws of the parameters from the posterior
    (:data:`DEFAULT_N_DRAWS` by default), a replicated data set is
    simulated under the observed design: every unit gets a time from the
    drawn model within its own truncation window, and is censored as the
    data were -- a censored unit at its own observed censoring time, a
    failed one at a time drawn from the Kaplan-Meier estimate of the
    censoring distribution beyond its failure (the design of the
    univariate family's bootstrap, so fixed-time censoring is reproduced
    exactly and random censoring from the censoring times observed). Two
    discrepancies are compared with their replicates:

    - the Anderson-Darling statistic of the Kaplan-Meier estimate of
      ``u = F(x | theta)`` against the uniform (the univariate family's
      statistic, :func:`~surpyval.diagnostics.univariate.gof_statistics`),
      a *realised* discrepancy: for each draw ``theta`` it is computed for
      the observed and for the replicated data at the same ``theta``;
    - the failures in time bins: the observed failures counted in up to
      :data:`MAX_BINS` bins (one per :data:`FAILURES_PER_BIN` failures,
      between the observed failure times' quantiles), against their mean
      over the replicates, by Pearson's ``sum (O - E)^2 / E``. With few
      failures there is one bin, and the check is of the number of
      failures.

    The posterior predictive p-value is the share of draws whose
    replicate is at least as extreme as the data, ``(1 + k) / (1 + B)``
    as the package's bootstrap p-values. It is a check, not an exact
    test: under a correct model it is conservative (it piles up near 0.5
    rather than being uniform; Robins, van der Vaart and Ventura, 2000),
    so a small value is strong evidence against the model and a moderate
    one is not evidence for it. Left- and interval-censored data have no
    such check (a replicate would need the inspection times, which the
    data do not record), and the rows say so and warn.
``goodness_of_fit``
    The univariate family's goodness-of-fit tests (Anderson-Darling,
    Kolmogorov-Smirnov and Cramer-von Mises with parametric bootstrap
    p-values; see :mod:`surpyval.diagnostics.univariate`) of the same
    distribution fitted to the same data by maximum likelihood,
    ``model.mle``. They are run on the maximum-likelihood fit, not at the
    posterior's centre: the bootstrap's null distribution is that of a
    maximum-likelihood refit, and the statistic at any other point would
    not be comparable with it. They check the distribution, which the
    prior does not change. Where there is no maximum-likelihood fit (no
    failures, or a likelihood with no finite maximum) or the univariate
    family refuses it, the check is left out, and a note says why.
``prior_data_conflict``
    Whether the prior and the data disagree, by Box's (1980) prior
    predictive check in its normal form (Evans and Moshonov, 2006;
    Presanis et al., 2013): on each parameter's unconstrained scale (log
    for a positive one, logit for a proportion), the prior and the
    likelihood are each approximated by a normal (the prior's mean and
    covariance; the maximum-likelihood estimate and the inverse of minus
    the log-likelihood's Hessian there). If the prior is right, the
    estimate is a draw from the prior plus estimation error, so the gap
    between the two centres, ``delta``, has covariance ``S = V_prior +
    V_data``, and ``delta' S^-1 delta`` is chi-square with one degree of
    freedom per parameter. In plain words: how many standard deviations
    of disagreement separate what the prior says from what the data say.
    One row per parameter with a proper prior, and a joint row when more
    than one has one. A flat parameter says nothing to disagree with and
    is not checked; a power prior's old data are a likelihood, not a
    belief, and are not checked either. Where the data alone have no
    maximum-likelihood fit the rows warn, saying so.
``grid_accuracy``
    For a grid posterior, two rows: the posterior mass in the grid's
    outermost cells (a screen that warns above :data:`EDGE_MASS`: the
    grid may cut off a tail), and the largest change in any free
    parameter's 2.5%, 50% and 97.5% points when the posterior is computed
    again on a finer grid (twice the points a dimension, at least the
    grid's default start and at most its largest; the finer grid runs
    its own accuracy check), as a fraction of the finer 95% interval's
    width. For a Laplace approximation, one row: the same change against
    the grid posterior. The change is not a test: it warns at :data:`GRID_WARN`
    (1% of the interval's width) and fails at :data:`GRID_FAIL` (5%), a
    numerical error that large changes what the summary reports. A closed
    form (``method == "conjugate"``) is exact and another sampler's draws
    (``from_draws``) have their sampler's own diagnostics: neither has
    rows, and a note says so.

What is diagnosed
-----------------
A Bayesian fit of a continuous distribution with its data. ``diagnose()``
refuses, saying what to do: a model without data (``from_draws`` without
``data=``, or one restored without them) and a discrete distribution (a
``Bernoulli``, whose one parameter fits the proportion of events
exactly). Weighted data cannot arise: the Bayesian fits refuse weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from surpyval.utils.warnings import ignoring_warnings

from . import univariate
from ._shared import _fmt, _note
from .registry import AdviceRule, Check, Context, Family, Panel, register

#: The posterior draws (one replicated data set each) of the posterior
#: predictive checks when ``diagnose()`` is not given ``n_boot``.
DEFAULT_N_DRAWS = 1000
#: The most bins of the failures-in-bins check, and the failures a bin.
MAX_BINS = 10
FAILURES_PER_BIN = 5
#: The posterior mass in a grid's outermost cells above which the edge
#: screen warns.
EDGE_MASS = 1e-6
#: The change of a summary on a finer grid (or against the grid, for a
#: Laplace approximation), as a fraction of the 95% interval's width, at
#: which the grid-accuracy row warns, and at which it fails.
GRID_WARN = 0.01
GRID_FAIL = 0.05
#: The quantiles of each parameter the grid-accuracy check compares.
_QUANTILES = (0.025, 0.5, 0.975)


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
def _data(model: Any) -> dict:
    """The model's data as ``x``, ``c``, ``n``, ``t`` arrays (``x`` the
    left column of interval data)."""
    data = model.data
    c = np.asarray(data.c)
    x = np.asarray(data.x, dtype=float).reshape(c.size, -1)[:, 0]
    return {
        "x": x,
        "c": c,
        "n": np.asarray(data.n, dtype=float),
        "t": np.asarray(data.t, dtype=float).reshape(-1, 2),
    }


def _simulable(model: Any) -> bool:
    c = np.asarray(model.data.c)
    return bool(np.all((c == 0) | (c == 1)))


def _bin_edges(x: np.ndarray, n: np.ndarray) -> np.ndarray:
    """The inner edges of the failure bins: the quantiles of the failure
    times (counted ``n`` times each), one bin per
    :data:`FAILURES_PER_BIN` failures, at most :data:`MAX_BINS`."""
    failures = int(round(float(n.sum())))
    k = int(np.clip(failures // FAILURES_PER_BIN, 1, MAX_BINS))
    if k == 1:
        return np.array([], dtype=float)
    times = np.repeat(x, n.astype(int))
    return np.unique(np.quantile(times, np.arange(1, k) / k))


def _counts(x: np.ndarray, n: Any, edges: np.ndarray) -> np.ndarray:
    w = None if n is None else np.asarray(n, dtype=float)
    return np.bincount(
        np.searchsorted(edges, x, side="right"),
        weights=w,
        minlength=edges.size + 1,
    ).astype(float)


def _pearson(counts: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """``sum (O - E)^2 / E`` over the bins with ``E > 0``, for each row of
    ``counts``."""
    keep = expected > 0
    diff = np.atleast_2d(counts)[:, keep] - expected[keep]
    return np.sum(diff**2 / expected[keep], axis=1)


# ---------------------------------------------------------------------------
# The posterior predictive checks
# ---------------------------------------------------------------------------
def posterior_predictive(
    model: Any, n_draws: int = DEFAULT_N_DRAWS, random_state: Any = None
) -> dict:
    """
    The posterior predictive checks of a Bayesian fit: ``n_draws``
    replicated data sets, one from each of ``n_draws`` posterior draws,
    simulated under the observed censoring and truncation design (see
    :mod:`surpyval.diagnostics.bayes` for the scheme and the
    discrepancies).

    Returns a dictionary: ``p``, the posterior predictive p-values by
    discrepancy (``"AD"``, the realised Anderson-Darling discrepancy, and
    ``"bins"``, the failures in time bins); ``observed`` and
    ``replicated``, the Anderson-Darling discrepancy of the data and of
    the replicate at each draw; ``edges``, the bins' inner edges;
    ``observed_counts`` and ``replicated_counts`` (one row a draw), the
    failures in each bin; ``expected``, their mean over the replicates;
    ``pearson``, the observed and replicated ``sum (O - E)^2 / E``; and
    ``n_draws``.

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.diagnostics.bayes import posterior_predictive
    >>> x = sp.Weibull.random(30, 10, 2, random_state=1)
    >>> model = sp.Weibull.fit_bayes(
    ...     x, priors={"beta": sp.prior.around(2, between=(1, 4))})
    >>> out = posterior_predictive(model, 200, random_state=2)
    >>> sorted(out["p"]), out["replicated"].shape
    (['AD', 'bins'], (200,))
    >>> bool(out["p"]["AD"] > 0.05)
    True
    """
    from surpyval.univariate.regression._bootstrap import check_n_boot
    from surpyval.utils.rng import as_generator

    n_draws = check_n_boot(n_draws)
    data = _data(model)
    design = univariate._design(SimpleNamespace(data=data))
    rng = as_generator(random_state)
    t_units = np.column_stack([design.tl, design.tr])
    failed = data["c"] == 0
    edges = _bin_edges(data["x"][failed], data["n"][failed])
    observed_counts = _counts(data["x"][failed], data["n"][failed], edges)
    longest = float(np.max(design.x))
    observed = np.empty(n_draws)
    replicated = np.empty(n_draws)
    counts = np.empty((n_draws, edges.size + 1))
    with ignoring_warnings():
        drawn = model.draw_models(n_draws, random_state=rng)
    for i, m in enumerate(drawn):
        with np.errstate(all="ignore"):
            u = univariate.transform(m, data["x"], data["t"])
            observed[i] = univariate.gof_statistics(u, failed, data["n"])["AD"]
            lo, hi = univariate._window(m, t_units)
            q = lo + rng.uniform(size=design.x.size) * (hi - lo)
            T = np.asarray(m.qf(q), dtype=float)
        C = design.censoring(rng)
        censored = ~(T <= C)
        x = np.where(censored, C, T)
        # A unit that never fails (a limited failure population) and is not
        # censored by the censoring distribution is censored at the last
        # time observed.
        x = np.where(np.isfinite(x), x, longest)
        with np.errstate(all="ignore"):
            u = univariate.transform(m, x, t_units)
            replicated[i] = univariate.gof_statistics(u, ~censored)["AD"]
        counts[i] = _counts(x[~censored], None, edges)
    expected = counts.mean(axis=0)
    pearson_obs = float(_pearson(observed_counts, expected)[0])
    pearson_rep = _pearson(counts, expected)
    tie = 1e-12
    p_ad = (1 + np.sum(replicated >= observed - tie)) / (1 + n_draws)
    p_bins = (1 + np.sum(pearson_rep >= pearson_obs - tie)) / (1 + n_draws)
    return {
        "p": {"AD": float(p_ad), "bins": float(p_bins)},
        "observed": observed,
        "replicated": replicated,
        "edges": edges,
        "observed_counts": observed_counts,
        "replicated_counts": counts,
        "expected": expected,
        "pearson": (pearson_obs, pearson_rep),
        "n_draws": n_draws,
    }


def _ppc(ctx: Context) -> dict | None:
    """The report's posterior predictive checks, or None for data they
    cannot simulate."""
    if not _simulable(ctx.model):
        return None
    n = DEFAULT_N_DRAWS if ctx.n_boot is None else ctx.n_boot
    return ctx.cached(
        "ppc",
        lambda: posterior_predictive(ctx.model, n, ctx.random_state),
    )


def _dist_name(model: Any) -> str:
    return univariate._label(model)


def _ppc_rows(ctx: Context) -> list[dict]:
    ppc = _ppc(ctx)
    tests = (
        ("posterior predictive: Anderson-Darling", "AD"),
        ("posterior predictive: failures in time bins", "bins"),
    )
    if ppc is None:
        detail = (
            "not computed: the data are left- or interval-censored, and a "
            "replicate would need the inspection times, which they do not "
            "record"
        )
        return [
            {
                "test": test,
                "statistic": float("nan"),
                "p": float("nan"),
                "detail": detail,
                "_key": key,
            }
            for test, key in tests
        ]
    rows = []
    name = _dist_name(ctx.model)
    for test, key in tests:
        p = ppc["p"][key]
        if key == "AD":
            statistic = float(np.median(ppc["observed"]))
            what = "the replicates are as far from the {} as the data".format(
                name
            )
        else:
            statistic = ppc["pearson"][0]
            bins = ppc["expected"].size
            what = (
                "the replicates' failures {} as far from expected as the "
                "data's".format(
                    f"in {bins} time bins are" if bins > 1 else "are"
                )
            )
        share = "{:.1%} of {} draws".format(p, ppc["n_draws"])
        if p < 2 * ctx.alpha:
            detail = (
                "the {} does not reproduce the data: {} in only {}".format(
                    name, what, share
                )
            )
        else:
            detail = f"{what} in {share}"
        rows.append(
            {
                "test": test,
                "statistic": statistic,
                "p": p,
                "detail": detail,
                "_key": key,
            }
        )
    return rows


def _ppc_panels(ctx: Context) -> list[Panel]:
    ppc = _ppc(ctx)
    if ppc is None:
        return []

    def scatter(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        obs, rep = ppc["observed"], ppc["replicated"]
        ax.scatter(obs, rep, s=6, color=PALETTE[0], alpha=0.5)
        top = float(np.nanmax(np.r_[obs, rep]))
        ax.plot([0, top], [0, top], color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel("Anderson-Darling, observed data")
        ax.set_ylabel("Anderson-Darling, replicate")

    def bins(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        k = ppc["expected"].size
        level = 1 - options.alpha
        lo, hi = np.quantile(
            ppc["replicated_counts"], [(1 - level) / 2, (1 + level) / 2], 0
        )
        where = np.arange(k)
        ax.bar(where, ppc["expected"], color=PALETTE[0], alpha=0.4)
        ax.vlines(where, lo, hi, color=PALETTE[0])
        ax.plot(where, ppc["observed_counts"], "o", color=INK["primary"], ms=4)
        ax.set_xticks(where)
        ax.set_xlabel("time bin (observed: points)")
        ax.set_ylabel("failures")

    return [
        Panel(
            "realised discrepancy (AD p = {})".format(_fmt(ppc["p"]["AD"])),
            scatter,
        ),
        Panel(
            "failures in time bins (p = {})".format(_fmt(ppc["p"]["bins"])),
            bins,
        ),
    ]


# ---------------------------------------------------------------------------
# Goodness of fit of the maximum-likelihood fit (the univariate family's)
# ---------------------------------------------------------------------------
def _ml_context(ctx: Context) -> Context | None:
    """A univariate report's context on the maximum-likelihood fit of the
    same data, or None (with a note) where the univariate checks do not
    apply."""

    def build() -> Context | None:
        with ignoring_warnings():
            mle = ctx.model.mle
        if mle is None:
            _note(
                ctx,
                "Goodness of fit: not checked, because the data alone have "
                "no maximum-likelihood fit to test (no failures, or a "
                "likelihood with no finite maximum); the posterior "
                "predictive checks still apply.",
            )
            return None
        try:
            univariate._validate(mle)
        except (ValueError, NotImplementedError) as e:
            _note(ctx, "Goodness of fit: not checked. " + str(e))
            return None
        return Context(
            mle,
            ctx.alpha,
            [],
            [],
            n_boot=ctx.n_boot,
            random_state=ctx.random_state,
        )

    return ctx.cached("ml_context", build)


def _gof_rows(ctx: Context) -> list[dict]:
    sub = _ml_context(ctx)
    if sub is None:
        return []
    rows = []
    for row in univariate._gof_rows(sub):
        row = {k: v for k, v in row.items() if k != "alternative"}
        row["test"] = row["test"].replace(
            "(parametric bootstrap)", "(parametric bootstrap, ML fit)"
        )
        rows.append(row)
    for text in sub.notes:
        _note(ctx, text)
    return rows


def _gof_panels(ctx: Context) -> list[Panel]:
    sub = _ml_context(ctx)
    if sub is None:
        return []
    return univariate._gof_panels(sub)


# ---------------------------------------------------------------------------
# Prior-data conflict
# ---------------------------------------------------------------------------
def _prior_moments(model: Any, bound: Any) -> tuple[list[str], Any, Any]:
    """The names of the free parameters with a proper prior, and the
    prior's mean and covariance of their unconstrained values."""
    from surpyval.bayes.prior import MultivariateNormal

    if bound.proper:
        from surpyval.bayes.fitting import prior_model

        with ignoring_warnings():
            prior = prior_model(
                model.dist,
                model.priors,
                _bounds_data=model.data,
                offset=model.offset,
                lfp=model.lfp,
                zi=model.zi,
                fixed={
                    k: v
                    for k, v in model.fixed.items()
                    if k not in model.priors.fixed()
                }
                or None,
            )
        post = prior.posterior
        order = [post.parameter_names.index(n) for n in bound.names]
        phi = bound.to_phi(post.points[:, order])
        w = post.weights
        mean = w @ phi
        centred = phi - mean
        cov = centred.T @ (centred * w[:, None])
        return list(bound.free_names), mean, cov
    names: list[str] = []
    means: list[float] = []
    variances: dict[tuple[str, str], float] = {}
    rng = np.random.default_rng(0)
    for name, p in model.priors.single.items():
        if p.is_fixed or not p.proper or name not in bound.free_names:
            continue
        t = bound.transforms[bound.names.index(name)]
        draws = np.asarray(p.draw_phi(20000, t, rng), dtype=float)
        names.append(name)
        means.append(float(np.mean(draws)))
        variances[(name, name)] = float(np.var(draws))
    for joint in model.priors.joint:
        if not isinstance(joint, MultivariateNormal):
            continue
        joint_names = list(joint.names or [])
        for i, a in enumerate(joint_names):
            names.append(a)
            means.append(float(joint.mean[i]))
            for j, b in enumerate(joint_names):
                variances[(a, b)] = float(joint.cov[i, j])
    cov = np.array(
        [[variances.get((a, b), 0.0) for b in names] for a in names]
    ).reshape(len(names), len(names))
    return names, np.array(means), cov


def prior_data_conflict(model: Any) -> dict:
    """
    The prior-data conflict check of a Bayesian fit: Box's prior
    predictive check in its normal form (see
    :mod:`surpyval.diagnostics.bayes`).

    Returns a dictionary: ``names``, the parameters checked (those with a
    proper prior); ``prior_mean`` and ``prior_cov``, the prior's mean and
    covariance of their unconstrained values (log, logit); ``data_mean``
    and ``data_cov``, the maximum-likelihood estimate and its covariance
    there; ``z``, each parameter's gap in standard deviations of ``S =
    prior_cov + data_cov``; ``p``, each one's two-sided p-value;
    ``joint``, ``(statistic, df, p)`` of the chi-square over all of them;
    and ``reason``, why there is nothing to check (``None`` otherwise).

    Examples
    --------
    >>> import surpyval as sp
    >>> from surpyval.diagnostics.bayes import prior_data_conflict
    >>> x = sp.Weibull.random(40, 10, 2, random_state=1)
    >>> model = sp.Weibull.fit_bayes(
    ...     x, priors={"beta": sp.prior.around(6, between=(5, 7))})
    >>> out = prior_data_conflict(model)
    >>> out["names"], bool(out["p"]["beta"] < 0.01)
    (['beta'], True)
    """
    from scipy import stats

    from surpyval.bayes.engines.grid import find_mode, hessian
    from surpyval.bayes.likelihood import LogLikelihood

    bound = model._bound()
    out: dict[str, Any] = {"names": [], "reason": None}
    try:
        names, prior_mean, prior_cov = _prior_moments(model, bound)
    except ValueError as e:
        out["names"] = list(bound.free_names)
        out["reason"] = "the prior alone could not be computed ({})".format(
            str(e).split(". ")[0]
        )
        return out
    if not names:
        out["reason"] = (
            "no parameter has a proper prior, so the prior says nothing "
            "the data could disagree with"
        )
        return out
    with ignoring_warnings():
        mle = model.mle
    if mle is None or getattr(mle, "maximum", None) == "no finite maximum":
        out["names"] = names
        out["reason"] = (
            "the data alone have no maximum-likelihood fit (no failures, or "
            "a likelihood with no finite maximum), so there is no estimate "
            "to compare with the prior"
        )
        return out
    ll = LogLikelihood(
        model.dist, model.data, model.offset, model.lfp, model.zi
    )

    def loglik(phi: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            v = np.asarray(ll(bound.full_theta(phi)), dtype=float)
        return np.where(np.isnan(v), -np.inf, v)

    values = dict(zip(mle.dist.parameter_names, np.atleast_1d(mle.params)))
    values.update(
        gamma=getattr(mle, "gamma", 0.0),
        lfp_p=getattr(mle, "lfp_p", 1.0),
        f0=getattr(mle, "f0", 0.0),
    )
    theta = np.array([[float(values[n]) for n in bound.names]])
    start = bound.to_phi(theta)[0]
    reason = (
        "the data alone do not pin the parameters down (the "
        "log-likelihood is flat in some direction at its maximum)"
    )
    try:
        mode, cov, _ = find_mode(loglik, [start])
        h = np.clip(0.2 * np.sqrt(np.diag(cov)), 1e-5, 1.0)
        H = -hessian(loglik, mode, h)
        H = 0.5 * (H + H.T)
        ok = bool(np.all(np.isfinite(H))) and bool(
            np.all(np.linalg.eigvalsh(H) > 1e-8)
        )
    except (ValueError, np.linalg.LinAlgError):
        ok = False
    if not ok:
        out["names"] = names
        out["reason"] = reason
        return out
    data_cov_all = np.linalg.inv(H)
    idx = [bound.free_names.index(n) for n in names]
    data_mean = mode[idx]
    data_cov = data_cov_all[np.ix_(idx, idx)]
    S = prior_cov + data_cov
    delta = data_mean - prior_mean
    z = delta / np.sqrt(np.diag(S))
    statistic = float(delta @ np.linalg.solve(S, delta))
    out.update(
        names=names,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        data_mean=data_mean,
        data_cov=data_cov,
        z={n: float(v) for n, v in zip(names, z)},
        p={n: float(2 * stats.norm.sf(abs(v))) for n, v in zip(names, z)},
        joint=(
            statistic,
            len(names),
            float(stats.chi2.sf(statistic, len(names))),
        ),
    )
    return out


def _conflict(ctx: Context) -> dict:
    return ctx.cached("conflict", lambda: prior_data_conflict(ctx.model))


def _natural(bound: Any, name: str, phi: float) -> float:
    t = bound.transforms[bound.names.index(name)]
    return float(t.to_theta(np.atleast_1d(phi))[0])


def _conflict_rows(ctx: Context) -> list[dict]:
    out = _conflict(ctx)
    if not out["names"]:
        _note(
            ctx,
            "Prior-data conflict: not checked, because " + out["reason"] + ".",
        )
        return []
    if out["reason"] is not None:
        return [
            {
                "test": f"prior vs data: {name}",
                "statistic": float("nan"),
                "p": float("nan"),
                "df": 1,
                "detail": "not computed: " + out["reason"],
                "_name": name,
            }
            for name in out["names"]
        ]
    bound = ctx.cached("bound", ctx.model._bound)
    rows = []
    for j, name in enumerate(out["names"]):
        z, p = out["z"][name], out["p"][name]
        prior_at = _natural(bound, name, out["prior_mean"][j])
        data_at = _natural(bound, name, out["data_mean"][j])
        rows.append(
            {
                "test": f"prior vs data: {name}",
                "statistic": z**2,
                "df": 1,
                "p": p,
                "detail": "the prior centres {} at {}, the data alone at {}: "
                "{} standard deviations apart".format(
                    name, _fmt(prior_at), _fmt(data_at), _fmt(abs(z))
                ),
                "_name": name,
                "_prior": prior_at,
                "_data": data_at,
            }
        )
    if len(out["names"]) > 1:
        statistic, df, p = out["joint"]
        rows.append(
            {
                "test": "prior vs data: all parameters with a prior",
                "statistic": statistic,
                "df": df,
                "p": p,
                "detail": "joint chi-square over {}".format(
                    ", ".join(out["names"])
                ),
                "_name": None,
            }
        )
    return rows


def _conflict_panels(ctx: Context) -> list[Panel]:
    out = _conflict(ctx)
    if not out["names"] or out["reason"] is not None:
        return []
    bound = ctx.cached("bound", ctx.model._bound)
    post = ctx.model.posterior

    def draw(ax: Any, options: Any) -> None:
        from scipy import stats

        from surpyval.plotting import PALETTE

        z = stats.norm.ppf(1 - options.alpha / 2)
        labels = []
        for j, name in enumerate(out["names"]):
            spans = []
            for mean, var in (
                (out["prior_mean"][j], out["prior_cov"][j, j]),
                (out["data_mean"][j], out["data_cov"][j, j]),
            ):
                sd = np.sqrt(var)
                spans.append(
                    [
                        _natural(bound, name, mean - z * sd),
                        _natural(bound, name, mean),
                        _natural(bound, name, mean + z * sd),
                    ]
                )
            a = options.alpha
            lo, med, hi = post.quantile(name, [a / 2, 0.5, 1 - a / 2])
            spans.append([float(lo), float(med), float(hi)])
            for k, (lo_k, mid, hi_k) in enumerate(spans):
                y = 3 * j + k
                ax.plot([lo_k, hi_k], [y, y], color=PALETTE[k], lw=2)
                ax.plot([mid], [y], "o", color=PALETTE[k], ms=4)
            labels += [f"{name}: prior", f"{name}: data", f"{name}: posterior"]
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        from matplotlib.ticker import NullFormatter, ScalarFormatter

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel(f"{1 - options.alpha:.0%} intervals")

    return [Panel("prior, data and posterior", draw)]


# ---------------------------------------------------------------------------
# Grid accuracy
# ---------------------------------------------------------------------------
def _refit(model: Any, method: str, points: int | None) -> Any:
    fixed = {
        k: v for k, v in model.fixed.items() if k not in model.priors.fixed()
    }
    with ignoring_warnings():
        return model.dist.fit_bayes(
            model.data,
            priors=model.priors,
            offset=model.offset,
            lfp=model.lfp,
            zi=model.zi,
            fixed=fixed or None,
            method=method,
            points=points,
        )


def _change(model: Any, other: Any) -> tuple[float, str]:
    """The largest change of a free parameter's quantiles between the
    model and ``other``, as a fraction of ``other``'s 95% interval's
    width, and where it is."""
    worst, where = 0.0, ""
    for name in model.posterior.free_names:
        a = np.asarray(model.posterior.quantile(name, _QUANTILES), float)
        b = np.asarray(other.posterior.quantile(name, _QUANTILES), float)
        width = abs(b[-1] - b[0])
        width = width if width > 0 else 1.0
        rel = np.abs(a - b) / width
        k = int(np.argmax(rel))
        if rel[k] > worst or not where:
            worst = float(rel[k])
            where = f"{name}'s {100 * _QUANTILES[k]:g}% point"
    return worst, where


def _accuracy_verdict(change: float) -> str:
    if change >= GRID_FAIL:
        return "fail"
    if change >= GRID_WARN:
        return "warn"
    return "ok"


def _edge_mass(model: Any) -> float:
    post = model.posterior
    shape = post.shape
    W = post.weights.reshape(shape)
    mask = np.zeros(shape, dtype=bool)
    for axis in range(1, len(shape)):
        index: list[Any] = [slice(None)] * len(shape)
        index[axis] = [0, shape[axis] - 1]
        mask[tuple(index)] = True
    return float(W[mask].sum())


def _grid_rows(ctx: Context) -> list[dict]:
    from surpyval.bayes.engines.grid import MAX_POINTS, START_POINTS

    model = ctx.model
    if model.method not in ("grid", "laplace"):
        return []
    d = len(model.posterior.free_names)
    rows = []
    if model.method == "grid":
        mass = _edge_mass(model)
        rows.append(
            {
                "test": "posterior mass at the grid's edge",
                "statistic": mass,
                "verdict": "warn" if mass > EDGE_MASS else "ok",
                "detail": "{} of the posterior in the grid's outermost "
                "cells{}".format(
                    _fmt(mass),
                    (
                        ": the grid may cut off a tail"
                        if mass > EDGE_MASS
                        else ""
                    ),
                ),
                "_kind": "edge",
            }
        )
        m = max(int(v) for v in model.diagnostics.get("points_per_dim", [0]))
        points = int(max(START_POINTS[d], min(2 * m - 1, MAX_POINTS[d])))
        test = "summaries on a finer grid"
        against = "a grid of {} points a parameter (this one: {})".format(
            points,
            " x ".join(
                str(v) for v in model.diagnostics.get("points_per_dim", [])
            ),
        )
        method = "grid"
    else:
        points = None
        test = "Laplace approximation against the grid"
        against = "the grid posterior"
        method = "grid"
    try:
        other = ctx.cached("finer", lambda: _refit(model, method, points))
    except ValueError as e:
        rows.append(
            {
                "test": test,
                "statistic": float("nan"),
                "verdict": "warn",
                "detail": "not computed: {} could not be computed ({})".format(
                    against, str(e).split(". ")[0]
                ),
                "_kind": "not computed",
            }
        )
        return rows
    change, where = _change(model, other)
    rows.append(
        {
            "test": test,
            "statistic": change,
            "verdict": _accuracy_verdict(change),
            "detail": "against {}, {} moves by {:.2%} of its 95% interval's "
            "width".format(against, where, change),
            "_kind": model.method,
            "_points": points,
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Refusals and notes
# ---------------------------------------------------------------------------
def _validate(model: Any) -> None:
    dist = model.dist
    name = getattr(dist, "name", type(dist).__name__)
    if getattr(dist, "discrete", False):
        raise ValueError(
            "diagnose() checks a distribution of times, and a {} model has "
            "none: its one parameter is the event probability p, whose "
            "posterior fits the proportion of events. See what the prior "
            "did with model.compare() and model.prior_check().".format(name)
        )
    if model.data is None:
        raise ValueError(
            "diagnose() needs the data the model was fitted to, which this "
            "model does not have (it was built from posterior draws without "
            "data=, or restored without them). Pass the data to "
            "BayesianParametric.from_draws(..., data=(x, c, n, t)), or "
            "refit with fit_bayes."
        )


def _notes(ctx: Context) -> list[str]:
    notes = []
    ppc = _ppc(ctx)
    if ppc is not None:
        notes.append(
            "Posterior predictive checks: {} posterior draws, each with a "
            "data set replicated under the observed design (every unit "
            "within its truncation window; a censored unit censored at its "
            "own observed censoring time, a failed one at a time drawn from "
            "the censoring distribution beyond its failure). A p-value is "
            "the share of replicates at least as far from the model as the "
            "data; the check is conservative, so a moderate p-value is no "
            "evidence for the model. diagnose(n_boot=..., random_state=...) "
            "sets the draws and their seed.".format(ppc["n_draws"])
        )
    notes.append(
        "Prior-data conflict: Box's prior predictive check in its normal "
        "form, the gap between the prior's centre and the "
        "maximum-likelihood estimate on each parameter's log (or logit) "
        "scale, in standard deviations of the two together."
    )
    if ctx.model.method == "conjugate":
        notes.append(
            "Grid accuracy: not checked, because the posterior is a closed "
            "form ({}), exact.".format(ctx.model.diagnostics.get("case", ""))
        )
    elif ctx.model.method == "draws":
        notes.append(
            "Grid accuracy: not checked, because the posterior is another "
            "sampler's draws; check their convergence with that sampler's "
            "diagnostics (R-hat, effective sample size), kept in "
            "model.diagnostics."
        )
    return notes


# ---------------------------------------------------------------------------
# Advice
# ---------------------------------------------------------------------------
def _distribution_message(row: dict, ctx: Context) -> str:
    return (
        "The {model} does not reproduce the data. fit_best(x, c, n, t, "
        "return_table=True) ranks the distributions by AIC; refit the one "
        "it chooses with its fit_bayes and priors on its own parameters, "
        "and diagnose that.".format(model=_dist_name(ctx.model))
    )


def _prior_to_blame(row: Any, ctx: Context) -> bool:
    """Whether a failed posterior predictive check is the prior's doing:
    the prior and the data disagree, or the maximum-likelihood fit of the
    same distribution passes its goodness-of-fit tests."""
    from .report import verdict

    if any(
        verdict(r.get("p"), ctx.alpha) == "fail" for r in _conflict_rows(ctx)
    ):
        return True
    gof = _gof_rows(ctx)
    return bool(gof) and all(
        verdict(r.get("p"), ctx.alpha) == "ok" for r in gof
    )


def _prior_pulls_message(row: dict, ctx: Context) -> str:
    return (
        "The posterior does not reproduce the data, but the data do not "
        "rule out the {model} itself: the prior pulls the posterior away "
        "from what the data say. model.compare() sets the prior, the "
        "maximum-likelihood fit and the posterior side by side; check the "
        "prior's source, and widen it (prior.around(value, between=(low, "
        "high))) if it was for other units or conditions.".format(
            model=_dist_name(ctx.model)
        )
    )


def _conflict_message(row: dict, ctx: Context) -> str:
    if row["_name"] is None:
        return (
            "The priors and the data disagree taken together. "
            "model.compare() sets the prior, the maximum-likelihood fit and "
            "the posterior side by side for each parameter."
        )
    return (
        "The prior and the data disagree on {name}: the prior centres it "
        "at {prior}, the data alone at {data}. Neither is taken as right: "
        "check where the prior came from and the data (model.compare() "
        "sets the two side by side). If the prior was for other units or "
        "conditions, widen it (prior.around(value, between=(low, high)) "
        "with a wider range) so that the data can move the "
        "posterior.".format(
            name=row["_name"],
            prior=_fmt(row["_prior"]),
            data=_fmt(row["_data"]),
        )
    )


def _edge_message(row: dict, ctx: Context) -> str:
    return (
        "The grid's edge holds {mass} of the posterior: a parameter has a "
        "long tail the grid may cut off. Give it a prior "
        "(prior.around(value, between=(low, high))) or refit with more "
        "points ({name}.fit_bayes(..., points=)).".format(
            mass=_fmt(row["statistic"]), name=ctx.model.dist.name
        )
    )


def _grid_message(row: dict, ctx: Context) -> str:
    return (
        "The grid posterior is not accurate to {warn:.0%} of an interval's "
        "width (a finer grid moves it by {change:.2%}). Refit with more "
        "points, {name}.fit_bayes(..., points={points}).".format(
            warn=GRID_WARN,
            change=row["statistic"],
            name=ctx.model.dist.name,
            points=row["_points"],
        )
    )


def _laplace_message(row: dict, ctx: Context) -> str:
    return (
        "The Laplace approximation is off from the grid posterior by "
        "{change:.2%} of an interval's width: the posterior is not normal "
        "on the parameters' log scale. Refit with {name}.fit_bayes(..., "
        "method='grid'), or 'auto'.".format(
            change=row["statistic"], name=ctx.model.dist.name
        )
    )


def _per_dist(
    check: str, kind: str, message: Any, feature: str, *extra: str
) -> list[AdviceRule]:
    """One rule per named distribution (so that each names its own
    ``fit_bayes``), and one, naming no ``fit_bayes``, for any other."""

    def of(name: str) -> Any:
        return (
            lambda row, ctx: row["_kind"] == kind
            and ctx.model.dist.name == name
        )

    rules = [
        AdviceRule(
            check,
            message,
            (feature.format(name), *extra),
            when=of(name),
            on=("fail", "warn"),
        )
        for name in univariate.NAMED
    ]
    rules.append(
        AdviceRule(
            check,
            message,
            extra,
            when=lambda row, ctx: row["_kind"] == kind
            and ctx.model.dist.name not in univariate.NAMED,
            on=("fail", "warn"),
        )
    )
    return rules


def _is_bayesian(model: Any) -> bool:
    from surpyval.bayes.model import BayesianParametric

    return isinstance(model, BayesianParametric)


#: The Bayesian family: its checks, in the order of the table, and its
#: advice.
BAYES = register(
    Family(
        name="Bayesian fit",
        issue=867,
        applies=_is_bayesian,
        validate=_validate,
        covariates=lambda model: [],
        checks=(
            Check(
                "posterior_predictive",
                "Data replicated from the posterior, under the observed "
                "censoring and truncation, look like the data.",
                rows=_ppc_rows,
                panels=_ppc_panels,
            ),
            Check(
                "goodness_of_fit",
                "The data are a sample from the distribution fitted by "
                "maximum likelihood.",
                rows=_gof_rows,
                panels=_gof_panels,
            ),
            Check(
                "prior_data_conflict",
                "The data are plausible under the prior: the prior's centre "
                "and the data's estimate are within their joint spread.",
                rows=_conflict_rows,
                panels=_conflict_panels,
            ),
            Check(
                "grid_accuracy",
                "The posterior was computed accurately: a finer grid (or, "
                "for a Laplace approximation, the grid) gives the same "
                "summaries.",
                rows=_grid_rows,
            ),
        ),
        advice=(
            AdviceRule(
                "posterior_predictive",
                _distribution_message,
                ("fit_best(return_table=)",),
                when=lambda row, ctx: not _prior_to_blame(row, ctx),
            ),
            AdviceRule(
                "posterior_predictive",
                _prior_pulls_message,
                ("bayes.BayesianParametric.compare", "prior.around(between=)"),
                when=_prior_to_blame,
            ),
            AdviceRule(
                "goodness_of_fit",
                _distribution_message,
                ("fit_best(return_table=)",),
            ),
            AdviceRule(
                "prior_data_conflict",
                _conflict_message,
                ("bayes.BayesianParametric.compare", "prior.around(between=)"),
                when=lambda row, ctx: row["_name"] is not None,
            ),
            AdviceRule(
                "prior_data_conflict",
                _conflict_message,
                ("bayes.BayesianParametric.compare",),
                when=lambda row, ctx: row["_name"] is None,
            ),
            *_per_dist(
                "grid_accuracy",
                "edge",
                _edge_message,
                "{}.fit_bayes(points=)",
                "prior.around(between=)",
            ),
            *_per_dist(
                "grid_accuracy", "grid", _grid_message, "{}.fit_bayes(points=)"
            ),
            *_per_dist(
                "grid_accuracy",
                "laplace",
                _laplace_message,
                "{}.fit_bayes(method=)",
            ),
        ),
        notes=_notes,
    )
)
