"""WAIC and leave-one-out cross-validation of Bayesian fits, and the
comparison of fits by them.

The references, none of which depends on the platform:

- the pointwise log-likelihood against the summed one and against a
  likelihood of each row alone (truncation is per row);
- the conjugate fits' closed forms against refitting without each unit
  (LOO) and against numerical integration over the exact posterior
  (WAIC);
- the grid's quadrature against refitting without each unit, and against
  the closed forms where the same posterior is also conjugate;
- Pareto-smoothed importance sampling against ArviZ's (values stored by
  ``scripts/reference/reference_arviz_loo.py``) and against the closed
  form within its Monte Carlo error, with the seed fixed in advance.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

import surpyval as sp
from surpyval.bayes import LogLikelihood
from surpyval.bayes.criteria import good_k, gpd_fit, psis

ARVIZ = json.loads((Path(__file__).parent / "arviz_loo.json").read_text())


def _no_warnings(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return f(*args, **kwargs)


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _without_one(data, j):
    """The data with one unit of row ``j`` left out."""
    n = np.asarray(data.n).copy()
    n[j] -= 1
    keep = n > 0
    return sp.SurpyvalData(
        x=np.asarray(data.x)[keep],
        c=np.asarray(data.c)[keep],
        n=n[keep],
        t=np.asarray(data.t)[keep],
    )


# ----------------------------------------------------------------------
# The pointwise log-likelihood
# ----------------------------------------------------------------------
MIXED = dict(
    x=[0, 1, 2, 3, [1, 4], 5, 6, 7],
    c=[0, 0, 1, -1, 2, 1, -1, 0],
    n=[2, 1, 2, 3, 1, 1, 2, 1],
    tl=[-1, 0, 0.5, 0.5, 0, 0, 1, 0],
    tr=[np.inf] * 5 + [9, np.inf, np.inf],
)

FORMS = [
    (sp.Weibull, {}, [[3.0, 1.5], [5.0, 2.0]]),
    (sp.Weibull, dict(offset=True, lfp=True), [[-0.5, 3.0, 1.5, 0.8]]),
    (sp.Weibull, dict(lfp=True, zi=True), [[3.0, 1.5, 0.6, 0.1]]),
    (sp.Weibull, dict(zi=True), [[3.0, 1.5, 0.1]]),
    (sp.Normal, {}, [[1.0, 2.0], [3.0, 0.7]]),
    (sp.Gamma, {}, [[2.0, 0.5]]),
]


@pytest.mark.parametrize("dist, form, points", FORMS)
def test_the_rows_sum_to_the_likelihood(dist, form, points):
    data = sp.SurpyvalData(**MIXED)
    ll = LogLikelihood(dist, data, **form)
    P = np.asarray(points)
    rows = ll.pointwise(P)
    assert rows.shape == (len(P), len(data.c))
    finite = np.isfinite(ll(P))
    np.testing.assert_allclose(
        (rows @ data.n)[finite], ll(P)[finite], rtol=1e-12
    )


@pytest.mark.parametrize("dist, form, points", FORMS)
def test_each_row_is_its_own_likelihood(dist, form, points):
    """Each row against the likelihood of a data set of that row alone:
    the row's own truncation window, not the data's."""
    data = sp.SurpyvalData(**MIXED)
    P = np.asarray(points)
    rows = LogLikelihood(dist, data, **form).pointwise(P)
    for j in range(len(data.c)):
        alone = sp.SurpyvalData(
            x=[data.x[j]], c=[data.c[j]], n=[1], t=[data.t[j]]
        )
        expect = LogLikelihood(dist, alone, **form)(P)
        np.testing.assert_allclose(rows[:, j], expect, rtol=1e-12)


def test_pointwise_refuses_the_wrong_number_of_parameters():
    ll = LogLikelihood(sp.Weibull, sp.SurpyvalData(x=[1.0, 2.0]))
    with pytest.raises(ValueError, match="2 values in its last axis"):
        ll.pointwise(np.array([1.0, 2.0, 3.0]))


# ----------------------------------------------------------------------
# Conjugate fits: closed forms
# ----------------------------------------------------------------------
GAMMA_PRIOR = {"failure_rate": sp.Gamma.from_params([2.0, 500.0])}
EXPONENTIAL = dict(
    x=[120, 340, 340, 560, 800, 1500, 900],
    c=[0, 0, 0, 0, 0, 1, 1],
    tl=[0, 0, 0, 100, 0, 0, 200],
)
NORMAL = dict(x=[1.2, 2.0, 2.0, 3.5, -0.4, 6.0])
NORMAL_PRIOR = {"mu": sp.Normal.from_params([0.0, 10.0])}


def _exponential(**kw):
    return sp.Exponential.fit_bayes(**EXPONENTIAL, priors=GAMMA_PRIOR, **kw)


def _normal(**kw):
    return sp.Normal.fit_bayes(
        **NORMAL, fixed={"sigma": 1.5}, priors=NORMAL_PRIOR, **kw
    )


def _bernoulli():
    return sp.Bernoulli.fit_bayes(
        [0, 1, 1, 0, 1, 1, 1],
        priors={"p": sp.Beta.from_params([2.0, 2.0])},
    )


def _predictive(model, j):
    """``log p(y_j | y_-j)`` from the refit's exact posterior, in closed
    form with SciPy: the Lomax for the exponential (at the time since
    entry: the exponential forgets its age), the normal, and the
    beta-Bernoulli."""
    data = model.data
    sub = _without_one(data, j)
    dist = model.dist
    if dist.name == "Exponential":
        m = sp.Exponential.fit_bayes(sub, priors=GAMMA_PRIOR)
        a, b = m.posterior.exact["params"]
        tl = data.t[j, 0]
        y = data.x[j] - (max(tl, 0.0) if np.isfinite(tl) else 0.0)
        lomax = stats.lomax(a, scale=b)
        return lomax.logpdf(y) if data.c[j] == 0 else lomax.logsf(y)
    if dist.name == "Normal":
        m = sp.Normal.fit_bayes(sub, fixed={"sigma": 1.5}, priors=NORMAL_PRIOR)
        mean, sd = m.posterior.exact["params"]
        return stats.norm.logpdf(data.x[j], mean, np.hypot(1.5, sd))
    m = sp.Bernoulli.fit_bayes(
        sub, priors={"p": sp.Beta.from_params([2.0, 2.0])}
    )
    a, b = m.posterior.exact["params"]
    return np.log((a if data.x[j] == 1 else b) / (a + b))


@pytest.mark.parametrize("make", [_exponential, _normal, _bernoulli])
def test_conjugate_loo_is_refitting_without_each_unit(make):
    model = make()
    assert model.method == "conjugate"
    res = _no_warnings(model.loo)
    assert res.method == "closed form"
    expect = [_predictive(model, j) for j in range(len(model.data.c))]
    np.testing.assert_allclose(res.pointwise, expect, rtol=1e-12)
    assert res.elpd == pytest.approx(np.asarray(expect) @ model.data.n)


def _log_terms(model, j):
    """``log p(y_j | theta)`` as a function of the one free parameter."""
    d = model.data
    if model.dist.name == "Exponential":
        tl = d.t[j, 0]
        y = d.x[j] - (max(tl, 0.0) if np.isfinite(tl) else 0.0)
        return lambda lam: (np.log(lam) if d.c[j] == 0 else 0.0) - lam * y
    if model.dist.name == "Normal":
        return lambda mu: stats.norm.logpdf(d.x[j], mu, 1.5)
    return lambda p: np.log(p if d.x[j] == 1 else 1 - p)


@pytest.mark.parametrize("make", [_exponential, _normal, _bernoulli])
def test_conjugate_waic_is_the_integral_over_the_posterior(make):
    model = make()
    res = _quiet(model.waic)
    assert res.method == "closed form"
    exact = model.posterior.exact
    family = {
        "gamma": lambda a, b: stats.gamma(a, scale=1 / b),
        "normal": stats.norm,
        "beta": stats.beta,
    }[exact["family"]](*exact["params"])
    lo, hi = family.ppf(1e-15), family.isf(1e-15)
    for j in range(len(model.data.c)):
        g = _log_terms(model, j)

        def moment(h):
            return integrate.quad(
                lambda v: h(v) * family.pdf(v),
                lo,
                hi,
                epsabs=0,
                epsrel=1e-12,
                limit=200,
            )[0]

        m1 = moment(g)
        var = moment(lambda v: (g(v) - m1) ** 2)
        lppd = np.log(moment(lambda v: np.exp(g(v))))
        assert res.p_pointwise[j] == pytest.approx(var, rel=1e-8)
        assert res.pointwise[j] == pytest.approx(lppd - var, rel=1e-8)


def test_the_grid_agrees_with_the_closed_forms():
    """The same posteriors on the grid: its quadrature against the
    closed forms, row by row."""
    for make in (_exponential, _normal):
        conj, grid = make(), make(method="grid")
        assert grid.method == "grid"
        for what in ("loo", "waic"):
            a = _quiet(getattr(conj, what))
            b = _quiet(getattr(grid, what))
            assert b.method == "quadrature"
            np.testing.assert_allclose(b.pointwise, a.pointwise, atol=1e-6)
            np.testing.assert_allclose(b.p_pointwise, a.p_pointwise, atol=1e-6)


# ----------------------------------------------------------------------
# Grid fits: quadrature against refitting
# ----------------------------------------------------------------------
WEIBULL = dict(
    x=[310, 480, 620, 790, 950, 1100, 1300, 1600, 2000, 700],
    c=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    n=[1, 1, 1, 2, 1, 1, 1, 1, 3, 1],
    tl=[0, 0, 100, 0, 0, 300, 0, 0, 0, 0],
)
WEIBULL_PRIOR = {"beta": sp.prior.around(2, between=(1, 4))}


def test_grid_loo_is_refitting_without_each_unit():
    model = sp.Weibull.fit_bayes(**WEIBULL, priors=WEIBULL_PRIOR)
    res = _no_warnings(model.loo)
    assert res.method == "quadrature"
    assert np.all(res.edge < 1e-3)
    data = model.data
    for j in range(len(data.c)):
        refit = sp.Weibull.fit_bayes(
            _without_one(data, j), priors=WEIBULL_PRIOR
        )
        ll = LogLikelihood(sp.Weibull, data).pointwise(refit.posterior.points)
        w = refit.posterior.weights
        assert res.pointwise[j] == pytest.approx(
            np.log(w @ np.exp(ll[:, j])), abs=1e-4
        )


def test_the_lppd_is_the_posterior_predictive_density():
    """For rows without truncation, the log pointwise predictive density
    is the log of the model's own predictive ``df`` or ``sf``."""
    model = sp.Weibull.fit_bayes(
        WEIBULL["x"], WEIBULL["c"], WEIBULL["n"], priors=WEIBULL_PRIOR
    )
    model.predictive_table = False
    res = _quiet(model.waic)
    d = model.data
    lppd = res.pointwise + res.p_pointwise
    expect = np.where(d.c == 0, np.log(model.df(d.x)), np.log(model.sf(d.x)))
    np.testing.assert_allclose(lppd, expect, rtol=1e-10)


def test_waic_and_loo_agree_on_a_well_behaved_fit():
    x = sp.Weibull.random(60, 1000, 2.0, random_state=20261120)
    model = sp.Weibull.fit_bayes(x, priors=WEIBULL_PRIOR)
    a = _no_warnings(model.waic)
    b = _no_warnings(model.loo)
    assert abs(a.elpd - b.elpd) < 0.1
    assert abs(a.p - b.p) < 0.1


def test_a_loo_posterior_past_the_grid_warns():
    """One failure among many survivors, and no prior on alpha: without
    the failure nothing bounds alpha from above, and its leave-one-out
    posterior runs to the grid's edge (2.5% of its mass in the outermost
    cells). (With a uniform prior on alpha up to 1e6 the share is 2e-4,
    under the threshold.)"""
    model = sp.Weibull.fit_bayes(
        [500, 2000],
        c=[0, 1],
        n=[1, 30],
        priors={"beta": sp.prior.around(2, between=(1, 4))},
    )
    with pytest.warns(UserWarning, match="reaches the edge of the grid"):
        res = model.loo()
    assert 0 in res.flagged


# ----------------------------------------------------------------------
# Draws: Pareto-smoothed importance sampling
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(ARVIZ["references"]))
def test_psis_and_waic_agree_with_arviz(case):
    ref = ARVIZ["references"][case]
    spec = ref["case"]
    S = spec["draws"]
    mu = spec["posterior_mean"] + spec["posterior_sd"] * stats.norm.ppf(
        (np.arange(S) + 0.5) / S
    )
    y = np.asarray(spec["y"])
    data = sp.SurpyvalData(x=y)
    draws = np.c_[mu, np.ones(S)]
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Normal, draws, ["mu", "sigma"], data=data
    )
    order = np.argsort(y)  # the data's rows are sorted
    values = ref["values"]
    loo = _quiet(model.loo)
    waic = _quiet(model.waic)
    np.testing.assert_allclose(
        loo.pointwise, np.asarray(values["elpd_loo"])[order], rtol=1e-10
    )
    np.testing.assert_allclose(
        loo.pareto_k, np.asarray(values["pareto_k"])[order], rtol=1e-8
    )
    assert loo.p == pytest.approx(values["p_loo"], rel=1e-9)
    assert loo.good_k == pytest.approx(values["good_k"], rel=1e-14)
    np.testing.assert_allclose(
        waic.pointwise, np.asarray(values["elpd_waic"])[order], rtol=1e-10
    )
    assert waic.p == pytest.approx(values["p_waic"], rel=1e-9)


def test_psis_loo_is_the_closed_form_within_its_monte_carlo_error():
    """Exact draws from the normal-normal posterior, seed fixed in
    advance: PSIS-LOO within four Monte Carlo standard errors of the
    closed form, the error estimated by the delta method from the same
    draws."""
    exact = _normal()
    mean, sd = exact.posterior.exact["params"]
    rng = np.random.default_rng(462)
    S = 4000
    mu = rng.normal(mean, sd, S)
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Normal,
        np.c_[mu, np.full(S, 1.5)],
        ["mu", "sigma"],
        data=exact.data,
    )
    res = _no_warnings(model.loo)
    assert res.method == "psis"
    assert np.all(res.pareto_k < 0.5)
    truth = exact.loo()
    ll = LogLikelihood(sp.Normal, exact.data).pointwise(
        np.c_[mu, np.full(S, 1.5)]
    )
    r = np.exp(-ll)
    g = (r / r.mean(axis=0)) @ exact.data.n
    mcse = np.std(g, ddof=1) / np.sqrt(S)
    assert abs(res.elpd - truth.elpd) < 4 * mcse


def test_an_outlier_has_a_high_pareto_k_and_is_counted():
    """Draws of a normal mean with one observation far out: its
    leave-one-out posterior is far from the posterior, and the warning
    counts it (twice: its row has a count of 2)."""
    x = np.r_[np.linspace(-1.5, 1.5, 19), 9.0]
    n = np.r_[np.ones(19, int), 2]
    S = 4000
    rng = np.random.default_rng(4620)
    prec = n.sum()
    mu = rng.normal((x @ n) / prec, 1 / np.sqrt(prec), S)
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Normal,
        np.c_[mu, np.ones(S)],
        ["mu", "sigma"],
        data=sp.SurpyvalData(x=x, n=n),
    )
    with pytest.warns(
        UserWarning, match=r"2 of the 21 observations \(1 distinct row\) have"
    ):
        res = model.loo()
    assert res.pareto_k[-1] > 0.7
    assert list(res.flagged) == [len(x) - 1]


def test_psis_on_flat_ratios_changes_nothing():
    log_w, k = psis(np.zeros(1000))
    np.testing.assert_allclose(np.exp(log_w), 1e-3)


def test_too_few_draws_leaves_the_tail_alone():
    log_w, k = psis(np.log(np.arange(1.0, 21.0)))
    assert k == np.inf
    np.testing.assert_allclose(
        np.exp(log_w), np.arange(1.0, 21.0) / 210, rtol=1e-12
    )


def test_gpd_fit_recovers_a_known_shape():
    """The expected order statistics of a generalised Pareto with shape
    0.3 and scale 2: the fit (with its weak prior towards 0.5) is close
    to 0.3."""
    u = (np.arange(2000) + 0.5) / 2000
    x = 2.0 * np.expm1(-0.3 * np.log1p(-u)) / 0.3
    k, sigma = gpd_fit(x)
    assert k == pytest.approx(0.3, abs=0.01)
    assert sigma == pytest.approx(2.0, rel=0.05)


def test_good_k_follows_the_sample_size():
    assert good_k(100) == pytest.approx(0.5)
    assert good_k(10_000) == 0.7


def test_loo_refuses_too_few_draws():
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Exponential,
        np.linspace(0.01, 0.02, 10)[:, None],
        data=sp.SurpyvalData(x=[50.0, 80.0]),
    )
    with pytest.raises(ValueError, match="at least 25 draws"):
        model.loo()


# ----------------------------------------------------------------------
# Counts, the standard error and the refusals
# ----------------------------------------------------------------------
def test_counts_are_identical_rows():
    """``n = k`` is ``k`` units: the sums and the standard error are those
    of the units written out."""
    model = _quiet(
        sp.Weibull.fit_bayes,
        [300, 500, 900, 1400],
        c=[0, 0, 0, 1],
        n=[2, 1, 3, 4],
        priors=WEIBULL_PRIOR,
    )
    repeated = _quiet(
        sp.Weibull.fit_bayes,
        [300, 300, 500, 900, 900, 900] + [1400] * 4,
        c=[0] * 6 + [1] * 4,
        priors=WEIBULL_PRIOR,
    )
    for what in ("waic", "loo"):
        a = _quiet(getattr(model, what))
        b = _quiet(getattr(repeated, what))
        assert a.elpd == pytest.approx(b.elpd, rel=1e-12)
        units = np.repeat(a.pointwise, a.n.astype(int))
        assert a.elpd == pytest.approx(units.sum(), rel=1e-12)
        assert a.se == pytest.approx(
            np.sqrt(len(units)) * np.std(units, ddof=1), rel=1e-12
        )
        assert a.ic == -2 * a.elpd


def test_waic_and_loo_need_the_data():
    model = sp.bayes.BayesianParametric.from_draws(
        sp.Exponential, np.linspace(0.01, 0.02, 100)[:, None]
    )
    for what in ("waic", "loo"):
        with pytest.raises(ValueError, match="give data= to from_draws"):
            getattr(model, what)()


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
X = [310, 480, 620, 790, 950, 1100, 1300, 1600]


def test_compare_ranks_by_elpd_with_the_paired_standard_error():
    fits = {
        "exponential": sp.Exponential.fit_bayes(
            X, priors={"failure_rate": sp.prior.between(1e-4, 1e-2)}
        ),
        "weibull": sp.Weibull.fit_bayes(X, priors=WEIBULL_PRIOR),
    }
    out = _quiet(sp.bayes.compare, fits)
    assert out.criterion == "loo"
    assert list(out.table) == ["weibull", "exponential"]
    assert out.best == "weibull"
    a, b = out.results["exponential"], out.results["weibull"]
    row = out.table["exponential"]
    assert row["elpd_diff"] == pytest.approx(a.elpd - b.elpd)
    diff = a.pointwise - b.pointwise
    assert row["se_diff"] == pytest.approx(
        np.sqrt(len(diff)) * np.std(diff, ddof=1)
    )
    assert out.table["weibull"]["elpd_diff"] == 0.0
    assert np.isnan(out.table["weibull"]["se_diff"])
    assert "Comparison by LOO" in repr(out)


def test_compare_names_a_list_by_distribution():
    a = sp.Weibull.fit_bayes(X, priors=WEIBULL_PRIOR)
    b = sp.Weibull.fit_bayes(
        X, priors={"beta": sp.prior.around(3, between=(2, 5))}
    )
    out = _quiet(sp.bayes.compare, [a, b], criterion="waic")
    assert set(out.table) == {"Weibull 1", "Weibull 2"}


def test_compare_refuses_what_does_not_compare():
    w = sp.Weibull.fit_bayes(X, priors=WEIBULL_PRIOR)
    other = sp.Weibull.fit_bayes(X[:-1], priors=WEIBULL_PRIOR)
    with pytest.raises(ValueError, match="fitted to different data"):
        sp.bayes.compare({"a": w, "b": other})
    with pytest.raises(ValueError, match="Compare maximum-likelihood"):
        sp.bayes.compare({"a": w, "b": sp.Weibull.fit(X)})
    with pytest.raises(ValueError, match="at least two fits"):
        sp.bayes.compare([w])
    with pytest.raises(ValueError, match="criterion must be"):
        sp.bayes.compare([w, w], criterion="aic")


def test_compare_says_which_fit_warns():
    w = sp.Weibull.fit_bayes(X, priors=WEIBULL_PRIOR)
    e = sp.Exponential.fit_bayes(
        X, priors={"failure_rate": sp.prior.between(1e-4, 1e-2)}
    )
    with pytest.warns(UserWarning, match="^'weibull': WAIC may be"):
        sp.bayes.compare({"weibull": w, "exp": e}, criterion="waic")


@pytest.mark.parametrize(
    "make",
    [_exponential, lambda: sp.Weibull.fit_bayes(X, priors=WEIBULL_PRIOR)],
)
def test_a_saved_fit_gives_the_same_values(make):
    model = make()
    loaded = sp.bayes.BayesianParametric.from_dict(model.to_dict())
    for what in ("loo", "waic"):
        a = _quiet(getattr(model, what))
        b = _quiet(getattr(loaded, what))
        assert (a.method, a.elpd, a.p) == (b.method, b.elpd, b.p)
