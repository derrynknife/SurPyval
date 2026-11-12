"""Zero-inflated regression (``ZeroInflated``): covariates on the share of
units dead on arrival and on the life of those that work.

Its two routes (two stages where every row says whether its unit was
dead on arrival, joint otherwise) are checked against each other, against
the univariate ``fit(zi=True)`` and against the likelihood written out
by hand.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, logit

import surpyval as sp
from surpyval.univariate.regression.distributional.zero_inflated import (
    _incidence_fit,
)

#: The truth of :func:`_simulate`, in the model's order (the scale's
#: coefficient with SurPyval's sign: alpha(z) = alpha exp(-b'z)).
_TRUTH = {
    "alpha": 100.0,
    "beta": 2.0,
    "f0": expit(-2.0),
    "alpha_coef_0": 0.4,
    "alpha_coef_1": 0.0,
    "f0_coef_0": 1.0,
    "f0_coef_1": 0.5,
}


def _simulate(seed, n, end=(80.0, 300.0)):
    """Units with a binary ``z`` and a continuous ``a``: dead on arrival
    with probability ``expit(-2 + z + 0.5 a)``; a unit that works has a
    Weibull life, shape 2, scale ``100 exp(-0.4 z)``, censored at a
    uniform time in ``end``."""
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, n)
    a = rng.normal(size=n)
    doa = rng.random(n) < expit(-2.0 + 1.0 * z + 0.5 * a)
    life = 100.0 * np.exp(-0.4 * z) * rng.weibull(2.0, n)
    stop = rng.uniform(*end, n)
    x = np.where(doa, 0.0, np.minimum(life, stop))
    c = np.where(doa, 0, (life > stop).astype(int))
    return x, c, np.column_stack([z, a])


@pytest.fixture(scope="module")
def fitted():
    x, c, Z = _simulate(4, 2000)
    return sp.WeibullZeroInflated.fit(x, Z, c=c), x, c, Z


def _inspected(seed, n):
    """:func:`_simulate`'s units, a third of them inspected only: a unit
    found failed at its first inspection (at 30) is left censored there
    (perhaps dead on arrival), a later failure lies in a 30-wide
    interval, and a quarter of those found failed at the first
    inspection are recorded as an interval [0, 30]."""
    x, c, Z = _simulate(seed, n)
    rng = np.random.default_rng(seed + 1)
    inspected = rng.random(n) < 1 / 3
    failed = c == 0
    first = inspected & failed & (x < 30.0)
    as_interval = first & (rng.random(n) < 0.25)
    later = inspected & failed & (x >= 30.0)
    lo = np.where(later, np.floor(x / 30.0) * 30.0, x)
    hi = np.where(later, lo + 30.0, x)
    lo = np.where(first, np.where(as_interval, 0.0, 30.0), lo)
    hi = np.where(first, 30.0, hi)
    c = np.where(first & ~as_interval, -1, c)
    c = np.where(later | as_interval, 2, c)
    return np.column_stack([lo, hi]), c, Z


def _hand_neg_ll(theta, xx, c, Z, tl=None, link="logit"):
    """The zero-inflated likelihood written out with scipy, for a Weibull
    latency with covariates on its scale and on f0, at ``theta =
    (log alpha, log beta, g(f0), alpha coefficients, f0 coefficients)``.
    """
    p = Z.shape[1]
    alpha = np.exp(theta[0] - Z @ theta[3 : 3 + p])
    beta = np.exp(theta[1])
    eta = theta[2] + Z @ theta[3 + p :]
    f0 = {"logit": expit, "probit": stats.norm.cdf}[link](eta)
    dist = stats.weibull_min(beta, scale=alpha)
    lo, hi = xx[:, 0], xx[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cases = {
            "doa": np.log(f0),
            "fail": np.log1p(-f0) + dist.logpdf(lo),
            "right": np.log1p(-f0) + dist.logsf(lo),
            "by": np.log(f0 + (1 - f0) * dist.cdf(hi)),
            "window": np.log1p(-f0) + np.log(dist.cdf(hi) - dist.cdf(lo)),
        }
    ll = np.select(
        [
            (c == 0) & (lo == 0),
            c == 0,
            c == 1,
            (c == -1) | ((c == 2) & (lo == 0)),
            c == 2,
        ],
        [
            cases["doa"],
            cases["fail"],
            cases["right"],
            cases["by"],
            cases["window"],
        ],
    )
    if tl is not None:
        late = tl > 0
        ll = ll - np.where(
            late, np.log1p(-f0) + dist.logsf(np.where(late, tl, 1.0)), 0.0
        )
    return -np.sum(ll)


def _hand_fit(xx, c, Z, tl=None, link="logit", start=None):
    p = Z.shape[1]
    theta0 = (
        np.r_[np.log(100.0), np.log(2.0), -2.0, np.zeros(2 * p)]
        if start is None
        else start
    )
    res = minimize(
        _hand_neg_ll,
        theta0,
        args=(xx, c, Z, tl, link),
        method="BFGS",
        options={"gtol": 1e-8},
    )
    res = minimize(
        _hand_neg_ll,
        res.x,
        args=(xx, c, Z, tl, link),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 40000},
    )
    return res


def _as_theta(model):
    """The model's parameters on the scale of :func:`_hand_neg_ll`."""
    params = np.asarray(model.params, dtype=float)
    return np.r_[
        np.log(params[0]),
        np.log(params[1]),
        logit(params[2]),
        params[3:],
    ]


# -- reductions ---------------------------------------------------------------


@pytest.mark.parametrize(
    "latency", [sp.Weibull, sp.LogNormal, sp.LogLogistic, sp.Gamma]
)
@pytest.mark.parametrize("method", ["two-stage", "joint"])
def test_no_covariates_is_the_univariate_zero_inflated_fit(latency, method):
    x, c, Z = _simulate(23, 600)
    univariate = latency.fit(x, c=c, zi=True)
    model = sp.ZeroInflated(latency).fit(
        x, Z, c=c, covariates={}, method=method
    )
    assert model.fit_route == method
    assert model.parameter_names == [*latency.parameter_names, "f0"]
    assert_allclose(model.neg_ll(), univariate.neg_ll(), rtol=1e-9)
    assert_allclose(model.params[:-1], univariate.params, rtol=1e-4)
    assert_allclose(model.params[-1], univariate.f0, rtol=1e-4)
    t = [0.0, 20.0, 80.0, 250.0]
    assert_allclose(model.sf(t, [0, 0]), univariate.sf(t), rtol=1e-4)
    assert_allclose(model.ff(0.0, [0, 0]), univariate.f0, rtol=1e-4)
    at = model.distribution_at([0, 0])
    assert_allclose(at.sf(t), model.sf(t, [0, 0]), rtol=1e-12)


def test_no_covariates_with_left_censoring_is_the_univariate_fit():
    # The joint route; an interval [0, t] holds the jump at 0, so it is
    # the univariate fit's left censoring at t.
    xx, c, Z = _inspected(31, 900)
    model = sp.WeibullZeroInflated.fit(xx, Z, c=c, covariates={})
    assert model.fit_route == "joint"
    from_zero = (c == 2) & (xx[:, 0] == 0)
    c_uni = np.where(from_zero, -1, c)
    x_uni = np.where(from_zero[:, None], xx[:, 1:], xx)
    univariate = sp.Weibull.fit(x_uni, c=c_uni, zi=True)
    assert_allclose(model.neg_ll(), univariate.neg_ll(), rtol=1e-9)
    assert_allclose(model.params[:2], univariate.params, rtol=1e-4)
    assert_allclose(model.params[2], univariate.f0, rtol=1e-4)


# -- the two routes -----------------------------------------------------------


def test_two_stages_are_the_joint_maximum(fitted):
    model, x, c, Z = fitted
    assert model.fit_route == "two-stage"
    assert model.maximum == "verified"
    joint = sp.WeibullZeroInflated.fit(x, Z, c=c, method="joint")
    assert joint.fit_route == "joint"
    assert joint.maximum == "verified"
    assert_allclose(model.neg_ll(), joint.neg_ll(), rtol=1e-11)
    assert_allclose(model.params, joint.params, rtol=1e-4, atol=1e-5)
    se = np.sqrt(np.diag(model.covariance()))
    assert_allclose(se, np.sqrt(np.diag(joint.covariance())), rtol=1e-3)
    # No parameter is shared: the covariance is block diagonal
    names = model.parameter_names
    incidence = np.array([name.startswith("f0") for name in names])
    cov = model.covariance()
    assert np.all(cov[np.ix_(incidence, ~incidence)] == 0)


def test_two_stages_are_a_binary_and_a_latency_regression(fitted):
    model, x, c, Z = fitted
    # Stage 1: logistic regression of "dead on arrival" on every row
    first = _incidence_fit((x == 0).astype(float), Z, np.ones(len(x)), "logit")
    assert_allclose(model.params[[2, 5, 6]], first, rtol=1e-6)
    # Stage 2: the latency's regression on the units that worked
    worked = x > 0
    second = sp.WeibullDistributional.fit(
        x[worked], Z[worked], c=c[worked], covariates={"alpha": [0, 1]}
    )
    assert_allclose(model.params[[0, 1, 3, 4]], second.params, rtol=1e-6)
    # The incidence's covariance is the binary regression's
    eta = logit(first[0]) + Z @ first[1:]
    pr = expit(eta)
    X = np.column_stack([np.ones(len(x)), Z])
    info = X.T @ (X * (pr * (1 - pr))[:, None])
    cov = np.linalg.inv(info)
    se = np.sqrt(np.diag(model.covariance()))
    assert_allclose(se[5:], np.sqrt(np.diag(cov))[1:], rtol=1e-4)


def test_two_stages_with_center_and_weights():
    x, c, Z = _simulate(5, 800)
    plain = sp.WeibullZeroInflated.fit(x, Z, c=c)
    centred = sp.WeibullZeroInflated.fit(x, Z, c=c, center=True)
    assert centred.fit_route == "two-stage"
    assert_allclose(centred.neg_ll(), plain.neg_ll(), rtol=1e-11)
    row = [1.0, 0.3]
    assert_allclose(centred.sf(50.0, row), plain.sf(50.0, row), rtol=1e-6)
    weighted = sp.WeibullZeroInflated.fit(x, Z, c=c, w=np.full(800, 2.0))
    assert weighted.fit_route == "two-stage"
    assert_allclose(weighted.params, plain.params, rtol=1e-5)


def test_joint_route_matches_the_likelihood_by_hand():
    xx, c, Z = _inspected(32, 1200)
    assert np.any(c == -1) and np.any((c == 2) & (xx[:, 0] == 0))
    model = sp.WeibullZeroInflated.fit(xx, Z, c=c)
    assert model.fit_route == "joint"
    assert model.maximum == "verified"
    with pytest.raises(ValueError, match="method='two-stage' needs every"):
        sp.WeibullZeroInflated.fit(xx, Z, c=c, method="two-stage")
    theta = _as_theta(model)
    # the same likelihood ...
    assert_allclose(model.neg_ll(), _hand_neg_ll(theta, xx, c, Z), rtol=1e-10)
    # ... and the same maximum, found by scipy from the truth
    res = _hand_fit(xx, c, Z)
    assert res.fun >= model.neg_ll() - 1e-7
    assert_allclose(theta, res.x, rtol=2e-4, atol=2e-4)


def test_late_entry_says_nothing_of_f0():
    x, c, Z = _simulate(33, 1500)
    rng = np.random.default_rng(34)
    # Some units enter observation at an age, if still working then
    entry = np.where(rng.random(1500) < 0.3, rng.uniform(5, 60, 1500), 0.0)
    keep = (x > entry) | (entry == 0)
    x, c, Z, entry = x[keep], c[keep], Z[keep], entry[keep]
    model = sp.WeibullZeroInflated.fit(x, Z, c=c, tl=entry)
    assert model.fit_route == "two-stage"
    assert model.maximum == "verified"
    # Stage 1 is the binary regression of the units that entered at 0
    at_zero = entry == 0
    first = _incidence_fit(
        (x[at_zero] == 0).astype(float),
        Z[at_zero],
        np.ones(at_zero.sum()),
        "logit",
    )
    assert_allclose(model.params[[2, 5, 6]], first, rtol=1e-6)
    # and the whole matches the likelihood by hand
    xx = np.column_stack([x, x])
    res = _hand_fit(xx, c, Z, tl=entry, start=_as_theta(model))
    assert_allclose(
        model.neg_ll(), _hand_neg_ll(_as_theta(model), xx, c, Z, entry)
    )
    assert res.fun >= model.neg_ll() - 1e-7
    assert_allclose(_as_theta(model), res.x, rtol=2e-4, atol=2e-4)
    # a truncation at 0 is no truncation
    zeros = sp.WeibullZeroInflated.fit(
        x[at_zero], Z[at_zero], c=c[at_zero], tl=np.zeros(at_zero.sum())
    )
    none = sp.WeibullZeroInflated.fit(x[at_zero], Z[at_zero], c=c[at_zero])
    assert_allclose(zeros.params, none.params, rtol=1e-10)


def test_right_truncation_without_late_entry_is_joint():
    # Units seen only if they failed by 150: the window holds the jump
    x, c, Z = _simulate(35, 1200)
    seen = (c == 0) & (x <= 150.0)
    model = sp.WeibullZeroInflated.fit(
        x[seen], Z[seen], tr=np.full(seen.sum(), 150.0)
    )
    assert model.fit_route == "joint"
    assert np.isfinite(model.neg_ll())


# -- recovery and the model's functions ---------------------------------------


def test_recovers_both_parts(fitted):
    model, *_ = fitted
    assert model.parameter_names == list(_TRUTH)
    for name, value in _TRUTH.items():
        lower, upper = model.param_cb(name)
        assert lower < value < upper, (name, lower, value, upper)


def test_sign_convention(fitted):
    model, *_ = fitted
    # a positive f0 coefficient: more units dead on arrival
    assert model.params[5] > 0
    assert model.doa_probability([1, 0]) > model.doa_probability([0, 0])
    assert_allclose(
        logit(model.doa_probability([1, 0]))
        - logit(model.doa_probability([0, 0])),
        model.params[5],
    )
    # a positive scale coefficient: a shorter life among those that work
    assert model.params[3] > 0
    assert model.qf(0.5, [1, 0], susceptible=True) < model.qf(
        0.5, [0, 0], susceptible=True
    )
    assert model.coefficient_signs == {"alpha": -1.0, "beta": 1.0, "f0": 1.0}


def test_population_and_latency_functions(fitted):
    model, *_ = fitted
    row = [1.0, 0.3]
    f0 = model.doa_probability(row)
    t = np.array([10.0, 50.0, 120.0])
    Su = model.sf(t, row, susceptible=True)
    assert_allclose(model.sf(t, row), (1 - f0) * Su, rtol=1e-12)
    assert_allclose(model.ff(t, row), f0 + (1 - f0) * (1 - Su), rtol=1e-12)
    assert_allclose(model.ff(0.0, row), f0, rtol=1e-12)
    assert_allclose(model.sf(0.0, row), 1 - f0, rtol=1e-12)
    assert_allclose(model.sf(-1.0, row), 1.0)
    assert_allclose(model.ff(0.0, row, susceptible=True), 0.0)
    fu = model.df(t, row, susceptible=True)
    assert_allclose(model.df(t, row), (1 - f0) * fu, rtol=1e-12)
    assert_allclose(model.hf(t, row), fu / Su, rtol=1e-12)
    assert_allclose(model.Hf(t, row), -np.log((1 - f0) * Su), rtol=1e-12)
    alpha, beta, _ = model.parameters_at([row])[0]
    assert_allclose(Su, np.exp(-((t / alpha) ** beta)), rtol=1e-12)
    # quantiles: 0 up to f0, then the latency's
    q = model.qf([0.5 * f0, f0, 0.5], row)
    assert q[0] == 0 and q[1] == 0
    assert_allclose(model.ff(q[2], row), 0.5, rtol=1e-9)
    qu = model.qf([0.1, 0.5], row, susceptible=True)
    assert_allclose(model.ff(qu, row, susceptible=True), [0.1, 0.5])
    from scipy.special import gamma

    mean_u = alpha * gamma(1 + 1 / beta)
    assert_allclose(model.mean(row, susceptible=True), mean_u)
    assert_allclose(model.mean(row), (1 - f0) * mean_u)
    draws, at = model.random(4000, [row], random_state=3)
    assert at.shape == (4000, 2)
    dead = draws == 0
    assert abs(dead.mean() - f0) < 3 * np.sqrt(f0 * (1 - f0) / 4000)
    assert_allclose(np.median(draws[~dead]), qu[1], rtol=0.05)


def test_bounds_and_serialisation(fitted):
    model, *_ = fitted
    row = [1.0, 0.3]
    t = [20.0, 90.0, 200.0]
    wald = model.cb(t, row)
    assert np.all(wald[:, 0] < model.sf(t, row))
    assert np.all(model.sf(t, row) < wald[:, 1])
    cb = model.parameter_cb("f0", [row])
    assert cb[0, 0] < model.doa_probability(row) < cb[0, 1]
    lower, upper = model.param_cb("f0_coef_0", method="lr")
    assert lower < model.params[5] < upper

    restored = sp.from_dict(model.to_dict())
    assert type(restored).__name__ == "ZeroInflatedModel"
    assert restored.latency.name == "Weibull"
    assert_allclose(restored.sf(t, row), model.sf(t, row), rtol=1e-12)
    assert_allclose(restored.ff(0.0, row), model.ff(0.0, row), rtol=1e-12)
    assert_allclose(
        restored.sf(t, row, susceptible=True),
        model.sf(t, row, susceptible=True),
        rtol=1e-12,
    )
    assert_allclose(restored.cb(t, row), model.cb(t, row), rtol=1e-8)
    import pickle

    again = pickle.loads(pickle.dumps(model))
    assert_allclose(again.sf(t, row), model.sf(t, row), rtol=1e-12)
    text = repr(model)
    assert "Zero-Inflated Regression SurPyval Model" in text
    assert "logit(f0) ~ 1 + Z0 + Z1" in text


def test_links_formulas_and_named_columns():
    x, c, Z = _simulate(25, 1500)
    df = pd.DataFrame({"x": x, "c": c, "lot": Z[:, 0], "load": Z[:, 1]})
    by_formula = sp.WeibullZeroInflated.fit_from_df(
        df,
        x_col="x",
        c_col="c",
        formulas={"f0": "lot + load", "alpha": "lot"},
    )
    assert by_formula.fit_route == "two-stage"
    assert by_formula.parameter_names == [
        "alpha",
        "beta",
        "f0",
        "alpha_lot",
        "f0_lot",
        "f0_load",
    ]
    by_columns = sp.WeibullZeroInflated.fit(
        x, Z, c=c, covariates={"f0": [0, 1], "alpha": [0]}
    )
    assert_allclose(by_formula.neg_ll(), by_columns.neg_ll(), rtol=1e-9)
    assert_allclose(by_formula.params, by_columns.params, rtol=1e-5)
    row = pd.DataFrame({"lot": [1.0], "load": [0.3]})
    assert_allclose(
        by_formula.doa_probability(row),
        by_columns.doa_probability([1.0, 0.3]),
        rtol=1e-5,
    )
    named = sp.WeibullZeroInflated.fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["lot", "load"]
    )
    assert named.parameter_names[3:] == [
        "alpha_lot",
        "alpha_load",
        "f0_lot",
        "f0_load",
    ]

    for link in ("probit", "cloglog"):
        model = sp.ZeroInflated(sp.Weibull, incidence=link).fit(x, Z, c=c)
        joint = sp.ZeroInflated(sp.Weibull, incidence=link).fit(
            x, Z, c=c, method="joint"
        )
        assert model.incidence_link == link
        assert model.maximum == "verified"
        assert_allclose(model.neg_ll(), joint.neg_ll(), rtol=1e-10)
        # another link, nearly the same fitted probabilities
        assert_allclose(
            model.doa_probability([[0, 0], [1, 0]]),
            by_columns.doa_probability([[0, 0], [1, 0]]),
            atol=0.02,
        )
    for latency in (sp.LogNormal, sp.LogLogistic, sp.Gamma, sp.Exponential):
        model = sp.ZeroInflated(latency).fit(x, Z, c=c)
        assert model.latency is latency
        assert model.fit_route == "two-stage"
        assert 0 < model.doa_probability([0, 0]) < 1


def test_probit_joint_matches_the_likelihood_by_hand():
    xx, c, Z = _inspected(36, 900)
    model = sp.ZeroInflated(sp.Weibull, incidence="probit").fit(xx, Z, c=c)
    assert model.fit_route == "joint"
    params = np.asarray(model.params, dtype=float)
    theta = np.r_[np.log(params[:2]), stats.norm.ppf(params[2]), params[3:]]
    assert_allclose(
        model.neg_ll(),
        _hand_neg_ll(theta, xx, c, Z, link="probit"),
        rtol=1e-10,
    )


def test_tests_bootstrap_and_penalised_fit():
    x, c, Z = _simulate(29, 900)
    model = sp.WeibullZeroInflated.fit(x, Z, c=c)
    # the global test's null is the zero-inflated fit without covariates
    res = model.likelihood_ratio_test()
    null = sp.Weibull.fit(x, c=c, zi=True)
    assert res["df"] == 4
    assert_allclose(res["log_likelihood_null"], -null.neg_ll(), rtol=1e-9)
    part = model.likelihood_ratio_test("f0")
    held = sp.WeibullZeroInflated.fit(x, Z, c=c, covariates={"alpha": [0, 1]})
    assert part["df"] == 2
    assert_allclose(part["log_likelihood_null"], -held.neg_ll(), rtol=1e-8)
    # the parametric bootstrap draws units dead on arrival as 0
    row = [1.0, 0.0]
    wald = model.cb([50.0, 150.0], row)
    boot = model.cb(
        [50.0, 150.0], row, method="bootstrap", n_boot=40, random_state=1
    )
    assert np.all(np.isfinite(boot))
    assert_allclose(boot, wald, atol=0.04)
    penalised = sp.WeibullZeroInflated.fit_penalised(x, Z, c=c, penalizer=0.05)
    assert penalised.fit_route == "joint"
    assert np.all(
        np.abs(penalised.params[3:]) <= np.abs(model.params[3:]) + 1e-6
    )


# -- refusals -----------------------------------------------------------------


def test_refusals():
    with pytest.raises(ValueError, match="not available with a Normal"):
        sp.ZeroInflated(sp.Normal)
    with pytest.raises(ValueError, match="takes the latency's distribution"):
        sp.ZeroInflated("Weibull")
    with pytest.raises(ValueError, match="not a link of a probability"):
        sp.ZeroInflated(sp.Weibull, incidence="logistic")
    with pytest.raises(ValueError, match="give incidence only"):
        sp.ZeroInflated(sp.Weibull, links={"f0": "probit"})
    with pytest.raises(ValueError, match="built by ZeroInflated"):
        sp.ZeroInflatedFitter(sp.Weibull)
    x, c, Z = _simulate(26, 300)
    with pytest.raises(ValueError, match="method = 'exact'"):
        sp.WeibullZeroInflated.fit(x, Z, c=c, method="exact")
    with pytest.raises(ValueError, match="No unit failed at time 0"):
        sp.WeibullZeroInflated.fit(x[x > 0], Z[x > 0], c=c[x > 0])
    with pytest.raises(ValueError, match="Every unit that could have been"):
        sp.WeibullZeroInflated.fit(
            np.r_[0.0, 0.0, 5.0], [0.0, 1.0, 1.0], tl=[0, 0, 2]
        )
    with pytest.raises(ValueError, match="time below 0"):
        sp.WeibullZeroInflated.fit([-1.0, 0.0, 3.0], [0.0, 1.0, 1.0])
    assert repr(sp.WeibullZeroInflated) == (
        "WeibullZeroInflated: zero-inflated regression fitter "
        "(Weibull latency)"
    )
