"""Tests for the shared-frailty proportional-hazards model."""

import json
import warnings
from unittest import mock

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

import surpyval as surv
import surpyval as sp
from surpyval import (
    ExponentialFrailty,
    Frailty,
    Weibull,
    WeibullFrailty,
    WeibullPH,
)
from surpyval.tests._helpers import weibull_ph_data
from surpyval.univariate.regression.frailty import frailty_fitter
from surpyval.univariate.regression.frailty.frailty_model import (
    FrailtyModel,
)


def _sim(seed=7, G=80, per=6, alpha=12.0, shape=1.8, beta=0.9, theta=0.6):
    rng = np.random.default_rng(seed)
    groups, x, c, Z = [], [], [], []
    for g in range(G):
        u = rng.gamma(1.0 / theta, theta)
        for _ in range(per):
            z = rng.normal(0, 1)
            eta = np.exp(beta * z) * u
            t = alpha * (-np.log(rng.uniform()) / eta) ** (1.0 / shape)
            obs = min(t, 30.0)
            groups.append(g)
            x.append(obs)
            c.append(0 if t <= 30.0 else 1)
            Z.append(z)
    return (
        np.array(x),
        np.array(c),
        np.array(Z).reshape(-1, 1),
        np.array(groups),
    )


def test_marginal_likelihood_matches_numerical_integration():
    # Gold standard: the closed-form gamma-frailty group likelihood must equal
    # brute-force integration of the conditional likelihood over the frailty.
    rng = np.random.default_rng(0)
    fitter = WeibullFrailty
    dp = np.array([9.0, 1.8])
    theta = 0.7
    for _ in range(5):
        m = rng.integers(2, 6)
        x = np.abs(rng.weibull(2.0, m)) * 10 + 0.5
        c = rng.integers(0, 2, m)
        c[0] = 0  # ensure an event
        eta = np.exp(rng.normal(0, 0.5, m))

        H0 = Weibull.Hf(x, *dp)
        h0 = Weibull.hf(x, *dp)
        event = c == 0
        D = int(event.sum())
        H = float(np.sum(eta * H0))
        it = 1.0 / theta
        closed = (
            np.sum(np.log(h0[event]) + np.log(eta[event]))
            - it * np.log(theta)
            - gammaln(it)
            + gammaln(D + it)
            - (D + it) * np.log(H + it)
        )

        prod_event = np.prod(h0[event] * eta[event])
        logconst = it * np.log(theta) + gammaln(it)

        def integrand(u):
            cond = (u**D) * prod_event * np.exp(-u * H)
            fu = np.exp((it - 1) * np.log(u) - u / theta - logconst)
            return cond * fu

        val, _ = quad(integrand, 0, np.inf, limit=200)
        assert closed == pytest.approx(np.log(val), abs=1e-6)
    assert fitter is WeibullFrailty  # keep the fixture referenced


def test_recovers_known_parameters():
    x, c, Z, groups = _sim()
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    assert m.dist_params[0] == pytest.approx(12.0, rel=0.2)
    assert m.dist_params[1] == pytest.approx(1.8, rel=0.2)
    assert m.beta[0] == pytest.approx(0.9, abs=0.2)
    assert m.theta == pytest.approx(0.6, abs=0.2)
    assert m.n_groups == 80
    # theta CI excludes 0 for this clearly-heterogeneous data
    lo, hi = m.param_cb("theta")
    assert 0 < lo < m.theta < hi


def test_marginal_sf_is_gamma_laplace_transform():
    x, c, Z, groups = _sim(seed=2)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    t = np.array([4.0, 9.0, 15.0])
    z = np.array([0.3])
    eta = np.exp(z @ m.beta)
    H0 = Weibull.Hf(t, *m.dist_params)
    expected = (1.0 + m.theta * eta * H0) ** (-1.0 / m.theta)
    assert np.allclose(m.sf(t, z), expected)


def test_posterior_frailty_mean_is_about_one():
    x, c, Z, groups = _sim(seed=3)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    vals = np.array(list(m.frailties.values()))
    assert vals.mean() == pytest.approx(1.0, abs=0.1)


def test_conditional_orders_by_frailty():
    x, c, Z, groups = _sim(seed=4)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    t = np.array([8.0])
    z = np.array([0.0])
    # a higher frailty means a higher hazard, hence lower survival
    assert m.sf(t, z, frailty=2.0) < m.sf(t, z, frailty=0.5)


def test_no_covariate_frailty():
    rng = np.random.default_rng(5)
    G, per, theta = 60, 8, 0.5
    groups, x, c = [], [], []
    for g in range(G):
        u = rng.gamma(1.0 / theta, theta)
        for _ in range(per):
            t = 10.0 * (-np.log(rng.uniform()) / u) ** (1.0 / 2.0)
            groups.append(g)
            x.append(min(t, 25.0))
            c.append(0 if t <= 25.0 else 1)
    m = WeibullFrailty.fit(
        x=np.array(x), c=np.array(c), groups=np.array(groups)
    )
    assert m.beta.size == 0
    assert m.theta == pytest.approx(0.5, abs=0.2)
    assert m.sf(np.array([8.0])).shape == (1,)


def test_fit_from_df_formula_round_trips():
    rng = np.random.default_rng(6)
    n = 400
    import pandas as pd

    df = pd.DataFrame(
        {
            "t": rng.weibull(1.8, n) * 10 + 0.5,
            "c": rng.integers(0, 2, n),
            "load": rng.uniform(1, 5, n),
            "site": rng.choice(["A", "B", "C"], n),
            "unit": rng.integers(0, 50, n),
        }
    )
    m = WeibullFrailty.fit_from_df(
        df, x_col="t", c_col="c", group_col="unit", formula="load + site"
    )
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    assert type(restored).__name__ == "FrailtyModel"
    raw = pd.DataFrame({"load": [3.0, 2.0], "site": ["B", "A"]})
    t = np.array([5.0, 10.0])
    assert np.allclose(m.sf(t, raw), restored.sf(t, raw))
    # a known group's conditional prediction round-trips too
    g = m.group_labels[0]
    assert np.allclose(
        m.sf(t, raw.iloc[[0]], group=g),
        restored.sf(t, raw.iloc[[0]], group=g),
    )


def test_serialisation_round_trip_arrays():
    x, c, Z, groups = _sim(seed=8, G=40, per=5)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    t = np.array([5.0, 12.0])
    z = np.array([0.4])
    assert np.allclose(m.sf(t, z), restored.sf(t, z))
    assert restored.theta == pytest.approx(m.theta)
    assert restored.frailties == m.frailties


def test_guard_single_group():
    x, c, Z, groups = _sim(seed=9, G=1, per=20)
    with pytest.raises(ValueError, match="two groups"):
        WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)


def test_guard_unsupported_censoring():
    with pytest.raises(ValueError, match="right-censored"):
        WeibullFrailty.fit(
            x=np.array([1.0, 2.0, 3.0, 4.0]),
            c=np.array([-1, 0, 0, 1]),
            groups=np.array([0, 0, 1, 1]),
        )


def test_guard_unknown_family():
    # A ValueError naming the choices (principle 2); "lognormal" is a
    # family since #343 (it raised NotImplementedError before).
    with pytest.raises(ValueError, match="'gamma' or 'lognormal'"):
        Frailty(Weibull, family="weibull")


def test_exponential_frailty_available():
    x, c, Z, groups = _sim(seed=10, G=50, per=5)
    m = ExponentialFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    assert m.dist.name == "Exponential"
    assert m.theta > 0


def test_theta_zero_gives_ph_limit_not_nan():
    # theta -> 0 is the no-frailty PH limit: Hf = eta * H0. Dividing by a
    # zero theta (frailty-free data, or a restored model) gave NaN (#262).
    m = FrailtyModel.__new__(FrailtyModel)
    m.dist = Weibull
    m.dist_params = np.array([10.0, 3.0])
    m.beta = np.array([])
    m.theta = 0.0
    m.family = "gamma"
    m._frailties = {}
    m.feature_names = None
    m.formula = None
    m._model_spec = None

    x = np.array([2.0, 5.0, 10.0])
    assert np.allclose(m.Hf(x), Weibull.Hf(x, 10.0, 3.0))
    assert np.all(np.isfinite(m.sf(x)))


def test_no_frailty_in_the_data_reduces_to_the_ph_fit():
    # With no shared frailty the variance goes to its boundary, theta -> 0.
    # The marginal likelihood written directly cancels catastrophically
    # there (terms of size theta^-1 log theta^-1), which let the optimiser
    # chase round-off to a nonsense optimum or divide by an underflowed
    # theta. It now tends to the proportional-hazards likelihood.
    from surpyval import WeibullPH
    from surpyval.univariate.regression.frailty import WeibullFrailty

    rng = np.random.default_rng(1)
    n = 300
    Z = rng.normal(size=(n, 1))
    groups = rng.integers(0, 30, n)
    x = 10.0 * rng.weibull(1.5, n) * np.exp(-0.8 * Z[:, 0] / 1.5)
    c = (x > 12).astype(int)
    x = np.minimum(x, 12.0)

    ph = WeibullPH.fit(x=x, Z=Z, c=c)
    fr = WeibullFrailty.fit(x, Z=Z, c=c, groups=groups)
    assert fr.theta < 1e-3
    assert fr._neg_ll == pytest.approx(ph._neg_ll, abs=1e-4)
    assert fr.beta[0] == pytest.approx(ph.params[-1], rel=1e-3)
    assert np.allclose(fr.dist_params, ph.params[:2], rtol=1e-3)
    assert np.all(np.isfinite(list(fr.frailties.values())))


def test_stable_group_likelihood_matches_the_direct_formula():
    from scipy.special import gammaln

    from surpyval.univariate.regression.frailty.frailty_fitter import (
        _group_frailty_ll,
    )

    D = np.array([0.0, 1.0, 3.0, 7.0, 2.5])
    H = np.array([0.2, 1.3, 2.9, 6.0, 1.7])
    for theta in (5.0, 0.7, 0.05, 1e-3):
        it = 1.0 / theta
        direct = (
            -it * np.log(theta)
            - gammaln(it)
            + gammaln(D + it)
            - (D + it) * np.log(H + it)
        )
        assert np.allclose(_group_frailty_ll(D, H, theta), direct, atol=1e-9)
    # and the no-frailty limit is -H
    assert np.allclose(_group_frailty_ll(D, H, 1e-14), -H, atol=1e-10)
    assert np.allclose(_group_frailty_ll(D, H, 0.0), -H)


# ---------------------------------------------------------------------------
# Frailty fits take the gradient ladder first (#515):
# ``optimise_ph`` on the likelihood's autograd gradient, kept
# when it is a verified optimum.
# ---------------------------------------------------------------------------


def _survey_data(seed):
    rng = np.random.default_rng(seed)
    n = 400
    Z = rng.normal(size=(n, 3))
    Z[:, 0] = rng.integers(0, 2, n)
    t = 10 * rng.weibull(1.5, n) * np.exp(-(Z @ [0.5, 0.1, -0.3]) / 1.5)
    ct = rng.uniform(0, 1.2 * np.quantile(t, 0.8), n)
    c = (ct < t).astype(int)
    x = np.maximum(np.ceil(np.minimum(t, ct) * 10) / 10, 0.1)
    groups = np.random.default_rng(seed).integers(0, 40, n)
    return dict(x=x, Z=Z, c=c, groups=groups)


def _shared_frailty_data():
    # The docstring example: thirty groups of six, gamma frailty of
    # variance 0.5.
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 6)
    u = rng.gamma(2.0, 0.5, 30)[groups]
    Z = rng.binomial(1, 0.5, (180, 1))
    H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
    return dict(x=10 * H**0.5, Z=Z, groups=groups)


# The negative log-likelihood the Nelder-Mead-then-BFGS ladder reached
# (before #515). On the survey data it stopped with theta at 1e-10 and
# 1e-21 for the Weibull baseline, 0.10 nats short of the maximum at theta
# 0.02; on the others it found the maximum.
_OLD_NEG_LL = {
    ("shared", "WeibullFrailty"): 553.2587426928156,
    ("shared", "GammaFrailty"): 555.6163844533303,
    ("shared", "LogNormalFrailty"): 567.7838014631002,
    ("survey0", "WeibullFrailty"): 607.937511288359,
    ("survey0", "GammaFrailty"): 608.6993140296784,
    ("survey1", "WeibullFrailty"): 570.8795379993384,
    ("survey1", "LogNormalFrailty"): 585.9436498029434,
    ("survey2", "ExponentialFrailty"): 643.5745250742405,
}


def _data(case):
    if case == "shared":
        return _shared_frailty_data()
    return _survey_data(int(case[-1]))


@pytest.mark.parametrize("case,name", sorted(_OLD_NEG_LL))
def test_reaches_at_least_the_old_ladders_optimum(case, name):
    model = getattr(sp, name).fit(**_data(case))
    old = _OLD_NEG_LL[(case, name)]
    assert model._neg_ll <= old + 1e-6


@pytest.mark.parametrize("name", ["WeibullFrailty", "GammaFrailty"])
def test_no_nelder_mead_when_the_gradient_ladder_converges(name):
    methods = []
    real = frailty_fitter.minimize

    def recording(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return real(*args, **kwargs)

    with mock.patch.object(frailty_fitter, "minimize", recording):
        model = getattr(sp, name).fit(**_shared_frailty_data())
    assert np.isfinite(model._neg_ll)
    assert "Nelder-Mead" not in methods, methods


def test_a_variance_heading_for_zero_is_carried_to_its_limit():
    # BFGS in log(theta) stops at theta ~ 1e-7 on data with no frailty,
    # with the likelihood still rising towards theta = 0; the fit carries
    # it on to where the likelihood no longer changes.
    rng = np.random.default_rng(4)
    x = rng.weibull(1.5, 300) * 10
    Z = rng.normal(size=(300, 2))
    groups = rng.integers(0, 30, 300)
    model = sp.GammaFrailty.fit(x, Z=Z, groups=groups)
    ph = sp.GammaPH.fit(x, Z)
    assert model.theta < 1e-12
    assert model._neg_ll == pytest.approx(ph._neg_ll, abs=1e-6)


# ---------------------------------------------------------------------------
# ``param_cb`` with theta at its boundary; information criteria
# comparable with PH.
# ---------------------------------------------------------------------------


def _frailty_free():
    rng = np.random.default_rng(2)
    z = rng.normal(size=300)
    x = 20 * (-np.log(rng.uniform(size=300)) / np.exp(0.8 * z)) ** (1 / 1.8)
    g = np.repeat(np.arange(30), 10)
    return x, z.reshape(-1, 1), g


def test_frailty_param_cb_theta_at_boundary_is_warning_free():
    x, Z, g = _frailty_free()
    model = WeibullFrailty.fit(x=x, Z=Z, groups=g)
    assert model.theta < 1e-8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lower, upper = model.param_cb("theta")
        model.standard_errors()
    assert lower == 0.0
    assert upper == np.inf


def test_frailty_information_criteria_compare_with_ph():
    x, Z, g = _frailty_free()
    frailty = WeibullFrailty.fit(x=x, Z=Z, groups=g)
    ph = WeibullPH.fit(x=x, Z=Z)
    # theta -> 0 is the PH model, so the likelihoods agree and the frailty
    # model pays for one more parameter.
    assert frailty.neg_ll() == pytest.approx(ph.neg_ll(), abs=1e-6)
    assert frailty.aic() == pytest.approx(ph.aic() + 2, abs=1e-5)
    assert frailty.bic() == pytest.approx(ph.bic() + np.log(300), abs=1e-5)
    restored = type(frailty).from_dict(frailty.to_dict())
    assert restored.aic() == pytest.approx(frailty.aic())
    assert restored.bic() == pytest.approx(frailty.bic())


# ---------------------------------------------------------------------------
# The length of ``groups`` and the ``param_cb`` name are checked.
# ---------------------------------------------------------------------------


def test_frailty_groups_length_and_param_cb_name():
    x, Z = weibull_ph_data()
    groups = np.repeat(np.arange(40), 5)
    with pytest.raises(ValueError, match="'groups' has 199"):
        WeibullFrailty.fit(x, Z=Z, groups=groups[:-1])
    model = WeibullFrailty.fit(x, Z=Z, groups=groups)
    with pytest.raises(ValueError, match="Unknown parameter 'gamma'"):
        model.param_cb("gamma")


# ---------------------------------------------------------------------------
# #388: a missing frailty group.
# ---------------------------------------------------------------------------


def _frailty_data():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(8), 6).astype(float)
    u = rng.gamma(2.0, 0.5, 8)[groups.astype(int)]
    Z = rng.binomial(1, 0.5, (48, 1)).astype(float)
    x = 10 * (rng.exponential(1, 48) / (u * np.exp(0.5 * Z[:, 0]))) ** 0.5
    return x, Z, groups


@pytest.mark.parametrize("missing", [np.nan, None])
def test_frailty_drops_a_row_with_a_missing_group(missing):
    # A NaN label used to be a group of its own (n_groups 9, not 8), and a
    # None label raised TypeError.
    x, Z, groups = _frailty_data()
    labels = list(groups)
    labels[3] = missing
    init = [8.0, 2.0, 0.0, 0.5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.WeibullFrailty.fit(x, Z=Z, groups=labels, init=init)
    dropped = [str(w.message) for w in caught if "Dropped" in str(w.message)]
    assert dropped == ["Dropped 1 of 48 rows with a missing group label."]
    assert model.n_groups == 8
    keep = np.arange(48) != 3
    ref = sp.WeibullFrailty.fit(
        x[keep], Z=Z[keep], groups=groups[keep], init=init
    )
    np.testing.assert_allclose(model.sf(5.0, [1.0]), ref.sf(5.0, [1.0]))


def test_frailty_refuses_every_group_missing():
    x, Z, _ = _frailty_data()
    with pytest.raises(ValueError, match="Every group label is missing"):
        sp.WeibullFrailty.fit(x, Z=Z, groups=[None] * 48)


def test_frailty_predicts_nan_for_a_missing_group():
    x, Z, groups = _frailty_data()
    model = sp.WeibullFrailty.fit(x, Z=Z, groups=groups, init=[8, 2, 0, 0.5])
    assert np.isnan(model.sf([5.0, 6.0], [1.0], group=np.nan)).all()
    assert np.isfinite(model.sf(5.0, [1.0], group=groups[0]))


def test_605_covariance_is_a_method():
    x, c, Z, groups = _sim()
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    cov = m.covariance()
    assert type(cov) is np.ndarray and cov.shape == (4, 4)
    # The attribute's spelling, deprecated in v0.23, is gone
    with pytest.raises(TypeError):
        m.covariance[0]
    # Without one, the call says why
    m._covariance = None
    with pytest.raises(ValueError, match="no parameter covariance"):
        m.covariance()


# -- param_cb(method="lr") (#617) -------------------------------------------
def _gamma_frailty_nll(theta_vec, x, c, Z, groups):
    """The marginal negative log-likelihood of a Weibull gamma-frailty
    model, written out here from its closed form (see
    ``test_marginal_likelihood_matches_numerical_integration``)."""
    alpha, shape, beta, theta = theta_vec
    eta = np.exp(beta * Z[:, 0])
    H0 = (x / alpha) ** shape
    h0 = shape / alpha * (x / alpha) ** (shape - 1.0)
    event = c == 0
    ll = np.sum(np.log(h0[event] * eta[event]))
    it = 1.0 / theta
    for g in np.unique(groups):
        rows = groups == g
        D = event[rows].sum()
        H = np.sum(eta[rows] * H0[rows])
        ll += (
            -it * np.log(theta)
            - gammaln(it)
            + gammaln(D + it)
            - (D + it) * np.log(H + it)
        )
    return -ll


def test_617_frailty_param_cb_lr_is_where_the_profile_deviance_is_chi2():
    from scipy.optimize import minimize
    from scipy.stats import chi2

    x, c, Z, groups = _sim(seed=11, G=30, per=5)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    nll_hat = _gamma_frailty_nll(m.params, x, c, Z, groups)
    assert nll_hat == pytest.approx(m.neg_ll(), rel=1e-10)
    crit = chi2.ppf(0.95, 1)
    for j, name in ((3, "theta"), (2, "coef_0")):
        lo, hi = m.param_cb(name, method="lr")
        assert lo < m.params[j] < hi
        others = [i for i in range(4) if i != j]
        for b in (lo, hi):

            def nll(v, b=b):
                p = np.empty(4)
                p[j] = b
                p[others] = v
                p[0], p[1] = np.exp(p[0]), np.exp(p[1])
                return _gamma_frailty_nll(p, x, c, Z, groups)

            v = m.params[others].copy()
            v[0], v[1] = np.log(v[0]), np.log(v[1])
            res = minimize(
                nll,
                v,
                method="Nelder-Mead",
                options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000},
            )
            dev = 2.0 * (res.fun - nll_hat)
            assert dev == pytest.approx(crit, abs=1e-4)


def test_617_frailty_param_cb_lr_options_and_the_edge_of_theta():
    x, c, Z, groups = _sim(seed=12, G=30, per=5)
    m = WeibullFrailty.fit(x=x, Z=Z, c=c, groups=groups)
    # Wald stays the default.
    assert np.array_equal(m.param_cb("coef_0"), m.param_cb("coef_0", 0.05))
    assert np.array_equal(
        m.param_cb("coef_0"), m.param_cb("coef_0", method="wald")
    )
    two = m.param_cb("theta", alpha_ci=0.2, method="profile")
    upper = m.param_cb("theta", alpha_ci=0.1, bound="upper", method="lr")
    assert upper.shape == (1,) and upper[0] == two[1]
    with pytest.raises(ValueError, match="method"):
        m.param_cb("theta", method="bootstrap")
    with pytest.raises(ValueError, match="Unknown parameter"):
        m.param_cb("gamma", method="lr")
    # A model restored from a dict keeps no data.
    restored = FrailtyModel.from_dict(m.to_dict())
    with pytest.raises(ValueError, match="data"):
        restored.param_cb("theta", method="lr")
    # No frailty in the data: theta's interval reaches its edge, 0.
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0 = WeibullFrailty.fit(
            rng.weibull(1.5, 200) * 10, groups=np.repeat(np.arange(40), 5)
        )
    lo, hi = m0.param_cb("theta", method="lr")
    assert lo == 0.0 and 0.0 < hi < 1.0


def test_617_frailty_param_cb_lr_lognormal_and_aliased():
    x, c, Z, groups = _sim(seed=13, G=30, per=5)
    m = Frailty(Weibull, family="lognormal").fit(x=x, Z=Z, c=c, groups=groups)
    lo, hi = m.param_cb("theta", method="lr")
    assert lo < m.theta < hi
    # A constant column is aliased: no interval, as Wald gives none.
    Z2 = np.column_stack([Z, np.ones(len(x))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = WeibullFrailty.fit(x=x, Z=Z2, c=c, groups=groups)
    assert np.all(np.isnan(m2.param_cb("coef_1", method="lr")))
    lo, hi = m2.param_cb("coef_0", method="lr")
    assert lo < m2.beta[0] < hi
