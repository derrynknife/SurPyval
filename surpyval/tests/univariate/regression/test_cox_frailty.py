"""The shared gamma frailty model with a Cox baseline (#342): CoxFrailty.

References: R survival's ``coxph(... + frailty(id, dist = "gamma"))`` on the
kidney catheter data (McGilchrist and Aisbett 1991), stored by
scripts/reference/reference_r_frailty.R.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.special import gammaln

import surpyval as sp
from surpyval import CoxFrailty, CoxPH
from surpyval.datasets import load_kidney
from surpyval.tests.reference._data import reference


def _kidney(extra=False):
    df = load_kidney()
    cols = [df["age"], (df["sex"] == 2).astype(float)]
    if extra:
        cols.append((df["disease"] == "PKD").astype(float))
    Z = np.column_stack(cols)
    return df["time"].values, 1 - df["status"].values, Z, df["id"].values


def _simulate(seed, G=60, per=5, theta=0.5, beta=0.7):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), per)
    u = rng.gamma(1 / theta, theta, G)[g]
    Z = rng.normal(size=(G * per, 1))
    # Weibull(10, 1.5) baseline: H(t) = (t / 10)^1.5 times the multiplier
    H = rng.exponential(size=G * per) / (np.exp(beta * Z[:, 0]) * u)
    t = 10.0 * H ** (1 / 1.5)
    c = (t > 25.0).astype(int)
    return np.minimum(t, 25.0), c, Z, g


@pytest.fixture(scope="module")
def kidney_fits():
    x, c, Z, g = _kidney()
    return {
        ties: CoxFrailty.fit(x, Z=Z, c=c, groups=g, tie_method=ties)
        for ties in ("efron", "breslow")
    }


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_kidney_matches_r_coxph(kidney_fits, ties):
    ref = reference("r_frailty", "kidney_cox_gamma_" + ties)["values"]
    m = kidney_fits[ties]
    # theta maximises a flat profile: R's optimize and ours agree to 1e-5
    assert m.theta == pytest.approx(ref["theta"], rel=2e-5)
    assert m.log_likelihood == pytest.approx(ref["loglik"], abs=1e-8)
    assert m.log_likelihood_no_frailty == pytest.approx(
        ref["loglik_cox"], abs=1e-8
    )
    np.testing.assert_allclose(m.beta, ref["beta"], rtol=0, atol=2e-6)
    se = dict(zip(m.parameter_names, m.standard_errors()))
    np.testing.assert_allclose(
        [se["coef_0"], se["coef_1"]], ref["se"], rtol=1e-4
    )
    np.testing.assert_allclose(
        np.log(list(m.frailties.values())),
        ref["log_frailties"],
        rtol=0,
        atol=2e-5,
    )


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_kidney_at_rs_theta_is_rs_fit(ties):
    # At R's theta the fit is R's to its own tolerance: EM's fixed point
    # is the maximum of the penalised partial likelihood R maximises, with
    # Efron's ties too.
    ref = reference("r_frailty", "kidney_cox_gamma_" + ties)["values"]
    x, c, Z, g = _kidney()
    m = CoxFrailty.fit(
        x, Z=Z, c=c, groups=g, tie_method=ties, theta=ref["theta"]
    )
    np.testing.assert_allclose(m.beta, ref["beta"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        np.log(list(m.frailties.values())),
        ref["log_frailties"],
        rtol=0,
        atol=1e-7,
    )
    assert m.log_likelihood == pytest.approx(ref["loglik"], abs=1e-9)
    assert np.isnan(m.standard_errors()[-1])  # theta was given


def test_kidney_with_disease_has_no_frailty():
    # R's theta is 5e-9 there: the profile is maximal at theta = 0, and the
    # fit is the Cox fit.
    ref = reference("r_frailty", "kidney_cox_gamma_no_frailty")["values"]
    x, c, Z, g = _kidney(extra=True)
    m = CoxFrailty.fit(x, Z=Z, c=c, groups=g)
    cox = CoxPH.fit(x, Z, c=c)
    assert m.theta == 0.0
    np.testing.assert_allclose(m.beta, cox.beta, rtol=1e-9)
    np.testing.assert_allclose(m.beta, ref["beta"], rtol=0, atol=2e-6)
    assert m.log_likelihood == pytest.approx(ref["loglik_cox"], abs=1e-8)
    assert set(m.frailties.values()) == {1.0}
    t = np.array([10.0, 100.0, 300.0])
    z = [50.0, 1.0, 0.0]
    np.testing.assert_allclose(m.sf(t, z), cox.sf(t, z), rtol=1e-9)


def test_em_fixed_point_maximises_the_penalised_partial_likelihood():
    # Independent of the EM code: the score of R's penalised partial
    # likelihood -- CoxPH's Efron partial likelihood with an indicator per
    # group, minus (1/theta) sum(exp(w) - w) -- is zero at the fit.
    x, c, Z, g = _kidney()
    m = CoxFrailty.fit(x, Z=Z, c=c, groups=g, theta=0.6)
    labels, inv = np.unique(g, return_inverse=True)
    w = np.log([m.frailties[str(k)] for k in labels])
    ind = np.zeros((x.size, labels.size))
    ind[np.arange(x.size), inv] = 1.0
    _, jac = CoxPH.create_efron_ll_jac_hess(
        x.astype(float),
        np.column_stack([Z, ind]),
        c,
        np.ones(x.size),
        np.full(x.size, -np.inf),
    )
    score = -np.asarray(jac(np.r_[m.beta, w])[0])
    score[2:] -= (np.exp(w) - 1.0) / 0.6
    np.testing.assert_allclose(score, 0.0, atol=1e-7)


def test_breslow_i_likelihood_is_the_marginal_likelihood():
    # With Breslow's ties the I-likelihood is the likelihood with the
    # frailties integrated out (here by quadrature) and the baseline at
    # its nonparametric maximum, plus sum(D) - sum(d log d) over the
    # event times.
    x, c, Z, g = _kidney()
    theta = 0.45
    m = CoxFrailty.fit(
        x, Z=Z, c=c, groups=g, tie_method="breslow", theta=theta
    )
    eta = np.exp(Z @ m.beta)
    jump = dict(zip(m.x, m.h0))
    event = c == 0
    loglik = np.sum(np.log([jump[t] for t in x[event]]) + np.log(eta[event]))
    H = m._H0(x)
    for k in np.unique(g):
        i = g == k
        D, A = np.sum(event[i]), np.sum(eta[i] * H[i])
        a = 1 / theta

        def density(u):
            return np.exp(
                D * np.log(u)
                - u * A
                + (a - 1) * np.log(u)
                - u * a
                + a * np.log(a)
                - gammaln(a)
            )

        loglik += np.log(quad(density, 0, np.inf, epsabs=0, epsrel=1e-12)[0])
    _, d = np.unique(x[event], return_counts=True)
    expected = loglik + event.sum() - np.sum(d * np.log(d))
    assert m.log_likelihood == pytest.approx(expected, abs=1e-8)


def test_theta_zero_is_the_cox_model():
    x, c, Z, g = _simulate(1)
    m = CoxFrailty.fit(x, Z=Z, c=c, groups=g, theta=0.0)
    cox = CoxPH.fit(x, Z, c=c)
    np.testing.assert_allclose(m.beta, cox.beta, rtol=1e-10)
    np.testing.assert_allclose(m.H0, cox.H0, rtol=1e-8)
    np.testing.assert_allclose(
        m.standard_errors()[0], cox.standard_errors()[0], 1e-6
    )


def test_predictions():
    x, c, Z, g = _simulate(2)
    m = CoxFrailty.fit(x, Z=Z, c=c, groups=g)
    t = np.array([0.1, 3.0, 8.0, 20.0, 1e3])
    z = [0.5]
    s = np.exp(0.5 * m.beta[0]) * m._H0(t)
    # the marginal curve is the gamma's Laplace transform
    np.testing.assert_allclose(
        m.sf(t, z), (1 + m.theta * s) ** (-1 / m.theta), rtol=1e-12
    )
    # a group's curve uses its posterior frailty
    u = m.frailties["3"]
    np.testing.assert_allclose(m.sf(t, z, group=3), np.exp(-u * s), 1e-12)
    # 1 before the first time and held after the last
    assert m.sf([1e-6], z)[0] == 1.0
    assert m.sf([1e3], z)[0] == m.sf([1e4], z)[0]
    # hf is the jump of Hf at each baseline time: the jumps add up
    jumps = m.hf(m.x, np.full((m.x.size, 1), 0.5))
    np.testing.assert_allclose(
        np.cumsum(jumps), m.Hf(m.x, np.full((m.x.size, 1), 0.5)), rtol=1e-9
    )


def test_recovers_parameters_on_simulated_data():
    # One larger data set: beta and theta within 2.5 standard errors, the
    # baseline at t = 10 near its true value 1 (Weibull(10, 1.5)).
    x, c, Z, g = _simulate(3, G=200)
    m = CoxFrailty.fit(x, Z=Z, c=c, groups=g)
    se = dict(zip(m.parameter_names, m.standard_errors()))
    assert abs(m.beta[0] - 0.7) < 2.5 * se["coef_0"]
    assert abs(m.theta - 0.5) < 2.5 * se["theta"]
    lo, hi = m.param_cb("theta")
    assert 0 < lo < m.theta < hi
    assert m._H0(np.array([10.0]))[0] == pytest.approx(1.0, rel=0.25)


def test_summary_repr_serialisation_and_data_frame():
    x, c, Z, g = _simulate(4)
    m = CoxFrailty.fit(x, Z=Z, c=c, groups=g)
    assert list(m.summary().index) == [
        ("coefficients", "coef_0"),
        ("frailty", "theta"),
    ]
    assert m.parameter_names == ["coef_0", "theta"]
    np.testing.assert_array_equal(m.params, [m.beta[0], m.theta])
    text = repr(m)
    assert "unspecified (Cox); efron ties" in text
    assert "I-likelihood" in text
    restored = sp.from_dict(json.loads(json.dumps(m.to_dict())))
    assert type(restored).__name__ == "CoxFrailtyModel"
    t = np.array([2.0, 9.0])
    np.testing.assert_array_equal(m.sf(t, [0.2]), restored.sf(t, [0.2]))
    np.testing.assert_array_equal(
        m.sf(t, [0.2], group=5), restored.sf(t, [0.2], group=5)
    )
    assert repr(restored) == text
    df = pd.DataFrame({"t": x, "cens": c, "z": Z[:, 0], "grp": g})
    from_df = CoxFrailty.fit_from_df(
        df, x_col="t", c_col="cens", group_col="grp", Z_cols="z"
    )
    np.testing.assert_allclose(from_df.params, m.params, rtol=1e-9)
    np.testing.assert_allclose(
        from_df.sf(t, pd.DataFrame({"z": [0.2, 0.2]})), m.sf(t, [0.2]), 1e-12
    )


def test_no_covariates():
    x, c, _, g = _simulate(5)
    m = CoxFrailty.fit(x, c=c, groups=g)
    assert m.beta.size == 0 and m.theta > 0
    assert m.parameter_names == ["theta"]
    assert m.sf([5.0]).shape == (1,)


def test_invalid_options_raise_value_errors():
    x, c, Z, g = _simulate(6)
    with pytest.raises(ValueError, match="tie_method"):
        CoxFrailty.fit(x, Z=Z, c=c, groups=g, tie_method="exact")
    with pytest.raises(ValueError, match="theta"):
        CoxFrailty.fit(x, Z=Z, c=c, groups=g, theta=-1.0)
    with pytest.raises(ValueError, match="two groups"):
        CoxFrailty.fit(x, Z=Z, c=c, groups=np.zeros(x.size))
    with pytest.raises(ValueError, match="right-censored"):
        CoxFrailty.fit(x, Z=Z, c=np.where(c == 1, -1, 0), groups=g)


def test_monotone_likelihood_warns_once():
    # A covariate that is 1 on exactly the censored rows: no finite
    # maximum, said once (CoxPH's warning), not a stream of warnings.
    x, c, Z, g = _simulate(7)
    Z = np.column_stack([Z, c.astype(float)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CoxFrailty.fit(x, Z=Z, c=c, groups=g)
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert "No finite maximum: the partial" in messages[0]


def test_604_cox_frailty_model_comparison_values(kidney_fits):
    m = kidney_fits["efron"]
    # The integrated log-likelihood, penalised by the two coefficients and
    # theta; BIC's n the events (#604)
    assert isinstance(m.log_likelihood, float)
    assert m.neg_ll() == -m.log_likelihood
    assert m.aic() == pytest.approx(2 * 3 - 2 * m.log_likelihood)
    events = float((load_kidney()["status"] == 1).sum())
    assert m.bic() == pytest.approx(3 * np.log(events) + 2 * m.neg_ll())
    restored = sp.from_dict(m.to_dict())
    for name in ("neg_ll", "aic", "aic_c", "bic"):
        assert getattr(restored, name)() == getattr(m, name)()
    assert restored.log_likelihood_no_frailty == m.log_likelihood_no_frailty
    # A theta given is not estimated
    x, c, Z, g = _kidney()
    fixed = CoxFrailty.fit(x, Z=Z, c=c, groups=g, theta=0.5)
    assert fixed.aic() == pytest.approx(2 * 2 + 2 * fixed.neg_ll())


def test_604_cox_frailty_old_likelihood_names_are_deprecated(kidney_fits):
    m = kidney_fits["efron"]
    with pytest.warns(DeprecationWarning, match="'log_likelihood'"):
        assert m.loglik == m.log_likelihood
    with pytest.warns(DeprecationWarning, match="log_likelihood_no_frailty"):
        assert m.loglik_no_frailty == m.log_likelihood_no_frailty
    # A dict written before v0.23
    old = m.to_dict()
    old["loglik"] = -old.pop("_neg_ll")
    old["loglik_no_frailty"] = old.pop("log_likelihood_no_frailty")
    for key in ("k", "n_events_weighted", "n_obs_weighted"):
        del old[key]
    legacy = sp.from_dict(old)
    assert legacy.log_likelihood == m.log_likelihood
    assert legacy.log_likelihood_no_frailty == m.log_likelihood_no_frailty
    assert legacy.aic() == m.aic()


def _em_state(ties, theta, seed=3, G=80, per=6, tied=False):
    from surpyval.univariate.regression.frailty import cox_frailty as cf
    from surpyval.univariate.regression.frailty.frailty_fitter import (
        grouped_data,
    )

    x, c, Z, g = _simulate(seed, G=G, per=per)
    Z = np.column_stack([Z, np.random.default_rng(seed).normal(size=len(x))])
    if tied:
        x = np.ceil(x)
    x, Zm, c, w, labels, inv = grouped_data(x, Z, c, None, g)
    em = cf._CoxFrailtyEM(
        x, Zm - Zm.mean(axis=0), c, w, inv, labels.shape[0], ties
    )
    beta, log_u, _ = em.em(theta, tol=1e-10)
    return em, beta, log_u


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("theta", [1e-12, 0.05, 0.5, 20.0])
def test_551_covariance_from_the_blocks_is_the_full_inverse(ties, tied, theta):
    # The Schur complement of the frailty block, solved by conjugate
    # gradients, is the coefficients' block of the inverse of the full
    # penalised information, to the last digits.
    em, beta, log_u = _em_state(ties, max(theta, 1e-6), tied=tied)
    got = em._schur_covariance(theta, beta, log_u)
    full = em._dense_beta_covariance(theta, beta, log_u)
    np.testing.assert_allclose(got, full, rtol=1e-11)


def test_551_covariance_does_not_form_the_group_indicators(monkeypatch):
    # The full information has a column per group: O(n G^2) to form and
    # O(G^3) to invert, 22 s at G = 4000 (n = 1e4). The covariance takes
    # the partial likelihood's information in the coefficients alone.
    em, beta, log_u = _em_state("efron", 0.5)
    widths = []
    generator = em.generator

    def spy(x, Z, *args):
        widths.append(Z.shape[1])
        return generator(x, Z, *args)

    monkeypatch.setattr(em, "generator", spy)
    cov = em.beta_covariance(0.5, beta, log_u)
    assert max(widths) <= em.p + 1 < em.G
    assert np.all(np.linalg.eigvalsh(cov) > 0)
