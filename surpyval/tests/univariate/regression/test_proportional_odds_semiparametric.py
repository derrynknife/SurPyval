"""The semi-parametric proportional odds model, ``ProportionalOdds``
(#341).

The fit is checked against its own definition here -- the derivatives
against finite differences, the O(m) solve against a dense one, the
profile information against the numerical second derivative of the
profile likelihood, the maximum against a general-purpose optimiser --
and against R in ``surpyval/tests/reference/test_proportional_odds.py``.
The coverage study is in ``surpyval/tests/calibration``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import surpyval as sp
from surpyval.univariate.regression.proportional_odds.proportional_odds import (  # noqa: E501
    _inner,
    _POLikelihood,
    _profile_fit,
)


def _po_data(n=120, seed=1, beta=(1.0, -0.5), truncate=False):
    """Proportional odds data with a log-logistic baseline (alpha 10,
    shape 2): failure odds (t / 10)^2 exp(-beta'Z)."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, n), rng.normal(size=n)])
    U = rng.uniform(size=n)
    T = 10 * (U / (1 - U) * np.exp(Z @ np.asarray(beta))) ** 0.5
    C = rng.uniform(0, 30, n)
    x = np.round(np.minimum(T, C), 1)  # rounded: some ties
    c = (T > C).astype(float)
    tl = np.full(n, -np.inf)
    if truncate:
        tl = np.where(rng.uniform(size=n) < 0.3, 0.5 * x, -np.inf)
    w = rng.integers(1, 3, n).astype(float)
    return x, c, w, tl, Z


def _likelihood(truncate=True):
    x, c, w, tl, Z = _po_data(truncate=truncate)
    lik = _POLikelihood(x, c, w, tl, Z - Z.mean(axis=0))
    return lik, lik.start(x, w, tl)


def test_derivatives_match_finite_differences():
    lik, u = _likelihood()
    gamma = np.array([-0.3, 0.2])
    der = lik.derivatives(u, gamma)
    m, h = lik.m, 1e-6
    I_m, I_p = np.eye(m), np.eye(2)

    def num_grad(f, v, eye):
        return np.array([(f(v + h * e) - f(v - h * e)) / (2 * h) for e in eye])

    g_u = num_grad(lambda v: lik.value(v, gamma), u, I_m)
    g_g = num_grad(lambda v: lik.value(u, v), gamma, I_p)
    np.testing.assert_allclose(der["grad_u"], g_u, atol=1e-6)
    np.testing.assert_allclose(der["grad_gamma"], g_g, atol=1e-6)

    # The negative Hessian, from differences of the analytic gradient.
    U = np.triu(np.ones((m, m)))
    G = np.diag(der["g"])
    N_uu = np.diag(der["D"]) - G @ U @ np.diag(der["w"]) @ U.T @ G
    num_uu = -np.column_stack(
        [
            (
                lik.derivatives(u + h * e, gamma)["grad_u"]
                - lik.derivatives(u - h * e, gamma)["grad_u"]
            )
            / (2 * h)
            for e in I_m
        ]
    )
    np.testing.assert_allclose(N_uu, num_uu, atol=1e-5 * np.abs(N_uu).max())
    num_ug = -np.column_stack(
        [
            (
                lik.derivatives(u, gamma + h * e)["grad_u"]
                - lik.derivatives(u, gamma - h * e)["grad_u"]
            )
            / (2 * h)
            for e in I_p
        ]
    )
    np.testing.assert_allclose(der["N_ug"], num_ug, atol=1e-5)
    num_gg = -np.column_stack(
        [
            (
                lik.derivatives(u, gamma + h * e)["grad_gamma"]
                - lik.derivatives(u, gamma - h * e)["grad_gamma"]
            )
            / (2 * h)
            for e in I_p
        ]
    )
    np.testing.assert_allclose(der["N_gg"], num_gg, atol=1e-5)


def test_structured_solve_matches_a_dense_solve():
    lik, u = _likelihood()
    gamma = np.array([-0.3, 0.2])
    u, der = _inner(lik, gamma, u, 1e-12)
    m = lik.m
    U = np.triu(np.ones((m, m)))
    G = np.diag(der["g"])
    N_uu = np.diag(der["D"]) - G @ U @ np.diag(der["w"]) @ U.T @ G
    rhs = np.random.default_rng(0).normal(size=(m, 3))
    dense = np.linalg.solve(N_uu, rhs)
    np.testing.assert_allclose(lik.solve_uu(der, rhs), dense, rtol=1e-9)
    np.testing.assert_allclose(
        lik.solve_uu(der, rhs[:, 0]), dense[:, 0], rtol=1e-9
    )


def test_profile_information_is_the_profile_likelihoods_curvature():
    # Murphy and van der Vaart (2000): the standard errors are those of
    # the profile likelihood. The Schur complement used for them is its
    # negative Hessian, which a numerical second difference of the
    # profile log-likelihood reproduces.
    lik, u = _likelihood()
    gamma, u, der, S, ok, _ = _profile_fit(lik, u, np.zeros(2), 1e-10, 50)
    assert ok

    def pl(g):
        return _inner(lik, g, u, 1e-14)[1]["value"]

    h = 1e-4
    num = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = h * np.eye(2)[i], h * np.eye(2)[j]
            num[i, j] = -(
                pl(gamma + ei + ej)
                - pl(gamma + ei - ej)
                - pl(gamma - ei + ej)
                + pl(gamma - ei - ej)
            ) / (4 * h * h)
    np.testing.assert_allclose(S, num, rtol=1e-5)


def test_fit_is_the_joint_maximum():
    # A general-purpose optimiser over all the jumps and coefficients,
    # started anywhere, finds no higher likelihood.
    lik, u0 = _likelihood()
    gamma, u, der, _, ok, _ = _profile_fit(lik, u0, np.zeros(2), 1e-10, 50)
    assert ok
    m = lik.m

    def neg(theta):
        d = lik.derivatives(theta[:m], theta[m:])
        return -d["value"], -np.r_[d["grad_u"], d["grad_gamma"]]

    rng = np.random.default_rng(3)
    start = np.r_[u0 + rng.normal(0, 0.5, m), rng.normal(0, 0.5, 2)]
    res = minimize(neg, start, jac=True, method="L-BFGS-B")
    assert -res.fun <= der["value"] + 1e-8
    np.testing.assert_allclose(res.x[m:], gamma, atol=1e-3)


def test_sign_and_baseline_match_the_parametric_po():
    # On data from a log-logistic proportional odds model the NPMLE and
    # the parametric PO(LogLogistic) fit estimate the same coefficients,
    # with the same (survival odds) sign, and the same baseline odds.
    x, c, _, _, Z = _po_data(n=2000, seed=5)
    semi = sp.ProportionalOdds.fit(x, Z, c=c)
    par = sp.PO(sp.LogLogistic).fit(x, Z, c=c)
    np.testing.assert_allclose(semi.beta, par.params[2:], atol=0.05)
    assert np.all(np.abs(semi.beta - [1.0, -0.5]) < 3 * semi.se)
    t = np.array([5.0, 10.0, 20.0])
    np.testing.assert_allclose(
        semi.sf(t, [0.0, 0.0]), par.sf(t, [0.0, 0.0]), atol=0.02
    )
    # A positive coefficient is a longer life, as for the parametric model.
    assert semi.sf(10.0, [1.0, 0.0]) > semi.sf(10.0, [0.0, 0.0])
    assert semi.concordance() > 0.5


def test_delayed_entry_splits_exactly():
    # A row observed on (0, x] has the likelihood of the same subject
    # censored at s and then entering at s: the product of 1 / (1 + G(s)
    # e) and the conditional term is the original term, so the fits agree.
    x, c, _, _, Z = _po_data(n=80, seed=2)
    whole = sp.ProportionalOdds.fit(x, Z, c=c)
    s = 0.5 * x[:20]
    xs = np.r_[x, s]
    cs = np.r_[c, np.ones(20)]
    tl = np.r_[
        np.where(np.arange(80) < 20, np.r_[s, np.zeros(60)], -np.inf),
        np.full(20, -np.inf),
    ]
    split = sp.ProportionalOdds.fit(xs, np.r_[Z, Z[:20]], c=cs, tl=tl)
    np.testing.assert_allclose(split.beta, whole.beta, rtol=1e-8)
    np.testing.assert_allclose(split.se, whole.se, rtol=1e-6)
    np.testing.assert_allclose(
        split.sf(x, Z), whole.sf(x, Z), rtol=1e-8, atol=1e-12
    )


def test_invariant_to_a_monotone_change_of_time():
    # The likelihood uses the order of the times only.
    x, c, _, _, Z = _po_data(n=100, seed=4)
    a = sp.ProportionalOdds.fit(x, Z, c=c)
    b = sp.ProportionalOdds.fit(np.log(x), Z, c=c)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(
        a.sf(x, Z), b.sf(np.log(x), Z), rtol=1e-10, atol=1e-14
    )


def test_functions_agree_with_the_baseline():
    x, c, _, _, Z = _po_data(n=60, seed=6)
    model = sp.ProportionalOdds.fit(x, Z, c=c)
    z = [1.0, 0.3]
    t = model.x
    G = model.G0 * np.exp(-model.beta @ np.asarray(z))
    np.testing.assert_allclose(model.sf(t, z), 1 / (1 + G), rtol=1e-12)
    # hf and df are the jumps of Hf and ff at the baseline times.
    np.testing.assert_allclose(
        np.cumsum(model.hf(t, z)), model.Hf(t, z), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.cumsum(model.df(t, z)), model.ff(t, z), rtol=1e-10
    )
    # Before the first time nothing has happened; after the last it holds.
    assert model.sf(t[0] - 1, z) == 1.0
    assert model.sf(t[-1] + 100, z) == model.sf(t[-1], z)
    # The jumps are where the events are.
    assert np.all((model.g0 > 0) == (model.d > 0))


def test_center_gives_the_same_model():
    x, c, _, _, Z = _po_data(n=80, seed=7)
    Z = Z + [0.0, 50.0]
    a = sp.ProportionalOdds.fit(x, Z, c=c)
    b = sp.ProportionalOdds.fit(x, Z, c=c, center=True)
    np.testing.assert_allclose(b.center, Z.mean(axis=0))
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-12)
    np.testing.assert_allclose(a.sf(x, Z), b.sf(x, Z), rtol=1e-10)


def test_baseline_far_from_the_covariates_is_refused():
    x, c, _, _, Z = _po_data(n=80, seed=7)
    Z = Z + [0.0, 1e5]
    with pytest.raises(ValueError, match="center=True") as err:
        sp.ProportionalOdds.fit(x, Z, c=c)
    model = sp.ProportionalOdds.fit(x, Z, c=c, center=True)
    assert np.all(np.isfinite(model.sf(x, Z)))
    # Worded as CoxPH's and FineGray's refusals (one helper): the failure
    # odds at Z = 0 are exp(beta'center) times those at the means.
    s = float(model.beta @ model.center)
    message = str(err.value)
    assert "beta'center = {:.4g}".format(s) in message
    assert "is exp({:.4g}) times".format(s) in message


@pytest.mark.parametrize("flag", [-1, 2])
def test_left_and_interval_censoring_are_refused(flag):
    x, c, _, _, Z = _po_data(n=20)
    xx = np.column_stack([x, x])
    cc = c.copy()
    cc[0] = flag
    if flag == 2:
        xx[0, 1] = xx[0, 0] + 1
    with pytest.raises(ValueError, match=r"right-censored \(c=1\)"):
        sp.ProportionalOdds.fit(xx, Z, c=cc)


def test_two_sided_truncation_is_refused():
    x, c, _, _, Z = _po_data(n=20)
    with pytest.raises(ValueError, match="left truncation"):
        sp.ProportionalOdds.fit(
            x, Z, c=c, tl=np.column_stack([np.zeros(20), np.full(20, 99.0)])
        )


def test_no_events_is_refused():
    with pytest.raises(ValueError, match="at least one event"):
        sp.ProportionalOdds.fit([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], c=[1, 1, 1])


def test_no_finite_maximum_warns_once_at_the_caller():
    # A level with no events: its survival odds run off to infinity.
    x, c, _, _, Z = _po_data(n=60, seed=8)
    Z[:, 0] = c
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = sp.ProportionalOdds.fit(x, Z, c=c)
    messages = [str(w.message) for w in caught]
    assert len(caught) == 1, messages
    assert "No finite maximum" in messages[0] and "[0]" in messages[0]
    assert caught[0].filename == __file__
    assert model.beta[0] > 5


def test_ordinary_fit_does_not_warn():
    x, c, w, tl, Z = _po_data(n=200, seed=9, truncate=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.ProportionalOdds.fit(x, Z, c=c, n=w, tl=tl)


def test_summary_param_cb_and_frame_entry_points_agree():
    x, c, w, tl, Z = _po_data(n=100, seed=10, truncate=True)
    model = sp.ProportionalOdds.fit(x, Z, c=c, n=w, tl=tl)
    table = model.summary()
    for k, name in enumerate(model.parameter_names):
        lo, hi = model.param_cb(name)
        assert lo == pytest.approx(table["coef lower 95%"].iloc[k])
        assert hi == pytest.approx(table["coef upper 95%"].iloc[k])
    df = pd.DataFrame({"x": x, "c": c, "w": w, "tl": tl})
    df["a"], df["b"] = Z[:, 0], Z[:, 1]
    framed = sp.ProportionalOdds.fit_from_df(
        df, x_col="x", c_col="c", n_col="w", tl_col="tl", formula="a + b"
    )
    np.testing.assert_allclose(framed.beta, model.beta, rtol=1e-12)
    assert list(framed.summary().index) == ["a", "b"]
    assert "survival odds ratio" in repr(framed)
    restored = sp.ProportionalOddsModel.from_json(framed.to_json())
    np.testing.assert_array_equal(restored.sf(x, df), framed.sf(x, df))
    np.testing.assert_array_equal(restored.covariance(), framed.covariance())


def test_no_convergence_warning_on_ordinary_fits():
    # The Newton decrement is exactly 0 at the maximum of some fits; it was
    # taken for a profile that is not concave there, and 6 of 1000
    # ordinary fits ran out of iterations at their maximum and warned that
    # they had not converged (two of them among these 120).
    rng = np.random.default_rng(20261202)
    beta = np.array([1.0, -0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(120):
            Z = np.column_stack(
                [rng.binomial(1, 0.5, 400), rng.normal(size=400)]
            )
            U = rng.uniform(size=400)
            T = 10 * (U / (1 - U) * np.exp(Z @ beta)) ** 0.5
            C = rng.uniform(0, 30, 400)
            x, c = np.minimum(T, C)[:200], (T > C)[:200].astype(int)
            model = sp.ProportionalOdds.fit(x, Z[:200], c=c)
            assert model.n_iter < 10


def test_truncated_fit_with_a_non_concave_start_is_not_aliased():
    # With delayed entry the profile likelihood need not be concave at
    # beta = 0. Its information there was the aliasing check's yardstick,
    # and a negative eigenvalue aliased a continuous covariate (one fit in
    # 1000 of the left-truncation study). Aliasing is decided by the
    # covariates themselves now, and the fit reaches the maximum.
    rng = np.random.default_rng(384)
    Z = np.column_stack([rng.binomial(1, 0.5, 40), rng.normal(size=40)])
    U = rng.uniform(size=40)
    T = 10 * (U / (1 - U) * np.exp(Z @ [1.0, -0.5])) ** 0.5
    tl = rng.uniform(0, 5, 40)
    keep = T > tl
    C = np.maximum(rng.uniform(0, 30, 40), tl + 0.01)
    x, c = np.minimum(T, C)[keep], (T > C)[keep].astype(float)
    Z, tl = Z[keep], tl[keep]
    w = np.ones(x.size)
    lik = _POLikelihood(x, c, w, tl, Z - Z.mean(axis=0))
    _, der = _inner(lik, np.zeros(2), lik.start(x, w, tl), 1e-12)
    assert np.linalg.eigvalsh(lik.schur(der)).min() < 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.ProportionalOdds.fit(x, Z, c=c, tl=tl)
    np.testing.assert_allclose(model.beta, [0.5132, -0.9434], atol=1e-4)


def test_604_proportional_odds_model_comparison_values():
    x, c, w, _, Z = _po_data()
    model = sp.ProportionalOdds.fit(x, Z, c=c, n=w)
    # The profile likelihood, penalised by the coefficients (the baseline
    # is profiled out, as a Cox model's is); BIC's n the events (#604)
    ll = model.log_likelihood
    assert isinstance(ll, float) and model.neg_ll() == -ll
    assert model.aic() == pytest.approx(2 * 2 - 2 * ll)
    events = w[c == 0].sum()
    assert model.bic() == pytest.approx(2 * np.log(events) - 2 * ll)
    restored = sp.from_dict(model.to_dict())
    for name in ("neg_ll", "aic", "aic_c", "bic"):
        assert getattr(restored, name)() == getattr(model, name)()
    # A dict written before v0.23 stored the log-likelihood itself
    old = model.to_dict()
    old["log_likelihood"] = -old.pop("_neg_ll")
    del old["ic_n"]
    assert sp.from_dict(old).bic() == pytest.approx(model.bic(), rel=1e-15)
