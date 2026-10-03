"""Lin & Ying (1994) semi-parametric additive hazards model.

Validation strategy (no R in CI): the closed-form estimator is checked
against a brute-force numerical integration of the estimating equation, its
point estimates against simulation from a known additive model, and its
sandwich standard errors against the empirical spread of repeated fits.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import AdditiveHazards, ExponentialAH
from surpyval.datasets import load_rossi_static
from surpyval.univariate.regression.additive_hazards.additive_hazards import (
    AdditiveHazardsModel,
)


def _simulate(N, seed, lambda0=0.5, beta=(0.30, -0.15), tau=6.0):
    # h(t | Z) = lambda0 + beta'Z is constant in t, so survival is
    # exponential with rate lambda0 + beta'Z; recovering beta validates the
    # estimator against a known truth. Covariates are kept small so the
    # hazard stays positive.
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta)
    Z = rng.uniform(-1, 1, size=(N, beta.size))
    rate = lambda0 + Z @ beta
    Z, rate = Z[rate > 0], rate[rate > 0]
    x = rng.exponential(1.0 / rate)
    c = (x > tau).astype(int)
    x = np.minimum(x, tau)
    return x, c, Z


def test_closed_form_matches_brute_force_integration():
    # A (a time integral over the risk sets) and b (a sum over events) must
    # match a direct numerical evaluation of the Lin-Ying estimating
    # equation.
    x = np.array([2.0, 5.0, 1.0, 8.0, 4.0])
    c = np.array([0, 0, 1, 0, 0])
    Z = np.array([[0.5], [1.0], [0.2], [-0.3], [0.8]])
    model = AdditiveHazards.fit(x, Z, c=c)

    grid = np.linspace(0, x.max(), 200_000)
    dt = grid[1] - grid[0]
    A_bf = 0.0
    for t in grid:
        at_risk = x >= t
        if at_risk.sum() == 0:
            continue
        Zt = Z[at_risk]
        centered = Zt - Zt.mean(0)
        A_bf += (centered[:, :, None] * centered[:, None, :]).sum(0) * dt
    b_bf = np.zeros(1)
    for i in np.where(c == 0)[0]:
        b_bf += Z[i] - Z[x >= x[i]].mean(0)

    assert np.allclose(model._A, A_bf, atol=1e-3)
    assert np.allclose(model._b, b_bf, atol=1e-4)
    assert np.allclose(model.beta, np.linalg.solve(A_bf, b_bf), atol=1e-3)


def test_estimating_equation_is_solved():
    x, c, Z = _simulate(500, 0)
    model = AdditiveHazards.fit(x, Z, c=c)
    # beta = A^-1 b, so A beta must reproduce b exactly.
    assert np.allclose(model._A @ model.beta, model._b)


def test_recovers_known_beta():
    x, c, Z = _simulate(20000, 1)
    model = AdditiveHazards.fit(x, Z, c=c)
    assert np.allclose(model.beta, [0.30, -0.15], atol=0.02)
    assert np.all(
        np.abs(model.beta - [0.30, -0.15]) < 3 * model.standard_errors()
    )


def test_sandwich_se_matches_empirical_spread():
    # The Lin-Ying sandwich SE should track the actual sampling spread of
    # the estimate across repeated samples.
    ests, ses = [], []
    for s in range(80):
        x, c, Z = _simulate(1500, 100 + s)
        model = AdditiveHazards.fit(x, Z, c=c)
        ests.append(model.beta)
        ses.append(model.standard_errors())
    empirical_sd = np.std(ests, axis=0)
    mean_se = np.mean(ses, axis=0)
    assert np.allclose(empirical_sd, mean_se, rtol=0.2)


def test_p_values_shape_and_significance():
    x, c, Z = _simulate(20000, 2)
    model = AdditiveHazards.fit(x, Z, c=c)
    assert model.p_values.shape == (2,)
    # Both effects are real and the sample is large, so both are significant.
    assert np.all(model.p_values < 0.05)
    assert model.covariance().shape == (2, 2)
    assert np.allclose(
        model.standard_errors(), np.sqrt(np.diag(model.covariance()))
    )


def test_counts_equivalent_to_repeated_rows():
    x = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    c = np.array([0, 0, 1, 0, 0])
    Z = np.array([[0.4], [0.7], [0.7], [-0.2], [0.9]])
    m_rep = AdditiveHazards.fit(
        np.repeat(x, 2), np.repeat(Z, 2, axis=0), c=np.repeat(c, 2)
    )
    m_cnt = AdditiveHazards.fit(x, Z, c=c, n=np.full(5, 2))
    assert np.allclose(m_rep.beta, m_cnt.beta)
    assert np.allclose(m_rep.H0, m_cnt.H0)


def test_baseline_survival_matches_exponential():
    # With covariate effects removed (Z = 0) the fitted survival should be
    # the constant-baseline exponential the data were generated from.
    x, c, Z = _simulate(20000, 3)
    model = AdditiveHazards.fit(x, Z, c=c)
    t = np.array([0.5, 1.0, 2.0, 3.0])
    sf0 = model.sf(t, np.array([[0.0, 0.0]]))
    assert np.allclose(sf0, np.exp(-0.5 * t), atol=0.03)


def test_prediction_methods_are_consistent():
    x, c, Z = _simulate(2000, 4)
    model = AdditiveHazards.fit(x, Z, c=c)
    t = np.array([1.0, 2.0, 3.0])
    Z0 = np.array([[0.2, -0.1]])
    assert np.allclose(model.ff(t, Z0), 1 - model.sf(t, Z0))
    assert np.allclose(model.df(t, Z0), model.hf(t, Z0) * model.sf(t, Z0))
    # H(t | Z) = H0(t) + t * beta'Z is linear in t beyond the baseline step.
    assert model.Hf(t, Z0).shape == (3,)


def test_fit_from_df_retains_feature_names():
    rng = np.random.default_rng(5)
    Z = rng.uniform(-1, 1, size=(500, 2))
    rate = 0.5 + Z @ np.array([0.3, -0.15])
    x = rng.exponential(1 / rate)
    df = pd.DataFrame({"time": x, "event": 1, "age": Z[:, 0], "dose": Z[:, 1]})
    # surpyval censoring convention: 0 = observed event, 1 = censored.
    df["c"] = 0
    model = AdditiveHazards.fit_from_df(
        df, x_col="time", Z_cols=["age", "dose"], c_col="c"
    )
    assert model.feature_names == ["age", "dose"]
    # Prediction accepts a DataFrame and selects the right columns.
    pred = model.sf([1.0, 2.0], df[["age", "dose"]].iloc[[0]])
    assert pred.shape == (2,)


def test_rejects_interval_and_left_censoring():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="right-censored"):
        AdditiveHazards.fit(x, np.array([[0.5], [0.7]]), c=np.array([2, 2]))


def test_estimate_holds_past_the_last_time():
    # There is no risk set past the last observed time, so the estimate
    # holds there like every other semi-parametric one (#400); it used to
    # keep drifting at the last interval's rate beta'(Z - Zbar).
    x, c, Z = _simulate(300, 5)
    model = AdditiveHazards.fit(x, Z, c=c)
    last = model.x[-1]
    t = last * np.array([1.0, 1.5, 10.0, 100.0])
    Z0 = np.array([0.8, -0.9])
    H = model.Hf(t, Z0)
    assert np.all(H == H[0]), H
    assert np.all(model.sf(t, Z0) == model.sf(last, Z0))
    assert np.all(model.hf(t[1:], Z0) == 0.0)
    assert np.all(model.df(t[1:], Z0) == 0.0)
    # Inside the data nothing changed: the hazard still moves.
    assert model.Hf(0.5 * last, Z0) < H[0]


def test_hf_is_nan_at_a_nan_time():
    x, c, Z = _simulate(300, 5)
    model = AdditiveHazards.fit(x, Z, c=c)
    out = model.hf(np.array([1.0, np.nan]), np.array([0.2, -0.1]))
    assert np.isfinite(out[0]) and np.isnan(out[1]), out


# ---------------------------------------------------------------------------
# The bandwidth on coincident event times (#289).
# ---------------------------------------------------------------------------


class TestRound2FollowUps:
    def test_ah_bandwidth_coincident_events(self):
        # 289: nearly-tied event times used to collapse the bandwidth to
        # the floor and return Dirac spikes (~1.6e9).
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = AdditiveHazards.fit(x=[5.0, 5.0 + 1e-9], Z=[[0.0], [0.1]])
        hf = float(np.ravel(m.hf([5.0], np.array([0.05])))[0])
        assert np.isfinite(hf)
        assert hf < 100.0


# ---------------------------------------------------------------------------
# #277: Lin-Ying ``hf``/``df`` on the hazard scale; ``phi()`` on
# additive models.
# ---------------------------------------------------------------------------


class TestLinYingHazardRate:
    def test_hf_df_on_hazard_scale(self):
        # 277: hf used to return ~beta'Z (the baseline jump vanishes as
        # n grows); it must estimate h0(t) + beta'Z.
        np.random.seed(2)
        n = 20000
        Z = np.random.uniform(size=(n, 1))
        t = np.random.exponential(1 / (0.5 + 0.7 * Z[:, 0]))
        m = AdditiveHazards.fit(x=t, Z=Z)
        z = np.array([0.5])
        hf = np.ravel(m.hf([0.5, 1.0, 1.5], z))
        assert np.all(np.abs(hf - 0.85) < 0.12)
        df = float(np.ravel(m.df([1.0], z))[0])
        assert df == pytest.approx(0.85 * np.exp(-0.85), rel=0.15)

    def test_parametric_ah_phi_raises_not_implemented(self):
        np.random.seed(4)
        Z = np.random.uniform(size=(300, 1))
        t = np.random.exponential(1 / (0.5 + 0.7 * Z[:, 0]))
        m = ExponentialAH.fit(x=t, Z=Z)
        with pytest.raises(NotImplementedError, match="additive"):
            m.phi(np.array([0.5]))


# ---------------------------------------------------------------------------
# Predictions do not depend on covariate centring.
# ---------------------------------------------------------------------------


def test_lin_ying_hf_is_invariant_to_covariate_centring():
    rng = np.random.default_rng(21)
    Z = rng.uniform(0, 2, size=(200, 1))
    T = rng.exponential(1 / (0.1 + 0.05 * Z[:, 0]))
    C = rng.uniform(2, 20, 200)
    x, c = np.minimum(T, C), (T > C).astype(int)
    m1 = AdditiveHazards.fit(x=x, Z=Z, c=c)
    m2 = AdditiveHazards.fit(x=x, Z=Z + 3, c=c)
    t = np.array([0.01, 1.0, 3.0, 5.5, 40.0])
    np.testing.assert_allclose(m1.Hf(t, [0.5]), m2.Hf(t, [3.5]), rtol=1e-10)
    # The drift is stored, so a restored model predicts the same.
    restored = AdditiveHazardsModel.from_dict(
        json.loads(json.dumps(m1.to_dict()))
    )
    np.testing.assert_allclose(restored.Hf(t, [0.5]), m1.Hf(t, [0.5]))


# ---------------------------------------------------------------------------
# #376: the Lin-Ying survival stays a survival.
# ---------------------------------------------------------------------------


def _lin_ying():
    rng = np.random.default_rng(11)
    Z = rng.normal(size=(120, 2))
    x = rng.exponential(1 / (0.2 + 0.05 * Z[:, 0] - 0.04 * Z[:, 1]).clip(0.01))
    c = (x > 8).astype(int)
    return sp.AdditiveHazards.fit(np.minimum(x, 8), Z, c=c)


@pytest.mark.parametrize("z", [[0.0, 0.0], [-3.0, 3.0], [2.0, -2.0]])
def test_lin_ying_survival_is_in_bounds_and_non_increasing(z):
    # Row (-3, 3) has a negative hazard; the survival used to climb above
    # 1 (the conformance fixture reached 57.8 at Z = (-2, 2) and 1.21 at
    # Z = 0, inside the data) and could be inf.
    model = _lin_ying()
    t = np.linspace(-1.0, 12.0, 2001)
    sf = model.sf(t, np.array(z))
    assert np.all((sf >= 0) & (sf <= 1))
    assert np.all(np.diff(sf) <= 0)
    assert np.all(sf[t <= 0] == 1.0)
    hf = model.hf(t, np.array(z))
    assert np.all(hf >= 0)


def test_lin_ying_hf_is_the_running_maximum_of_the_estimate():
    model = _lin_ying()
    t = np.sort(np.concatenate([np.linspace(0, 10, 4001), model.x]))
    for z in ([0.0, 0.0], [-3.0, 3.0], [1.0, -1.0]):
        bz = np.asarray(z) @ model.beta
        held = np.minimum(t, model.x[-1])
        H = model._baseline_H(held) + held * bz
        envelope = np.maximum(np.maximum.accumulate(H), 0.0)
        np.testing.assert_allclose(
            model.Hf(t, np.array(z)), envelope, rtol=1e-12, atol=1e-15
        )


def test_lin_ying_predictions_inside_the_data_are_the_estimate():
    # The docstring's Rossi prediction is where the estimate is at its
    # running maximum: unchanged.
    df = load_rossi_static()
    model = sp.AdditiveHazards.fit(
        df["week"].values,
        df[["fin", "age", "prio"]].values,
        c=1 - df["arrest"].values,
    )
    np.testing.assert_allclose(
        model.sf([20, 52], [1, 25, 3]), [0.9269, 0.7725], atol=5e-5
    )


def test_605_cov_is_deprecated_for_covariance():
    df = load_rossi_static()
    model = AdditiveHazards.fit(
        df["week"].values, df[["fin", "age"]].values, c=1 - df["arrest"].values
    )
    with pytest.warns(DeprecationWarning, match=r"use 'covariance\(\)'"):
        np.testing.assert_array_equal(model.cov, model.covariance())
    d = model.to_dict()
    assert "covariance" in d and "cov" not in d
    d["cov"] = d.pop("covariance")
    np.testing.assert_array_equal(
        sp.from_dict(d).covariance(), model.covariance()
    )
