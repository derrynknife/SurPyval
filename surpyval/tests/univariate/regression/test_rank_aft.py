"""Rank-based (Gehan and log-rank) semi-parametric AFT regression.

No other package is used as a reference here (the R reference script,
``scripts/reference/reference_r_rank_aft.R``, is checked by
``test_against_r_rank_aft`` once its output exists). The estimator is
checked from first principles instead:

- the Gehan estimate minimises a convex loss that is a linear programme:
  the exact LP minimiser (``scipy.optimize.linprog``) is found on small
  data with censoring and tied times, the step estimating function
  changes sign across it, and the smoothed estimate tends to it as the
  smoothing covariance shrinks, inside a bound proved from the loss;
- the log-rank estimate is the fixed point of iterated weighted Gehan
  fits: the same iteration on exact LPs gives the fixed point the
  smoothed one tends to;
- the step estimating functions change sign across the reported
  estimates;
- with no censoring and a symmetric error the Gehan estimate agrees with
  least squares on the log times, in expectation over seeded data sets;
- on one large censored data set it agrees with ``BuckleyJames``, which
  estimates the same coefficients;
- the standard errors are checked against a simulation's spread and
  coverage in ``calibration/test_coverage_rank_aft.py``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surpyval import BuckleyJames, RankAFT
from surpyval.tests._helpers import counted_regression_data
from surpyval.univariate.regression import RankAFTModel
from surpyval.univariate.regression.rank_aft import rank_aft as ra
from surpyval.univariate.regression.regression_data import NoLikelihoodError

REFERENCE = (
    Path(__file__).parents[2] / "reference" / "data" / "rank_aft_r.json"
)


def _data(N, seed, censor_hi=15.0, rounding=1, errors="normal"):
    """``log T = 1 + gamma'Z + eps`` with ``gamma = (0.5, -1)``, ``Z`` a
    binary and a uniform covariate, uniform censoring on (1, censor_hi),
    times rounded to ``rounding`` decimals (ties). Returns ``x, Z, c``
    and the textbook slope ``gamma``."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack(
        [rng.binomial(1, 0.5, N), rng.uniform(-1, 1, N)]
    ).astype(float)
    if errors == "normal":
        eps = 0.5 * rng.normal(size=N)
    else:
        eps = 0.5 * np.log(rng.exponential(size=N))
    T = np.exp(1.0 + Z @ [0.5, -1.0] + eps)
    C = rng.uniform(1.0, censor_hi, N)
    x = np.minimum(T, C)
    if rounding is not None:
        x = np.round(x, rounding)
    return x, Z, (T > C).astype(int), np.array([0.5, -1.0])


def _arrays(x, Z, c):
    return np.log(x), (c == 0).astype(float), Z, np.ones(len(x))


# -- the Gehan estimate and its linear programme -----------------------------


def test_gehan_lp_is_the_minimiser_of_the_loss():
    x, Z, c, _ = _data(40, 11)
    assert 0.2 < c.mean() < 0.5 and np.unique(x).size < x.size
    Y, d, Z, w = _arrays(x, Z, c)
    g = ra.gehan_lp(Y, d, Z, w)
    best = ra.gehan_loss(g, Y, d, Z, w)
    # The loss written out pair by pair, independently of the sorted sums.
    e = Y - Z @ g
    pairs = (d[:, None] * np.maximum(e[None, :] - e[:, None], 0.0)).sum()
    assert best == pytest.approx(pairs, rel=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(200):
        trial = g + rng.normal(0, 0.05, 2)
        assert ra.gehan_loss(trial, Y, d, Z, w) >= best - 1e-9
    # The step estimating function (the loss's gradient) changes sign
    # across the minimiser along each coefficient.
    for k in range(2):
        step = 1e-6 * np.eye(2)[k]
        below = ra.gehan_function(g - step, Y, d, Z, w)[k]
        above = ra.gehan_function(g + step, Y, d, Z, w)[k]
        assert below <= 0 <= above


def test_smoothed_gehan_tends_to_the_lp_as_the_smoothing_shrinks():
    # The smoothed loss exceeds the exact one by at most
    # sum_ij delta_i r_ij phi(0) (E(d + rN)^+ - d^+ <= r / sqrt(2 pi)), so
    # at the smoothed minimiser the exact loss is within that of the LP's.
    x, Z, c, _ = _data(40, 11)
    Y, d, Z, w = _arrays(x, Z, c)
    g_lp = ra.gehan_lp(Y, d, Z, w)
    best = ra.gehan_loss(g_lp, Y, d, Z, w)
    G0 = ra._start_gamma(Z, w)
    gaps, bounds, dist = [], [], []
    for s in (1.0, 1e-2, 1e-4):
        G = s * G0
        g = ra._fit_gamma(Y, d, Z, w, "gehan", 1e-8, 200, G, update=False)[0]
        L = np.linalg.cholesky(G)
        ev = d == 1
        r = np.linalg.norm((Z[ev, None, :] - Z[None, :, :]) @ L, axis=2).sum()
        gaps.append(ra.gehan_loss(g, Y, d, Z, w) - best)
        bounds.append(r / np.sqrt(2 * np.pi))
        dist.append(np.max(np.abs(g - g_lp)))
    assert all(0 <= gp <= b for gp, b in zip(gaps, bounds))
    assert gaps[-1] < 1e-2 * gaps[0]
    # Within a twentieth of a standard error of the LP at the smallest.
    model = RankAFT.fit(x, Z, c=c)
    assert np.all(dist[-1] < 0.05 * model.standard_errors())


def test_smoothed_gehan_is_close_to_the_lp_estimate():
    x, Z, c, _ = _data(100, 12, errors="extreme")
    Y, d, Z, w = _arrays(x, Z, c)
    g_lp = ra.gehan_lp(Y, d, Z, w)
    model = RankAFT.fit(x, Z, c=c)
    assert model.converged
    assert np.all(np.abs(-model.beta - g_lp) < 0.25 * model.standard_errors())


def test_smoothed_estimate_is_a_root_and_the_step_function_changes_sign():
    x, Z, c, _ = _data(100, 12, errors="extreme")
    Y, d, Z, w = _arrays(x, Z, c)
    for weights, step in (
        ("gehan", ra.gehan_function),
        ("logrank", ra.logrank_function),
    ):
        model = RankAFT.fit(x, Z, c=c, rank_weights=weights)
        g = -model.beta
        G = model._smoothing
        if weights == "gehan":
            U = ra.smoothed_gehan(g, Y, d, Z, w, G)[1]
            scale = np.abs(ra.smoothed_gehan(g, Y, d, Z, w, G)[2]).max()
        else:
            s = ra._smoothed_sums(g, Y, d, Z, w, ra._chol(G))
            U = (s["U"] / s["R"][:, None]).sum(axis=0)
            scale = np.abs(s["H"] / s["R"][:, None, None]).sum(axis=0).max()
        # The smoothed function is zero to well within a hundredth of a
        # standard error's worth of its slope.
        assert np.all(np.abs(U) < 1e-2 * scale * model.standard_errors())
        for k in range(2):
            h = 0.5 * model.standard_errors()[k] * np.eye(2)[k]
            assert step(g - h, Y, d, Z, w)[k] < 0 < step(g + h, Y, d, Z, w)[k]


# -- log-rank: iterated weighted Gehan fits ----------------------------------


def _lp_step(g, Y, d, Z, w):
    """One exact weighted Gehan fit, the weights 1 / R at ``g``."""
    R, _ = ra._at_or_above(Y - Z @ g, w, Z)
    return ra.gehan_lp(Y, d, Z, w, np.where(d == 1, w / R, 0.0))


def _iterated_lp(Y, d, Z, w, g=None):
    """The exact iteration of weighted Gehan fits, from ``g`` (the exact
    Gehan estimate when ``None``), to its fixed point."""
    g = ra.gehan_lp(Y, d, Z, w) if g is None else g
    for _ in range(50):
        new = _lp_step(g, Y, d, Z, w)
        if np.allclose(new, g, rtol=0.0, atol=1e-10):
            return new
        g = new
    raise AssertionError("the iterated LP did not settle")


def test_logrank_is_near_the_exact_iterations_fixed_points():
    # The log-rank step function has several roots close together (its
    # jumps are a pair's covariate difference over a number at risk); the
    # exact iteration of weighted Gehan LPs reaches one of them from the
    # Gehan estimate, and the smoothed fixed point tends, as the smoothing
    # shrinks, to one of them too (here not the same one: on these data
    # the exact fixed points from the two starts are 0.06 standard errors
    # apart). So: the reported estimate is within a quarter of a standard
    # error of the exact fixed point; the smoothed estimate at a
    # ten-thousandth of the smoothing is itself an exact fixed point to a
    # hundredth of a standard error (one exact weighted Gehan fit from it
    # barely moves it), and the exact iteration from it settles within a
    # twentieth of a standard error.
    x, Z, c, _ = _data(100, 12, errors="extreme")
    Y, d, Z, w = _arrays(x, Z, c)
    g_fix = _iterated_lp(Y, d, Z, w)
    model = RankAFT.fit(x, Z, c=c, rank_weights="logrank")
    assert model.converged
    assert np.all(np.abs(-model.beta - g_fix) < 0.25 * model.standard_errors())
    small = ra._fit_gamma(
        Y, d, Z, w, "logrank", 1e-8, 500, 1e-4 * model._smoothing, False
    )
    assert small[4]
    g_s = small[0]
    assert np.all(
        np.abs(_lp_step(g_s, Y, d, Z, w) - g_s)
        < 0.01 * model.standard_errors()
    )
    g_s_fix = _iterated_lp(Y, d, Z, w, g_s)
    assert np.all(np.abs(g_s_fix - g_s) < 0.05 * model.standard_errors())
    assert np.all(
        np.abs(-model.beta - g_s_fix) < 0.25 * model.standard_errors()
    )


# -- agreement with other estimators ----------------------------------------


def test_no_censoring_symmetric_errors_agrees_with_least_squares():
    # Both are unbiased for the slope with symmetric errors and no
    # censoring: their mean difference over seeded data sets is within
    # three Monte Carlo standard errors of zero.
    rng = np.random.default_rng(31)
    reps, N = 60, 80
    diff = np.empty((reps, 2))
    for r in range(reps):
        Z = np.column_stack([rng.binomial(1, 0.5, N), rng.uniform(-1, 1, N)])
        Y = 1.0 + Z @ [0.5, -1.0] + rng.logistic(0, 0.3, N)
        X = np.column_stack([np.ones(N), Z])
        ols = np.linalg.lstsq(X, Y, rcond=None)[0][1:]
        diff[r] = -RankAFT.fit(np.exp(Y), Z).beta - ols
    mcse = diff.std(axis=0, ddof=1) / np.sqrt(reps)
    assert np.all(np.abs(diff.mean(axis=0)) < 3 * mcse)


def test_agrees_with_buckley_james_on_the_same_data():
    x, Z, c, gamma = _data(1000, 5, rounding=None)
    rank = RankAFT.fit(x, Z, c=c)
    lr = RankAFT.fit(x, Z, c=c, rank_weights="logrank")
    bj = BuckleyJames.fit(x, Z, c=c)
    for m in (rank, lr, bj):
        assert np.all(np.abs(m.beta + gamma) < 3 * rank.standard_errors())
    assert np.all(np.abs(rank.beta - bj.beta) < 2 * rank.standard_errors())
    assert np.all(np.abs(lr.beta - bj.beta) < 2 * rank.standard_errors())


# -- the API -----------------------------------------------------------------


def test_counts_equal_repeated_rows():
    x, Z, n, c = counted_regression_data()
    a = RankAFT.fit(x, Z, c=c, n=n)
    b = RankAFT.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    )
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-6)
    np.testing.assert_allclose(
        a.standard_errors(), b.standard_errors(), rtol=1e-6
    )


def test_row_order_does_not_matter():
    x, Z, c, _ = _data(60, 3)
    perm = np.random.default_rng(1).permutation(60)
    a = RankAFT.fit(x, Z, c=c)
    b = RankAFT.fit(x[perm], Z[perm], c=c[perm])
    assert np.array_equal(a.beta, b.beta)


def test_predictions_read_the_residual_km():
    x, Z, c, _ = _data(200, 4)
    m = RankAFT.fit(x, Z, c=c)
    t = np.array([1.0, 3.0, 8.0])
    sf = m.sf(t, [1.0, 0.0])
    assert np.all(np.diff(sf) <= 0) and np.all((0 <= sf) & (sf <= 1))
    np.testing.assert_allclose(m.ff(t, [1.0, 0.0]), 1 - sf)
    np.testing.assert_allclose(m.Hf(t, [1.0, 0.0]), -np.log(sf))
    q = m.qf(0.3, [1.0, 0.0])
    assert m.ff(q, [1.0, 0.0]) >= 0.3
    # A positive coefficient shortens life.
    assert m.sf(5.0, [0.0, 1.0]) < m.sf(5.0, [0.0, 0.0])


def test_param_cb_wald_and_bootstrap():
    x, Z, c, _ = _data(150, 6)
    m = RankAFT.fit(x, Z, c=c)
    wald = m.param_cb()
    np.testing.assert_allclose(
        wald,
        np.column_stack(
            [
                m.beta - 1.959964 * m.standard_errors(),
                m.beta + 1.959964 * m.standard_errors(),
            ]
        ),
        rtol=1e-6,
    )
    lower = m.param_cb("coef_0", bound="lower")
    assert lower == pytest.approx(
        m.beta[0] - 1.644854 * m.standard_errors()[0], rel=1e-6
    )
    boot = m.param_cb(method="bootstrap", n_boot=40, random_state=1)
    assert np.all(boot[:, 0] < m.beta) and np.all(m.beta < boot[:, 1])
    with pytest.raises(ValueError, match="no likelihood"):
        m.param_cb("coef_0", method="lr")
    table = m.summary()
    np.testing.assert_allclose(table["se(coef)"], m.standard_errors())
    with pytest.raises(NoLikelihoodError):
        m.aic()


def test_refusals():
    x, Z, c, _ = _data(30, 7)
    with pytest.raises(ValueError, match="rank_weights"):
        RankAFT.fit(x, Z, c=c, rank_weights="peto")
    with pytest.raises(ValueError, match="positive"):
        RankAFT.fit(np.r_[0.0, x[1:]], Z, c=c)
    with pytest.raises(ValueError, match="at least one event"):
        RankAFT.fit(x, Z, c=np.ones(30))
    with pytest.raises(ValueError, match="right-censored"):
        RankAFT.fit(x, Z, c=np.r_[-1, c[1:]])


def test_aliased_column():
    x, Z, c, _ = _data(60, 8)
    Z3 = np.column_stack([Z, np.ones(60)])
    with pytest.warns(UserWarning, match="cannot be estimated"):
        m = RankAFT.fit(x, Z3, c=c)
    assert np.isnan(m.beta[2]) and np.isnan(m.standard_errors()[2])
    np.testing.assert_allclose(m.beta[:2], RankAFT.fit(x, Z, c=c).beta)


def test_fit_from_df_and_serialisation():
    x, Z, c, _ = _data(80, 9)
    df = pd.DataFrame({"x": x, "c": c, "a": Z[:, 0], "b": Z[:, 1]})
    m = RankAFT.fit_from_df(df, "x", ["a", "b"], c_col="c")
    np.testing.assert_allclose(m.beta, RankAFT.fit(x, Z, c=c).beta)
    assert m.parameter_names == ["a", "b"]
    back = RankAFTModel.from_json(m.to_json())
    np.testing.assert_allclose(back.beta, m.beta)
    np.testing.assert_allclose(back.covariance(), m.covariance())
    np.testing.assert_allclose(
        back.sf([2.0, 5.0], df[["a", "b"]].iloc[:2]),
        m.sf([2.0, 5.0], df[["a", "b"]].iloc[:2]),
    )


# -- R -----------------------------------------------------------------------


@pytest.mark.skipif(
    not REFERENCE.exists(),
    reason="the R rank-AFT reference file has not been generated yet",
)
def test_against_r_rank_aft():  # pragma: no cover - needs the file
    refs = json.loads(REFERENCE.read_text())
    for name, ref in refs["cases"].items():
        model = RankAFT.fit(
            np.array(ref["x"]),
            np.array(ref["Z"]),
            c=1 - np.array(ref["delta"]),
            rank_weights=ref["rank_weights"],
        )
        # R reports the textbook slope; surpyval its negative.
        np.testing.assert_allclose(
            -model.beta, ref["coef"], rtol=1e-3, err_msg=name
        )
        np.testing.assert_allclose(
            model.standard_errors(), ref["se"], rtol=5e-2, err_msg=name
        )
