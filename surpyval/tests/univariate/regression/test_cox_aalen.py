"""The Cox-Aalen model (multiplicative Cox terms on an additive Aalen
baseline).

R's ``timereg::cox.aalen`` (2.0.7) is the reference on two cases,
written by ``scripts/reference/reference_r_cox_aalen.R`` to
``tests/reference/data/cox_aalen_r.json`` (``test_against_r_timereg``):
the coefficients and cumulative coefficients agree to 1e-12 relative,
the standard errors to 0.4% (timereg's own variance estimator). The model
is also checked from first principles:

- with the baseline alone additive it is the package's ``CoxPH`` with
  Breslow ties: coefficients, model-based and robust standard errors,
  predicted cumulative hazards and their bounds, with ties and with
  delayed entry;
- with every covariate additive it is the package's ``AalenAdditive``:
  the cumulative coefficients and their variances;
- the score, written out again from the paper with explicit matrices at
  each event time, is zero at the estimate, and its derivative is the
  one Newton's method uses;
- seeded simulation recovers both parts with coverage
  (``calibration/test_coverage_cox_aalen.py``).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surpyval import AalenAdditive, CoxAalen, CoxPH
from surpyval.datasets import load_rossi_static
from surpyval.univariate.regression import CoxAalenModel
from surpyval.univariate.regression.cox_aalen import cox_aalen as ca
from surpyval.univariate.regression.regression_data import NoLikelihoodError

REFERENCE = (
    Path(__file__).parents[2] / "reference" / "data" / "cox_aalen_r.json"
)


def _rossi():
    df = load_rossi_static()
    x = df["week"].to_numpy(dtype=float)
    c = 1 - df["arrest"].to_numpy()
    Z = df[["fin", "age", "prio"]].to_numpy(dtype=float)
    return x, c, Z


def _simulate(N, seed, beta=0.7, entry=False):
    """``h = (0.2 + 0.3 a 1(t < 2)) exp(beta z)``: a binary ``z`` acting
    multiplicatively and a uniform ``a`` adding to the hazard until 2.
    Censored at a uniform time on (2, 10); with ``entry``, units enter at
    a uniform time on (0, 1) and only those still alive are kept."""
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, N).astype(float)
    a = rng.uniform(0, 1, N)
    e = rng.exponential(size=N) * np.exp(-beta * z)
    early = 0.2 + 0.3 * a
    t = np.where(e < 2 * early, e / early, 2 + (e - 2 * early) / 0.2)
    end = rng.uniform(2, 10, N)
    x, c = np.minimum(t, end), (t > end).astype(int)
    tl = np.zeros(N)
    if entry:
        tl = rng.uniform(0, 1, N)
        keep = x > tl
        x, c, tl, z, a = x[keep], c[keep], tl[keep], z[keep], a[keep]
    return x, c, tl, np.column_stack([z, a])


# -- the two parent models ---------------------------------------------------


def test_baseline_alone_additive_is_cox_breslow():
    x, c, Z = _rossi()
    model = CoxAalen.fit(x, Z, c=c)
    cox = CoxPH.fit(x, Z, c=c, tie_method="breslow")
    np.testing.assert_allclose(model.beta, cox.params, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        model.standard_errors(), cox.standard_errors(), rtol=1e-8
    )
    np.testing.assert_allclose(
        model.robust_covariance(), cox.robust_covariance(), rtol=1e-8
    )
    t = np.array([1.0, 5.0, 20.0, 52.0])
    for z in ([0, 25, 3], [1, 30, 0]):
        np.testing.assert_allclose(model.Hf(t, z), cox.Hf(t, z), rtol=1e-8)
        np.testing.assert_allclose(
            model.cb(t, z, on="Hf"), cox.cb(t, z, on="Hf"), rtol=1e-7
        )
        np.testing.assert_allclose(
            model.cb(t, z, bound_type="log"),
            cox.cb(t, z, bound_type="log"),
            rtol=1e-7,
        )


def test_baseline_alone_additive_is_cox_with_delayed_entry():
    x, c, tl, Z = _simulate(400, 1, entry=True)
    x = np.round(x, 1)  # ties
    tl = np.minimum(np.round(tl, 1), x - 0.05)
    model = CoxAalen.fit(x, Z, c=c, tl=tl)
    cox = CoxPH.fit(x, Z, c=c, tl=tl, tie_method="breslow")
    np.testing.assert_allclose(model.beta, cox.params, rtol=0, atol=1e-10)
    np.testing.assert_allclose(model.covariance(), cox.covariance(), rtol=1e-8)
    t = np.array([0.5, 2.0, 6.0])
    np.testing.assert_allclose(
        model.Hf(t, [1.0, 0.5]), cox.Hf(t, [1.0, 0.5]), rtol=1e-8
    )
    np.testing.assert_allclose(
        model.cb(t, [1.0, 0.5]), cox.cb(t, [1.0, 0.5]), rtol=1e-7
    )


# The second data set's fit stops before its last event (one is left).
@pytest.mark.filterwarnings("ignore:CoxAalen stopped at")
def test_every_covariate_additive_is_aalen():
    for data in (_rossi(), _simulate(300, 2, entry=True)):
        if len(data) == 3:
            x, c, Z = data
            tl = None
        else:
            x, c, tl, Z = data
        model = CoxAalen.fit(x, Z, c=c, tl=tl, additive=range(Z.shape[1]))
        aalen = AalenAdditive.fit(x, Z, c=c, tl=tl)
        assert model.beta.size == 0
        np.testing.assert_allclose(model.x, aalen.x)
        np.testing.assert_allclose(model.A, aalen.B, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.diagonal(model._A_cov, axis1=1, axis2=2),
            aalen.B_var,
            rtol=1e-10,
            atol=1e-15,
        )
        t = np.array([1.0, 3.0, 8.0])
        np.testing.assert_allclose(
            model.sf(t, Z[0]), aalen.sf(t, Z[0]), rtol=1e-12
        )


# -- the estimating equations, written out again -----------------------------


def _paper_score(beta, x, c, n, tl, X, Z, times):
    """``U(beta) = sum_t Z'(I - W X (X'WX)^-1 X') dN(t)`` with explicit
    matrices over the units at risk at each event time (Scheike and
    Zhang 2002)."""
    U = np.zeros(Z.shape[1])
    for t in times:
        R = (tl < t) & (x >= t)
        W = np.diag(n[R] * np.exp(Z[R] @ beta))
        XR, ZR = X[R], Z[R]
        dN = n[R] * ((x[R] == t) & (c[R] == 0))
        proj = W @ XR @ np.linalg.inv(XR.T @ W @ XR) @ XR.T
        U += ZR.T @ (np.eye(R.sum()) - proj) @ dN
    return U


def test_the_score_from_the_paper_is_zero_at_the_estimate():
    x, c, tl, Z = _simulate(250, 3, entry=True)
    model = CoxAalen.fit(x, Z, c=c, tl=tl, additive=[1])
    X = np.column_stack([np.ones(x.size), Z[:, 1]])
    Zm = Z[:, [0]]
    n = np.ones(x.size)
    U = _paper_score(model.beta, x, c, n, tl, X, Zm, model.x)
    # Its scale: the score's standard deviation, sqrt(Sigma).
    scale = 1.0 / model.standard_errors()
    assert np.all(np.abs(U) < 1e-8 * scale)
    # Away from the estimate it is not zero, and the package's score
    # agrees with the explicit one there too.
    off = model.beta + 2 * model.standard_errors()
    np.testing.assert_allclose(
        ca.score(off, x, c, n, tl, X, Zm, model.x),
        _paper_score(off, x, c, n, tl, X, Zm, model.x),
        rtol=1e-9,
    )
    assert np.all(np.abs(_paper_score(off, x, c, n, tl, X, Zm, model.x)) > 1)


def test_the_derivative_is_the_scores():
    x, c, tl, Z = _simulate(250, 4)
    Z = np.column_stack([Z, np.random.default_rng(0).normal(size=x.size)])
    X = np.column_stack([np.ones(x.size), Z[:, 1]])
    Zm = Z[:, [0, 2]]
    n = np.ones(x.size)
    times = ca._event_times(x, c, n, tl, X)
    beta = np.array([0.5, -0.2])
    I_ = ca._pieces(x, c, n, tl, X, Zm, beta, times)["I"]
    h = 1e-6
    numeric = np.column_stack(
        [
            -(
                _paper_score(beta + h * e, x, c, n, tl, X, Zm, times)
                - _paper_score(beta - h * e, x, c, n, tl, X, Zm, times)
            )
            / (2 * h)
            for e in np.eye(2)
        ]
    )
    np.testing.assert_allclose(I_, numeric, rtol=1e-6)


# -- the API -----------------------------------------------------------------


def test_negative_additive_hazard_is_held():
    # A protective additive covariate whose estimated effect makes the
    # additive part fall for a unit with a large value of it.
    x, c, _, Z = _simulate(500, 5)
    model = CoxAalen.fit(x, Z, c=c, additive=[1])
    row = [0.0, 5.0]
    raw = model.cumulative_coefficients(model.x).to_numpy() @ [1.0, 5.0]
    assert np.any(np.diff(raw) < 0)
    H = model.Hf(model.x, row)
    np.testing.assert_allclose(
        H, np.maximum.accumulate(np.maximum(raw, 0.0)), rtol=1e-12
    )
    assert np.all(np.diff(model.sf(model.x, row)) <= 0)


def test_predictions_and_bounds():
    x, c, _, Z = _simulate(400, 6)
    model = CoxAalen.fit(x, Z, c=c, additive=[1])
    t = np.array([0.5, 1.0, 3.0, 6.0])
    sf = model.sf(t, [1.0, 0.5])
    np.testing.assert_allclose(model.ff(t, [1.0, 0.5]), 1 - sf)
    band = model.cb(t, [1.0, 0.5])
    assert band.shape == (4, 2)
    assert np.all((band[:, 0] <= sf) & (sf <= band[:, 1]))
    q = model.qf(0.3, [1.0, 0.5])
    assert model.ff(q, [1.0, 0.5]) >= 0.3
    draws, rows = model.random(5, [[1.0, 0.5]], random_state=1)
    assert draws.shape == (5,) and rows.shape == (5, 2)
    A_band = model.cumulative_coefficients_cb([1.0, 3.0])
    A = model.cumulative_coefficients([1.0, 3.0]).to_numpy()
    assert np.all((A_band[..., 0] < A) & (A < A_band[..., 1]))
    lo, hi = model.param_cb("coef_0")
    assert model.param_cb().shape == (1, 2)
    assert lo < model.beta[0] < hi
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.param_cb("coef_0", method="lr")
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(1.0, [1.0, 0.5], method="bootstrap")
    with pytest.raises(NoLikelihoodError):
        model.aic()


def test_counts_equal_repeated_rows():
    x, c, _, Z = _simulate(120, 7)
    x = np.round(x, 0) + 1
    rng = np.random.default_rng(2)
    n = rng.integers(1, 4, x.size)
    a = CoxAalen.fit(x, Z, c=c, n=n, additive=[1])
    b = CoxAalen.fit(
        np.repeat(x, n),
        np.repeat(Z, n, axis=0),
        c=np.repeat(c, n),
        additive=[1],
    )
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.covariance(), b.covariance(), rtol=1e-9)
    np.testing.assert_allclose(a.A, b.A, rtol=1e-9, atol=1e-14)


def test_fit_from_df_names_and_serialisation():
    df = load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    model = CoxAalen.fit_from_df(
        df,
        "week",
        ["fin", "age", "prio"],
        c_col="censored",
        additive=["fin"],
    )
    assert model.parameter_names == ["age", "prio"]
    assert model.term_names == ["baseline", "fin"]
    x, c, Z = _rossi()
    same = CoxAalen.fit(x, Z, c=c, additive=[0])
    np.testing.assert_allclose(model.beta, same.beta)
    frame = CoxAalen.fit(
        x,
        pd.DataFrame(Z, columns=["fin", "age", "prio"]),
        c=c,
        additive="fin",
    )
    np.testing.assert_allclose(frame.beta, same.beta)
    back = CoxAalenModel.from_json(model.to_json())
    rows = df[["fin", "age", "prio"]].iloc[:3]
    np.testing.assert_allclose(back.sf(20.0, rows), model.sf(20.0, rows))
    np.testing.assert_allclose(back.cb(20.0, rows), model.cb(20.0, rows))
    np.testing.assert_allclose(back.covariance(), model.covariance())


def test_refusals_and_aliasing():
    x, c, _, Z = _simulate(80, 8)
    with pytest.raises(ValueError, match="additive"):
        CoxAalen.fit(x, Z, c=c, additive=[5])
    with pytest.raises(ValueError, match="additive"):
        CoxAalen.fit(x, Z, c=c, additive=["a"])
    with pytest.raises(ValueError, match="right-censored"):
        CoxAalen.fit(x, Z, c=np.r_[2, c[1:]])
    with pytest.raises(ValueError, match="at least one event"):
        CoxAalen.fit(x, Z, c=np.ones(x.size))
    Z3 = np.column_stack([Z, np.ones(x.size)])
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = CoxAalen.fit(x, Z3, c=c, additive=[1])
    assert np.isnan(model.beta[1])
    np.testing.assert_allclose(
        model.beta[0], CoxAalen.fit(x, Z, c=c, additive=[1]).beta[0]
    )


def _stops_early(seed=0, N=400):
    """An additive binary ``a`` whose ``a = 1`` units have all left by
    t = 3 while the others fail on to t = 10: past 3, ``X'WX`` is
    singular. The hazard is ``0.18 exp(0.3 z)`` (``a`` has no effect),
    so the truth at t = 9 for ``z = 0`` is ``exp(-1.62) = 0.198``."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.5, N).astype(float)
    z = rng.normal(size=N)
    t = rng.exponential(1 / (0.18 * np.exp(0.3 * z)), N)
    end = np.where(a == 1, rng.uniform(0.5, 3, N), 10.0)
    return np.minimum(t, end), (t > end).astype(int), np.c_[z, a]


def test_a_fit_that_stops_early_warns_and_predicts_nan_beyond():
    x, c, Z = _stops_early()
    with pytest.warns(
        UserWarning, match=r"stopped at t = 2\.98.*later events"
    ):
        model = CoxAalen.fit(x, Z, c=c, additive=[1])
    stop = model.stopped_at
    assert stop == model.x[-1] < x[c == 0].max()
    t = np.array([1.0, stop, 9.0])
    for out in (model.sf(t, [0, 0]), model.ff(t, [0, 0]), model.Hf(t, [0, 0])):
        assert np.all(np.isfinite(out[:2])) and np.isnan(out[2])
    assert np.isnan(model.cb(t, [0, 0])[2]).all()
    assert np.isfinite(model.cb(t, [0, 0])[:2]).all()
    assert np.isnan(model.qf(0.9, [0, 0]))
    assert np.isnan(model.cumulative_coefficients([9.0]).to_numpy()).all()
    assert np.isnan(model.cumulative_coefficients_cb([9.0])).all()
    back = CoxAalenModel.from_json(model.to_json())
    assert back.stopped_at == stop and np.isnan(back.sf(9.0, [0, 0]))
    # A fit that does not stop says so with nan, and predicts after its
    # last event time as before (held).
    x2, c2, _, Z2 = _simulate(200, 9)
    full = CoxAalen.fit(x2, Z2, c=c2, additive=[1])
    assert np.isnan(full.stopped_at)
    assert np.isfinite(full.sf(1e3, [1.0, 0.5]))


# -- R -----------------------------------------------------------------------


def test_against_r_timereg():
    refs = json.loads(REFERENCE.read_text())
    assert set(refs["cases"]) == {"lung-sex-additive", "simulated"}
    for name, ref in refs["cases"].items():
        Z = np.array(ref["Z"], dtype=float)
        model = CoxAalen.fit(
            np.array(ref["x"]),
            Z,
            c=1 - np.array(ref["delta"]),
            additive=ref["additive"],
        )
        assert np.isnan(model.stopped_at), name
        np.testing.assert_allclose(
            model.beta, ref["gamma"], rtol=1e-10, err_msg=name
        )
        # timereg's var.gamma is its own estimator of the same variance:
        # 0.14% (lung) and 0.34% (simulated) from covariance().
        np.testing.assert_allclose(
            model.standard_errors(), ref["se_gamma"], rtol=1e-2, err_msg=name
        )
        np.testing.assert_allclose(
            model.cumulative_coefficients(ref["times"]).to_numpy(),
            np.array(ref["cum"]),
            rtol=1e-10,
            atol=1e-12,
            err_msg=name,
        )
