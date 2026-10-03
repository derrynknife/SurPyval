"""Cox residuals and the proportional-hazards test (#211).

The residual identities are exact at the MLE and are the strongest
correctness checks available: Schoenfeld, score and martingale residuals each
sum to (approximately) zero because the fitted ``beta`` solves the score
equation. The proportional-hazards test is additionally checked for **power**
(it rejects a genuine time-varying-coefficient violation) and **calibration**
(under true proportional hazards its p-values are ~Uniform).
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval import CoxPH
from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    check_ph,
    compute_residuals,
)


def _ph_data(seed=0, n=200, censor=0.25):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.8 * Z[:, 0] - 0.5 * Z[:, 1]
    x = (rng.exponential(1.0, n) / np.exp(lin)) ** (1 / 1.5)
    c = (rng.random(n) < censor).astype(int)
    return x, Z, c


# -- residual identities at the MLE -----------------------------------------


def test_schoenfeld_residuals_sum_to_zero():
    x, Z, c = _ph_data()
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    sch = m.compute_residuals("schoenfeld")
    assert sch.shape == ((c == 0).sum(), Z.shape[1])
    np.testing.assert_allclose(sch.sum(axis=0), 0.0, atol=1e-6)


def test_score_residuals_sum_to_score_zero():
    x, Z, c = _ph_data(seed=1)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    score = m.compute_residuals("score")
    assert score.shape == Z.shape
    # The score residuals sum to the total score, which is zero at the MLE.
    np.testing.assert_allclose(score.sum(axis=0), 0.0, atol=1e-5)


def test_martingale_residuals_sum_to_zero_and_bounded():
    x, Z, c = _ph_data(seed=2)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    mart = m.compute_residuals("martingale")
    assert mart.shape == (len(x),)
    assert abs(float(mart.sum())) < 1e-6
    # Martingale residuals lie in (-inf, 1].
    assert mart.max() <= 1.0 + 1e-9


def test_deviance_residuals_finite_and_symmetrising():
    x, Z, c = _ph_data(seed=3)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    dev = m.compute_residuals("deviance")
    mart = m.compute_residuals("martingale")
    assert np.all(np.isfinite(dev))
    # Deviance residuals share the martingale's sign and are more symmetric.
    assert np.all(np.sign(dev[mart != 0]) == np.sign(mart[mart != 0]))
    assert abs(float(dev.mean())) < abs(float(mart.mean())) + 0.5


def test_dfbeta_shape_and_scale():
    x, Z, c = _ph_data(seed=4)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    dfb = m.compute_residuals("dfbeta")
    assert dfb.shape == Z.shape
    # No single observation should dominate the coefficient (well-behaved data)
    assert np.all(np.abs(dfb).max(axis=0) < np.abs(m.beta) + 1.0)


def test_scaled_schoenfeld_centres_on_beta():
    x, Z, c = _ph_data(seed=5)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    scaled = m.compute_residuals("scaled_schoenfeld")
    # Under proportional hazards the scaled residuals fluctuate about beta.
    np.testing.assert_allclose(scaled.mean(axis=0), m.beta, atol=0.15)


# -- residuals respect delayed entry ----------------------------------------


def test_residuals_respect_left_truncation():
    rng = np.random.default_rng(6)
    n = 200
    Z = rng.normal(0, 1, (n, 1))
    x = (rng.exponential(1.0, n) / np.exp(0.6 * Z[:, 0])) ** (1 / 1.4)
    tl = np.minimum(rng.uniform(0, 0.3, n), x * 0.5)
    c = (rng.random(n) < 0.2).astype(int)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c, tl=tl)
    # Martingale residuals use the fitted Breslow baseline directly, so they
    # sum to zero exactly.
    assert abs(float(m.compute_residuals("martingale").sum())) < 1e-6
    # Score residuals use exact risk-set membership ({tl < tau <= x}); the
    # fitted beta comes from a truncated partial likelihood that buckets
    # truncation times onto the event-time grid, so under truncation the sum
    # is O(grid error) rather than machine zero -- small relative to the
    # O(1) per-observation residual scale, not a residual bug.
    score = m.compute_residuals("score")
    rms = float(np.sqrt((score**2).mean()))
    assert abs(float(score.sum(axis=0)[0])) < 0.1 * rms * np.sqrt(n)


# -- the proportional-hazards test ------------------------------------------


def test_ph_test_does_not_reject_true_ph():
    x, Z, c = _ph_data(seed=7)
    ph = sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph()
    assert ph.loc["GLOBAL", "df"] == Z.shape[1]
    assert ph.loc["GLOBAL", "p"] > 0.05
    assert len(ph) == Z.shape[1] + 1


def test_ph_test_detects_violation():
    # Z0 drives only the early half of the events -> a time-varying effect.
    rng = np.random.default_rng(8)
    n = 400
    Z = rng.normal(0, 1, (n, 1))
    u = rng.random(n)
    x = np.where(
        u < 0.5,
        np.exp(-1.5 * Z[:, 0]) * rng.exponential(1, n),
        rng.exponential(5, n),
    )
    c = np.zeros(n)
    ph = sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph()
    assert ph.loc["GLOBAL", "p"] < 0.01


def test_ph_test_is_calibrated_under_null():
    # Under true proportional hazards the global p-value is ~Uniform(0, 1);
    # the rejection rate at 0.05 should be close to 0.05 (not systematically
    # small, which would indicate a mis-scaled statistic).
    def one(seed):
        x, Z, c = _ph_data(seed=1000 + seed, n=150, censor=0.2)
        return sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph().loc["GLOBAL", "p"]

    pvals = np.array([one(s) for s in range(80)])
    assert 0.0 <= pvals.min() and pvals.max() <= 1.0
    assert (pvals < 0.05).mean() < 0.15  # not over-rejecting
    assert 0.35 < pvals.mean() < 0.65  # centred near 0.5


@pytest.mark.parametrize("transform", ["km", "rank", "identity", "log"])
def test_ph_test_transforms_run(transform):
    x, Z, c = _ph_data(seed=9)
    ph = sp.CoxPH.fit(x=x, Z=Z, c=c).check_ph(transform=transform)
    assert ph.attrs["transform"] == transform
    assert np.isfinite(ph.loc["GLOBAL", "statistic"])


# -- guards ------------------------------------------------------------------


def test_unknown_residual_kind_raises():
    x, Z, c = _ph_data(seed=10)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    with pytest.raises(ValueError, match="'kind' must be one of"):
        m.compute_residuals("bogus")


def test_unknown_transform_raises():
    x, Z, c = _ph_data(seed=11)
    m = sp.CoxPH.fit(x=x, Z=Z, c=c)
    with pytest.raises(ValueError, match="transform"):
        m.check_ph(transform="bogus")


def test_per_covariate_names_from_dataframe():
    import pandas as pd

    rng = np.random.default_rng(12)
    n = 150
    df = pd.DataFrame(
        {
            "time": (rng.exponential(1.0, n)) ** (1 / 1.4),
            "temp": rng.normal(0, 1, n),
            "volt": rng.normal(0, 1, n),
        }
    )
    m = sp.CoxPH.fit_from_df(df, x_col="time", Z_cols=["temp", "volt"])
    ph = m.check_ph()
    assert list(ph.index) == ["temp", "volt", "GLOBAL"]


def _ph_test_dataset():
    # Correlated covariates with a time-varying effect on the first, so
    # the per-covariate statistics genuinely differ between conventions.
    np.random.seed(11)
    n = 200
    Z = np.random.normal(size=(n, 2))
    Z[:, 1] = 0.5 * Z[:, 0] + np.random.normal(size=n)
    u = np.random.uniform(size=n)
    x = -np.log(u) / (0.1 * np.exp(0.8 * Z[:, 0]))
    cut = np.quantile(x, 0.8)
    c = (x > cut).astype(int)
    return np.minimum(x, cut), Z, c


def test_check_ph_matches_lifelines_convention():
    # Grambsch-Therneau per-covariate statistic d (Vu)_j^2 / (Sgc2 V_jj)
    # with the km transform as 1 - KM(t) (#262). Reference values computed
    # with lifelines 0.30.3 ``proportional_hazard_test(time_transform="km")``
    # on this exact dataset.
    x, Z, c = _ph_test_dataset()
    model = sp.CoxPH.fit(x=x, Z=Z, c=c)
    res = model.check_ph(transform="km")
    stats = res["statistic"].iloc[:2].tolist()
    assert stats[0] == pytest.approx(0.178375, abs=2e-4)
    assert stats[1] == pytest.approx(0.182189, abs=2e-4)
    pvals = res["p"].iloc[:2].tolist()
    assert pvals[0] == pytest.approx(0.672773, abs=2e-4)
    assert pvals[1] == pytest.approx(0.669498, abs=2e-4)


def test_check_ph_is_a_table_like_summary():
    # #514: check_ph returned a nested dict while summary() is a table;
    # it is now R's cox.zph table: a row per covariate and a GLOBAL row,
    # with the statistic, its degrees of freedom and the p-value. The
    # dict is still what the diagnostics function returns.
    import pandas as pd

    from surpyval.univariate.regression.proportional_hazards.diagnostics import (  # noqa: E501
        check_ph,
    )

    x, Z, c = _ph_test_dataset()
    model = sp.CoxPH.fit(x=x, Z=Z, c=c)
    table = model.check_ph()
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == ["statistic", "df", "p"]
    assert list(table.index) == ["coef_0", "coef_1", "GLOBAL"]
    assert table.index.name == "covariate"
    assert table["df"].tolist() == [1, 1, 2]
    assert table.attrs["transform"] == "km"
    res = check_ph(model)
    assert table.loc["GLOBAL", "statistic"] == res["global"]["statistic"]
    assert table.loc["GLOBAL", "p"] == res["global"]["p_value"]
    assert table["p"].iloc[:2].tolist() == [
        e["p_value"] for e in res["per_covariate"]
    ]


# -- the risk sums are computed in one pass (performance sweep) -------------


def _risk_sum_reference(model):
    # The per-event-time loop the residuals used, kept as the reference:
    # a mask of every row at each event time (O(n K)).
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        cox_at_risk_mask,
    )
    from surpyval.univariate.regression.proportional_hazards.diagnostics import (  # noqa: E501
        _require_cox,
    )

    data = _require_cox(model)
    x, c, n, Z, tl = (data[k] for k in ("x", "c", "n", "Z", "tl"))
    beta = np.asarray(model.beta, dtype=float)
    w = n * np.exp(Z @ beta)
    times = np.unique(x[c == 0])
    A, B, A_own, B_own, Zbar = [], [], [], [], []
    for tau in times:
        r = cox_at_risk_mask(x, tl, tau)
        s0, s1 = w[r].sum(), (w[r, None] * Z[r]).sum(axis=0)
        ev = (x == tau) & (c == 0)
        m = int(round(n[ev].sum()))
        if model.tie_method == "efron" and m > 1:
            f = np.arange(m) / m
            s0_l = s0 - f * w[ev].sum()
            e_l = (s1 - f[:, None] * (w[ev, None] * Z[ev]).sum(axis=0)) / (
                s0_l[:, None]
            )
            Zbar.append(e_l.mean(axis=0))
            A.append((1 / s0_l).sum())
            B.append((e_l / s0_l[:, None]).sum(axis=0))
            A_own.append(((1 - f) / s0_l).sum())
            B_own.append(((1 - f)[:, None] * e_l / s0_l[:, None]).sum(axis=0))
        else:
            Zbar.append(s1 / s0)
            A.append(m / s0)
            B.append(s1 / s0 * m / s0)
            A_own.append(A[-1])
            B_own.append(B[-1])
    A, B, A_own, B_own, Zbar = map(np.array, (A, B, A_own, B_own, Zbar))
    score = np.zeros_like(Z)
    for i in range(len(x)):
        win = (times > tl[i]) & (times <= x[i])
        if not win.any():
            continue
        a, b = A[win].sum(), B[win].sum(axis=0)
        if c[i] == 0:
            k = np.searchsorted(times, x[i])
            a, b = a + A_own[k] - A[k], b + B_own[k] - B[k]
            score[i] += Z[i] - Zbar[k]
        score[i] -= np.exp(Z[i] @ beta) * (Z[i] * a - b)
    return Zbar, score * n[:, None]


@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_residuals_match_the_per_time_loop(tie_method):
    # Ties, delayed entry and counts: the one-pass risk sums give the
    # Schoenfeld and score residuals of the loop over the event times.
    rng = np.random.default_rng(3)
    n = 300
    Z = rng.normal(0, 1, (n, 2)) * [1.0, 4.0] + [0.0, 50.0]
    x = np.round(10 * rng.weibull(1.5, n) * np.exp(-0.3 * Z[:, 0]), 0) + 1
    c = (rng.random(n) < 0.3).astype(int)
    tl = np.where(rng.random(n) < 0.4, x * rng.uniform(0, 0.9, n), 0.0)
    counts = rng.integers(1, 4, n)
    model = sp.CoxPH.fit(x=x, Z=Z, c=c, n=counts, tl=tl, tie_method=tie_method)
    Zbar, score = _risk_sum_reference(model)
    events = np.flatnonzero(c == 0)
    times = np.unique(x[events])
    sch = model.compute_residuals("schoenfeld")
    Zc = Z - model._fit_center if model._fit_center is not None else Z
    expected_sch = Zc[events] - Zbar[np.searchsorted(times, x[events])]
    np.testing.assert_allclose(sch, expected_sch, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        model.compute_residuals("score"), score, rtol=1e-10, atol=1e-12
    )


def test_residuals_do_not_loop_over_event_times(monkeypatch):
    # The residuals used a mask of every row at each event time, O(n K):
    # dfbeta took 4.5 s on 1e4 rows with distinct times and 47 s on 3e4
    # (0.014 s and 0.046 s now). The risk sums now come from one pass.
    from surpyval.univariate.regression.proportional_hazards import (
        diagnostics as dg,
    )

    calls = []

    def counting(x, tl, tau):
        calls.append(tau)
        return (tl < tau) & (x >= tau)

    monkeypatch.setattr(dg, "cox_at_risk_mask", counting, raising=False)
    x, Z, c = _ph_data(n=500)
    model = sp.CoxPH.fit(x=x, Z=Z, c=c)
    for kind in ("martingale", "schoenfeld", "score", "dfbeta"):
        model.compute_residuals(kind)
    model.check_ph()
    assert calls == []


# ---------------------------------------------------------------------------
# #279: Efron tie corrections in the Cox residuals, ``check_ph``
# and the robust standard errors.
# ---------------------------------------------------------------------------


class TestEfronDiagnostics:
    @staticmethod
    def _tied_fit():
        np.random.seed(11)
        n = 120
        Z = np.column_stack(
            [np.random.binomial(1, 0.5, n), np.random.normal(size=n)]
        )
        u = np.random.uniform(size=n)
        t = -np.log(u) / (0.3 * np.exp(0.7 * Z[:, 0] - 0.4 * Z[:, 1]))
        x = np.ceil(np.clip(t, 0.5, 6)).astype(float)
        c = (np.random.uniform(size=n) < 0.2).astype(int)
        return x, c, Z

    def test_residual_sums_vanish_at_mle(self):
        # 279: these identities only hold when the residuals use the
        # same tie handling as the fitted likelihood.
        x, c, Z = self._tied_fit()
        for method in ("efron", "breslow"):
            m = CoxPH.fit(x=x, Z=Z, c=c, tie_method=method)
            assert compute_residuals(m, "martingale").sum() == pytest.approx(
                0.0, abs=1e-8
            )
            assert np.abs(
                compute_residuals(m, "score").sum(axis=0)
            ).max() == pytest.approx(0.0, abs=1e-8)

    def test_check_ph_matches_lifelines_under_ties(self):
        # Reference values from lifelines 0.30.3 on this exact dataset
        # (km transform; identity and log also agree — lifelines is not
        # a CI dependency, so the values are pinned).
        x, c, Z = self._tied_fit()
        m = CoxPH.fit(x=x, Z=Z, c=c, tie_method="efron")
        res = check_ph(m, transform="km")
        stats = [e["statistic"] for e in res["per_covariate"]]
        assert stats[0] == pytest.approx(1.3936, abs=2e-3)
        assert stats[1] == pytest.approx(0.0013, abs=2e-3)

    def test_dfbeta_tracks_exact_leave_one_out(self):
        x, c, Z = self._tied_fit()
        m = CoxPH.fit(x=x, Z=Z, c=c, tie_method="efron")
        dfb = compute_residuals(m, "dfbeta")
        # Spot-check 15 rows of exact leave-one-out influence.
        rows = np.arange(0, 120, 8)
        loo = np.zeros((rows.size, 2))
        for r, i in enumerate(rows):
            keep = np.ones(120, dtype=bool)
            keep[i] = False
            mi = CoxPH.fit(x=x[keep], Z=Z[keep], c=c[keep], tie_method="efron")
            loo[r] = m.beta - mi.beta
        corr = np.corrcoef(dfb[rows, 0], loo[:, 0])[0, 1]
        assert corr > 0.99
