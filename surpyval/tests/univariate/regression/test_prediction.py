"""Quantiles and conditional survival of the regression models.

``qf(p, Z)`` (#571) inverts each parametric regression family from its
own cumulative hazard; ``cs(x, given, Z)`` (#581) is the univariate
models' conditional survival with covariates, on every regression model
with ``Hf(x, Z)``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.life_models import Power

P = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.999999])
ZQ = np.array([[0, 0.2], [1, 0.9], [0, 0.5], [1, 0.1], [0, 0.0], [1, 1.0]])


def _data(seed=3, n=150):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.binomial(1, 0.5, n), rng.uniform(0, 1, n)])
    scale = np.exp(-0.4 * Z[:, 0] - 0.3 * Z[:, 1])
    x = 10 * rng.weibull(2.0, n) * scale + 0.5
    c = (rng.uniform(size=n) < 0.2).astype(int)
    return x, Z, c


FAMILIES = [
    f"{dist}{kind}"
    for kind in ("PH", "AFT", "PO", "AH")
    for dist in ("Weibull", "LogNormal", "Normal", "Gumbel", "Exponential")
]


@pytest.mark.parametrize("name", FAMILIES)
def test_571_qf_inverts_ff_for_every_family(name):
    # A regression model had no qf: "B10 at high load" meant a root
    # search on sf by hand (#571).
    x, Z, c = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, name).fit(x, Z, c=c)
        q = model.qf(P, ZQ)
        ff, sf = model.ff(q, ZQ), model.sf(q, ZQ)
    # Relative to the smaller tail, where the probability is resolved: ff
    # below 1/2, sf above it. Newton's finish (#828) takes every family to
    # rounding; an additive model's H = H0 + x beta'Z cancels near the
    # start of its support, so it keeps fewer digits there.
    upper = P > 0.5
    err = np.where(upper, np.abs(sf - (1 - P)) / (1 - P), np.abs(ff - P) / P)
    tol = 1e-8 if name.endswith("AH") else 1e-12
    np.testing.assert_array_less(err, tol)


@pytest.mark.parametrize("name", ["NormalAH", "GumbelAH", "LogisticAH"])
def test_828_qf_is_exact_near_the_start_of_the_support(name):
    # The issue's data: the quantiles of a small p are just above where
    # the support starts (x* < 0 for beta'Z > 0 on these baselines). The
    # solver's tolerance, relative to |t|, left sf(qf(1 - s)) off by up to
    # 1.8e-5 of s; Newton's finish on H leaves the cancellation in
    # H0 + x beta'Z there, below 1e-8 of it.
    rng = np.random.default_rng(0)
    x = rng.weibull(1.5, 300) * 100
    Z = rng.uniform(0, 1, (300, 1))
    x = x * np.exp(0.5 * Z[:, 0])
    s = np.logspace(-9, np.log10(0.5), 40)
    z = np.repeat([[0.5]], s.size, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, name).fit(x, Z=Z)
        low = model.ff(model.qf(s, z), z)
        high = model.sf(model.qf(1 - s, z), z)
    assert np.max(np.abs(low - s) / s) < 1e-8
    assert np.max(np.abs(high - (1 - (1 - s))) / s) < 1e-8


def test_571_qf_matches_the_closed_forms():
    x, Z, c = _data()
    aft = sp.WeibullAFT.fit(x, Z, c=c)
    alpha, beta, b0, b1 = aft.params
    phi = np.exp(ZQ @ np.array([b0, b1]))
    closed = alpha * (-np.log1p(-P)) ** (1 / beta) / phi
    np.testing.assert_allclose(aft.qf(P, ZQ), closed, rtol=1e-11)
    # Proportional hazards: S0(t) ** phi = 1 - p.
    ph = sp.WeibullPH.fit(x, Z, c=c)
    alpha, beta, b0, b1 = ph.params
    phi = np.exp(ZQ @ np.array([b0, b1]))
    closed = alpha * (-np.log1p(-P) / phi) ** (1 / beta)
    np.testing.assert_allclose(ph.qf(P, ZQ), closed, rtol=1e-11)
    # An accelerated life model: the distribution's own qf at the life.
    rng = np.random.default_rng(0)
    s = rng.choice([1.0, 2.0, 3.0], 150)
    al = sp.AcceleratedLife(sp.Weibull, Power).fit(
        100 / s * rng.weibull(3, 150), s
    )
    q = al.qf(P, s[:6])
    np.testing.assert_array_less(
        np.abs(al.ff(q, s[:6]) - P) / np.minimum(P, 1 - P), 1e-10
    )


def test_571_qf_pairs_rows_like_sf():
    x, Z, c = _data()
    model = sp.WeibullPH.fit(x, Z, c=c)
    grid = model.qf(P, ZQ[:2], grid=True)
    assert grid.shape == (2, 6)
    np.testing.assert_allclose(grid[1], model.qf(P, ZQ[1]), rtol=1e-12)
    assert model.qf(P, ZQ[:1]).shape == (6,)
    assert model.qf(0.5, ZQ).shape == (6,)
    assert np.ndim(model.qf(0.5, ZQ[0])) == 0
    with pytest.raises(ValueError, match="covariate rows"):
        model.qf(P, ZQ[:4])
    # The ends of the support, a missing probability, and a mistake.
    np.testing.assert_array_equal(
        model.qf([0.0, 1.0, np.nan], ZQ[:3]), [0.0, np.inf, np.nan]
    )
    with pytest.warns(UserWarning, match="outside"):
        assert np.isnan(model.qf(10.0, ZQ[0]))
    normal = sp.NormalAFT.fit(x, Z, c=c)
    np.testing.assert_array_equal(normal.qf([0, 1], ZQ[:2]), [-np.inf, np.inf])


def test_571_qf_reads_data_frames_and_the_centre():
    x, Z, c = _data()
    df = pd.DataFrame({"x": x, "c": c, "a": Z[:, 0], "b": Z[:, 1] + 1000})
    model = sp.WeibullPH.fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["a", "b"], center=True
    )
    new = pd.DataFrame({"b": ZQ[:, 1] + 1000, "a": ZQ[:, 0]})
    q = model.qf(P, new)
    np.testing.assert_allclose(model.ff(q, new), P, rtol=1e-9)


def _cs_models():
    x, Z, c = _data()
    z1 = Z[:, :1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "WeibullPH": sp.WeibullPH.fit(x, Z, c=c),
            "LogNormalAFT": sp.LogNormalAFT.fit(x, Z, c=c),
            "CoxPH": sp.CoxPH.fit(x, Z, c=c),
            "ProportionalOdds": sp.ProportionalOdds.fit(x, Z, c=c),
            "AdditiveHazards": sp.AdditiveHazards.fit(x, z1, c=c),
            "BuckleyJames": sp.BuckleyJames.fit(x, Z, c=c),
        }


@pytest.mark.parametrize("name", list(_cs_models()))
def test_581_cs_is_the_ratio_of_survival(name):
    # Regression models had no cs; "P(failure within a year | survived to
    # its age, its covariates)" was 1 - sf(a + 1, Z) / sf(a, Z) by hand.
    model = _cs_models()[name]
    k = 1 if name == "AdditiveHazards" else 2
    rows = ZQ[:3, :k]
    ages = np.array([1.0, 4.0, 7.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = model.cs(1.5, ages, rows)
        ratio = model.sf(ages + 1.5, rows) / model.sf(ages, rows)
    np.testing.assert_allclose(got, ratio, rtol=1e-9)
    assert np.ndim(model.cs(1.5, 4.0, rows[0])) == 0


def test_581_cs_stays_exact_where_survival_underflows():
    x, Z, c = _data()
    model = sp.WeibullPH.fit(x, Z, c=c)
    row = ZQ[1]
    age = 10.0
    while model.sf(age, row) > 0:
        age *= 1.5
    with np.errstate(all="ignore"):
        ratio = model.sf(age + 0.1, row) / model.sf(age, row)
    assert np.isnan(ratio)  # 0 / 0
    H = model.Hf([age, age + 0.1], row)
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(
        model.cs(0.1, age, row), np.exp(H[0] - H[1]), rtol=1e-12
    )
    assert 0 < model.cs(0.1, age, row) < 1


def test_581_cs_grid_and_cox_strata():
    x, Z, c = _data()
    model = sp.WeibullPH.fit(x, Z, c=c)
    g = model.cs([1.0, 2.0], [3.0, 5.0], ZQ[:3], grid=True)
    assert g.shape == (3, 2)
    np.testing.assert_allclose(g[2], model.cs([1.0, 2.0], [3.0, 5.0], ZQ[2]))
    strata = np.arange(len(x)) % 2
    cox = sp.CoxPH.fit(x, Z, c=c, strata=strata)
    got = cox.cs(1.0, 3.0, ZQ[0], stratum=1)
    ratio = cox.sf(4.0, ZQ[0], stratum=1) / cox.sf(3.0, ZQ[0], stratum=1)
    np.testing.assert_allclose(got, ratio, rtol=1e-12)


# ---------------------------------------------------------------------------
# #662: qf on the semi-parametric and frailty models, mean(Z)
# ---------------------------------------------------------------------------


def _grouped(n=150):
    x, Z, c = _data(n=n)
    return x, Z, c, np.arange(n) % 15


def _step_models():
    x, Z, c, g = _grouped()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "CoxPH": (sp.CoxPH.fit(x, Z, c=c), {}),
            "ProportionalOdds": (sp.ProportionalOdds.fit(x, Z, c=c), {}),
            "BuckleyJames": (sp.BuckleyJames.fit(x, Z, c=c), {}),
            "AdditiveHazards": (sp.AdditiveHazards.fit(x, Z, c=c), {}),
            "CoxFrailty": (sp.CoxFrailty.fit(x, Z, c=c, groups=g), {}),
            "CoxFrailty[group]": (
                sp.CoxFrailty.fit(x, Z, c=c, groups=g),
                {"group": 3},
            ),
        }


@pytest.mark.parametrize("name", list(_step_models()))
def test_662_qf_is_the_first_time_the_curve_reaches_p(name):
    # CoxPH, ProportionalOdds, BuckleyJames, AdditiveHazards and CoxFrailty
    # had no qf (AttributeError); the changelog said regression models do.
    model, extra = _step_models()[name]
    p = np.array([0.0, 0.05, 0.3, 0.5])
    q = model.qf(p, ZQ[:4], **extra)
    assert q.shape == (4,) and np.all(np.isfinite(q))
    F = model.ff(q, ZQ[:4], **extra)
    assert np.all(F >= p - 1e-9)
    # A step earlier the curve is short of p (a jump crossed it at q).
    before = model.ff(q * (1 - 1e-6), ZQ[:4], **extra)
    assert np.all(before[1:] < p[1:])
    # Never reached by the end of the curve: nan, as KaplanMeier.qf.
    end = model.ff(np.full(4, 1e6), ZQ[:4], **extra)
    beyond = model.qf(np.minimum(end + 0.005, 1.0), ZQ[:4], **extra)
    assert np.isnan(beyond[end < 0.99]).all()
    # Every model's qf rule (#611): outside [0, 1] is nan, with a warning.
    with pytest.warns(UserWarning, match="outside"):
        assert np.isnan(model.qf(1.5, ZQ[0], **extra))


def test_662_cox_qf_grid_strata_and_shape():
    x, Z, c = _data()
    model = sp.CoxPH.fit(x, Z, c=c)
    grid = model.qf([0.1, 0.3], ZQ[:3], grid=True)
    assert grid.shape == (3, 2)
    np.testing.assert_array_equal(grid[:, 1], model.qf(0.3, ZQ[:3]))
    assert np.ndim(model.qf(0.3, ZQ[0])) == 0
    strata = np.arange(len(x)) % 2
    st = sp.CoxPH.fit(x, Z, c=c, strata=strata)
    q = st.qf(0.3, ZQ[0], stratum=1)
    assert st.ff(q, ZQ[0], stratum=1) >= 0.3 - 1e-9
    with pytest.raises(ValueError, match="stratum"):
        st.qf(0.3, ZQ[0])


@pytest.mark.parametrize("name", ["WeibullFrailty", "LogNormalFrailty"])
def test_662_parametric_frailty_qf_inverts_ff(name):
    x, Z, c, g = _grouped()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(sp, name).fit(x, Z, c=c, groups=g)
    for extra in ({}, {"group": 3}, {"frailty": 2.0}):
        q = model.qf(P[1:-1], ZQ[1:-1], **extra)
        np.testing.assert_allclose(
            model.ff(q, ZQ[1:-1], **extra), P[1:-1], rtol=1e-10
        )


def test_662_mean_at_constant_covariates():
    from scipy.special import gamma

    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    x, Z, c = _data()
    model = sp.WeibullAFT.fit(x, Z, c=c)
    alpha, beta, b0, b1 = model.params
    scale = alpha / np.exp(ZQ[:3] @ np.array([b0, b1]))
    np.testing.assert_allclose(
        model.mean(ZQ[:3]), scale * gamma(1 + 1 / beta), rtol=1e-8
    )
    assert np.ndim(model.mean(ZQ[0])) == 0
    assert np.isnan(model.mean([np.nan, 0.5]))
    # An accelerated life model's MTTF is the distribution's closed form,
    # which mean_tvc reproduces.
    stress = np.repeat([20.0, 30.0, 40.0], 30)
    t = 1000 * stress**-1.2 * np.random.default_rng(1).weibull(2, 90)
    al = sp.AcceleratedLife(sp.Weibull, Power).fit(t, Z=stress)
    _, shape, a, n = al.params
    np.testing.assert_allclose(
        al.mean([10.0, 25.0]),
        a * np.array([10.0, 25.0]) ** n * gamma(1 + 1 / shape),
        rtol=1e-12,
    )
    tvc = al.mean_tvc(StepSchedule.constant([10.0]))
    assert al.mean(10.0) == pytest.approx(tvc, rel=1e-8)
    restored = sp.from_dict(al.to_dict())
    assert restored.mean(10.0) == al.mean(10.0)


# ---------------------------------------------------------------------------
# #657: a Z of the wrong width is refused, naming the covariates
# ---------------------------------------------------------------------------

WIDTH = r"The model has 2 covariates \(coef_0, coef_1\); Z gives 3 per row"


def _width_models():
    x, Z, c, g = _grouped(n=90)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "WeibullPH": sp.WeibullPH.fit(x, Z, c=c),
            "LogNormalAFT": sp.LogNormalAFT.fit(x, Z, c=c),
            "WeibullAH": sp.WeibullAH.fit(x, Z, c=c),
            "CoxPH": sp.CoxPH.fit(x, Z, c=c),
            "ProportionalOdds": sp.ProportionalOdds.fit(x, Z, c=c),
            "AdditiveHazards": sp.AdditiveHazards.fit(x, Z, c=c),
            "BuckleyJames": sp.BuckleyJames.fit(x, Z, c=c),
            "CoxFrailty": sp.CoxFrailty.fit(x, Z, c=c, groups=g),
            "WeibullFrailty": sp.WeibullFrailty.fit(x, Z, c=c, groups=g),
        }


@pytest.mark.parametrize("name", list(_width_models()))
def test_657_wrong_covariate_width_is_named(name):
    # numpy's "operands could not be broadcast together with shapes (3,)
    # (2,)" (Cox) or "shapes (3,) and (2,) not aligned" (WeibullPH).
    model = _width_models()[name]
    for fn in ("sf", "ff", "Hf", "hf", "df", "qf"):
        method = getattr(model, fn, None)
        if method is None:
            continue
        query = [0.5] if fn == "qf" else [5.0]
        with pytest.raises(ValueError, match=WIDTH):
            method(query, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=WIDTH):
            method(query, [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match=WIDTH):
        model.cs(1.0, 2.0, [1.0, 2.0, 3.0])
    if hasattr(model, "cb"):
        with pytest.raises(ValueError, match=WIDTH):
            model.cb([5.0], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=WIDTH):
            model.quantile_cb([0.5], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=WIDTH):
            model.mean([1.0, 2.0, 3.0])
    # The right width still predicts, a row per time.
    assert np.shape(model.sf([5.0, 6.0], ZQ[:2])) == (2,)


def test_657_accelerated_life_stresses_and_times():
    # A 1-D stress per time, against more times, was a raw broadcast
    # error; it is the row-count message every regression gives.
    stress = np.repeat([1.0, 2.0, 4.0], 30)
    t = 1000 * stress**-1.2 * np.random.default_rng(1).weibull(2, 90)
    al = sp.AcceleratedLife(sp.Weibull, Power).fit(t, Z=stress)
    with pytest.raises(ValueError, match="Z has 2 covariate rows for 3"):
        al.sf(np.array([1e2, 2e2, 3e2]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="1 stress column"):
        al.sf([1e2], [[1.0, 2.0]])
    np.testing.assert_allclose(
        al.sf([1e2, 2e2], [1.0, 2.0]),
        [al.sf(1e2, 1.0), al.sf(2e2, 2.0)],
    )
    restored = sp.from_dict(al.to_dict())
    with pytest.raises(ValueError, match="Z has 2 covariate rows for 3"):
        restored.sf(np.array([1e2, 2e2, 3e2]), np.array([1.0, 2.0]))
