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
        back = model.ff(q, ZQ)
    # Relative to the smaller tail, where the probability is resolved.
    tail = np.minimum(P, 1 - P)
    tol = 1e-7 if name.endswith("AH") else 1e-10
    np.testing.assert_array_less(np.abs(back - P) / tail, tol)


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
