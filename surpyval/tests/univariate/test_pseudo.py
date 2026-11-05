"""Properties of the pseudo-observations and of the regressions on them
(the R checks are in ``tests/reference/test_pseudo.py``)."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from surpyval import (
    CompetingRisks,
    KaplanMeier,
    pseudo_cif,
    pseudo_regression,
    pseudo_regression_from_df,
    pseudo_rmst,
    pseudo_survival,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(20261105)
    m = 60
    z = rng.integers(0, 2, m).astype(float)
    x = np.round(rng.exponential(10 * np.exp(-0.6 * z)), 1) + 0.1
    c = (rng.uniform(size=m) < 0.3).astype(float)
    e = np.where(c == 0, rng.choice(["a", "b"], m), None).astype(object)
    return x, c, z, e


def test_without_censoring_the_pseudo_values_are_the_outcomes():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    assert_allclose(
        pseudo_survival(x, [2, 5]), (x[:, None] > [2, 5]) * 1.0, atol=1e-12
    )
    assert_allclose(pseudo_rmst(x, 5), np.minimum(x, 5), atol=1e-12)
    e = np.array(["a", "b", "a", "a", "b", "b", "a", "b"], dtype=object)
    assert_allclose(
        pseudo_cif(x, e, 4, "a"), ((x <= 4) & (e == "a")) * 1.0, atol=1e-12
    )


def test_the_mean_pseudo_value_is_the_estimate(data):
    x, c, _, e = data
    km = KaplanMeier.fit(x, c)
    t = [2.0, 5.0, 12.0]
    assert_allclose(pseudo_survival(x, t, c).mean(axis=0), km.sf(t))
    tau = 8.0
    grid = np.linspace(0, tau, 400001)
    area = np.sum(km.sf(grid[:-1]) * np.diff(grid))  # left Riemann sum
    assert_allclose(pseudo_rmst(x, tau, c).mean(), area, rtol=1e-4)
    cr = CompetingRisks.fit(x, e, c)
    assert_allclose(pseudo_cif(x, e, t, "a", c).mean(axis=0), cr.cif(t, "a"))


def test_cifs_and_survival_add_up_to_one_unit_by_unit(data):
    x, c, _, e = data
    t = [3.0, 7.0]
    total = (
        pseudo_cif(x, e, t, "a", c)
        + pseudo_cif(x, e, t, "b", c)
        + pseudo_survival(x, t, c)
    )
    assert_allclose(total, 1.0, atol=1e-12)


def test_counts_are_copies(data):
    x, c, z, e = data
    n = np.arange(len(x)) % 3 + 1
    xr, cr_, zr = (np.repeat(a, n) for a in (x, c, z))
    er = np.repeat(e, n)
    assert_allclose(
        np.repeat(pseudo_survival(x, [4, 8], c, n), n, axis=0),
        pseudo_survival(xr, [4, 8], cr_),
    )
    assert_allclose(
        np.repeat(pseudo_rmst(x, 8, c, n), n), pseudo_rmst(xr, 8, cr_)
    )
    assert_allclose(
        np.repeat(pseudo_cif(x, e, 6, "b", c, n), n),
        pseudo_cif(xr, er, 6, "b", cr_),
    )
    a = pseudo_regression(x, z, c, n, quantity="rmst", tau=8)
    b = pseudo_regression(xr, zr, cr_, quantity="rmst", tau=8)
    assert_allclose(a.coef, b.coef, rtol=1e-10)
    assert_allclose(a.se, b.se, rtol=1e-10)


def test_row_order_does_not_matter(data):
    x, c, z, _ = data
    perm = np.random.default_rng(1).permutation(len(x))
    assert_allclose(
        pseudo_survival(x, 5, c)[perm], pseudo_survival(x[perm], 5, c[perm])
    )
    a = pseudo_regression(x, z, c, t=[4, 8])
    b = pseudo_regression(x[perm], z[perm], c[perm], t=[4, 8])
    assert_allclose(a.coef, b.coef, rtol=1e-9)


def test_rmst_identity_coefficient_is_minus_the_difference(data):
    # With one binary covariate the identity-link fit is the difference of
    # the two groups' mean pseudo-values, with SurPyval's sign.
    x, c, z, _ = data
    y = pseudo_rmst(x, 8, c)
    res = pseudo_regression(x, z, c, quantity="rmst", tau=8)
    diff = y[z == 1].mean() - y[z == 0].mean()
    assert_allclose(res.coef, [y[z == 0].mean(), -diff], rtol=1e-10)
    assert res.coef[1] > 0  # z = 1 shortens life in these data
    assert_allclose(res.predict([0, 1]), [y[z == 0].mean(), y[z == 1].mean()])


def test_rescaling_a_covariate_rescales_its_coefficient(data):
    x, c, z, _ = data
    a = pseudo_regression(x, z, c, t=[4, 8])
    b = pseudo_regression(x, 10 * z, c, t=[4, 8])
    assert_allclose(b.coef[-1] * 10, a.coef[-1], rtol=1e-8)
    assert_allclose(b.se[-1] * 10, a.se[-1], rtol=1e-8)


def test_predictions_and_bounds(data):
    x, c, z, _ = data
    res = pseudo_regression(x, z, c, t=[4, 8], link="logit")
    s = res.predict([0, 1], t=8)
    cb = res.predict_cb([0, 1], t=8)
    assert np.all((cb[:, 0] < s) & (s < cb[:, 1]))
    assert np.all((cb >= 0) & (cb <= 1))
    assert s[1] < s[0]  # positive coefficient, lower survival
    with pytest.raises(ValueError, match="one of the model's times"):
        res.predict([0], t=5)
    with pytest.raises(ValueError, match="Give 't'"):
        res.predict([0])


def test_summary_and_to_dict(data):
    x, c, z, _ = data
    res = pseudo_regression(x, z, c, quantity="rmst", tau=8, names=["z"])
    table = res.summary()
    assert list(table.index) == ["Intercept", "z"]
    assert table["exp(coef)"].isna().all()  # identity link
    assert_allclose(table["p"], res.p)
    out = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    assert out["quantity"] == "rmst" and out["names"] == ["Intercept", "z"]
    assert "shorter life" in out["sign"]
    assert "Pseudo-observation regression" in repr(res)
    lo, hi = res.ci.T
    assert np.all((lo < res.coef) & (res.coef < hi))


def test_from_df_matches_arrays(data):
    x, c, z, e = data
    df = pd.DataFrame({"x": x, "c": c, "z": z, "e": e})
    a = pseudo_regression_from_df(
        df, "x", "z", c_col="c", e_col="e", quantity="cif", cause="a", t=6
    )
    b = pseudo_regression(x, z, c, quantity="cif", e=e, cause="a", t=6)
    assert_allclose(a.coef, b.coef)
    assert a.names == ["Intercept", "z"]


def test_missing_covariates_are_dropped_with_a_warning(data):
    x, c, z, _ = data
    z2 = z.copy()
    z2[[0, 5]] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = pseudo_regression(x, z2, c, quantity="rmst", tau=8)
    assert any("Dropped 2 of 60" in str(w.message) for w in caught)
    keep = np.isfinite(z2)
    ref = pseudo_regression(x[keep], z[keep], c[keep], quantity="rmst", tau=8)
    assert_allclose(res.coef, ref.coef)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(t=5, tl=1.0), "delayed entry"),
        (dict(t=500), "past the largest time"),
        (dict(t=5, link="log"), "not available"),
        (dict(quantity="cif", t=5), "needs the causes"),
        (dict(quantity="rmst"), "needs 'tau'"),
        (dict(), "needs the time"),
        (dict(quantity="median", t=5), "use one of"),
    ],
)
def test_refusals(data, kwargs, match):
    x, c, z, _ = data
    with pytest.raises(ValueError, match=match):
        pseudo_regression(x, z, c, **kwargs)


def test_more_refusals(data):
    x, c, z, _ = data
    with pytest.raises(ValueError, match="left .* or interval"):
        pseudo_survival([1.0, 2.0, 3.0], 2, [0, -1, 1])
    with pytest.raises(ValueError, match="collinear"):
        pseudo_regression(x, np.c_[z, 2 * z], c, t=5)
    with pytest.raises(ValueError, match="No failures by t"):
        pseudo_regression(x, z, c, t=0.05)
    with pytest.raises(ValueError, match="No failures at or before tau"):
        pseudo_rmst([5.0, 6.0, 7.0], 4, [1, 0, 0])


def test_large_sample_is_fast():
    import time

    rng = np.random.default_rng(7)
    x = rng.weibull(1.2, 5000)
    c = (rng.uniform(size=5000) < 0.3).astype(float)
    start = time.perf_counter()
    pseudo_survival(x, [0.5, 1.0], c)
    assert time.perf_counter() - start < 10


def test_rmst_difference_coverage():
    # 300 simulated two-group trials (exponential lives, rates 0.1 and
    # 0.15, uniform censoring on [0, 30]); the 95% interval of the RMST
    # difference to tau = 10 covers the true one, (1 - e^(-l tau)) / l,
    # about 95% of the time.
    rng = np.random.default_rng(4041)
    tau = 10.0
    rates = np.array([0.1, 0.15])
    true = (1 - np.exp(-rates * tau)) / rates
    beta = -(true[1] - true[0])  # SurPyval's sign
    covered = 0
    reps = 300
    for _ in range(reps):
        z = np.repeat([0.0, 1.0], 100)
        life = rng.exponential(1 / rates[z.astype(int)])
        cens = rng.uniform(0, 30, z.size)
        x, c = np.minimum(life, cens), (cens < life).astype(float)
        lo, hi = pseudo_regression(x, z, c, quantity="rmst", tau=tau).ci[1]
        covered += lo <= beta <= hi
    assert 0.92 <= covered / reps <= 0.98
