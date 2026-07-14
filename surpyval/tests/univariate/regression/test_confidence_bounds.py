"""Delta-method confidence bounds for the parametric regression models.

Every parametric regression fitter (AFT, PH, PO, additive hazards,
accelerated life) produces a ``ParametricRegressionModel``, so the bounds are
tested once on that shared surface across a few representative families. The
checks are structural (bounds bracket the point estimate, stay in range, nest
with the confidence level, respect parameter support) plus one coverage check
that the interval width is calibrated, not just ordered.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import (
    ExponentialPH,
    WeibullAFT,
    WeibullAH,
    WeibullPH,
    WeibullPO,
)

ALL_FAMILIES = [WeibullPH, WeibullAFT, WeibullAH, WeibullPO, ExponentialPH]


def _data(seed, N=800, beta=(0.6, -0.3)):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-1, 1, size=(N, len(beta)))
    scale = 10.0 * np.exp(-(Z @ np.asarray(beta)) / 2.0)
    x = scale * rng.weibull(2.0, size=N)
    c = (rng.uniform(size=N) < 0.2).astype(int)
    return x, Z, c


@pytest.mark.parametrize("F", ALL_FAMILIES)
def test_parameter_names_and_covariance_shape(F):
    x, Z, c = _data(0)
    m = F.fit(x=x, Z=Z, c=c)
    names = m.parameter_names()
    # distribution params first, then one coefficient per covariate.
    assert names[-2:] == ["beta_0", "beta_1"]
    k = len(names)
    assert m.covariance().shape == (k, k)
    se = m.standard_errors()
    assert se.shape == (k,)
    assert np.all(np.isfinite(se)) and np.all(se > 0)


@pytest.mark.parametrize("F", ALL_FAMILIES)
def test_sf_bounds_bracket_and_bounded(F):
    x, Z, c = _data(1)
    m = F.fit(x=x, Z=Z, c=c)
    xs = np.array([2.0, 5.0, 10.0])
    Z0 = np.array([0.2, -0.1])
    sf = m.sf(xs, Z0)
    cb = m.cb(xs, Z0, on="sf")
    assert cb.shape == (3, 2)
    lo, hi = cb[:, 0], cb[:, 1]
    assert np.all(lo <= sf) and np.all(sf <= hi)
    assert np.all(lo >= 0) and np.all(hi <= 1)
    assert np.all(lo <= hi)


@pytest.mark.parametrize("on", ["sf", "ff", "Hf", "hf", "df"])
def test_all_functions_bracket_estimate(on):
    x, Z, c = _data(2)
    m = WeibullAFT.fit(x=x, Z=Z, c=c)
    xs = np.array([3.0, 8.0])
    Z0 = np.array([0.1, -0.2])
    est = getattr(m, on)(xs, Z0)
    cb = m.cb(xs, Z0, on=on)
    assert np.all(cb[:, 0] <= est) and np.all(est <= cb[:, 1])
    # ff/sf are probabilities.
    if on in ("sf", "ff"):
        assert np.all(cb >= 0) and np.all(cb <= 1)
    # hf/df/Hf are non-negative.
    if on in ("hf", "df", "Hf"):
        assert np.all(cb >= 0)


def test_one_sided_bounds_ordered():
    x, Z, c = _data(3)
    m = WeibullPH.fit(x=x, Z=Z, c=c)
    xs = np.array([2.0, 6.0])
    Z0 = np.array([0.3, 0.0])
    for on in ["sf", "ff", "Hf", "hf", "df"]:
        est = getattr(m, on)(xs, Z0)
        lower = m.cb(xs, Z0, on=on, bound="lower")
        upper = m.cb(xs, Z0, on=on, bound="upper")
        assert np.all(lower <= est) and np.all(est <= upper)


def test_tighter_confidence_gives_wider_bounds():
    # A smaller alpha_ci (higher confidence) must widen the interval.
    x, Z, c = _data(4)
    m = WeibullAFT.fit(x=x, Z=Z, c=c)
    xs = np.array([2.0, 5.0, 9.0])
    Z0 = np.array([0.2, -0.2])
    cb95 = m.cb(xs, Z0, on="sf", alpha_ci=0.05)
    cb90 = m.cb(xs, Z0, on="sf", alpha_ci=0.10)
    assert np.all(cb95[:, 0] <= cb90[:, 0])  # 95% lower is below 90% lower
    assert np.all(cb90[:, 1] <= cb95[:, 1])  # 95% upper is above 90% upper


def test_param_cb_brackets_and_respects_support():
    x, Z, c = _data(5)
    m = WeibullPH.fit(x=x, Z=Z, c=c)
    names = m.parameter_names()
    for nm in names:
        idx = names.index(nm)
        lo, hi = m.param_cb(nm)
        assert lo <= m.params[idx] <= hi
        assert lo <= hi
    # Weibull scale/shape are strictly positive; their lower bound must be too.
    assert m.param_cb("alpha")[0] > 0
    assert m.param_cb("beta")[0] > 0


def test_param_cb_one_sided():
    x, Z, c = _data(6)
    m = WeibullAFT.fit(x=x, Z=Z, c=c)
    # One-sided bounds return a single value (as a length-1 array, matching
    # the two-sided [lower, upper] convention).
    lower = np.ravel(m.param_cb("beta_0", bound="lower"))[0]
    upper = np.ravel(m.param_cb("beta_0", bound="upper"))[0]
    beta0 = m.params[m.parameter_names().index("beta_0")]
    assert lower <= beta0 <= upper


def test_fixed_parameter_has_zero_variance():
    x, Z, c = _data(7)
    m = WeibullAFT.fit(x=x, Z=Z, c=c, fixed={"beta_1": 0.0})
    names = m.parameter_names()
    j = names.index("beta_1")
    cov = m.covariance()
    assert np.allclose(cov[j, :], 0.0) and np.allclose(cov[:, j], 0.0)
    # The free parameters still have positive variance.
    assert m.standard_errors()[names.index("beta_0")] > 0


def test_coverage_of_coefficient_interval():
    # The 95% Wald interval for a coefficient should cover the truth close to
    # 95% of the time -- a check that the covariance magnitude is calibrated,
    # not merely that the interval is ordered.
    true_beta0 = 0.5
    covered = 0
    reps = 120
    for s in range(reps):
        rng = np.random.default_rng(1000 + s)
        N = 500
        Z = rng.uniform(-1, 1, size=(N, 2))
        phi = np.exp(Z @ [true_beta0, -0.4])
        x = 10.0 * rng.weibull(2.0, size=N) / phi
        c = (rng.uniform(size=N) < 0.15).astype(int)
        m = WeibullAFT.fit(x=x, Z=Z, c=c)
        lo, hi = m.param_cb("beta_0")
        covered += lo <= true_beta0 <= hi
    # Allow Monte-Carlo slack (finite-sample Wald slightly under-covers).
    assert 0.88 <= covered / reps <= 1.0


def test_cb_accepts_dataframe_covariates():
    x, Z, c = _data(8)
    df = pd.DataFrame({"t": x, "c": c, "age": Z[:, 0], "dose": Z[:, 1]})
    m = WeibullAFT.fit_from_df(
        df, x_col="t", Z_cols=["age", "dose"], c_col="c"
    )
    row = df[["age", "dose"]].iloc[[0]]
    cb = m.cb([1.0, 2.0], row, on="sf")
    assert cb.shape == (2, 2)
    assert np.all(cb >= 0) and np.all(cb <= 1)


def test_cb_rejects_bad_arguments():
    x, Z, c = _data(9)
    m = WeibullPH.fit(x=x, Z=Z, c=c)
    with pytest.raises(ValueError, match="`on` must be one of"):
        m.cb([1.0], [0.0, 0.0], on="nonsense")
    with pytest.raises(ValueError, match="`bound` must be"):
        m.cb([1.0], [0.0, 0.0], bound="sideways")
    with pytest.raises(ValueError, match="Unknown parameter"):
        m.param_cb("beta_99")


def test_plot_with_bounds_returns_axes():
    import matplotlib

    matplotlib.use("Agg")
    x, Z, c = _data(10)
    m = WeibullPH.fit(x=x, Z=Z, c=c)
    ax = m.plot(plot_bounds=True)
    assert ax is not None
    # A band (fill_between) adds a PolyCollection to the axes.
    assert len(ax.collections) >= 1
