"""Properties of the confidence bounds on a Cox model's predicted curve,
``SemiParametricRegressionModel.cb`` (R's ``survfit(fit, newdata)``,
checked against R in ``surpyval/tests/reference/test_cox_curve.py``)."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval import SemiParametricRegressionModel
from surpyval.datasets import load_rossi_static
from surpyval.univariate.regression.proportional_hazards._curve_variance import (  # noqa: E501
    curve_components,
)


@pytest.fixture(scope="module")
def rossi():
    df = load_rossi_static()
    return (
        df["week"].values,
        1 - df["arrest"].values,
        df[["fin", "age", "prio"]].values.astype(float),
    )


@pytest.fixture(scope="module")
def model(rossi):
    x, c, Z = rossi
    return sp.CoxPH.fit(x, Z, c=c)


ROW = [0, 25, 3]
TIMES = np.array([0.5, 1.0, 10.0, 25.5, 52.0, 60.0])


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_running_sums_hold_the_models_baseline(rossi, ties, center, weighted):
    # The variance's baseline is the model's own: its running sum at each
    # event time is H0 there, wherever the baseline is reported.
    x, c, Z = rossi
    w = np.linspace(0.5, 2.0, x.size) if weighted else None
    m = sp.CoxPH.fit(x, Z, c=c, w=w, tie_method=ties, center=center)
    data = m._fit_data
    parts = curve_components(
        data["x"],
        data["c"],
        data["n"],
        data.get("w"),
        data["tl"],
        data["Z"],
        m._coef(),
        m._fit_center,
        m._center(),
        ties,
    )
    H0 = m._baseline_step(m.x, m.H0, parts["times"])
    assert_allclose(parts["A"], H0, rtol=1e-10)


def test_bounds_hold_the_estimate_and_are_one_before_the_first_event(model):
    for bound_type in ("log", "log-log", "plain"):
        b = model.cb(TIMES, ROW, bound_type=bound_type)
        sf = model.sf(TIMES, ROW)
        assert np.all(b[:, 0] <= sf) and np.all(sf <= b[:, 1])
        # Before the first arrest (week 1) the curve is known
        assert_allclose(b[0], [1.0, 1.0])
        assert np.all(b[1:, 1] - b[1:, 0] > 0)


def test_centring_and_a_shifted_covariate_leave_the_bounds_alone(rossi):
    # The bounds are of the unit at Z, whatever the baseline is reported
    # at; and moving a covariate's origin (in the data and the query)
    # moves nothing.
    x, c, Z = rossi
    ref = sp.CoxPH.fit(x, Z, c=c).cb(TIMES, ROW)
    assert_allclose(sp.CoxPH.fit(x, Z, c=c, center=True).cb(TIMES, ROW), ref)
    shift = np.array([0.0, 100.0, -3.0])
    shifted = sp.CoxPH.fit(x, Z + shift, c=c)
    assert_allclose(shifted.cb(TIMES, np.array(ROW) + shift), ref, rtol=1e-8)


def test_time_scale_invariance(rossi):
    x, c, Z = rossi
    ref = sp.CoxPH.fit(x, Z, c=c).cb(TIMES, ROW)
    days = sp.CoxPH.fit(7 * x, Z, c=c).cb(7 * TIMES, ROW)
    assert_allclose(days, ref, rtol=1e-10)


def test_counts_are_repeated_rows():
    rng = np.random.default_rng(3)
    x = rng.integers(1, 15, 40).astype(float)
    c = (rng.uniform(size=40) < 0.3).astype(int)
    Z = rng.normal(size=(40, 2))
    n = rng.integers(1, 4, 40)
    counted = sp.CoxPH.fit(x, Z, c=c, n=n)
    repeated = sp.CoxPH.fit(
        np.repeat(x, n), np.repeat(Z, n, axis=0), c=np.repeat(c, n)
    )
    t = np.array([2.0, 6.0, 11.0])
    assert_allclose(
        counted.cb(t, [0.3, -1.0]), repeated.cb(t, [0.3, -1.0]), rtol=1e-8
    )


def test_one_sided_and_transformed(model):
    two = model.cb(TIMES, ROW, alpha_ci=0.1)
    assert_allclose(
        model.cb(TIMES, ROW, bound="lower", alpha_ci=0.05), two[:, 0]
    )
    assert_allclose(
        model.cb(TIMES, ROW, bound="upper", alpha_ci=0.05), two[:, 1]
    )
    ff = model.cb(TIMES, ROW, on="ff", alpha_ci=0.1)
    assert_allclose(ff, 1 - two[:, ::-1], atol=1e-15)
    Hf = model.cb(TIMES, ROW, on="Hf", alpha_ci=0.1)
    assert_allclose(Hf, -np.log(two[:, ::-1]), atol=1e-12)
    # The non-parametric estimates' names are accepted too
    assert_allclose(
        model.cb(TIMES, ROW, bound_type="exp"),
        model.cb(TIMES, ROW, bound_type="log-log"),
    )
    assert_allclose(
        model.cb(TIMES, ROW, bound_type="normal"),
        model.cb(TIMES, ROW, bound_type="plain"),
    )


def test_rows_pair_with_times_and_missing_values_give_nan(model):
    rows = np.array([[0, 25, 3], [1, 30, 0], [np.nan, 30, 0]])
    t = np.array([20.0, 30.0, 40.0])
    both = model.cb(t, rows)
    for k in range(2):
        assert_allclose(both[k], model.cb(t[k], rows[k]))
    assert np.isnan(both[2]).all()
    assert np.isnan(model.cb([np.nan, 10.0], ROW)[0]).all()
    assert model.cb(10.0, ROW).shape == (2,)
    assert model.cb([[10.0, 20.0]], ROW, bound="lower").shape == (1, 2)


def test_stratum_is_taken_and_checked():
    df = load_rossi_static()
    m = sp.CoxPH.fit(
        df["week"].values,
        df[["age", "prio"]].values,
        c=1 - df["arrest"].values,
        strata=df["fin"].values,
    )
    a = m.cb(TIMES, [25, 3], stratum=0)
    b = m.cb(TIMES, [25, 3], stratum=1)
    assert not np.allclose(a, b)
    sf = m.sf(TIMES, [25, 3], 1)
    assert np.all((b[:, 0] <= sf) & (sf <= b[:, 1]))
    assert np.isnan(m.cb(TIMES, [25, 3], stratum=np.nan)).all()
    with pytest.raises(ValueError, match="pass stratum="):
        m.cb(TIMES, [25, 3])


def test_dataframe_rows(rossi):
    df = load_rossi_static()
    df["c"] = 1 - df["arrest"]
    m = sp.CoxPH.fit_from_df(
        df, x_col="week", c_col="c", Z_cols=["fin", "age", "prio"]
    )
    row = pd.DataFrame({"fin": [0], "age": [25], "prio": [3]})
    x, c, Z = rossi
    assert_allclose(m.cb(TIMES, row), sp.CoxPH.fit(x, Z, c=c).cb(TIMES, ROW))


def test_refusals(model, rossi):
    with pytest.raises(ValueError, match="'bound_type' must be one of"):
        model.cb(10.0, ROW, bound_type="logit")
    with pytest.raises(ValueError, match="Wald bounds only"):
        model.cb(10.0, ROW, method="lr")
    with pytest.raises(ValueError, match="'on' must be one of"):
        model.cb(10.0, ROW, on="hf")
    restored = SemiParametricRegressionModel.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="Refit the model"):
        restored.cb(10.0, ROW)
    x, c, Z = rossi
    penalised = sp.CoxPH.fit_penalised(x, Z, c=c, penalizer=0.1)
    with pytest.raises(ValueError):
        penalised.cb(10.0, ROW)


def test_coverage_of_the_true_survival():
    # Exponential baseline (rate 0.1) and one covariate with effect 0.7;
    # 300 data sets of 150 units, censored at 12. The 95% bound (log,
    # R's default) on S(6 | z = 1) should cover the truth about 95% of
    # the time: 0.92 to 0.98 is the binomial band of 300 draws.
    rng = np.random.default_rng(20261105)
    truth = np.exp(-0.1 * 6 * np.exp(0.7))
    covered = {"log": 0, "log-log": 0, "plain": 0}
    reps = 300
    for _ in range(reps):
        z = rng.uniform(0, 2, 150)
        t = rng.exponential(1 / (0.1 * np.exp(0.7 * z)))
        c = (t > 12).astype(int)
        m = sp.CoxPH.fit(np.minimum(t, 12), z[:, None], c=c)
        for bound_type in covered:
            lo, hi = m.cb(6.0, [1.0], bound_type=bound_type)
            covered[bound_type] += lo <= truth <= hi
    for bound_type, k in covered.items():
        assert 0.92 <= k / reps <= 0.98, (bound_type, k / reps)
