"""
Regression tests for the #261 serialisation/robustness batch and the
#259 TVC prediction/alignment batch.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval as surv
from surpyval import Weibull, WeibullPH
from surpyval.univariate.regression import (
    AcceleratedLife,
    BuckleyJames,
    CoxPH,
)
from surpyval.univariate.regression.accelerated_life import Power
from surpyval.univariate.regression.additive_hazards.additive_hazards import (
    AdditiveHazards,
)
from surpyval.univariate.regression.proportional_hazards.diagnostics import (
    robust_covariance,
)


def _df(seed=7, n=240):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], n)
    age = rng.normal(50, 10, n)
    x = (
        10
        * np.exp(-0.3 * (sex == "M") + 0.01 * (age - 50))
        * (-np.log(rng.uniform(size=n))) ** (1 / 2)
    )
    return pd.DataFrame(
        {"time": x, "sex": sex, "age": age, "cens": np.zeros(n)}
    )


# -- #261: serialisation & robustness ---------------------------------------


def test_buckley_james_formula_round_trip():
    df = _df()
    m = BuckleyJames.fit_from_df(
        df, x_col="time", formula="age + I(age**2) + sex", c_col="cens"
    )
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    new = pd.DataFrame({"age": [55.0], "sex": ["M"]})
    assert np.allclose(m.sf(5.0, new), restored.sf(5.0, new))


def test_additive_hazards_formula_round_trip():
    df = _df()
    m = AdditiveHazards.fit_from_df(
        df, x_col="time", formula="age + sex", c_col="cens"
    )
    restored = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    new = pd.DataFrame({"age": [55.0, 45.0], "sex": ["M", "F"]})
    assert np.allclose(m.sf([5.0, 10.0], new), restored.sf([5.0, 10.0], new))


def test_second_generation_serialisation_keeps_covariance():
    np.random.seed(2)
    x = Weibull.random(200, 10, 3)
    Z = np.random.normal(size=(200, 1))
    m = WeibullPH.fit(x, Z=Z)
    gen1 = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    gen2 = surv.from_dict(json.loads(json.dumps(gen1.to_dict())))
    cb = gen2.param_cb("beta_0")
    assert np.all(np.isfinite(cb))


def test_regression_random_dispatches_to_fitter():
    np.random.seed(3)
    x = Weibull.random(150, 10, 3)
    Z = np.random.normal(size=(150, 1))
    m = WeibullPH.fit(x, Z=Z)
    out = m.random(20, Z[:1])
    # The PH fitter's sampler returns (x, Z) arrays.
    assert np.all(np.asarray(out[0]) > 0)


def test_accelerated_life_refit_and_fixed():
    np.random.seed(4)
    stress = np.repeat([1.0, 2.0, 3.0, 4.0], 50)
    x = (100 / stress) * (-np.log(np.random.uniform(size=len(stress)))) ** (
        1 / 3
    )
    fitter = AcceleratedLife(Weibull, Power)
    m1 = fitter.fit(x, Z=stress)  # 1-D stress vector (#261)
    assert np.isfinite(np.atleast_1d(m1.sf([50.0], np.array([1.0])))).all()
    # A second fit on the same fitter instance used to corrupt the
    # parameter map; and user-fixed parameters were dropped from
    # ``model.fixed`` so SEs were reported for constrained parameters.
    m2 = fitter.fit(x, Z=stress, fixed={"beta": 3.0})
    assert "beta" in m2.fixed
    assert m2.params[1] == pytest.approx(3.0, abs=1e-9)


def test_ph_fit_accepts_plain_list_Z():
    np.random.seed(5)
    x = Weibull.random(100, 10, 3)
    Z = [[float(v)] for v in np.random.normal(size=100)]
    m = WeibullPH.fit(x, Z=Z)
    assert np.isfinite(m.params).all()


def test_fit_accepts_ndarray_init():
    np.random.seed(6)
    x = Weibull.random(100, 10, 3)
    m = Weibull.fit(x, init=np.array([10.0, 3.0]))
    assert np.isfinite(m.params).all()


def test_restored_parametric_model_bic_aicc_and_reserialisation():
    np.random.seed(7)
    x = Weibull.random(100, 10, 3)
    m = Weibull.fit(x)
    restored = surv.from_dict(
        json.loads(json.dumps(m.to_dict(with_data=True)))
    )
    assert restored.bic() == pytest.approx(m.bic())
    assert restored.aic_c() == pytest.approx(m.aic_c())
    assert restored.support[0] == 0.0
    # Re-serialising a restored model used to crash on list data.
    again = restored.to_dict(with_data=True)
    assert "data" in again


def test_invalid_cb_arguments_raise_value_error():
    np.random.seed(8)
    m = Weibull.fit(Weibull.random(80, 10, 3))
    with pytest.raises(ValueError):
        m.cb([5.0], on="bogus")
    with pytest.raises(ValueError):
        m.param_cb("alpha", bound="both")


def test_interval_bound_below_support_raises():
    with pytest.raises(ValueError):
        Weibull.fit(xl=np.array([-0.5, 1, 2]), xr=np.array([0.5, 2, 3]))
    with pytest.raises(ValueError):
        Weibull.fit(np.array([0.0, 1.0, 2.0, 5.0]), c=np.array([-1, 0, 0, 0]))


# -- #259: TVC prediction and alignment -------------------------------------


def _tvc_fit(seed=4, n=80):
    rng = np.random.default_rng(seed)
    rows_i, rows_xl, rows_xr, rows_c, rows_z = [], [], [], [], []
    for i in range(n):
        z1 = rng.normal()
        t = rng.exponential(np.exp(-0.4 * z1)) * 4
        change = min(3.0, 0.6 * t)
        tend = min(t, 8.0)
        rows_i += [i, i]
        rows_xl += [0.0, change]
        rows_xr += [change, tend]
        rows_c += [1, 0 if t < 8.0 else 1]
        rows_z += [[z1], [z1 + 1.0]]
    return (
        np.array(rows_i),
        np.array(rows_xl),
        np.array(rows_xr),
        np.array(rows_c),
        np.array(rows_z),
    )


def test_tvc_prediction_uses_old_covariate_at_change_time():
    # (xl, xr] convention: a baseline jump exactly at a covariate change
    # time belongs to the OLD covariate, matching the fitted likelihood.
    i, xl, xr, c, Z = _tvc_fit()
    m = CoxPH.fit_tvc(i, xl, xr, c, Z)
    sched = m.Hf_tvc(
        [3.0], Z=np.array([[0.0], [1.0]]), xl=np.array([0.0, 3.0])
    )
    const = m.Hf_tvc([3.0], Z=np.array([[0.0]]), xl=np.array([0.0]))
    assert float(np.atleast_1d(sched)[0]) == pytest.approx(
        float(np.atleast_1d(const)[0])
    )


def test_tvc_hf_query_independent_of_other_query_points():
    i, xl, xr, c, Z = _tvc_fit()
    m = CoxPH.fit_tvc(i, xl, xr, c, Z)
    Zs = np.array([[0.0], [1.0]])
    starts = np.array([0.0, 3.0])
    single = float(np.atleast_1d(m.Hf_tvc([3.0], Z=Zs, xl=starts))[0])
    paired = float(np.atleast_1d(m.Hf_tvc([3.0, 3.5], Z=Zs, xl=starts))[0])
    assert single == pytest.approx(paired)


def test_tvc_cluster_labels_aligned_after_internal_sort():
    i, xl, xr, c, Z = _tvc_fit()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(i))
    m_sorted = CoxPH.fit_tvc(i, xl, xr, c, Z)
    m_shuffled = CoxPH.fit_tvc(i[perm], xl[perm], xr[perm], c[perm], Z[perm])
    cov_sorted = robust_covariance(m_sorted, cluster=i)
    cov_shuffled = robust_covariance(m_shuffled, cluster=i[perm])
    assert np.allclose(cov_sorted, cov_shuffled)
    # And a TVC fit defaults to clustering by subject.
    assert np.allclose(robust_covariance(m_sorted), cov_sorted)


def test_degenerate_tvc_fit_degrades_instead_of_crashing():
    i = np.array([0, 0, 1, 1])
    xl = np.array([0.0, 1.0, 0.0, 1.0])
    xr = np.array([1.0, 2.0, 1.0, 2.0])
    c = np.array([1, 0, 1, 0])
    Z = np.array([[1.0], [1.0], [1.0], [1.0]])
    m = CoxPH.fit_tvc(i, xl, xr, c, Z)
    # Singular information: NaN p-values are the correct signal.
    assert m.p_values is not None
