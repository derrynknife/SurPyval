"""Serialisation of the semi-parametric regression models.

The three semi-parametric regression result classes -- Cox proportional
hazards (``SemiParametricRegressionModel``), the Lin-Ying additive-hazards
model (``AdditiveHazardsModel``), and the Buckley-James AFT
(``BuckleyJamesModel``) -- round-trip through ``to_dict``/``from_dict`` (and
the JSON file variants). Each stores its coefficients plus the fitted
nonparametric baseline (or residual survival), so the restored model predicts
identically. Cox's ``phi`` and TVC prediction, the additive model's
covariance, and Buckley-James's bootstrap CI all survive the round-trip.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval as surv
from surpyval import AdditiveHazards, BuckleyJames
from surpyval.datasets import load_rossi_static
from surpyval.tests._helpers import weibull_ph_data
from surpyval.univariate.regression import (
    CoxPH,
    SemiParametricRegressionModel,
)
from surpyval.univariate.regression.additive_hazards.additive_hazards import (
    AdditiveHazardsModel,
)
from surpyval.univariate.regression.buckley_james.buckley_james import (
    BuckleyJamesModel,
)


def _semipar_data(seed=0, n=80):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.3 * Z[:, 0] - 0.2 * Z[:, 1]
    x = np.abs(rng.weibull(1.5, n) * 20 * np.exp(-lin)) + 0.5
    c = np.zeros(n)
    return x, Z, c


# -- Cox ------------------------------------------------------------------


def test_cox_round_trip_predictions():
    rossi = load_rossi_static().assign(c=lambda d: 1 - d["arrest"])
    Zc = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    model = CoxPH.fit_from_df(
        rossi, x_col="week", c_col="c", Z_cols=Zc, tie_method="efron"
    )
    restored = SemiParametricRegressionModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    Zq = np.array([1.0, 30.0, 1.0, 1.0, 0.0, 1.0, 3.0])
    t = np.array([10.0, 30.0, 52.0])
    for fn in ("hf", "Hf", "sf", "ff", "df"):
        a = np.asarray(getattr(model, fn)(t, Zq), dtype=float)
        b = np.asarray(getattr(restored, fn)(t, Zq), dtype=float)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14), fn
    assert np.allclose(model.beta, restored.beta)
    assert model.tie_method == restored.tie_method
    assert restored.feature_names == Zc


def test_cox_json_file_round_trip(tmp_path):
    x, Z, c = _semipar_data()
    model = CoxPH.fit(x, Z, c=c)
    fp = tmp_path / "cox.json"
    model.to_json(fp)
    restored = SemiParametricRegressionModel.from_json(fp)
    t = np.array([5.0, 15.0])
    Zq = np.array([0.3, -0.4])
    assert np.allclose(
        np.asarray(model.sf(t, Zq), dtype=float),
        np.asarray(restored.sf(t, Zq), dtype=float),
    )


def test_cox_tvc_round_trip():
    # a small start-stop (time-varying-covariate) fit: each subject has two
    # contiguous intervals with a covariate that changes at the split.
    rng = np.random.default_rng(3)
    n = 40
    ident, xl, xr, c, Zrows = [], [], [], [], []
    for s in range(n):
        split = 3.0
        end = split + np.abs(rng.weibull(1.4)) * 6.0 + 0.5
        z0 = rng.normal(0, 1)
        ident += [s, s]
        xl += [0.0, split]
        xr += [split, end]
        c += [
            1,
            0,
        ]  # censored split, then event (c=0) on the terminal interval
        Zrows += [[z0], [z0 + 0.2]]
    model = CoxPH.fit_tvc(
        np.array(ident),
        np.array(xl),
        np.array(xr),
        np.array(c),
        np.array(Zrows),
    )
    assert model.is_tvc
    restored = SemiParametricRegressionModel.from_dict(model.to_dict())
    assert restored.is_tvc
    s = np.array([0.0])
    st = np.array([8.0])
    Zpath = np.array([[0.5]])
    ta, sa, Ha = model.predict_tvc(s, st, Zpath)
    tb, sb, Hb = restored.predict_tvc(s, st, Zpath)
    assert np.allclose(ta, tb) and np.allclose(sa, sb) and np.allclose(Ha, Hb)


def test_cox_from_dict_rejects_wrong_model():
    with pytest.raises(ValueError, match="SemiParametricRegressionModel"):
        SemiParametricRegressionModel.from_dict({"model": "Other"})


# -- Lin-Ying additive hazards --------------------------------------------


def test_additive_hazards_round_trip():
    x, Z, c = _semipar_data(seed=1)
    model = AdditiveHazards.fit(x, Z, c=c)
    restored = AdditiveHazardsModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    xs = np.array([5.0, 15.0, 30.0])
    Zq = np.array([0.4, -0.2])
    for fn in ("hf", "Hf", "sf", "ff", "df"):
        a = np.asarray(getattr(model, fn)(xs, Zq), dtype=float)
        b = np.asarray(getattr(restored, fn)(xs, Zq), dtype=float)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14), fn
    assert np.allclose(model.beta, restored.beta)
    # covariance / standard errors survive
    assert np.allclose(model.cov, restored.cov)
    assert np.allclose(model.se, restored.se)


def test_additive_hazards_json_file_round_trip(tmp_path):
    x, Z, c = _semipar_data(seed=2)
    model = AdditiveHazards.fit(x, Z, c=c)
    fp = tmp_path / "ah.json"
    model.to_json(fp)
    restored = AdditiveHazardsModel.from_json(fp)
    xs = np.array([5.0, 20.0])
    Zq = np.array([0.1, 0.1])
    assert np.allclose(
        np.asarray(model.Hf(xs, Zq), dtype=float),
        np.asarray(restored.Hf(xs, Zq), dtype=float),
    )


def test_additive_hazards_rejects_wrong_model():
    with pytest.raises(ValueError, match="AdditiveHazardsModel"):
        AdditiveHazardsModel.from_dict({"model": "Other"})


# -- Buckley-James --------------------------------------------------------


def test_buckley_james_round_trip():
    x, Z, c = _semipar_data(seed=4)
    model = BuckleyJames.fit(x, Z, c=c)
    restored = BuckleyJamesModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    xs = np.array([5.0, 15.0, 30.0])
    Zq = np.array([0.4, -0.2])
    for fn in ("sf", "ff", "Hf"):
        a = np.asarray(getattr(model, fn)(xs, Zq), dtype=float)
        b = np.asarray(getattr(restored, fn)(xs, Zq), dtype=float)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14), fn
    assert np.allclose(model.beta, restored.beta)


def test_buckley_james_bootstrap_ci_survives_round_trip():
    x, Z, c = _semipar_data(seed=5)
    model = BuckleyJames.fit(x, Z, c=c)
    restored = BuckleyJamesModel.from_dict(model.to_dict())
    ci1 = model.bootstrap_ci(n_boot=50, random_state=1)
    ci2 = restored.bootstrap_ci(n_boot=50, random_state=1)
    assert np.allclose(ci1, ci2)


def test_buckley_james_json_file_round_trip(tmp_path):
    x, Z, c = _semipar_data(seed=6)
    model = BuckleyJames.fit(x, Z, c=c)
    fp = tmp_path / "bj.json"
    model.to_json(fp)
    restored = BuckleyJamesModel.from_json(fp)
    xs = np.array([5.0, 20.0])
    Zq = np.array([0.1, 0.1])
    assert np.allclose(
        np.asarray(model.sf(xs, Zq), dtype=float),
        np.asarray(restored.sf(xs, Zq), dtype=float),
    )


def test_buckley_james_rejects_wrong_model():
    with pytest.raises(ValueError, match="BuckleyJamesModel"):
        BuckleyJamesModel.from_dict({"model": "Other"})


# ---------------------------------------------------------------------------
# Formula fits round-trip (#261).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# A restored Cox model's residuals say they need the data; Cox
# serialises to strict JSON.
# ---------------------------------------------------------------------------


def test_restored_cox_residuals_explain_missing_data():
    x, Z = weibull_ph_data()
    restored = SemiParametricRegressionModel.from_dict(
        CoxPH.fit(x, Z).to_dict()
    )
    with pytest.raises(ValueError, match="restored"):
        restored.compute_residuals()


def test_cox_to_dict_is_strict_json_and_reads_old_dicts():
    x, Z = weibull_ph_data()
    model = CoxPH.fit(x, Z)
    text = json.dumps(model.to_dict(), allow_nan=False)
    restored = SemiParametricRegressionModel.from_dict(json.loads(text))
    np.testing.assert_allclose(
        restored.sf([3.0], [0.5]), model.sf([3.0], [0.5])
    )
    old = model.to_dict()
    old["tl"] = [-np.inf] * 200
    assert np.all(SemiParametricRegressionModel.from_dict(old).tl == -np.inf)
    tl = np.r_[np.zeros(100), np.full(100, -np.inf)]
    delayed = CoxPH.fit(x, Z, tl=tl)
    text = json.dumps(delayed.to_dict(), allow_nan=False)
    np.testing.assert_array_equal(
        SemiParametricRegressionModel.from_dict(json.loads(text)).tl, tl
    )
