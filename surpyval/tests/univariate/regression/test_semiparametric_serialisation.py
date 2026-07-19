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
import pytest

from surpyval import AdditiveHazards, BuckleyJames
from surpyval.datasets import load_rossi_static
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
    rossi = load_rossi_static()
    Zc = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
    model = CoxPH.fit_from_df(
        rossi, x_col="week", c_col="arrest", Z_cols=Zc, method="efron"
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
    ident, start, stop, event, Zrows = [], [], [], [], []
    for s in range(n):
        split = 3.0
        end = split + np.abs(rng.weibull(1.4)) * 6.0 + 0.5
        z0 = rng.normal(0, 1)
        ident += [s, s]
        start += [0.0, split]
        stop += [split, end]
        event += [0, 1]  # event on the terminal interval
        Zrows += [[z0], [z0 + 0.2]]
    model = CoxPH.fit_tvc(
        np.array(ident),
        np.array(start),
        np.array(stop),
        np.array(event),
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
    ci1 = model.bootstrap_ci(n_boot=50, seed=1)
    ci2 = restored.bootstrap_ci(n_boot=50, seed=1)
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
