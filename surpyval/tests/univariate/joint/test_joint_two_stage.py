"""The two-stage fit of a joint model, ``fit(..., how="two-stage")``.

The quick, approximate estimate: each item's path from its readings
alone (pulled towards the fleet's average where it has few readings),
then the survival part with each item's predicted path taken as a known
covariate. The checks: it is exactly where the joint fit starts (the same
code); its survival part's covariance is the inverse of the plug-in
likelihood's Hessian (against finite differences), and its marker part
has none; it says it is approximate in its printout, summary and
prediction; it refuses the likelihood and the criteria, and
``compare_links``; and it saves, loads and predicts as a joint model
does.
"""

import functools
import json

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import JointModel, WeibullJointAFT, WeibullJointPH
from surpyval.tests.univariate.joint._fleet import fleet, simulate_aft
from surpyval.univariate.joint import DynamicPrediction, joint_ph
from surpyval.univariate.joint._markers import _plug_in_fit, survival_stage

#: A pump still running at 200 days with four readings.
ITEM = dict(markers={0: 3.0, 60: 3.4, 120: 3.9, 180: 4.3}, given=200)


@functools.cache
def _data():
    return fleet(150, seed=11)


@functools.cache
def _quick():
    return WeibullJointPH.fit(_data(), how="two-stage")


class _Started(Exception):
    pass


def test_it_is_where_the_joint_fit_starts(monkeypatch):
    seen = {}

    def caught(model, arrays, theta):
        seen["theta"] = theta
        raise _Started

    monkeypatch.setattr(joint_ph, "_maximise", caught)
    with pytest.raises(_Started):
        WeibullJointPH.fit(_data())
    start = seen["theta"]
    quick = _quick()
    got = quick._theta()
    for name in ("mu", "Delta", "L", "sigma", "alpha", "beta", "zeta"):
        np.testing.assert_allclose(
            np.asarray(got[name], float),
            np.asarray(start[name], float),
            rtol=1e-9,
            atol=1e-12,
            err_msg=name,
        )


def test_the_model_says_how_it_was_fitted():
    quick = _quick()
    assert quick.how == "two-stage"
    assert quick.maximum == "not applicable"
    assert WeibullJointPH.fit(_data()).how == "MLE"
    text = repr(quick)
    assert "Fit                 : two-stage (approximate" in text
    assert "standard errors understated" in text
    assert "maximum" not in text.split("Fit")[1].split("\n")[0]
    parts = quick.summary().index.get_level_values("part")
    stage_one = [p for p in parts if "stage 1: no standard errors" in p]
    stage_two = [p for p in parts if "standard errors understated" in p]
    assert len(stage_one) == 6 and len(stage_two) == 3
    assert stage_two[-1].startswith("link: current value")


def test_only_the_survival_part_has_standard_errors():
    quick = _quick()
    se = quick.standard_errors()
    names = quick.parameter_names
    survival = names.index("alpha")
    assert np.all(np.isnan(se[:survival]))
    assert np.all(np.isfinite(se[survival:])) and np.all(se[survival:] > 0)
    table = quick.summary()
    assert table["se(coef)"].isna().sum() == survival


def test_the_covariance_is_the_plug_in_likelihoods():
    """The survival part's covariance against the inverse of a
    finite-difference Hessian of the plug-in likelihood (the search's own
    objective, a unit of weight)."""
    quick = _quick()
    seen = {}

    def caught(model, arrays, theta):
        seen["args"] = (model, arrays, theta)
        raise _Started

    # stage one's answer, as the fit hands it to stage two
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(joint_ph, "_two_stage", caught)
        with pytest.raises(_Started):
            WeibullJointPH.fit(_data(), how="two-stage")
    model, arrays, theta = seen["args"]
    _, cov, reason = survival_stage(model, arrays, theta)
    assert reason is None
    _, objective, v, scales, _ = _plug_in_fit(model, arrays, theta)
    total = float(np.sum(arrays.w))
    h = 1e-4
    k = v.size
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (
                objective(v + e_i + e_j)
                - objective(v + e_i - e_j)
                - objective(v - e_i + e_j)
                + objective(v - e_i - e_j)
            ) / (4 * h * h)
    to_u = np.concatenate([np.ones(k - scales.size), scales])
    expected = np.linalg.inv(H * total) * np.outer(to_u, to_u)
    np.testing.assert_allclose(cov, expected, rtol=1e-4)
    # reported: marker_effect is zeta itself, beta on the log scale
    names = quick.parameter_names
    se = dict(zip(names, quick.standard_errors()))
    est = dict(zip(names, quick.params))
    assert se["marker_effect"] == pytest.approx(np.sqrt(cov[-1, -1]))
    assert se["beta"] == pytest.approx(est["beta"] * np.sqrt(cov[1, 1]))


@pytest.mark.parametrize("name", ["aic", "aic_c", "bic", "neg_ll"])
def test_the_likelihood_and_the_criteria_are_refused(name):
    with pytest.raises(ValueError, match="A two-stage fit has no") as caught:
        getattr(_quick(), name)()
    assert "how='MLE'" in str(caught.value)
    assert "score_predictions" in str(caught.value)


def test_the_log_likelihood_is_refused():
    with pytest.raises(ValueError, match="two-stage"):
        _quick().log_likelihood


def test_compare_links_refuses_it():
    with pytest.raises(ValueError, match="compare_links ranks the links"):
        WeibullJointPH.compare_links(
            _data(), links=["value", "slope"], how="two-stage"
        )


def test_an_unknown_how_is_refused():
    with pytest.raises(ValueError, match="'how' must be one of"):
        WeibullJointPH.fit(_data(), how="MPP")


def test_it_predicts_and_bounds():
    quick = _quick()
    x = np.array([250.0, 400.0, 700.0])
    pred = quick.predict(**ITEM)
    assert isinstance(pred, DynamicPrediction)
    sf = pred.sf(x)
    np.testing.assert_array_equal(sf, quick.sf(x, **ITEM))
    band = pred.cb(x)
    assert np.all(np.isfinite(band))
    assert np.all((band[:, 0] < sf) & (sf < band[:, 1]))
    assert "two-stage fit (approximate" in repr(pred)
    d = pred.to_dict()
    assert d["how"] == "two-stage" and d["sf_lower"] is not None
    assert WeibullJointPH.fit(_data()).predict(**ITEM).to_dict()["how"] == (
        "MLE"
    )


def test_it_saves_and_loads():
    quick = _quick()
    d = quick.to_dict()
    # SurPyval 0.28 and older would present it as the joint fit: refused
    assert d["schema"] == 5 and d["how"] == "two-stage"
    assert "how" not in WeibullJointPH.fit(_data()).to_dict()
    restored = surpyval.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    assert type(restored) is JointModel
    assert restored.how == "two-stage"
    assert restored.maximum == "not applicable"
    np.testing.assert_array_equal(restored.params, quick.params)
    x = np.array([250.0, 400.0])
    np.testing.assert_array_equal(restored.cb(x, **ITEM), quick.cb(x, **ITEM))
    with pytest.raises(ValueError, match="A two-stage fit has no"):
        restored.aic()
    bad = dict(d, how="moments")
    with pytest.raises(ValueError, match="'MLE' or 'two-stage'"):
        JointModel.from_dict(bad)


def test_a_dataframe_takes_how():
    items = _data().to_items()
    df = pd.DataFrame(
        {
            "age": [it.value for it in items],
            "running": [int(it.outcome == "right_censored") for it in items],
            "readings": [it.markers for it in items],
        }
    )
    model = WeibullJointPH.fit_from_df(
        df,
        x_col="age",
        c_col="running",
        markers_col="readings",
        how="two-stage",
    )
    assert model.how == "two-stage"
    np.testing.assert_allclose(model.params, _quick().params, rtol=1e-10)


def test_the_accelerated_model_fits_in_two_stages():
    data = simulate_aft(150, seed=31, value=0.45)
    model = WeibullJointAFT.fit(data, how="two-stage")
    assert model.how == "two-stage" and model.survival == "AFT"
    assert model.to_dict()["schema"] == 5
    effect = dict(zip(model.parameter_names, model.params))["marker_effect"]
    se = dict(zip(model.parameter_names, model.standard_errors()))
    assert abs(effect - 0.45) < 4 * se["marker_effect"]
