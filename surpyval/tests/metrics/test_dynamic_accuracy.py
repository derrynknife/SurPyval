"""Accuracy of dynamic predictions at an age, over a window:
``dynamic_auc``, ``dynamic_brier`` and ``dynamic_calibration``.

The checks: a hand-worked example with a unit censored inside the window
and one entering late; with no censoring inside the window, the plain AUC
and mean squared error of the window's outcomes for a fitted joint model;
the same numbers as ``auc_td`` and ``brier_score`` on the units at risk;
readings after the age are never used; a model of the ages alone ranks
like a coin toss and the joint model beats it; cross-validation by unit;
``LandmarkCox`` as a like-for-like baseline; calibration tables; counts;
saving; and the refusals. The bias of the estimates over simulated data
sets with censoring is a calibration study in
``tests/calibration/test_metric_bias.py``.
"""

import functools
import json
import warnings

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from surpyval import (
    ItemLifetime,
    LandmarkCox,
    SurpyvalData,
    Weibull,
    WeibullJointPH,
    WeibullPH,
)
from surpyval.metrics import (
    DynamicAccuracy,
    DynamicCalibration,
    auc_td,
    brier_score,
    dynamic_auc,
    dynamic_brier,
    dynamic_calibration,
)
from surpyval.tests.univariate.joint._fleet import fleet, simulate


class _GivenRisk:
    """A model whose chance of failing in any window from age 1 is the
    unit's one covariate (its survival is 1 to age 1, then 1 - Z)."""

    def sf(self, x, Z):
        x = np.asarray(x, dtype=float)
        risk = np.asarray(Z, dtype=float)[:, 0]
        return np.where(x <= 1.0, 1.0, 1.0 - risk)


def _hand_data():
    # Six units at risk at 1, one censored inside the window (1, 5] at 3,
    # and a seventh that enters at 2, after the age scored.
    return SurpyvalData(
        x=[2, 3, 4, 6, 7, 8, 9],
        c=[0, 1, 0, 0, 1, 0, 0],
        tl=[0, 0, 0, 0, 0, 0, 2],
        Z=[[0.9], [0.5], [0.4], [0.3], [0.6], [0.1], [0.99]],
        i=list("abcdefg"),
    )


def test_hand_worked_example_with_censoring_in_the_window():
    # Cases: the failures at 2 and 4; controls: those past 5 (6, 7, 8);
    # the unit censored at 3 is neither. The censoring survival of the
    # units at risk at 1 drops to 1 - 1/5 at 3 (five at risk then), so the
    # failure at 2 weighs 1, that at 4 and each control 1 / 0.8.
    data = _hand_data()
    auc = dynamic_auc(_GivenRisk(), data, at=1, horizon=4)
    brier = dynamic_brier(_GivenRisk(), data, at=1, horizon=4)
    # AUC: the case at 0.9 outranks all three controls (0.3, 0.6, 0.1),
    # the one at 0.4 two of them
    assert auc.estimate()[0] == pytest.approx((3 + 1.25 * 2) / (2.25 * 3))
    # Brier: (0.1^2 + 1.25 0.6^2 + 1.25 (0.3^2 + 0.6^2 + 0.1^2)) / 6
    expected = (0.01 + 1.25 * 0.36 + 1.25 * (0.09 + 0.36 + 0.01)) / 6
    assert brier.estimate()[0] == pytest.approx(expected)
    row = brier.scores.iloc[0]
    assert (row["n_at_risk"], row["n_events"], row["n_censored"]) == (6, 2, 1)


def test_the_same_as_auc_td_and_brier_score_on_the_units_at_risk():
    rng = np.random.default_rng(11)
    n = 200
    z = rng.integers(0, 2, n).astype(float)
    t = 10 * rng.weibull(1.5, n) * np.exp(-0.7 * z)
    cens = rng.uniform(0, 25, n)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    data = SurpyvalData(x=x, c=c, Z=z[:, None], covariate_names=["z"])
    model = WeibullPH.fit(data)
    s, w = 4.0, 5.0
    keep = x > s
    S = model.sf(np.full(keep.sum(), s + w), z[keep, None]) / model.sf(
        np.full(keep.sum(), s), z[keep, None]
    )
    want_auc = auc_td(x[keep], c[keep], 1 - S, [s + w])[1]
    want_bs = brier_score(x[keep], c[keep], S[:, None], [s + w])[1]
    got_auc = dynamic_auc(model, data, at=s, horizon=w).estimate()
    got_bs = dynamic_brier(model, data, at=s, horizon=w).estimate()
    assert np.isclose(got_auc, want_auc, rtol=1e-12).all()
    assert np.isclose(got_bs, want_bs, rtol=1e-12).all()
    # there was censoring inside the window
    assert np.any(keep & (x <= s + w) & (c == 1))


@functools.cache
def _fleet_and_model():
    data = fleet(120, seed=1)
    return data, WeibullJointPH.fit(data)


def _joint_risk(model, data, s, w):
    """Each unit at risk at ``s``: (age, flag, chance of failing by s + w
    given its readings to ``s``), straight from the model."""
    out = []
    for k, item in enumerate(data.to_items()):
        if item.value <= s:
            continue
        known = {a: v for a, v in item.markers.items() if a <= s}
        risk = float(model.ff(s + w, markers=known, given=s))
        out.append((item.value, int(item.outcome == "right_censored"), risk))
    return np.array(out)


def test_no_censoring_in_the_window_is_the_plain_auc_and_squared_error():
    # Followed to 730 days, so nobody is censored before 380 or 480
    data, model = _fleet_and_model()
    w = 180.0
    auc = dynamic_auc(model, data, at=[200, 300], horizon=w)
    brier = dynamic_brier(model, data, at=[200, 300], horizon=w)
    for k, s in enumerate([200.0, 300.0]):
        x, c, risk = _joint_risk(model, data, s, w).T
        failed = x <= s + w
        assert not np.any(failed & (c == 1))
        cases, controls = risk[failed], risk[~failed]
        wins = (cases[:, None] > controls[None, :]).sum()
        ties = (cases[:, None] == controls[None, :]).sum()
        plain_auc = (wins + ties / 2) / (cases.size * controls.size)
        assert auc.estimate()[k] == pytest.approx(plain_auc, rel=1e-12)
        mse = np.mean((failed - risk) ** 2)
        assert brier.estimate()[k] == pytest.approx(mse, rel=1e-12)
        assert auc.scores["n_at_risk"].iloc[k] == x.size


def test_readings_after_the_age_are_not_used():
    data, model = _fleet_and_model()
    s, w = 200.0, 180.0
    changed = []
    for item in data.to_items():
        markers = {
            a: (v if a <= s else v + 50.0) for a, v in item.markers.items()
        }
        outcome = {item.outcome: item.value}
        changed.append(ItemLifetime(item.id, **outcome, markers=markers))
    other = SurpyvalData.from_items(changed)
    for score in (dynamic_auc, dynamic_brier):
        a = score(model, data, at=s, horizon=w).estimate()
        b = score(model, other, at=s, horizon=w).estimate()
        np.testing.assert_array_equal(a, b)


def test_ages_alone_rank_like_a_coin_toss_and_the_readings_do_better():
    data, model = _fleet_and_model()
    models = {"age only": Weibull.fit(data.without("m")), "joint": model}
    auc = dynamic_auc(models, data, at=[200, 300], horizon=180)
    np.testing.assert_array_equal(auc.estimate("age only"), [0.5, 0.5])
    # 120 pumps: the joint model ranks better than a coin toss at both
    # ages (0.64 at 200 days, 0.74 at 300)
    assert np.all(auc.estimate("joint") > 0.6)
    table = auc.comparisons
    assert list(table["model"]) == ["joint", "joint"]
    assert list(table["reference"]) == ["age only", "age only"]
    assert np.all(table["difference"] > 0)
    assert np.all(np.isfinite(table["p_value"]))
    brier = dynamic_brier(models, data, at=[200, 300], horizon=180)
    assert np.all(brier.estimate("joint") < brier.estimate("age only"))
    # the intervals contain the estimates, at the level asked for
    narrow = dynamic_brier(models, data, at=200, horizon=180, alpha_ci=0.1)
    wide = dynamic_brier(models, data, at=200, horizon=180)
    rows = narrow.scores
    assert np.all((rows["lower"] <= rows["estimate"]))
    assert np.all((rows["estimate"] <= rows["upper"]))
    assert np.all(
        (wide.scores["upper"] - wide.scores["lower"])
        > (rows["upper"] - rows["lower"])
    )
    assert "90% two-sided" in repr(narrow)


def test_cross_validation_refits_without_each_group():
    data = fleet(80, seed=2)
    fitters = {"age only": Weibull, "joint": WeibullJointPH}
    a = dynamic_auc(
        fitters,
        data,
        at=[150, 300],
        horizon=180,
        cross_validate=2,
        random_state=5,
    )
    b = dynamic_auc(
        fitters,
        data,
        at=[150, 300],
        horizon=180,
        cross_validate=2,
        random_state=5,
    )
    assert_frame_equal(a.scores, b.scores)
    assert a.cross_validate == 2
    assert "Cross-validated" in repr(a)
    # held out, the age-only fits differ between the groups, so its AUC
    # is no longer exactly 0.5; the joint model's differs from in-sample
    assert np.all(np.isfinite(a.estimate("joint")))
    in_sample = dynamic_auc(
        WeibullJointPH.fit(data), data, at=[150, 300], horizon=180
    )
    assert not np.allclose(a.estimate("joint"), in_sample.estimate())
    # the same units are scored either way
    np.testing.assert_array_equal(
        a.scores["n_at_risk"].to_numpy()[:2],
        in_sample.scores["n_at_risk"].to_numpy(),
    )
    # a function of the data is a fitter too
    c = dynamic_brier(
        lambda d: Weibull.fit(d.without("m")),
        data,
        at=150,
        horizon=180,
        cross_validate=2,
        random_state=5,
    )
    d = dynamic_brier(
        Weibull, data, at=150, horizon=180, cross_validate=2, random_state=5
    )
    np.testing.assert_allclose(c.estimate(), d.estimate())


def test_landmark_cox_is_a_like_for_like_baseline():
    data, _ = _fleet_and_model()
    s, w = 200.0, 180.0
    lc = LandmarkCox.fit(data, at=s, horizon=w)
    result = dynamic_auc(lc, data, at=s, horizon=w)
    # by hand: the last reading at or before s, and the model's ff
    x, c, last = [], [], []
    for item in data.to_items():
        if item.value <= s:
            continue
        x.append(item.value)
        c.append(int(item.outcome == "right_censored"))
        last.append(item.markers[max(a for a in item.markers if a <= s)])
    risk = lc.ff(np.full(len(x), s + w), np.array(last)[:, None], s)
    want = auc_td(np.array(x), np.array(c), risk, [s + w])[1]
    np.testing.assert_allclose(result.estimate(), want, rtol=1e-12)
    # refitted at the ages scored, under cross-validation
    cv = dynamic_auc(
        LandmarkCox, data, at=s, horizon=w, cross_validate=3, random_state=1
    )
    assert np.isfinite(cv.estimate()).all()
    with pytest.raises(ValueError, match="'horizon' is 200"):
        dynamic_auc(lc, data, at=s, horizon=200.0)


def test_units_without_a_prediction_are_left_out_with_a_warning():
    items, _ = simulate(n=60, seed=3)
    # one unit's readings start after the age scored: a landmark model
    # knows nothing of it then
    late = next(i for i in items if i.value > 400)
    markers = {a: v for a, v in late.markers.items() if a >= 90}
    outcome = {late.outcome: late.value}
    items = [
        (
            ItemLifetime(late.id, **outcome, markers=markers)
            if i.id == late.id
            else i
        )
        for i in items
    ]
    data = SurpyvalData.from_items(items)
    joint = WeibullJointPH.fit(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lc = LandmarkCox.fit(data, at=60, horizon=180)
    at_risk = int(np.sum(data.x > 60))
    with pytest.warns(UserWarning, match="1 of {} at 60".format(at_risk)):
        result = dynamic_brier(
            {"joint": joint, "landmark": lc}, data, at=60, horizon=180
        )
    assert set(result.scores["n_at_risk"]) == {at_risk - 1}


def test_calibration_table():
    data, model = _fleet_and_model()
    s, w = 200.0, 180.0
    edges = [0, 0.2, 0.4, 0.6, 1]
    result = dynamic_calibration(model, data, at=s, horizon=w, bins=edges)
    table = result.table
    assert list(table["bin"]) == [1, 2, 3, 4]
    assert table["risk_from"].tolist() == edges[:-1]
    x, c, risk = _joint_risk(model, data, s, w).T
    assert table["n"].sum() == x.size
    for row in table.itertuples():
        inside = (risk >= row.risk_from) & (
            (risk < row.risk_to) | (row.bin == 4)
        )
        assert row.n == inside.sum()
        if row.n:
            # no censoring in the window: the share that failed
            assert row.predicted == pytest.approx(risk[inside].mean())
            failed = x[inside] <= s + w
            assert row.observed == pytest.approx(failed.mean())
            assert row.n_events == failed.sum()
    # groups of about equal size
    deciles = dynamic_calibration(model, data, at=s, horizon=w)
    sizes = deciles.table["n"].to_numpy()
    assert sizes.size == 10 and sizes.max() - sizes.min() <= 1
    assert np.all(np.diff(deciles.table["predicted"]) > 0)
    assert "95% two-sided" in repr(deciles)


def test_calibration_with_censoring_is_kaplan_meier():
    # The hand data in one group from age 1: failures at 2 and 4 with 6
    # and 4 at risk (the unit censored at 3 leaves between them)
    result = dynamic_calibration(
        _GivenRisk(), _hand_data(), at=1, horizon=4, bins=1
    )
    row = result.table.iloc[0]
    assert row["observed"] == pytest.approx(1 - (5 / 6) * (3 / 4))
    assert row["lower"] < row["observed"] < row["upper"]
    assert (row["n"], row["n_events"], row["n_censored"]) == (6, 2, 1)


def test_counts_are_identical_units():
    rng = np.random.default_rng(4)
    t = np.ceil(10 * rng.weibull(1.5, 120))
    x, c = np.minimum(t, 15), (t > 15).astype(int)
    grouped = SurpyvalData(x=x, c=c)
    assert np.any(grouped.n > 1)
    model = Weibull.fit(grouped)
    one_each = SurpyvalData(x=x, c=c, i=list(range(x.size)))
    assert np.all(one_each.n == 1)
    for score in (dynamic_brier, dynamic_auc):
        a = score(model, grouped, at=[2, 5], horizon=4)
        b = score(model, one_each, at=[2, 5], horizon=4)
        np.testing.assert_allclose(a.estimate(), b.estimate())
        np.testing.assert_array_equal(
            a.scores["n_at_risk"], b.scores["n_at_risk"]
        )


def test_no_unit_at_risk_gives_nan():
    data = _hand_data()
    result = dynamic_brier(_GivenRisk(), data, at=[1, 20], horizon=4)
    assert np.isnan(result.estimate()[1])
    assert result.scores["n_at_risk"].tolist() == [6, 0]


def test_results_save_to_strict_json():
    data = _hand_data()
    models = {"a": _GivenRisk(), "b": _GivenRisk()}
    result = dynamic_auc(models, data, at=[1, 20], horizon=4)
    text = json.dumps(result.to_dict(), allow_nan=False)
    back = DynamicAccuracy.from_dict(json.loads(text))
    np.testing.assert_array_equal(back.estimate("b"), result.estimate("b"))
    assert back.models == ["a", "b"] and back.metric == "auc"
    assert len(back.comparisons) == 2
    cal = dynamic_calibration(models, data, at=1, horizon=4, bins=[0, 0.5, 1])
    text = json.dumps(cal.to_dict(), allow_nan=False)
    back = DynamicCalibration.from_dict(json.loads(text))
    assert back.bins == [0.0, 0.5, 1.0]
    np.testing.assert_allclose(
        back.table["observed"].to_numpy(dtype=float),
        cal.table["observed"].to_numpy(dtype=float),
    )
    with pytest.raises(ValueError, match="DynamicAccuracy"):
        DynamicAccuracy.from_dict(cal.to_dict())


def test_refusals():
    data = _hand_data()
    model = _GivenRisk()
    with pytest.raises(ValueError, match="'data' must be a SurpyvalData"):
        dynamic_auc(model, [1, 2, 3], at=1, horizon=4)
    interval = SurpyvalData(x=[[1, 2], [3, 3], [4, 4]], c=[2, 0, 1])
    with pytest.raises(ValueError, match="interval censored"):
        dynamic_auc(Weibull.fit([1, 2, 3, 4]), interval, at=1, horizon=4)
    with pytest.raises(ValueError, match="'horizon'"):
        dynamic_auc(model, data, at=1, horizon=0)
    with pytest.raises(ValueError, match="'at'"):
        dynamic_auc(model, data, at=[1, 1], horizon=4)
    with pytest.raises(ValueError, match="is a fitter"):
        dynamic_auc(Weibull, data.without("covariates"), at=1, horizon=4)
    with pytest.raises(ValueError, match="cross_validate refits"):
        dynamic_auc(model, data, at=1, horizon=4, cross_validate=2)
    with pytest.raises(ValueError, match="'cross_validate'"):
        dynamic_auc(Weibull, data, at=1, horizon=4, cross_validate=1)
    with pytest.raises(ValueError, match="'alpha_ci'"):
        dynamic_brier(model, data, at=1, horizon=4, alpha_ci=1.5)
    with pytest.raises(ValueError, match="'bins'"):
        dynamic_calibration(model, data, at=1, horizon=4, bins=[0.2, 1])
    with pytest.raises(ValueError, match="'bins'"):
        dynamic_calibration(model, data, at=1, horizon=4, bins=0)
    with pytest.raises(ValueError, match="'model'"):
        dynamic_brier({}, data, at=1, horizon=4)
    with pytest.raises(ValueError, match="'model' must be a fitted model"):
        dynamic_brier(object(), data, at=1, horizon=4)
    with pytest.raises(ValueError, match="takes covariates"):
        dynamic_brier(model, data.without("covariates"), at=1, horizon=4)
