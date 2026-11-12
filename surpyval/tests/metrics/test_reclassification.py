"""idi_nri: the IDI and the NRI against hand counts, their invariances,
the resampling, serialisation and the refusals. The R comparison is in
tests/reference/test_reclassification.py."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.metrics import RiskReclassification, idi_nri

X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
C = np.array([0, 0, 1, 0, 0, 1, 0, 1])
OLD = np.array([0.5, 0.4, 0.3, 0.5, 0.2, 0.3, 0.2, 0.1])
NEW = np.array([0.6, 0.5, 0.2, 0.4, 0.2, 0.2, 0.1, 0.1])


def _est(res):
    return res.estimates.set_index("measure")["estimate"]


def test_hand_count():
    # Events by 4.5: units 1, 2 and 4; unit 3 censored at 3 makes
    # G(4-) = 5/6, so unit 4 weighs 6/5. Non-events: units 5 to 8.
    res = idi_nri(X, C, OLD, NEW, 4.5, cutoffs=[0.25, 0.45], bounds="none")
    e = _est(res)
    w = np.array([1, 1, 6 / 5])
    d = np.array([0.1, 0.1, -0.1])
    assert_allclose(e["idi_events"], (w * d).sum() / w.sum(), atol=1e-15)
    assert_allclose(e["idi_nonevents"], 0.05, atol=1e-15)
    assert_allclose(e["p_up_events"], 2 / 3.2, atol=1e-15)
    assert_allclose(e["p_down_events"], 1.2 / 3.2, atol=1e-15)
    assert_allclose(e["p_up_nonevents"], 0.0, atol=1e-15)
    assert_allclose(e["p_down_nonevents"], 0.5, atol=1e-15)
    assert_allclose(res.nri, (2 - 1.2) / 3.2 + 0.5, atol=1e-15)
    assert_allclose(res.idi, e["idi_events"] + e["idi_nonevents"])
    # categories: <0.25, [0.25, 0.45), >=0.45; unit 1 stays high, unit 2
    # goes up (0.4 -> 0.5), unit 4 down (0.5 -> 0.4)
    assert_allclose(e["p_up_events_categorical"], 1 / 3.2, atol=1e-15)
    assert_allclose(e["p_down_events_categorical"], 1.2 / 3.2, atol=1e-15)
    assert_allclose(e["nri_categorical_nonevents"], 0.25, atol=1e-15)
    assert res.n_events == 3 and res.n_nonevents == 4 and res.n == 8
    table = res.tables["events"]
    assert table.to_numpy().sum() == 3
    assert table.loc["[0.25, 0.45)", ">= 0.45"] == 1


def test_same_risks_give_zero():
    res = idi_nri(X, C, OLD, OLD, 4.5, cutoffs=[0.3], bounds="none")
    assert (res.estimates["estimate"] == 0).all()


def test_swapping_the_models_reverses_the_signs():
    a = _est(idi_nri(X, C, OLD, NEW, 4.5, cutoffs=[0.25], bounds="none"))
    b = _est(idi_nri(X, C, NEW, OLD, 4.5, cutoffs=[0.25], bounds="none"))
    for m in ("idi", "idi_events", "nri", "nri_nonevents", "nri_categorical"):
        assert_allclose(a[m], -b[m], atol=1e-15)


def test_order_of_rows_does_not_matter():
    perm = np.random.default_rng(0).permutation(X.size)
    a = idi_nri(X, C, OLD, NEW, 4.5, bounds="none")
    b = idi_nri(X[perm], C[perm], OLD[perm], NEW[perm], 4.5, bounds="none")
    assert_allclose(_est(a), _est(b), atol=1e-15)


def test_no_censoring_is_the_plain_difference_of_means():
    rng = np.random.default_rng(1)
    x = rng.exponential(1, 200)
    old = rng.uniform(0, 1, 200)
    new = np.clip(old + rng.normal(0, 0.1, 200), 0, 1)
    c = np.zeros(200, dtype=int)
    for censoring in ("ipcw", "km"):
        res = idi_nri(x, c, old, new, 1.0, censoring=censoring, bounds="none")
        ev, ne = x <= 1, x > 1
        d = new - old
        assert_allclose(res.idi, d[ev].mean() - d[ne].mean(), atol=1e-13)
        up, down = d > 0, d < 0
        nri = (up[ev].mean() - down[ev].mean()) + (
            down[ne].mean() - up[ne].mean()
        )
        assert_allclose(res.nri, nri, atol=1e-13)


def test_units_do_not_matter():
    a = idi_nri(X, C, OLD, NEW, 4.5, censoring="km", bounds="none")
    b = idi_nri(60 * X, C, OLD, NEW, 270.0, censoring="km", bounds="none")
    assert_allclose(_est(a), _est(b), atol=1e-15)


def test_seeded_resampling_is_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(3)
    n = 150
    z = rng.normal(size=n)
    t = rng.exponential(np.exp(-z))
    cens = rng.exponential(2, n)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    old = 1 - np.exp(-0.5 * np.ones(n))
    new = 1 - np.exp(-0.5 * np.exp(z))
    for bounds in ("perturbation", "bootstrap"):
        a = idi_nri(
            x, c, old, new, 0.5, bounds=bounds, n_resamples=200, random_state=7
        )
        b = idi_nri(
            x, c, old, new, 0.5, bounds=bounds, n_resamples=200, random_state=7
        )
        assert a.estimates.equals(b.estimates)
        e = a.estimates.set_index("measure")
        for m in ("idi", "nri"):
            assert e.loc[m, "lower"] < e.loc[m, "estimate"]
            assert e.loc[m, "estimate"] < e.loc[m, "upper"]
            assert e.loc[m, "p_value"] < 0.05
        assert np.isnan(e.loc["p_up_events", "p_value"])


def test_round_trip_and_repr():
    res = idi_nri(
        X,
        C,
        OLD,
        NEW,
        4.5,
        cutoffs=[0.25, 0.45],
        n_resamples=50,
        random_state=1,
    )
    d = json.loads(json.dumps(res.to_dict(), allow_nan=False))
    back = RiskReclassification.from_dict(d)
    assert back.estimates.equals(res.estimates)
    for name, table in res.tables.items():
        assert back.tables[name].equals(table)
    assert back.nri_categorical == res.nri_categorical
    assert "not a proper scoring rule" in repr(res)
    plain = idi_nri(X, C, OLD, NEW, 4.5, bounds="none")
    again = RiskReclassification.from_dict(plain.to_dict())
    assert again.nri_categorical is None and again.tables == {}
    assert np.isnan(again.estimates["lower"]).all()
    with pytest.raises(ValueError, match="RiskReclassification"):
        RiskReclassification.from_dict({"result": "PredictionScores"})


def test_top_level_names():
    assert sp.idi_nri is idi_nri
    assert sp.RiskReclassification is RiskReclassification


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(c=[0, 0, 2, 0, 0, 1, 0, 1]), "interval"),
        (dict(risk_new=NEW * 10), "1 - S\\(time"),
        (dict(risk_new=np.append(NEW[:-1], np.nan)), "missing"),
        (dict(risk_old=OLD[:-1]), "one predicted risk per unit"),
        (dict(time=0.5), "both are needed"),
        (dict(time=9.0), "both are needed"),
        (dict(cutoffs=[0.4, 0.2]), "'cutoffs'"),
        (dict(cutoffs=[0.0, 0.2]), "'cutoffs'"),
        (dict(censoring="fh"), "'censoring'"),
        (dict(bounds="wald"), "'bounds'"),
        (dict(n_resamples=1), "'n_resamples'"),
        (dict(alpha_ci=1.5), "'alpha_ci'"),
    ],
)
def test_refusals(kwargs, match):
    args = dict(x=X, c=C, risk_old=OLD, risk_new=NEW, time=4.5)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        idi_nri(**args)


def test_degenerate_resamples_warn_with_counts():
    with pytest.warns(UserWarning, match="of the 200 resamples"):
        idi_nri(
            X,
            C,
            OLD,
            NEW,
            4.5,
            bounds="bootstrap",
            n_resamples=200,
            random_state=1,
            censoring="km",
        )
