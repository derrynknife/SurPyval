"""The Aalen additive hazards family of ``diagnose()``: its statistics
against R (``timereg::aalen``'s test for time-invariant effects, stored in
``data/r_aalen_additive.json`` by
``scripts/reference/reference_r_aalen_diagnostics.R``), the checks on data
built to pass or to fail them, the advice, the refusals and the plots.

Tolerances: the cumulative coefficients are closed form, so the
Cramér-von Mises and Kolmogorov-Smirnov statistics agree with timereg's
to rounding (``rtol=1e-8``) once they are taken the same way. Two
differences are by design, and the tests show each:

- timereg's straight line runs to the last event time, with ``B`` held
  at its last estimate over event times past the point where the risk set
  no longer separates the covariates; SurPyval's runs to the last time
  ``B`` is estimated. On data where the two are the same time (the
  additive fixture) the statistics are timereg's; on the veteran data
  (estimated to day 411 of 999) they are timereg's on timereg's grid.
- timereg's supremum is over the event times; SurPyval's is the supremum
  of the step function's distance from the line, which also takes the
  left limits, so it is at least timereg's.

The p-values are not compared: timereg resamples each unit's martingale
residuals, which rejected constant effects in 47% (n = 100) and 17%
(n = 300) of simulated data sets at the 5% level, against SurPyval's
3-3.5% (``tests/calibration/test_diagnose_aalen.py``).
"""

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.diagnostics import Diagnosis
from surpyval.diagnostics import aalen_additive as aa
from surpyval.tests.reference._data import fixture
from surpyval.univariate.regression.additive_hazards.aalen_additive import (
    _invariance_statistics,
)
from surpyval.utils.weights import WeightedRefusal

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXACT = dict(rtol=1e-8, atol=1e-12)
DATA = Path(__file__).parent / "data" / "r_aalen_additive.json"


def _reference(ref_id):
    with open(DATA) as f:
        entry = json.load(f)["references"][ref_id]
    return {
        k: np.array(v, dtype=float) if k != "terms" else v
        for k, v in entry["values"].items()
    }


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


def _untie(d, time, status):
    """The ties broken as the R script breaks them."""
    rows = np.arange(d[time].size)
    order = np.lexsort((rows, -d[status], d[time]))
    d = {k: v[order] for k, v in d.items()}
    t = d[time]
    rank = np.zeros(t.size)
    for k in range(1, t.size):
        rank[k] = rank[k - 1] + 1 if t[k] == t[k - 1] else 0
    d[time] = t + 0.001 * rank
    return d


def _veteran():
    d = _untie(fixture("veteran"), "time", "status")
    Z = pd.DataFrame(
        {
            "trt": d["trt"],
            "karno": d["karno"],
            "age": d["age"],
            "cell": d["celltype"],
        }
    )
    return sp.AalenAdditive.fit(d["time"], Z, c=1 - d["status"])


def _additive():
    d = fixture("additive")
    Z = pd.DataFrame({"z1": d["z1"], "z2": d["z2"]})
    return sp.AalenAdditive.fit(d["x"], Z, c=d["c"])


def _right_only_ks(times, B):
    """timereg's supremum: over the event times only."""
    line = times[:, None] / times[-1] * B[-1][None, :]
    return np.abs(B - line).max(axis=0)


def _row(d, check, name=None):
    table = d.table[d.table["check"] == check]
    if name is not None:
        table = table[table["covariates"] == (name,)]
    return table.iloc[0]


def _sample(rng, n, early_z2):
    """``h = 0.1 + 0.08 z1 + early_z2 z2`` to t = 4, then ``0.1 + 0.08 z1
    + 0.1 z2``; follow-up ends uniformly on (2, 14) and at 12."""
    z1 = rng.binomial(1, 0.5, n).astype(float)
    z2 = rng.uniform(0.0, 1.0, n)
    early = 0.1 + 0.08 * z1 + early_z2 * z2
    late = 0.1 + 0.08 * z1 + 0.1 * z2
    e = rng.exponential(1.0, n)
    t = np.where(e < 4.0 * early, e / early, 4.0 + (e - 4.0 * early) / late)
    end = np.minimum(rng.uniform(2.0, 14.0, n), 12.0)
    Z = pd.DataFrame({"z1": z1, "z2": z2})
    return sp.AalenAdditive.fit(np.minimum(t, end), Z, c=(t > end).astype(int))


def _convex(rng, n):
    """A hazard convex in ``z``, which the additive line takes below 0
    near ``z = 1``."""
    z = rng.uniform(0.0, 1.0, n)
    t = rng.exponential(1.0 / (0.02 + 0.6 * (1.0 - z) ** 4))
    end = np.minimum(rng.uniform(2.0, 14.0, n), 12.0)
    return sp.AalenAdditive.fit(
        np.minimum(t, end),
        pd.DataFrame({"z": z}),
        c=(t > end).astype(int),
    )


# -- R references -------------------------------------------------------------
def test_statistics_match_timereg_where_the_estimate_reaches_the_end():
    model = _additive()
    ref = _reference("constant_effects_additive")
    assert_allclose(model.x, ref["time"], rtol=0, atol=0)
    assert_allclose(model.B, ref["cum"], **EXACT)
    table = model.constant_effects_test(n_sim=10, random_state=0)
    assert list(table.index) == ref["terms"]
    assert_allclose(table["CvM"], ref["CvM"], **EXACT)
    assert_allclose(_right_only_ks(model.x, model.B), ref["KS"], **EXACT)
    # SurPyval's supremum takes the left limits too: at least timereg's
    assert np.all(table["KS"].to_numpy() >= ref["KS"] - 1e-12)


def test_statistics_match_timereg_on_its_grid_veteran():
    model = _veteran()
    ref = _reference("constant_effects_veteran")
    # timereg runs its grid to the last event time and holds B there;
    # SurPyval's estimate stops where the risk set cannot separate the
    # covariates.
    assert ref["time"][-1] == 999 and model.x[-1] == 411
    B = model.cumulative_coefficients(ref["time"]).to_numpy()
    assert_allclose(B, ref["cum"], **EXACT)
    _, cvm = _invariance_statistics(ref["time"], B[None])
    assert_allclose(cvm[0], ref["CvM"], **EXACT)
    assert_allclose(_right_only_ks(ref["time"], B), ref["KS"], **EXACT)


def test_lung_differs_from_timereg_only_at_its_last_event_time():
    # SurPyval estimates the last step, where 4 units are at risk for the
    # 4 terms (an exact fit); timereg holds B there. Before it they agree.
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    d = {k: v[keep] for k, v in d.items()}
    d["status"] = 1 - d["c"]
    d = _untie(d, "time", "status")
    Z = pd.DataFrame({k: d[k] for k in ("age", "sex", "ph_ecog")})
    model = sp.AalenAdditive.fit(d["time"], Z, c=d["c"])
    ref = _reference("constant_effects_lung")
    assert_allclose(model.x, ref["time"], rtol=0, atol=0)
    assert_allclose(model.B[:-1], ref["cum"][:-1], **EXACT)
    assert_allclose(ref["cum"][-1], ref["cum"][-2], **EXACT)
    assert np.sum(d["time"] >= model.x[-1]) == 4
    assert np.abs(model.B[-1] - ref["cum"][-1]).max() > 1


# -- the report ---------------------------------------------------------------
def test_the_report_is_the_models_own_test():
    model = _veteran()
    d = model.diagnose(n_boot=500, random_state=3)
    assert isinstance(d, Diagnosis)
    assert d.family == "Aalen additive hazards"
    table = model.constant_effects_test(n_sim=500, random_state=3)
    pd.testing.assert_frame_equal(d.tables["constant_effects"], table)
    for name in ["trt", "karno", "age", "cell"]:
        row = _row(d, "constant_effects", name)
        assert_allclose(row["statistic"], table.loc[name, "KS"])
        assert_allclose(row["p"], table.loc[name, "KS p"])
        assert row["verdict"] == "ok"
    # Seeded: the same seed, the same report
    again = model.diagnose(n_boot=500, random_state=3)
    pd.testing.assert_frame_equal(d.table, again.table)


def test_a_changing_effect_is_reported_not_failed():
    # karno's effect fades (timereg's p is 0.008; this test's about 0.04)
    d = _veteran().diagnose(random_state=0)
    karno = _row(d, "constant_effects", "karno")
    assert karno["p"] < 0.05
    assert karno["verdict"] == "ok"
    assert "changes with time" in karno["detail"]
    advice = [a for a in d.advice if a.check == "constant_effects"]
    assert len(advice) == 1 and advice[0].covariates == ("karno",)
    assert "smoothed_coefficients" in advice[0]
    assert not any("AdditiveHazards.fit" in a for a in d.advice)


def test_constant_effects_point_to_the_lin_ying_model():
    # Constant effects; z2's test rejects here by chance (p = 0.027, a 1
    # in 20 event), so the advice is not given until only z1 is checked.
    model = _quiet(_sample, np.random.default_rng(41), 400, 0.1)
    d = model.diagnose(random_state=1)
    assert _row(d, "constant_effects", "z1")["p"] >= 0.05
    assert _row(d, "constant_effects", "z2")["p"] < 0.05
    assert not any("AdditiveHazards.fit" in a for a in d.advice)
    d = model.diagnose(columns=["z1"], random_state=1)
    advice = [a for a in d.advice if "AdditiveHazards.fit" in a]
    assert len(advice) == 1 and advice[0].features == ("AdditiveHazards.fit",)
    assert _row(d, "non_negative_hazard")["verdict"] == "ok"
    assert d.ok


def test_negative_fitted_hazards_fail_with_advice():
    model = _quiet(_convex, np.random.default_rng(42), 300)
    d = model.diagnose(random_state=2)
    row = _row(d, "non_negative_hazard")
    assert row["p"] < 0.05 and row["verdict"] == "fail"
    assert not d.ok
    advice = [a for a in d.advice if a.check == "non_negative_hazard"]
    assert len(advice) == 1
    assert advice[0].features == ("CoxPH.fit", "Hf")
    # Another additive model is not advised
    assert not any("AdditiveHazards.fit" in a for a in d.advice)
    # The units it names are those near z = 1
    res = aa.falls(model, n_sim=200, random_state=2)
    flagged = res["rows"][(res["row_p"] < 0.05), 0]
    assert flagged.size and flagged.min() > 0.5


def test_falls_are_drawdowns_below_the_running_maximum():
    model = _quiet(_convex, np.random.default_rng(43), 200)
    res = aa.falls(model, n_sim=50, random_state=0)
    H, H_star = res["H"], res["H_star"]
    assert np.all(H_star >= H - 1e-12) and np.all(H_star >= 0)
    assert np.all(np.diff(H_star, axis=0) >= 0)
    # H is B_0 + B'Z at the event times, for each distinct row
    X = np.column_stack([np.ones(res["rows"].shape[0]), res["rows"]])
    assert_allclose(H, model.B @ X.T, **EXACT)
    assert_allclose(res["largest"], (H_star - H).max())
    # The prediction is the running maximum
    k = H.shape[1] // 2
    assert_allclose(model.Hf(model.x, res["rows"][k]), H_star[:, k])
    assert 0 <= res["p"] <= 1 and 0 <= res["unit_share"] <= 1


def test_columns_and_alpha():
    model = _veteran()
    d = model.diagnose(columns=["karno"], alpha=0.01, random_state=0)
    rows = d.table[d.table["check"] == "constant_effects"]
    assert list(rows["covariates"]) == [("karno",)]
    assert d.columns == ["karno"]


def test_an_aliased_covariate_warns():
    rng = np.random.default_rng(44)
    model = _quiet(_sample, rng, 200, 0.1)
    x, c, n, tl, Z = model._fit_data
    Z = pd.DataFrame({"z1": Z[:, 0], "z2": Z[:, 1], "twice": 2 * Z[:, 1]})
    aliased = _quiet(sp.AalenAdditive.fit, x, Z, c=c)
    d = aliased.diagnose(n_boot=50, random_state=0)
    row = _row(d, "constant_effects", "twice")
    assert row["verdict"] == "warn" and "aliased" in row["detail"]


def test_plots_draw():
    d = _veteran().diagnose(n_boot=50, random_state=0)
    fig = d.plot()
    assert len(fig.axes) == 5
    fig = d.plot("non_negative_hazard")
    assert len(fig.axes) == 1
    plt.close("all")


# -- refusals -----------------------------------------------------------------
def test_a_weighted_fit_is_refused_saying_what_to_use():
    model = _veteran()
    x, c, n, tl, Z = model._fit_data
    w = np.random.default_rng(0).uniform(0.5, 2.0, x.size)
    weighted = _quiet(sp.AalenAdditive.fit, x, Z, c=c, w=w)
    with pytest.raises(WeightedRefusal, match="cumulative_coefficients_cb"):
        weighted.diagnose()


def test_a_restored_model_is_refused():
    back = sp.from_json(_additive().to_json())
    with pytest.raises(ValueError, match="refit with AalenAdditive.fit"):
        back.diagnose()


@pytest.mark.parametrize("n_boot", [0, 2.5, True])
def test_a_bad_n_boot_is_refused(n_boot):
    with pytest.raises(ValueError, match="n_boot"):
        _additive().diagnose(n_boot=n_boot)
