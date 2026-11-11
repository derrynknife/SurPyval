"""Event marks (competing-risks recurrent events).

An item can experience events of several mutually exclusive types over time.
``handle_xicn`` now carries a per-row mark ``e``; the nonparametric
``CauseSpecificMCF`` routes through it (gaining validation, sorting and
truncation), and the parametric ``CauseSpecificNHPP`` fits a separate intensity
model per cause. For a marked Poisson process the cause-specific processes are
independent thinned Poisson processes, so each cause's intensity is the NHPP
fit to that cause's events over the full observation window -- these tests
check that mark plumbing, the routing, and independent parameter recovery.
"""

import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np
import pytest

import surpyval as surv
from surpyval import handle_xicn
from surpyval.recurrent import HPP, CoxLewis, CrowAMSAA, Duane
from surpyval.recurrent.competing_risks import (
    CauseSpecificMCF,
    CauseSpecificNHPP,
)

# --- marks in handle_xicn ---------------------------------------------------


def test_handle_xicn_sorts_marks_with_events():
    x = np.array([3, 1, 2, 5])
    i = np.array([1, 1, 1, 1])
    c = np.array([0, 0, 0, 1])
    e = np.array(["B", "A", "A", None], dtype=object)
    data = handle_xicn(x, i, c, e=e)
    # marks travel with their event through the (i, x) sort
    assert list(data.x) == [1.0, 2.0, 3.0, 5.0]
    assert list(data.e) == ["A", "A", "B", None]
    assert data.event_types == ["A", "B"]


def test_handle_xicn_normalises_missing_marks_to_none():
    # NaN / pandas-NA style missing marks become the single None sentinel, so
    # they are not mistaken for a distinct cause.
    x = np.array([1.0, 2.0, 3.0])
    e = np.array(["A", np.nan, "A"], dtype=object)
    data = handle_xicn(x, np.array([1, 1, 1]), np.array([0, 0, 1]), e=e)
    assert data.event_types == ["A"]
    assert data.e[1] is None


def test_handle_xicn_marks_length_checked():
    with pytest.raises(ValueError, match="x and e must have the same length"):
        handle_xicn(np.array([1.0, 2.0]), e=np.array(["A"]))


def test_marks_rejected_with_windows():
    with pytest.raises(ValueError, match="does not support event-type"):
        handle_xicn(
            np.array([3.0]),
            np.array([1]),
            e=np.array(["A"]),
            windows={1: [(0, 10)]},
        )


# --- CauseSpecificMCF routed through handle_xicn ----------------------------


def test_cause_specific_mcf_counts_by_cause():
    x = [3, 1, 5, 2, 4, 6]
    i = [1, 1, 1, 2, 2, 2]
    c = [0, 0, 1, 0, 0, 1]
    e = ["A", "B", None, "A", "A", None]
    model = CauseSpecificMCF.fit(x, i, c, e=e)
    assert model.event_types == ["A", "B"]
    # three A events over two items -> MCF reaches 1.5; one B event -> 0.5
    assert np.isclose(model.mcf(6, "A"), 1.5)
    assert np.isclose(model.mcf(6, "B"), 0.5)


def test_cause_specific_mcf_requires_marks():
    with pytest.raises(ValueError, match="required for a cause-specific"):
        CauseSpecificMCF.fit([1, 2, 3], [1, 1, 1], [0, 0, 1])


def test_cause_specific_mcf_validates_via_handler():
    # Routing through handle_xicn means malformed input is now caught (here a
    # right-censoring row that is not an item's last row).
    with pytest.raises(ValueError, match="right censored"):
        CauseSpecificMCF.fit(
            [1, 2, 3],
            [1, 1, 1],
            c=[1, 0, 0],
            e=["A", "A", None],
        )


def test_cause_specific_mcf_from_df():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "t": [3, 1, 5, 2, 4, 6],
            "item": [1, 1, 1, 2, 2, 2],
            "cens": [0, 0, 1, 0, 0, 1],
            "mark": ["A", "B", np.nan, "A", "A", np.nan],
        }
    )
    model = CauseSpecificMCF.fit_from_df(
        df, "t", "mark", i_col="item", c_col="cens"
    )
    assert model.event_types == ["A", "B"]
    assert np.isclose(model.mcf(6, "A"), 1.5)
    assert model.df is df


# --- CauseSpecificNHPP: independent per-cause intensity recovery -------------


def _simulate_two_cause_marks(paramsA, paramsB, T, items, seed):
    truthA = CrowAMSAA.from_params(paramsA)
    truthB = CrowAMSAA.from_params(paramsB)
    dA = truthA.time_terminated_simulation_data(
        T, items=items, random_state=seed
    )
    dB = truthB.time_terminated_simulation_data(
        T, items=items, random_state=seed + 1
    )
    xs, ii, cc, ee = [], [], [], []
    for item in range(items):
        for xv in dA.x[(dA.i == item) & (dA.c == 0)]:
            xs.append(xv)
            ii.append(item)
            cc.append(0)
            ee.append("A")
        for xv in dB.x[(dB.i == item) & (dB.c == 0)]:
            xs.append(xv)
            ii.append(item)
            cc.append(0)
            ee.append("B")
        xs.append(T)
        ii.append(item)
        cc.append(1)
        ee.append(None)
    return (
        np.array(xs),
        np.array(ii),
        np.array(cc),
        np.array(ee, dtype=object),
    )


def test_cause_specific_nhpp_recovers_independent_intensities():
    x, i, c, e = _simulate_two_cause_marks(
        [6.0, 1.3], [10.0, 0.8], T=25.0, items=500, seed=1
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    assert model.event_types == ["A", "B"]
    assert np.allclose(model.models["A"].params, [6.0, 1.3], rtol=0.15)
    assert np.allclose(model.models["B"].params, [10.0, 0.8], rtol=0.15)


def test_cause_specific_nhpp_total_cif_is_sum_of_causes():
    x, i, c, e = _simulate_two_cause_marks(
        [6.0, 1.3], [10.0, 0.8], T=25.0, items=200, seed=3
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    grid = np.array([5.0, 12.0, 25.0])
    total = model.total_cif(grid)
    parts = model.cif(grid, "A") + model.cif(grid, "B")
    assert np.allclose(total, parts)


def test_cause_specific_nhpp_per_cause_models_carry_inference():
    x, i, c, e = _simulate_two_cause_marks(
        [6.0, 1.3], [10.0, 0.8], T=25.0, items=100, seed=5
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    # each per-cause model is a full ParametricRecurrenceModel
    a = model.models["A"]
    assert np.isfinite(a.aic())
    assert a.iif(10.0) > 0
    assert a.cif(25.0) > a.cif(5.0)


def test_cause_specific_nhpp_requires_marks():
    with pytest.raises(ValueError, match="required for a cause-specific"):
        CauseSpecificNHPP.fit([1, 2, 3], [1, 1, 1], [0, 0, 1])


def test_cause_specific_nhpp_rejects_interval_censoring():
    data = handle_xicn(
        np.array([[1.0, 2.0], [3.0, 3.0]]),
        np.array([1, 1]),
        c=np.array([2, 1]),
        n=np.array([1, 1]),
        e=np.array(["A", None], dtype=object),
    )
    with pytest.raises(ValueError, match="exact"):
        CauseSpecificNHPP.fit_from_recurrent_data(data)


def test_cause_specific_nhpp_item_with_no_events_of_a_cause():
    # Item 2 never fails from cause B, but its window still contributes to B's
    # compensator (so B's intensity is not overestimated). The fit must run and
    # give a smaller B intensity than a data set where both items fail from B.
    x = [2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
    i = [1, 1, 1, 2, 2, 2]
    c = [0, 0, 1, 0, 0, 1]
    e = ["A", "B", None, "A", "A", None]  # only item 1 has a B event
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    assert set(model.event_types) == {"A", "B"}
    # B seen once over two full windows -> positive but small intensity
    cif_b = float(np.asarray(model.cif(8.0, "B")))
    cif_a = float(np.asarray(model.cif(8.0, "A")))
    assert cif_b > 0
    assert cif_b < cif_a


# --- CauseSpecificNHPP: bounds per cause, totals, growth and forecasts ------


def _hand_case():
    # Two items observed to 10 (exposure 20): five "a" events, three "b".
    x = [1.0, 2.0, 4.0, 5.0, 7.0, 3.0, 6.0, 8.0, 10.0, 10.0]
    i = [1, 1, 2, 2, 1, 2, 1, 2, 1, 2]
    c = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    e = ["a", "a", "a", "a", "a", "b", "b", "b", None, None]
    return x, i, c, e


def _poisson_quantile(mean, q):
    # The smallest count whose cumulative probability reaches q.
    k, term = 0, math.exp(-mean)
    total = term
    while total < q:
        k += 1
        term *= mean / k
        total += term
    return k


def test_hpp_per_mode_bounds_totals_and_forecast_by_hand():
    # A constant rate per mode is events / exposure, with variance
    # rate^2 / events (the inverse information of the Poisson likelihood),
    # so on the log scale a mode's bounds are its estimate times
    # exp(+-z / sqrt(events)). The total's variance is the sum, N / E^2
    # for N events of any kind, so its bounds are the total times
    # exp(+-z / sqrt(N)): those of one HPP fitted to every event.
    x, i, c, e = _hand_case()
    model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e, dist=HPP)
    exposure = Fraction(20)
    events = {"a": 5, "b": 3}
    rate = {k: Fraction(n) / exposure for k, n in events.items()}
    assert rate == {"a": Fraction(1, 4), "b": Fraction(3, 20)}
    total_rate = rate["a"] + rate["b"]
    assert total_rate == Fraction(2, 5)
    z = NormalDist().inv_cdf(0.975)
    t = 8.0

    for k, n in events.items():
        lam = float(rate[k])
        assert model.models[k].params[0] == pytest.approx(lam, rel=1e-12)
        spread = math.exp(z / math.sqrt(n))
        assert model.cif_cb(t, k) == pytest.approx(
            [t * lam / spread, t * lam * spread], rel=1e-6
        )
        assert model.iif_cb(t, k) == pytest.approx(
            [lam / spread, lam * spread], rel=1e-6
        )
        assert float(model.mtbf(t, k)) == pytest.approx(1 / lam, rel=1e-12)
        assert model.mtbf_cb(t, k) == pytest.approx(
            [1 / (lam * spread), spread / lam], rel=1e-6
        )

    lam = float(total_rate)
    spread = math.exp(z / math.sqrt(8))
    assert float(model.total_cif(t)) == pytest.approx(t * lam, rel=1e-12)
    assert float(model.total_iif(t)) == pytest.approx(lam, rel=1e-12)
    assert float(model.total_mtbf(t)) == pytest.approx(2.5, rel=1e-12)
    assert model.total_cif_cb(t) == pytest.approx(
        [t * lam / spread, t * lam * spread], rel=1e-6
    )
    assert model.total_iif_cb(t) == pytest.approx(
        [lam / spread, lam * spread], rel=1e-6
    )
    assert model.total_mtbf_cb(t) == pytest.approx(
        [1 / (lam * spread), spread / lam], rel=1e-6
    )
    # one-sided at 10%: z of 0.90
    z90 = NormalDist().inv_cdf(0.90)
    assert float(
        model.total_cif_cb(t, alpha_ci=0.1, bound="lower")
    ) == pytest.approx(t * lam * math.exp(-z90 / math.sqrt(8)), rel=1e-6)
    # the pooled fit of every event, whatever its mode
    pooled = HPP.fit(x, i=i, c=c)
    assert model.total_cif_cb(t) == pytest.approx(pooled.cif_cb(t), rel=1e-6)

    # The next 5 time units of the two items (each at 10): 2 * 5 * rate.
    mean = {k: float(2 * 5 * rate[k]) for k in events}
    assert mean == {"a": 2.5, "b": 1.5}
    for k in events:
        f = model.forecast(horizon=5.0, event=k)
        assert f.expected == pytest.approx([mean[k]], rel=1e-12)
        assert f.lower[0] == _poisson_quantile(mean[k], 0.025)
        assert f.upper[0] == _poisson_quantile(mean[k], 0.975)
    f = model.forecast(horizon=5.0)
    assert f.expected == pytest.approx([4.0], rel=1e-12)
    assert (f.lower[0], f.upper[0]) == (1.0, 8.0)
    assert f.upper[0] == _poisson_quantile(4.0, 0.975)
    assert f.lower[0] == _poisson_quantile(4.0, 0.025)
    assert list(f.units) == [1, 2]


def test_growth_refuses_a_baseline_with_no_trend_parameter():
    x, i, c, e = _hand_case()
    model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e, dist=HPP)
    with pytest.raises(ValueError, match="dist=HPP") as info:
        model.growth()
    assert "Refit with dist=CrowAMSAA" in str(info.value)
    # the constant rates are still summarised
    table = model.summary()
    assert list(table.index) == [("a", "lambda"), ("b", "lambda")]
    assert table["estimate"].to_numpy() == pytest.approx([0.25, 0.15])
    assert table["se"].to_numpy() == pytest.approx(
        [0.25 / math.sqrt(5), 0.15 / math.sqrt(3)], rel=1e-6
    )


def _simulate_marks(truth, T, items, seed):
    """Marked data from one Crow-AMSAA process per cause (the package's
    simulator, a seed per cause), every item observed to ``T``."""
    xs, ii, cc, ee = [], [], [], []
    for k, (cause, params) in enumerate(truth.items()):
        d = CrowAMSAA.from_params(params).time_terminated_simulation_data(
            T, items=items, random_state=seed + k
        )
        seen = d.c == 0
        xs.extend(d.x[seen])
        ii.extend(d.i[seen])
        cc.extend([0] * int(seen.sum()))
        ee.extend([cause] * int(seen.sum()))
        ids = np.unique(d.i)
    xs.extend([T] * len(ids))
    ii.extend(ids)
    cc.extend([1] * len(ids))
    ee.extend([None] * len(ids))
    return (
        np.array(xs, dtype=float),
        np.array(ii),
        np.array(cc),
        np.array(ee, dtype=object),
    )


# Three modes of 30 items observed to 100, about 8 events of each a item:
# one improving, one level, one worsening.
_TRUTH = {
    "improving": [3.125, 0.6],
    "level": [12.5, 1.0],
    "worsening": [31.5, 1.8],
}
_T = 100.0


@pytest.fixture(scope="module")
def three_modes():
    x, i, c, e = _simulate_marks(_TRUTH, _T, items=30, seed=20261111)
    return CauseSpecificNHPP.fit(x, i, c, e=e)


def test_recovery_of_three_modes_within_their_bounds(three_modes):
    # Seed, sizes and the 99% level fixed before the result was seen.
    model = three_modes
    alpha = 0.01
    level = "99%"
    table = model.growth(alpha_ci=alpha)
    assert list(table.index) == ["improving", "level", "worsening"]
    truth = {k: CrowAMSAA.from_params(p) for k, p in _TRUTH.items()}
    for cause, (_, beta) in _TRUTH.items():
        row = table.loc[cause]
        assert row["parameter"] == "beta"
        assert row["lower " + level] < beta < row["upper " + level], row
        rate = float(truth[cause].iif(_T))
        assert row["iif lower " + level] < rate < row["iif upper " + level]
        assert (
            row["mtbf lower " + level] < 1 / rate < row["mtbf upper " + level]
        )
        lo, hi = model.cif_cb(_T, cause, alpha_ci=alpha)
        assert lo < float(truth[cause].cif(_T)) < hi
    assert list(table["trend"]) == ["decreasing", "none", "increasing"]

    total_cif = sum(float(m.cif(_T)) for m in truth.values())
    total_iif = sum(float(m.iif(_T)) for m in truth.values())
    lo, hi = model.total_cif_cb(_T, alpha_ci=alpha)
    assert lo < total_cif < hi
    lo, hi = model.total_iif_cb(_T, alpha_ci=alpha)
    assert lo < total_iif < hi
    lo, hi = model.total_mtbf_cb(_T, alpha_ci=alpha)
    assert lo < 1 / total_iif < hi
    # the worsening mode carries most of the intensity at the end
    assert table["share"].idxmax() == "worsening"


def test_modes_sum_to_the_totals(three_modes):
    model = three_modes
    xt = np.array([5.0, 40.0, 100.0, 150.0])
    causes = model.event_types
    assert np.allclose(
        model.total_cif(xt), sum(model.cif(xt, k) for k in causes), rtol=1e-12
    )
    assert np.allclose(
        model.total_iif(xt), sum(model.iif(xt, k) for k in causes), rtol=1e-12
    )
    assert np.allclose(
        model.total_mtbf(xt),
        1 / sum(1 / model.mtbf(xt, k) for k in causes),
        rtol=1e-12,
    )
    # the total's variance on the natural scale is the sum of the modes'
    z = NormalDist().inv_cdf(0.975)
    total = model.total_cif(xt)
    se = np.log(model.total_cif_cb(xt)[:, 1] / total) * total / z
    var = 0.0
    for k in causes:
        est = model.cif(xt, k)
        var = var + (np.log(model.cif_cb(xt, k)[:, 1] / est) * est / z) ** 2
    assert np.allclose(se**2, var, rtol=1e-9)


def test_bounds_through_the_model_are_the_modes_own(three_modes):
    model = three_modes
    xt = np.array([10.0, 100.0])
    for cause in model.event_types:
        own = model.models[cause]
        for bound in ("two-sided", "lower", "upper"):
            kw = {"alpha_ci": 0.1, "bound": bound}
            assert np.array_equal(
                model.cif_cb(xt, cause, **kw), own.cif_cb(xt, **kw)
            )
            assert np.array_equal(
                model.iif_cb(xt, cause, **kw), own.iif_cb(xt, **kw)
            )
            assert np.array_equal(
                model.mtbf_cb(xt, cause, **kw), own.mtbf_cb(xt, **kw)
            )
        assert np.array_equal(model.mtbf(xt, cause), own.mtbf(xt))
        # Crow's exact bounds at the end of this time-terminated test
        assert np.array_equal(
            model.mtbf_cb(_T, cause, method="crow"),
            own.mtbf_cb(_T, method="crow"),
        )
    with pytest.raises(ValueError, match="Unknown cause 'x'"):
        model.cif_cb(xt, "x")
    with pytest.raises(ValueError, match="Unknown cause 'x'"):
        model.forecast(horizon=1.0, event="x")
    with pytest.raises(ValueError, match="alpha_ci"):
        model.total_cif_cb(xt, alpha_ci=1.5)
    with pytest.raises(ValueError, match="bound"):
        model.total_iif_cb(xt, bound="both")


@pytest.mark.parametrize("dist", [CrowAMSAA, Duane, CoxLewis, HPP])
def test_one_mode_total_bounds_equal_that_modes(dist):
    x, i, c, e = _simulate_marks({"only": [8.0, 1.4]}, 60.0, 6, seed=7)
    model = CauseSpecificNHPP.fit(x, i, c, e=e, dist=dist)
    xt = np.array([[5.0, 30.0], [60.0, 90.0]])
    for bound in ("two-sided", "lower", "upper"):
        kw = {"alpha_ci": 0.2, "bound": bound}
        pairs = [
            (model.total_cif_cb(xt, **kw), model.cif_cb(xt, "only", **kw)),
            (model.total_iif_cb(xt, **kw), model.iif_cb(xt, "only", **kw)),
            (model.total_mtbf_cb(xt, **kw), model.mtbf_cb(xt, "only", **kw)),
        ]
        for total, own in pairs:
            assert total.shape == own.shape
            assert total.shape == xt.shape + ((2,) if bound[0] == "t" else ())
            assert np.allclose(total, own, rtol=1e-12, atol=0)
    assert np.shape(model.total_cif_cb(30.0)) == (2,)
    assert np.shape(model.total_mtbf_cb(30.0, bound="lower")) == ()


@pytest.mark.parametrize(
    "dist, name, null", [(Duane, "alpha", 1.0), (CoxLewis, "beta", 0.0)]
)
def test_growth_reads_each_baselines_trend_parameter(dist, name, null):
    truth = {"down": [3.125, 0.6], "up": [31.5, 1.8]}
    x, i, c, e = _simulate_marks(truth, 100.0, 10, seed=11)
    model = CauseSpecificNHPP.fit(x, i, c, e=e, dist=dist)
    table = model.growth()
    assert list(table["parameter"]) == [name, name]
    assert list(table["no trend"]) == [null, null]
    assert list(table["trend"]) == ["decreasing", "increasing"]
    for cause in model.event_types:
        own = model.models[cause]
        row = table.loc[cause]
        assert row[["lower 95%", "upper 95%"]].to_numpy(
            float
        ) == pytest.approx(own.param_cb(name))


def test_growth_table_is_consistent_with_the_bounds(three_modes):
    model = three_modes
    for alpha in (0.05, 0.2):
        level = f"{100 * (1 - alpha):g}%"
        table = model.growth(alpha_ci=alpha)
        assert table["share"].sum() == pytest.approx(1.0)
        for cause in model.event_types:
            own = model.models[cause]
            row = table.loc[cause]
            interval = own.param_cb("beta", alpha_ci=alpha)
            assert [row["lower " + level], row["upper " + level]] == (
                pytest.approx(interval)
            )
            # "none" exactly when the interval holds the no-trend value
            inside = interval[0] <= 1.0 <= interval[1]
            assert (row["trend"] == "none") == inside
            assert (row["p_value"] >= alpha) == inside
            assert [
                row["iif lower " + level],
                row["iif upper " + level],
            ] == pytest.approx(own.iif_cb(_T, alpha_ci=alpha))
            assert [
                row["mtbf lower " + level],
                row["mtbf upper " + level],
            ] == pytest.approx(own.mtbf_cb(_T, alpha_ci=alpha))
            assert row["share"] == pytest.approx(
                float(own.iif(_T) / model.total_iif(_T))
            )
    # read at another time
    early = model.growth(x=20.0)
    assert early.loc["level", "iif"] == pytest.approx(
        float(model.iif(20.0, "level"))
    )
    assert early["share"].idxmax() == "improving"
    with pytest.raises(ValueError, match="x must be one finite time"):
        model.growth(x=[10.0, 20.0])


def test_growth_and_summary_keep_tuple_labels():
    truth = {("pump", 1): [8.0, 1.4], ("pump", 2): [10.0, 0.8]}
    x, i, c, e = _simulate_marks(truth, 60.0, 6, seed=3)
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    assert list(model.growth().index) == [("pump", 1), ("pump", 2)]
    summary = model.summary()
    assert summary.index.names == ["event", "parameter"]
    assert list(summary.index) == [
        (("pump", 1), "alpha"),
        (("pump", 1), "beta"),
        (("pump", 2), "alpha"),
        (("pump", 2), "beta"),
    ]


def test_summary_stacks_the_modes_summaries(three_modes):
    model = three_modes
    table = model.summary(alpha_ci=0.1)
    assert list(table.columns) == ["estimate", "se", "lower 90%", "upper 90%"]
    for cause in model.event_types:
        own = model.models[cause].summary(alpha_ci=0.1)
        assert np.allclose(table.loc[cause].to_numpy(), own.to_numpy())
        assert list(table.loc[cause].index) == list(own.index)


def test_forecasts_of_the_modes_sum_to_the_total(three_modes):
    model = three_modes
    horizon = [10.0, 25.0]
    every = model.forecast(horizon=horizon, alpha_ci=0.1)
    each = [
        model.forecast(horizon=horizon, event=k) for k in model.event_types
    ]
    assert np.allclose(every.expected, sum(f.expected for f in each))
    assert np.allclose(every.per_unit, sum(f.per_unit for f in each))
    assert np.allclose(
        every.period_expected, sum(f.period_expected for f in each)
    )
    # 30 fitted items, each from the end of its observation
    assert every.units.size == 30
    gained = model.total_cif(_T + np.array(horizon)) - model.total_cif(_T)
    assert np.allclose(every.expected, 30 * gained, rtol=1e-12)
    # the Poisson interval of the total's mean
    assert every.upper[0] == _poisson_quantile(every.expected[0], 0.95)
    assert every.lower[0] == _poisson_quantile(every.expected[0], 0.05)
    assert np.all(every.variance == every.expected)

    # surpyval.forecast reads the model as its forecast method does
    same = surv.forecast(model, horizon=horizon, alpha_ci=0.1)
    assert np.array_equal(same.expected, every.expected)
    assert np.array_equal(same.upper, every.upper)

    # given ages, counts and a limit, as the single-process forecast
    kw = {"age": [0.0, 50.0], "horizon": 20.0, "n": [3, 2], "limit": 60.0}
    every = model.forecast(**kw)
    each = [model.forecast(event=k, **kw) for k in model.event_types]
    own = [surv.forecast(model.models[k], **kw) for k in model.event_types]
    assert np.allclose(every.expected, sum(f.expected for f in each))
    for f, g in zip(each, own):
        assert np.array_equal(f.expected, g.expected)
    expected = 3 * model.total_cif(20.0) + 2 * (
        model.total_cif(60.0) - model.total_cif(50.0)
    )
    assert every.expected == pytest.approx([float(expected)], rel=1e-12)
    with pytest.raises(ValueError, match="takes no covariates"):
        surv.forecast(model, age=[1.0], horizon=1.0, Z=[[1.0]])
    with pytest.raises(ValueError, match="horizon is required"):
        model.forecast()


def test_mse_fit_refuses_bounds_saying_how_to_get_them():
    x, i, c, e = _simulate_marks(
        {"A": [8.0, 1.4], "B": [10.0, 0.8]}, 60.0, 6, seed=5
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e, how="MSE")
    for call in (
        lambda: model.total_cif_cb(10.0),
        lambda: model.cif_cb(10.0, "A"),
        lambda: model.growth(),
        lambda: model.summary(),
    ):
        with pytest.raises(ValueError, match="refit with how='MLE'"):
            call()
    # the point values and the forecast need no likelihood
    assert float(model.total_iif(10.0)) > 0
    assert model.forecast(horizon=5.0).expected[0] > 0
