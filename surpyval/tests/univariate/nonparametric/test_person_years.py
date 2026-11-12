"""surpyval.person_years: the exposure table's cells are what a hand count
gives, counts equal repetition, row order and the time unit change
nothing, the bounds and ratios are the closed forms, it round-trips
through strict JSON, and what it cannot count is refused with the fix.
R's ``pyears`` is compared in tests/reference/test_person_years.py."""

import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import surpyval as sp
from surpyval import PersonYears, person_years, poisson_rate_bounds

X = np.array([1500.0, 4200.0, 2500.0, 6000.0, 800.0])
C = np.array([0, 1, 0, 0, 1])
SITE = np.array(["A", "A", "B", "B", "A"])
AGE0 = np.array([0.0, 0.0, 1000.0, 0.0, 3500.0])
BREAKS = [0, 2000, 4000, 8000]


def _table(**kw):
    args = dict(
        factors={"site": SITE},
        timescales={"age": (AGE0, BREAKS)},
    )
    args.update(kw)
    return person_years(X, C, **args)


def test_cells_are_the_hand_count():
    py = _table()
    t = py.table.set_index(["site", "age"])
    # A: (0, 1500] event in 0-2000; (0, 4200] censored; age 3500 + (0, 800]
    assert t.loc[("A", 0.0), "exposure"] == 1500 + 2000
    assert t.loc[("A", 2000.0), "exposure"] == 2000 + 500
    assert t.loc[("A", 4000.0), "exposure"] == 200 + 300
    assert t.loc[("A", 0.0), "events"] == 1
    assert t.loc[("A", 2000.0), "units"] == 2
    # B: age 1000 + (0, 2500] event at age 3500; (0, 6000] event at 6000
    assert t.loc[("B", 0.0), "exposure"] == 1000 + 2000
    assert t.loc[("B", 2000.0), "exposure"] == 1500 + 2000
    assert t.loc[("B", 4000.0), "exposure"] == 2000
    assert t.loc[("B", 2000.0), "events"] == 1
    assert t.loc[("B", 4000.0), "events"] == 1
    assert py.units == 5
    assert t["exposure"].sum() == X.sum()
    assert list(py.table.columns[:3]) == ["site", "age", "age_end"]


def test_an_event_on_an_edge_is_in_the_band_ending_there():
    py = person_years([2000.0], [0], timescales={"age": (0, BREAKS)})
    assert py.table["events"].tolist() == [1, 0, 0]
    assert py.table["exposure"].tolist() == [2000.0, 0.0, 0.0]


def test_counts_equal_repetition_and_order_does_not_matter():
    n = np.array([2, 1, 3, 1, 2])
    py = _table(n=n)
    rep = np.repeat(np.arange(5), n)
    repeated = person_years(
        X[rep],
        C[rep],
        factors={"site": SITE[rep]},
        timescales={"age": (AGE0[rep], BREAKS)},
    )
    pd.testing.assert_frame_equal(py.table, repeated.table)
    order = np.array([3, 0, 4, 2, 1])
    shuffled = person_years(
        X[order],
        C[order],
        n=n[order],
        factors={"site": SITE[order]},
        timescales={"age": (AGE0[order], BREAKS)},
    )
    pd.testing.assert_frame_equal(py.table, shuffled.table)


def test_rescaling_time_rescales_exposure_and_rates():
    py = _table()
    k = 3.5
    scaled = person_years(
        X * k,
        C,
        factors={"site": SITE},
        timescales={"age": (AGE0 * k, np.array(BREAKS) * k)},
    )
    assert_allclose(scaled.table["exposure"], py.table["exposure"] * k)
    assert_allclose(scaled.table["rate"], py.table["rate"] / k)
    assert_allclose(scaled.table["rate_upper"], py.table["rate_upper"] / k)


def test_per_scales_the_rates_only():
    py, per = _table(), _table(per=1000)
    assert_allclose(per.table["rate"], 1000 * py.table["rate"])
    assert_allclose(per.table["rate_lower"], 1000 * py.table["rate_lower"])
    assert_allclose(per.table["exposure"], py.table["exposure"])


def test_garwood_bounds_are_the_chi_square_quantiles():
    lo, hi = poisson_rate_bounds([0, 4], [10.0, 20.0], alpha_ci=0.1)
    assert lo[0] == 0
    assert_allclose(hi[0], stats.chi2.ppf(0.95, 2) / 2 / 10)
    assert_allclose(lo[1], stats.chi2.ppf(0.05, 8) / 2 / 20)
    assert_allclose(hi[1], stats.chi2.ppf(0.95, 10) / 2 / 20)
    lo, hi = poisson_rate_bounds(1, 0.0)
    assert np.isnan(lo) and np.isnan(hi)


def test_cells_with_no_time_have_no_rate():
    py = _table(timescales={"age": (AGE0, [0, 2000, 4000, 8000, 9000])})
    empty = py.table["exposure"] == 0
    assert empty.any()
    assert py.table.loc[empty, "rate"].isna().all()
    assert py.table.loc[empty, "units"].eq(0).all()


def test_late_entry_counts_time_from_entry():
    py = person_years([10.0], [0], tl=[4.0], timescales={"t": (0, [0, 5, 20])})
    assert py.table["exposure"].tolist() == [1.0, 5.0]
    since = person_years(
        [10.0], [0], tl=[4.0], timescales={"since": (-4.0, [0, 2, 20])}
    )
    assert since.table["exposure"].tolist() == [2.0, 4.0]


def test_time_off_the_bands_is_reported_with_a_warning():
    with pytest.warns(UserWarning, match="outside the bands"):
        py = person_years([10.0, 3.0], [0, 0], timescales={"t": (0, [2, 5])})
    assert py.offtable == {"exposure": 9.0, "events": 1}
    assert py.table["exposure"].tolist() == [4.0]
    assert py.table["events"].tolist() == [1]


def test_margins_add_up_and_recompute_rates():
    py = _table()
    by_site = py.margins("site")
    assert by_site["exposure"].tolist() == [
        py.table.loc[py.table["site"] == s, "exposure"].sum() for s in "AB"
    ]
    assert "units" not in by_site.columns
    total = py.margins().iloc[0]
    assert total["exposure"] == X.sum()
    assert_allclose(total["rate"], total["events"] / total["exposure"])
    by_age = py.margins(["age"])
    assert by_age["age_end"].tolist() == [2000.0, 4000.0, 8000.0]
    with pytest.raises(ValueError, match="dimensions are"):
        py.margins("region")


def test_categorical_factor_keeps_its_order():
    cat = pd.Categorical(SITE, categories=["B", "A"])
    py = _table(factors={"site": cat})
    assert list(py.table["site"].cat.categories) == ["B", "A"]
    assert py.table["site"].iloc[0] == "B"


def test_no_dimensions_gives_one_total_cell():
    py = person_years(X, C)
    assert len(py.table) == 1
    assert py.table["exposure"].iloc[0] == X.sum()
    assert py.table["events"].iloc[0] == 3


def test_rate_ratio_is_the_conditional_binomial():
    py = _table()
    rr = py.rate_ratio({"site": "B"}, {"site": "A"})
    d1, d0 = rr["events"], rr["reference_events"]
    T1, T0 = rr["exposure"], rr["reference_exposure"]
    assert_allclose(rr["ratio"], (d1 / T1) / (d0 / T0))
    lo = stats.beta.ppf(0.025, d1, d0 + 1)
    assert_allclose(rr["lower"], lo / (1 - lo) * T0 / T1)
    p = stats.binomtest(int(d1), int(d1 + d0), T1 / (T1 + T0)).pvalue
    assert_allclose(rr["p_value"], p)
    one_cell = py.rate_ratio(
        {"site": "B", "age": 4000.0}, {"site": "A", "age": 0.0}
    )
    assert one_cell["events"] == 1 and one_cell["exposure"] == 2000


def test_rate_ratio_with_no_events_is_unbounded():
    py = person_years([5.0, 6.0], [1, 1], factors={"g": ["a", "b"]})
    rr = py.rate_ratio({"g": "a"}, {"g": "b"})
    assert np.isnan(rr["ratio"]) and rr["lower"] == 0
    assert rr["upper"] == np.inf and rr["p_value"] == 1


def test_rate_ratio_refuses_overlapping_or_unknown_cells():
    py = _table()
    with pytest.raises(ValueError, match="in common"):
        py.rate_ratio({"site": "A"}, {"age": 0.0})
    with pytest.raises(ValueError, match="not a level"):
        py.rate_ratio({"site": "C"}, {"site": "A"})
    with pytest.raises(ValueError, match="not a dimension"):
        py.rate_ratio({"region": "C"}, {"site": "A"})
    with pytest.raises(ValueError, match="dict of labels"):
        py.rate_ratio("A", {"site": "A"})


def test_expected_events_and_observed_to_expected():
    rates = pd.DataFrame({"age": [0.0, 2000.0, 4000.0], "rate": [1, 2, 4]})
    py = _table(rates=rates, per=1000)
    t = py.table
    assert_allclose(
        t["expected"],
        t["age"].map({0: 1, 2000: 2, 4000: 4}) * t["exposure"] / 1000,
    )
    assert_allclose(t["oe_ratio"], t["events"] / t["expected"])
    lo, hi = poisson_rate_bounds(t["events"], t["expected"])
    assert_allclose(t["oe_upper"], hi)
    scalar = _table(rates=2.0, per=1000)
    assert_allclose(scalar.table["expected"], 2 * t["exposure"] / 1000)
    as_dict = _table(
        rates={"age": [0, 2000, 4000], "rate": [1, 2, 4]}, per=1000
    )
    assert_allclose(as_dict.table["expected"], t["expected"])
    total = py.margins().iloc[0]
    assert_allclose(total["oe_ratio"], 3 / t["expected"].sum())
    assert "Observed / expected" in repr(py)


def test_rates_that_do_not_cover_the_cells_are_refused():
    with pytest.raises(ValueError, match="no rate for 2 cells"):
        _table(rates={"age": [0.0, 2000.0], "rate": [1, 2]})
    with pytest.raises(ValueError, match="more than one rate"):
        _table(rates={"age": [0.0, 0.0, 2000, 4000], "rate": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="not factors or time scales"):
        _table(rates={"region": ["x"], "rate": [1]})
    with pytest.raises(ValueError, match="'rate' column"):
        _table(rates={"age": [0.0]})
    with pytest.raises(ValueError, match="at least 0"):
        _table(rates=-1.0)


def test_round_trips_through_strict_json():
    rates = pd.DataFrame({"site": ["A", "B"], "rate": [0.5, 0.7]})
    with pytest.warns(UserWarning):
        py = _table(
            rates=rates,
            per=1000,
            timescales={"age": (AGE0, [500, 2000, 4000, np.inf])},
        )
    d = py.to_dict()
    text = json.dumps(d, allow_nan=False)
    back = PersonYears.from_dict(json.loads(text))
    pd.testing.assert_frame_equal(back.table, py.table)
    assert back.offtable == py.offtable
    assert back.units == py.units
    assert (back.factors, back.timescales) == (py.factors, py.timescales)
    assert back.rate_ratio({"site": "B"}, {"site": "A"}) == py.rate_ratio(
        {"site": "B"}, {"site": "A"}
    )
    with pytest.raises(ValueError, match="to_dict wrote"):
        PersonYears.from_dict({"result": "Other"})
    with pytest.raises(ValueError, match="no 'table'"):
        PersonYears.from_dict({k: v for k, v in d.items() if k != "table"})


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(c=[0, 2, 1, 0, 0]), "interval .c=2. censored"),
        (dict(c=[0, -1, 1, 0, 0]), "Turnbull"),
        (dict(tl=[0, 0, 3000, 0, 0]), "at or before their entry"),
        (dict(n=[1, 0, 1, 1, 1]), "positive counts"),
        (dict(c=[0, 3, 1, 0, 0]), "0 .event. or 1"),
        (dict(per=0), "'per' must be a positive"),
        (dict(alpha_ci=1.2), "alpha_ci"),
        (dict(factors={"site": ["A"] * 4}), "one per row"),
        (dict(factors={"site": ["A", None, "B", "B", "A"]}), "missing for 1"),
        (dict(factors={"rate": SITE}), "column of the table"),
        (dict(factors=["site"]), "must be a dict"),
        (dict(timescales={"age": [0, 10]}), r"\(start, breaks\)"),
        (dict(timescales={"age": (0, [5, 1])}), "increasing"),
        (dict(timescales={"age": {"start": 0}}), "needs 'breaks'"),
        (
            dict(timescales={"age": ([0, 1, np.nan, 2, 3], BREAKS)}),
            "missing or infinite for 1",
        ),
    ],
)
def test_refusals_name_the_argument_and_the_fix(kwargs, match):
    args = dict(c=C, factors={"site": SITE})
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        person_years(X, **args)


def test_interval_rows_and_surpyval_data_are_refused():
    with pytest.raises(ValueError, match="inspection"):
        person_years([[1.0, 2.0], [3.0, 4.0]], [2, 2])
    with pytest.raises(TypeError, match="sorted"):
        person_years(sp.SurpyvalData(X, C))
