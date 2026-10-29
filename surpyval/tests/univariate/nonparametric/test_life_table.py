"""surpyval.life_table: the counts per interval are what a hand count and
lifelines give, and the actuarial estimates are Klein and Moeschberger's
(2003, section 5.4) formulas, as R's ``KMsurv::lifetab`` computes them,
evaluated here independently in exact arithmetic."""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp

# A small right censored sample with late entry, untied with the
# boundaries 0, 2, 4, 6, 8 and 10.
X = [0.7, 1.3, 1.9, 2.2, 2.6, 3.1, 3.4, 4.8, 5.5, 6.2, 7.9, 9.4]
C = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
TL = [0, 0, 1.5, 0, 0, 2.5, 0, 0, 0, 3.0, 0, 0]
EDGES = [0, 2, 4, 6, 8, 10]


def _lifetab(edges, nsubs_first, entered, nlost, nevent):
    """Klein and Moeschberger's equations 5.4.1-5.4.6 (``lifetab``), in
    exact arithmetic, with a late entrant counted as half at risk, as a
    withdrawal is."""
    out = []
    at_start = Fraction(nsubs_first)
    S = Fraction(1)
    greenwood = Fraction(0)
    for j in range(len(nevent)):
        Y = at_start - Fraction(nlost[j], 2) + Fraction(entered[j], 2)
        d = Fraction(nevent[j])
        q = d / Y
        p = 1 - q
        var_S = S**2 * greenwood
        f = var_f = h = var_h = None
        if np.isfinite(edges[j + 1]) and q > 0:
            b = Fraction(edges[j + 1]) - Fraction(edges[j])
            f = S * q / b
            h = d / (b * (Y - d / 2))
            var_f = (S * q / b) ** 2 * (greenwood + p / (Y * q))
            var_h = h**2 / (Y * q) * (1 - (h * b / 2) ** 2)
        out.append((at_start, Y, q, S, var_S, f, var_f, h, var_h))
        greenwood += q / (Y * p) if p > 0 else 0
        S *= p
        at_start += entered[j] - nlost[j] - nevent[j]
    return out


def test_counts_match_lifelines_survival_table():
    # lifelines 0.30.3:
    #   survival_table_from_events(x, 1 - c, collapse=True,
    #                              intervals=[0, 2, 4, 6, 8, 10])
    # gives observed [2, 3, 1, 2, 0], censored [1, 1, 1, 0, 1] and
    # at_risk [12, 9, 5, 3, 1]. Its intervals are (a, b], ours [a, b);
    # no time is on a boundary, so they agree.
    table = sp.life_table(X, C, intervals=EDGES)
    assert_array_equal(table["events"], [2, 3, 1, 2, 0])
    assert_array_equal(table["censored"], [1, 1, 1, 0, 1])
    assert_array_equal(table["at_risk_start"], [12, 9, 5, 3, 1])
    assert_array_equal(table["entered"], 0)
    assert_array_equal(table["start"], EDGES[:-1])
    assert_array_equal(table["end"], EDGES[1:])


def test_late_entry_counts_match_lifelines():
    # lifelines 0.30.3, survival_table_from_events(x, 1 - c,
    # birth_times=tl): entrance 9 at 0 and 1 each at 1.5, 2.5 and 3.0;
    # at_risk at the first event after each boundary (0.7, 2.2, 4.8, 6.2,
    # 9.4) is 9, 7, 5, 3 and 1.
    table = sp.life_table(X, C, tl=TL, intervals=EDGES)
    assert_array_equal(table["entered"], [1, 2, 0, 0, 0])
    assert_array_equal(table["at_risk_start"], [9, 7, 5, 3, 1])
    assert_array_equal(table["events"], [2, 3, 1, 2, 0])
    assert_array_equal(table["censored"], [1, 1, 1, 0, 1])
    # half of each late entrant and each withdrawal is at risk
    assert_allclose(table["at_risk"], [9.0, 7.5, 4.5, 3.0, 0.5])


@pytest.mark.parametrize("late", [False, True])
def test_estimates_are_klein_and_moeschberger(late):
    table = sp.life_table(X, C, tl=TL if late else None, intervals=EDGES)
    # The last interval has no events, so q = 0 and its standard errors
    # are 0 (the exact formulas divide by q there): leave it to the next
    # test.
    rows = 4
    ref = _lifetab(
        EDGES,
        table["at_risk_start"][0],
        table["entered"],
        table["censored"],
        table["events"],
    )[:rows]
    cols = ["at_risk_start", "at_risk", "q", "surv"]
    for k, name in enumerate(cols):
        assert_allclose(
            table[name][:rows], [float(r[k]) for r in ref], rtol=1e-14
        )
    assert_allclose(
        table["se_surv"][:rows],
        [math.sqrt(r[4]) for r in ref],
        rtol=1e-14,
    )
    for k, name in ((5, "pdf"), (7, "hazard")):
        assert_allclose(
            table[name][:rows], [float(r[k]) for r in ref], rtol=1e-14
        )
        assert_allclose(
            table["se_" + name][:rows],
            [math.sqrt(r[k + 1]) for r in ref],
            rtol=1e-13,
        )


def test_an_interval_without_events_has_zero_rates_and_errors():
    table = sp.life_table(X, C, intervals=EDGES)
    last = table.iloc[-1]
    assert last["q"] == 0 and last["pdf"] == 0 and last["hazard"] == 0
    assert last["se_pdf"] == 0 and last["se_hazard"] == 0
    assert last["surv"] > 0 and last["se_surv"] > 0


def test_without_withdrawals_survival_is_kaplan_meier():
    # Every event at an interval's start: the actuarial survival at each
    # start is the Kaplan-Meier survival just before it, and the Greenwood
    # standard errors agree.
    x = [1, 1, 2, 3, 3, 3, 5, 6, 8, 9]
    table = sp.life_table(x, intervals=[0, 1, 2, 3, 5, 6, 8, 9, 10])
    km = sp.KaplanMeier.fit(x)
    starts = table["start"].to_numpy()[1:]
    below = starts - 1e-9
    assert_allclose(table["surv"][1:], km.sf(below), rtol=1e-14)
    se_km = np.sqrt(
        np.array(
            [
                km.sf(t) ** 2
                * sum(
                    d / (r * (r - d))
                    for xi, r, d in zip(km.x, km.r, km.d)
                    if xi < t + 1e-12 and r > d
                )
                for t in below
            ]
        )
    )
    assert_allclose(table["se_surv"][1:], se_km, rtol=1e-12)


def test_grouped_counts_as_lifetab_takes_them():
    # lifetab(tis, ninit, nlost, nevent)'s input as rows at each
    # interval's start, with counts.
    tis = [0, 2, 3, 5, 7, np.inf]
    nlost = [2, 3, 6, 22, 21]
    nevent = [77, 71, 119, 75, 109]
    ninit = 927
    x = np.repeat(tis[:-1], 2)
    c = np.tile([0, 1], 5)
    n = np.ravel(np.column_stack([nevent, nlost]))
    others = ninit - sum(nlost) - sum(nevent)
    x = np.append(x, 9.0)
    c = np.append(c, 1)
    n = np.append(n, others)
    table = sp.life_table(x, c, n, intervals=tis)
    assert_array_equal(table["at_risk_start"][:5], [927, 848, 774, 649, 552])
    ref = _lifetab(tis, ninit, [0] * 5, nlost[:4] + [21 + others], nevent)
    assert_allclose(table["surv"], [float(r[3]) for r in ref], rtol=1e-14)
    assert np.isnan(table["pdf"].iloc[-1])
    assert np.isnan(table["hazard"].iloc[-1])


def test_number_of_intervals_ends_open():
    table = sp.life_table(X, C, intervals=4)
    assert len(table) == 4
    assert table["end"].iloc[-1] == np.inf
    assert table["events"].sum() + table["censored"].sum() == len(X)


def test_default_intervals_cover_the_data():
    table = sp.life_table(X, C)
    assert table["start"].iloc[0] == 0
    assert table["end"].iloc[-1] > max(X)
    assert table["events"].sum() == C.count(0)


def test_data_and_model_inputs_agree_with_arrays():
    ref = sp.life_table(X, C, tl=TL, intervals=EDGES)
    data = sp.SurpyvalData(X, C, tl=TL)
    km = sp.KaplanMeier.fit(X, c=C, tl=TL)
    for given in (data, km):
        out = sp.life_table(given, intervals=EDGES)
        assert out.equals(ref)
    with pytest.raises(TypeError, match="leave out"):
        sp.life_table(data, C)


def test_counts_equal_repeated_rows():
    a = sp.life_table([1, 2, 3, 5], [0, 1, 0, 0], [2, 1, 3, 1], intervals=3)
    b = sp.life_table(
        [1, 1, 2, 3, 3, 3, 5], [0, 0, 1, 0, 0, 0, 0], intervals=3
    )
    assert a.equals(b)


def test_entry_on_a_boundary_is_at_risk_from_it():
    table = sp.life_table([3, 5], tl=[2, 0], intervals=[0, 2, 4, 6])
    assert_array_equal(table["at_risk_start"], [1, 2, 1])
    assert_array_equal(table["entered"], 0)


@pytest.mark.parametrize("c", [[0, -1, 0], [0, 2, 0]])
def test_left_and_interval_censoring_refused(c):
    x = [1, 2, 3] if c[1] == -1 else [1, [2, 3], 4]
    with pytest.raises(ValueError, match="Turnbull"):
        sp.life_table(x, c)


def test_right_truncation_refused():
    with pytest.raises(ValueError, match="Turnbull"):
        sp.life_table(sp.SurpyvalData([1, 2, 3], tr=10))


@pytest.mark.parametrize("bad", [[0], [0, 2, 1], [0, np.nan, 3], 0, 1.5])
def test_bad_intervals_refused(bad):
    with pytest.raises(ValueError, match="intervals"):
        sp.life_table(X, C, intervals=bad)


def test_rows_before_the_table_warn_with_counts():
    with pytest.warns(UserWarning, match="2 of the 12 units"):
        table = sp.life_table(X, C, intervals=[1.5, 4, 8])
    assert table["at_risk_start"].iloc[0] == 10


def test_no_warning_in_ordinary_use():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.life_table(X, C, tl=TL, intervals=EDGES)


@pytest.mark.parametrize("x", [[1, 2, 3, 4], [0.1 * k for k in range(1, 31)]])
def test_default_intervals_count_every_unit(x):
    table = sp.life_table(x)
    assert table["events"].sum() == len(x)


# -- gaps found by the 0.26 test audit ---------------------------------------


def test_a_model_restored_without_its_data_is_refused():
    """Audit C5. A Kaplan-Meier restored from its dictionary holds the
    curve but not the rows, so there is nothing to count; the refusal
    says to pass the times and flags."""
    km = sp.KaplanMeier.fit(X, c=C)
    d = km.to_dict()
    d.pop("data", None)
    with pytest.raises(ValueError, match="does not hold its data"):
        sp.life_table(sp.from_dict(d))


def test_a_bool_number_of_intervals_is_refused():
    """Audit C6. ``True`` is not a number of intervals."""
    with pytest.raises(ValueError, match="intervals"):
        sp.life_table([1, 2, 3, 4], intervals=True)


def test_negative_times_start_the_table_at_the_smallest_time():
    """Audit C6."""
    table = sp.life_table([-2, -1, 0.5, 1], intervals=2)
    assert table["start"][0] == -2
    assert table["events"].sum() == 4


def test_a_largest_time_on_the_last_boundary_gets_an_open_interval():
    """Audit C6. The unit failing at 4 is counted in an open last
    interval [4, inf)."""
    table = sp.life_table([1, 2, 3, 4], intervals=[0, 2, 4])
    assert len(table) == 3
    assert table["end"].iloc[-1] == np.inf
    assert table["events"].sum() == 4


def test_an_interval_with_nobody_at_risk_has_nan_rates():
    """Audit C6. Nobody is at risk in [5, 10), so its conditional
    probabilities, density and hazard are not defined."""
    table = sp.life_table([1, 2, 3, 4], intervals=[0, 5, 10])
    row = table.iloc[1]
    assert row["at_risk_start"] == 0
    for column in ("q", "p", "pdf", "hazard"):
        assert np.isnan(row[column]), column


def test_all_equal_times_count_every_unit():
    """Audit C6."""
    table = sp.life_table([3, 3, 3, 3], [0, 0, 1, 1])
    assert table["events"].sum() == 2
    assert table["censored"].sum() == 2
