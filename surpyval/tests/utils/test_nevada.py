"""The warranty Nevada table: reading the triangle, writing it back, and
forecasting from it.

References:

- a hand-worked triangle (below), each row of ``x``, ``c``, ``n`` counted
  by hand;
- ReliaSoft's documented Nevada chart (shipments of 100, 140 and 150
  units in June to August, returns in July to September): its
  failures/suspensions table, its Weibull fit (eta 6.6951, beta 2.4928)
  and its next-month forecast of each shipment's returns (11.748, 11.035
  and 5.796, from probabilities rounded to three figures; 29 in all).
  Numbers only, typed from the documentation.
"""

import numpy as np
import pytest

import surpyval as surv
from surpyval.utils.nevada import NevadaTable

# Hand-worked: four monthly shipments, observed to the end of month 4
SHIPPED = [50, 60, 70, 80]
CALENDAR = [
    [1, 2, 0, 3],
    [None, 4, 1, 2],
    [None, None, 0, 5],
    [None, None, None, 6],
]
AGE = [
    [1, 2, 0, 3],
    [4, 1, 2],
    [0, 5],
    [6],
]


def test_calendar_and_age_layouts_read_the_same():
    cal = NevadaTable(SHIPPED, CALENDAR)
    age = NevadaTable(SHIPPED, AGE, layout="age")
    np.testing.assert_array_equal(cal.observed, [4, 3, 2, 1])
    np.testing.assert_array_equal(age.observed, [4, 3, 2, 1])
    np.testing.assert_array_equal(
        np.nan_to_num(cal.returns, nan=-1), np.nan_to_num(age.returns, nan=-1)
    )
    np.testing.assert_array_equal(cal.survivors, [44, 53, 65, 74])
    np.testing.assert_array_equal(cal.age, [4.0, 3.0, 2.0, 1.0])


def test_hand_worked_xcnt():
    table = NevadaTable(SHIPPED, CALENDAR)
    # Returns by period of service: k=1: 1+4+0+6, k=2: 2+1+5, k=3: 0+2,
    # k=4: 3; survivors 44 at 4, 53 at 3, 65 at 2, 74 at 1.
    x, c, n = table.to_xcnt()
    np.testing.assert_array_equal(
        x,
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 1], [2, 2], [3, 3], [4, 4]],
    )
    np.testing.assert_array_equal(c, [2, 2, 2, 2, 1, 1, 1, 1])
    np.testing.assert_array_equal(n, [11, 8, 2, 3, 74, 65, 53, 44])
    assert n.sum() == sum(SHIPPED)
    x, c, n = table.to_xcnt("end")
    np.testing.assert_array_equal(x, [1, 2, 3, 4, 1, 2, 3, 4])
    np.testing.assert_array_equal(c, [0, 0, 0, 0, 1, 1, 1, 1])
    x, _, _ = table.to_xcnt("middle")
    np.testing.assert_array_equal(x[:4], [0.5, 1.5, 2.5, 3.5])


def test_period_scales_the_ages():
    table = NevadaTable(SHIPPED, CALENDAR, period=30.0)
    x, c, n = table.to_xcnt()
    np.testing.assert_array_equal(x[0], [0, 30])
    np.testing.assert_array_equal(table.age, [120, 90, 60, 30])


def test_cumulative_cells_are_differenced():
    cum = [
        [1, 3, 3, 6],
        [None, 4, 5, 7],
        [None, None, 0, 5],
        [None, None, None, 6],
    ]
    a = NevadaTable(SHIPPED, cum, cumulative=True)
    b = NevadaTable(SHIPPED, CALENDAR)
    np.testing.assert_array_equal(
        np.nan_to_num(a.returns, nan=-1), np.nan_to_num(b.returns, nan=-1)
    )


@pytest.mark.parametrize("layout", ["calendar", "age"])
def test_round_trip(layout):
    table = NevadaTable(
        SHIPPED, CALENDAR if layout == "calendar" else AGE, layout=layout
    )
    back = NevadaTable.from_returns(SHIPPED, *table.to_returns())
    assert back.layout == "calendar"
    np.testing.assert_array_equal(
        np.nan_to_num(back.to_array(), nan=-1),
        np.nan_to_num(table.to_array(), nan=-1),
    )
    expected = np.array(CALENDAR, dtype=float)
    np.testing.assert_array_equal(
        np.nan_to_num(back.to_array(), nan=-1),
        np.nan_to_num(expected, nan=-1),
    )
    np.testing.assert_array_equal(
        back.to_array("age")[3], [6, np.nan, np.nan, np.nan]
    )
    for got, want in zip(back.to_xcnt(), table.to_xcnt()):
        np.testing.assert_array_equal(got, want)


def test_from_returns_one_row_per_unit():
    shipment = [0] * 1 + [0] * 2 + [0] * 3 + [1] * 4 + [1] + [1] * 2
    shipment += [2] * 5 + [3] * 6
    age = [1] + [2] * 2 + [4] * 3 + [1] * 4 + [2] + [3] * 2 + [2] * 5
    age += [1] * 6
    table = NevadaTable.from_returns(SHIPPED, shipment, age)
    np.testing.assert_array_equal(
        np.nan_to_num(table.to_array(), nan=-1),
        np.nan_to_num(np.array(CALENDAR, dtype=float), nan=-1),
    )


def test_from_returns_with_observed_is_age_layout():
    table = NevadaTable.from_returns([10, 10], [0, 1], [1, 2], observed=[3, 3])
    assert table.layout == "age"
    np.testing.assert_array_equal(table.returns, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(table.survivors, [9, 9])


def test_published_nevada_example():
    table = NevadaTable(
        [100, 140, 150], [[3, 3, 5], [None, 2, 4], [None, None, 4]]
    )
    # The documented failures/suspensions table
    x, c, n = table.to_xcnt("end")
    np.testing.assert_array_equal(x, [1, 2, 3, 1, 2, 3])
    np.testing.assert_array_equal(c, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(n, [9, 7, 5, 146, 134, 89])
    model = surv.Weibull.fit(table.to_data(ages="end"))
    assert model.alpha == pytest.approx(6.6951, abs=5e-4)
    assert model.beta == pytest.approx(2.4928, abs=5e-4)
    ahead = table.forecast(surv.Weibull.from_params([6.6951, 2.4928]), 1)
    # Conditional probabilities Q(1 | 3), Q(1 | 2), Q(1 | 1)
    np.testing.assert_allclose(
        ahead.probability.ravel(), [0.132, 0.0824, 0.0397], atol=5e-4
    )
    np.testing.assert_allclose(
        ahead.unit_expected.ravel(), [11.748, 11.035, 5.796], atol=0.06
    )
    assert round(float(ahead.expected[0])) == 29


def test_forecast_is_forecast_of_the_cohorts():
    table = NevadaTable(SHIPPED, CALENDAR)
    model = surv.Weibull.fit(table.to_data())
    got = table.forecast(model, horizon=[1, 2, 3], limit=4)
    want = surv.forecast(
        model,
        age=[4, 3, 2, 1],
        n=[44, 53, 65, 74],
        horizon=[1, 2, 3],
        limit=4,
    )
    np.testing.assert_array_equal(got.expected, want.expected)
    F = model.ff
    a = np.array([4.0, 3, 2, 1])
    end = np.minimum(a + 3, 4)
    by_hand = np.sum(
        np.array([44, 53, 65, 74]) * (F(end) - F(a)) / model.sf(a)
    )
    assert got.expected[-1] == pytest.approx(by_hand, rel=1e-10)


def test_fit_matches_per_unit_data():
    # Simulated cohorts: the pooled table fits to the same maximum as the
    # unpooled rows, one per cohort and period.
    rng = np.random.default_rng(3)
    S = 12
    shipped = rng.integers(200, 400, S)
    grid = np.full((S, S), np.nan)
    rows_x, rows_c, rows_n = [], [], []
    for i, m in enumerate(shipped):
        observed = S - i
        k = np.ceil(40 * rng.weibull(1.3, m))
        grid[i, i:] = [np.sum(k == j) for j in range(1, observed + 1)]
        for j in range(1, observed + 1):
            if np.sum(k == j):
                rows_x.append([j - 1, j])
                rows_c.append(2)
                rows_n.append(int(np.sum(k == j)))
        rows_x.append([observed, observed])
        rows_c.append(1)
        rows_n.append(int(np.sum(k > observed)))
    table = NevadaTable(shipped, grid)
    a = surv.Weibull.fit(table.to_data())
    b = surv.Weibull.fit(x=np.array(rows_x, float), c=rows_c, n=rows_n)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-6)
    assert a.neg_ll() == pytest.approx(b.neg_ll(), abs=1e-8)


def test_repr():
    table = NevadaTable(SHIPPED, AGE, layout="age")
    text = repr(table)
    assert text.splitlines()[0] == (
        "Nevada table: 4 shipments, 260 units, 24 returns (age layout)"
    )
    assert "age 1" in text and "survivors" in text


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(shipped=[5, 5], returns=[[6], [None, 0]]), "Shipment 0"),
        (dict(shipped=[5, 5], returns=[[1, 0], [1, 0]]), "before shipment 1"),
        (dict(shipped=[5], returns=[[1.5]]), "whole, non-negative"),
        (dict(shipped=[5], returns=[[-1]]), "whole, non-negative"),
        (dict(shipped=[5, 5], returns=[[1, 1]]), "one shipment quantity"),
        (
            dict(shipped=[5, 5, 5], returns=[[1, 1], [None, 1], [None]]),
            "3 shipments but 2",
        ),
        (dict(shipped=[5], returns=[[1]], layout="diag"), "'layout'"),
        (dict(shipped=[5], returns=[[1]], period=0), "'period'"),
        (dict(shipped=[5], returns=[[2, 1]], cumulative=True), "cannot"),
        (
            dict(shipped=[5], returns=[[1, None, 1]], layout="age"),
            "after a blank",
        ),
        (dict(shipped=[5], returns=[[1]], observed=[1]), "'observed' is"),
    ],
)
def test_refusals_say_what_to_do(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NevadaTable(**kwargs)


def test_from_returns_refusals():
    with pytest.raises(ValueError, match="at least 1"):
        NevadaTable.from_returns([5], [0], [0])
    with pytest.raises(ValueError, match="names row 3"):
        NevadaTable.from_returns([5], [3], [1])
    with pytest.raises(ValueError, match="in service for 1"):
        NevadaTable.from_returns([5, 5], [1], [2], observed=[2, 1])
    with pytest.raises(ValueError, match="'ages'"):
        NevadaTable([5], [[1]]).to_xcnt("start")


def test_top_level_name():
    assert surv.NevadaTable is NevadaTable


# -- shipments spread through their period ------------------------------------


def test_spread_reads_every_age_half_a_period_younger():
    start = NevadaTable(SHIPPED, CALENDAR, period=2.0)
    table = NevadaTable(SHIPPED, CALENDAR, period=2.0, shipping="spread")
    np.testing.assert_array_equal(table.age, [7.0, 5.0, 3.0, 1.0])
    np.testing.assert_array_equal(table.survivors, start.survivors)
    x, c, n = table.to_xcnt()
    # Period 2: the first period's returns in (0, 1], the k-th in
    # (2k - 3, 2k - 1]; survivors at 2m - 1.
    np.testing.assert_array_equal(
        x,
        [[0, 1], [1, 3], [3, 5], [5, 7], [1, 1], [3, 3], [5, 5], [7, 7]],
    )
    np.testing.assert_array_equal(c, [2, 2, 2, 2, 1, 1, 1, 1])
    np.testing.assert_array_equal(n, [11, 8, 2, 3, 74, 65, 53, 44])
    x_end, _, _ = table.to_xcnt("end")
    np.testing.assert_array_equal(x_end[:4], [1, 3, 5, 7])
    x_mid, _, _ = table.to_xcnt("middle")
    np.testing.assert_array_equal(x_mid[:4], [0.5, 2, 4, 6])
    assert "shipments spread" in repr(table).splitlines()[0]
    ahead = table.forecast(surv.Weibull.from_params([6, 2]), horizon=1)
    expected = surv.forecast(
        surv.Weibull.from_params([6, 2]),
        age=[7.0, 5.0, 3.0, 1.0],
        n=table.survivors,
        horizon=1,
    )
    np.testing.assert_array_equal(ahead.expected, expected.expected)
    back = NevadaTable.from_returns(
        SHIPPED, *table.to_returns(), period=2.0, shipping="spread"
    )
    assert back.shipping == "spread"


def test_start_is_unchanged_by_the_option():
    # Bit for bit: the default reading is the reading before the option.
    a = NevadaTable(SHIPPED, CALENDAR, period=30.44)
    b = NevadaTable(SHIPPED, CALENDAR, period=30.44, shipping="start")
    for ages in ("interval", "end", "middle"):
        for u, v in zip(a.to_xcnt(ages), b.to_xcnt(ages)):
            np.testing.assert_array_equal(u, v)
    x, _, _ = a.to_xcnt("middle")
    np.testing.assert_array_equal(x[:4], (np.arange(1, 5) - 0.5) * 30.44)


def test_spread_recovers_a_simulation_better_than_the_start_reading():
    # 12 monthly shipments of 500 units, each unit going into service at a
    # uniform time in its shipment month; Weibull(eta 6, beta 2) lives in
    # months; 40 tables, seed 715. The start reading takes every unit as
    # in service from the start of the month, so its ages run old.
    rng = np.random.default_rng(715)
    S, N, truth = 12, 500, np.array([6.0, 2.0])
    fits = {"start": [], "spread": []}
    for _ in range(40):
        grid = np.full((S, S), np.nan)
        for i in range(S):
            m = S - i
            u = rng.uniform(0, 1, N)
            t = truth[0] * rng.weibull(truth[1], N)
            k = np.ceil(u + t).astype(int)
            grid[i, :m] = np.bincount(k[k <= m], minlength=m + 1)[1 : m + 1]
        for shipping, out in fits.items():
            table = NevadaTable(
                np.full(S, N), grid, layout="age", shipping=shipping
            )
            out.append(surv.Weibull.fit(table.to_data()).params)
    start = np.array(fits["start"]).mean(axis=0)
    spread = np.array(fits["spread"]).mean(axis=0)
    # Measured: start (6.566, 2.263), spread (6.024, 1.984).
    np.testing.assert_allclose(spread, truth, rtol=0.02)
    assert np.all(np.abs(spread - truth) < np.abs(start - truth) / 5)


# -- returns known only as totals ---------------------------------------------


def test_totals_are_left_censored_at_the_shipment_age():
    calendar = [row[:] for row in CALENDAR]
    calendar[1] = [None, 0, 0, 0]  # shipment 1: 7 returns, by when unknown
    table = NevadaTable(SHIPPED, calendar, totals=[None, 7, None, None])
    np.testing.assert_array_equal(table.survivors, [44, 53, 65, 74])
    np.testing.assert_array_equal(np.isnan(table.totals), [1, 0, 1, 1])
    x, c, n = table.to_xcnt()
    # By period of service, without shipment 1: k=1: 1+0+6, k=2: 2+5,
    # k=3: none, k=4: 3; then the survivors; then shipment 1's 7 returns,
    # left censored at its age, 3.
    np.testing.assert_array_equal(
        x,
        [[0, 1], [1, 2], [3, 4]] + [[1, 1], [2, 2], [3, 3], [4, 4]] + [[3, 3]],
    )
    np.testing.assert_array_equal(c, [2, 2, 2, 1, 1, 1, 1, -1])
    np.testing.assert_array_equal(n, [7, 7, 3, 74, 65, 53, 44, 7])
    x, c, n = table.to_xcnt("end")
    np.testing.assert_array_equal(x[-1:], [3])
    np.testing.assert_array_equal(c[-1:], [-1])
    # The fit is the fit of the same rows given directly.
    direct = surv.Weibull.fit(*table.to_xcnt())
    assert surv.Weibull.fit(table.to_data()).params == pytest.approx(
        direct.params
    )
    assert "total 7" in repr(table)
    spread = NevadaTable(
        SHIPPED, calendar, totals=[None, 7, None, None], shipping="spread"
    )
    assert spread.to_xcnt()[0][-1].tolist() == [2.5, 2.5]
    age = NevadaTable(
        SHIPPED,
        [[1, 2, 0, 3], [], [0, 5], [6]],
        layout="age",
        observed=[4, 3, 2, 1],
        totals=[None, 7, None, None],
    )
    for u, v in zip(age.to_xcnt(), table.to_xcnt()):
        np.testing.assert_array_equal(u, v)


def test_totals_recover_a_simulation():
    # Half the shipments report only their total returns to date: the fit
    # still recovers the lives (Weibull eta 6, beta 2; 12 shipments of
    # 2000; 20 tables, seed 1715).
    rng = np.random.default_rng(1715)
    S, N, truth = 12, 2000, np.array([6.0, 2.0])
    fits = []
    for _ in range(20):
        grid = np.full((S, S), np.nan)
        totals = np.full(S, np.nan)
        for i in range(S):
            m = S - i
            k = np.ceil(truth[0] * rng.weibull(truth[1], N)).astype(int)
            counts = np.bincount(k[k <= m], minlength=m + 1)[1 : m + 1]
            if i % 2:
                totals[i] = counts.sum()
                grid[i, :m] = 0
            else:
                grid[i, :m] = counts
        table = NevadaTable(np.full(S, N), grid, layout="age", totals=totals)
        fits.append(surv.Weibull.fit(table.to_data()).params)
    np.testing.assert_allclose(np.mean(fits, axis=0), truth, rtol=0.02)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(shipped=[5], returns=[[1]], shipping="late"), "'shipping'"),
        (
            dict(shipped=[5, 5], returns=[[1, 0], [None, 0]], totals=[1]),
            "one per shipment",
        ),
        (
            dict(
                shipped=[5, 5], returns=[[1, 0], [None, 0]], totals=[1, None]
            ),
            "returns in its row",
        ),
        (
            dict(
                shipped=[5, 5], returns=[[0, 0], [None, 0]], totals=[6, None]
            ),
            "Shipment 0",
        ),
        (
            dict(
                shipped=[5, 5],
                returns=[[0, 0], []],
                layout="age",
                totals=[None, 2],
            ),
            "no periods in service",
        ),
        (dict(shipped=[5], returns=[[0]], totals=[1.5]), "whole"),
    ],
)
def test_totals_and_shipping_refusals(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NevadaTable(**kwargs)
