"""``EventHistoryData.without("covariates")`` on the arrays gives what
the record route gives: each item's ItemHistory without its covariates,
then ``EventHistoryData.from_items``.

Random data -- ties within and across items, items with no events (only
an end of observation), items with no end row, delayed entry and right
truncation, intervals and counts before a first look, numeric, text and
object ids, observation windows, unsorted hand-built data -- is reshaped
both ways. Where the record route gives data, the array route must give
the same data bit for bit (every array with its dtype, and every
attribute); where it refuses, the array route must refuse with the same
message. The proportional-intensity NHPP's default start, a fit to the
covariate-free data, must then be the same too."""

import warnings

import numpy as np
import pytest

from surpyval import CoxLewis, CrowAMSAA, Duane, EventHistoryData
from surpyval.recurrent.regression.nhpp_proportional_intensity import (
    ProportionalIntensityNHPP,
)
from surpyval.utils import data_reshape, history_reshape
from surpyval.utils.recurrent_utils import handle_xicn

PARTS = ("covariates",)
ATTRIBUTES = (
    "ids",
    "Z",
    "covariate_names",
    "window_map",
    "observation_windows",
    "time_varying_covariates",
    "states",
    "attributes",
    "unit",
)
ARRAYS = ("x", "i", "c", "n", "tl", "tr", "e", "interarrival_times")


def record_route(data):
    """The old route, item by item through the records."""
    return data._map(lambda r: data_reshape.without(r, PARTS))


def outcome(route, data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = route(data)
        except Exception as error:  # noqa: BLE001 - compared below
            out = (type(error), str(error))
    return out, [(w.category, str(w.message)) for w in caught]


def assert_same_data(a, b):
    for name in ARRAYS:
        left, right = getattr(a, name, None), getattr(b, name, None)
        if left is None or right is None:
            assert left is None and right is None, name
            continue
        left, right = np.asarray(left), np.asarray(right)
        assert left.dtype == right.dtype, (name, left.dtype, right.dtype)
        assert left.shape == right.shape, name
        if left.dtype.kind == "f":
            # bit for bit, NaN and the sign of zero included
            assert np.array_equal(
                left.view(np.uint64), right.view(np.uint64)
            ), name
        else:
            assert np.array_equal(left, right), name
    for name in ATTRIBUTES:
        left, right = getattr(a, name, None), getattr(b, name, None)
        assert type(left) is type(right), name
        assert repr(left) == repr(right), name
    assert sorted(vars(a)) == sorted(vars(b))


def assert_routes_agree(data):
    """Returns whether the array route was taken."""
    old, old_warnings = outcome(record_route, data)
    new, new_warnings = outcome(lambda d: d.without(*PARTS), data)
    assert new_warnings == old_warnings
    if isinstance(old, tuple):
        assert new == old
        return False
    assert not isinstance(new, tuple), new
    assert_same_data(new, old)
    return history_reshape.without_covariates(data) is not None


# -- random data ----------------------------------------------------------


def random_rows(rng, two_d, string_ids, n_items):
    x, i, c, n, tl, tr, Z = [], [], [], [], [], [], []
    for k in range(n_items):
        iid = (
            f"unit-{k:0{int(rng.integers(1, 3))}d}"
            if string_ids
            else int(rng.integers(0, 40) * 10 + k)
        )
        entry = float(rng.choice([0, 0, -np.inf, rng.integers(1, 5)]))
        start = 0.0 if not np.isfinite(entry) else entry
        rows = []
        decimals = int(rng.choice([0, 0, 1, 3]))
        events = np.round(
            start + 0.01 + rng.random(rng.integers(0, 6)) * 20, decimals
        )
        events = events[events > start]
        if len(events) and rng.random() < 0.4:
            events = np.append(events, rng.choice(events, rng.integers(1, 3)))
        rows += [([t, t], 0, 1) for t in events]
        if two_d and not len(events) and rng.random() < 0.6:
            low = start + float(rng.integers(0, 4))
            for _ in range(int(rng.integers(1, 3))):
                high = low + float(rng.integers(1, 4))
                rows.append(([low, high], 2, int(rng.integers(1, 4))))
                low = high
        if two_d and rng.random() < 0.25:
            first = min([r[0][0] for r in rows] + [start + 6.0])
            left = float(np.round(start + (first - start) * rng.random(), 1))
            if left > start:
                rows.append(([left, left], -1, int(rng.integers(1, 3))))
        last = max([r[0][1] for r in rows], default=start + 1.0)
        if rng.random() < 0.8 or not rows:
            end = last + float(rng.choice([0, 0.5, 3]))
            rows.append(([end, end], 1, 1))
        else:
            end = last
        finite = [r[0][1] for r in rows if r[1] != 1]
        if finite and rng.random() < 0.2:
            # right truncated: the first failure came by then
            top = float(min(finite)) + float(rng.choice([0, 0.5]))
            top = max(top, end + 1.0) if rng.random() < 0.5 else top
        else:
            top = np.inf
        z = rng.normal(size=2)
        if rng.random() < 0.1:
            z[0] = np.nan
        for xx, cc, nn in rows:
            x.append(xx)
            i.append(iid)
            c.append(cc)
            n.append(nn)
            tl.append(entry)
            tr.append(top)
            Z.append(z)
    return x, i, c, n, tl, tr, Z


def random_data(seed):
    rng = np.random.default_rng(seed)
    two_d = rng.random() < 0.4
    string_ids = rng.random() < 0.4
    x, i, c, n, tl, tr, Z = random_rows(
        rng, two_d, string_ids, int(rng.integers(1, 8))
    )
    x = np.array(x, dtype=float)
    if not two_d and rng.random() < 0.8:
        x = x[:, 0]
    if rng.random() < 0.2 and np.all(x == np.round(x)):
        x = x.astype(int)
    order = rng.permutation(len(c))
    i = np.array(i)
    if rng.random() < 0.15:
        i = i.astype(object)
    columns = dict(
        x=x[order],
        i=i[order],
        c=np.array(c)[order],
        n=np.array(n)[order],
        tl=np.array(tl)[order],
        tr=np.array(tr)[order],
        Z=np.array(Z)[order],
    )
    if not np.isfinite(columns["tl"]).any():
        del columns["tl"]
    if not np.isfinite(columns["tr"]).any():
        del columns["tr"]
    names = ["age", "load"] if rng.random() < 0.5 else None
    how = rng.choice(["handle_xicn", "hand built", "no covariates"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if how == "hand built":
            data = EventHistoryData(
                columns["x"],
                columns["i"],
                columns["c"],
                columns["n"],
                tl=columns.get("tl"),
                tr=columns.get("tr"),
                unit=rng.choice([None, "hour"]),
            )
            data.Z = columns["Z"]
        elif how == "no covariates":
            del columns["Z"]
            data = handle_xicn(**columns)
        else:
            data = handle_xicn(**columns, unit=rng.choice([None, "hour"]))
    if data.Z is not None and names is not None:
        data.covariate_names = names
    return data


def random_datasets(count, first_seed=0):
    out = []
    seed = first_seed
    while len(out) < count:
        seed += 1
        try:
            out.append((seed, random_data(seed)))
        except ValueError:
            continue
    return out


DATASETS = random_datasets(600)


@pytest.mark.parametrize(
    "seed, data", DATASETS[:400], ids=[str(s) for s, _ in DATASETS[:400]]
)
def test_random_data_routes_agree(seed, data):
    assert_routes_agree(data)


def test_array_route_is_taken_for_most_random_data():
    taken = [assert_routes_agree(data) for _, data in DATASETS[400:]]
    assert np.mean(taken) > 0.6


# -- data the records refuse, or change -----------------------------------


def _windows():
    # handle_xicn takes no covariates with windows; set by hand
    data = handle_xicn(
        [1, 3, 12, 4, 15],
        [1, 1, 1, 2, 2],
        [0, 0, 0, 0, 0],
        windows={1: [(0, 5), (10, 20)], 2: [(0, 20)]},
    )
    data.Z = np.ones((len(data.x), 1))
    return data


EDGE_CASES = {
    "failure at the top of an inspection window": dict(
        x=[[1, 3], [3, 3], [8, 8], [2, 2], [6, 6]],
        i=[1, 1, 1, 2, 2],
        c=[2, 0, 1, 0, 1],
        n=[1, 1, 1, 1, 1],
    ),
    "end of observation at entry": dict(
        x=[2, 2, 5, 9],
        i=["a", "b", "b", "b"],
        c=[1, 0, 0, 1],
        tl=[2, 0, 0, 0],
    ),
    "blank id": dict(x=[1, 2, 3, 4], i=["", "", "b", "b"], c=[0, 1, 0, 1]),
    "left censored count, then a failure at that age": dict(
        x=[[3, 3], [3, 3], [8, 8]],
        i=[1, 1, 1],
        c=[-1, 0, 1],
        n=[2, 1, 1],
    ),
    "adjacent inspection windows": dict(
        x=[[1, 3], [3, 5], [8, 8]],
        i=[1, 1, 1],
        c=[2, 2, 1],
        n=[1, 2, 1],
    ),
    "item with only an end of observation": dict(
        x=[5, 3, 4], i=[1, 2, 2], c=[1, 0, 1]
    ),
    "negative ages after a negative entry": dict(
        x=[-1, 2], i=[1, 1], c=[0, 1], tl=[-3, -3]
    ),
    "a 2-D x with no intervals": dict(x=[[1, 1], [3, 3]], i=[1, 1], c=[0, 1]),
    "float ids": dict(x=[1, 2, 3], i=[1.5, 1.5, 2.0], c=[0, 1, 0]),
}


@pytest.mark.parametrize("case", sorted(EDGE_CASES))
def test_edge_cases_routes_agree(case):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = handle_xicn(**EDGE_CASES[case])
    data.Z = np.ones((len(data.x), 1))
    assert_routes_agree(data)


def test_windows_take_the_record_route():
    data = _windows()
    assert history_reshape.without_covariates(data) is None
    assert_routes_agree(data)


def test_hand_built_data_watched_past_right_truncation_agrees():
    # Built by hand (no per-item checks): watched past its right
    # truncation age with no failure by then, which the records refuse
    data = EventHistoryData(
        [[1, 4], [9, 9], [1, 1], [9, 9]],
        [1, 1, 2, 2],
        [2, 1, 0, 1],
        1,
        tr=[3, 3, 20, 20],
    )
    data.Z = np.zeros((4, 1))
    assert history_reshape.without_covariates(data) is None
    assert_routes_agree(data)


def test_hand_built_data_with_two_ends_agrees():
    # Two ends of observation for one item: the record keeps the first
    data = EventHistoryData([1, 5, 4], [1, 1, 1], [0, 1, 1], 1)
    data.Z = np.zeros((3, 1))
    assert_routes_agree(data)


def test_hand_built_data_with_repeated_window_agrees():
    # The same inspection window twice: the record counts it once
    data = EventHistoryData(
        [[0, 2], [0, 2], [5, 5]], [1, 1, 1], [2, 2, 1], [1, 2, 1]
    )
    data.Z = np.zeros((3, 1))
    assert_routes_agree(data)


# -- the start of the proportional-intensity NHPP -------------------------


def record_start(data, dist, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(history_reshape, "without_covariates", lambda data: None)
        return ProportionalIntensityNHPP._baseline_start(data, dist)


@pytest.mark.parametrize("dist", [Duane, CrowAMSAA, CoxLewis])
def test_start_is_the_same(dist, monkeypatch):
    starts = 0
    for _, data in DATASETS[:120]:
        if data.Z is None:
            continue
        # The start's warnings (a small random dataset's baseline fit may
        # not reach a verified maximum) are the same both ways too
        new, new_warnings = outcome(
            lambda d: ProportionalIntensityNHPP._baseline_start(d, dist), data
        )
        old, old_warnings = outcome(
            lambda d: record_start(d, dist, monkeypatch), data
        )
        assert new_warnings == old_warnings
        assert np.array_equal(new.view(np.uint64), old.view(np.uint64))
        starts += not np.array_equal(new, np.ones(len(new)))
    assert starts > 20


def test_fit_is_the_same(monkeypatch):
    rng = np.random.default_rng(7)
    x, i, c, Z = [], [], [], []
    for k in range(40):
        times = np.sort(rng.random(rng.poisson(4)) * 50)
        x += [*times, 50.0]
        c += [0] * len(times) + [1]
        i += [f"m{k}"] * (len(times) + 1)
        Z += [[rng.normal(), float(k % 2)]] * (len(times) + 1)
    data = handle_xicn(np.array(x), np.array(i), np.array(c), Z=np.array(Z))
    new = ProportionalIntensityNHPP.fit_from_recurrent_data(data)
    with monkeypatch.context() as patch:
        patch.setattr(history_reshape, "without_covariates", lambda data: None)
        old = ProportionalIntensityNHPP.fit_from_recurrent_data(data)
    assert np.array_equal(new.params, old.params)
    assert new.log_likelihood == old.log_likelihood


# -- data the records refuse but the fit takes ----------------------------


@pytest.mark.parametrize(
    "case",
    [
        "failure at the top of an inspection window",
        "end of observation at entry",
        "blank id",
    ],
)
@pytest.mark.parametrize("dist", [Duane, CrowAMSAA, CoxLewis])
def test_start_where_the_records_refuse(case, dist):
    # It used to be every baseline parameter at one; it is now the fit
    # of the baseline to the data with its covariates set aside
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = handle_xicn(**EDGE_CASES[case])
    data.Z = np.arange(len(data.x), dtype=float)[:, None]
    with pytest.raises(ValueError):
        record_route(data)
    start = ProportionalIntensityNHPP._baseline_start(data, dist)
    free = history_reshape.covariates_dropped(data)
    assert free.Z is None and data.Z is not None
    expected = dist.fit_from_recurrent_data(free).params
    assert np.array_equal(start, expected)
    assert not np.array_equal(start, np.ones(len(start)))
