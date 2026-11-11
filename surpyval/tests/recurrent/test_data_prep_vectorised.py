"""
The recurrent data preparation works on every item at once: the checks
``handle_xicn`` makes of each item, the gaps between events, each row's
previous time and each item's first row and last time. These tests hold
it to the item-by-item loops it replaced, kept here as private reference
copies, on random data with ties, single-event items, items with no
events, truncation, interval counts and gapped observation windows: the
same arrays bit for bit, and the same error (or none) for the same input.
"""

import numpy as np
import pytest

import surpyval.utils.recurrent_utils as ru
from surpyval import EventHistoryData, SurpyvalData, handle_xicn
from surpyval.recurrent import CrowAMSAA

# -- the item-by-item forms replaced, as they were ----------------------------


def _old_interarrival(data):
    _, idx = np.unique(data.i, return_index=True)
    arrival_times = np.split(data.x, idx)[1:]
    return np.concatenate([np.diff(arr, prepend=0) for arr in arrival_times])


def _old_previous_x(data, min_x=0):
    x_previous = []
    for item in np.unique(data.i):
        mask_item = data.i == item
        x_prev_item = np.roll(data.x[mask_item], shift=1, axis=0)
        tl_item = float(data.tl[mask_item][0])
        x_prev_item[0] = tl_item if np.isfinite(tl_item) else min_x
        x_previous.append(x_prev_item)
    return np.concatenate(x_previous)


def _old_first_events(data):
    return SurpyvalData(
        np.array([data.x[data.i == item][0] for item in data.ids]),
        np.array([data.c[data.i == item][0] for item in data.ids]),
        np.array([data.n[data.i == item][0] for item in data.ids]),
    )


def _same_lifetimes(new, old):
    for name in ("x", "c", "n", "tl", "tr"):
        a, b = getattr(new, name), getattr(old, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name


def _old_censoring_positions(c, idx, unique_i, items_given):
    ru._check_censoring_positions_by_item(
        unique_i, np.split(c, idx)[1:], items_given
    )


def _old_observation_windows(x, tl_arr, tr_arr, c, idx, unique_i):
    ru._check_observation_windows_by_item(
        x, tl_arr, tr_arr, idx, unique_i, np.split(c, idx)[1:]
    )


def _old_static_covariates(Z_arr, idx, unique_i):
    if Z_arr is None:
        return
    for ii, Z_i in zip(unique_i, np.split(Z_arr, idx)[1:]):
        if not np.all(Z_i == Z_i[0]):
            raise ValueError(f"Item {ru.item_label(ii)} has covariates Z")


def _old_interval_overlaps(x, idx, unique_i, c=None, e=None):
    if x.ndim != 2:
        return
    kinds = (
        np.zeros(len(x), dtype=object)
        if e is None
        else np.asarray(e, dtype=object)
    )
    ends = np.zeros(len(x), dtype=bool) if c is None else (np.asarray(c) == 1)
    for ii, rows in zip(unique_i, np.split(np.arange(len(x)), idx)[1:]):
        groups: dict = {}
        for r in rows:
            if e is None or not ends[r]:
                key = kinds[r] if kinds[r] is not None else ("masked",)
                groups.setdefault(key, []).append(r)
        for kind, members in groups.items():
            arr = x[sorted(members, key=lambda r: (x[r, 0], x[r, 1]))]
            same = (arr[1:] == arr[:-1]).all(axis=1) & (
                arr[1:, 0] != arr[1:, 1]
            )
            of = "" if e is None else f" of type {kind!r}"
            if same.any():
                raise ValueError(
                    f"Item {ru.item_label(ii)} has the same inspection window"
                    f"{of} twice: give it once, with n the number of "
                    "failures it found."
                )
            if (arr[:-1][:, 1] > arr[1:][:, 0]).any():
                raise ValueError(
                    f"Item {ru.item_label(ii)} has overlapping intervals{of}: "
                    "a type's counts are of its own inspections, which "
                    "follow one another (windows of different types may "
                    "overlap)."
                )


def _static_message(Z_arr, idx, unique_i):
    """The new check's message, cut to what the reference copy says."""
    try:
        ru._check_static_covariates(Z_arr, idx, unique_i)
    except ValueError as error:
        raise ValueError(str(error).split(" that change")[0]) from None


# -- random data -------------------------------------------------------------


def _random_rows(rng, n_items, intervals=False, string_ids=False):
    """Rows of random histories: ties within and across items (integer
    times), items with one event, items with only an end of observation,
    items with no end of observation."""
    x, i, c, n = [], [], [], []
    for k in range(n_items):
        label = f"u{k:03d}" if string_ids else int(rng.integers(0, 10**6))
        if not string_ids:
            label = k * 7 + 3  # distinct, unsorted relative to input order
        kind = rng.integers(0, 4)
        n_events = {0: 0, 1: 1}.get(kind, int(rng.integers(2, 8)))
        times = np.sort(rng.integers(1, 30, n_events).astype(float))
        if intervals:  # one inspection per time (repeats are refused)
            times = np.unique(times)
            n_events = times.size
        end = (times.max() if n_events else 0.0) + float(rng.integers(0, 4))
        has_end = n_events == 0 or rng.uniform() < 0.7
        if has_end and end == 0.0:
            end = 1.0
        for t in times:
            if intervals and rng.uniform() < 0.4:
                lo = t - 0.5
                x.append([lo, t])
                c.append(2)
                n.append(int(rng.integers(1, 4)))
            else:
                x.append([t, t] if intervals else t)
                c.append(0)
                n.append(1)
            i.append(label)
        if has_end:
            x.append([end, end] if intervals else end)
            c.append(1)
            n.append(1)
            i.append(label)
    order = rng.permutation(len(x))
    return (
        np.asarray(x, dtype=float)[order],
        np.asarray(i)[order],
        np.asarray(c)[order],
        np.asarray(n)[order],
    )


def _datasets(seed):
    rng = np.random.default_rng(seed)
    x, i, c, n = _random_rows(rng, 40)
    yield handle_xicn(x, i, c, n)
    # Delayed entry (tl per item, below every event) and right truncation
    # (tr at an item's last time).
    ids = np.unique(i)
    tl = {}
    for item in ids:
        tl[item] = float(np.min(x[i == item])) - rng.uniform(0.1, 1.5)
    yield handle_xicn(x, i, c, n, tl=np.array([tl[k] for k in i]))
    x2, i2, c2, n2 = _random_rows(rng, 30, intervals=True, string_ids=True)
    yield handle_xicn(x2, i2, c2, n2)
    # Gapped observation windows.
    times = np.sort(rng.integers(1, 40, 60).astype(float))
    items = rng.integers(0, 6, times.size)
    windows = {int(k): [(0.0, 15.5), (20.5, 41.0)] for k in np.unique(items)}
    keep = (times < 15.5) | (times > 20.5)
    yield handle_xicn(times[keep], items[keep], windows=windows)


@pytest.mark.parametrize("seed", range(6))
def test_prepared_arrays_bit_identical(seed):
    for data in _datasets(seed):
        if data.x.ndim == 1:
            new = data.get_interarrival_times()
            old = _old_interarrival(data)
            assert new.dtype == old.dtype
            assert np.array_equal(new, old)
            assert np.array_equal(data.interarrival_times, old)
        for min_x in (0, 0.0, -2.5):
            new = data.get_previous_x(min_x)
            old = _old_previous_x(data, min_x)
            assert new.dtype == old.dtype and new.shape == old.shape
            assert np.array_equal(new, old)
        _same_lifetimes(data._first_event_times(), _old_first_events(data))


def test_rows_not_grouped_by_item_as_before():
    # Built by hand, not sorted by item: the same arrays (in the item-
    # grouped order the loops returned) as before.
    data = EventHistoryData(
        np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]),
        np.array([2, 1, 2, 3, 1, 3, 2]),
        np.array([0, 0, 0, 0, 1, 1, 1]),
        np.ones(7, dtype=int),
        tl=np.array([0.5, -np.inf, 0.5, -np.inf, -np.inf, -np.inf, 0.5]),
    )
    assert np.array_equal(data.get_previous_x(), _old_previous_x(data))
    assert np.array_equal(
        data.get_interarrival_times(), _old_interarrival(data)
    )
    _same_lifetimes(data._first_event_times(), _old_first_events(data))


def test_integer_times_keep_their_dtype():
    data = EventHistoryData(
        np.array([1, 4, 2, 7]), np.array([1, 1, 2, 2]), 0, 1
    )
    assert data.get_interarrival_times().dtype == _old_interarrival(data).dtype
    assert np.array_equal(data.get_interarrival_times(), [1, 3, 2, 5])
    assert np.array_equal(data.get_previous_x(), _old_previous_x(data))


@pytest.mark.parametrize("seed", range(3))
def test_fitted_ages_as_before(seed):
    data = next(_datasets(seed))
    model = CrowAMSAA.fit_from_recurrent_data(data)
    item = np.asarray(data.i)
    x = np.asarray(data.x, dtype=float).reshape(item.size, -1).max(axis=1)
    old = np.array([x[item == u].max() for u in np.unique(item)])
    assert np.array_equal(model._ages(None), old)


# -- the item checks: the same error, or none, for the same input -----------


def _outcome(*args, **kwargs):
    try:
        data = handle_xicn(*args, **kwargs)
    except ValueError as error:
        return "error", str(error)
    return "ok", (data.x.tobytes(), data.i.tolist(), data.c.tobytes())


def _old_checks(monkeypatch):
    monkeypatch.setattr(
        ru, "_check_censoring_positions", _old_censoring_positions
    )
    monkeypatch.setattr(
        ru, "_check_observation_windows", _old_observation_windows
    )
    monkeypatch.setattr(ru, "_check_interval_overlaps", _old_interval_overlaps)


def _corrupted(rng, intervals):
    x, i, c, n = _random_rows(rng, 12, intervals=intervals)
    kwargs = {}
    roll = rng.integers(0, 6)
    rows = rng.integers(0, len(c), int(rng.integers(1, 3)))
    if roll == 0:  # censoring codes moved: two ends, an end not last, ...
        c = c.copy()
        c[rows] = rng.choice([-1, 0, 1], rows.size)
        if intervals:
            c[(c == -1)] = 0
    elif roll == 1:  # truncation bounds, sometimes not one per item
        tl = np.full(len(c), -np.inf)
        for item in np.unique(i):
            lowest = np.min(x[i == item])
            tl[i == item] = lowest - rng.choice([1.0, 0.0, -0.5])
        tl[rows] -= rng.choice([0.0, 0.25])
        kwargs["tl"] = tl
    elif roll == 2:  # right truncation, some at or before the last time
        tr = np.full(len(c), np.inf)
        for item in np.unique(i):
            top = np.max(x[i == item])
            tr[i == item] = top + rng.choice([1.0, 0.0, -0.5, -40.0])
        kwargs["tr"] = tr
    elif roll == 3:  # an event before the origin
        x = x.copy()
        x[rows] = -1.0
    elif roll == 4 and intervals:  # a window given twice / overlapping
        x = np.vstack([x, x[rows]])
        x[-1] = x[-1] - rng.choice([0.0, 0.25])
        i = np.append(i, i[rows])
        c = np.append(c, np.full(rows.size, 2))
        n = np.append(n, np.ones(rows.size, dtype=int))
    return (x, i, c, n), kwargs


@pytest.mark.parametrize("intervals", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_item_checks_same_outcome(monkeypatch, seed, intervals):
    rng = np.random.default_rng(1000 + seed)
    args, kwargs = _corrupted(rng, intervals)
    new = _outcome(*args, **kwargs)
    with monkeypatch.context() as patched:
        _old_checks(patched)
        old = _outcome(*args, **kwargs)
    assert new == old


@pytest.mark.parametrize("seed", range(10))
def test_static_covariate_check_same_outcome(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, 15)
    idx = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    Z = np.repeat(rng.normal(size=(15, 2)), sizes, axis=0)
    if seed % 2:
        Z[rng.integers(0, len(Z))] += 1.0
    unique_i = np.arange(15)
    results = []
    for check in (_static_message, _old_static_covariates):
        try:
            check(Z, idx, unique_i)
            results.append(None)
        except ValueError as error:
            results.append(str(error))
    assert results[0] == results[1]
