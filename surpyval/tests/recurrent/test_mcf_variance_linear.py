"""The Lawless-Nadeau MCF variance in linear time (#521).

The variance summed each item's deviations over the whole time grid, one
item at a time, and found each item's observation window with a pass
over every row: O(items x times), 4.6 s at 5k items. It is now
accumulated from each item's own rows and window ends. These check it
against the sum over items, written out here as it was, on plain,
truncated, windowed, tied and cause-specific data, and that doubling the
items no longer quadruples the time.
"""

import time

import numpy as np
import pytest

from surpyval import handle_xicn
from surpyval.recurrent import CauseSpecificMCF, NonParametricCounting
from surpyval.recurrent.nonparametric.mcf import _lawless_nadeau_var
from surpyval.univariate.competing_risks.labels import label_mask


def _by_items(data, x, r, d, counted=None):
    # The sum over items, as the variance was computed before #521.
    x_out = data.midpoints if data.x.ndim == 2 else data.x
    is_event = (data.c == 0) | (data.c == 2) | (data.c == -1)
    if counted is not None:
        is_event = is_event & counted
    col = np.searchsorted(x, x_out)
    dm = np.where(r > 0, d / np.where(r > 0, r, 1), 0.0)
    inv_r = np.where(r > 0, 1.0 / np.where(r > 0, r, 1), 0.0)
    window_map = getattr(data, "window_map", None) or {}
    clusters: dict = {}
    for item in data.items:
        rows = data.i == item
        entry = float(data.tl[rows][0])
        upper = data.x if data.x.ndim == 1 else data.x[:, 1]
        exit_ = float(upper[rows].max())
        tr = float(data.tr[rows][0])
        exit_ = max(exit_, tr) if np.isfinite(tr) else exit_
        at_risk = (entry <= x) & (x <= exit_)
        n_k = np.bincount(
            col[rows & is_event],
            weights=data.n[rows & is_event],
            minlength=len(x),
        )
        dev = at_risk * inv_r * (n_k - dm)
        key = window_map[item][0] if item in window_map else item
        clusters[key] = clusters.get(key, 0.0) + dev
    total = np.zeros(len(x))
    for dev in clusters.values():
        total += np.cumsum(dev) ** 2
    return total


def _items(n_items, seed=0, spread=1.0, entry=False, tr=False):
    rng = np.random.default_rng(seed)
    x, i, c, tl, trs = [], [], [], [], []
    for k in range(n_items):
        end = round(float(rng.uniform(50, 100)), 2)
        rate = rng.gamma(2.0, spread) / 10.0
        t = np.cumsum(rng.exponential(1 / rate, 40))
        start = float(rng.uniform(0, 20)) if entry else -np.inf
        # On a 0.001 grid (a few ties across items), inside (start, end)
        t = np.unique(np.floor(t * 1000) / 1000)
        t = t[(t < end) & (t > start)]
        close = [] if tr else [end]
        x += t.tolist() + close
        i += [k] * (len(t) + len(close))
        c += [0] * len(t) + [1] * len(close)
        tl += [start] * (len(t) + len(close))
        trs += [end if tr else np.inf] * (len(t) + len(close))
    return [np.array(v) for v in (x, i, c, tl, trs)]


def _check(data, counted=None, xrd=None):
    x, r, d = data.to_xrd() if xrd is None else xrd
    got = _lawless_nadeau_var(data, x, r, d, counted=counted)
    want = _by_items(data, x, r, d, counted=counted)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
    return got


@pytest.mark.parametrize("spread", [1.0, 20.0])
def test_matches_the_sum_over_items(spread):
    # spread 20: nearly every item reaches the generator's 40 events,
    # so the items' counts are alike and the sums of squares nearly
    # cancel (the case the blocks are for).
    x, i, c, _, _ = _items(300, seed=2, spread=spread)
    _check(handle_xicn(x, i, c))


def test_truncated_items():
    x, i, c, tl, tr = _items(300, seed=1, entry=True, tr=True)
    _check(handle_xicn(x, i, c, tl=tl, tr=tr))


def test_windowed_items_are_one_cluster():
    rng = np.random.default_rng(5)
    xs, ii, windows = [], [], {}
    for k in range(60):
        w = [(0.0, 30.0), (40.0, 80.0)] if k % 2 else [(5.0, 70.0)]
        windows[k] = w
        for a, b in w:
            ev = np.unique(np.round(rng.uniform(a, b, 4), 1))
            ev = ev[(ev > a) & (ev <= b)]
            xs += ev.tolist()
            ii += [k] * len(ev)
    model = NonParametricCounting.fit(
        np.array(xs), i=np.array(ii), c=np.zeros(len(xs), int), windows=windows
    )
    _check(model.data)


def test_ties_and_a_single_item():
    data = handle_xicn(
        [1, 2, 2, 5, 3, 3.5, 4, 6, 1, 2, 7],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    )
    _check(data)
    single = NonParametricCounting.fit([1, 2, 3, 5], c=[0, 0, 0, 1])
    np.testing.assert_array_equal(single.var, 0.0)


def test_cause_specific():
    # One cause's events count; the other's are non-events.
    rng = np.random.default_rng(3)
    x, i, c, _, _ = _items(80, seed=4)
    e = np.where(c == 0, rng.choice(["a", "b"], len(x)), None)
    model = CauseSpecificMCF.fit(x, i=i, c=c, e=e)
    data = handle_xicn(x, i, c, e=e)
    got = _check(
        data,
        counted=label_mask(data.e, "a"),
        xrd=data.to_cause_specific_xrd("a"),
    )
    np.testing.assert_allclose(model.models["a"].var, got, rtol=1e-12)


def _seconds(n_items):
    x, i, c, _, _ = _items(n_items, seed=6)
    data = handle_xicn(x, i, c)
    xs, r, d = data.to_xrd()
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        _lawless_nadeau_var(data, xs, r, d)
        best = min(best, time.perf_counter() - start)
    return best


def test_linear_in_the_items():
    # Eight times the items (and as many more distinct times) took about
    # fifty times as long (46 to 53, best of 3); it now takes about ten.
    ratio = _seconds(2400) / _seconds(300)
    assert ratio < 25, ratio
