"""
The imperfect-repair likelihoods evaluate their per-item recursions for all
the items at once (#515). These tests hold the vectorised forms to the
per-item, per-event loops they replaced, kept here as private reference
copies: the ARA and Kijima virtual ages bit for bit, the G1 likelihood to
the last digits (it now sums the terms with ``np.sum`` rather than a running
total).
"""

import numpy as np
import pytest

import surpyval as surv
from surpyval import handle_xicn
from surpyval.recurrent import ARA, GeneralizedOneRenewal, GeneralizedRenewal
from surpyval.recurrent.renewal.ara import ARAVirtualAges, ara_virtual_ages
from surpyval.recurrent.renewal.generalized_one_renewal import (
    _outside_open_bounds,
)
from surpyval.recurrent.renewal.generalized_renewal import (
    KijimaIIVirtualAges,
    _previous_in_item,
)
from surpyval.recurrent.renewal.renewal_model import event_positions

MEMORIES = [1, 2, 3, 5, 8, 9, 130, np.int64(4), np.inf]


def _old_ara_virtual_ages(arrival_times, rho, m):
    T = np.asarray(arrival_times, dtype=float)
    length = T.size
    v = np.zeros(length)
    for i in range(1, length):
        upper = i if np.isinf(m) else min(int(m), i)
        j = np.arange(upper)
        v[i] = T[i - 1] - rho * np.sum(((1.0 - rho) ** j) * T[i - 1 - j])
    return v


def _old_kijima_ii(previous_interarrival_times, q):
    v = 0
    return np.array(
        [v := q * (v + x) for x in previous_interarrival_times]  # noqa
    )


def _old_previous(values, i):
    _, idx = np.unique(i, return_index=True)
    return np.concatenate(
        [
            np.concatenate([[0], np.atleast_1d(arr)])[:-1]
            for arr in np.split(values, idx)[1:]
        ]
    )


def _old_g1_negll(x, i, c, n, dist):
    def negll_func(params):
        q = params[0]
        dist_params = params[1:]
        if not q > -1 or _outside_open_bounds(dist_params, dist.bounds):
            return np.inf
        log1p_q = np.log1p(q)
        ll = 0.0
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for item in set(i):
                mask_item = i == item
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])
                for j in range(0, len(x_item)):
                    log_cj = j * log1p_q
                    xj = x_item[j] * np.exp(-log_cj)
                    if c_item[j] == 0:
                        ll += n_item[j] * (
                            dist.log_df(xj, *dist_params) - log_cj
                        )
                    elif c_item[j] == 1:
                        ll += n_item[j] * dist.log_sf(xj, *dist_params)
        if not np.isfinite(ll):
            return np.inf
        return -ll

    return negll_func


def _ragged(rng, n_items, longest):
    """Items of random lengths (1 row up to ``longest``), rows grouped by
    item, with interarrival times over several orders of magnitude."""
    lengths = rng.integers(1, longest + 1, n_items)
    gaps = [rng.exponential(10 ** rng.uniform(-3, 3), k) for k in lengths]
    i = np.repeat(np.arange(n_items), lengths)
    return gaps, i


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "n_items,longest", [(1, 1), (1, 40), (7, 12), (60, 5)]
)
def test_ara_virtual_ages_bit_identical(seed, n_items, longest):
    rng = np.random.default_rng(seed)
    gaps, i = _ragged(rng, n_items, longest)
    items = [np.cumsum(g) for g in gaps]
    x = np.concatenate(items)
    for m in MEMORIES:
        ages = ARAVirtualAges(x, i, m)
        for rho in (0.0, 1.0, 0.3, float(rng.uniform()), rng.uniform()):
            old = np.concatenate(
                [_old_ara_virtual_ages(a, rho, m) for a in items]
            )
            np.testing.assert_array_equal(ages(rho), old)
            np.testing.assert_array_equal(
                ara_virtual_ages(items[-1], rho, m),
                _old_ara_virtual_ages(items[-1], rho, m),
            )


def test_ara_virtual_ages_long_item_bit_identical():
    # More than 128 terms per sum: numpy's pairwise summation blocks.
    rng = np.random.default_rng(1)
    T = np.cumsum(rng.exponential(3.0, 300))
    for m in (129, 200, np.inf):
        np.testing.assert_array_equal(
            ara_virtual_ages(T, 0.37, m), _old_ara_virtual_ages(T, 0.37, m)
        )


def test_ara_virtual_ages_empty():
    assert ara_virtual_ages(np.array([]), 0.5, 2).shape == (0,)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize(
    "n_items,longest",
    [(1, 1), (1, 300), (5, 30), (9, 3), (40, 12), (200, 4)],
)
def test_kijima_ii_virtual_ages_bit_identical(seed, n_items, longest):
    # Covers the whole-array steps, the scalar tail (fewer than 8 items
    # left) and data that never reaches 8 items at a position.
    rng = np.random.default_rng(seed)
    gaps, i = _ragged(rng, n_items, longest)
    interarrival = np.concatenate(gaps)
    x = np.concatenate([np.cumsum(g) for g in gaps])
    previous = _previous_in_item(interarrival, i)
    np.testing.assert_array_equal(previous, _old_previous(interarrival, i))
    np.testing.assert_array_equal(
        _previous_in_item(x, i), _old_previous(x, i)
    )
    _, idx = np.unique(i, return_index=True)
    ages = KijimaIIVirtualAges(previous, i)
    for q in (0.0, 1.0, 2.5, 1e-300, 1e300, float(rng.uniform(0, 3))):
        with np.errstate(over="ignore", invalid="ignore"):
            old = np.concatenate(
                [_old_kijima_ii(a, q) for a in np.split(previous, idx)[1:]]
            )
            np.testing.assert_array_equal(ages(q), old)


def test_event_positions_any_row_order():
    i = np.array([3, 1, 3, 2, 1, 3])
    np.testing.assert_array_equal(event_positions(i), [0, 0, 1, 0, 1, 2])
    assert event_positions(np.array([])).shape == (0,)


def _fleet(seed, n_items=12):
    rng = np.random.default_rng(seed)
    xs, iis, cs = [], [], []
    for k in range(n_items):
        t = np.cumsum(rng.weibull(2.0, rng.integers(1, 9)) * 10)
        if rng.uniform() < 0.7:
            # Right-censored end of observation after the last failure.
            t = np.append(t, t[-1] + rng.uniform(0.1, 5))
            c = [0] * (t.size - 1) + [1]
        else:
            c = [0] * t.size
        xs += list(t)
        iis += [k] * t.size
        cs += c
    return handle_xicn(np.array(xs), np.array(iis), np.array(cs))


def _old_negll_ages(data, virtual_ages, dist, dist_params):
    interarrival = data.get_interarrival_times()
    x_new = interarrival + virtual_ages
    with np.errstate(divide="ignore"):
        log_sf_v = dist.log_sf(virtual_ages, *dist_params)
        ll_o = dist.log_df(x_new, *dist_params) - log_sf_v
        ll_right = dist.log_sf(x_new, *dist_params) - log_sf_v
    ll = np.where(data.c == 0, ll_o, 0.0)
    ll = np.where(data.c == 1, ll_right, ll)
    return -ll.sum()


@pytest.mark.parametrize("seed", range(3))
def test_ara_and_kijima_likelihoods_bit_identical(seed):
    data = _fleet(seed)
    _, idx = np.unique(data.i, return_index=True)
    items = np.split(data.x, idx)[1:]
    previous = _old_previous(data.get_interarrival_times(), data.i)
    rng = np.random.default_rng(seed)
    for _ in range(4):
        params = np.append(rng.uniform(0, 1), rng.uniform(1, 20, 2))
        r, alpha, beta = params
        for m in (1, 2, 5, np.inf):
            ages = np.concatenate(
                [_old_ara_virtual_ages(a, r, m) for a in items]
            )
            assert ARA.create_negll_func(data, surv.Weibull, m)(
                params
            ) == _old_negll_ages(data, ages, surv.Weibull, [alpha, beta])
        ages = np.concatenate(
            [_old_kijima_ii(a, r) for a in np.split(previous, idx)[1:]]
        )
        negll = GeneralizedRenewal.create_negll_func(
            data, surv.Weibull, kijima="ii"
        )
        assert negll(params) == _old_negll_ages(
            data, ages, surv.Weibull, [alpha, beta]
        )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "dist", [surv.Weibull, surv.LogNormal, surv.Gamma, surv.Exponential]
)
def test_g1_likelihood_matches_loop(seed, dist):
    rng = np.random.default_rng(seed)
    n_items = int(rng.choice([1, 3, 20]))
    lengths = rng.integers(1, 9, n_items)
    x = rng.weibull(1.5, lengths.sum()) * 10
    i = np.repeat(np.arange(n_items), lengths)
    c = np.zeros(x.size, dtype=int)
    last = np.cumsum(lengths) - 1
    c[last[rng.uniform(size=n_items) < 0.6]] = 1
    n = rng.integers(1, 4, x.size)
    new = GeneralizedOneRenewal.create_negll_func(x, i, c, n, dist)
    old = _old_g1_negll(x, i, c, n, dist)
    k = len(dist.parameter_names)
    for _ in range(6):
        params = np.concatenate(
            [[rng.uniform(-0.95, 3)], rng.uniform(0.3, 20, k)]
        )
        np.testing.assert_allclose(new(params), old(params), rtol=1e-12)
    # At q = -1 and outside the parameter bounds: both infinite.
    for params in ([-1.0] + [2.0] * k, [0.5] + [-1.0] * k):
        assert new(np.array(params)) == old(np.array(params)) == np.inf
    # A huge q with tiny parameters: finite, and still the same.
    params = np.array([50.0] + [1e-300] * k)
    np.testing.assert_allclose(new(params), old(params), rtol=1e-12)


def test_renewal_residuals_unchanged():
    # The time-rescaling residuals use the same vectorised virtual ages.
    data = _fleet(0)
    ara = ARA.fit_from_recurrent_data(data, m=2)
    _, idx = np.unique(data.i, return_index=True)
    ages = np.concatenate(
        [
            _old_ara_virtual_ages(a, ara.rho, 2)
            for a in np.split(data.x, idx)[1:]
        ]
    )
    x_new = data.get_interarrival_times() + ages
    with np.errstate(divide="ignore"):
        expected = ara.model.Hf(x_new) - ara.model.Hf(ages)
    np.testing.assert_array_equal(
        ARA._rescaled_increments(ara, data), expected
    )

    g1 = GeneralizedOneRenewal.fit_from_recurrent_data(data)
    scaled = np.concatenate(
        [
            np.asarray(arr, dtype=float)
            / (1.0 + g1.q) ** np.arange(len(arr))
            for arr in np.split(data.get_interarrival_times(), idx)[1:]
        ]
    )
    np.testing.assert_array_equal(
        GeneralizedOneRenewal._rescaled_increments(g1, data),
        g1.model.Hf(scaled),
    )
