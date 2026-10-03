"""The log-rank risk sets and variance without dense arrays (#515).

``_logrank_z_v`` built ``x_i[:, None] >= event_times`` arrays, O(n * m) in
time and memory: 13 s and 2 GB for 3e4 rows in three groups, and more
memory than a machine has at 1e5. It now counts with ``searchsorted`` and
``bincount``. The counts are whole numbers, so every sum is exact, and the
variance keeps the old products in the old order: the result must be
bit-identical to the dense implementation, kept here as the reference.
"""

import importlib
import tracemalloc

import numpy as np
import pytest

import surpyval
from surpyval.univariate.nonparametric.kaplan_meier import kaplan_meier
from surpyval.utils import xcnt_handler

# The module, not the ``logrank`` function the package exports under its name.
logrank_module = importlib.import_module(
    "surpyval.univariate.nonparametric.logrank"
)


def _dense_logrank_z_v(x, Z, c, n, groups, weighting, rho, gamma, tl=None):
    """The implementation before #515, verbatim, as the reference (it has
    no entry times, ``tl``, #576)."""
    assert tl is None
    k = groups.size
    x_g, c_g, n_g = [], [], []
    for g in groups:
        mask = Z == g
        x_i = np.atleast_1d(x)[mask]
        c_i = None if c is None else np.atleast_1d(c)[mask]
        n_i = None if n is None else np.atleast_1d(n)[mask]
        if x_i.size == 0:
            x_g.append(np.array([]))
            c_g.append(np.array([]))
            n_g.append(np.array([]))
            continue
        x_i, c_i, n_i, _ = xcnt_handler(x=x_i, c=c_i, n=n_i)
        if ((c_i != 0) & (c_i != 1)).any():
            raise ValueError(
                "Log-rank test can only be used with observed and "
                + "right censored data"
            )
        x_g.append(x_i)
        c_g.append(c_i)
        n_g.append(n_i)

    event_pool = [x_i[c_i == 0] for x_i, c_i in zip(x_g, c_g) if x_i.size > 0]
    event_times = (
        np.unique(np.concatenate(event_pool)) if event_pool else np.array([])
    )
    m = event_times.size
    if m == 0:
        return np.zeros(k), np.zeros((k, k)), np.zeros(k)

    r_gt = np.zeros((k, m))
    d_gt = np.zeros((k, m))
    for j, (x_i, c_i, n_i) in enumerate(zip(x_g, c_g, n_g)):
        if x_i.size == 0:
            continue
        r_gt[j] = (n_i[:, None] * (x_i[:, None] >= event_times)).sum(axis=0)
        d_gt[j] = (
            n_i[:, None]
            * ((x_i[:, None] == event_times) & (c_i == 0)[:, None])
        ).sum(axis=0)

    r_t = r_gt.sum(axis=0)
    d_t = d_gt.sum(axis=0)

    if weighting == "log-rank":
        w = np.ones(m)
    elif weighting == "gehan":
        w = r_t
    elif weighting == "tarone-ware":
        w = np.sqrt(r_t)
    else:
        S = kaplan_meier(r_t, d_t)
        S_prev = np.hstack([[1.0], S[:-1]])
        w = S_prev**rho * (1 - S_prev) ** gamma

    with np.errstate(all="ignore"):
        expected = np.where(r_t > 0, d_t * r_gt / r_t, 0.0)
    z = (w * (d_gt - expected)).sum(axis=1)

    with np.errstate(all="ignore"):
        hyper = np.where(r_t > 1, d_t * (r_t - d_t) / (r_t - 1), 0.0)
        prop = np.where(r_t > 0, r_gt / r_t, 0.0)
    V = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            delta = 1.0 if a == b else 0.0
            V[a, b] = (w**2 * hyper * prop[a] * (delta - prop[b])).sum()

    return z, V, expected.sum(axis=1)


WEIGHTINGS = [
    ("log-rank", 0, 0),
    ("gehan", 0, 0),
    ("tarone-ware", 0, 0),
    ("fleming-harrington", 1, 0),
    ("fleming-harrington", 0.5, 2),
]


def _datasets():
    rng = np.random.default_rng(515)
    for size, groups in [(1, 2), (2, 2), (7, 3), (60, 2), (300, 4)]:
        # One row per group at least, so every label is a group.
        Z = np.concatenate([np.arange(groups), rng.integers(0, groups, size)])
        N = Z.size
        x = rng.weibull(1.5, N) * (5 + Z)
        yield f"continuous {N}x{groups}", x, Z, None, None, None
        c = (rng.uniform(size=N) < 0.3).astype(int)
        yield f"censored {N}x{groups}", x, Z, c, None, None
        # Heavy ties across and within groups, with counts.
        xt = np.ceil(x)
        n = rng.integers(1, 5, N)
        yield f"tied with counts {N}x{groups}", xt, Z, c, n, None
        strata = rng.integers(0, 3, N)
        yield f"stratified {N}x{groups}", xt, Z, c, n, strata
    # String labels; a group whose members are all censored; a group only
    # at risk after the last event; no events at all.
    yield "string labels", [3, 5, 5, 8, 2, 4], list("aabbcc"), [
        0,
        1,
        0,
        0,
        1,
        0,
    ], None, None
    yield "group all censored", [1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1], [
        0,
        0,
        0,
        1,
        1,
        1,
    ], None, None
    yield "group after last event", [1, 2, 3, 9, 10], [0, 0, 0, 1, 1], [
        0,
        0,
        0,
        1,
        1,
    ], None, None
    yield "no events", [1, 2, 3, 4], [0, 1, 0, 1], [1, 1, 1, 1], None, None
    # A group absent from one stratum.
    yield "group missing from a stratum", [1, 2, 3, 4, 5, 6, 7, 8], [
        0,
        1,
        2,
        0,
        1,
        0,
        1,
        2,
    ], None, None, [0, 0, 0, 0, 1, 1, 1, 1]


def _fields(result):
    return (
        result.statistic,
        result.dof,
        result.p_value,
        result.weighting,
        result.strata,
    )


@pytest.mark.parametrize("weighting,rho,gamma", WEIGHTINGS)
def test_logrank_bit_identical_to_the_dense_implementation(
    monkeypatch, weighting, rho, gamma
):
    for label, x, Z, c, n, strata in _datasets():
        kwargs = dict(
            c=c, n=n, weighting=weighting, rho=rho, gamma=gamma, strata=strata
        )
        new = surpyval.logrank(x, Z, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(logrank_module, "_logrank_z_v", _dense_logrank_z_v)
            old = surpyval.logrank(x, Z, **kwargs)
        assert _fields(new) == _fields(old), label


@pytest.mark.parametrize("weighting,rho,gamma", WEIGHTINGS)
def test_z_v_and_expected_bit_identical(weighting, rho, gamma):
    for label, x, Z, c, n, _ in _datasets():
        x, Z = np.asarray(x, dtype=float), np.asarray(Z)
        c = None if c is None else np.asarray(c)
        n = None if n is None else np.asarray(n)
        args = (x, Z, c, n, np.unique(Z), weighting, rho, gamma)
        for new, old in zip(
            logrank_module._logrank_z_v(*args), _dense_logrank_z_v(*args)
        ):
            np.testing.assert_array_equal(new, old, err_msg=label)


def test_counts_equal_repeated_rows():
    # Counts are unit rows repeated: the at-risk and event counts, and so
    # the statistic, are the same either way.
    x = np.array([2.0, 3.0, 3.0, 5.0, 1.0, 3.0, 4.0])
    Z = np.array([0, 0, 0, 0, 1, 1, 1])
    c = np.array([0, 0, 1, 0, 0, 0, 1])
    n = np.array([2, 3, 1, 2, 1, 4, 2])
    grouped = surpyval.logrank(x, Z, c=c, n=n)
    expanded = surpyval.logrank(
        np.repeat(x, n), np.repeat(Z, n), c=np.repeat(c, n)
    )
    assert grouped.statistic == pytest.approx(expanded.statistic, rel=1e-12)
    assert grouped.dof == expanded.dof


def _three_groups(N):
    rng = np.random.default_rng(0)
    Z = rng.integers(0, 3, N)
    x = rng.weibull(1.5, N) * (10 + 2 * Z)
    c = (rng.uniform(size=N) < 0.3).astype(int)
    return x, Z, c


def test_memory_is_linear_in_the_sample():
    # 6,000 rows: the dense arrays peaked at about 100 MB (they grow with
    # n * m); the counts need well under 5 MB.
    x, Z, c = _three_groups(6_000)
    tracemalloc.start()
    try:
        surpyval.logrank(x, Z, c=c)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 5e6


def test_large_sample_runs():
    # 1e5 rows in three groups: the dense arrays needed ~56 GB.
    x, Z, c = _three_groups(100_000)
    result = surpyval.logrank(x, Z, c=c)
    assert result.dof == 2
    assert result.statistic > 0
