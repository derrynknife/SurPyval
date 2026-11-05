"""Checks that kill mutants of the log-rank tests
(``nonparametric/logrank.py``) that the suite let survive (#396).

Most are invariances the suite checked only with one row per unit: a
row with count ``n`` is ``n`` rows, under strata and with ``t0`` as
without. The others check closed forms: the two-group test with entry
times written out, the Fleming-Harrington test with ``rho = gamma = 0``
is the log-rank test, and Holm's adjusted p-values are capped at 1 (R's
``p.adjust``).
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval import logrank, pairwise_logrank
from surpyval.univariate.nonparametric.logrank import adjust_p_values


def _data():
    rng = np.random.default_rng(396)
    size = 36
    Z = np.repeat(["a", "b", "c"], size // 3)
    scale = np.select([Z == "a", Z == "b"], [8.0, 5.0], 6.0)
    x = np.round(rng.weibull(1.3, size) * scale, 1) + 0.1
    c = (rng.uniform(size=size) < 0.25).astype(int)
    n = rng.integers(1, 4, size)
    tl = np.where(np.arange(size) % 4 == 0, np.round(x * 0.3, 1), 0.0)
    strata = np.where(np.arange(size) % 2 == 0, "s1", "s2")
    return x, Z, c, n, tl, strata


def _repeat(n, *arrays):
    rows = np.repeat(np.arange(len(n)), n)
    return [a[rows] for a in arrays]


@pytest.mark.parametrize(
    "options",
    [
        {"strata": True},
        {"tl": True},
        {"t0": 4.0},
        {"t0": 4.0, "tl": True, "strata": True},
    ],
    ids=["strata", "entry", "t0", "t0-entry-strata"],
)
def test_counts_are_repeated_rows(options):
    """Kills, in ``_logrank``, the counts dropped within a stratum
    (``None if n is None or True``) and before the cut at ``t0``; in
    ``_cut_at``, the counts dropped (``or True``)."""
    x, Z, c, n, tl, strata = _data()
    kw = {}
    rep = {}
    xr, Zr, cr, tlr, sr = _repeat(n, x, Z, c, tl, strata)
    if options.get("strata"):
        kw["strata"], rep["strata"] = strata, sr
    if options.get("tl"):
        kw["tl"], rep["tl"] = tl, tlr
    if "t0" in options:
        kw["t0"] = rep["t0"] = options["t0"]
    counted = logrank(x, Z, c, n, **kw)
    repeated = logrank(xr, Zr, cr, **rep)
    assert_allclose(counted.statistic, repeated.statistic, rtol=1e-10)
    assert counted.dof == repeated.dof == 2


def _by_hand(x, Z, c, n, tl):
    """The two-group log-rank statistic written out: at each event time
    ``t`` the units at risk are those with ``tl < t <= x``; ``O - E`` and
    the hypergeometric variance summed over the event times."""
    o_e = var = 0.0
    for t in np.unique(x[c == 0]):
        at = (tl < t) & (x >= t)
        r, r1 = n[at].sum(), n[at & (Z == "a")].sum()
        dead = (x == t) & (c == 0)
        d, d1 = n[dead].sum(), n[dead & (Z == "a")].sum()
        o_e += d1 - d * r1 / r
        if r > 1:
            var += d * (r1 / r) * (1 - r1 / r) * (r - d) / (r - 1)
    return o_e**2 / var


@pytest.mark.parametrize("t0", [None, 4.0])
def test_entry_times_with_counts_by_hand(t0):
    """Kills, in ``_logrank_z_v``, the count of the rows not yet entered
    summed forward (``[::+1]``), the same sum where every count is 1;
    and, in ``_cut_at``, the entry times dropped after the cut at ``t0``
    (``or True``): the data cut at ``t0`` (later times censored there,
    rows entering at or after it dropped) keep their entry times."""
    x, Z, c, n, tl, _ = _data()
    two = Z != "c"
    x, Z, c, n, tl = x[two], Z[two], c[two], n[two], tl[two]
    res = logrank(x, Z, c, n, tl=tl, t0=t0)
    if t0 is not None:
        keep = tl < t0
        c = np.where(x > t0, 1, c)[keep]
        x = np.minimum(x, t0)[keep]
        Z, n, tl = Z[keep], n[keep], tl[keep]
    assert_allclose(res.statistic, _by_hand(x, Z, c, n, tl), rtol=1e-10)


def test_a_group_in_one_stratum_only():
    """Kills, in ``_logrank``, the expected counts of the strata kept from
    the last stratum alone (``E = E_s``): a group with events only in an
    earlier stratum was then left out as never expected to fail, one
    degree of freedom too few."""
    x = np.array([1.0, 2, 3, 4, 5, 6, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    Z = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "a", "b", "a", "b"])
    strata = np.array(["s1"] * 6 + ["s2"] * 6)
    res = logrank(x, Z, strata=strata)
    assert res.dof == 2


@pytest.mark.parametrize("call", ["logrank", "pairwise"])
def test_fleming_harrington_defaults_are_the_log_rank(call):
    """Kills the default ``rho = 1`` (``logrank`` and
    ``pairwise_logrank``) and ``gamma = 1`` (``pairwise_logrank``): the
    defaults are ``rho = gamma = 0``, whose weight ``S**0 (1 - S)**0`` is
    1, the log-rank test."""
    x, Z, c, _, _, _ = _data()
    if call == "logrank":
        fh = logrank(x, Z, c, weighting="fleming-harrington").statistic
        lr = logrank(x, Z, c).statistic
    else:
        fh = pairwise_logrank(x, Z, c, weighting="fleming-harrington").table[
            "statistic"
        ]
        lr = pairwise_logrank(x, Z, c).table["statistic"]
    assert_allclose(fh, lr, rtol=1e-12)


def test_t0_without_censoring_flags():
    """Kills, in ``_cut_at``, ``c`` left as ``np.asarray(None)`` (``c is
    None and False``) or ``np.zeros(None)``: with ``c`` not given every
    row failed, and the cut at ``t0`` must work as with ``c`` all 0."""
    x, Z, _, _, _, _ = _data()
    given = logrank(x, Z, np.zeros(x.size, dtype=int), t0=4.0).statistic
    assert_allclose(logrank(x, Z, t0=4.0).statistic, given, rtol=1e-12)


def test_holm_is_capped_at_one():
    """Kills, in ``adjust_p_values``, the cap of Holm's and BH's adjusted
    p-values made 2: R's ``p.adjust(c(0.5, 0.6, 0.7), "holm")`` is
    ``1 1 1`` (``0.5 * 3`` capped), ``"BH"`` is ``0.7 0.7 0.7``."""
    assert_allclose(adjust_p_values([0.5, 0.6, 0.7], "holm"), [1, 1, 1])
    assert_allclose(adjust_p_values([0.5, 0.6, 0.7], "bh"), [0.7] * 3)


@pytest.mark.parametrize(
    "sizes, warns",
    [
        ([1, 1, 5, 5], False),  # half the groups single: not most
        ([1, 1, 1, 5, 5], True),
        ([2, 2, 2, 5], False),  # two units is not one
    ],
)
def test_singleton_warning_threshold(sizes, warns):
    """Kills, in ``_logrank``, the singleton warning's thresholds: a group
    of two counted as single (``<= 2``), and "most groups" made ``>= k /
    2`` or ``> k / 3``."""
    Z = np.repeat(np.arange(len(sizes)), sizes)
    x = np.arange(1.0, Z.size + 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logrank(x, Z)
    single = [w for w in caught if "single member" in str(w.message)]
    assert bool(single) is warns


def test_singleton_warning_counts_the_units():
    """Kills, in ``_logrank``, the group sizes of the singleton warning
    taken as rows, not units (``n`` ignored): four groups of one row with
    a count of 5 each are groups of five units, not singletons."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        logrank([1.0, 2.0, 3.0, 4.0], [0, 1, 2, 3], n=[5, 5, 5, 5])
    assert not [w for w in caught if "single member" in str(w.message)]
