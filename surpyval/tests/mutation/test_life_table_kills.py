"""Checks that kill mutants of the actuarial life table
(``nonparametric/life_table.py``) that the suite let survive (#396).

The suite checked the table against R's ``KMsurv::lifetab`` with the
intervals given; the default intervals (the Freedman-Diaconis rule), the
intervals nobody is at risk in, counts with late entry and the data held
by a fitted model were not checked. The values here are closed forms
(the rule, and Klein and Moeschberger's equations 5.4.1-5.4.6 as the
docstring writes them) or the table of the rows repeated.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval import life_table

X = np.array([0.7, 1.2, 1.9, 2.4, 3.1, 3.3, 4.8, 5.5, 6.2, 7.9, 9.4, 12.0])
C = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
N = np.array([2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 3, 1])


def _repeat(n, *arrays):
    rows = np.repeat(np.arange(len(n)), n.astype(int))
    return [np.asarray(a)[rows] for a in arrays]


@pytest.mark.parametrize(
    "x, n",
    [(X, N), (X, None), (X - 3.0, None), (X / 10, None)],
    ids=["counts", "rows", "negative-times", "narrow"],
)
def test_default_intervals_are_freedman_diaconis(x, n):
    """Kills, in ``_edges``, the arithmetic edits of the default width
    ``2 IQR / m**(1/3)`` (``m`` the units, each count repeated; the
    quartiles at 25 and 75), the guard ``width > 0`` made ``> 1``, the
    number of intervals ``floor((top - origin) / width) + 1`` and the
    boundaries ``origin + width * arange(k + 1)``, from the origin (0, or
    the smallest time where it is negative), and the counts left out of
    the quartiles. No test used the default intervals."""
    times = x if n is None else np.repeat(x, n)
    origin = min(0.0, x.min())
    q75, q25 = np.percentile(times, [75, 25])
    width = 2 * (q75 - q25) / times.size ** (1 / 3)
    k = int(np.floor((x.max() - origin) / width)) + 1
    table = life_table(x, C, n)
    assert_allclose(table["start"], origin + width * np.arange(k))
    assert_allclose(table["end"], origin + width * np.arange(1, k + 1))
    units = np.ones(x.size) if n is None else n
    assert table["events"].sum() == units[C == 0].sum()


@pytest.mark.parametrize("low", [1.0, -1.0])
def test_default_width_of_tied_quartiles(low):
    """Kills, in ``_edges``, the fallback width where the quartiles tie
    (``IQR = 0``): a tenth of the span from the origin, ``(top - origin) /
    10``, with its edits (``* 10``, ``+ origin``, ``/ 11``, the ``and
    False`` of its condition)."""
    x = np.array([low, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0])
    origin = min(0.0, low)
    width = (9.0 - origin) / 10
    table = life_table(x)
    assert_allclose(table["start"], origin + width * np.arange(11))
    assert_allclose(table["end"].iloc[-1], origin + width * 11)


def test_every_time_at_the_origin():
    """Kills, in ``_edges``, the fallback width's condition ``top >
    origin`` made ``or True`` or ``>=`` (a width of 0) and its other
    value 1 made 2; and, for a number of intervals, ``not top > origin``
    made ``>=`` or the open interval ``np.array(None)``: with every time
    at 0 the table is one interval, ``[0, 1)`` by default and ``[0,
    inf)`` for a number of them."""
    zeros = np.zeros(3)
    table = life_table(zeros)
    assert table["start"].tolist() == [0.0] and table["end"].tolist() == [1.0]
    three = life_table(zeros, intervals=3)
    assert three["start"].tolist() == [0.0] and np.isinf(three["end"].iloc[0])
    assert three["events"].iloc[0] == 3


def test_entry_before_the_first_time_sets_the_origin():
    """Kills, in ``_origin``, the entry times left out (``and False``):
    the table starts at the earliest time or entry where it is
    negative."""
    table = life_table([1.0, 2.0, 3.0, 4.0], tl=[-2.0, 0.0, 0.0, 0.0])
    assert table["start"].iloc[0] == -2.0


def test_one_interval_and_two_boundaries_are_allowed():
    """Kills, in ``_edges``, the refusal of one interval (``int(k) <= 1``,
    ``< 2``) and of two boundaries (``size <= 2``, ``< 3``): both are a
    table of a single interval."""
    one = life_table(X, C, N, intervals=1)
    assert one["start"].tolist() == [0.0] and np.isinf(one["end"].iloc[0])
    two = life_table(X, C, N, intervals=[0, 20])
    assert two["start"].tolist() == [0.0] and two["end"].tolist() == [20.0]
    assert two["events"].iloc[0] == N[C == 0].sum()


def test_equal_boundaries_are_refused():
    """Kills, in ``_edges``, ``diff(edges) >= 0``: an interval of zero
    width is refused."""
    with pytest.raises(ValueError, match="increasing"):
        life_table(X, C, N, intervals=[0, 5, 5, 20])


def test_an_interval_nobody_is_at_risk_in():
    """Kills, in ``_actuarial``, the intervals with nobody at risk taken
    as having some (``Y >= 0``) or given ``p = 2`` in the product, and
    the Greenwood term of 1 there: the survival and its standard error
    are carried unchanged over an empty interval to the next. Here
    everyone has left by 4 (two failures, two withdrawals), and two
    units enter at 6.5."""
    x = np.array([1.0, 2.0, 2.5, 3.0, 7.0, 8.0])
    c = np.array([0, 0, 1, 1, 0, 0])
    tl = np.array([0.0, 0.0, 0.0, 0.0, 6.5, 6.5])
    table = life_table(x, c, tl=tl, intervals=[0, 2, 4, 6, 9])
    empty = table.iloc[2]
    assert empty["at_risk"] == 0 and np.isnan(empty["q"])
    # 1 - 1/4, then 1 - 1/2 (two at risk, one failure)
    assert_allclose(table["surv"].iloc[2:4], [0.375, 0.375])
    assert_allclose(table["se_surv"].iloc[3], table["se_surv"].iloc[2])
    # The two late entrants at 6.5 are at risk in [6, 9), half of it
    assert table["at_risk"].iloc[3] == 1.0


def test_standard_error_after_everyone_failed_is_zero():
    """Kills, in ``_actuarial``, ``se_surv`` where the survival is 0 left
    as ``0 * sqrt(inf)`` (``S >= 0``, nan) or set to 1: an interval
    after one where every unit failed has survival 0 and standard error
    0."""
    x = np.array([1.0, 1.5, 2.0])
    table = life_table(x, intervals=[0, 2.5, 5, 10])
    assert table["surv"].iloc[1] == 0 and table["se_surv"].iloc[1] == 0
    assert table["surv"].iloc[2] == 0 and table["se_surv"].iloc[2] == 0


def test_hazard_standard_error_on_a_width_other_than_two():
    """Kills, in ``_actuarial``, ``width**2`` made ``width * 2`` in the
    hazard's variance, the same where the width is 2 (R's ``lifetab``
    references): Klein and Moeschberger's (5.4.6), ``Var(h) = h**2 / (Y
    q) (1 - (h b / 2)**2)``, here with widths of 3."""
    x = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9]
    c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    table = life_table(x, c, intervals=[0, 3, 6, 9]).iloc[:3]
    h, Y, q = table["hazard"], table["at_risk"], table["q"]
    var = h**2 / (Y * q) * (1 - (h * 3 / 2) ** 2)
    assert_allclose(table["se_hazard"], np.sqrt(var), rtol=1e-12)


def test_counts_with_late_entry_are_repeated_rows():
    """Kills, in ``_counts``, the late entrants counted as rows, not by
    their counts (``weights=None`` or none); and, in ``_edges``, the
    default intervals taken without the counts (``and False``,
    ``np.all(None)``, ``n != n.round()``)."""
    tl = np.where(np.arange(X.size) % 3 == 1, X * 0.5, 0.0)
    counted = life_table(X, C, N, tl=tl)
    xr, cr, tlr = _repeat(N, X, C, tl)
    repeated = life_table(xr, cr, tl=tlr)
    late = tl > 0  # none of them on a boundary
    assert counted["entered"].sum() == N[late].sum()
    assert_allclose(
        counted.to_numpy(dtype=float), repeated.to_numpy(dtype=float)
    )


def test_a_fitted_model_with_counts():
    """Kills, in ``_data``, the counts of a fitted non-parametric model's
    data dropped (``None``, left out, ``d.get(None)``): its table is that
    of the arrays it was fitted to."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.KaplanMeier.fit(X, C, N)
    expected = life_table(X, C, N, intervals=[0, 3, 6, 9])
    got = life_table(model, intervals=[0, 3, 6, 9])
    assert_allclose(got.to_numpy(dtype=float), expected.to_numpy(dtype=float))
