"""A zero-width interval row through ``fit`` and ``fit_from_df``.

A row of a two-column ``x`` with ``xl == xr`` is an exact time:
``fit_from_df`` says so ("If xl[i] == xr[i] assumes observed",
``surpyval/utils/dataframe.py``), and the array path accepts such a row
with ``c = 0``. With ``c = 2`` the array path refuses it instead, in
``surpyval/utils/data_formats.py::_check_interval_flags``, with a message
about "only one failure time" that does not tell the user which row or
what to do. The strict xfails pin that disagreement; the passing tests
pin the behaviour both paths already share.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import Turnbull, Weibull


def _data():
    rng = np.random.default_rng(1)
    x = rng.weibull(2.0, 30) * 100
    X = np.c_[x, x + 1.0]
    X[3, 1] = X[3, 0]
    return X


def _row_3_exact(X):
    """Row 3 written as an exact time inside the two-column x."""
    c2 = np.full(30, 2)
    c2[3] = 0
    return X.copy(), c2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a c=2 row with xl == xr is refused by fit() with a message about "
        "'one failure time' but accepted as observed by fit_from_df()"
    ),
)
def test_zero_width_interval_row_weibull_fit():
    """Observed at e3412770: ``Weibull.fit(X, c=np.full(30, 2))`` raises
    ``ValueError: Censor flag indicates interval censored but only has
    one failure time`` (``data_formats._check_interval_flags``).
    Expected: the row is taken as exact, as ``fit_from_df`` does, giving
    the fit with row 3 flagged ``c = 0`` (or, if both paths should refuse,
    a message naming the row and the fix).
    """
    X = _data()
    m = Weibull.fit(X, c=np.full(30, 2))
    X2, c2 = _row_3_exact(X)
    np.testing.assert_allclose(
        m.params, Weibull.fit(X2, c=c2).params, rtol=1e-10
    )


def test_zero_width_interval_row_fit_from_df_is_exact():
    X = _data()
    df = pd.DataFrame({"xl": X[:, 0], "xr": X[:, 1]})
    m = Weibull.fit_from_df(df, xl_col="xl", xr_col="xr")
    X2, c2 = _row_3_exact(X)
    np.testing.assert_allclose(
        m.params, Weibull.fit(X2, c=c2).params, rtol=1e-10
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a c=2 row with xl == xr is refused by Turnbull.fit() with a "
        "message about 'one failure time' instead of being taken as exact"
    ),
)
def test_zero_width_interval_row_turnbull_fit():
    """Observed at e3412770: ``Turnbull.fit(X, c=np.full(30, 2))`` raises
    the same ``ValueError`` as ``Weibull.fit``. Expected: the estimate
    with row 3 flagged exact.
    """
    X = _data()
    m = Turnbull.fit(X, c=np.full(30, 2))
    X2, c2 = _row_3_exact(X)
    q = np.linspace(20, 200, 10)
    np.testing.assert_allclose(
        m.sf(q), Turnbull.fit(X2, c=c2).sf(q), rtol=0, atol=1e-10
    )


def test_all_interval_rows_agree_between_paths():
    rng = np.random.default_rng(1)
    x = rng.weibull(2.0, 30) * 100
    X = np.c_[x, x + 1.0]
    a = Weibull.fit(X, c=np.full(30, 2))
    df = pd.DataFrame({"xl": X[:, 0], "xr": X[:, 1]})
    b = Weibull.fit_from_df(df, xl_col="xl", xr_col="xr")
    np.testing.assert_allclose(a.params, b.params, rtol=1e-10)
