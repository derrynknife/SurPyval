"""A zero-width interval row through ``fit`` and ``fit_from_df``.

A row of a two-column ``x`` with ``xl == xr`` is an exact time:
``fit_from_df`` says so ("If xl[i] == xr[i] assumes observed",
``surpyval/utils/dataframe.py``), and the array path accepts such a row
with ``c = 0``. With ``c = 2`` the array path refused it (at e3412770,
in ``surpyval/utils/data_formats.py::_check_interval_flags``, with a
message about "only one failure time"); it now reads it as exact too,
so both paths agree.
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


def test_zero_width_interval_row_weibull_fit():
    """``Weibull.fit(X, c=np.full(30, 2))`` with a zero-width row takes
    it as exact, as ``fit_from_df`` does: the fit with row 3 flagged
    ``c = 0``. It raised "only has one failure time" at e3412770."""
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


def test_zero_width_interval_row_turnbull_fit():
    """``Turnbull.fit(X, c=np.full(30, 2))``: the estimate with row 3
    flagged exact (it raised as ``Weibull.fit`` did at e3412770)."""
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


def test_zero_width_interval_row_fit_from_df_with_flags():
    # A c column of 2 on every row, the zero-width one included: exact,
    # as without the column.
    X = _data()
    df = pd.DataFrame({"xl": X[:, 0], "xr": X[:, 1], "c": 2})
    m = Weibull.fit_from_df(df, xl_col="xl", xr_col="xr", c_col="c")
    X2, c2 = _row_3_exact(X)
    np.testing.assert_allclose(
        m.params, Weibull.fit(X2, c=c2).params, rtol=1e-10
    )


def test_reversed_interval_names_the_row():
    X = _data()
    X[[4, 7]] = X[[4, 7]][:, ::-1]
    with pytest.raises(ValueError, match=r"2 row\(s\) .*\(rows \[4, 7\]"):
        Weibull.fit(X, c=np.full(30, 2))
