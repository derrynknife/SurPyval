"""Polars and Arrow tables are taken wherever a pandas DataFrame is.

The conversion is one function, ``surpyval.utils.dataframe.as_pandas``,
called where each DataFrame entry point reads its frame; the registry
sweep in ``conformance/test_fit_paths.py`` checks that every entry point
calls it. These tests use the real libraries where they are installed
(and skip where not), and check that neither is imported unless the
object is theirs.
"""

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.utils.dataframe import as_pandas

DATA = {
    "x": [2.0, 3.5, 4.1, 5.0, 6.2, 7.7, 9.1, 11.0],
    "c": [0, 0, 1, 0, 0, 1, 0, 0],
    "n": [1, 2, 1, 1, 1, 1, 2, 1],
    "z": [0.1, 0.5, 0.2, 0.9, 0.4, 0.3, 0.8, 0.6],
}


def _frames():
    out = [pytest.param(lambda d: pd.DataFrame(d), id="pandas")]

    def polars(d):
        pl = pytest.importorskip("polars")
        pytest.importorskip("pyarrow")  # Polars' own to_pandas needs it
        return pl.DataFrame(d)

    def lazy(d):
        pl = pytest.importorskip("polars")
        pytest.importorskip("pyarrow")
        return pl.DataFrame(d).lazy()

    def arrow(d):
        pa = pytest.importorskip("pyarrow")
        return pa.table(d)

    def batch(d):
        pa = pytest.importorskip("pyarrow")
        return pa.RecordBatch.from_pandas(pd.DataFrame(d))

    return out + [
        pytest.param(polars, id="polars"),
        pytest.param(lazy, id="polars-lazy"),
        pytest.param(arrow, id="arrow"),
        pytest.param(batch, id="arrow-batch"),
    ]


@pytest.mark.parametrize("make", _frames())
def test_fit_from_df_is_the_same_fit(make):
    frame = make(DATA)
    pandas_frame = pd.DataFrame(DATA)
    got = sp.Weibull.fit_from_df(frame, x_col="x", c_col="c", n_col="n")
    want = sp.Weibull.fit_from_df(
        pandas_frame, x_col="x", c_col="c", n_col="n"
    )
    np.testing.assert_array_equal(got.params, want.params)
    got = sp.CoxPH.fit_from_df(frame, x_col="x", c_col="c", Z_cols=["z"])
    want = sp.CoxPH.fit_from_df(
        pandas_frame, x_col="x", c_col="c", Z_cols=["z"]
    )
    np.testing.assert_array_equal(got.params, want.params)
    km = sp.KaplanMeier.fit_from_df(frame, x_col="x", c_col="c")
    np.testing.assert_array_equal(
        km.R, sp.KaplanMeier.fit_from_df(pandas_frame, "x", "c").R
    )


@pytest.mark.parametrize("make", _frames())
def test_surpyval_data_from_dataframe(make):
    got = sp.SurpyvalData.from_dataframe(
        make(DATA), x_col="x", c_col="c", n_col="n"
    )
    want = sp.SurpyvalData.from_dataframe(
        pd.DataFrame(DATA), x_col="x", c_col="c", n_col="n"
    )
    np.testing.assert_array_equal(got.x, want.x)
    np.testing.assert_array_equal(got.c, want.c)
    np.testing.assert_array_equal(got.n, want.n)


@pytest.mark.parametrize("make", _frames())
def test_a_null_is_a_missing_value(make):
    data = {**DATA, "x": [None] + DATA["x"][1:]}
    frame = make(data)
    pandas_frame = pd.DataFrame(data)
    with pytest.raises(ValueError) as polars_error:
        sp.Weibull.fit_from_df(frame, x_col="x", c_col="c")
    with pytest.raises(ValueError) as pandas_error:
        sp.Weibull.fit_from_df(pandas_frame, x_col="x", c_col="c")
    assert str(polars_error.value) == str(pandas_error.value)


def test_a_frame_of_another_kind_is_refused_naming_the_options():
    with pytest.raises(ValueError, match="Polars DataFrame or a PyArrow"):
        sp.Weibull.fit_from_df({"x": [1.0, 2.0]}, x_col="x")


def test_pandas_is_not_touched():
    frame = pd.DataFrame(DATA)
    assert as_pandas(frame) is frame


def test_fitting_from_pandas_imports_neither_library():
    # In a fresh interpreter: whatever importing pandas itself brings in
    # (pandas loads parts of pyarrow where it is installed) is the
    # baseline, and importing surpyval and fitting from a pandas frame
    # adds nothing.
    code = textwrap.dedent("""
        import sys
        import pandas as pd

        def loaded():
            return {
                m
                for m in sys.modules
                if m.split(".")[0] in ("polars", "pyarrow")
            }

        # pandas loads parts of pyarrow itself when it builds a frame
        x = [1.0, 2.0, 3.5, 4.0, 6.0]
        df = pd.DataFrame({"x": x, "c": [0, 0, 1, 0, 0]})
        df = df.assign(z=[0.1, 0.4, 0.2, 0.9, 0.3])
        before = loaded()
        import surpyval as sp

        sp.Weibull.fit_from_df(df, x_col="x", c_col="c")
        sp.CoxPH.fit_from_df(df, x_col="x", c_col="c", Z_cols=["z"])
        sp.SurpyvalData.from_dataframe(df, x_col="x", c_col="c")
        extra = loaded() - before
        assert not extra, sorted(extra)[:5]
        print("ok")
        """)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.strip() == "ok"
