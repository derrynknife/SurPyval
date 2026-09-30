"""Durations and dates are refused wherever times are accepted (#480).

A ``timedelta64`` array converted to floats silently, in its storage
ticks: six durations in days were fitted in seconds (``timedelta64[s]``,
scale 5.0e5) or, multiplied by 1.37 first, in nanoseconds
(``timedelta64[ns]``, scale 6.9e14), whichever pandas happened to pick.
SurPyval has no unit of time, so it refuses them with a ``ValueError``
that names the argument and says how to convert, the same at every entry
point: the xcnt handler (and so ``SurpyvalData`` and every fitter), the
recurrent handler, ``fit_from_df``, and the model functions.
"""

import datetime

import numpy as np
import pandas as pd
import pytest

import surpyval as sp

DAYS = np.array([1.0, 2, 3, 5, 8, 13])
TD_S = pd.to_timedelta(DAYS, unit="D")  # timedelta64[s]
TD_NS = pd.to_timedelta(DAYS * 1.37, unit="D")  # timedelta64[ns]
DURATIONS = "holds durations"


@pytest.mark.parametrize(
    "x",
    [
        TD_S,
        TD_NS,
        pd.Series(TD_S),
        np.asarray(TD_S),
        list(TD_S),
        [datetime.timedelta(days=d) for d in DAYS],
        [[pd.Timedelta(days=1), pd.Timedelta(days=2)], 3.0, 4.0],
    ],
    ids=["index_s", "index_ns", "series", "numpy", "list", "python", "pairs"],
)
def test_a_fit_to_durations_raises_and_says_how_to_convert(x):
    # Before: Weibull.fit(TD_S).params was [5.03e5, 1.328], in seconds
    with pytest.raises(ValueError, match=DURATIONS) as info:
        sp.Weibull.fit(x)
    assert "pd.Timedelta(days=1)" in str(info.value)
    assert str(info.value).startswith("'x'")


def test_the_conversion_the_message_suggests_gives_days():
    model = sp.Weibull.fit(TD_S / pd.Timedelta(days=1))
    np.testing.assert_allclose(
        model.params, sp.Weibull.fit(DAYS).params, rtol=1e-12
    )


def test_dates_are_refused():
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-04-01"])
    with pytest.raises(ValueError, match="holds dates or times"):
        sp.Weibull.fit(dates)


@pytest.mark.parametrize(
    "call, name",
    [
        (lambda: sp.SurpyvalData(TD_S), "x"),
        (lambda: sp.KaplanMeier.fit(TD_S), "x"),
        (lambda: sp.Weibull.fit(xl=TD_S, xr=TD_S * 2), "xl"),
        (lambda: sp.Weibull.fit(DAYS, tl=pd.Timedelta(hours=1)), "tl"),
        (lambda: sp.Weibull.fit(DAYS, tr=TD_S * 2), "tr"),
        (lambda: sp.CoxPH.fit(TD_S, Z=np.arange(6.0)), "x"),
        (lambda: sp.WeibullPH.fit(TD_S, Z=np.arange(6.0)), "x"),
        (lambda: sp.handle_xicn(TD_S, i=[1, 1, 1, 2, 2, 2]), "x"),
        (
            lambda: sp.Weibull.fit_from_df(pd.DataFrame({"x": TD_S}), x="x"),
            "x",
        ),
        (
            lambda: sp.Weibull.fit_from_df(
                pd.DataFrame({"x": DAYS, "tl": TD_S * 0}), x="x", tl="tl"
            ),
            "tl",
        ),
        (
            lambda: sp.WeibullPH.fit_from_df(
                pd.DataFrame({"x": DAYS, "Z": [0, 1.0] * 3, "tl": TD_S * 0}),
                x_col="x",
                Z_cols=["Z"],
                tl_col="tl",
            ),
            "tl",
        ),
    ],
)
def test_every_entry_point_refuses_durations(call, name):
    with pytest.raises(ValueError, match=f"'{name}' {DURATIONS}"):
        call()


@pytest.mark.parametrize("method", ["sf", "ff", "df", "hf", "Hf"])
@pytest.mark.parametrize(
    "x", [TD_S, pd.Timedelta(days=1), np.timedelta64(1, "D")]
)
def test_a_model_function_refuses_durations(method, x):
    # Before: a TypeError from numpy ("ufunc 'subtract' cannot use
    # operands ...") or from pandas.
    model = sp.Weibull.fit(DAYS)
    with pytest.raises(ValueError, match=DURATIONS):
        getattr(model, method)(x)
    km = sp.KaplanMeier.fit(DAYS)
    with pytest.raises(ValueError, match=DURATIONS):
        getattr(km, method)(x)


def test_numbers_are_untouched():
    model = sp.Weibull.fit(pd.Series(DAYS))
    np.testing.assert_allclose(model.params, sp.Weibull.fit(DAYS).params)
    assert model.sf(pd.Series([1.0, 2.0])).shape == (2,)
