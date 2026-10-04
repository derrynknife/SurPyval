"""``fit_from_df`` on every fitter (#511).

The conformance suite (``conformance/test_fit_paths.py``) checks that
every registered model's ``fit_from_df`` gives the model its ``fit`` gives;
these tests check the shared reading of the columns: the names each
family uses, the options passed through, and the errors.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import multivariate as mv
from surpyval import recurrent as rc
from surpyval.univariate.competing_risks import FineGray

LOG = pd.DataFrame(
    {
        "hours": [120.0, 380, 610, 700, 90, 400, 520, 650],
        "truck": [1, 1, 1, 1, 2, 2, 2, 2],
        "c": [0, 0, 0, 1, 0, 0, 0, 1],
    }
)


def test_kaplan_meier_as_a_new_user_calls_it():
    # The issue's call: AttributeError before.
    df = pd.DataFrame({"days": [3.0, 5, 7, 9, 12], "c": [0, 1, 0, 0, 1]})
    km = sp.KaplanMeier.fit_from_df(df, x_col="days", c_col="c")
    ref = sp.KaplanMeier.fit(df["days"], df["c"])
    np.testing.assert_array_equal(km.R, ref.R)


def test_interval_and_truncation_columns():
    df = pd.DataFrame(
        {
            "lo": [1.0, 2.0, 4.0, 5.0, 7.0],
            "hi": [1.0, 3.5, 7.0, 5.0, 12.0],
            "entry": [0.0, 0.5, 0.0, 1.0, 0.0],
        }
    )
    got = sp.Turnbull.fit_from_df(
        df, xl_col="lo", xr_col="hi", tl_col="entry", tr_col=20
    )
    x = np.column_stack([df["lo"], df["hi"]])
    t = np.column_stack([df["entry"], np.full(5, 20.0)])
    ref = sp.Turnbull.fit(x, t=t)
    np.testing.assert_allclose(got.R, ref.R)


def test_crow_amsaa_event_log():
    # The issue's recurrent example, with the recurrent names (x_col,
    # i_col, c_col) of CauseSpecificNHPP.fit_from_df.
    got = rc.CrowAMSAA.fit_from_df(
        LOG, x_col="hours", i_col="truck", c_col="c"
    )
    ref = rc.CrowAMSAA.fit(LOG["hours"], LOG["truck"], LOG["c"])
    np.testing.assert_allclose(got.params, ref.params)


def test_options_pass_through_to_fit():
    got = rc.GeneralizedRenewal.fit_from_df(
        LOG, x_col="hours", i_col="truck", c_col="c", kijima="ii"
    )
    ref = rc.GeneralizedRenewal.fit(
        LOG["hours"], LOG["truck"], LOG["c"], kijima="ii"
    )
    np.testing.assert_allclose(got.params, ref.params)
    mix = sp.MixtureModel.fit_from_df(
        pd.DataFrame(
            {"x": np.r_[np.linspace(2, 6, 9), np.linspace(20, 40, 9)]}
        ),
        x_col="x",
        dist=sp.Weibull,
        m=2,
    )
    assert mix.m == 2


def test_a_column_the_fit_cannot_take_is_refused():
    df = pd.DataFrame({"x": [0, 1, 1, 0], "c": 0})
    # (ARA, the example here before, takes `tl` since #615)
    with pytest.raises(ValueError, match="Bernoulli.fit takes no `t`"):
        sp.Bernoulli.fit_from_df(df, x_col="x", tl_col="c")
    with pytest.raises(ValueError, match="Bernoulli.fit takes no `c`"):
        sp.Bernoulli.fit_from_df(df, x_col="x", c_col="c")


def test_an_unknown_column_names_the_argument():
    with pytest.raises(ValueError, match="i_col='unit' is not a column"):
        rc.HPP.fit_from_df(LOG, x_col="hours", i_col="unit")
    with pytest.raises(ValueError, match="x_col='days' is not a column"):
        sp.NelsonAalen.fit_from_df(LOG, x_col="days")


def test_not_a_data_frame():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        sp.KaplanMeier.fit_from_df({"x": [1, 2]}, x_col="x")


def test_durations_are_refused():
    df = pd.DataFrame({"x": pd.to_timedelta([1, 2, 3], unit="D")})
    with pytest.raises(ValueError, match="durations"):
        sp.KaplanMeier.fit_from_df(df, x_col="x")
    log = LOG.assign(hours=pd.to_timedelta(LOG["hours"], unit="h"))
    with pytest.raises(ValueError, match="durations"):
        rc.CrowAMSAA.fit_from_df(log, x_col="hours", i_col="truck")


def test_missing_censoring_flag_is_fits_error_without_a_raw_warning():
    # Weibull.fit_from_df cast the flags to int, so a missing flag became
    # -9223372036854775808 with a raw numpy RuntimeWarning (principle 22)
    # before fit refused it; now it reaches fit as it is, which names it
    # as missing (#576).
    df = pd.DataFrame({"x": np.arange(1.0, 11.0), "c": 0.0})
    df.loc[3, "c"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="'c' cannot contain NaN"):
            sp.Weibull.fit_from_df(df, x_col="x", c_col="c")


def test_fine_gray_blank_cause_is_censored():
    rng = np.random.default_rng(0)
    z = rng.binomial(1, 0.5, 120).astype(float)
    t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * z)))
    t_b = rng.exponential(1 / 0.05, 120)
    t_c = rng.uniform(0, 20, 120)
    x = np.minimum(np.minimum(t_a, t_b), t_c)
    cause = np.where(t_a < t_b, "a", "b").astype(object)
    cause[t_c < np.minimum(t_a, t_b)] = np.nan
    df = pd.DataFrame({"time": x, "cause": cause, "z": z})
    got = FineGray.fit_from_df(
        df, x_col="time", e_col="cause", Z_cols="z", event="a"
    )
    e = np.where(pd.isna(cause), None, cause)
    ref = FineGray.fit(x, z[:, None], e, event="a")
    np.testing.assert_allclose(got.beta, ref.beta)


def test_copula_reads_a_column_per_dimension():
    margins = [
        sp.Weibull.from_params([10, 2]),
        sp.Weibull.from_params([20, 3]),
    ]
    X = mv.Clayton.from_params([2.0], margins).random(100, random_state=0)
    df = pd.DataFrame(X, columns=["pump", "motor"])
    df["c_motor"] = (df["motor"] > 25).astype(int)
    df["c_pump"] = 0
    got = mv.Clayton.fit_from_df(
        df,
        x_cols=["pump", "motor"],
        c_cols=["c_pump", "c_motor"],
        margins=[sp.Weibull, sp.Weibull],
    )
    ref = mv.Clayton.fit(
        X,
        c=df[["c_pump", "c_motor"]].to_numpy(),
        margins=[sp.Weibull, sp.Weibull],
    )
    np.testing.assert_allclose(got.params, ref.params)


@pytest.mark.parametrize("old", ["x", "c", "n", "xl", "xr", "tl", "tr"])
def test_univariate_v021_names_are_gone(old):
    # Principle 21: every DataFrame entry point names its columns with a
    # ``_col`` suffix. The v0.21 names of Weibull.fit_from_df (x, c, n,
    # xl, xr, tl, tr), deprecated in v0.22, are removed in v0.23: each
    # is refused as an unknown argument, naming the new one, rather than
    # reaching ``fit`` (where ``x``, ``c``, ``n`` and ``tl`` are options).
    df = pd.DataFrame({"t": [3.0, 5, 7, 9], "cens": [0, 1, 0, 0]})
    with pytest.raises(
        TypeError,
        match=f"unexpected keyword argument '{old}'; name the column "
        f"with '{old}_col'",
    ):
        sp.Weibull.fit_from_df(df, x_col="t", **{old: "cens"})


def test_a_constant_truncation_is_a_number():
    # tl_col / tr_col take a number as well as a column: every row is
    # truncated at that value.
    df = pd.DataFrame({"t": [3.0, 5, 7, 9, 12, 15]})
    got = sp.Weibull.fit_from_df(df, x_col="t", tl_col=2.0)
    ref = sp.Weibull.fit(df["t"], tl=2.0)
    np.testing.assert_allclose(got.params, ref.params)


def _degradation_frame():
    rng = np.random.default_rng(3)
    units = np.repeat(np.arange(6), 5)
    t = np.tile(np.arange(1.0, 6.0), 6)
    rate = rng.uniform(0.8, 1.2, 6)[units]
    y = rate * t + rng.gamma(0.5, 0.2, t.size)
    return pd.DataFrame({"hours": t, "wear": y, "unit": units})


@pytest.mark.parametrize(
    "name, options",
    [
        ("DegradationAnalysis", dict(threshold=8.0, path="linear")),
        ("WienerProcess", dict(threshold=8.0)),
        ("GammaProcess", dict(threshold=8.0)),
    ],
)
def test_degradation_v021_names_are_gone(name, options):
    # x=, y=, i=, deprecated in v0.22, are removed in v0.23.
    from surpyval import degradation as dg

    fitter = getattr(dg, name)
    df = _degradation_frame()
    new = fitter.fit_from_df(
        df, x_col="hours", y_col="wear", i_col="unit", **options
    )
    assert np.isfinite(new.sf(4.0))
    for old in ("x", "y", "i"):
        with pytest.raises(
            TypeError, match=f"unexpected keyword argument '{old}'"
        ):
            fitter.fit_from_df(df, **{old: "hours"}, **options)


def test_destructive_degradation_and_copula_use_col_names():
    # New since v0.21, so their column names simply switched (no
    # deprecation): x_col / y_col / c_col, and x_cols / c_cols (a column
    # per dimension) for the copulas.
    from surpyval import degradation as dg

    rng = np.random.default_rng(1)
    age = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    df = pd.DataFrame(
        {
            "age": age,
            "strength": np.exp(4.0 - 0.02 * age + rng.normal(0, 0.1, 24)),
        }
    )
    got = dg.DestructiveDegradation.fit_from_df(
        df, x_col="age", y_col="strength", threshold=20
    )
    ref = dg.DestructiveDegradation.fit(age, df["strength"], threshold=20)
    np.testing.assert_allclose(got.sf([50.0]), ref.sf([50.0]))
    with pytest.raises(TypeError, match="x_cols"):
        mv.Clayton.fit_from_df(df, x=["age", "strength"])
