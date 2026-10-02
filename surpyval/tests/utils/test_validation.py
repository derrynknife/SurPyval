"""One message for an unknown option value, everywhere (principles 2 and
21): ``'<name>' must be one of <values>; got <value>`` from
``surpyval.utils.validation.check_option``.

The same refusal used to be worded a dozen ways across the models
("bound must be ...", "`on` must be one of (...)", "Unknown
confidence-bound method ...", "cb 'on' supports ...", "how must be
...", "Method must be in [...]", "Unrecognised baseline method"), some
without the value given or the values accepted.
"""

import re

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted
from surpyval.utils.validation import (
    BOUNDS,
    check_option,
    format_options,
    option_error,
)

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
C = np.array([0, 0, 1, 0, 0, 1, 0, 0])


def test_check_option_message():
    check_option("bound", "lower", BOUNDS)
    with pytest.raises(ValueError) as err:
        check_option("bound", "both", BOUNDS)
    assert str(err.value) == (
        "'bound' must be one of 'two-sided', 'lower' or 'upper'; got 'both'"
    )
    # One accepted value reads "must be", and a note follows the value.
    with pytest.raises(ValueError) as err:
        check_option("dist", "t", ("z",), "Use `bootstrap_cb`.")
    assert str(err.value) == "'dist' must be 'z'; got 't'. Use `bootstrap_cb`."


@pytest.mark.parametrize("value", [None, 1, np.array(["lower"]), ["lower"]])
def test_check_option_refuses_what_is_not_a_string(value):
    # An array is refused, not compared element-wise (which raised "The
    # truth value of an array ... is ambiguous").
    with pytest.raises(ValueError, match="'bound' must be one of"):
        check_option("bound", value, BOUNDS)


def test_format_options():
    assert format_options(("a",)) == "'a'"
    assert format_options(["a", "b"]) == "'a' or 'b'"
    assert format_options({"a": 1, "b": 2, "c": 3}) == "'a', 'b' or 'c'"
    assert str(option_error("k", 2, (None, "x"))) == (
        "'k' must be one of None or 'x'; got 2"
    )


def _weibull():
    return sp.Weibull.fit(X, C)


def _ph():
    return fitted(CASE_BY_NAME["CoxPH"])


# An unknown value of an option, model by model. Each used to give its own
# wording; every one now gives the message of ``check_option``.
def _refusals():
    """The calls, by label. (In a function: mypy checks the lambdas of a
    module-level table, and the singleton fitters look like classes to it.)"""
    return {
        "Parametric.cb bound": (lambda: _weibull().cb([2.0], bound="both")),
        "Parametric.cb on": (lambda: _weibull().cb([2.0], on="xf")),
        "Parametric.cb method": (lambda: _weibull().cb([2.0], method="boot")),
        "Parametric.cb lr bound": (
            lambda: _weibull().cb([2.0], method="lr", bound="both")
        ),
        "Parametric.param_cb bound": (
            lambda: _weibull().param_cb("alpha", bound="both")
        ),
        "Parametric.param_cb method": (
            lambda: _weibull().param_cb("alpha", method="boot")
        ),
        "Parametric.quantile_cb bound": (
            lambda: _weibull().quantile_cb(0.1, bound="both")
        ),
        "Parametric fit how": (lambda: sp.Weibull.fit(X, C, how="MXE")),
        "RoystonParmar.cb on": (
            lambda: fitted(CASE_BY_NAME["RoystonParmar"]).cb([2.0], on="hf")
        ),
        "RoystonParmar.fit scale": (
            lambda: sp.RoystonParmar.fit(X, C, scale="logit")
        ),
        "KaplanMeier.cb bound_type": (
            lambda: sp.KaplanMeier.fit(X, C).cb([2.0], bound_type="log")
        ),
        "KaplanMeier.band method": (
            lambda: sp.KaplanMeier.fit(X, C).band([2.0], method="eqp")
        ),
        "Turnbull turnbull_estimator": (
            lambda: sp.Turnbull.fit(X, C, turnbull_estimator="Greenwood")
        ),
        "ParametricRegression.cb on": (
            lambda: fitted(CASE_BY_NAME["WeibullPH"]).cb(
                [2.0], Z=CASE_BY_NAME["WeibullPH"].Z[0], on="xf"
            )
        ),
        "ParametricRegression.cb bound": (
            lambda: fitted(CASE_BY_NAME["WeibullPH"]).cb(
                [2.0], Z=CASE_BY_NAME["WeibullPH"].Z[0], bound="both"
            )
        ),
        "CoxPH tie_method": (
            lambda: sp.CoxPH.fit(X, Z=np.arange(8.0), c=C, tie_method="fast")
        ),
        "CoxPH residual kind": (lambda: _ph().compute_residuals("bogus")),
        "Frailty param_cb bound": (
            lambda: fitted(CASE_BY_NAME["WeibullFrailty"]).param_cb(
                "theta", bound="both"
            )
        ),
        "CompetingRisksProportionalHazards model": (
            lambda: sp.CompetingRisksProportionalHazards.fit(
                X, np.arange(8.0)[:, None], ["a", "b"] * 4, model="Weibull"
            )
        ),
        "concordance_index ties": (
            lambda: sp.metrics.concordance_index(X, C, X, ties="breslow")
        ),
        "fit_best metric": (lambda: sp.fit_best(X, C, metric="r2")),
        "NHPP how": (
            lambda: sp.recurrent.CrowAMSAA.fit([1, 2, 3, 4], how="MPS")
        ),
        "trend_test test": (
            lambda: sp.recurrent.HPP.fit(
                [1.0, 2.0, 3.0], i=[1, 1, 1]
            ).trend_test(test="nope")
        ),
        "recurrence residuals kind": (
            lambda: sp.recurrent.HPP.fit(
                [1.0, 2.0, 3.0], i=[1, 1, 1]
            ).residuals(kind="deviance")
        ),
    }


@pytest.mark.parametrize("label", sorted(_refusals()))
def test_an_unknown_option_value_has_one_message(label):
    with pytest.raises(ValueError) as err:
        _refusals()[label]()
    message = str(err.value)
    assert re.match(
        r"'\w+' must be (one of )?'[^;]+; got [^;]+(\. .*)?$", message
    ), message


def _frame():
    import pandas as pd

    return pd.DataFrame(
        {"x": X, "c": C, "z": np.arange(8.0), "e": ["a", "b"] * 4}
    )


def _no_covariates():
    df = _frame()
    return {
        "CoxPH": lambda: sp.CoxPH.fit_from_df(df, "x", c_col="c"),
        "BuckleyJames": lambda: sp.BuckleyJames.fit_from_df(
            df, "x", c_col="c"
        ),
        "CompetingRisksProportionalHazards": lambda: (
            sp.CompetingRisksProportionalHazards.fit_from_df(df, "x", "e")
        ),
        "WeibullPH": lambda: sp.WeibullPH.fit_from_df(df, "x", c_col="c"),
    }


@pytest.mark.parametrize("name", sorted(_no_covariates()))
def test_fit_from_df_without_covariates_has_one_message(name):
    # Cox, Buckley-James and the competing-risks PH model built their
    # design matrix with their own copy of ``design_matrix_from_df``, which
    # said "'Z_cols' or 'formula' cannot both be None".
    with pytest.raises(
        ValueError, match="^One of 'Z_cols' or 'formula' must be provided$"
    ):
        _no_covariates()[name]()


def _unknown_causes():
    def model(name):
        return fitted(CASE_BY_NAME[name])

    crph = "CompetingRisksProportionalHazards[Cox]"
    Z = CASE_BY_NAME[crph].Z[0]
    return {
        "ParametricCompetingRisks": lambda: model(
            "ParametricCompetingRisks"
        ).cif([1.0], event="zzz"),
        "CompetingRisksProportionalHazards.cif": lambda: model(crph).cif(
            [1.0], Z, "zzz"
        ),
        "CompetingRisksProportionalHazards.sf": lambda: model(crph).sf(
            [1.0], Z, event="zzz"
        ),
        "CompetingRisks": lambda: model("CompetingRisks[Kaplan-Meier]").sf(
            [1.0], event="zzz"
        ),
        "CauseSpecificNHPP": lambda: model("CauseSpecificNHPP").cif(
            [1.0], "zzz"
        ),
        "CauseSpecificMCF": lambda: model("CauseSpecificMCF").mcf(
            [1.0], "zzz"
        ),
    }


@pytest.mark.parametrize("name", sorted(_unknown_causes()))
def test_an_unknown_cause_has_one_message(name):
    # Worded six ways ("Unrecognised event type for this model", "Event
    # type not in model", "`event` must be one of the fitted causes ...",
    # ...), and a bare KeyError from CauseSpecificMCF.
    with pytest.raises(
        ValueError,
        match=r"^Unknown cause 'zzz'; the causes are \['a', 'b'\]$",
    ):
        _unknown_causes()[name]()


def _bad_alpha_ci():
    return {
        "KaplanMeier.band": lambda: sp.KaplanMeier.fit(X, C).band(
            [2.0], alpha_ci=1.5
        ),
        "Parametric.quantile_cb": lambda: _weibull().quantile_cb(
            0.1, alpha_ci=1.5
        ),
        "trend_test": lambda: sp.recurrent.HPP.fit([1.0, 2.0, 3.0]).trend_test(
            alpha_ci=1.5
        ),
    }


@pytest.mark.parametrize("name", sorted(_bad_alpha_ci()))
def test_an_alpha_ci_outside_zero_one_has_one_message(name):
    with pytest.raises(
        ValueError,
        match=r"^'alpha_ci' must be strictly between 0 and 1; got 1\.5",
    ):
        _bad_alpha_ci()[name]()


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"c": [0, 1]}, "'c'"),
        ({"n": [1, 2]}, "'n'"),
        ({"tl": [0, 0]}, "'tl' and 'tr'"),
    ],
)
def test_a_column_of_the_wrong_length_has_one_message(kwargs, name):
    # "censoring flag array must be same length as variable array" and the
    # like, beside "'c' must be the same length as 'x'".
    with pytest.raises(
        ValueError, match=f"^{name} must be the same length as 'x'$"
    ):
        sp.Weibull.fit([1.0, 2.0, 3.0], **kwargs)
