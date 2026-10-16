"""The shared ``diagnose()`` machinery (#867): the report, the verdicts,
the advice engine, the registry and the refusals of the families not built
yet. The Cox family is the one that exists, so the report is exercised on
Cox fits; ``test_cox.py`` covers the family's own checks.
"""

import importlib
import inspect
import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.datasets import load_rossi_static
from surpyval.diagnostics import Advice, Diagnosis, DiagnosticsFailed, registry
from surpyval.diagnostics.mixin import DiagnoseMixin
from surpyval.diagnostics.report import COLUMNS, verdict


@pytest.fixture(scope="module")
def rossi():
    df = load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    return df


@pytest.fixture(scope="module")
def failing(rossi):
    # age fails proportional hazards on the km scale (p = 0.012)
    return sp.CoxPH.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=["fin", "age", "prio"]
    )


@pytest.fixture(scope="module")
def passing(rossi):
    return sp.CoxPH.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=["fin", "prio"]
    )


# -- the verdict rule ---------------------------------------------------------
@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, "fail"),
        (0.049, "fail"),
        (0.05, "warn"),
        (0.099, "warn"),
        (0.1, "ok"),
        (1.0, "ok"),
        (float("nan"), "warn"),
        (None, "warn"),
    ],
)
def test_verdict_bands(p, expected):
    assert verdict(p, 0.05) == expected


# -- the report ---------------------------------------------------------------
def test_table_columns_and_types(failing):
    d = failing.diagnose()
    assert list(d.table.columns) == [
        "check",
        "transform",
        "test",
        "covariates",
        "statistic",
        "df",
        "p",
        "verdict",
        "detail",
    ]
    assert set(COLUMNS) <= set(d.table.columns)
    assert all(isinstance(c, tuple) for c in d.table["covariates"])
    assert str(d.table["df"].dtype) == "Int64"
    assert set(d.table["verdict"]) <= {"ok", "warn", "fail"}
    assert d.family == "Cox proportional hazards"
    assert d.model == "CoxPH"
    assert d.columns == ["fin", "age", "prio"]


def test_ok_and_raise_if_failed(failing, passing):
    good = passing.diagnose()
    assert good.ok
    assert good.raise_if_failed() is good
    bad = failing.diagnose()
    assert not bad.ok
    with pytest.raises(DiagnosticsFailed) as info:
        bad.raise_if_failed()
    message = str(info.value)
    assert "proportional_hazards" in message and "age" in message
    assert "Advice:" in message
    assert info.value.diagnosis is bad


def test_a_warning_does_not_fail(passing):
    out = passing.diagnose().to_dict()
    out["table"][0]["verdict"] = "warn"
    d = Diagnosis.from_dict(out)
    assert d.ok and d.raise_if_failed() is d
    out["table"][0]["verdict"] = "fail"
    assert not Diagnosis.from_dict(out).ok


def test_to_dict_is_strict_json_and_round_trips(failing):
    d = failing.diagnose()
    out = d.to_dict()
    text = json.dumps(out, allow_nan=False)
    back = Diagnosis.from_dict(json.loads(text))
    assert back.to_dict() == out
    pd.testing.assert_frame_equal(back.table, d.table)
    assert back.advice == d.advice
    assert [a.features for a in back.advice] == [a.features for a in d.advice]
    assert back.ok == d.ok and back.notes == d.notes
    assert out["ok"] is False and out["alpha"] == 0.05
    row = out["table"][-1]
    assert row["p"] is None and row["df"] is None  # a screen
    with pytest.raises(ValueError, match="from_dict"):
        back.plot()
    with pytest.raises(ValueError, match="to_dict"):
        Diagnosis.from_dict([1, 2])
    with pytest.raises(ValueError, match="newer"):
        Diagnosis.from_dict(dict(out, schema=99))


def test_advice_is_a_string_with_its_features():
    a = Advice("Do this.", "proportional_hazards", ["x"], ["CoxPH.fit"])
    assert a == "Do this." and isinstance(a, str)
    assert a.covariates == ("x",) and a.features == ("CoxPH.fit",)
    import pickle

    b = pickle.loads(pickle.dumps(a))
    assert b == a and b.features == a.features


def test_summary_prints_table_advice_and_notes(failing):
    text = repr(failing.diagnose())
    assert text.startswith("Cox proportional hazards diagnostics at alpha")
    assert "Advice:" in text and "Notes:" in text
    assert "fin, age, prio" in text  # a tuple is printed as its names


# -- columns= and alpha -------------------------------------------------------
def test_columns_checks_a_subset(failing):
    d = failing.diagnose(columns=["age"])
    single = [c for c in d.table["covariates"] if len(c) == 1]
    assert set(single) == {("age",)}
    # The joint test still covers every covariate
    glob = d.table[d.table["test"].str.contains("GLOBAL")]
    assert set(glob["covariates"]) == {("fin", "age", "prio")}
    assert d.columns == ["age"]
    assert failing.diagnose(columns="age").to_dict() == d.to_dict()
    full = failing.diagnose()
    both = full.table[full.table["covariates"] == ("age",)]
    pd.testing.assert_frame_equal(
        d.table[d.table["covariates"] == ("age",)].reset_index(drop=True),
        both.reset_index(drop=True),
    )


@pytest.mark.parametrize("bad", [["nope"], [0], [], "Age"])
def test_columns_must_name_covariates(failing, bad):
    with pytest.raises(ValueError, match="'columns'"):
        failing.diagnose(columns=bad)


def test_alpha_sets_the_threshold(failing):
    strict = failing.diagnose(alpha=0.01)
    loose = failing.diagnose(alpha=0.05)
    p = loose.table["p"]
    assert ((loose.table["verdict"] == "fail") == (p < 0.05)).all()
    tests = strict.table[strict.table["p"].notna()]
    assert ((tests["verdict"] == "fail") == (tests["p"] < 0.01)).all()
    assert strict.alpha == 0.01
    # age's km p-value (0.012) fails at 0.05, warns at 0.01
    age_km = (strict.table["covariates"] == ("age",)) & (
        strict.table["transform"] == "km"
    )
    assert strict.table.loc[age_km, "verdict"].item() == "warn"
    assert loose.table.loc[age_km, "verdict"].item() == "fail"


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, "0.05x", None, np.nan])
def test_alpha_must_be_a_probability(failing, bad):
    with pytest.raises(ValueError, match="'alpha'"):
        failing.diagnose(alpha=bad)


# -- plots --------------------------------------------------------------------
def test_plot_returns_a_figure(failing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    d = failing.diagnose()
    fig = d.plot(n_boot=10, random_state=1)
    assert isinstance(fig, Figure)
    # 3 PH, 2 functional form (fin is binary), 3 influence, outliers,
    # overall fit
    assert len(fig.axes) == 3 + 2 + 3 + 1 + 1
    plt.close(fig)
    fig = d.plot("overall_fit")
    assert len(fig.axes) == 1
    plt.close(fig)
    fig = d.plot(["proportional_hazards", "outliers"], n_boot=0)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_is_reproducible_with_a_seed(failing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = failing.diagnose()

    def band():
        fig = d.plot("proportional_hazards", n_boot=15, random_state=7)
        poly = fig.axes[0].collections[-1]
        out = np.array(poly.get_paths()[0].vertices)
        plt.close(fig)
        return out

    np.testing.assert_array_equal(band(), band())


@pytest.mark.parametrize(
    "which, n_boot",
    [("nope", 10), (["influence", "nope"], 10), ([], 10), (None, -1)],
)
def test_plot_refuses_bad_arguments(failing, which, n_boot):
    import matplotlib

    matplotlib.use("Agg")
    with pytest.raises(ValueError):
        failing.diagnose().plot(which, n_boot=n_boot)


def test_diagnose_does_not_import_matplotlib():
    code = textwrap.dedent("""
        import sys
        import surpyval as sp
        from surpyval.datasets import load_rossi_static
        df = load_rossi_static()
        m = sp.CoxPH.fit(df["week"].values, df[["fin", "age"]].values,
                         c=1 - df["arrest"].values)
        d = m.diagnose()
        d.to_dict(); repr(d)
        print(any(k == "matplotlib" or k.startswith("matplotlib.")
                  for k in sys.modules))
        """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(importlib.import_module("surpyval").__path__[0] + "/.."),
    )
    assert out.stdout.strip() == "False"


# -- the registry -------------------------------------------------------------
def _feature_exists(spec):
    """Whether an advised feature, spelt as AdviceRule.features spells it,
    exists: ``Name.method(arg=)`` on ``surpyval``, or ``method(arg=)`` on a
    fitted model class that has ``diagnose``."""
    name, _, arg = spec.partition("(")
    arg = arg.rstrip(")").rstrip("=")
    candidates = []
    obj = sp
    try:
        for part in name.split("."):
            obj = getattr(obj, part)
        candidates.append(obj)
    except AttributeError:
        pass

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    if "." not in name:
        candidates.extend(
            getattr(cls, name)
            for cls in subclasses(DiagnoseMixin)
            if hasattr(cls, name)
        )
    for obj in candidates:
        if not arg:
            return True
        if arg in inspect.signature(obj).parameters:
            return True
    return False


def test_every_advised_feature_exists():
    features = {
        f
        for family in registry.families()
        for r in family.advice
        for f in r.features
    }
    assert features
    missing = sorted(f for f in features if not _feature_exists(f))
    assert not missing, missing


def test_feature_check_catches_a_missing_one():
    assert not _feature_exists("CoxPH.fit(nope=)")
    assert not _feature_exists("CoxPH.no_such_method")
    assert _feature_exists("CoxPH.fit(strata=)")
    assert _feature_exists("summary(robust=)")


def _bare(path, **attributes):
    module, _, name = path.rpartition(".")
    cls = getattr(importlib.import_module(module), name)
    obj = cls.__new__(cls)
    for key, value in attributes.items():
        object.__setattr__(obj, key, value)
    return obj


REG = "surpyval.univariate.regression."


@pytest.mark.parametrize(
    "path, attributes, issue",
    [
        (REG + "FrailtyModel", {}, 872),
        (REG + "CoxFrailtyModel", {}, 872),
        (
            "surpyval.univariate.competing_risks."
            "CompetingRisksProportionalHazards",
            {},
            874,
        ),
        (
            "surpyval.univariate.competing_risks.regression.fine_gray."
            "FineGrayModel",
            {},
            874,
        ),
        (
            "surpyval.univariate.competing_risks.ParametricCompetingRisks",
            {},
            874,
        ),
        ("surpyval.univariate.parametric.Parametric", {}, 875),
        (REG + "BuckleyJamesModel", {}, 867),
    ],
)
def test_families_not_built_yet_refuse_naming_their_issue(
    path, attributes, issue
):
    model = _bare(path, **attributes)
    assert isinstance(model, DiagnoseMixin)
    with pytest.raises(NotImplementedError, match="#{}".format(issue)):
        model.diagnose()


def test_a_real_fit_of_a_pending_family_refuses(rossi):
    x, c = rossi["week"].values, rossi["censored"].values
    with pytest.raises(NotImplementedError, match="#875"):
        sp.Weibull.fit(x, c=c).diagnose()


class _Toy:
    """A stand-in model for a family registered by a test."""

    beta = np.array([0.5])
    feature_names = ["z"]


def test_a_family_plugs_in_by_listing_its_checks():
    # What the developer guide promises: a family is its checks and
    # rules, registered; the report, verdicts, advice, columns= and alpha
    # are shared.
    def rows(ctx):
        return [
            {
                "test": "toy test",
                "covariates": [ctx.names[j]],
                "statistic": 4.0,
                "df": 1,
                "p": 0.01,
                "detail": "toy",
            }
            for j in ctx.columns
        ]

    family = registry.Family(
        name="Toy",
        issue=0,
        applies=lambda model: isinstance(model, _Toy),
        checks=(registry.Check("toy", "Toys are fine.", rows=rows),),
        advice=(
            registry.AdviceRule(
                "toy",
                lambda row, ctx: "Fix {}.".format(row["covariates"][0]),
                ("CoxPH.fit",),
            ),
        ),
    )
    registry.families()  # load the built-in families first
    registry.register(family)
    try:
        d = registry.diagnose(_Toy())
        assert d.family == "Toy" and not d.ok
        assert d.table["verdict"].tolist() == ["fail"]
        assert d.advice == ["Fix z."]
        assert registry.diagnose(_Toy(), alpha=0.005).ok
        with pytest.raises(ValueError, match="draw"):
            d.plot()
    finally:
        registry._FAMILIES[:] = [
            f for f in registry._FAMILIES if f.name != "Toy"
        ]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "checks": (
                    registry.Check("a", "x", rows=lambda c: []),
                    registry.Check("a", "y", rows=lambda c: []),
                )
            },
            "two checks",
        ),
        (
            {
                "checks": (registry.Check("a", "x", rows=lambda c: []),),
                "advice": (
                    registry.AdviceRule("b", lambda r, c: "", ("CoxPH.fit",)),
                ),
            },
            "does not have",
        ),
        (
            {
                "checks": (registry.Check("a", "x", rows=lambda c: []),),
                "columns": ("verdict",),
            },
            "clash",
        ),
    ],
)
def test_register_refuses_an_inconsistent_family(kwargs, match):
    family = registry.Family(
        name="Broken", issue=0, applies=lambda m: False, **kwargs
    )
    with pytest.raises(ValueError, match=match):
        registry.register(family)
