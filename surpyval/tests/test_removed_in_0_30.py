"""What v0.29 deprecated is gone in v0.30, and each old name says what
replaced it (the removal hints of ``surpyval.utils.removed_names``).

- the 39 result, model, base-fitter and plot classes taken off the top
  level (``surpyval._MOVED``), which live in their subpackages;
- ``fit_from_surpyval_data`` and ``fit_from_recurrent_data``, now
  ``fit(data)``;
- the renewal fitters' ``fit_from_parameters``, now ``from_params``;
- ``concordance_estimate`` (top level, ``surpyval.metrics`` and the
  regression models), now ``concordance_index`` and ``concordance``;
- the internal steps of ``surpyval.utils`` and
  ``surpyval.univariate.nonparametric``;
- the 21 one-design test-planning functions, now the design functions;
- ``EquivalenceFitter.fit``'s ``x1`` and its ``tl`` as a pair per unit,
  and ``fit_from_df``'s ``x_cols`` and ``tl_cols``.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import recurrent as rc
from surpyval import test_planning
from surpyval.univariate import nonparametric
from surpyval.utils import removed_names

MOVED = sorted(sp._MOVED.items())


def test_thirty_nine_names_left_the_top_level():
    assert len(MOVED) == 39


@pytest.mark.parametrize("name, home", MOVED, ids=[n for n, _ in MOVED])
def test_a_moved_class_is_in_its_subpackage_only(name, home):
    assert hasattr(importlib.import_module(home), name)
    assert name not in sp.__all__
    assert name not in dir(sp)
    with pytest.raises(AttributeError) as err:
        getattr(sp, name)
    message = str(err.value)
    assert "removed in v0.30" in message
    assert f"from {home} import {name}" in message


def test_from_import_of_a_moved_class_names_its_subpackage():
    with pytest.raises(ImportError, match="surpyval.univariate.regression"):
        from surpyval import AFTFitter  # noqa: F401


def test_what_a_user_constructs_stays():
    for name in (
        "ProportionalHazardsFitter",
        "MixtureModel",
        "EquivalenceFitter",
        "EquivalenceModel",
    ):
        assert name in sp.__all__
        getattr(sp, name)


@pytest.mark.parametrize(
    "fitter, old",
    [
        (sp.Weibull, "fit_from_surpyval_data"),
        (sp.Hypoexponential, "fit_from_surpyval_data"),
        (rc.HPP, "fit_from_recurrent_data"),
        (rc.CrowAMSAA, "fit_from_recurrent_data"),
        (rc.NonParametricCounting, "fit_from_recurrent_data"),
        (rc.GeneralizedRenewal, "fit_from_recurrent_data"),
        (rc.ProportionalIntensityNHPP, "fit_from_recurrent_data"),
        (rc.CauseSpecificNHPP, "fit_from_recurrent_data"),
    ],
)
def test_the_old_data_entry_points_say_call_fit(fitter, old):
    assert not hasattr(fitter, old)
    with pytest.raises(AttributeError, match=r"v0\.30; call fit\(data\)"):
        getattr(fitter, old)
    # the private step fit(data) runs is still there
    assert callable(getattr(fitter, "_" + old))


@pytest.mark.parametrize(
    "fitter",
    [rc.GeneralizedRenewal, rc.GeneralizedOneRenewal, rc.ARA, rc.ARI],
)
def test_fit_from_parameters_says_from_params(fitter):
    with pytest.raises(AttributeError, match="v0.30; use 'from_params'"):
        fitter.fit_from_parameters


def test_concordance_estimate_is_gone_everywhere():
    for module in (sp, sp.metrics):
        assert not hasattr(module, "concordance_estimate")
        with pytest.raises(AttributeError, match="concordance_index"):
            module.concordance_estimate
    with pytest.raises(ImportError, match="removed in v0.30"):
        from surpyval.metrics import concordance_estimate  # noqa: F401
    x = np.array([5.0, 8.0, 12.0, 3.0, 9.0, 15.0, 7.0, 11.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 0])
    Z = np.array([[1.0], [0.0], [0.0], [1.0], [1.0], [0.0], [1.0], [0.0]])
    model = sp.CoxPH.fit(x, Z, c)
    with pytest.raises(AttributeError, match="use 'concordance'"):
        model.concordance_estimate


@pytest.mark.parametrize(
    "module, names",
    [
        (
            sp.utils,
            (
                "coerce_xcnt_x",
                "format_truncation",
                "group_xcnt",
                "xcnt_sort",
                "resolve_cr_censoring",
            ),
        ),
        (
            nonparametric,
            (
                "kaplan_meier",
                "nelson_aalen",
                "fleming_harrington",
                "turnbull",
                "greenwood_variance",
                "nelson_aalen_variance",
                "fleming_harrington_variance",
                "rank_adjust",
                "canonical_heuristic",
                "FIT_FUNCS",
                "VAR_FUNCS",
                "PLOTTING_METHODS",
            ),
        ),
        (test_planning, tuple(test_planning._REPLACED)),
    ],
    ids=["utils", "nonparametric", "test_planning"],
)
def test_the_internal_steps_and_one_design_functions_are_gone(module, names):
    for name in names:
        assert not hasattr(module, name), name
        with pytest.raises(AttributeError, match="removed in v0.30; use"):
            getattr(module, name)


def test_twenty_one_test_planning_functions_went():
    assert len(test_planning._REPLACED) == 21
    with pytest.raises(AttributeError, match=r"logrank_design\("):
        test_planning.logrank_events


def test_from_import_of_a_removed_subpackage_name():
    with pytest.raises(ImportError, match="removed in v0.30"):
        from surpyval.test_planning import logrank_events  # noqa: F401


def test_raise_removed_is_an_attribute_error_outside_an_import():
    with pytest.raises(AttributeError, match="v0.30; use X"):
        removed_names.raise_removed("m", "n", ("use X", "0.30"))


def _two_clocks():
    rng = np.random.default_rng(3)
    x = 100 * rng.weibull(2.0, 40)
    return x, x * rng.uniform(0.5, 2.0, 40)


def test_equivalence_x1_says_use_x():
    x, x2 = _two_clocks()
    with pytest.raises(TypeError, match="v0.30; use 'x'"):
        sp.WeibullEquivalence.fit(x1=x, x2=x2)


def test_equivalence_pair_entry_is_refused_with_the_fix():
    x, x2 = _two_clocks()
    pair = np.column_stack([np.zeros(40), np.zeros(40)])
    with pytest.raises(ValueError, match=r"v0\.30; .*SecondaryClock"):
        sp.WeibullEquivalence.fit(x, x2, tl=pair)


def test_equivalence_fit_from_df_column_lists_are_refused():
    x, x2 = _two_clocks()
    df = pd.DataFrame({"hours": x, "starts": x2})
    with pytest.raises(TypeError, match="v0.30; use x_col and x2_col"):
        sp.WeibullEquivalence.fit_from_df(df, x_cols=["hours", "starts"])
    with pytest.raises(ValueError, match="v0.30; give them one each"):
        sp.WeibullEquivalence.fit_from_df(df, ["hours", "starts"])
    model = sp.WeibullEquivalence.fit_from_df(
        df, x_col="hours", x2_col="starts"
    )
    assert model.clocks == ("hours", "starts")
