"""The public names of the regression families,
``surpyval.univariate.regression.kinds``."""

import importlib

import pytest

import surpyval as surv
from surpyval.univariate.regression import kinds

X = [10.0, 14.0, 21.0, 26.0, 35.0, 41.0, 52.0, 60.0]
Z = [[1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0]]


@pytest.mark.parametrize(
    "fitter, kind",
    [
        ("WeibullPH", kinds.PROPORTIONAL_HAZARD),
        ("WeibullAFT", kinds.ACCELERATED_FAILURE_TIME),
        ("WeibullPO", kinds.PROPORTIONAL_ODDS),
        ("WeibullAH", kinds.ADDITIVE_HAZARD),
    ],
)
def test_a_fitted_models_kind_is_one_of_the_public_names(fitter, kind):
    model = getattr(surv, fitter).fit(X, Z=Z)
    assert model.kind == kind
    assert model.to_dict()["kind"] == kind


def test_the_groups_partition_the_families_they_cover():
    assert kinds.TIME_SCALING == {
        kinds.ACCELERATED_FAILURE_TIME,
        kinds.ACCELERATED_LIFE,
    }
    assert kinds.HAZARD_BASED == {
        kinds.PROPORTIONAL_HAZARD,
        kinds.ADDITIVE_HAZARD,
    }
    assert not kinds.TIME_SCALING & kinds.HAZARD_BASED
    neither = {kinds.PROPORTIONAL_ODDS, kinds.DISTRIBUTIONAL}
    assert not neither & (kinds.TIME_SCALING | kinds.HAZARD_BASED)


def test_every_name_is_listed_and_distinct():
    names = [n for n in kinds.__all__ if isinstance(getattr(kinds, n), str)]
    assert len(names) == 6
    assert len({getattr(kinds, n) for n in names}) == 6


def test_the_former_private_module_still_gives_the_same_names():
    old = importlib.import_module("surpyval.univariate.regression._kinds")
    for name in kinds.__all__:
        assert getattr(old, name) is getattr(kinds, name)


@pytest.mark.parametrize(
    "fitter, log_linear",
    [
        ("WeibullPH", True),
        ("WeibullAFT", True),
        ("WeibullPO", True),
        ("WeibullAH", False),
    ],
)
def test_every_family_shares_the_regression_fitter_base(fitter, log_linear):
    # One base for every parametric regression family: the distribution,
    # the baseline functions it reads and the likelihood; and one fit for
    # the families whose covariates act through exp(beta'Z).
    from surpyval.univariate.regression._regression_fitter import (
        LogLinearRegressionFitter,
        RegressionFitter,
    )

    model = getattr(surv, fitter)
    assert isinstance(model, RegressionFitter)
    assert isinstance(model, LogLinearRegressionFitter) == log_linear
    for name in model._baseline_functions:
        assert getattr(model, name + "_dist") == getattr(surv.Weibull, name)
    # The additive hazards fitter adds its support barrier to the shared
    # likelihood; the others take it as it is.
    shared = type(model).neg_ll is RegressionFitter.neg_ll
    assert shared == log_linear


def test_the_accelerated_life_fitter_shares_the_base():
    from surpyval.life_models import Power
    from surpyval.univariate.regression._regression_fitter import (
        LogLinearRegressionFitter,
        RegressionFitter,
    )

    model = surv.AcceleratedLife(surv.Weibull, Power)
    assert isinstance(model, RegressionFitter)
    assert not isinstance(model, LogLinearRegressionFitter)
    assert model.df_dist == surv.Weibull.df
