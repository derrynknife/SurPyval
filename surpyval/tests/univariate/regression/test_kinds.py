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
