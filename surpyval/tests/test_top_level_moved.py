"""Base fitters, fitted models, results and plots left the top level in
v0.29.

A user calls ``WeibullAFT.fit`` or ``logrank``; the classes those return
(``AFTFitter``, ``LogRankResult``, ...) are for type hints and
``isinstance``, and live in their subpackages. ``surpyval.<name>`` still
gives the same object until v0.30, with a ``FutureWarning`` naming where
it lives.
"""

import importlib
import warnings

import pytest

import surpyval as sp
from surpyval.utils.deprecation import REMOVED_IN

MOVED = sorted(sp._MOVED.items())


def test_thirty_nine_names_moved():
    assert len(MOVED) == 39


@pytest.mark.parametrize("name, home", MOVED, ids=[n for n, _ in MOVED])
def test_each_is_in_its_subpackage_and_warns_at_the_top(name, home):
    there = getattr(importlib.import_module(home), name)
    with pytest.warns(FutureWarning) as caught:
        top = getattr(sp, name)
    assert top is there
    message = str(caught[0].message)
    assert f"'{name}' is deprecated" in message
    assert f"v{REMOVED_IN}" in message
    assert f"'{home}.{name}'" in message
    assert name not in sp.__all__
    assert name not in dir(sp)


def test_from_import_warns_and_gives_the_same_class():
    from surpyval.univariate.nonparametric import LogRankResult

    with pytest.warns(FutureWarning, match="surpyval.univariate"):
        from surpyval import LogRankResult as old
    assert old is LogRankResult


def test_isinstance_is_unchanged():
    from surpyval.univariate.regression import AFTFitter

    assert isinstance(sp.WeibullAFT, AFTFitter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(sp.WeibullAFT, sp.AFTFitter)


def test_what_a_user_constructs_stays():
    # A custom proportional hazards model is built from the base fitter,
    # and a mixture is constructed; the two-clock base classes have no
    # subpackage of their own.
    for name in (
        "ProportionalHazardsFitter",
        "MixtureModel",
        "EquivalenceFitter",
        "EquivalenceModel",
    ):
        assert name in sp.__all__
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            getattr(sp, name)


def test_star_import_has_none_of_them():
    namespace: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec("from surpyval import *", namespace)
    assert not set(sp._MOVED) & set(namespace)
