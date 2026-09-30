"""
The names v0.21 deprecated are gone in v0.22.0 (#422, principle 21).

An old argument name is now an unknown argument, so the call raises
Python's own ``TypeError``, and ``surpyval.experimental`` no longer
exists (``surpyval.beta.ml`` holds the survival tree and forest). A few
representative old names from each area are checked here.
"""

import importlib
import sys

import numpy as np
import pytest

import surpyval as sp
from surpyval.recurrent import CrowAMSAA, NonParametricCounting
from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted
from surpyval.univariate.competing_risks import CompetingRisks, FineGray

X = np.array([5.0, 10.0, 20.0])

# Recurrent events: three items.
XR = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0]
IR = [1] * 6 + [2] * 5 + [3] * 5
CR = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

# Competing risks: causes "a" and "b", with censoring.
XC = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ZC = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], float)[:, None]
EC = np.array(["a", "b", "a", None, "b", "a", "a", "b", None, "a"])


def _parametric_cb():
    fitted(CASE_BY_NAME["Weibull"]).cb(t=X)


def _cox_method():
    x = np.array([1, 1, 2, 2, 3, 3, 4, 5], float)
    Z = np.array([0, 1, 0, 1, 1, 0, 1, 0], float)[:, None]
    sp.CoxPH.fit(x, Z, method="breslow")


def _recurrent_confidence():
    NonParametricCounting.fit(XR, i=IR, c=CR).mcf_cb(X, confidence=0.9)


def _recurrent_seed():
    CrowAMSAA.fit(XR, i=IR, c=CR).count_terminated_simulation(3, 2, seed=1)


def _degradation_t():
    fitted(CASE_BY_NAME["WienerProcess"]).sf(t=X)


def _competing_risks_method():
    CompetingRisks.fit(XC, EC, method="Kaplan-Meier")


def _fine_gray_cause():
    FineGray.fit(XC, ZC, EC, cause="b")


OLD_NAMES = {
    "Parametric.cb(t=)": (_parametric_cb, "t"),
    "CoxPH.fit(method=)": (_cox_method, "method"),
    "NonParametricCounting.mcf_cb(confidence=)": (
        _recurrent_confidence,
        "confidence",
    ),
    "count_terminated_simulation(seed=)": (_recurrent_seed, "seed"),
    "WienerProcessModel.sf(t=)": (_degradation_t, "t"),
    "CompetingRisks.fit(method=)": (_competing_risks_method, "method"),
    "FineGray.fit(cause=)": (_fine_gray_cause, "cause"),
}


@pytest.mark.parametrize("call, old", OLD_NAMES.values(), ids=list(OLD_NAMES))
def test_old_name_is_an_unknown_argument(call, old):
    with pytest.raises(
        TypeError, match=f"unexpected keyword argument '{old}'"
    ):
        call()


def test_experimental_alias_is_gone():
    sys.modules.pop("surpyval.experimental", None)
    with pytest.raises(ImportError):
        importlib.import_module("surpyval.experimental")
    assert not hasattr(sp, "experimental")
    # Its contents live in surpyval.beta.ml.
    from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree

    assert SurvivalTree is not None and RandomSurvivalForest is not None
