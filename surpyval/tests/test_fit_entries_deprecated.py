"""The data entry points ``fit_from_surpyval_data`` and
``fit_from_recurrent_data`` are deprecated for ``fit(data)``.

Every fitter's ``fit`` takes the data object itself (a ``SurpyvalData``,
or the ``EventHistoryData`` of ``handle_xicn``), so the two public entry
points were a second and third way to do the same thing. Until v0.30 the
old names run the private method ``fit`` calls, with a ``FutureWarning``
saying to call ``fit(data)``; the fit is the same to the last digit.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import SurpyvalData
from surpyval import recurrent as rc
from surpyval.tests.conformance.registry_fixtures import recurrent_data
from surpyval.utils.deprecation import REMOVED_IN, RenamedAttribute
from surpyval.utils.recurrent_utils import handle_xicn

OLD_NAMES = ("fit_from_surpyval_data", "fit_from_recurrent_data")


def _public_fitters():
    """``(label, fitter, old name)`` for every public object that has one
    of the old entry points."""
    out = {}
    for module, names in (
        (sp, sp.__all__),
        (rc, [n for n in dir(rc) if not n.startswith("_")]),
    ):
        for name in names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(module, name, None)
            if obj is None:
                continue
            owner = obj if isinstance(obj, type) else type(obj)
            for old in OLD_NAMES:
                for klass in owner.__mro__:
                    if isinstance(klass.__dict__.get(old), RenamedAttribute):
                        out[(name, old)] = (obj, old)
                        break
    return [
        pytest.param(obj, old, id=f"{name}.{old}")
        for (name, old), (obj, old) in sorted(out.items())
    ]


FITTERS = _public_fitters()


def test_the_fitters_are_found():
    labels = {p.id for p in FITTERS}
    for expected in (
        "Weibull.fit_from_surpyval_data",
        "Hypoexponential.fit_from_surpyval_data",
        "CrowAMSAA.fit_from_recurrent_data",
        "HPP.fit_from_recurrent_data",
        "NonParametricCounting.fit_from_recurrent_data",
        "GeneralizedRenewal.fit_from_recurrent_data",
        "ProportionalIntensityNHPP.fit_from_recurrent_data",
        "CauseSpecificNHPP.fit_from_recurrent_data",
    ):
        assert expected in labels


@pytest.mark.parametrize("fitter, old", FITTERS)
def test_the_old_name_warns_and_names_fit(fitter, old):
    with pytest.warns(FutureWarning) as caught:
        method = getattr(fitter, old)
    message = str(caught[0].message)
    assert f".{old} is deprecated" in message
    assert f"v{REMOVED_IN}" in message
    assert "call fit(data)" in message
    assert method == getattr(fitter, "_" + old)


def _same(a, b):
    da = json.dumps(a.to_dict(), sort_keys=True, default=str)
    db = json.dumps(b.to_dict(), sort_keys=True, default=str)
    assert da == db


X = np.array([1.2, 3.1, 4.4, 7.0, 9.3, 10.5, 12.0, 15.2, 18.9, 20.0])
C = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1])


@pytest.mark.parametrize(
    "fitter, kw",
    [
        (sp.Weibull, {}),
        (sp.Weibull, {"how": "MPP"}),
        (sp.LogNormal, {"offset": True}),
        (sp.Exponential, {"fixed": {"failure_rate": 0.1}}),
        (sp.Hypoexponential, {"stages": 2}),
    ],
    ids=["Weibull", "Weibull-MPP", "LogNormal-offset", "fixed", "Hypo"],
)
def test_the_old_univariate_call_is_fit_of_the_data(fitter, kw):
    data = SurpyvalData(X, C)
    with pytest.warns(FutureWarning, match=r"call fit\(data\)"):
        old = fitter.fit_from_surpyval_data(data, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = fitter.fit(data, **kw)
    np.testing.assert_array_equal(old.params, new.params)
    assert old.log_likelihood == new.log_likelihood


def _recurrent(with_Z=False, with_e=False):
    d = recurrent_data(with_Z, with_e)
    kw = {}
    if with_Z:
        kw["Z"] = d["Z"]
    if with_e:
        kw["e"] = d["e"]
    return handle_xicn(d["x"], d["i"], d["c"], d["n"], **kw)


@pytest.mark.parametrize(
    "fitter, kw, with_Z, with_e",
    [
        (rc.HPP, {}, False, False),
        (rc.CrowAMSAA, {}, False, False),
        (rc.Duane, {}, False, False),
        (rc.CoxLewis, {}, False, False),
        (rc.GoelOkumoto, {}, False, False),
        (rc.NonParametricCounting, {}, False, False),
        (rc.GeneralizedRenewal, {}, False, False),
        (rc.GeneralizedOneRenewal, {}, False, False),
        (rc.ARA, {}, False, False),
        (rc.ARI, {}, False, False),
        (rc.ProportionalIntensityHPP, {}, True, False),
        (rc.ProportionalIntensityNHPP, {}, True, False),
        (rc.CauseSpecificMCF, {}, False, True),
        (rc.CauseSpecificNHPP, {}, False, True),
        (rc.CauseSpecificProportionalIntensity, {}, True, True),
    ],
    ids=lambda v: (
        getattr(v, "__name__", type(v).__name__)
        if not isinstance(v, (dict, bool))
        else None
    ),
)
def test_the_old_recurrent_call_is_fit_of_the_data(fitter, kw, with_Z, with_e):
    data = _recurrent(with_Z, with_e)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new = fitter.fit(data, **kw)
        warnings.simplefilter("always")
        with pytest.warns(FutureWarning, match=r"call fit\(data\)"):
            old = fitter.fit_from_recurrent_data(data, **kw)
    _same(old, new)
