"""Fits under an extreme change of the time unit.

Writing times in another unit (``t' = k t``) is a reparameterisation: a
fit to the rescaled data is the same model, its survival at ``k x`` the
original's at ``x``. A search that starts at a fixed value, judges its
steps against a floor of 1, or factorises a matrix whose entries span
the square of the unit change is not the same search in every unit, and
at the extremes it stopped short or failed:

- ``WeibullJointPH`` in seconds or in millions of the unit warned that
  its maximum was unverified, ``sf`` up to 0.99 off: the curvature of a
  slope a second is about 1e13 times one an hour's, and the random
  effects' modes, their grid and the outer search were factorised and
  floored in raw units.
- The additive hazards fits with times in millionths of the unit stopped
  at their start (every coefficient 0) and called it a maximum
  (``LogNormalAH`` ``sf`` 1.3 off); a coefficient there is a rate of
  about 1e5, searched against a floor of 1.
- ``NormalAL[GeneralLinear]``: a linear life's coefficients carry the
  unit of time, searched against a floor of 1 (``sf`` 3e-3 off).
- The distributional regressions on a location (Normal, Gumbel,
  Logistic): the location's coefficients, likewise (``sf`` up to 1e5
  off).
- ``HPP`` started its search at a rate of 1 in whatever unit, and in
  millionths or millions of the unit stayed there.
- ``GammaProcess`` and ``DestructiveDegradation`` took the density as a
  difference in steps of at least 1e-6 of the unit (``hf`` 1% and 26%
  off in millionths of the unit).

Each test fits a registry fixture in its own unit and in an extreme one
and compares the predictions; the tolerance is above the fits' own
agreement (noted per test) and below the old error. Every test fails
without its fix.
"""

import warnings

import numpy as np
import pytest

from surpyval.tests.conformance.checks import rescaled, rescaled_case
from surpyval.tests.conformance.registry import CASE_BY_NAME, call
from surpyval.tests.conformance.registry_families import WITH_COVARIATES

SECONDS = 3600.0  # the fixtures' unit as hours
YEARS = 1.0 / 8760.0
MICRO = 1e-6
MEGA = 1e6


def _fit(case, k):
    """The fit of ``case``'s fixture with every time multiplied by
    ``k``, and the warnings it gave that say its search did not end at a
    verified maximum."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = case.fit(rescaled(case, case.data(), k))
    unverified = [
        str(w.message)
        for w in caught
        if "verified maximum" in str(w.message)
        or "could not be evaluated or inverted" in str(w.message)
    ]
    return model, unverified


def _at(case, model, fn, x):
    Z = None
    if case.interface in WITH_COVARIATES:
        Zc = np.asarray(case.Z, dtype=float)
        Z = Zc[0] if Zc.ndim > 1 else Zc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(call(case, model, fn, x, Z), dtype=float)


def _same_model(name, k, rtol, functions=(("sf", 0),)):
    case = CASE_BY_NAME[name]
    ref, said = _fit(case, 1.0)
    assert not said, said
    model, said = _fit(case, k)
    assert not said, said
    moved = rescaled_case(case, k)
    for fn, power in functions:
        np.testing.assert_allclose(
            _at(moved, model, fn, case.x * k) * k**power,
            _at(case, ref, fn, case.x),
            rtol=rtol,
            err_msg=fn,
        )


# The joint fits agree to 1e-12 in every unit; old: sf 0.99 off in
# seconds and in a million units ([value+slope] 9.4), unverified.
JOINT = [
    (name, k)
    for name in (
        "WeibullJointPH",
        "WeibullJointPH[slope]",
        "WeibullJointPH[value+slope]",
    )
    for k in (SECONDS, MEGA)
]


@pytest.mark.parametrize(
    "name, k", JOINT, ids=[f"{n}-{k:g}" for n, k in JOINT]
)
def test_joint_fit_does_not_depend_on_the_unit(name, k):
    _same_model(name, k, 1e-8)


# The additive hazards fits agree to about 1e-5 (the search's own
# tolerance); old: in millionths of the unit sf 1.3 (LogNormal), 0.59
# (Normal), 77 (Gumbel), 1.0 (Logistic) off; WeibullAH and GammaAH warned
# in seconds or a million units.
ADDITIVE = [
    ("LogNormalAH", MICRO),
    ("NormalAH", MICRO),
    ("GumbelAH", MICRO),
    ("LogisticAH", MICRO),
    ("WeibullAH", SECONDS),
    ("WeibullAH", MEGA),
    ("GammaAH", MEGA),
    ("NormalAH", MEGA),
]


@pytest.mark.parametrize(
    "name, k", ADDITIVE, ids=[f"{n}-{k:g}" for n, k in ADDITIVE]
)
def test_additive_hazards_fit_does_not_depend_on_the_unit(name, k):
    _same_model(name, k, 1e-4, (("sf", 0), ("hf", 1)))


# Agreement about 1e-6; old: 3.1e-3 off (millionths), unverified there
# and in years.
@pytest.mark.parametrize("k", [MICRO, YEARS], ids=["micro", "years"])
def test_linear_life_fit_does_not_depend_on_the_unit(k):
    _same_model("NormalAL[GeneralLinear]", k, 1e-4)


# Agreement about 1e-5; old: Normal 0.51 and 3.4 (hf), Gumbel 1.5e5 and
# 1.3e3, Logistic 0.49 off, each unverified.
DISTRIBUTIONAL = [
    (name, k)
    for name in (
        "NormalDistributional",
        "GumbelDistributional",
        "LogisticDistributional",
    )
    for k in (MICRO, MEGA)
]


@pytest.mark.parametrize(
    "name, k", DISTRIBUTIONAL, ids=[f"{n}-{k:g}" for n, k in DISTRIBUTIONAL]
)
def test_distributional_fit_does_not_depend_on_the_unit(name, k):
    _same_model(name, k, 1e-4, (("sf", 0), ("hf", 1)))


# The rate is exact; old: a rate of 1, unverified, in millionths, years
# and a million units.
@pytest.mark.parametrize(
    "k", [MICRO, YEARS, MEGA], ids=["micro", "years", "mega"]
)
def test_hpp_rate_does_not_depend_on_the_unit(k):
    case = CASE_BY_NAME["HPP"]
    ref, _ = _fit(case, 1.0)
    model, said = _fit(case, k)
    assert not said, said
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.params * k, ref.params, rtol=1e-12)


# The process fit agrees to 2e-5; old: hf 9.7e-3 (millionths) and 1.4e-2
# (years) off. The destructive model's density: old 0.26 off, now 1e-11.
@pytest.mark.parametrize(
    "name, k, rtol",
    [
        ("GammaProcess", MICRO, 1e-4),
        ("GammaProcess", YEARS, 1e-4),
        ("DestructiveDegradation", MICRO, 1e-6),
    ],
    ids=["gamma-micro", "gamma-years", "destructive-micro"],
)
def test_density_does_not_depend_on_the_unit(name, k, rtol):
    case = CASE_BY_NAME[name]
    ref, _ = _fit(case, 1.0)
    model, _ = _fit(case, k)
    moved = rescaled_case(case, k)
    np.testing.assert_allclose(
        _at(moved, model, "df", case.x * k) * k,
        _at(case, ref, "df", case.x),
        rtol=rtol,
    )
