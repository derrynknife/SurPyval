"""#962: survival given an age depends on the covariate from that age on
only under proportional hazards and odds, additive hazards and Cox, so the
covariates need only start by then; accelerated failure time and life
still need them from 0."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import CovariatePath, ItemLifetime, StepSchedule

TIMES = np.array([0.5, 1.0, 4.0, 6.0])


def _fit(name):
    rng = np.random.default_rng(1)
    Z = rng.binomial(1, 0.5, (300, 1)).astype(float)
    x = rng.normal(3, 2, 300) - 0.8 * Z[:, 0]
    if not name.startswith(("Normal", "Logistic", "Gumbel")):
        x = np.abs(x) + 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, name).fit(x, Z)


def _origin(m):
    return -np.inf if float(m.distribution.support[0]) < 0 else 0.0


LATE = StepSchedule.from_changepoints([1, 2], [[0.0], [1.0]])
LATE_PATH = CovariatePath.from_points([1, 3], [0.0, 1.0])
HAZARD_FAMILIES = [
    "NormalPH",
    "LogisticPO",
    "GumbelPH",
    "NormalAH",
    "WeibullPH",
    "WeibullPO",
    "WeibullAH",
]


@pytest.mark.parametrize("name", HAZARD_FAMILIES)
def test_a_schedule_from_the_age_is_enough(name):
    m = _fit(name)
    for before in (0.0, 1.0):
        full = StepSchedule.from_changepoints(
            [_origin(m), 1, 2], [[before], [0.0], [1.0]]
        )
        np.testing.assert_allclose(
            m.sf_tvc(TIMES, LATE, given=1),
            m.sf_tvc(TIMES, full, given=1),
            rtol=1e-12,
        )
    np.testing.assert_allclose(
        m.mean_tvc(LATE, given=1), m.mean_tvc(full, given=1), rtol=1e-8
    )


@pytest.mark.parametrize("name", HAZARD_FAMILIES)
def test_a_path_from_the_age_is_enough(name):
    m = _fit(name)
    full = CovariatePath.from_points([-5, 1, 3], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        m.sf_tvc(TIMES, LATE_PATH, given=1),
        m.sf_tvc(TIMES, full, given=1),
        rtol=1e-9,
    )


@pytest.mark.parametrize("name", ["NormalPH", "WeibullPH", "LogisticPO"])
def test_bounds_from_the_age(name):
    m = _fit(name)
    full = StepSchedule.from_changepoints(
        [_origin(m), 1, 2], [[1.0], [0.0], [1.0]]
    )
    np.testing.assert_allclose(
        m.cb_tvc(TIMES, LATE, given=1),
        m.cb_tvc(TIMES, full, given=1),
        rtol=1e-9,
    )


@pytest.mark.parametrize("name", HAZARD_FAMILIES)
def test_covariates_starting_after_the_age_are_refused(name):
    m = _fit(name)
    with pytest.raises(ValueError, match=r"after the age given \(0.5\)"):
        m.sf_tvc(4, LATE, given=0.5)
    with pytest.raises(ValueError, match=r"after the age given \(0.5\)"):
        m.sf_tvc(4, LATE_PATH, given=0.5)


@pytest.mark.parametrize("name", ["NormalAFT", "WeibullAFT"])
def test_aft_still_needs_the_covariate_from_0(name):
    # The age reached by the given age depends on the covariate before it.
    m = _fit(name)
    a, b = (
        m.sf_tvc(
            4,
            StepSchedule.from_changepoints(
                [_origin(m), 1, 2], [[v], [0.0], [1.0]]
            ),
            given=1,
        )
        for v in (0.0, 1.0)
    )
    assert not np.isclose(a, b)
    with pytest.raises(ValueError, match="after the time origin"):
        m.sf_tvc(4, LATE, given=1)


def test_cox():
    m = _fit("CoxPH")
    full = StepSchedule.from_changepoints([0, 1, 2], [[1.0], [0.0], [1.0]])
    np.testing.assert_allclose(
        m.sf_tvc(TIMES, LATE, given=1),
        m.sf_tvc(TIMES, full, given=1),
        rtol=1e-12,
    )
    full_path = CovariatePath.from_points([0, 1, 3], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        m.sf_tvc(TIMES, LATE_PATH, given=1),
        m.sf_tvc(TIMES, full_path, given=1),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="after the age given"):
        m.sf_tvc(4, LATE, given=0.5)


def test_an_item_entering_late_given_its_entry():
    # The prediction counterpart of a record with left_truncation.
    m = _fit("WeibullPH")
    item = ItemLifetime(
        "a",
        observed=10,
        left_truncation=6,
        time_varying_covariates={5: [0.0], 8: [1.0]},
    )
    full = StepSchedule.from_changepoints([0, 8], [[0.0], [1.0]])
    np.testing.assert_allclose(
        m.sf_tvc([7, 9], item, given=6),
        m.sf_tvc([7, 9], full, given=6),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="Item 'a'.*after the age given"):
        m.sf_tvc([7, 9], item, given=4)


def test_without_given_the_rules_are_unchanged():
    with pytest.raises(ValueError, match="after the time origin"):
        _fit("WeibullPH").sf_tvc(4, LATE)
    from_0 = StepSchedule.from_changepoints([0, 2], [[0.0], [1.0]])
    with pytest.raises(ValueError, match="back to -inf.*given="):
        _fit("NormalPH").sf_tvc(4, from_0)
