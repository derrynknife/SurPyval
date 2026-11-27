"""A path of one's own with a kink (its slope jumps at a known age): the
integrals of the hazard over age are split there where the path names
the age in ``breaks``, as a spline's knots are."""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from surpyval import JointPH, Weibull
from surpyval.degradation.path_models import PathModel
from surpyval.tests.univariate.joint._fleet import simulate_path
from surpyval.univariate.joint import _likelihood as lk

KNEE = 365.0


class Hinge(PathModel):
    """``a + b t + c max(t - 365, 0)``: a line that turns at day 365
    (module level, to pickle), not saying where."""

    name = "Hinge"
    parameter_names = ["a", "b", "c"]

    def path(self, x, a, b, c):
        x = np.asarray(x, dtype=float)
        return a + b * x + c * np.maximum(x - KNEE, 0.0)

    def inv_path(self, y, a, b, c):
        raise NotImplementedError

    def _initial_guess(self, x, y):
        return [float(np.asarray(y)[0]), 0.0, 0.0]


class NamedHinge(Hinge):
    """The same line, naming its corner."""

    breaks = (KNEE,)


#: flat at 3, then rising 3 a year; Weibull PH (800, 2) at a marker of 4
A, B, C = 3.0, 0.0, 3.0 / 365
ALPHA, BETA, EFFECT, CENTRE = 800.0, 2.0, 0.9, 4.0
AGES = np.array([200.0, 400.0, 730.0])


def _cumhaz(path_model, lag=0.0, lo=None):
    """The model's cumulative hazard of the line at ``AGES`` (from
    ``lo``, else from 0)."""
    pf = lk.PathFunction(path_model, lag=lag)
    theta = {
        "mu": np.array([A, B, C]),
        "Delta": np.zeros((3, 0)),
        "L": np.array([[0.5]]),
        "sigma": 0.4,
        "alpha": ALPHA,
        "beta": BETA,
        "gamma": np.zeros(0),
        "zeta": np.array([EFFECT]),
    }
    n = AGES.size
    arrays = lk.JointArrays.build(
        np.ones(n),
        AGES,
        np.zeros(n),
        np.zeros(n),
        np.zeros((n, 0)),
        [np.array([0.0])] * n,
        [np.array([3.0])] * n,
    )
    params = np.broadcast_to(np.array([A, B, C]), (n, 1, 3)).copy()
    if lo is None:
        out = lk.log_cumhaz(theta, pf, params, arrays, AGES, CENTRE)
    else:
        out = lk.log_cumhaz_between(
            theta, pf, params, arrays, np.full(n, lo), AGES, CENTRE
        )
    return pf.cuts, np.exp(np.asarray(out))[:, 0]


def _exact(lag=0.0, lo=0.0):
    def h(s):
        m = Hinge().path(max(s - lag, 0.0), A, B, C)
        return (
            (BETA / ALPHA)
            * (s / ALPHA) ** (BETA - 1)
            * np.exp(EFFECT * (m - CENTRE))
        )

    out = []
    for t in AGES:
        corners = [x for x in (lag, KNEE + lag) if lo < x < t]
        out.append(quad(h, lo, t, points=corners or None, limit=200)[0])
    return np.array(out)


@pytest.mark.parametrize("lag", [0.0, 90.0], ids=["at once", "lagged"])
def test_a_named_kink_splits_the_integral(lag):
    cuts, got = _cumhaz(NamedHinge(), lag=lag)
    assert cuts == tuple(sorted({KNEE + lag} | ({lag} if lag else set())))
    np.testing.assert_allclose(got, _exact(lag), rtol=1e-10)


def test_a_named_kink_splits_an_interval():
    _, got = _cumhaz(NamedHinge(), lo=100.0)
    np.testing.assert_allclose(got, _exact(lo=100.0), rtol=1e-10)


def test_an_unnamed_kink_loses_the_rule_its_accuracy():
    # why ``breaks``: the one rule over the corner is off by 4e-4 of the
    # cumulative hazard past it (the docs give the number)
    cuts, got = _cumhaz(Hinge())
    assert cuts == ()
    error = np.abs(got / _exact() - 1.0)
    assert error[0] < 1e-12  # before the corner
    assert 1e-4 < error[1] < 1e-3


def test_a_fleet_on_a_kinked_path_fits():
    # each item's level and rate its own, the turn after day 365 common
    data = simulate_path(
        NamedHinge(),
        [3.0, 0.5 / 365, 2.0 / 365],
        [0.5, 0.3 / 365, 0.0],
        0.4,
        n=150,
        seed=4,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = JointPH(Weibull, NamedHinge(), varying=("a", "b")).fit(
            data, marker_reference=CENTRE
        )
    assert model.maximum == "verified", [str(w.message) for w in caught]
    names = list(model.parameter_names)
    k = names.index("marker_effect")
    z = (model.params[k] - EFFECT) / model.standard_errors()[k]
    assert abs(z) < 3


def test_an_unnamed_kink_is_warned_of_at_the_fit():
    data = simulate_path(
        NamedHinge(),
        [3.0, 0.5 / 365, 2.0 / 365],
        [0.5, 0.3 / 365, 0.0],
        0.4,
        n=150,
        seed=4,
    )
    with pytest.warns(UserWarning, match="seems to have a kink.*breaks"):
        JointPH(Weibull, Hinge(), varying=("a", "b")).fit(
            data, marker_reference=CENTRE
        )
    # a smooth path of one's own, and one naming its kink, are not
    for path in (NamedHinge(), Smooth()):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            JointPH(Weibull, path, varying=("a", "b")).fit(
                data, marker_reference=CENTRE
            )
        assert not [w for w in caught if "kink" in str(w.message)]


class Smooth(PathModel):
    """``a + b t + c t^2``, a smooth path of one's own."""

    name = "Smooth"
    parameter_names = ["a", "b", "c"]

    def path(self, x, a, b, c):
        x = np.asarray(x, dtype=float)
        return a + b * x + c * x**2

    def inv_path(self, y, a, b, c):
        raise NotImplementedError

    def _initial_guess(self, x, y):
        return [float(np.asarray(y)[0]), 0.0, 0.0]
