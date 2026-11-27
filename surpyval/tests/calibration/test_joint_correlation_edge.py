"""A joint fit near the edge of the valid spreads: :func:`joint_data`
on a quadratic path with its level, rate and curvature all varying.

Before the nodes near a nearly singular spread were placed in the
prior's own coordinates, the search ran to a correlation of 0.997
between the level and the rate where the 7-point grid overstated each
item's likelihood (by 1.8 in log for one item, against a Monte Carlo
integral), and reported 406.38, unverified: 73 better than the fit with
the curvature common, which it cannot truly be by more than a little.
With the nodes right, it stalled near a correlation of 0.997 between the
rate and the curvature (479.465, unverified) until a round whose step
failed tried the Newton step. It now reaches a verified maximum inside,
479.3549, no worse than the fit with the curvature common (479.6325).
The fit takes about two minutes, so it runs with the calibration
studies; nothing in it is random."""

import warnings

from surpyval import JointPH, Weibull
from surpyval.degradation import QuadraticPath
from surpyval.tests.conformance.registry_fixtures import joint_data


def test_quadratic_with_every_parameter_varying_reaches_a_maximum():
    data = joint_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        common = JointPH(Weibull, QuadraticPath, varying=("a", "b")).fit(
            **data
        )
        every = JointPH(Weibull, QuadraticPath, varying=("a", "b", "c")).fit(
            **data
        )
    assert every.maximum == "verified"
    assert abs(every.neg_ll() - 479.3549) < 1e-3
    assert every.neg_ll() <= common.neg_ll() + 1e-4
