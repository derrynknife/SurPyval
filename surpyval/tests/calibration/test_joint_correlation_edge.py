"""A joint fit near the edge of the valid spreads: :func:`joint_data`
on a quadratic path with its level, rate and curvature all varying.

Before the nodes near a singular spread were placed in the prior's own
coordinates, the search landed within 1e-10 of a singular correlation
matrix (corr_a_b 0.997), the one place the 7-point grid overstated the
likelihood (406.94 there; 494.19 at corr_a_b 1e-10 lower and anywhere
below), and reported 406.38, unverified. With the nodes right, it
stalled near a correlation of 0.997 between the rate and the curvature
(479.465, unverified, gradients of 0.3 standard errors) until a round
whose step failed tried minus the whitened gradient. It now reaches
479.3549, no worse than the fit with the curvature common (479.6325),
but unverified (its gradient still above the rule there).
The fit takes about two minutes, so it runs with the calibration
studies; nothing in it is random."""

import warnings

from surpyval import JointPH, Weibull
from surpyval.degradation import QuadraticPath
from surpyval.tests.conformance.registry_fixtures import joint_data


def test_quadratic_with_every_parameter_varying_reaches_the_maximum():
    data = joint_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        common = JointPH(Weibull, QuadraticPath, varying=("a", "b")).fit(
            **data
        )
        every = JointPH(Weibull, QuadraticPath, varying=("a", "b", "c")).fit(
            **data
        )
    assert abs(every.neg_ll() - 479.3549) < 1e-3
    # as it stands: the gradient there is still above the rule
    assert every.maximum == "unverified"
    assert every.neg_ll() <= common.neg_ll() + 1e-4
