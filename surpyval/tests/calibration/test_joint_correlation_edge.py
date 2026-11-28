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
and a maximum (the largest gradient 4e-4 standard errors; every
perturbation raises the negative log-likelihood), but on the edge of the
valid spreads with no pair at +-1: the curvature's own spread is about
1.4e-10, an exact linear combination of the level and the rate. That
edge is not held, so the maximum cannot be certified: it ends
unverified.
The fit takes about two minutes, so it runs with the calibration
studies; nothing in it is random."""

import sys
import warnings

import pytest

from surpyval import JointPH, Weibull
from surpyval.degradation import QuadraticPath
from surpyval.tests.conformance.registry_fixtures import joint_data


@pytest.mark.xfail(
    sys.platform.startswith("linux"),
    strict=True,
    reason="on the 0.30 release run (x86-64 Linux) the fit reached the same "
    "-log L, 479.3549, but was marked 'verified' on the unheld rank-one "
    "edge it cannot certify (macOS: 'unverified'); released as a known "
    "limitation",
)
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
    # a maximum on the rank-one edge, which is not held: uncertified
    assert every.maximum == "unverified"
    assert every.neg_ll() <= common.neg_ll() + 1e-4
