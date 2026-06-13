import numpy as np
import pytest

from surpyval import Exponential, Gamma, Normal
from surpyval.recurrent.renewal import GeneralizedOneRenewal


def test_g1_renewal():
    # Solution from:
    # Kaminskiy, M. P., and V. V. Krivtsov.
    # "G1-renewal process as repairable system model."
    # Reliability: Theory & Applications 5.3 (18) (2010): 7-14.
    # Ref:
    # https://arxiv.org/pdf/1006.3718.pdf

    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    model = GeneralizedOneRenewal.fit(x, dist=Exponential)
    life = 1.0 / model.model.params[0]

    assert np.allclose([0.232], model.q, atol=1e-3)
    assert np.allclose([4.781], life, atol=1e-3)


def test_g1_renewal_scale_family_dist():
    # The time-axis formulation lets G1 fit any non-negative lifetime
    # distribution, not just Weibull/Exponential. Gamma is a scale family
    # whose scale parameter is not the first positional parameter, so this
    # would have been silently mishandled by the old parameter-scaling code.
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()
    model = GeneralizedOneRenewal.fit(x, dist=Gamma)

    assert model.model.dist == Gamma
    assert np.all(np.isfinite(model.model.params))
    assert np.isfinite(model.q)


def test_g1_renewal_rejects_distribution_with_negative_support():
    # Distributions with support over negative values cannot be scaled into a
    # valid interarrival distribution and must be rejected up front.
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    with pytest.raises(ValueError, match="non-negative lifetime"):
        GeneralizedOneRenewal.fit(x, dist=Normal)

    with pytest.raises(ValueError, match="non-negative lifetime"):
        GeneralizedOneRenewal.fit_from_parameters([10, 2], 0.2, dist=Normal)
