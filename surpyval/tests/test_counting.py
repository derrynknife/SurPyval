import numpy as np

from surpyval import Exponential, GeneralizedOneRenewal


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
