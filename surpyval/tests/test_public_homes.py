"""Names downstream code needs, imported from their public homes."""

import numpy as np


def test_dynamic_prediction_from_the_joint_namespace():
    from surpyval.univariate.joint import DynamicPrediction
    from surpyval.univariate.joint.dynamic import DynamicPrediction as same

    assert DynamicPrediction is same


def test_conditional_gaps_from_recurrent():
    from surpyval import Weibull
    from surpyval.recurrent import conditional_gaps

    life = Weibull.from_params([10.0, 2.0])
    gaps = conditional_gaps(life, np.array([0.0, 5.0]), np.array([0.5, 0.5]))
    # H(v + x) = H(v) + log 2 with H(t) = (t / 10)^2.
    expected = 10 * np.sqrt(np.array([0.0, 0.25]) + np.log(2)) - [0.0, 5.0]
    np.testing.assert_allclose(gaps, expected, rtol=1e-10)
