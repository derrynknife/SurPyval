"""A joint fit of several markers whose spread is 0 in one of them: the
larger model is never worse than the one with that parameter common.

:func:`registry_fixtures.joint_markers_data`'s temperature rises at the
same rate (0.01 a day) in every item. Fitted with both markers' level and
rate varying, the search stalled 0.40 below the fit with the
temperature's rate common (997.95 against 997.55, unverified) before the
spread at 0 was handled on its boundary. Each fit takes minutes (four
varying parameters, two markers), so it runs with the calibration
studies; nothing in it is random.
"""

import warnings

from surpyval import JointPH, Weibull
from surpyval.degradation import LinearPath
from surpyval.tests.conformance.registry_fixtures import joint_markers_data


def test_two_markers_with_a_rate_common_in_fact():
    data = joint_markers_data()
    paths = {"vibration": LinearPath, "temperature": LinearPath}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        common = JointPH(
            Weibull,
            paths,
            varying={"vibration": ["a", "b"], "temperature": ["a"]},
        ).fit(**data)
        both = JointPH(
            Weibull,
            paths,
            varying={"vibration": ["a", "b"], "temperature": ["a", "b"]},
        ).fit(**data)
    assert both.neg_ll() <= common.neg_ll() + 1e-6 * abs(common.neg_ll())
