"""A joint search started at a spread too small for the integration to
resolve ends at the fit held at 0, not at an artefact.

On the two-mode fixture with a spline whose bend varies, the bend has no
spread: the fit holds ``sd_s1`` at 0 (985.8987). With the other
parameters fixed, the likelihood by the adaptive nodes is unchanged down
to ``sd_s1`` 1e-10, moves 0.003 at 1e-14, 2.3 at 1e-16 and 107 below
1e-18 (878.50): the nodes' offsets fall within rounding of the bend's
value, the prior's term at them is lost and the change of variables no
longer cancels it. A search is now bounded above that, and an answer in
effect at 0 that beats the held fit is taken as the artefact it is. The
search takes minutes, so it runs with the calibration studies; nothing
in it is random."""

import warnings

import numpy as np

from surpyval import JointPH, Weibull
from surpyval.degradation import SplinePath
from surpyval.tests.conformance.registry_fixtures import joint_causes_data
from surpyval.univariate.joint import joint_ph as jp

HELD = 985.8987


def test_a_search_from_a_tiny_spread_ends_held():
    data = joint_causes_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held = JointPH(
            Weibull,
            SplinePath.natural(df=2),
            varying=("level", "rate", "s1"),
        ).fit(**data)
    assert held.spreads_at_zero == ["s1"]
    assert abs(held.neg_ll() - HELD) < 1e-3
    # the full model, the bend's own spread started at 1e-18
    names = list(held.parameter_names)
    d = held.to_dict()
    d["params"][names.index("sd_s1")] = 1e-6
    from surpyval.univariate.joint import JointModel

    near = JointModel.from_dict(d)
    lik, arrays, u = jp.likelihood_parts(near, **data)
    theta = jp._theta_of(lik.layout, u)
    k = list(lik.layout.varying).index(
        list(held.path_model.parameter_names).index("s1")
    )
    theta["L"][k, :] = 0.0
    theta["L"][k, k] = 1e-18
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        answer = jp._maximise(lik, arrays, theta)
        fit_model, (u2, value, _, maximum), _, _ = jp._on_an_edge(
            lik, arrays, answer, []
        )
    assert np.isfinite(value)
    assert abs(value - held.neg_ll()) < 1e-4
