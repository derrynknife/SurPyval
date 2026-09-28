"""Targeted review of ``parametric/distributions/custom_distribution.py``
(#399).

Each test pins a bug found by reading the module adversarially. They are
strict expected failures until the bug is fixed.
"""

import numpy as np
import pytest

import surpyval as surv


def _weibull_hf(x, *params):
    return (x / params[0]) ** params[1]


def _data():
    np.random.seed(0)
    return surv.Weibull.random(50, 10, 2)


@pytest.mark.xfail(
    strict=True,
    reason="#437: a CustomDistribution parameter named 'k' overwrites the "
    "model's parameter count: AIC 290.08 (k = 2.29) against 289.50 for "
    "the same fit named 'shape'",
)
def test_parameter_named_k_keeps_the_information_criteria():
    x = _data()
    names = {}
    for shape_name in ("shape", "k"):
        dist = surv.CustomDistribution(
            "review_weibull_" + shape_name,
            _weibull_hf,
            ["lam", shape_name],
            ((0, None), (0, None)),
            (0, np.inf),
        )
        names[shape_name] = dist.fit(x)
    np.testing.assert_allclose(
        names["k"].aic(), names["shape"].aic(), rtol=1e-8
    )


@pytest.mark.xfail(
    strict=True,
    reason="#437: CustomDistribution accepts parameter names that collide "
    "with model attributes ('dist', 'data', 'lfp', 'zi', 'method'), which "
    "then break sf/bic/cb with AttributeError, IndexError or ValueError; "
    "the constructor should refuse them with a ValueError",
)
@pytest.mark.parametrize("name", ["dist", "data", "lfp", "zi", "method"])
def test_parameter_names_that_collide_with_the_model_are_refused(name):
    x = _data()
    try:
        dist = surv.CustomDistribution(
            "review_collide_" + name,
            _weibull_hf,
            ["lam", name],
            ((0, None), (0, None)),
            (0, np.inf),
        )
    except ValueError:
        return
    model = dist.fit(x)
    model.sf(5.0)
    model.bic()
    model.cb(5.0)
    surv.from_dict(model.to_dict()).sf(5.0)


@pytest.mark.xfail(
    strict=True,
    reason="#437: CustomDistribution.qf(-0.1) is the support's lower "
    "bound (0.0) where every built-in distribution returns nan",
)
def test_qf_outside_the_unit_interval_is_nan():
    dist = surv.CustomDistribution(
        "review_qf",
        _weibull_hf,
        ["lam", "shape"],
        ((0, None), (0, None)),
        (0, np.inf),
    )
    got = dist.qf(np.array([-0.1, 1.5]), 10.0, 3.0)
    assert np.isnan(surv.Weibull.qf(np.array([-0.1]), 10.0, 3.0)).all()
    assert np.isnan(got).all()
