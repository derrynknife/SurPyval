"""A model with no offset, limited-failure or zero-inflation part gives
its distribution's own sf, ff and df, without the transforms (#642)."""

import math

import numpy as np
import pytest

import surpyval as surv

X = np.array([-1.0, 0.0, 0.5, 2.0, 7.5, 40.0, np.nan, np.inf])


@pytest.mark.parametrize(
    "dist, params",
    [
        (surv.Weibull, [10, 2]),
        (surv.LogNormal, [1, 0.5]),
        (surv.Exponential, [0.1]),
        (surv.Gamma, [2, 0.3]),
        (surv.Geometric, [0.3]),
    ],
)
@pytest.mark.parametrize("fn", ["sf", "ff", "df"])
def test_a_plain_model_is_its_distribution(dist, params, fn):
    model = dist.from_params(params)
    got = getattr(model, fn)(X)
    want = getattr(model.dist, fn)(X, *params)
    np.testing.assert_array_equal(got, want)
    # a scalar query gives a scalar, and a -0.0 below the support is 0.0
    value = getattr(model, fn)(-1.0)
    assert np.ndim(value) == 0
    assert math.copysign(1.0, float(value)) == 1.0


def test_the_transforms_still_apply_off_the_plain_model():
    base = surv.Weibull.from_params([10, 2])
    lfp = surv.Weibull.from_params([10, 2], lfp_p=0.8)
    zi = surv.Weibull.from_params([10, 2], f0=0.1)
    shifted = surv.Weibull.from_params([10, 2], gamma=1.5)
    x = np.array([0.5, 2.0, 7.5])
    np.testing.assert_allclose(lfp.ff(x), 0.8 * base.ff(x))
    np.testing.assert_allclose(zi.ff(x), 0.1 + 0.9 * base.ff(x))
    np.testing.assert_allclose(shifted.sf(x), base.sf(x - 1.5))
