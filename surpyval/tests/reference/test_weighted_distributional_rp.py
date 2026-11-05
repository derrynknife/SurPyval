"""Distributional and Royston-Parmar regression with sampling weights
``w`` against R.

The data are ``_helpers.weighted_reference_data()`` (right censored, a
late entry on some rows). The values are printed by
``scripts/reference/reference_r_weighted_distributional_rp.R`` (R 4.6.2,
flexsurv 2.3.2, numDeriv 2016.8-1.1):

* the point estimates and log-likelihood are flexsurv's with
  ``weights = w`` (``flexsurvreg`` with ``anc = list(shape = ~ z1)`` for
  the Weibull, ``flexsurvspline`` for Royston-Parmar);
* the robust (sandwich) standard errors are computed by hand in R, since
  flexsurv's with weights are model based: the rows' scores by numDeriv,
  the bread the inverse of the numDeriv Hessian of the weighted negative
  log-likelihood, the meat ``sum w^2 s s'`` (HC0).

The Royston-Parmar fits are given the knots SurPyval places (on the event
times counted by ``n``, not weighted); flexsurv's default places them at
an approximate weighted quantile (DECISIONS.md, "Weights stage 2:
distributional and Royston-Parmar regression").

Tolerances: the estimates to 1e-5 (two optimisers stopping at the same
maximum); the log-likelihood to 1e-8 relative; the robust standard errors
to 1e-4 relative (R's are numDeriv's Richardson extrapolation, SurPyval's
autograd's).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import quietly, weighted_reference_data

# flexsurv order: log shape, log scale, scale z1, scale z2, shape z1
WD_EST = [
    0.7510066329,
    2.091663104,
    -0.6553470265,
    -0.6735067286,
    -0.4849280104,
]
WD_LL = -94.23844206
WD_ROBUST_SE = [
    0.1719977429,
    0.1786591526,
    0.2896760666,
    0.127485452,
    0.2912449562,
]
RP_KNOTS = [-2.302585093, 1.064710737, 2.116255515]
# gamma0, gamma1, gamma2, z1, z2 (flexsurv's sign)
RP = {
    "hazard": {
        "est": [
            -3.389144379,
            1.349579232,
            -0.02435347927,
            1.061342562,
            1.065571084,
        ],
        "ll": -95.83493361,
        "se": [
            0.5217365816,
            1.263487725,
            0.1376257119,
            0.4747426325,
            0.2450934709,
        ],
    },
    "odds": {
        "est": [
            -4.393119434,
            1.010156789,
            -0.1475781331,
            1.724415435,
            1.451494768,
        ],
        "ll": -96.53167201,
        "se": [
            0.7835755306,
            0.9557748702,
            0.1076282982,
            0.7146385715,
            0.3360660883,
        ],
    },
}


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _weibull(data, w):
    return quietly(
        sp.WeibullDistributional.fit,
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        w=w,
        covariates={"alpha": [0, 1], "beta": [0]},
    )


def test_weibull_distributional_matches_flexsurvreg(data):
    model = _weibull(data, data["w"])
    assert model.parameter_names == [
        "alpha",
        "beta",
        "alpha_coef_0",
        "alpha_coef_1",
        "beta_coef_0",
    ]
    log_shape, log_scale, s1, s2, b1 = WD_EST
    # SurPyval's sign: a positive coefficient on the scale is a shorter
    # life, so the scale's coefficients are flexsurv's negated.
    expected = [np.exp(log_scale), np.exp(log_shape), -s1, -s2, b1]
    assert_allclose(model.params, expected, rtol=1e-5, atol=1e-6)
    assert_allclose(model.log_likelihood, WD_LL, rtol=1e-8)
    se = model.standard_errors()
    sd_shape, sd_scale, sd_s1, sd_s2, sd_b1 = WD_ROBUST_SE
    # alpha and beta on their natural scale: the delta method of the logs
    expected_se = [
        np.exp(log_scale) * sd_scale,
        np.exp(log_shape) * sd_shape,
        sd_s1,
        sd_s2,
        sd_b1,
    ]
    assert_allclose(se, expected_se, rtol=1e-4)


@pytest.mark.parametrize("scale", ["hazard", "odds"])
def test_royston_parmar_matches_flexsurvspline(data, scale):
    fitter = sp.RoystonParmarRegression(df=2, scale=scale)
    model = quietly(
        fitter.fit,
        data["x"],
        data["Z"],
        c=data["c"],
        tl=data["tl"],
        w=data["w"],
    )
    assert_allclose(model.model.dist.knots, RP_KNOTS, rtol=1e-9)
    ref = RP[scale]
    est = np.array(ref["est"])
    # On the odds scale SurPyval's coefficients are flexsurv's negated
    # (a positive one is a shorter life).
    sign = 1.0 if scale == "hazard" else -1.0
    expected = [*est[:3], *(sign * est[3:])]
    assert_allclose(model.params, expected, rtol=1e-5, atol=1e-6)
    assert_allclose(model.log_likelihood, ref["ll"], rtol=1e-8)
    assert_allclose(model.standard_errors(), ref["se"], rtol=1e-4)
