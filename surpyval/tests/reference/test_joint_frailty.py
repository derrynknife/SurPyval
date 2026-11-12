"""The joint frailty model of recurrent and terminal events against R's
``frailtypack`` (``frailtyPenal`` with ``terminal()``, ``hazard =
"Weibull"``), on the fixture "joint_frailty" (250 units of a joint
gamma-frailty model; scripts/reference/reference_r_joint_frailty.R).

frailtypack integrates a gamma frailty with Gauss-Laguerre quadrature of
fixed nodes (``nb.gl = 32``) and a log-normal one with Gauss-Hermite
quadrature (``nb.gh = 32``); SurPyval with a trapezoid rule over each
unit's integrand, accurate to rounding (checked against scipy's adaptive
quadrature in ``tests/recurrent/test_joint_frailty.py``). So:

* with the log-normal frailty the two agree closely: every parameter to
  0.2% (relative) and the log-likelihood to 2e-3;
* with the gamma frailty frailtypack's quadrature is not exact (on R's
  readmission data its estimate of theta moves from 0.99 to 1.04 between
  ``nb.gl = 20`` and 32), so its estimates are those of an approximate
  likelihood. They are checked as such: SurPyval's maximum is at least
  the exact log-likelihood at frailtypack's estimates (so SurPyval's is
  the better maximum of the exact likelihood), that exact value is within
  0.05 of it, the reported log-likelihoods within 0.05, and the
  parameters within 3% (relative; 0.02 absolute on the coefficients).

frailtypack's gamma fit on the gap time scale fails on this fixture
(istop = 4, recorded in the R script), so there is no gap-gamma entry; on
the gap scale the log-normal one is compared.
"""

import warnings

import numpy as np
import pytest

from surpyval.recurrent import JointFrailty

from ._data import fixture, values


def _fit(time_scale, family):
    d = fixture("joint_frailty")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return JointFrailty.fit(
            d["x"],
            np.c_[d["z1"], d["z2"]],
            i=d["id"],
            c=d["c"],
            terminal=d["terminal"],
            time_scale=time_scale,
            family=family,
        )


def _reference(ref):
    v = values("r_frailtypack", ref)
    params = np.r_[
        v["recurrent_scale"],
        v["recurrent_shape"],
        v["terminal_scale"],
        v["terminal_shape"],
        v["theta"],
        v["alpha"],
        v["recurrent_coef"],
        v["terminal_coef"],
    ]
    return params, float(v["loglik"]), int(v["istop"])


@pytest.mark.parametrize(
    "time_scale, ref",
    [
        ("calendar", "frailtypack_calendar_lognormal"),
        ("gap", "frailtypack_gap_lognormal"),
    ],
)
def test_lognormal_frailty_matches_frailtypack(time_scale, ref):
    model = _fit(time_scale, "lognormal")
    params, loglik, istop = _reference(ref)
    assert istop == 1
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.params, params, rtol=2e-3)
    assert model.log_likelihood == pytest.approx(loglik, abs=2e-3)


def test_gamma_frailty_against_frailtypack():
    model = _fit("calendar", "gamma")
    params, loglik, istop = _reference("frailtypack_calendar_gamma")
    assert istop == 1
    assert model.maximum == "verified"
    at_theirs = -float(model._lik(params))
    # SurPyval's is the higher maximum of the exact likelihood ...
    assert model.log_likelihood >= at_theirs - 1e-8
    # ... and frailtypack's estimate is close to it.
    assert model.log_likelihood - at_theirs < 0.05
    assert model.log_likelihood == pytest.approx(loglik, abs=0.05)
    np.testing.assert_allclose(model.params[:6], params[:6], rtol=0.03)
    np.testing.assert_allclose(model.params[6:], params[6:], atol=0.02)
