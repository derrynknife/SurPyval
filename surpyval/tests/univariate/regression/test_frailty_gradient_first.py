"""Frailty fits take the gradient ladder first (#515).

``FrailtyFitter.fit`` ran Nelder-Mead (1800 to 4000 evaluations) and then
BFGS on a finite-difference gradient: 44.6 s for a GammaFrailty at 10 000
rows. It now runs ``optimise_ph`` on the likelihood's autograd gradient and
keeps that answer when it is a verified optimum, as the AFT and PO fits do
(#499). These tests pin that the answer is at least as good as the old
ladder's (whose values are recorded below), and that Nelder-Mead is not
called when the gradient ladder converges.
"""

from unittest import mock

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.regression.frailty import frailty_fitter


def _survey_data(seed):
    rng = np.random.default_rng(seed)
    n = 400
    Z = rng.normal(size=(n, 3))
    Z[:, 0] = rng.integers(0, 2, n)
    t = 10 * rng.weibull(1.5, n) * np.exp(-(Z @ [0.5, 0.1, -0.3]) / 1.5)
    ct = rng.uniform(0, 1.2 * np.quantile(t, 0.8), n)
    c = (ct < t).astype(int)
    x = np.maximum(np.ceil(np.minimum(t, ct) * 10) / 10, 0.1)
    groups = np.random.default_rng(seed).integers(0, 40, n)
    return dict(x=x, Z=Z, c=c, groups=groups)


def _shared_frailty_data():
    # The docstring example: thirty groups of six, gamma frailty of
    # variance 0.5.
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 6)
    u = rng.gamma(2.0, 0.5, 30)[groups]
    Z = rng.binomial(1, 0.5, (180, 1))
    H = rng.exponential(1, 180) / (u * np.exp(0.5 * Z[:, 0]))
    return dict(x=10 * H**0.5, Z=Z, groups=groups)


# The negative log-likelihood the Nelder-Mead-then-BFGS ladder reached
# (before #515). On the survey data it stopped with theta at 1e-10 and
# 1e-21 for the Weibull baseline, 0.10 nats short of the maximum at theta
# 0.02; on the others it found the maximum.
_OLD_NEG_LL = {
    ("shared", "WeibullFrailty"): 553.2587426928156,
    ("shared", "GammaFrailty"): 555.6163844533303,
    ("shared", "LogNormalFrailty"): 567.7838014631002,
    ("survey0", "WeibullFrailty"): 607.937511288359,
    ("survey0", "GammaFrailty"): 608.6993140296784,
    ("survey1", "WeibullFrailty"): 570.8795379993384,
    ("survey1", "LogNormalFrailty"): 585.9436498029434,
    ("survey2", "ExponentialFrailty"): 643.5745250742405,
}


def _data(case):
    if case == "shared":
        return _shared_frailty_data()
    return _survey_data(int(case[-1]))


@pytest.mark.parametrize("case,name", sorted(_OLD_NEG_LL))
def test_reaches_at_least_the_old_ladders_optimum(case, name):
    model = getattr(sp, name).fit(**_data(case))
    old = _OLD_NEG_LL[(case, name)]
    assert model._neg_ll <= old + 1e-6


@pytest.mark.parametrize("name", ["WeibullFrailty", "GammaFrailty"])
def test_no_nelder_mead_when_the_gradient_ladder_converges(name):
    methods = []
    real = frailty_fitter.minimize

    def recording(*args, **kwargs):
        methods.append(kwargs.get("method"))
        return real(*args, **kwargs)

    with mock.patch.object(frailty_fitter, "minimize", recording):
        model = getattr(sp, name).fit(**_shared_frailty_data())
    assert np.isfinite(model._neg_ll)
    assert "Nelder-Mead" not in methods, methods


def test_a_variance_heading_for_zero_is_carried_to_its_limit():
    # BFGS in log(theta) stops at theta ~ 1e-7 on data with no frailty,
    # with the likelihood still rising towards theta = 0; the fit carries
    # it on to where the likelihood no longer changes.
    rng = np.random.default_rng(4)
    x = rng.weibull(1.5, 300) * 10
    Z = rng.normal(size=(300, 2))
    groups = rng.integers(0, 30, 300)
    model = sp.GammaFrailty.fit(x, Z=Z, groups=groups)
    ph = sp.GammaPH.fit(x, Z)
    assert model.theta < 1e-12
    assert model._neg_ll == pytest.approx(ph._neg_ll, abs=1e-6)
