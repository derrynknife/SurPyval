"""The cost of likelihood-ratio bounds (#519).

A Weibull's band at 20 times on 1000 units took 20 s. Each evaluation of
the likelihood went mostly on guards that cannot change its value on
data inside the support (``Parametric._lr_lean_data``), and each time's
bound was searched for from the estimate and from the tips of the
parameters' walks. A two-parameter region's boundary is now traced once
(``Parametric._lr_trace``) and each search starts from its most extreme
traced point; the answer is taken when it is at least as far out as
every traced point, and otherwise the searches run as before.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtri as z

import surpyval as sp
import surpyval.univariate.parametric._likelihood_ratio as likelihood_ratio
import surpyval.univariate.parametric.parametric as parametric_module
from surpyval.tests._helpers import neg_ll_at
from surpyval.tests.conformance.registry import CASE_BY_NAME

CRIT_95 = z(0.975) ** 2


def _fitted(name):
    case = CASE_BY_NAME[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = case.fit(case.data())
    model._ensure_surv_data()
    return model


@pytest.mark.parametrize(
    "fitter", ["Weibull", "LogNormal", "Gamma", "Gumbel", "ExpoWeibull"]
)
def test_the_lean_likelihood_is_the_full_one_to_the_bit(fitter):
    # Every kind of observation: failures, right, left and interval
    # censored, and truncated windows.
    rng = np.random.default_rng(1)
    x = 5 + 3 * rng.weibull(2.0, 60)
    c = rng.choice([0, 1, -1], 60)
    xi = np.column_stack([x[:10], x[:10] + 1.0])
    with warnings.catch_warnings():
        # (an ExpoWeibull's three parameters on 60 points)
        warnings.simplefilter("ignore")
        model = getattr(sp, fitter).fit(
            xl=np.r_[x[10:], xi[:, 0]],
            xr=np.r_[x[10:], xi[:, 1]],
            c=np.r_[c[10:], np.full(10, 2)],
            tl=np.r_[np.zeros(50), np.full(10, 1.0)],
        )
    model._ensure_surv_data()
    assert model._lr_lean_data() is not None
    for _ in range(20):
        theta = np.asarray(model.params) * np.exp(
            0.2 * rng.normal(size=len(model.params))
        )
        assert model._lr_raw_neg_ll(theta) == neg_ll_at(model, theta)


def test_a_support_set_by_the_parameters_takes_the_full_path():
    # The Uniform's support is its parameters: no lean likelihood.
    model = sp.Uniform.fit(np.array([1.0, 2.0, 3.5, 4.0, 6.0]))
    model._ensure_surv_data()
    assert model._lr_lean_data() is None


def test_a_band_takes_fewer_evaluations(monkeypatch):
    # The registry's Weibull, ten times: 14,461 likelihood evaluations for
    # the band before, 5,968 with the traced region (and the parameters'
    # own walks, 2,675, as before).
    model = _fitted("Weibull")
    for name in model.parameter_names:
        model.param_cb(name, method="lr")
    calls = []
    raw = parametric_module.Parametric._lr_raw_neg_ll

    def counted(self, theta):
        calls.append(1)
        return raw(self, theta)

    monkeypatch.setattr(
        parametric_module.Parametric, "_lr_raw_neg_ll", counted
    )
    model.cb(np.linspace(2.0, 15.0, 10), method="lr")
    assert len(calls) < 8000


def test_a_likelihood_is_evaluated_once_per_point(monkeypatch):
    # The searches ask for the same point again (SLSQP's function and
    # gradient calls, a search's result checked after it): a quarter of
    # a band's likelihood evaluations, each O(n), were repeats. The
    # likelihoods are kept, so the band is the same to the bit.
    model = _fitted("Weibull")
    times = np.linspace(2.0, 15.0, 10)
    seen = []
    lean = likelihood_ratio._lean_neg_ll

    def recorded(dist, data, theta):
        seen.append(np.asarray(theta, dtype=float).tobytes())
        return lean(dist, data, theta)

    monkeypatch.setattr(likelihood_ratio, "_lean_neg_ll", recorded)
    band = model.cb(times, method="lr")
    assert len(seen) > 1000
    assert len(set(seen)) == len(seen)
    # The same band from a model that keeps no likelihoods

    class KeepsNothing(dict):
        def __setitem__(self, key, value):
            pass

    fresh = _fitted("Weibull")
    fresh.__dict__["_lr_nll_memo"] = (fresh.surv_data, KeepsNothing())
    np.testing.assert_array_equal(fresh.cb(times, method="lr"), band)


def test_a_bound_is_the_extreme_on_the_region_boundary():
    # The LogNormal's hazard at 0.5, far below its data: log hf there
    # falls steeply across the region, and the old search stopped where
    # the deviance was up to 1e-6 past crit, 4e-8 of the bound beyond the
    # region (and a search from a traced start, 1.2e-6). The bound is the
    # minimum of the hazard over the region's boundary, found here along
    # rays in the Wald metric.
    model = _fitted("LogNormal")
    lower = model.cb(np.array([0.5]), on="hf", method="lr")[0, 0]
    theta_hat = np.asarray(model.params, dtype=float)
    nll_hat = neg_ll_at(model, theta_hat)
    L = np.linalg.cholesky(np.asarray(model.hess_inv, dtype=float))

    def boundary(angle):
        d = L @ np.array([np.cos(angle), np.sin(angle)])

        def excess(r):
            nll = neg_ll_at(model, theta_hat + r * d)
            return 2.0 * (nll - nll_hat) - CRIT_95

        hi = 2.0
        while excess(hi) < 0:
            hi *= 2.0
        r = brentq(excess, 0.0, hi, xtol=1e-15, rtol=1e-15)
        return theta_hat + r * d

    def log_hf(angle):
        hf = model.dist.hf(np.array([0.5]), *boundary(angle))
        return float(np.log(hf[0]))

    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    k = int(np.argmin([log_hf(a) for a in angles]))
    best = minimize_scalar(
        log_hf, bracket=(angles[k - 1], angles[k], angles[k + 1]), tol=1e-12
    )
    assert lower == pytest.approx(np.exp(best.fun), rel=1e-9, abs=0)
