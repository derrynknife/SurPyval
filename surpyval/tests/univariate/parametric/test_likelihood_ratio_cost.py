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

import functools
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


def _windows_weibull():
    # Interval-censored, left-truncated and right-truncated rows
    x = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit(
            xl=x, xr=x + 1.0, c=np.full(8, 2), tl=0.5, tr=np.full(8, 15.0)
        )
    model._ensure_surv_data()
    return model


def test_602_windows_skip_the_guarded_functions(monkeypatch):
    # Interval and truncated windows went through the distribution's
    # wrapped ff and log_sf (``_array_inputs``, ``_support_guarded``):
    # a Weibull on six interval-censored units took 243 us an
    # evaluation, against 43 us on 1000 exact ones. The windows' data
    # part is kept and the functions are called unwrapped, as the
    # exact and censored terms are.
    model = _windows_weibull()
    lean = model._lr_lean_data()
    assert lean is not None and lean[3] is not None and lean[4] is not None
    calls = []
    dist_type = type(model.dist)
    for name in ("ff", "log_sf", "log_ff"):
        wrapped = getattr(dist_type, name)

        @functools.wraps(wrapped)
        def counted(self, x, *params, wrapped=wrapped):
            calls.append(1)
            return wrapped(self, x, *params)

        monkeypatch.setattr(dist_type, name, counted)
    theta = np.asarray(model.params) * 1.01
    model._lr_raw_neg_ll(theta)
    assert calls == []


@pytest.mark.parametrize(
    "scale",
    # Near the estimate; windows in the upper tail (F(l) > 1/2, from
    # log_sf); in the lower tail (F(r) below the smallest normal float,
    # from log_ff); and windows of no probability at all
    [1.0, 0.05, 1e3, 1e40],
)
def test_602_the_lean_windows_are_the_full_ones_to_the_bit(scale):
    model = _windows_weibull()
    alpha, beta = model.params
    for theta in ([alpha * scale, beta], [alpha * scale, 4.0 * beta]):
        theta = np.asarray(theta, dtype=float)
        lean = model._lr_raw_neg_ll(theta)
        full = neg_ll_at(model, theta)
        assert lean == full or (np.isnan(lean) and np.isnan(full))


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


# ---------------------------------------------------------------------------
# #587: a band's times share what each search learns
# ---------------------------------------------------------------------------
ISSUE_587_X = [[1, 2], [2, 3], [3, 5], 4, [4, 6], [5, 8]]


def _evaluations(model, call, monkeypatch):
    """The likelihoods ``call(model)`` evaluates."""
    seen = []
    lean = likelihood_ratio._lean_neg_ll

    def counted(dist, data, theta):
        seen.append(1)
        return lean(dist, data, theta)

    with monkeypatch.context() as patch:
        patch.setattr(likelihood_ratio, "_lean_neg_ll", counted)
        call(model)
    return len(seen)


def _small_weibull():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.Weibull.fit(ISSUE_587_X)


def test_587_a_band_time_takes_a_quarter_of_the_evaluations(monkeypatch):
    # A probability plot's band (the Weibull on six interval-censored
    # units of #477): each time's two sides took 854 likelihood
    # evaluations beyond those of the region, most of them in the check
    # beyond each answer, which SLSQP started at the answer and crawled
    # from to the target. It now starts beside the answer, the searches
    # run in the coordinates scaled by the Wald standard errors, and a
    # time starts from its neighbour's answer: 222.
    times = np.linspace(1.0, 12.0, 41)
    one = _evaluations(
        _small_weibull(), lambda m: m.cb(times[:1], method="lr"), monkeypatch
    )
    band = _evaluations(
        _small_weibull(), lambda m: m.cb(times, method="lr"), monkeypatch
    )
    per_time = (band - one) / (len(times) - 1)
    assert per_time < 400


def test_587_the_region_is_found_once_per_level(monkeypatch):
    # Every band, quantile and mean bound at a level searches the same
    # likelihood region; it was traced again for each call.
    model = _fitted("Weibull")
    traced = []
    trace = parametric_module.Parametric._lr_trace

    def counted(self, *args):
        traced.append(1)
        return trace(self, *args)

    monkeypatch.setattr(parametric_module.Parametric, "_lr_trace", counted)
    model.cb(np.array([4.0, 8.0]), method="lr")
    model.cb(np.array([13.0]), on="hf", method="lr")
    model.quantile_cb([0.1, 0.5], method="lr")
    model.mean_cb(method="lr")
    assert len(traced) == 1
    # Another level is another region
    model.cb(np.array([4.0]), method="lr", alpha_ci=0.1)
    assert len(traced) == 2


def test_587_a_band_is_each_time_alone_to_the_search_tolerance():
    # Each time's search starts from its neighbour's answer, so a band is
    # searched in time order whatever the order asked for, and each
    # bound is the one the time gives alone to the search's tolerance.
    times = np.array([9.0, 2.0, 5.5])
    band = _small_weibull().cb(times, on="ff", method="lr")
    order = np.argsort(times)
    reordered = _small_weibull().cb(times[order], on="ff", method="lr")
    np.testing.assert_array_equal(band[order], reordered)
    for i, t in enumerate(times):
        alone = _small_weibull().cb(np.array([t]), on="ff", method="lr")
        np.testing.assert_allclose(band[i], alone[0], rtol=1e-9, atol=0)


def test_587_a_band_bound_is_the_extreme_on_the_region_boundary():
    # The scaled search from the trace, on the small sample whose region
    # is far from an ellipse: each bound is the extreme of F over the
    # region's boundary, found here along rays in the Wald metric.
    model = _small_weibull()
    times = np.array([4.5, 6.25])
    band = model.cb(times, on="ff", method="lr")
    model._ensure_surv_data()
    theta_hat = np.asarray(model.params, dtype=float)
    nll_hat = neg_ll_at(model, theta_hat)
    log_hat = np.log(theta_hat)
    cov = np.asarray(model.hess_inv) / np.outer(theta_hat, theta_hat)
    L = np.linalg.cholesky(cov)

    def boundary(angle):
        d = L @ np.array([np.cos(angle), np.sin(angle)])

        def excess(r):
            nll = neg_ll_at(model, np.exp(log_hat + r * d))
            return 2.0 * (nll - nll_hat) - CRIT_95

        hi = 2.0
        while excess(hi) < 0:
            hi *= 2.0
        r = brentq(excess, 0.0, hi, xtol=1e-15, rtol=1e-15)
        return np.exp(log_hat + r * d)

    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    for t, (lower, upper) in zip(times, band):

        def ff(angle, t=t):
            return float(model.dist.ff(np.array([t]), *boundary(angle))[0])

        for sign, bound in ((1.0, lower), (-1.0, upper)):
            values = [sign * ff(a) for a in angles]
            k = int(np.argmin(values))
            best = minimize_scalar(
                lambda a: sign * ff(a),
                bracket=(angles[k - 1], angles[k], angles[k + 1]),
                tol=1e-12,
            )
            assert bound == pytest.approx(sign * best.fun, rel=1e-9, abs=0)


# ---------------------------------------------------------------------------
# #609: an extreme found again is not checked again
# ---------------------------------------------------------------------------
def test_609_an_answer_found_again_is_not_checked_again(monkeypatch):
    # Each side's searches from the estimate, the walks' tips and the
    # edge valleys mostly end on one extreme, and each answer was checked
    # (a constrained search a hair beyond it, from two starts): two thirds
    # of a NegativeBinomial band's likelihood evaluations. An answer
    # within that hair of one that has checked out is the same extreme.
    model = _fitted("Weibull")
    free = [0, 1]
    region = model._lr_region(free, CRIT_95)

    def psi(theta):
        return float(np.log(model.dist.sf(np.array([8.0]), *theta)[0]))

    search = likelihood_ratio._PsiBoundSearch(
        model, psi, free, CRIT_95, (-700.0, 700.0), region[0]
    )
    lower, _ = search.run(True, False, region[1], region[2])
    # (the answer, before it is taken onto the boundary along a ray)
    done = search.checked[-1.0]
    assert done == pytest.approx(lower, rel=1e-9, abs=0)
    at = next(u for p, u in search.reached if p == done)
    solves = []
    solve = likelihood_ratio._PsiBoundSearch.solve

    def counted(self, *args):
        solves.append(1)
        return solve(self, *args)

    monkeypatch.setattr(likelihood_ratio._PsiBoundSearch, "solve", counted)
    assert search.checks_out(-1.0, at) == done
    assert solves == []
    # One further from it than the hair (the estimate) is checked
    assert search.checks_out(-1.0, search.u_hat) is None
    assert solves == [1]


def test_609_a_valley_at_its_limit_is_probed_once(monkeypatch):
    # A NegativeBinomial's r runs to infinity along a valley whose profile
    # deviance has reached its limit (the Poisson's) by the deepest point
    # of the walk: the slice through the point before it was searched as
    # well, a fifth of the registry's bounds' time (98 s, now 80 s), for
    # answers equal to 1e-14. An ExpoWeibull's valleys, still changing
    # there, are probed as before.
    model = _fitted("NegativeBinomial")
    faces = []
    extreme_far = likelihood_ratio._PsiBoundSearch.extreme_far

    def counted(self, direction, start, level, *args, face=None, **kw):
        if face is not None:
            # (the side of the bound, and the parameter held)
            faces.append((direction, face[0]))
        return extreme_far(
            self, direction, start, level, *args, face=face, **kw
        )

    monkeypatch.setattr(
        likelihood_ratio._PsiBoundSearch, "extreme_far", counted
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, hi = model.cb(5.0, method="lr")
    assert lo < model.sf(5.0) < hi
    assert faces
    # One slice a face and side of the bound, not two
    assert len(faces) == len(set(faces))
