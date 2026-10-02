"""Confidence bounds of a renewal model whose restoration parameter is on
the edge of its range (#461).

A Kijima ``q`` driven to 0, or an ARA / ARI ``rho`` to 1, has no Wald
interval: the numerical Hessian took steps across the edge, gave it a
negative variance (and the other parameters too: -10.6 for a Weibull
``alpha``), and ``param_cb`` returned nan with a warning for both. The
restoration parameter's interval is now the profile-likelihood one --
the values the likelihood-ratio test does not reject, the restricted model
refitted at each, as ``repair_test`` refits it -- which is one-sided, from
the edge; the other parameters' Wald intervals are those of the model
held on the edge.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import chi2

import surpyval as sp
from surpyval.recurrent import ARA, GeneralizedRenewal
from surpyval.tests._helpers import fresh_conformance_fit, no_warnings
from surpyval.tests.conformance.registry import CASES, fitted

X = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2])
C = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1])
I = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3])


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


@pytest.mark.parametrize(
    "fitter, name, edge", [(GeneralizedRenewal, "q", 0.0), (ARA, "rho", 1.0)]
)
def test_restoration_on_its_edge_has_a_profile_interval(fitter, name, edge):
    model = fitter.fit(X, I, C)
    assert abs(model.params[0] - edge) < 1e-6
    two = _quiet(model.param_cb, name)
    assert np.all(np.isfinite(two))
    # One-sided, from the edge.
    far = 1 if edge == 0.0 else 0
    assert two[1 - far] == edge and two[far] != edge
    # Its far end is where twice the drop of the profile log-likelihood
    # is the chi-squared(1) 95% point.
    restricted, _ = model._restricted_maximum(two[far], [model._mle[1:]])
    drop = 2 * (model.log_likelihood + restricted)
    assert drop == pytest.approx(chi2.ppf(0.95, 1), abs=1e-5)
    # A one-sided bound is the end of the two-sided bound at twice the
    # level; towards the edge it is the edge itself.
    toward, away = ("lower", "upper") if edge == 0.0 else ("upper", "lower")
    two_10 = _quiet(model.param_cb, name, alpha_ci=0.1)
    assert _quiet(model.param_cb, name, bound=toward) == [edge]
    np.testing.assert_allclose(
        _quiet(model.param_cb, name, bound=away), [two_10[far]], rtol=1e-6
    )
    # A higher level is wider.
    assert abs(two[far] - edge) > abs(two_10[far] - edge)
    wide = _quiet(model.param_cb, name, alpha_ci=0.01)
    assert abs(wide[far] - edge) > abs(two[far] - edge)


def test_others_are_the_wald_bounds_of_the_model_on_the_edge():
    # With q = 0 the generalized renewal process is the renewal process:
    # its alpha and beta are the Weibull fit to the times between
    # failures, and their covariance that fit's. (The Hessian across the
    # edge gave alpha a standard error of 0.509 here, and -10.6 as a
    # variance on the conformance fixture.)
    model = GeneralizedRenewal.fit(X, I, C)
    gaps, cens = [], []
    for item in np.unique(I):
        gaps += list(np.diff(np.r_[0.0, X[I == item]]))
        cens += list(C[I == item])
    weibull = sp.Weibull.fit(np.array(gaps), c=np.array(cens))
    np.testing.assert_allclose(model.params[1:], weibull.params, rtol=1e-5)
    cov = model.covariance()
    assert np.isnan(cov[0]).all() and np.isnan(cov[:, 0]).all()
    np.testing.assert_allclose(cov[1:, 1:], weibull.hess_inv, rtol=2e-3)
    for k, name in enumerate(("alpha", "beta"), start=1):
        bounds = _quiet(model.param_cb, name)
        assert bounds[0] < model.params[k] < bounds[1]


@pytest.mark.parametrize("name", ["GeneralizedRenewal", "ARA", "ARI"])
def test_conformance_fixtures_have_bounds_for_every_parameter(name):
    # The pinned cases: q = 2.7e-16, rho = 1 - 3e-16 and rho = 1.0, whose
    # param_cb was nan with a warning for the restoration parameter and
    # for alpha.
    model = fitted([c for c in CASES if c.name == name][0])
    for parameter, value in zip(model.parameter_names, model.params):
        bounds = _quiet(model.param_cb, parameter)
        assert np.all(np.isfinite(bounds))
        assert bounds[0] <= value <= bounds[1]


def test_repr_prints_the_profile_interval_and_says_so():
    model = GeneralizedRenewal.fit(X, I, C)
    text = repr(model)
    assert "profile-likelihood one, from the" in text
    line = [r for r in text.splitlines() if r.strip().startswith("q ")][0]
    assert line.split()[-3:] == ["nan", "0", "0.09235"]
    table = model.summary()
    np.testing.assert_allclose(
        table.loc["q", ["lower 95%", "upper 95%"]].to_numpy(float),
        model.param_cb("q"),
    )


def test_interior_restoration_keeps_its_wald_interval():
    # Off the edge nothing changes: the Wald interval on the log scale,
    # from the Hessian over every parameter (repair_test's example, q =
    # 2.63).
    rng = np.random.default_rng(8)
    rows = []
    for k in range(8):
        T = rng.uniform(6000, 12000)
        N = rng.poisson(2e-4 * T**1.35)
        ts = np.sort(T * rng.random(N) ** (1 / 1.35))
        rows += [(h, k, 0) for h in ts] + [(T, k, 1)]
    x, i, c = map(np.array, zip(*rows))
    model = GeneralizedRenewal.fit(x, i, c)
    assert model.params[0] > 1  # inside its range
    cov = model.covariance()
    np.testing.assert_allclose(
        cov, numerical_hessian_inverse(model), rtol=1e-12
    )
    var = cov[0, 0]
    q = model.params[0]
    want = q * np.exp(np.array([-1, 1]) * 1.959963984540054 * np.sqrt(var) / q)
    np.testing.assert_allclose(_quiet(model.param_cb, "q"), want, rtol=1e-8)


def numerical_hessian_inverse(model):
    from surpyval.utils.linalg import numerical_hessian

    return np.linalg.inv(numerical_hessian(model._neg_ll, model._mle))


# ---------------------------------------------------------------------------
# #411: the conformance fixture's fit, ``q`` at 2.7e-16.
# ---------------------------------------------------------------------------


# -- #411: a Wald bound that does not exist warns and is nan ----------------
def test_param_cb_at_a_restoration_boundary_is_one_sided():
    # GeneralizedRenewal's fixture fit puts q at 2.7e-16, where the inverse
    # Hessian's diagonal is negative: param_cb was [nan, nan] with only
    # numpy's raw "invalid value encountered in sqrt". Since #461 q gets
    # its one-sided profile-likelihood interval from the boundary, and
    # alpha the Wald interval of the model held there, without a warning.
    model = fresh_conformance_fit("GeneralizedRenewal")
    with np.errstate(all="raise"):
        q = no_warnings(model.param_cb, "q")
        alpha = no_warnings(model.param_cb, "alpha")
    assert q[0] == 0 and 0 < q[1] < 1
    assert np.all(np.isfinite(alpha)) and alpha[0] < alpha[1]
