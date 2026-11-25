"""The built-in ``Gompertz`` against R's flexsurv and closed forms.

``scripts/reference/reference_r_distributions.R`` writes, in
``data/r_distributions.json``, flexsurv 2.3.2's ``dgompertz``,
``pgompertz`` and ``qgompertz`` at fixed parameters (positive, negative
and zero shapes, and a shape of 1e-6), and its ``flexsurvreg`` fits to
survival's ``lung`` data in years: right censored, left truncated, and
with ``sex`` on the rate (proportional hazards).

Tolerances as for the generalised gamma (``test_generalized_gamma.py``):
the functions to ``rtol=1e-10``; the fits' log-likelihoods to 1e-6, the
estimates to a hundredth of R's standard error, the standard errors to
``rtol=1e-3``.
"""

import numpy as np
import pytest
from autograd import grad
from numpy.testing import assert_allclose
from scipy import integrate

import surpyval as surv
from surpyval import Gompertz as GZ

from ._data import fixture, reference

LOGLIK = dict(abs=1e-6)
EST_IN_SE = 0.01
SE = dict(rtol=1e-3)


@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    return np.asarray(d["time"], float) / 365.25, np.asarray(d["c"]), d["sex"]


def test_functions_match_flexsurv():
    ref = reference("r_distributions", "functions_gompertz")
    t = np.array(ref["settings"]["t"], dtype=float)
    p = np.array(ref["settings"]["p"], dtype=float)
    v = ref["values"]
    for k, params in enumerate(ref["settings"]["params"]):
        assert_allclose(GZ.df(t, *params), v["pdf"][k], rtol=1e-10)
        assert_allclose(GZ.log_df(t, *params), v["log_pdf"][k], rtol=1e-10)
        assert_allclose(GZ.ff(t, *params), v["cdf"][k], rtol=1e-10)
        assert_allclose(GZ.log_sf(t, *params), v["log_sf"][k], rtol=1e-10)
        q = np.array(v["quantile"][k], dtype=float)
        mine = GZ.qf(p, *params)
        # flexsurv gives NaN past the plateau, SurPyval inf
        assert np.all(np.isinf(mine[np.isnan(q)]))
        ok = ~np.isnan(q)
        assert_allclose(mine[ok], q[ok], rtol=1e-10)


def test_shape_zero_is_the_exponential():
    t = np.array([0.0, 0.5, 2.0, 10.0, 80.0])
    assert_allclose(GZ.sf(t, 0.0, 0.3), surv.Exponential.sf(t, 0.3), rtol=0)
    # a shape of 1e-12 adds H a t / 2 to the cumulative hazard
    t = t[:4]
    assert_allclose(
        GZ.Hf(t, 1e-12, 0.3), 0.3 * t * (1 + 1e-12 * t / 2), rtol=1e-15
    )
    assert GZ.mean(0.0, 0.3) == pytest.approx(1 / 0.3, rel=1e-15)


def test_derivative_in_the_shape_is_exact_through_zero():
    def log_sf(a, x):
        return GZ.log_sf(x, a, 0.3)

    for x in (0.5, 2.0, 10.0):
        # d/da of -(b/a)(e^{ax} - 1) at a = 0 is -b x^2 / 2
        assert grad(log_sf)(0.0, x) == pytest.approx(
            -0.3 * x**2 / 2, rel=1e-14
        )
        h = 1e-5
        fd = (log_sf(1e-6 + h, x) - log_sf(1e-6 - h, x)) / (2 * h)
        assert grad(log_sf)(1e-6, x) == pytest.approx(fd, rel=1e-8)


def test_negative_shape_has_a_plateau():
    a, b = -0.2, 0.1
    floor = np.exp(b / a)
    assert GZ.sf(np.inf, a, b) == pytest.approx(floor, rel=1e-15)
    assert GZ.sf(1e6, a, b) == pytest.approx(floor, rel=1e-15)
    # the quantiles stop at the fraction that ever fails
    assert np.isfinite(GZ.qf(0.99 * (1 - floor), a, b))
    assert GZ.qf(1 - floor + 1e-9, a, b) == np.inf
    assert GZ.mean(a, b) == np.inf
    draws = GZ.random(2000, a, b, random_state=1)
    assert np.mean(np.isinf(draws)) == pytest.approx(floor, abs=0.03)


@pytest.mark.parametrize(
    "params", [(0.1, 0.01), (1e-5, 0.5), (2.0, 1e-3), (50.0, 1e-3)]
)
def test_mean_and_moments_match_integration(params):
    for m in (1, 2):
        numeric = integrate.quad(
            lambda t: m * t ** (m - 1) * GZ.sf(t, *params),
            0,
            np.inf,
            limit=500,
        )[0]
        assert float(GZ.moment(m, *params)) == pytest.approx(
            numeric, rel=1e-10
        )
    assert float(GZ.mean(*params)) == pytest.approx(
        float(GZ.moment(1, *params)), rel=1e-12
    )


def test_quantiles_invert_in_both_tails():
    for a, b in ((0.1, 0.01), (2.0, 1e-3), (-0.1, 0.5)):
        p = np.array([1e-12, 1e-4, 0.3])
        assert_allclose(GZ.ff(GZ.qf(p, a, b), a, b), p, rtol=1e-12)
    s = np.array([1e-300, 1e-20, 0.4])
    assert_allclose(GZ.sf(GZ.isf(s, 0.1, 0.01), 0.1, 0.01), s, rtol=1e-12)


def _check_fit(model, ref_id):
    v = reference("r_distributions", ref_id)["values"]
    est = np.array(v["est"])
    assert -model.neg_ll() == pytest.approx(v["loglik"], **LOGLIK)
    assert np.all(np.abs(model.params - est) < EST_IN_SE * np.array(v["se"]))
    assert_allclose(model.standard_errors(), v["se"], **SE)


def test_fit_matches_flexsurv(lung):
    x, c, _ = lung
    _check_fit(GZ.fit(x, c=c), "lung_gompertz")


def test_truncated_fit_matches_flexsurv(lung):
    x, c, _ = lung
    tl = np.floor(np.asarray(fixture("lung")["time"], float) / 4) / 365.25
    _check_fit(GZ.fit(x, c=c, tl=tl), "lung_gompertz_truncated")


def test_proportional_hazards_matches_flexsurv(lung):
    # flexsurv's covariate multiplies the rate by exp(beta sex): the
    # proportional hazards model, with the same sign
    x, c, sex = lung
    _check_fit(surv.PH(GZ).fit(x, sex, c=c), "lung_gompertz_sex")


def test_simulation_recovery():
    """2000 lifetimes, censored at 40: every estimate within three standard
    errors of the truth (seed fixed before the result was seen)."""
    truth = np.array([0.08, 0.01])
    rng = np.random.default_rng(20261126)
    t = GZ.random(2000, *truth, random_state=rng)
    c = (t > 40).astype(int)
    model = GZ.fit(np.minimum(t, 40.0), c=c)
    assert np.all(np.abs(model.params - truth) < 3 * model.standard_errors())


def test_in_fit_best():
    from surpyval.fit_best import distributions

    assert GZ in distributions
