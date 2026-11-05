"""The documented ``CustomDistribution`` recipes against R and scipy.

The recipes are run exactly as the documentation writes them: each test
takes the code cell that defines a recipe out of
``docs/Parametric SurPyval Modelling.rst`` and executes it, so a change to
the page is a change to what is tested.

``scripts/reference/reference_r_custom_recipes.R`` writes
``data/r_custom_recipes.json``: flexsurv 2.3.2's survival functions
(``pgengamma``, ``pgenf``) at fixed parameters, and its fits
(``flexsurvreg``, ``dist = "gengamma"`` and ``"genf"``, with and without
``sex``) to survival's ``lung`` data. The inverse Gaussian, which flexsurv
does not have, is checked against ``scipy.stats.invgauss`` and a censored
likelihood written with it and maximised by ``scipy.optimize``.

Tolerances. The survival functions are the same function at the same
parameters: ``rtol=1e-10``. The fits come from different optimisers
(flexsurv's ``optim``, BFGS at ``reltol = 1e-14``, against SurPyval's), so
the maximised log-likelihoods must agree to ``1e-6`` (largest seen 2e-10)
and each estimate to within a hundredth of R's standard error (largest
seen 7e-6 of one, flexsurv's ``P``, whose likelihood is flat); the standard
errors (R's from ``optimHess`` by differences, SurPyval's by automatic
differentiation) to ``rtol=1e-3`` (largest seen 6e-6).
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize, stats

import surpyval as surv

from ._data import fixture, reference

REPO = Path(__file__).resolve().parents[3]
PAGE = REPO / "docs" / "Parametric SurPyval Modelling.rst"

pytestmark = pytest.mark.skipif(
    not PAGE.exists(), reason="needs the repository's docs folder"
)

LOGLIK = dict(abs=1e-6)
EST_IN_SE = 0.01
SE = dict(rtol=1e-3)


def _cells():
    """The code of every ``jupyter-execute`` cell on the page."""
    lines = PAGE.read_text().splitlines()
    cells, i = [], 0
    while i < len(lines):
        if lines[i].strip() != ".. jupyter-execute::":
            i += 1
            continue
        i += 1
        while i < len(lines) and lines[i].strip().startswith(":"):
            i += 1
        body = []
        while i < len(lines) and (
            lines[i] == "" or lines[i].startswith("    ")
        ):
            body.append(lines[i][4:])
            i += 1
        cells.append("\n".join(body))
    return cells


def _recipe(definition, name):
    """Run the documented cell that holds ``def <definition>(`` and return
    the ``CustomDistribution`` it names ``name``. The cell's own fitting
    lines are cut at the first blank line after the distribution."""
    found = [c for c in _cells() if re.search(rf"def {definition}\(", c)]
    assert len(found) == 1, f"one cell defines {definition}"
    code = found[0]
    # keep the imports and definitions; stop before the first fit
    end = code.find(".fit(")
    if end != -1:
        code = code[: code.rfind("\n\n", 0, end)]
    from autograd import numpy as anp

    namespace = {"surv": surv, "np": anp, "stats": stats}
    with warnings.catch_warnings():
        # a second construction under the same name only replaces the first
        warnings.simplefilter("ignore")
        exec(compile(code, str(PAGE), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def gen_gamma():
    return _recipe("ggp_Hf", "GenGammaPrentice")


@pytest.fixture(scope="module")
def gen_f():
    return _recipe("gfp_Hf", "GenFPrentice")


@pytest.fixture(scope="module")
def inv_gauss():
    return _recipe("ig_Hf", "InvGauss")


@pytest.fixture(scope="module")
def lung():
    d = fixture("lung")
    return d["time"], d["c"], d["sex"]


# ----------------------------------------------------------------------
# The survival functions


def test_gengamma_sf_matches_flexsurv(gen_gamma):
    ref = reference("r_custom_recipes", "sf_gengamma")
    t = np.array(ref["settings"]["t"], dtype=float)
    for params, sf in zip(ref["settings"]["params"], ref["values"]["sf"]):
        assert_allclose(gen_gamma.sf(t, *params), sf, rtol=1e-10)


def test_genf_sf_matches_flexsurv(gen_f):
    ref = reference("r_custom_recipes", "sf_genf")
    t = np.array(ref["settings"]["t"], dtype=float)
    for params, sf in zip(ref["settings"]["params"], ref["values"]["sf"]):
        assert_allclose(gen_f.sf(t, *params), sf, rtol=1e-10)


def test_gengamma_special_cases(gen_gamma):
    t = np.array([0.5, 2.0, 5.0, 12.0])
    # Q = 0 is the lognormal exactly, Q = 1 the Weibull, Q = sigma the gamma
    assert_allclose(
        gen_gamma.sf(t, 1.0, 0.7, 0.0),
        stats.lognorm(0.7, scale=np.e).sf(t),
        rtol=1e-12,
    )
    assert_allclose(
        gen_gamma.sf(t, 1.0, 0.7, 1.0),
        stats.weibull_min(1 / 0.7, scale=np.e).sf(t),
        rtol=1e-12,
    )
    a = 0.7**-2
    assert_allclose(
        gen_gamma.sf(t, 1.0, 0.7, 0.7),
        stats.gamma(a, scale=np.e / a).sf(t),
        rtol=1e-10,
    )


def test_inverse_gaussian_matches_scipy(inv_gauss):
    t = np.array([0.05, 0.5, 2.0, 10.0, 60.0])
    ref = stats.invgauss(2.0 / 3.0, scale=3.0)
    assert_allclose(inv_gauss.sf(t, 2.0, 3.0), ref.sf(t), rtol=1e-12)
    # the hazard, as the page checks it: the density far in the tail
    # (survival 1e-12) is the hazard times the survival, to about 1e-6
    assert_allclose(
        inv_gauss.hf(t, 2.0, 3.0), ref.pdf(t) / ref.sf(t), rtol=1e-10
    )


# ----------------------------------------------------------------------
# The fits on lung against flexsurvreg


def _check_fit(model, ref_id, flip_last=False):
    v = reference("r_custom_recipes", ref_id)["values"]
    est = np.array(v["est"])
    if flip_last:
        # flexsurv's coefficient shifts mu; SurPyval's AFT is its negative
        est[-1] = -est[-1]
    assert -model.neg_ll() == pytest.approx(v["loglik"], **LOGLIK)
    assert np.all(np.abs(model.params - est) < EST_IN_SE * np.array(v["se"]))
    assert_allclose(model.standard_errors(), v["se"], **SE)


def test_gengamma_fit_matches_flexsurv(gen_gamma, lung):
    x, c, _ = lung
    _check_fit(gen_gamma.fit(x, c=c), "lung_gengamma")


def test_gengamma_aft_matches_flexsurv(gen_gamma, lung):
    x, c, sex = lung
    model = surv.AFT(gen_gamma).fit(x, sex, c=c)
    _check_fit(model, "lung_gengamma_sex", flip_last=True)


def test_genf_fit_matches_flexsurv(gen_f, lung):
    x, c, _ = lung
    _check_fit(gen_f.fit(x, c=c), "lung_genf")


def test_genf_aft_matches_flexsurv(gen_f, lung):
    x, c, sex = lung
    model = surv.AFT(gen_f).fit(x, sex, c=c)
    _check_fit(model, "lung_genf_sex", flip_last=True)


def test_gengamma_fixed_q_is_weibull_and_lognormal(gen_gamma, lung):
    x, c, _ = lung
    q1 = gen_gamma.fit(x, c=c, fixed={"Q": 1})
    weibull = surv.Weibull.fit(x, c=c)
    assert -q1.neg_ll() == pytest.approx(-weibull.neg_ll(), abs=1e-6)
    assert_allclose(
        [np.exp(q1.params[0]), 1 / q1.params[1]], weibull.params, rtol=1e-5
    )
    q0 = gen_gamma.fit(x, c=c, fixed={"Q": 0})
    lognormal = surv.LogNormal.fit(x, c=c)
    assert -q0.neg_ll() == pytest.approx(-lognormal.neg_ll(), abs=1e-6)
    assert_allclose(q0.params[:2], lognormal.params, rtol=1e-5)


# ----------------------------------------------------------------------
# Every censoring type, against a likelihood written with scipy


def _stacy(mu, sigma, Q):
    a, k = Q**-2, Q / sigma
    return stats.gengamma(a, k, scale=np.exp(mu) * a ** (-1 / k))


def test_gengamma_fits_every_censoring_type(gen_gamma):
    rng = np.random.default_rng(3)
    t = _stacy(2.0, 0.5, 0.5).rvs(400, random_state=rng)
    t = t[t > 1]
    x = np.c_[t, t]
    c = np.zeros(t.size, dtype=int)
    x[t < 3], c[t < 3] = 3.0, -1
    mid = (t >= 3) & (t < 9)
    lo = 3 + 2 * np.floor((t[mid] - 3) / 2)
    x[mid], c[mid] = np.c_[lo, lo + 2], 2
    x[t > 15], c[t > 15] = 15.0, 1
    assert set(np.unique(c)) == {-1, 0, 1, 2}

    model = gen_gamma.fit(x, c=c, tl=1.0)

    def neg_ll(p):
        # every row is conditional on surviving to 1 (left truncation)
        d = _stacy(*p)
        lo, hi = np.where(c == -1, 1.0, x[:, 0]), x[:, 1]
        with np.errstate(divide="ignore"):
            ll = np.select(
                [c == 0, c == 1],
                [d.logpdf(x[:, 0]), d.logsf(x[:, 0])],
                np.log(d.cdf(hi) - d.cdf(lo)),
            )
        return -(ll.sum() - t.size * d.logsf(1.0))

    assert model.neg_ll() == pytest.approx(neg_ll(model.params), rel=1e-10)
    by_scipy = optimize.minimize(
        neg_ll,
        model.params + [0.05, -0.02, 0.1],
        method="Nelder-Mead",
        options=dict(xatol=1e-9, fatol=1e-12, maxiter=20000),
    )
    assert by_scipy.fun == pytest.approx(model.neg_ll(), abs=1e-6)
    assert_allclose(model.params, by_scipy.x, rtol=1e-4)


# ----------------------------------------------------------------------
# Inverse Gaussian regression, against a likelihood written with scipy


@pytest.fixture(scope="module")
def ig_data():
    rng = np.random.default_rng(11)
    Z = rng.binomial(1, 0.5, 300).astype(float)
    t = stats.invgauss(2 / 3, scale=3.0 * np.exp(-0.6 * Z)).rvs(
        random_state=rng
    )
    c = (t > 5).astype(int)
    return np.minimum(t, 5.0), Z, c


def _maximise(f, start):
    return optimize.minimize(
        f,
        start,
        method="Nelder-Mead",
        options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000),
    )


def test_inverse_gaussian_aft_matches_scipy(inv_gauss, ig_data):
    t, Z, c = ig_data
    model = surv.AFT(inv_gauss).fit(t, Z, c=c)

    def neg_ll(p):
        mu, lam, beta = p
        d = stats.invgauss(mu / lam, scale=lam * np.exp(-beta * Z))
        return -np.sum(np.where(c == 0, d.logpdf(t), d.logsf(t)))

    by_scipy = _maximise(neg_ll, [1.5, 2.5, 0.3])
    assert model.neg_ll() == pytest.approx(by_scipy.fun, rel=1e-10)
    assert_allclose(model.params, by_scipy.x, rtol=1e-5)


def test_inverse_gaussian_ph_matches_scipy(inv_gauss, ig_data):
    t, Z, c = ig_data
    model = surv.PH(inv_gauss).fit(t, Z, c=c)

    def neg_ll(p):
        mu, lam, beta = p
        d = stats.invgauss(mu / lam, scale=lam)
        log_h = d.logpdf(t) - d.logsf(t) + beta * Z
        return -np.sum(
            np.where(c == 0, log_h, 0) + d.logsf(t) * np.exp(beta * Z)
        )

    by_scipy = _maximise(neg_ll, [1.5, 2.5, 0.3])
    assert model.neg_ll() == pytest.approx(by_scipy.fun, rel=1e-10)
    assert_allclose(model.params, by_scipy.x, rtol=1e-5)
