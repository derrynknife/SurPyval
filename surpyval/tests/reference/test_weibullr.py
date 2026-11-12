"""Rank-regression fits, their pivotal bounds and the reduced bias
adjustment against R's WeibullR 1.2.4 (``r_weibullr.json``, written by
``scripts/reference/reference_r_weibullr.R``), and the pivotal bounds'
inversion against exact intervals.

What is checked, and how closely:

- The rank-regression (x on y, Benard's positions) fit agrees with
  WeibullR's to 1e-9, complete, type II censored and with suspensions,
  for the Weibull and the LogNormal.
- The RBA factors agree with WeibullR's ``rba`` to 1e-12 for r = 3 to
  30, 50, 100 and 343 (above 343 failures WeibullR returns 1, where the
  factor is in fact 0.997: SurPyval computes it). WeibullR's
  ``"mle-rba"`` fit is its ``"mle"`` fit with the Weibull shape, or the
  lognormal sdlog, multiplied by that factor for the number of failures
  (to 1e-12), and SurPyval's ``rba()`` multiplies its own fit by the
  same factor. The two maximum-likelihood fits themselves differ by
  about 1e-4 (WeibullR's complete-sample lognormal meanlog is 4.651327,
  the mean of the logs 4.651412), so they are compared to 1e-3.
- WeibullR's ``pivotal-rr`` bounds are constructed differently from
  the exact pivotal intervals used here (percentiles of the simulated
  log quantiles ``u* + b* z_p``, rotated about the fitted line, where
  Lawless 2003, section 5.1, reads those of ``(u* - z_p) / b*``). The
  tolerance fixed in advance (10% on each end of the 90% interval at
  p = 0.1 and 0.5) did not hold: on the type II design the upper bound
  on the median life is 223 here against 162. The constructions differ,
  so the bounds are recorded, not compared; both are checked to bracket
  the fitted quantile. (WeibullR's "Datum" column is the median of its
  simulation, not the fitted quantile, so it is not compared either.)
  SurPyval's coverage is checked by ``calibration/test_coverage_pivotal.py``.
- The inversion itself (:mod:`surpyval.univariate.parametric._pivotal`'s
  ``location_ends``, ``scale_ends``, ``quantile_ends`` and
  ``standardised_ends``) is checked against exact intervals: fed the
  pivots of the normal maximum-likelihood fit, whose distributions are
  known in closed form (Student's t for the mean, chi-square for the
  standard deviation, the noncentral t for a quantile, as for a normal
  tolerance bound), it gives those intervals up to the Monte Carlo error
  of 200,000 simulated samples. No published table of exact
  rank-regression pivots exists to compare with (Thoman, Bain and
  Antle's and Lawless's tables are for maximum-likelihood estimates).
"""

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.univariate.parametric import _pivotal
from surpyval.univariate.parametric._rba import rba_factor

from ._data import fixture, reference, values

SOURCE = "r_weibullr"
DESIGNS = ("complete", "type2", "mixed")
DISTS = {"weibull": sp.Weibull, "lognormal2p": sp.LogNormal}


def _data(design):
    d = fixture("rank_regression")
    if design == "complete":
        return d["x"], d["c"]
    if design == "type2":
        return d["x_type2"], d["c_type2"]
    return d["x"], d["c_mixed"]


@pytest.mark.parametrize("dist", sorted(DISTS))
@pytest.mark.parametrize("design", DESIGNS)
def test_rank_regression_fit(dist, design):
    ref = values(SOURCE, f"mrr_{dist}_xony_{design}")
    x, c = _data(design)
    model = DISTS[dist].fit(x, c=c, how="MPP", heuristic="Benard", rr="x")
    np.testing.assert_allclose(model.params, [ref["p1"], ref["p2"]], 1e-9)


@pytest.mark.parametrize("dist", sorted(DISTS))
@pytest.mark.parametrize("design", DESIGNS)
def test_pivotal_bounds_bracket_the_same_fit(dist, design):
    """Not a comparison of the bounds (see the module docstring): both
    bracket the same fitted quantile."""
    ref = values(SOURCE, f"mrr_{dist}_xony_{design}")
    x, c = _data(design)
    model = DISTS[dist].fit(x, c=c, how="MPP", heuristic="Benard", rr="x")
    cb = model.quantile_cb(
        ref["unrel"], alpha_ci=0.1, method="pivotal", random_state=1234
    )
    q = model.qf(ref["unrel"])
    assert np.all(cb[:, 0] < q) and np.all(q < cb[:, 1])
    assert np.all(ref["lower"] < q) and np.all(q < ref["upper"])


def test_rba_factors():
    ref = values(SOURCE, "rba_factors")
    r = ref["r"].astype(int)
    keep = r <= 343
    for key, family, basis in (
        ("weibull_median", "Weibull", "median"),
        ("weibull_mean", "Weibull", "mean"),
        ("lognormal", "LogNormal", "median"),
    ):
        ours = [rba_factor(int(k), family, basis) for k in r[keep]]
        np.testing.assert_allclose(ours, ref[key][keep], rtol=1e-12)
    # WeibullR's shortcut above 343 failures
    assert ref["weibull_median"][~keep][0] == 1.0
    assert rba_factor(344, "Weibull") == pytest.approx(0.99745, abs=1e-5)


@pytest.mark.parametrize("dist", sorted(DISTS))
@pytest.mark.parametrize("design", ("complete", "mixed"))
def test_mle_rba(dist, design):
    entry = reference(SOURCE, f"mle_rba_{dist}_{design}")
    ref = entry["values"]
    x, c = _data(design)
    family = "Weibull" if dist == "weibull" else "LogNormal"
    factor = rba_factor(int(np.sum(c == 0)), family)
    # WeibullR's own: the same factor on its maximum-likelihood fit
    assert ref["p1"] == ref["mle_p1"]
    assert ref["p2"] / ref["mle_p2"] == pytest.approx(factor, rel=1e-12)
    fit = DISTS[dist].fit(x, c=c)
    np.testing.assert_allclose(
        fit.params, [ref["mle_p1"], ref["mle_p2"]], rtol=1e-3
    )
    adjusted = fit.rba()
    np.testing.assert_allclose(
        adjusted.params, fit.params * [1.0, factor], rtol=1e-14
    )


# -- the inversion against exact intervals -----------------------------------
N, SIMS, A = 8, 200_000, 0.025


@pytest.fixture(scope="module")
def normal_pivots():
    rng = np.random.default_rng(20261113)
    z = rng.standard_normal((SIMS, N))
    return z.mean(axis=1), z.std(axis=1)  # the MLE: ddof = 0


def test_inversion_matches_exact_normal_intervals(normal_pivots):
    u_star, b_star = normal_pivots
    xbar, sd_mle = 10.0, 2.0
    s = sd_mle * np.sqrt(N / (N - 1))
    # The mean: Student's t
    t = stats.t.ppf(1 - A, N - 1)
    exact = [xbar - t * s / np.sqrt(N), xbar + t * s / np.sqrt(N)]
    got = _pivotal.location_ends(u_star, b_star, xbar, sd_mle, A)
    np.testing.assert_allclose(got, exact, rtol=0.01)
    # The standard deviation: n sd_mle^2 / sigma^2 is chi-square(n - 1)
    chi = stats.chi2.ppf([1 - A, A], N - 1)
    exact = sd_mle * np.sqrt(N / chi)
    np.testing.assert_allclose(
        _pivotal.scale_ends(b_star, sd_mle, A), exact, rtol=0.01
    )
    # A quantile y_p = mu + z_p sigma: sqrt(n) (xbar - y_p) / s is
    # noncentral t(n - 1, -sqrt(n) z_p)
    p = np.array([0.01, 0.1, 0.5, 0.9])
    zp = stats.norm.ppf(p)
    nct = stats.nct(N - 1, -np.sqrt(N) * zp)
    exact_lo = xbar - s * nct.ppf(1 - A) / np.sqrt(N)
    exact_hi = xbar - s * nct.ppf(A) / np.sqrt(N)
    lo, hi = _pivotal.quantile_ends(u_star, b_star, zp, xbar, sd_mle, A)
    # In units of s: the first version of this check compared each bound
    # relative to itself (rtol=0.015), which a bound near 0 (the 1%
    # quantile's lower bound is -0.60) cannot meet. Over six other seeds
    # the standardised errors had a standard deviation of about 0.01 at
    # p = 0.01 and 0.005 elsewhere, centred on 0.
    np.testing.assert_allclose((lo - exact_lo) / s, 0.0, atol=0.015)
    np.testing.assert_allclose((hi - exact_hi) / s, 0.0, atol=0.015)
    # The band on F inverts the quantile bounds: at the exact upper bound
    # on y_p, the lower bound on (y - mu) / sigma is z_p, so F's is p
    z_lo, _ = _pivotal.standardised_ends(
        u_star, b_star, exact_hi, xbar, sd_mle, A
    )
    np.testing.assert_allclose(z_lo, zp, atol=0.02)
