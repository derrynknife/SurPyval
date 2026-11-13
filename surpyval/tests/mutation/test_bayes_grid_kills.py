"""Checks that kill mutants of the Bayesian grid engine
(``bayes/engines/grid.py``) that the suite let survive (#396).

The suite checked that ``method="laplace"`` is labelled approximate, not
what it approximates with. It is the normal approximation at the
posterior mode in the search space (here ``mu`` and ``log sigma``), so
its intervals are closed forms of the mode and the curvature there.

The grid's quantiles (``lattice_quantile`` and ``_spread_quantiles``)
are checked against posteriors known apart from the engine: the normal
model's closed forms, and a brute-force grid for the log-logistic's mean,
which is infinite for part of the posterior.
"""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2, fisk, norm, t

import surpyval as sp

X = np.array([4.1, 5.3, 6.0, 4.8, 5.5, 6.7, 5.1, 4.4, 5.9, 5.0])


def test_laplace_intervals_are_the_normal_at_the_mode():
    """Kills, in ``laplace_grid``, the normal's log density edited
    (``+0.5``, ``-1.5``, ``z * 2``, ``z**3``, ``/``), the lattice made
    one-sided (``lo = +6``) and, in ``hessian``, edits of its finite
    differences: the curvature at the mode of the normal likelihood,
    with the default priors, is ``n / sigma**2`` for ``mu`` and ``2 n``
    for ``log sigma``, with no cross term, so the 95% intervals are ``mu
    +- 1.96 sigma / sqrt(n)`` and ``sigma exp(+-1.96 / sqrt(2 n))``,
    ``mu`` and ``sigma`` the mode. The grid's quadrature holds them to
    about 1e-4."""
    model = sp.Normal.fit_bayes(X, method="laplace")
    mu, sigma = model.diagnostics["mode"]
    n, z = X.size, norm.ppf(0.975)
    assert_allclose(mu, X.mean(), rtol=1e-6)
    assert_allclose(
        np.ravel(model.param_cri("mu")),
        mu + np.array([-1, 1]) * z * sigma / np.sqrt(n),
        rtol=2e-4,
    )
    assert_allclose(
        np.ravel(model.param_cri("sigma")),
        sigma * np.exp(np.array([-1, 1]) * z / np.sqrt(2 * n)),
        rtol=2e-4,
    )


# The normal model's grid posterior with the default priors (flat in
# ``mu`` and in ``log sigma``) has closed forms: ``mu`` is Student's t
# about the sample mean, scale ``s / sqrt(n)`` and ``n - 1`` degrees of
# freedom, and ``(n - 1) s**2 / sigma**2`` is chi-squared with ``n - 1``.
# The values sit between 0 and 2, across 1, so the quantile's choice of
# scale (logit only for values inside (0, 1)) matters.
XN = np.array([1.12, 0.95, 1.04, 0.88, 1.21, 0.99, 1.07, 0.93, 1.15, 1.01])
# The grid's own accuracy target: each end within 1e-3 of the interval's
# width (``grid.TOLERANCE`` for two parameters).
GRID_TOL = 1e-3


def _within_width(got, ref):
    got, ref = np.ravel(got), np.ravel(ref)
    assert np.all(np.abs(got - ref) <= GRID_TOL * (ref[1] - ref[0])), (
        got,
        ref,
    )


def _sf_quantile(x, p):
    """The posterior ``p`` quantile of ``sf(x)`` of the normal model,
    and of ``1 - sf(x)``, computed apart: ``sf(x) <= r`` when ``u = (mu
    - x) / sigma <= Phi^-1(r)``, and given ``sigma`` the posterior of
    ``mu`` is normal about the sample mean, so ``P(u <= k)`` is a
    one-dimensional integral over the chi-squared of ``sigma``."""
    n, xb, s = XN.size, XN.mean(), XN.std(ddof=1)

    def cdf_u(k):
        def f(v):
            z = k + (x - xb) * np.sqrt(v) / (s * np.sqrt(n - 1))
            return chi2.pdf(v, n - 1) * norm.cdf(np.sqrt(n) * z)

        return quad(f, 0, np.inf, limit=200, epsabs=1e-13)[0]

    k = brentq(lambda k: cdf_u(k) - p, -200, 200, xtol=1e-12)
    return norm.cdf(k), norm.sf(k)


def test_normal_parameter_intervals_are_the_closed_forms():
    """Kills, in ``lattice_quantile``, the logit scale taken for values
    below 2 (``col < 2``): ``mu``'s values, about 0.8 to 1.2, then lose
    every point above 1 (the logit of a value over 1 is not a number),
    and its 97.5% point falls below its 2.5% point."""
    model = sp.Normal.fit_bayes(XN)
    n, xb, s = XN.size, XN.mean(), XN.std(ddof=1)
    _within_width(
        model.param_cri("mu"),
        xb + t.ppf([0.025, 0.975], n - 1) * s / np.sqrt(n),
    )
    _within_width(
        model.param_cri("sigma"),
        np.sqrt((n - 1) * s**2 / chi2.ppf([0.975, 0.025], n - 1)),
    )


def test_normal_reliability_intervals_against_the_integral():
    """Kills, in ``lattice_quantile``, the logit scale taken for values
    with a 0 (``col >= 0``) or a 1 (``col <= 1``) among them. Far into
    the upper tail ``sf`` underflows to 0 at some of the posterior's
    points, and early in life it rounds to 1 at some: their logits are
    infinite, the points' mass was dropped and the interval moved (to
    an upper end of 1 for a reliability of 1e-97). Mid-life the ends are
    compared directly, in the tails on the log scale (the reliability
    spans hundreds of decades there), early in life through the
    unreliability ``1 - sf``, which keeps the digits."""
    model = sp.Normal.fit_bayes(XN)
    for x in [0.8, 1.0, 1.2]:
        ref = [_sf_quantile(x, p)[0] for p in (0.025, 0.975)]
        _within_width(model.cri(x), ref)
    for x in [1.6, 2.0, 3.0]:
        ref = [_sf_quantile(x, p)[0] for p in (0.025, 0.975)]
        _within_width(np.log(model.cri(x)), np.log(ref))
    # x = 5: the reliability underflows to 0 at the posterior's lower
    # end; its upper end is about 4e-97.
    lo, hi = np.ravel(model.cri(5.0))
    ref_hi = _sf_quantile(5.0, 0.975)[0]
    assert lo == 0.0
    assert_allclose(np.log(hi), np.log(ref_hi), rtol=GRID_TOL)
    # x = 0.2 and 0.5: the reliability rounds to 1 at the posterior's
    # upper end; the unreliability's upper end (from the reliability's
    # lower end) is about 7e-6 and 3e-3.
    for x in [0.2, 0.5]:
        hi = np.ravel(model.cri(x, on="ff"))[1]
        assert_allclose(hi, _sf_quantile(x, 0.025)[1], rtol=GRID_TOL)


# The log-logistic's mean is ``alpha (pi / beta) / sin(pi / beta)`` for
# ``beta > 1`` and infinite otherwise. Eight values with a small shape
# leave the posterior (flat priors in ``log alpha`` and ``log beta``) about
# 4.7% of its mass at ``beta <= 1``.
XL = np.array(
    [2.0626, 4.5833, 25.3329, 12.4745, 2.2102, 8.3577, 9.4564, 3.3062]
)


def _loglogistic_posterior():
    """The posterior by brute force: a fine grid in ``log alpha`` and
    ``log beta`` (wide enough that its edges carry less than 1e-9),
    every point weighted by its likelihood."""
    la = np.linspace(np.log(0.01), np.log(1e4), 1201)
    lb = np.linspace(np.log(0.05), np.log(20), 1201)
    a, b = np.meshgrid(np.exp(la), np.exp(lb), indexing="ij")
    lp = fisk.logpdf(XL[:, None, None], b[None], scale=a[None]).sum(axis=0)
    w = np.exp(lp - lp.max())
    return a.ravel(), b.ravel(), w.ravel() / w.sum()


def test_mean_interval_is_infinite_where_the_mean_is():
    """Kills, in ``_spread_quantiles``, the infinite answer dropped for a
    level past the finite values' mass (the condition ``None``, or
    ``finite_mass`` times 2) and, in ``lattice_quantile``, ``None`` in
    place of that answer: the upper end of the mean's 95% interval came
    out finite (about 4e7) or not a number, though more than 2.5% of the
    posterior (4.7%, by brute force) has an infinite mean. The lower end
    is the brute-force posterior's 2.5% point, to the grid's tolerance
    (it is 2.6e-4 off): this also kills, in ``lattice_quantile``, the log
    scale given up for positive values with one at or below 1 (``col >
    1``), which leaves the means, spanning seven decades, on their own
    scale and the lower end 1.7e-3 off."""
    a, b, w = _loglogistic_posterior()
    assert w[b <= 1].sum() > 0.025
    with np.errstate(all="ignore"):
        mean = np.where(b > 1, a * (np.pi / b) / np.sin(np.pi / b), np.inf)
    order = np.argsort(mean)
    lower = mean[order][np.searchsorted(np.cumsum(w[order]), 0.025)]

    lo, hi = sp.LogLogistic.fit_bayes(XL).mean_cri()
    assert hi == np.inf
    assert_allclose(lo, lower, rtol=GRID_TOL)


def test_mean_interval_with_every_mean_infinite():
    """Kills, in ``_spread_quantiles``, the answer for no finite value at
    all edited (``np.full`` of ``None``, without a shape or without a
    fill value) or its guard moved to one value (``c.size == 1``, so no
    values at all reached the sums): with the shape
    fixed at 0.8 every mean is infinite, and so is each end of the
    interval; the fit raised instead."""
    model = sp.LogLogistic.fit_bayes(XL, priors={"beta": sp.prior.fixed(0.8)})
    assert np.all(model.mean_cri() == np.inf)
