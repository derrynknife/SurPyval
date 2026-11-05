"""Checks that kill mutants of the Bayesian fits' vectorised likelihood
(``bayes/likelihood.py``) that the suite let survive (#396), against
closed forms of the Weibull likelihood (``beta = 1`` is the
exponential)."""

import numpy as np
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.bayes import log_likelihood_fn
from surpyval.bayes.likelihood import _log1mexp


def test_censoring_flags_are_used():
    """Kills, in ``log_likelihood_fn``, ``c`` dropped (``c=None`` or left
    out): the censored rows were taken as failures. With the exponential
    (``beta = 1``) the log-likelihood is ``-sum(x) / alpha - d log
    alpha``, ``d`` the failures."""
    x = np.array([2.0, 3.0, 5.0, 7.0])
    c = np.array([0, 1, 0, 1])
    ll = log_likelihood_fn(sp.Weibull, x, c=c)
    assert_allclose(ll(np.array([4.0, 1.0])), -17 / 4 - 2 * np.log(4))


def test_intervals_alone():
    """Kills, in ``LogLikelihood._observed``, a log-likelihood of 1 (not
    0) from the observed failures where there are none: data of
    intervals alone. The window ``(40, 41]`` of the unit exponential has
    probability ``exp(-40) (1 - exp(-1))`` (far in the upper tail, where
    the engine takes ``log S(l) + log(1 - S(r) / S(l))``)."""
    ll = log_likelihood_fn(sp.Weibull, xl=[40.0], xr=[41.0])
    expected = -40 + np.log1p(-np.exp(-1.0))
    assert_allclose(ll(np.array([1.0, 1.0])), expected, rtol=1e-10)


def test_log1mexp_near_zero():
    """Kills, in ``_log1mexp``, the small-argument branch dropped
    (``small = None``): ``log1p(-exp(-d))`` loses the digits of ``d``
    near 0, where ``log(-expm1(-d))`` keeps them."""
    d = np.array([1e-12, 1e-6, 1e-3])
    expected = np.log(-np.expm1(-d))
    assert_allclose(_log1mexp(d), expected, rtol=1e-13)


def test_limited_failure_window_from_the_support_bound():
    """Kills, in ``LogLikelihood._windows``, the lower end of a window
    that starts at the support's bound (``xl = 0``, not evaluated) taken
    as ``F = 1`` rather than 0 in the limited-failure-population form:
    ``P(0 < T <= 1) = p F(1)``, ``p`` the share that can fail."""
    ll = log_likelihood_fn(sp.Weibull, xl=[0.0], xr=[1.0], lfp=True)
    expected = np.log(0.5 * (1 - np.exp(-1.0)))
    assert_allclose(ll(np.array([1.0, 1.0, 0.5])), expected, rtol=1e-12)


def test_parameters_outside_their_space_have_no_likelihood():
    """Kills, in ``LogLikelihood._evaluate``, the nan of a parameter point
    outside the distribution's parameter space (a negative scale or
    shape) left as nan or made ``+inf`` rather than ``-inf``: a sampler
    or grid given ``log_likelihood_fn`` must see such points as
    impossible, not as the best."""
    ll = log_likelihood_fn(sp.Weibull, [1.0, 2.0])
    out = ll(np.array([[-1.0, 1.0], [1.0, -2.0]]))
    assert np.all(out == -np.inf)


def test_zero_inflated_windows_ending_or_starting_at_zero():
    """Kills, in ``LogLikelihood._windows``, the zero-inflated mass ``f0``
    counted only for a window end above 0 (``xr > 0``, ``xl > 0``,
    ``* (xl >= 0)`` made ``/``): the mass at 0 is in a window ``(xl, 0]``
    and below one ``(0, xr]``. With ``f0 = 0.2`` and the unit exponential,
    ``P(-1 < T <= 0) = 0.2`` and ``P(0 < T <= 1) = 0.8 (1 - exp(-1))``."""
    ll = log_likelihood_fn(sp.Weibull, xl=[-1.0, 0.0], xr=[0.0, 1.0], zi=True)
    expected = np.log(0.2) + np.log(0.8 * (1 - np.exp(-1.0)))
    assert_allclose(ll(np.array([1.0, 1.0, 0.2])), expected, rtol=1e-12)
