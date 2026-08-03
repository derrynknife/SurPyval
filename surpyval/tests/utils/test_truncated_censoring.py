"""Censored observations under truncation (#310).

A censored observation is only ever known to lie inside its own
truncation window -- it could not have been observed otherwise -- so its
likelihood numerator has to be the probability of that intersection. The
package used the unconditional form (``F(x)`` for left censoring,
``S(x)`` for right), which counts territory the truncation has already
ruled out. That makes the contribution exceed one, and the excess grows
without bound as the fitted distribution's mass slides out of the
window, so every likelihood had an unbounded direction and the optimiser
ran off down it and reported success at nonsense parameters.

The fix hands such rows to the likelihood as *intervals* -- ``[tl, x]``
and ``[x, tr]`` -- so the existing interval term, already a difference of
CDFs, computes the right thing. These tests pin the routing, the
resulting numerator, the boundedness, and above all that untruncated
data is untouched.
"""

import numpy as np
import pytest

from surpyval import (
    Exponential,
    Gamma,
    Gumbel,
    LogLogistic,
    LogNormal,
    Logistic,
    Normal,
    Weibull,
)
from surpyval.utils.surpyval_data import SurpyvalData

INF = np.inf


# -- routing -----------------------------------------------------------


def _mixed_data():
    # one of each: left censored with a finite tl, right censored with a
    # finite tr, right censored with no truncation, and an observation.
    return SurpyvalData(
        x=np.array([5.0, 3.0, 7.0, 4.0]),
        c=np.array([-1, 1, 1, 0]),
        n=np.ones(4, dtype=np.int64),
        tl=np.array([2.0, -INF, -INF, 2.0]),
        tr=np.array([INF, 9.0, INF, 9.0]),
    )


def test_censored_rows_with_a_finite_bound_are_recast_as_intervals():
    d = _mixed_data()
    # The left-censored row (x=5, tl=2) and the right-censored row
    # (x=3, tr=9) become intervals; the right-censored row with no right
    # truncation stays put.
    assert sorted(zip(d.x_il, d.x_ir)) == [(2.0, 5.0), (3.0, 9.0)]
    np.testing.assert_array_equal(d.x_r, [7.0])
    np.testing.assert_array_equal(d.x_l, [])
    np.testing.assert_array_equal(d.x_o, [4.0])


def test_recast_interval_is_the_correct_conditional_numerator():
    d = _mixed_data()
    model = Weibull.from_params([6.0, 2.0])

    def F(v):
        return model.ff(np.atleast_1d(v))[0]

    got = dict(zip(d.x_il, d.x_ir))
    # left censored at 5, entered at 2: P(2 < X <= 5) = F(5) - F(2)
    assert got[2.0] == 5.0
    np.testing.assert_allclose(
        F(5.0) - F(2.0), F(got[2.0]) - F(2.0), rtol=0, atol=0
    )
    # right censored at 3, right truncated at 9: P(3 < X <= 9)
    assert got[3.0] == 9.0
    np.testing.assert_allclose(
        F(9.0) - F(3.0), F(got[3.0]) - F(3.0), rtol=0, atol=0
    )


def test_right_censoring_without_truncation_keeps_the_exact_path():
    # Expressing right censoring as 1 - F(x) loses all precision once
    # F(x) rounds to one -- log_sf of -49 comes back as -inf -- and the
    # optimiser does evaluate the likelihood that far out. Untruncated
    # right-censored rows must stay on log_sf.
    d = SurpyvalData(
        x=np.array([1.0, 2.0, 3.0]),
        c=np.array([0, 1, 1]),
        n=np.ones(3, dtype=np.int64),
    )
    np.testing.assert_array_equal(d.x_r, [2.0, 3.0])
    assert d.x_il.size == 0


def test_untruncated_data_is_split_exactly_as_before():
    d = SurpyvalData(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        c=np.array([0, 1, -1, 0]),
        n=np.ones(4, dtype=np.int64),
    )
    np.testing.assert_array_equal(d.x_o, [1.0, 4.0])
    np.testing.assert_array_equal(d.x_r, [2.0])
    np.testing.assert_array_equal(d.x_l, [3.0])
    assert d.x_il.size == 0


def test_covariates_follow_their_rows_into_the_interval_bucket():
    # If a row changes bucket and its covariates do not follow, the
    # regression likelihood pairs every observation with the wrong
    # covariate row.
    Z = np.array([[10.0], [20.0], [30.0], [40.0]])
    d = SurpyvalData(
        x=np.array([5.0, 3.0, 7.0, 4.0]),
        c=np.array([-1, 1, 1, 0]),
        n=np.ones(4, dtype=np.int64),
        tl=np.array([2.0, -INF, -INF, 2.0]),
        tr=np.array([INF, 9.0, INF, 9.0]),
        Z=Z,
    )
    assert d.Z_i.shape[0] == d.x_il.shape[0]
    assert d.Z_r.shape[0] == d.x_r.shape[0]
    assert d.Z_o.shape[0] == d.x_o.shape[0]
    # The two recast rows are the ones carrying 10 and 20.
    assert sorted(d.Z_i.ravel().tolist()) == [10.0, 20.0]
    assert d.Z_r.ravel().tolist() == [30.0]
    assert d.Z_o.ravel().tolist() == [40.0]


# -- the bug itself ----------------------------------------------------


def _heavy_left_censored(seed=7, size=200):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0.0, 0.6, size)  # true mu = 0
    cut = np.quantile(base, 0.85)
    c = np.where(base < cut, -1, 0)
    x = np.where(c == -1, cut, base)
    return x, c, np.full(size, x.min() * 0.5)


def test_issue_310_lognormal_no_longer_runs_away():
    # Fitted mu was -7.81 with neg_ll = -inf and res.success True.
    x, c, tl = _heavy_left_censored()
    model = LogNormal.fit(x, c=c, tl=tl)
    assert np.isfinite(model.neg_ll())
    assert -2.0 < model.params[0] < 1.0
    assert 0.0 < model.params[1] < 3.0


def test_the_truncated_likelihood_is_bounded_above():
    # The sharp version: walking mu away from the data used to raise the
    # log-likelihood without limit. It must now fall off.
    x, c, tl = _heavy_left_censored()
    fitted = LogNormal.fit(x, c=c, tl=tl)
    best = -fitted.neg_ll()
    data = SurpyvalData(
        x=x,
        c=c,
        n=np.ones(x.size, dtype=np.int64),
        tl=tl,
        tr=np.full(x.size, INF),
    )
    for mu in (-3.0, -5.0, -8.0, -15.0, -40.0):
        ll = LogNormal._log_likelihood(data, mu, 0.6, 0.0, 0.0, 1.0)
        assert not (ll > best), f"log-likelihood at mu={mu} beats the fit"


def test_right_censoring_under_right_truncation_recovers_the_truth():
    # The combination that used to warn as contradictory (#195). It is
    # the ordinary flux-limited setup: a detector registers an event --
    # so it happened at or before ``tr`` -- but cannot resolve where in
    # the remaining window it fell, so it is censored at ``x``. The
    # event is simply in ``(x, tr]``, and the fit must be consistent.
    # 15,000 retained draws land inside 1% of the truth, which the 5%
    # tolerance below covers comfortably; 50,000 took twice as long for
    # no extra confidence.
    alpha, beta, tr_bound, cens_at = 10.0, 2.0, 14.0, 8.0
    draws = Weibull.random(60_000, alpha, beta)
    detected = draws[draws <= tr_bound][:15_000]  # right truncation

    c = (detected > cens_at).astype(int)
    x = np.where(c == 1, cens_at, detected)
    assert 0.3 < c.mean() < 0.6, "test needs substantial right censoring"

    model = Weibull.fit(
        x,
        c=c,
        n=np.ones(x.size, dtype=np.int64),
        tr=np.full(x.size, tr_bound),
    )
    np.testing.assert_allclose(model.params[0], alpha, rtol=0.05)
    np.testing.assert_allclose(model.params[1], beta, rtol=0.05)

    # And the correction earns its keep: ignoring the truncation biases
    # the scale down, because only the shorter-lived units are seen.
    naive = Weibull.fit(x, c=c, n=np.ones(x.size, dtype=np.int64))
    assert naive.params[0] < alpha * 0.95


DISTRIBUTIONS = [
    Weibull,
    LogNormal,
    Normal,
    Gamma,
    LogLogistic,
    Gumbel,
    Exponential,
    Logistic,
]


def _sample(dist, rng, size):
    if dist in (Normal, Logistic, Gumbel):
        return rng.normal(5.0, 1.0, size)
    return rng.lognormal(0.0, 0.6, size)


@pytest.mark.parametrize(
    "dist", DISTRIBUTIONS, ids=[d.name for d in DISTRIBUTIONS]
)
@pytest.mark.parametrize("side", ["left", "right", "both"])
@pytest.mark.parametrize("censoring", ["heavy_left", "heavy_right"])
def test_every_distribution_gives_a_finite_fit(dist, side, censoring):
    # 21 of these 48 combinations returned a non-finite neg_ll, with
    # parameters like Weibull alpha = 1.3e81.
    #
    # The right-censored-with-right-truncation combinations raise the
    # contradictory-data warning from ``xcnt_handler`` -- that data is
    # odd and stays flagged. What changes here is that it now yields the
    # coherent conditional P(x < X <= tr) instead of an unbounded
    # direction for the optimiser to run down.
    rng = np.random.default_rng(7)
    size = 200
    x = _sample(dist, rng, size)
    if censoring == "heavy_left":
        cut = np.quantile(x, 0.85)
        c = np.where(x < cut, -1, 0)
        x = np.where(c == -1, cut, x)
    else:
        cut = np.quantile(x, 0.12)
        c = (x > cut).astype(int)
        x = np.where(c == 1, cut, x)

    kwargs = {"c": c, "n": np.ones(size, dtype=np.int64)}
    lo = x.min() - abs(x.min()) * 0.5
    hi = x.max() + abs(x.max())
    if side in ("left", "both"):
        kwargs["tl"] = np.full(size, lo)
    if side in ("right", "both"):
        kwargs["tr"] = np.full(size, hi)

    model = dist.fit(x, **kwargs)
    assert np.isfinite(model.neg_ll())
    assert np.isfinite(np.asarray(model.params, dtype=float)).all()
    # No runaway: every fitted parameter stays within a sane magnitude of
    # the data it came from.
    assert (np.abs(np.asarray(model.params, dtype=float)) < 1e6).all()
