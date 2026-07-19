"""Stochastic-process degradation models: Wiener and Gamma.

These model the degradation increments directly as a stochastic process and
derive the failure-time distribution from the process's first passage to the
threshold. The tests check maximum-likelihood parameter recovery from
simulated processes, the internal consistency of the induced failure-time
distribution (``sf``/``ff``/``df``/``qf``/``random``/``mean``), the
cross-check of the Wiener life against a direct first-passage simulation, the
monotone-only guard on the Gamma process, and the input validation shared by
both.
"""

import numpy as np
import pytest

from surpyval.degradation import GammaProcess, WienerProcess
from surpyval.degradation.process_models import ProcessRUL


def _simulate_wiener(mu, sigma, units, npts, dt, seed):
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for u in range(units):
        t = np.arange(npts) * dt
        incr = rng.normal(mu * dt, sigma * np.sqrt(dt), size=npts - 1)
        w = np.concatenate([[0.0], np.cumsum(incr)])
        xs.append(t)
        ys.append(w)
        ids.append(np.full(npts, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def _simulate_gamma(alpha, beta, units, npts, dt, seed):
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for u in range(units):
        t = np.arange(npts) * dt
        incr = rng.gamma(alpha * dt, 1.0 / beta, size=npts - 1)
        w = np.concatenate([[0.0], np.cumsum(incr)])
        xs.append(t)
        ys.append(w)
        ids.append(np.full(npts, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


# --- Wiener -----------------------------------------------------------------


def test_wiener_recovers_parameters():
    x, y, i = _simulate_wiener(0.5, 0.3, units=60, npts=40, dt=0.5, seed=0)
    m = WienerProcess.fit(x, y, i, threshold=10.0)
    assert abs(m.mu - 0.5) < 0.05
    assert abs(m.sigma - 0.3) < 0.03
    # mean time to failure is threshold / mu
    assert np.isclose(m.mean(), 10.0 / m.mu)


def test_wiener_distribution_is_internally_consistent():
    x, y, i = _simulate_wiener(0.5, 0.3, units=60, npts=40, dt=0.5, seed=1)
    m = WienerProcess.fit(x, y, i, threshold=10.0)
    # sf + ff = 1
    t = np.array([5.0, 15.0, 30.0])
    assert np.allclose(m.sf(t) + m.ff(t), 1.0)
    # qf inverts ff
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(m.ff(m.qf(p)), p, atol=1e-4)
    # density integrates to one
    grid = np.linspace(1e-3, 100, 8000)
    assert abs(np.trapezoid(m.df(grid), grid) - 1.0) < 1e-2
    # scalar in -> scalar out
    assert np.isscalar(m.ff(10.0)) and np.isscalar(m.df(10.0))
    # random draws match the analytic mean
    assert abs(m.random(20000, random_state=3).mean() - m.mean()) < 0.5


def test_wiener_matches_direct_first_passage_simulation():
    # The Inverse-Gaussian first-passage law should agree with a fine-grid
    # Euler simulation of the process crossing the threshold.
    mu, sigma, D = 0.5, 0.3, 10.0
    m = WienerProcess.fit(
        *_simulate_wiener(mu, sigma, 80, 40, 0.5, seed=2), threshold=D
    )
    rng = np.random.default_rng(5)
    fdt = 0.005
    fp = []
    for _ in range(3000):
        w, s = 0.0, 0
        while w < D and s < 50000:
            w += rng.normal(mu * fdt, sigma * np.sqrt(fdt))
            s += 1
        fp.append(s * fdt)
    fp = np.array(fp)
    assert abs(fp.mean() - m.mean()) < 1.0
    assert abs((fp <= 15).mean() - m.ff(15.0)) < 0.03


def test_wiener_rejects_non_positive_drift():
    # a flat / decreasing signal gives mu <= 0 and a defective life
    x = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    y = np.array([0.0, -0.1, -0.2, 0.0, 0.0, -0.1])
    i = np.array([1, 1, 1, 2, 2, 2])
    with pytest.raises(ValueError, match="drift"):
        WienerProcess.fit(x, y, i, threshold=10.0)


def test_wiener_predict_rul():
    x, y, i = _simulate_wiener(0.5, 0.3, 60, 40, 0.5, seed=4)
    m = WienerProcess.fit(x, y, i, threshold=10.0)
    rul = m.predict_rul(6.0)
    assert isinstance(rul, ProcessRUL)
    lo, hi = rul.rul_interval
    assert lo < rul.rul < hi
    # remaining life over distance 4 is shorter than full life over 10
    assert rul.rul < m.mean()
    # already-failed state
    done = m.predict_rul(12.0)
    assert done.prob_already_failed == 1.0 and done.rul == 0.0


# --- Gamma ------------------------------------------------------------------


def test_gamma_recovers_parameters():
    x, y, i = _simulate_gamma(2.0, 1.0, units=80, npts=30, dt=0.5, seed=0)
    g = GammaProcess.fit(x, y, i, threshold=20.0)
    assert abs(g.alpha - 2.0) < 0.2
    assert abs(g.beta - 1.0) < 0.15


def test_gamma_distribution_is_internally_consistent():
    x, y, i = _simulate_gamma(2.0, 1.0, units=80, npts=30, dt=0.5, seed=1)
    g = GammaProcess.fit(x, y, i, threshold=20.0)
    t = np.array([4.0, 10.0, 18.0])
    assert np.allclose(g.sf(t) + g.ff(t), 1.0)
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(g.ff(g.qf(p)), p, atol=1e-4)
    assert np.isscalar(g.ff(10.0)) and np.isscalar(g.df(10.0))
    # ff is monotone increasing
    assert np.all(np.diff(g.ff(np.linspace(1, 30, 50))) >= -1e-9)
    assert abs(g.random(8000, random_state=3).mean() - g.mean()) < 0.5


def test_gamma_rejects_non_monotone_degradation():
    x = np.array([0, 1, 2], dtype=float)
    y = np.array([0.0, 5.0, 3.0])  # decreases
    i = np.array([1, 1, 1])
    with pytest.raises(ValueError, match="monotone"):
        GammaProcess.fit(x, y, i, threshold=10.0)


def test_gamma_predict_rul():
    x, y, i = _simulate_gamma(2.0, 1.0, 80, 30, 0.5, seed=4)
    g = GammaProcess.fit(x, y, i, threshold=20.0)
    rul = g.predict_rul(12.0)
    assert isinstance(rul, ProcessRUL)
    lo, hi = rul.rul_interval
    assert lo < rul.rul < hi


# --- shared input validation ------------------------------------------------


@pytest.mark.parametrize("fitter", [WienerProcess, GammaProcess])
def test_requires_at_least_one_increment(fitter):
    # every unit has a single measurement -> no increments
    with pytest.raises(ValueError, match="increment"):
        fitter.fit([1.0, 2.0], [0.5, 0.6], [1, 2], threshold=5.0)


@pytest.mark.parametrize("fitter", [WienerProcess, GammaProcess])
def test_requires_increasing_times(fitter):
    with pytest.raises(ValueError, match="increasing"):
        fitter.fit([0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1, 1, 1], threshold=5.0)


@pytest.mark.parametrize("fitter", [WienerProcess, GammaProcess])
def test_rejects_mismatched_lengths(fitter):
    with pytest.raises(ValueError, match="same length"):
        fitter.fit([0.0, 1.0], [0.0], [1, 1], threshold=5.0)
