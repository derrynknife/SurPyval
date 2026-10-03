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

import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import brentq, minimize
from scipy.special import gammainc

from surpyval.degradation import (
    GammaProcess,
    GammaProcessModel,
    WienerProcess,
    WienerProcessModel,
)
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


@pytest.mark.parametrize(
    "model",
    [
        WienerProcessModel(mu=0.5, sigma=0.8, threshold=100),
        GammaProcessModel(alpha=2.0, beta=4.0, threshold=100),
    ],
    ids=["wiener", "gamma"],
)
def test_585_process_quantiles_are_found_all_at_once(model, monkeypatch):
    # qf (and the gamma process's random) ran one brentq per probability,
    # a dozen scalar CDF evaluations each: qf of 5000 probabilities took
    # 3 s on the Wiener process. They are now solved together.
    p = np.linspace(0.001, 0.999, 2000)
    ff = model._ff_distance
    calls = []

    def counted(t, distance):
        calls.append(1)
        return ff(t, distance)

    monkeypatch.setattr(model, "_ff_distance", counted)
    q = model.qf(p)
    assert len(calls) < 200
    monkeypatch.undo()
    # the roots brentq finds, to its tolerance
    expected = [
        brentq(lambda t: ff(np.array([t]), 100.0)[0] - pk, 1e-12, 1e4)
        for pk in p[::50]
    ]
    np.testing.assert_allclose(q[::50], expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(model.ff(q), p, rtol=1e-12)
    # missing, the ends, and a quantile below the bracket's 1e-12 start
    # (brentq refused that bracket: qf(1e-300) raised on the gamma process)
    edge = model.qf([np.nan, 0.0, 1e-300, 1.0])
    assert np.isnan(edge[0]) and edge[1] == 0.0 and edge[3] == np.inf
    assert 0.0 <= edge[2] < q[0]


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


# ---------------------------------------------------------------------------
# Wiener ``sf``/``ff``/``hf`` stay finite when
# ``2 D mu / sigma**2`` is large and ``sigma = 0`` is refused;
# ``GammaProcessModel.mean`` is right when ``beta * threshold``
# is large; zero gamma increments are censored at the
# measurement resolution.
# ---------------------------------------------------------------------------


def test_wiener_life_finite_for_large_drift_to_noise() -> None:
    model = WienerProcessModel(1.0, 0.3, 35.0)
    nu, lam = 35.0, 35.0**2 / 0.09
    ref = stats.invgauss(mu=nu / lam, scale=lam)
    t = np.array([30.0, 35.0, 40.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf = model.sf(t)
        ff = model.ff(t)
        hf = np.asarray(model.hf(np.array([35.0, 1e3, 1e6])))
        Hf = np.asarray(model.Hf(np.array([40.0, 100.0])))
    assert np.allclose(sf, ref.sf(t), rtol=1e-9)
    assert np.allclose(ff, ref.cdf(t), rtol=1e-9)
    # the hazard tends to mu**2 / (2 sigma**2) far in the tail
    assert np.isfinite(hf).all()
    assert hf[-1] == pytest.approx(1.0 / (2 * 0.09), rel=1e-3)
    assert np.isfinite(Hf).all() and Hf[1] > Hf[0] > 0


def test_wiener_realistic_fit_quantiles_and_rul() -> None:
    rng = np.random.default_rng(0)
    t = np.tile(np.arange(0, 61.0), 5)
    i = np.repeat(np.arange(5), 61)
    y = np.hstack(
        [np.r_[0, np.cumsum(rng.normal(1, 0.3, 60))] for _ in range(5)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = WienerProcess.fit(t, y, i, threshold=40)
        sf = model.sf([35.0, 40.0, 45.0])
        q = model.qf([0.1, 0.5])
        rul = model.predict_rul(20.0)
    assert np.isfinite(sf).all() and np.all(np.diff(sf) < 0)
    assert np.allclose(model.ff(q), [0.1, 0.5])
    assert 15 < rul.rul < 25


def test_wiener_noise_free_is_refused() -> None:
    with pytest.raises(ValueError, match="sigma is 0"):
        WienerProcess.fit([0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1], 10.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        WienerProcessModel(1.0, 0.0, 10.0)


def test_process_predict_rul_alpha_ci_validated() -> None:
    model = WienerProcessModel(0.5, 0.4, 10.0)
    with pytest.raises(ValueError, match="alpha_ci"):
        model.predict_rul(2.0, alpha_ci=1.5)


def test_gamma_mean_with_large_beta_threshold() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = GammaProcessModel(3.0, 1500.0, 30.0).mean()
    # E[T] ~ (beta D + 1/2) / alpha for a gamma process
    assert mean == pytest.approx((1500.0 * 30.0 + 0.5) / 3.0, rel=1e-6)
    assert GammaProcessModel(3.0, 1.5, 30.0).mean() == pytest.approx(
        (45.0 + 0.5) / 3.0, rel=1e-4
    )


def _gamma_data(
    step: "float | None",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    xs, ys, ids = [], [], []
    for u in range(10):
        t = np.arange(0, 21.0)
        dy = rng.gamma(2.0, 1 / 4.0, 20)
        xs.append(t)
        ys.append(np.r_[0, np.cumsum(dy)])
        ids.append(np.full(21, u))
    x, y, i = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    if step is not None:
        y = np.round(y / step) * step
    return x, y, i


def test_gamma_zero_increments_censored_at_resolution() -> None:
    x, y, i = _gamma_data(0.2)
    dt = np.concatenate([np.diff(x[i == u]) for u in range(10)])
    dy = np.concatenate([np.diff(y[i == u]) for u in range(10)])
    zero = dy == 0
    assert zero.sum() > 0
    model = GammaProcess.fit(x, y, i, threshold=10.0)

    # the censored likelihood, maximised directly
    def nll(v: np.ndarray) -> float:
        a, b = np.exp(v)
        pos = stats.gamma.logpdf(dy[~zero], a * dt[~zero], scale=1 / b).sum()
        cens = np.log(gammainc(a * dt[zero], b * 0.2)).sum()
        return -(pos + cens)

    ref = minimize(
        nll, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-9}
    )
    assert np.allclose([model.alpha, model.beta], np.exp(ref.x), rtol=1e-4)
    # the old 1e-12 nudge cut alpha about six-fold; now it stays near the
    # fit to the unrounded data
    exact = GammaProcess.fit(*_gamma_data(None), threshold=10.0)
    assert model.alpha == pytest.approx(exact.alpha, rel=0.25)
    # an explicit resolution is used as given
    coarse = GammaProcess.fit(x, y, i, threshold=10.0, resolution=0.4)
    assert coarse.alpha != pytest.approx(model.alpha)


def test_gamma_no_zero_increments_unchanged_and_resolution_checked() -> None:
    x, y, i = _gamma_data(None)
    a = GammaProcess.fit(x, y, i, threshold=10.0)
    b = GammaProcess.fit(x, y, i, threshold=10.0, resolution=0.5)
    assert (a.alpha, a.beta) == (b.alpha, b.beta)
    with pytest.raises(ValueError, match="resolution"):
        GammaProcess.fit(*_gamma_data(0.2), threshold=10.0, resolution=-1.0)
    with pytest.raises(ValueError, match="every increment is zero"):
        GammaProcess.fit([0, 1, 2], [1.0, 1.0, 1.0], [1, 1, 1], 10.0)


def test_gamma_zero_increments_with_stress() -> None:
    rng = np.random.default_rng(3)
    xs, ys, ids, Zs = [], [], [], []
    for u, z in enumerate(np.repeat([0.0, 1.0], 8)):
        t = np.arange(0, 21.0)
        dy = rng.gamma(2.0 * np.exp(0.7 * z), 1 / 4.0, 20)
        xs.append(t)
        ys.append(np.round(np.r_[0, np.cumsum(dy)], 1))
        ids.append(np.full(21, u))
        Zs.append(np.full(21, z))
    x, y, i, Z = (np.concatenate(v) for v in (xs, ys, ids, Zs))
    model = GammaProcess.fit(x, y, i, threshold=10.0, Z=Z, stress_ref=[0.0])
    assert model.gamma is not None
    assert model.gamma[0] == pytest.approx(0.7, abs=0.2)
