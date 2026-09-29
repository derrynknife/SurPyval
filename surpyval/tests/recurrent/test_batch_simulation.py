"""
The recurrent simulations advance every sequence together, one event per
round (#362). Given the same uniforms, each family's batch sampler must
draw exactly what the one-sequence-at-a-time definition of the process
draws. The references below are written independently of the samplers:
closed forms for Weibull lifetimes and Crow-AMSAA intensities, and a
scalar root find for ARI.
"""

from typing import Callable

import numpy as np
import pytest
from scipy.optimize import brentq

from surpyval import Weibull
from surpyval.recurrent import (
    ARA,
    ARI,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)
from surpyval.recurrent.renewal.renewal_model import (
    conditional_gaps,
    solve_bracketed,
)
from surpyval.recurrent.simulation import simulate_sequences

ALPHA, BETA = 10.0, 2.0


def _weibull_gap(age: float, u: float) -> float:
    """Residual life from age ``age``: H(age + x) = H(age) - log(u)."""
    target = (age / ALPHA) ** BETA - np.log(u)
    return ALPHA * target ** (1.0 / BETA) - age


def _kijima(kind: str, q: float) -> Callable:
    def sequence(us: np.ndarray) -> list:
        age, gaps = 0.0, []
        for u in us:
            gap = _weibull_gap(age, u)
            age = age + q * gap if kind == "i" else q * (age + gap)
            gaps.append(gap)
        return gaps

    return sequence


def _g1(q: float) -> Callable:
    def sequence(us: np.ndarray) -> list:
        return [
            (1 + q) ** j * ALPHA * (-np.log1p(-u)) ** (1 / BETA)
            for j, u in enumerate(us)
        ]

    return sequence


def _ara(rho: float, m: float) -> Callable:
    def sequence(us: np.ndarray) -> list:
        arrivals: list = []
        gaps = []
        for u in us:
            n = len(arrivals)
            upper = n if np.isinf(m) else min(int(m), n)
            recent = arrivals[::-1][:upper]
            weights = (1 - rho) ** np.arange(upper)
            age = (arrivals[-1] if arrivals else 0.0) - rho * float(
                np.sum(weights * np.asarray(recent))
            )
            gap = _weibull_gap(age, u)
            arrivals.append((arrivals[-1] if arrivals else 0.0) + gap)
            gaps.append(gap)
        return gaps

    return sequence


def _ari(alpha: float, beta: float, rho: float, m: float) -> Callable:
    def cif(t: float) -> float:
        return (t / alpha) ** beta

    def iif(t: float) -> float:
        return beta / alpha * (t / alpha) ** (beta - 1)

    def sequence(us: np.ndarray) -> list:
        t = 0.0
        lams: list = []
        gaps = []
        for u in us:
            n = len(lams)
            upper = n if np.isinf(m) else min(int(m), n)
            recent = np.asarray(lams[::-1][:upper])
            reduction = rho * float(
                np.sum((1 - rho) ** np.arange(upper) * recent)
            )
            energy = -np.log(u)
            start = t

            def g(x: float) -> float:
                return cif(start + x) - cif(start) - reduction * x - energy

            gap = brentq(g, 0.0, 1e6, xtol=1e-13, rtol=1e-15)
            t += gap
            lams.append(iif(t))
            gaps.append(gap)
        return gaps

    return sequence


def _cases():
    # Unannotated: mypy reads the singleton fitters as classes.
    return {
        "kijima-i": (
            GeneralizedRenewal.fit_from_parameters(
                [ALPHA, BETA], 0.5, kijima="i", dist=Weibull
            ),
            _kijima("i", 0.5),
        ),
        "kijima-ii": (
            GeneralizedRenewal.fit_from_parameters(
                [ALPHA, BETA], 0.7, kijima="ii", dist=Weibull
            ),
            _kijima("ii", 0.7),
        ),
        # q = 1: the virtual age grows without bound, so H passes the point
        # where the quantile function is used and the root finder takes over.
        "kijima-i-long": (
            GeneralizedRenewal.fit_from_parameters(
                [ALPHA, BETA], 1.0, kijima="i", dist=Weibull
            ),
            _kijima("i", 1.0),
        ),
        "g1": (
            GeneralizedOneRenewal.fit_from_parameters(
                [ALPHA, BETA], 0.2, dist=Weibull
            ),
            _g1(0.2),
        ),
        "ara-m2": (
            ARA.fit_from_parameters([ALPHA, BETA], rho=0.4, m=2, dist=Weibull),
            _ara(0.4, 2),
        ),
        "ara-inf": (
            ARA.fit_from_parameters(
                [ALPHA, BETA], rho=0.4, m=np.inf, dist=Weibull
            ),
            _ara(0.4, np.inf),
        ),
        "ari-m1": (
            ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, dist=CrowAMSAA),
            _ari(20.0, 1.5, 0.5, 1),
        ),
        "ari-inf": (
            ARI.fit_from_parameters(
                [20.0, 1.5], 0.5, m=np.inf, dist=CrowAMSAA
            ),
            _ari(20.0, 1.5, 0.5, np.inf),
        ),
    }


CASES = _cases()


@pytest.mark.parametrize("name", list(CASES))
def test_batch_sampler_matches_one_sequence_at_a_time(name: str) -> None:
    model, reference = CASES[name]
    n, rounds = 40, 30
    u = np.random.default_rng(0).uniform(size=(n, rounds))
    # Sequences drop out as they would when they pass their close, so the
    # sampler sees a shrinking set of running sequences.
    stops = np.random.default_rng(1).integers(5, rounds + 1, size=n)
    step = model._new_batch_sampler(n)
    gaps = np.full((n, rounds), np.nan)
    for k in range(rounds):
        idx = np.flatnonzero(stops > k)
        gaps[idx, k] = step(idx, u[idx, k])
    for s in range(n):
        expected = np.asarray(reference(u[s, : stops[s]]))
        # A gap comes from solving for the next age, v + x, so it carries
        # an error relative to that age (bounded by the running time), not
        # to the gap itself; about 1e-10 of it where the quantile function
        # inverts a cumulative hazard near its limit of 20.
        scale = np.cumsum(expected)
        error = np.abs(gaps[s, : stops[s]] - expected)
        assert np.all(error <= 1e-9 * scale), s


def test_intensity_sampler_inverts_the_cif():
    model = CrowAMSAA.from_params([ALPHA, BETA])
    n, rounds = 30, 20
    u = np.random.default_rng(2).uniform(size=(n, rounds))
    step = model._new_batch_sampler(n)
    times = np.cumsum(
        np.column_stack([step(np.arange(n), u[:, k]) for k in range(rounds)]),
        axis=1,
    )
    # cif((t / alpha) ** beta) rises by -log(u) from one event to the next.
    cif = (times / ALPHA) ** BETA
    rises = np.diff(np.column_stack([np.zeros(n), cif]), axis=1)
    assert np.allclose(rises, -np.log(u), rtol=1e-10)


def test_conditional_gaps_past_the_quantile_limit() -> None:
    lifetime = Weibull.from_params([ALPHA, BETA])
    ages = np.array([0.0, 10.0, 44.0, 46.0, 200.0, 1e4])
    u = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2])
    gaps = conditional_gaps(lifetime, ages, u)
    expected = np.array([_weibull_gap(a, x) for a, x in zip(ages, u)])
    # Accurate relative to the age reached, ages + gaps: to about 3e-10
    # at a cumulative hazard just under 20 (the third age), where the
    # quantile function is still used, and to round-off past it.
    reached = ages + expected
    assert np.all(np.abs(gaps - expected) <= 1e-9 * reached)
    past = ((ages / ALPHA) ** BETA - np.log(u)) > 20
    assert np.all(np.abs(gaps - expected)[past] <= 1e-14 * reached[past])


def test_solve_bracketed_converges_like_bisection_at_worst() -> None:
    # A flat-then-steep function defeats plain regula falsi.
    target = np.array([1e-3, 0.5, 20.0, 1e6])

    def g(x: np.ndarray, sel: np.ndarray) -> np.ndarray:
        return x**9 - target[sel]

    root = solve_bracketed(
        g, np.zeros(4), np.full(4, 10.0), -target, 10.0**9 - target
    )
    assert np.allclose(root, target ** (1 / 9), rtol=1e-14)


def test_simulate_sequences_close_and_count():
    model = CrowAMSAA.from_params([ALPHA, BETA])
    close = np.array([25.0, np.inf, 40.0, np.inf])
    count = np.array([0, 3, 0, 5])
    run = simulate_sequences(
        model._new_batch_sampler(4),
        4,
        np.random.default_rng(3),
        close=close,
        count=count,
    )
    for k in range(4):
        x, c = run.x[run.i == k], run.c[run.i == k]
        assert np.all(np.diff(x) > 0)
        if np.isfinite(close[k]):
            assert c[-1] == 1 and x[-1] == close[k] and np.all(c[:-1] == 0)
            assert np.all(x[:-1] <= close[k])
        else:
            assert x.size == count[k] and np.all(c == 0)
    assert not run.stalled and not run.hit_max_events


def test_a_restored_renewal_model_simulates():
    # A model rebuilt from its dict gets its sampler back from its fitter.
    from surpyval.recurrent import RenewalModel

    model = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.2)
    restored = RenewalModel.from_dict(model.to_dict())
    grid = np.array([5.0, 20.0])
    assert np.array_equal(
        restored.mcf(grid, items=50, random_state=4),
        model.mcf(grid, items=50, random_state=4),
    )
