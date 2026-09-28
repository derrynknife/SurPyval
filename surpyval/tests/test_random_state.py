"""
One seeding rule for every SurPyval method that draws: ``random_state``
(or ``seed``) ``None`` seeds from numpy's global RNG, so ``np.random.seed``
controls the draw, while an explicit seed gives a stream that neither
depends on nor advances the global state (#361).

These methods used to build ``np.random.default_rng(None)`` -- fresh OS
entropy on every call -- so ``np.random.seed`` had no effect: a
Kaplan-Meier node in a seeded RePyability RBD gave a different answer on
every run while every parametric node was reproducible.
"""

from typing import Any, Callable

import numpy as np
import pytest

import surpyval as surv
from surpyval.degradation import DegradationAnalysis, WienerProcess
from surpyval.multivariate import Clayton, Frank, Gaussian, Gumbel
from surpyval.univariate.competing_risks.parametric.parametric_competing_risks import (  # noqa: E501
    ParametricCompetingRisks,
)


def _flat(result: Any) -> np.ndarray:
    """The drawn numbers in ``result`` as one float array."""
    if isinstance(result, tuple):
        return np.concatenate([_flat(r) for r in result])
    if hasattr(result, "samples"):
        return np.asarray(result.samples, dtype=float).ravel()
    arr = np.asarray(result)
    if arr.dtype.names:
        return np.asarray(arr["x"], dtype=float)
    return arr.astype(float).ravel()


def _km() -> Any:
    return surv.KaplanMeier.fit(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))


def _competing_risks() -> Any:
    rng = np.random.default_rng(0)
    t1 = rng.weibull(2.0, 60) * 30
    t2 = rng.weibull(1.2, 60) * 45
    x = np.minimum(t1, t2)
    e = np.where(t1 < t2, 1, 2)
    return ParametricCompetingRisks.fit(x, e)


def _degradation() -> Any:
    rng = np.random.default_rng(0)
    t = np.arange(1.0, 9.0)
    xs, ys, ids = [], [], []
    for u in range(6):
        b = rng.normal(1.0, 0.2)
        xs.append(t)
        ys.append(1 + b * t + rng.normal(0, 0.3, t.size))
        ids.append(np.full(t.size, u))
    return DegradationAnalysis.fit(
        np.concatenate(xs), np.concatenate(ys), np.concatenate(ids), 15.0
    )


def _wiener() -> Any:
    rng = np.random.default_rng(1)
    t = np.tile(np.arange(0.0, 11.0), 5)
    i = np.repeat(np.arange(5), 11)
    steps = rng.normal(1.0, 0.3, (5, 10))
    y = np.column_stack([np.zeros(5), np.cumsum(steps, axis=1)]).ravel()
    return WienerProcess.fit(t, y, i, threshold=10.0)


# Each entry draws with the given ``random_state``.
DRAWS: dict[str, Callable[[Any], Any]] = {
    "KaplanMeier.random": lambda rs: _km().random(8, random_state=rs),
    "KaplanMeier.bootstrap_cb": lambda rs: _km().bootstrap_cb(
        np.array([15.0, 35.0]), B=20, random_state=rs
    ),
    "ParametricCompetingRisks.random": lambda rs: _competing_risks().random(
        6, random_state=rs
    ),
    "Clayton.sample_uv": lambda rs: Clayton.sample_uv(5, [2.0], rs),
    "Gumbel.sample_uv": lambda rs: Gumbel.sample_uv(5, [2.0], rs),
    "Frank.sample_uv": lambda rs: Frank.sample_uv(5, [3.0], rs),
    "Gaussian.sample_uv": lambda rs: Gaussian.sample_uv(5, [0.5], rs),
    "WienerProcess.random": lambda rs: _wiener().random(6, random_state=rs),
    "DegradationAnalysis.random": lambda rs: _degradation().random(
        6, random_state=rs
    ),
    "DegradationAnalysis.induced_life": lambda rs: _degradation()
    .induced_life(n_samples=50, random_state=rs)
    .random(6, random_state=rs),
    "DegradationAnalysis.predict_rul": lambda rs: _degradation().predict_rul(
        [1.0, 2.0], [2.0, 3.1], n_samples=50, random_state=rs
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _restore_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("name", list(DRAWS))
def test_global_seed_reproduces_an_unseeded_draw(name: str) -> None:
    draw = DRAWS[name]
    np.random.seed(0)
    a = _flat(draw(None))
    np.random.seed(0)
    b = _flat(draw(None))
    np.random.seed(1)
    c = _flat(draw(None))
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)


@pytest.mark.parametrize("name", list(DRAWS))
def test_an_explicit_seed_ignores_and_keeps_the_global_state(
    name: str,
) -> None:
    draw = DRAWS[name]
    np.random.seed(1)
    a = _flat(draw(7))
    after = np.random.uniform(size=4)
    np.random.seed(1)
    untouched = np.random.uniform(size=4)
    np.random.seed(2)
    b = _flat(draw(7))
    assert np.array_equal(a, b, equal_nan=True)
    # Drawing with an explicit seed does not advance the global stream.
    assert np.array_equal(after, untouched)


def test_kaplan_meier_draw_reproducible_like_a_parametric_one() -> None:
    """The report in #361: seeding the global RNG made a Weibull draw
    reproducible but not a Kaplan-Meier one."""
    km = _km()
    w = surv.Weibull.from_params([100, 2])
    np.random.seed(0)
    a = km.random(8), w.random(3)
    np.random.seed(0)
    b = km.random(8), w.random(3)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_explicit_seed_streams_are_unchanged() -> None:
    """An explicit seed still means ``np.random.default_rng(seed)``. A
    non-parametric draw is ``qf(u)`` for one uniform per draw from it, with
    ``inf`` where the estimate never reaches ``u``."""
    km = _km()

    def expected(rng):
        q = np.ravel(km.qf(1.0 - rng.random(8)))
        return np.where(np.isnan(q), np.inf, q)

    assert np.array_equal(
        km.random(8, random_state=5), expected(np.random.default_rng(5))
    )
    gen = np.random.default_rng(3)
    same = np.random.default_rng(3)
    assert np.array_equal(km.random(8, random_state=gen), expected(same))
