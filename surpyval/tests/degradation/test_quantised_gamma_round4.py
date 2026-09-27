"""Round 4: the quantised (gauge-rounded) likelihood of ``GammaProcess``.

Readings rounded to a gauge whose step is comparable to the increments bias
the ordinary fit, even with its zero increments censored: every non-zero
increment is rounded too. ``GammaProcess.fit(..., gauge=step)`` fits the
probability that each unit's true path passes through the gauge bins of its
readings instead.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist

from surpyval.degradation import GammaProcess
from surpyval.degradation.process_models import _quantised_log_likelihood


def _gauge_data(
    step: "float | None",
    seed: int = 0,
    n_units: int = 10,
    start: "float | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gamma(2 dt, rate 4) wear read at t = 0..20 (mean increment 0.5),
    rounded to the nearest ``step``: the round-3 report's example."""
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for u in range(n_units):
        t = np.arange(0, 21.0)
        dy = rng.gamma(2.0, 1 / 4.0, 20)
        y0 = rng.uniform(0, 3) if start is None else start
        xs.append(t)
        ys.append(y0 + np.r_[0, np.cumsum(dy)])
        ids.append(np.full(21, u))
    x, y, i = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    if step is not None:
        y = np.round(y / step) * step
    return x, y, i


def _cdf(x: float, k: float, beta: float) -> float:
    return float(gammainc(k, beta * max(x, 0.0)))


# -- the likelihood itself ---------------------------------------------------


@pytest.mark.parametrize(
    "d1, d2, k1, k2, beta, gauge",
    [
        (0.0, 0.5, 1.5, 0.8, 3.0, 0.5),  # a zero increment, a shape < 1
        (1.0, 0.0, 0.6, 0.4, 2.0, 0.5),  # both shapes < 1 (singular)
        (2.0, 1.0, 6.0, 3.0, 2.5, 1.0),
    ],
)
def test_path_probability_matches_quadrature(
    d1: float, d2: float, k1: float, k2: float, beta: float, gauge: float
) -> None:
    # Three readings in the bins [0, g), [d1, d1 + g), [d1 + d2, ... + g),
    # the first level uniform over its bin. Integrating that start out
    # leaves a single integral over the middle level y1:
    #   P = 1/g int (G(y1) - G(y1 - g)) (G(d1 + d2 + g - y1) - G(d1 + d2 - y1))
    def middle(y1: float) -> float:
        into = _cdf(y1, k1, beta) - _cdf(y1 - gauge, k1, beta)
        out = _cdf(d1 + d2 + gauge - y1, k2, beta) - _cdf(
            d1 + d2 - y1, k2, beta
        )
        return into * out

    ref = np.log(quad(middle, d1, d1 + gauge, epsrel=1e-12)[0] / gauge)
    delta, i, k = np.array([d1, d2]), np.zeros(3), np.array([k1, k2])
    exact = _quantised_log_likelihood(delta, i, gauge, 16, None)(k, beta)
    fine = _quantised_log_likelihood(delta, i, gauge, 64, None)(k, beta)
    assert exact == pytest.approx(ref, abs=2e-3)
    # the grid converges as the square of the cell width
    assert abs(fine - ref) < abs(exact - ref) / 8
    assert fine == pytest.approx(ref, abs=1e-4)
    # treating the two rounded increments as independent is further off
    indep = _quantised_log_likelihood(delta, i, gauge, 1, None)(k, beta)
    assert abs(indep - ref) > 10 * abs(exact - ref)


@pytest.mark.parametrize("start", [0.0, 0.25])
def test_path_probability_from_exact_start(start: float) -> None:
    # the first level is the point ``start`` above the first bin's bottom
    d1, d2, k1, k2, beta, gauge = 0.0, 0.5, 1.5, 0.8, 3.0, 0.5

    def middle(y1: float) -> float:
        out = _cdf(d1 + d2 + gauge - y1, k2, beta) - _cdf(
            d1 + d2 - y1, k2, beta
        )
        return float(gamma_dist.pdf(y1 - start, k1, scale=1 / beta)) * out

    ref = np.log(quad(middle, d1, d1 + gauge, epsrel=1e-12)[0])
    delta, k = np.array([d1, d2]), np.array([k1, k2])
    ll = _quantised_log_likelihood(delta, np.zeros(3), gauge, 64, start)
    assert ll(k, beta) == pytest.approx(ref, abs=1e-4)


def test_independent_is_the_exact_likelihood_of_single_increments() -> None:
    # With one increment per unit there is no dependence to ignore: the
    # independent approximation and the exact recursion then agree, so the
    # independent fit equals the exact fit of every increment as its own
    # unit.
    x, y, i = _gauge_data(0.5, n_units=4)
    indep = GammaProcess.fit(
        x, y, i, 10.0, gauge=0.5, gauge_method="independent"
    )
    xs = np.column_stack([x[:-1], x[1:]])[np.diff(i) == 0].ravel()
    ys = np.column_stack([y[:-1], y[1:]])[np.diff(i) == 0].ravel()
    split = np.repeat(np.arange(len(xs) // 2), 2)
    exact = GammaProcess.fit(xs, ys, split, 10.0, gauge=0.5)
    assert exact.alpha == pytest.approx(indep.alpha, rel=1e-4)
    assert exact.beta == pytest.approx(indep.beta, rel=1e-4)


# -- bias removal -----------------------------------------------------------


def test_gauge_removes_rounding_bias_of_the_report_example() -> None:
    # the round-3 report: rounded to 0.5, the censored fit gives alpha 5.19
    # (unrounded 2.19) and a mean life of 17.5 (unrounded 21.2)
    raw = GammaProcess.fit(*_gauge_data(None, start=0.0), threshold=10.0)
    x, y, i = _gauge_data(0.5, start=0.0)
    censored = GammaProcess.fit(x, y, i, threshold=10.0)
    assert censored.alpha > 2 * raw.alpha
    for method in ("exact", "independent"):
        model = GammaProcess.fit(
            x, y, i, threshold=10.0, gauge=0.5, gauge_method=method
        )
        assert model.alpha == pytest.approx(raw.alpha, rel=0.1)
        assert model.beta == pytest.approx(raw.beta, rel=0.1)
        assert model.mean() == pytest.approx(raw.mean(), rel=0.02)


def test_gauge_is_unbiased_over_replicates() -> None:
    # Over replicate data sets with a gauge twice the mean increment, the
    # censored fit is an order of magnitude off; the quantised fits
    # average near the truth (alpha 2, beta 4, mean life 20.25).
    fits: dict[str, list[tuple[float, float]]] = {
        "censored": [],
        "exact": [],
        "independent": [],
    }
    for seed in range(6):
        x, y, i = _gauge_data(1.0, seed=100 + seed)
        m = GammaProcess.fit(x, y, i, 10.0)
        fits["censored"].append((m.alpha, float(m.mean())))
        for method in ("exact", "independent"):
            m = GammaProcess.fit(x, y, i, 10.0, gauge=1.0, gauge_method=method)
            fits[method].append((m.alpha, float(m.mean())))
    censored = np.mean(fits["censored"], axis=0)
    assert censored[0] > 8.0 and censored[1] < 15.0
    for method in ("exact", "independent"):
        alpha, life = np.mean(fits[method], axis=0)
        assert alpha == pytest.approx(2.0, rel=0.2)
        assert life == pytest.approx(20.25, rel=0.05)


def test_gauge_with_stress() -> None:
    # the censored fit also shrinks the stress coefficient (true 0.7)
    rng = np.random.default_rng(3)
    xs, ys, ids, Zs = [], [], [], []
    for u, z in enumerate(np.repeat([0.0, 1.0], 8)):
        t = np.arange(0, 21.0)
        dy = rng.gamma(2.0 * np.exp(0.7 * z), 1 / 4.0, 20)
        xs.append(t)
        ys.append(rng.uniform(0, 2) + np.r_[0, np.cumsum(dy)])
        ids.append(np.full(21, u))
        Zs.append(np.full(21, z))
    x, y_raw, i, Z = (np.concatenate(v) for v in (xs, ys, ids, Zs))
    y = np.round(y_raw / 0.5) * 0.5
    raw = GammaProcess.fit(x, y_raw, i, 10.0, Z=Z, stress_ref=[0.0])
    censored = GammaProcess.fit(x, y, i, 10.0, Z=Z, stress_ref=[0.0])
    model = GammaProcess.fit(x, y, i, 10.0, Z=Z, stress_ref=[0.0], gauge=0.5)
    assert raw.gamma is not None and censored.gamma is not None
    assert model.gamma is not None
    assert censored.gamma[0] < raw.gamma[0] - 0.1
    assert model.gamma[0] == pytest.approx(raw.gamma[0], abs=0.03)
    # (rounding costs information, so alpha is only within its larger
    # sampling error of the unrounded fit)
    assert censored.alpha > 1.3 * raw.alpha
    assert model.alpha == pytest.approx(raw.alpha, rel=0.15)
    assert model.stress_ref is not None and model.stress_ref[0] == 0.0
    # the reference stress only re-expresses alpha
    other = GammaProcess.fit(x, y, i, 10.0, Z=Z, stress_ref=[1.0], gauge=0.5)
    assert other.gamma is not None
    assert other.gamma[0] == pytest.approx(model.gamma[0], rel=1e-3)
    assert other.alpha == pytest.approx(
        model.alpha * np.exp(model.gamma[0]), rel=1e-3
    )


# -- the rounding convention and the start -----------------------------------


def test_rounding_convention_only_matters_with_exact_start() -> None:
    x, y, i = _gauge_data(0.5, start=0.0)
    near = GammaProcess.fit(x, y, i, 10.0, gauge=0.5)
    floor = GammaProcess.fit(x, y, i, 10.0, gauge=0.5, rounding="floor")
    # only the differences between bins enter the likelihood
    assert floor.alpha == pytest.approx(near.alpha, rel=1e-6)
    near_0 = GammaProcess.fit(x, y, i, 10.0, gauge=0.5, exact_start=True)
    floor_0 = GammaProcess.fit(
        x, y, i, 10.0, gauge=0.5, rounding="floor", exact_start=True
    )
    assert near_0.alpha != pytest.approx(near.alpha, rel=1e-3)
    assert floor_0.alpha != pytest.approx(near_0.alpha, rel=1e-3)


def test_exact_start_with_floored_readings_from_zero() -> None:
    # new units start at exactly zero wear and the gauge truncates
    x, y_raw, i = _gauge_data(None, seed=7, n_units=12, start=0.0)
    y = np.floor(y_raw / 0.5) * 0.5
    raw = GammaProcess.fit(x, y_raw, i, 10.0)
    censored = GammaProcess.fit(x, y, i, 10.0)
    model = GammaProcess.fit(
        x, y, i, 10.0, gauge=0.5, rounding="floor", exact_start=True
    )
    assert censored.alpha > 2 * raw.alpha
    assert model.alpha == pytest.approx(raw.alpha, rel=0.1)
    assert model.mean() == pytest.approx(raw.mean(), rel=0.03)


# -- inputs -----------------------------------------------------------------


def test_gauge_input_checks() -> None:
    x, y, i = _gauge_data(0.5)
    with pytest.raises(ValueError, match="resolution and gauge"):
        GammaProcess.fit(x, y, i, 10.0, gauge=0.5, resolution=0.5)
    for bad in (0.0, -0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="gauge must be a positive"):
            GammaProcess.fit(x, y, i, 10.0, gauge=bad)
    with pytest.raises(ValueError, match="not on the grid"):
        GammaProcess.fit(x, y, i, 10.0, gauge=0.3)
    with pytest.raises(ValueError, match="rounding must be"):
        GammaProcess.fit(x, y, i, 10.0, gauge=0.5, rounding="up")
    with pytest.raises(ValueError, match="gauge_method must be"):
        GammaProcess.fit(x, y, i, 10.0, gauge=0.5, gauge_method="fast")
    for kwargs in (
        {"rounding": "floor"},
        {"exact_start": True},
        {"gauge_method": "independent"},
    ):
        with pytest.raises(ValueError, match="only meaningful with gauge"):
            GammaProcess.fit(x, y, i, 10.0, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="decreases"):
        GammaProcess.fit(
            [0, 1, 2], [0.0, 1.0, 0.5], [1, 1, 1], 10.0, gauge=0.5
        )
    with pytest.raises(ValueError, match="every increment is zero"):
        GammaProcess.fit(
            [0, 1, 2], [1.0, 1.0, 1.0], [1, 1, 1], 10.0, gauge=0.5
        )
    with pytest.raises(ValueError, match="stress_ref is only meaningful"):
        GammaProcess.fit(x, y, i, 10.0, gauge=0.5, stress_ref=[0.0])


def test_gauge_tolerates_floating_point_readings() -> None:
    # readings such as 12.3 carry representation error; their differences
    # are snapped to whole gauge steps, so a tenth-step gauge still fits
    x, y, i = _gauge_data(None)
    tenths = np.round(y, 1)
    assert not np.allclose(
        np.diff(tenths) * 10,
        np.round(np.diff(tenths) * 10),
        rtol=0,
        atol=1e-15,
    )
    model = GammaProcess.fit(x, tenths, i, 10.0, gauge=0.1)
    scaled = GammaProcess.fit(x, np.round(y * 10), i, 100.0, gauge=1.0)
    # the same readings in tenths: alpha is unchanged, beta scales
    assert model.alpha == pytest.approx(scaled.alpha, rel=1e-4)
    assert model.beta == pytest.approx(10 * scaled.beta, rel=1e-4)


def test_units_of_different_lengths_and_single_readings() -> None:
    # the recursion advances all units together; a short unit drops out,
    # and a unit with one reading (no increment) is ignored as before
    x, y, i = _gauge_data(0.5, n_units=4)
    keep = ~((i == 1) & (x > 8)) & ~((i == 3) & (x > 0))
    model = GammaProcess.fit(x[keep], y[keep], i[keep], 10.0, gauge=0.5)
    without = ~((i == 1) & (x > 8)) & (i != 3)
    same = GammaProcess.fit(
        x[without], y[without], i[without], 10.0, gauge=0.5
    )
    assert model.alpha == pytest.approx(same.alpha, rel=1e-8)
    # the likelihood of the path of each unit is the product over units
    ys, ids = y[without], i[without]
    dy = np.diff(ys)[np.diff(ids) == 0]
    k = np.full(dy.size, 2.0)
    joint = _quantised_log_likelihood(dy, ids, 0.5, 16, None)(k, 4.0)
    parts = sum(
        _quantised_log_likelihood(
            np.diff(ys[ids == u]), ids[ids == u], 0.5, 16, None
        )(np.full((ids == u).sum() - 1, 2.0), 4.0)
        for u in np.unique(ids)
    )
    assert joint == pytest.approx(parts, rel=1e-12)
