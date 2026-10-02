"""Data makers and small helpers shared by several test modules."""

import numpy as np


def linear_degradation_units(
    n: int = 6, seed: int = 0, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rising linear units measured at t = 1..8, crossing 15 around t = 14."""
    rng = np.random.default_rng(seed)
    t = np.arange(1.0, 9.0)
    xs, ys, ids = [], [], []
    for u in range(n):
        b = rng.normal(1.0, 0.2)
        xs.append(t)
        ys.append(1 + b * t + rng.normal(0, noise, t.size))
        ids.append(np.full(t.size, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def linear_degradation_units_with_extremes() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """The linear units plus one already past 15 at t = 1 (unit 6) and one
    below it trending away (unit 7)."""
    x, y, i = linear_degradation_units()
    rng = np.random.default_rng(1)
    t = np.arange(1.0, 9.0)
    x = np.concatenate([x, t, t])
    y = np.concatenate(
        [
            y,
            16 + t + rng.normal(0, 0.3, t.size),
            5 - 0.5 * t + rng.normal(0, 0.3, t.size),
        ]
    )
    i = np.concatenate([i, np.full(t.size, 6), np.full(t.size, 7)])
    return x, y, i


REPAIR_FLEET_X = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]


REPAIR_FLEET_I = [1] * 6 + [2] * 5


REPAIR_FLEET_C = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def competing_risks_regression_data(seed=1, n=120):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    t1 = rng.exponential(1 / (0.2 * np.exp(Z @ [0.5, -0.3])))
    t2 = rng.exponential(1 / (0.1 * np.exp(Z @ [-0.4, 0.2])))
    tc = rng.exponential(8, n)
    x = np.minimum.reduce([t1, t2, tc])
    e = np.where(tc < np.minimum(t1, t2), None, np.where(t1 < t2, 1, 2))
    return x, Z, e.astype(object)
