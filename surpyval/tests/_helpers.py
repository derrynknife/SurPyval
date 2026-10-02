"""Data makers and small helpers shared by several test modules."""

import warnings

import numpy as np

import surpyval as sp
from surpyval import Turnbull


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


def sharp_drop_long_tail_data():
    # A sharp drop followed by a long flat tail makes an ordinary cubic
    # spline overshoot below zero.
    return np.array([1.0, 2.0, 3.0, 3.2, 3.4, 8.0, 12.0, 20.0])


def no_warnings(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return func(*args, **kwargs)


# The 11-row mixed-censoring example from the docs review.
TURNBULL_MIXED_CENSORING = dict(
    x=[1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12],
    c=[1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0],
    n=[1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1],
)


def fit_turnbull_quietly(**kwargs):
    with warnings.catch_warnings():
        # Some of these fits are deliberately slow to converge; the
        # warning is not what is under test.
        warnings.simplefilter("ignore", UserWarning)
        return Turnbull.fit(**kwargs)


def small_kaplan_meier():
    return sp.KaplanMeier.fit([1, 2, 3, 4, 5], c=[0, 1, 0, 0, 1])
