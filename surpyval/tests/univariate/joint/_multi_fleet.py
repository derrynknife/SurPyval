"""A simulated fleet of items read with two markers (several-marker joint
model tests): each item's vibration and temperature are straight lines
with their own start and rate, the four correlated across the markers;
vibration is read monthly and temperature every two months, each with
its own noise; the hazard is Weibull proportional hazards in both
markers' current values."""

from __future__ import annotations

import numpy as np

#: The truth. The two markers' starts are correlated 0.5 across them.
TRUTH = {
    "vibration.a": 3.0,
    "vibration.b": 1.0 / 365,
    "temperature.a": 60.0,
    "temperature.b": 5.0 / 365,
    "sd_vibration.a": 0.6,
    "sd_vibration.b": 0.5 / 365,
    "sd_temperature.a": 4.0,
    "sd_temperature.b": 3.0 / 365,
    "corr": np.array(
        [
            [1.0, 0.2, 0.5, 0.0],
            [0.2, 1.0, 0.0, 0.3],
            [0.5, 0.0, 1.0, 0.1],
            [0.0, 0.3, 0.1, 1.0],
        ]
    ),
    "vibration.sd_noise": 0.5,
    "temperature.sd_noise": 2.0,
    "alpha": 800.0,
    "beta": 2.0,
    "vibration.marker_effect": 0.8,
    "temperature.marker_effect": 0.08,
}
#: The markers' values at which ``alpha`` is the baseline's.
CENTRES = (4.0, 65.0)


def covariance(truth: dict) -> np.ndarray:
    sd = np.array(
        [
            truth["sd_vibration.a"],
            truth["sd_vibration.b"],
            truth["sd_temperature.a"],
            truth["sd_temperature.b"],
        ]
    )
    return truth["corr"] * np.outer(sd, sd)


def simulate(
    n: int = 200,
    seed: int = 0,
    follow: float = 730.0,
    every: tuple = (30.0, 60.0),
    truth: dict | None = None,
    centres: tuple = CENTRES,
) -> tuple[list, list, list, np.ndarray]:
    """``(x, c, markers, b)``: ``n`` items followed to ``follow`` days or
    their failure, their readings ``[{"vibration": {age: value},
    "temperature": {...}}, ...]`` and each item's true path parameters
    ``b`` ``(n, 4)``. The failure ages are drawn by inverting the
    cumulative hazard on a one-day grid (exact for lines to well under a
    day)."""
    truth = dict(TRUTH if truth is None else truth)
    rng = np.random.default_rng(seed)
    mean = np.array(
        [
            truth["vibration.a"],
            truth["vibration.b"],
            truth["temperature.a"],
            truth["temperature.b"],
        ]
    )
    b = rng.multivariate_normal(mean, covariance(truth), size=n)
    grid = np.linspace(0.0, follow, int(follow) * 4 + 1)
    alpha, beta = truth["alpha"], truth["beta"]
    z1 = truth["vibration.marker_effect"]
    z2 = truth["temperature.marker_effect"]
    x, c, markers = [], [], []
    for i in range(n):
        m1 = b[i, 0] + b[i, 1] * grid
        m2 = b[i, 2] + b[i, 3] * grid
        h = (
            beta
            / alpha
            * (np.maximum(grid, 1e-12) / alpha) ** (beta - 1)
            * np.exp(z1 * (m1 - centres[0]) + z2 * (m2 - centres[1]))
        )
        H = np.concatenate(
            [[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(grid))]
        )
        target = rng.exponential()
        if target < H[-1]:
            life = float(np.interp(target, H, grid))
            end, flag = life, 0
        else:
            end, flag = follow, 1
        readings = {}
        for name, step, (k0, k1), sd in (
            ("vibration", every[0], (0, 1), truth["vibration.sd_noise"]),
            ("temperature", every[1], (2, 3), truth["temperature.sd_noise"]),
        ):
            ages = np.arange(0.0, end, step)
            y = b[i, k0] + b[i, k1] * ages + rng.normal(0, sd, ages.size)
            readings[name] = dict(zip(ages.tolist(), y.tolist()))
        x.append(end)
        c.append(flag)
        markers.append(readings)
    return x, c, markers, b


#: Two failure modes for :func:`simulate_causes`: bearing failures that
#: follow vibration, seal failures that follow temperature (each mode's
#: Weibull ``alpha`` and ``beta`` at the markers' :data:`CENTRES`, and its
#: effect of each marker's current value).
CAUSES = {
    "bearing": {
        "alpha": 1100.0,
        "beta": 2.0,
        "vibration.marker_effect": 0.8,
        "temperature.marker_effect": 0.0,
    },
    "seal": {
        "alpha": 1400.0,
        "beta": 1.5,
        "vibration.marker_effect": 0.0,
        "temperature.marker_effect": 0.08,
    },
}


def simulate_causes(
    n: int = 200,
    seed: int = 0,
    follow: float = 730.0,
    every: tuple = (30.0, 60.0),
    truth: dict | None = None,
    causes: dict | None = None,
    centres: tuple = CENTRES,
) -> tuple[list, list, list, list, np.ndarray]:
    """``(x, c, e, markers, b)``: :func:`simulate`'s fleet failing by the
    first of several failure modes (``causes``, :data:`CAUSES`), each
    with its own Weibull hazard in both markers' current values; ``e``
    is each failure's mode (``None`` for an item still running). Each
    mode's latent age is drawn by inverting its own cumulative hazard
    on a quarter-day grid. The generator's draws are in a fixed order:
    the path parameters, then for each item every mode's target in the
    order of ``causes``, then its readings' noise."""
    truth = dict(TRUTH if truth is None else truth)
    causes = dict(CAUSES if causes is None else causes)
    rng = np.random.default_rng(seed)
    mean = np.array(
        [
            truth["vibration.a"],
            truth["vibration.b"],
            truth["temperature.a"],
            truth["temperature.b"],
        ]
    )
    b = rng.multivariate_normal(mean, covariance(truth), size=n)
    grid = np.linspace(0.0, follow, int(follow) * 4 + 1)
    x, c, e, markers = [], [], [], []
    for i in range(n):
        m1 = b[i, 0] + b[i, 1] * grid
        m2 = b[i, 2] + b[i, 3] * grid
        end, flag, mode = follow, 1, None
        for name, cause in causes.items():
            alpha, beta = cause["alpha"], cause["beta"]
            h = (
                beta
                / alpha
                * (np.maximum(grid, 1e-12) / alpha) ** (beta - 1)
                * np.exp(
                    cause["vibration.marker_effect"] * (m1 - centres[0])
                    + cause["temperature.marker_effect"] * (m2 - centres[1])
                )
            )
            H = np.concatenate(
                [[0.0], np.cumsum(0.5 * (h[1:] + h[:-1]) * np.diff(grid))]
            )
            target = rng.exponential()
            if target < H[-1]:
                life = float(np.interp(target, H, grid))
                if life < end:
                    end, flag, mode = life, 0, name
        readings = {}
        for name, step, (k0, k1), sd in (
            ("vibration", every[0], (0, 1), truth["vibration.sd_noise"]),
            ("temperature", every[1], (2, 3), truth["temperature.sd_noise"]),
        ):
            ages = np.arange(0.0, end, step)
            y = b[i, k0] + b[i, k1] * ages + rng.normal(0, sd, ages.size)
            readings[name] = dict(zip(ages.tolist(), y.tolist()))
        x.append(end)
        c.append(flag)
        e.append(mode)
        markers.append(readings)
    return x, c, e, markers, b
