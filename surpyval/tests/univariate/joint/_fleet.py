"""A simulated fleet for the joint model tests (#891): the issue's pump
fleet, each pump's vibration a straight line with its own start and rate
read monthly with noise, and its hazard Weibull proportional hazards in
the line's current value."""

from __future__ import annotations

import numpy as np

from surpyval import ItemLifetime, SurpyvalData

#: The truth: vibration (mm/s) starts at 3.0 (sd 0.6) and rises 1.0 a
#: year (sd 0.5, correlation 0.2), read with noise sd 0.7; the hazard is
#: Weibull (alpha 800 days, beta 2) at a vibration of 4.0 mm/s, with 0.9
#: per mm/s.
TRUTH = {
    "a": 3.0,
    "b": 1.0 / 365,
    "sd_a": 0.6,
    "sd_b": 0.5 / 365,
    "corr_a_b": 0.2,
    "sd_noise": 0.7,
    "alpha": 800.0,
    "beta": 2.0,
    "marker_effect": 0.9,
}
CENTRE = 4.0


def _cumhaz(t, m0, m1, alpha, beta, effect, centre, lin):
    """The cumulative hazard to ``t`` of lines ``m0 + m1 t`` (vectorised,
    by 64-point Gauss-Legendre on ``v = (s/t)^beta``)."""
    x, w = np.polynomial.legendre.leggauss(64)
    v, w = (x + 1) / 2, w / 2
    u = np.power(v, 3.0)
    weight = w * 3 * v**2
    s = t[:, None] * u[None, :] ** (1 / beta)
    inner = np.sum(
        weight * np.exp(effect * (m0[:, None] + m1[:, None] * s - centre)),
        axis=1,
    )
    return np.exp(lin) * (t / alpha) ** beta * inner


def simulate(
    n=300,
    seed=0,
    every=30.0,
    follow=730.0,
    truth=None,
    centre=CENTRE,
    gamma=0.0,
    delta=0.0,
    entry=False,
):
    """``(items, lines)``: ``n`` pumps followed to ``follow`` days (or
    their failure), read every ``every`` days from their entry; with a
    0/1 covariate (half the fleet) acting on the hazard by ``gamma`` and
    on the line by ``delta`` where either is not 0; with ``entry``, half
    the pumps come under observation at a random age up to 300 days
    (left truncated), the fleet drawn until ``n`` reach it."""
    truth = dict(TRUTH if truth is None else truth)
    rng = np.random.default_rng(seed)
    sd0, sd1 = truth["sd_a"], truth["sd_b"]
    r = truth["corr_a_b"]
    D = np.array([[sd0**2, r * sd0 * sd1], [r * sd0 * sd1, sd1**2]])
    covariate = gamma != 0.0 or delta != 0.0
    items, lines = [], []
    k = 0
    while len(items) < n:
        size = 4 * n
        b = rng.multivariate_normal([0.0, 0.0], D, size=size)
        z = (
            rng.integers(0, 2, size).astype(float)
            if covariate
            else 0 * b[:, 0]
        )
        m0 = truth["a"] + b[:, 0] + delta * z
        m1 = truth["b"] + b[:, 1]
        target = rng.exponential(size=size)
        lo, hi = np.zeros(size), np.full(size, 50 * follow)
        for _ in range(80):
            mid = (lo + hi) / 2
            H = _cumhaz(
                mid,
                m0,
                m1,
                truth["alpha"],
                truth["beta"],
                truth["marker_effect"],
                centre,
                gamma * z,
            )
            lo, hi = np.where(H < target, mid, lo), np.where(
                H < target, hi, mid
            )
        life = (lo + hi) / 2
        start = (
            np.where(rng.random(size) < 0.5, rng.uniform(0, 300, size), 0.0)
            if entry
            else np.zeros(size)
        )
        for j in range(size):
            if len(items) == n:
                break
            if life[j] <= start[j]:
                continue  # failed before it came under observation
            end = min(life[j], follow + start[j])
            ages = np.arange(start[j], end + 1e-9, every)
            readings = (
                m0[j]
                + m1[j] * ages
                + rng.normal(0, truth["sd_noise"], ages.size)
            )
            outcome = "observed" if life[j] <= end else "right_censored"
            items.append(
                ItemLifetime(
                    k,
                    **{outcome: float(end)},
                    left_truncation=float(start[j]) if start[j] > 0 else None,
                    covariates={"z": z[j]} if covariate else None,
                    markers=dict(zip(ages.tolist(), readings.tolist())),
                )
            )
            lines.append((m0[j], m1[j]))
            k += 1
    return items, np.array(lines)


def fleet(n=300, seed=0, **kwargs):
    """The fleet as a SurpyvalData."""
    items, _ = simulate(n=n, seed=seed, **kwargs)
    return SurpyvalData.from_items(items)


def simulate_path(
    path_model,
    mean,
    sd,
    sd_noise,
    n=300,
    seed=0,
    every=30.0,
    follow=730.0,
    alpha=800.0,
    beta=2.0,
    effect=0.9,
    centre=CENTRE,
):
    """``n`` items whose marker follows ``path_model`` with parameters
    normal about ``mean`` with standard deviations ``sd`` (0 for a common
    one, independent), read every ``every`` days with noise ``sd_noise``
    and followed to ``follow`` days, failing at a Weibull PH hazard in
    the path's current value (``effect`` per unit, baseline ``alpha``,
    ``beta`` at the marker ``centre``)."""
    rng = np.random.default_rng(seed)
    mean, sd = np.asarray(mean, float), np.asarray(sd, float)
    grid = np.linspace(0, 20 * follow, 40001)
    items = []
    for k in range(n):
        theta = mean + sd * rng.normal(size=mean.size)
        with np.errstate(all="ignore"):
            m = path_model.path(grid, *theta)
            h = (
                (beta / alpha)
                * (grid / alpha) ** (beta - 1)
                * np.exp(effect * (m - centre))
            )
        h = np.where(np.isfinite(h), h, 1e300)
        H = np.concatenate(
            [[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(grid))]
        )
        life = float(np.interp(rng.exponential(), H, grid))
        end = min(life, follow)
        ages = np.arange(0.0, end + 1e-9, every)
        readings = path_model.path(ages, *theta) + rng.normal(
            0, sd_noise, ages.size
        )
        outcome = "observed" if life <= follow else "right_censored"
        items.append(
            ItemLifetime(
                k,
                **{outcome: float(end)},
                markers=dict(zip(ages.tolist(), readings.tolist())),
            )
        )
    return SurpyvalData.from_items(items)
