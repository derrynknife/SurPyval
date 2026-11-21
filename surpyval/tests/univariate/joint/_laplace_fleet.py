"""A six-parameter path and its fleets, for the tests of the joint
model's Laplace integral: a marker that is a sum of six smooth bumps
along the age, the height of each an item's own.

The path is smooth on purpose. The likelihood integrates the hazard
along a path on fixed points in the age, which a path with a kink can
slip between: a first design, straight lines between the marker's values
at six ages, gave a likelihood with no maximum on one fleet (an item
that failed just after a knot had a parameter that only its hazard at
the failure itself saw, and no point of the integral did)."""

from __future__ import annotations

import numpy as np

from surpyval.degradation import PathModel

#: The centres (years) and width of the bumps, and the truth: each
#: item's height of each bump is an independent normal about ``MEANS``
#: with sd ``SDS``, read with noise; the hazard is Weibull proportional
#: hazards in the marker's value, quoted at a marker of ``CENTRE``.
KNOTS = np.linspace(0.0, 3.0, 6)
WIDTH = 0.6
MEANS = np.array([1.5, 1.2, 1.4, 1.5, 1.6, 2.1])
SDS = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
NOISE, ALPHA, BETA, EFFECT, CENTRE = 0.2, 2.5, 2.0, 0.8, 3.6


class KnotPath(PathModel):
    """``sum_j k_j exp(-(t - KNOTS_j)^2 / (2 WIDTH^2))``: a path of six
    parameters, linear in them (module level, to pickle)."""

    linear_in_parameters = True
    name = "Bumps"
    parameter_names = [f"k{j}" for j in range(6)]

    @staticmethod
    def basis(x):
        """Each bump at ``x``: ``x.shape + (6,)``."""
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x[..., None] - KNOTS) / WIDTH) ** 2)

    def path(self, x, *params):
        B = self.basis(x)
        out = 0.0
        for j, value in enumerate(params):
            out = out + np.asarray(value, dtype=float) * B[..., j]
        return out

    def inv_path(self, y, *params):
        return np.full(np.shape(y), np.nan)

    def jacobian(self, x, *params):
        return self.basis(x)

    def fit(self, x, y):
        coef, *_ = np.linalg.lstsq(
            self.basis(x), np.asarray(y, dtype=float), rcond=None
        )
        return coef

    def _initial_guess(self, x, y):
        return [float(np.mean(y)) / 2.5] * 6


KNOT_PATH = KnotPath()


def simulate(n, seed, follow=3.0, every=0.1, noise=NOISE, effect=EFFECT):
    """``n`` items of the bump path, read every ``every`` years (with
    noise of sd ``noise``) until they fail or reach ``follow``: the
    arrays of a fit (``x``, ``c``, ``markers``)."""
    rng = np.random.default_rng(seed)
    theta = MEANS + SDS * rng.standard_normal((n, 6))
    s = np.linspace(0, follow, 3001)
    m = theta @ KnotPath.basis(s).T
    h = (BETA / ALPHA) * (s / ALPHA) ** (BETA - 1)
    h = h * np.exp(effect * (m - CENTRE))
    steps = (h[:, 1:] + h[:, :-1]) / 2 * np.diff(s)
    H = np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1)
    target = rng.exponential(size=n)
    x, c, markers = [], [], []
    for i in range(n):
        failed = H[i, -1] > target[i]
        life = float(np.interp(target[i], H[i], s)) if failed else follow
        ages = np.arange(0.0, life + 1e-9, every)
        y = KnotPath.basis(ages) @ theta[i] + rng.normal(0, noise, ages.size)
        x.append(life)
        c.append(0 if failed else 1)
        markers.append(dict(zip(ages.tolist(), y.tolist())))
    return dict(x=np.array(x), c=np.array(c), markers=markers)
