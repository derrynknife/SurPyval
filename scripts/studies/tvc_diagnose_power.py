"""Does ``diagnose()`` of a ``fit_tvc`` model flag a misspecified
time-varying effect, and only then?

Two simulations, each with a null scenario (the fitted model is the true
one) and a misspecified one, ``R`` replicates of ``n`` units each:

PH (``WeibullPH.fit_tvc``, the ``proportional_hazards`` check)
    Each unit has a fixed covariate ``z1 ~ N(0, 1)`` and a time-varying
    one, ``z2(t)``, 0 until a switch time ``s ~ U(0.3, 1.5)`` and 1 after
    it (as a transplant is in the Stanford heart data). The hazard is a
    Weibull's (scale 2, shape 1.5) times ``exp(0.5 z1 + b(t) z2(t))``.
    Null: ``b(t) = 0.7``. Misspecified: ``b(t) = 1.5`` before ``t = 1``
    and 0 after (an effect that wears off), which a constant coefficient
    cannot follow. Reported: how often the step test of ``z2`` (and the
    GLOBAL test) fails at ``alpha = 0.05``, and how often that of ``z1``
    does (its effect is constant in both scenarios).

AFT (``WeibullAFT.fit_tvc``, the ``common_shape`` check)
    Each unit has a group ``g`` (0 or 1, fixed) and the same ``z2(t)``;
    its accumulated age is ``psi(t) = int_0^t exp(0.3 g + 0.6 z2(u)) du``
    and its life ``psi(T) ~ Weibull(scale 2, shape k_g)``. Null: ``k_0 =
    k_1 = 1.5``. Misspecified: ``k_0 = 1.2``, ``k_1 = 2.5`` (the group
    changes the shape, not only the time scale). Reported: how often the
    test of one shape for both groups fails at ``alpha = 0.05``.

Censoring is uniform on (1, 5) in both. The checks are called through
the family's rows (``ph_aft._ph_rows`` and ``ph_aft._shape_rows``), not
the whole of ``diagnose()``, to leave out its AIC refits, which do not
bear on the question.

    python scripts/studies/tvc_diagnose_power.py [reps] [n]

Defaults: 200 replicates of n = 200 units, about ten minutes on one core.
The seed (``default_rng(20261105)`` per scenario) was fixed before the
results were seen.

Results (5 Nov 2026, surpyval dev, 200 replicates, n = 200; Monte Carlo
standard errors in brackets, ``sqrt(p (1 - p) / R)``):

========  ============  ==============  ==============
scenario  test          null            misspecified
========  ============  ==============  ==============
PH        step, z2      0.040 (0.014)   0.830 (0.027)
PH        step, GLOBAL  0.045 (0.015)   0.815 (0.027)
PH        step, z1      0.050 (0.015)   0.105 (0.022)
AFT       common shape  0.050 (0.015)   1.000 (0.000)
========  ============  ==============  ==============

Every fit gave usable tests (200 of 200 in each scenario). Under the null
each test fails at about its nominal 5%. The misspecified effect of
``z2`` is flagged in 83% of the samples (the GLOBAL test 82%), and a shape
per group in all of them. The step test of ``z1``, whose effect is
constant, fails in 10.5% of the misspecified samples, about twice its
nominal rate: a constant coefficient for ``z2`` misfits the hazard of the
units that have switched, and part of that misfit lands on ``z1``'s
step. When two covariates fail together, the per-window plot
(``d.plot('proportional_hazards')``) shows which coefficient moves.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

import surpyval as sp
from surpyval.diagnostics import ph_aft
from surpyval.diagnostics.registry import Context

ALPHA = 0.05
SCALE = 2.0


def _switches(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.3, 1.5, n)


def _rows(
    T: np.ndarray, c: np.ndarray, s: np.ndarray, fixed: np.ndarray
) -> tuple:
    """Start-stop rows: a unit that outlives its switch has two."""
    n = T.size
    one = T <= s
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), s[~one]]
    xr = np.r_[np.where(one, T, s), T[~one]]
    cc = np.r_[np.where(one, c, 1), c[~one]]
    z2 = np.r_[np.zeros(n), np.ones((~one).sum())]
    Z = np.column_stack([fixed[i], z2])
    return i, xl, xr, cc, Z


def _ph_times(rng: np.random.Generator, n: int, misspecified: bool) -> tuple:
    """Exact draws of the PH scenario by inverting the cumulative
    hazard, piece by piece."""
    beta = 1.5
    z1 = rng.normal(size=n)
    s = _switches(rng, n)
    E = rng.exponential(size=n)
    T = np.empty(n)
    for u in range(n):
        if misspecified:
            cuts = sorted({s[u], 1.0})
        else:
            cuts = [s[u]]
        edges = [0.0] + cuts + [np.inf]
        left = E[u]
        for a, b in zip(edges[:-1], edges[1:]):
            z2 = 1.0 if a >= s[u] else 0.0
            effect = (1.5 if a < 1.0 else 0.0) if misspecified else 0.7
            r = np.exp(0.5 * z1[u] + effect * z2)
            H0a = (a / SCALE) ** beta
            piece = np.inf if np.isinf(b) else r * ((b / SCALE) ** beta - H0a)
            if left <= piece:
                T[u] = SCALE * (H0a + left / r) ** (1 / beta)
                break
            left -= piece
    C = rng.uniform(1.0, 5.0, n)
    return np.minimum(T, C), (T > C).astype(int), s, z1


def _aft_times(rng: np.random.Generator, n: int, misspecified: bool) -> tuple:
    g = rng.integers(0, 2, n).astype(float)
    s = _switches(rng, n)
    k = np.where(g == 1, 2.5, 1.2) if misspecified else np.full(n, 1.5)
    age = SCALE * rng.weibull(k)
    before = np.exp(0.3 * g)
    after = np.exp(0.3 * g + 0.6)
    T = np.where(
        age <= before * s, age / before, s + (age - before * s) / after
    )
    C = rng.uniform(1.0, 5.0, n)
    return np.minimum(T, C), (T > C).astype(int), s, g


def _p(rows: list[dict], covariates: tuple) -> float:
    (row,) = (r for r in rows if r["covariates"] == covariates)
    return row["p"]


def run(reps: int, n: int) -> None:
    names = ["z1", "z2"]
    for misspecified in (False, True):
        rng = np.random.default_rng(20261105)
        hits = np.zeros(3)
        done = 0
        for _ in range(reps):
            T, c, s, z1 = _ph_times(rng, n, misspecified)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sp.WeibullPH.fit_tvc(*_rows(T, c, s, z1))
                ctx = Context(model, ALPHA, names, [0, 1])
                rows = ph_aft._ph_rows(ctx)
            p = [_p(rows, ("z2",)), _p(rows, ("z1", "z2")), _p(rows, ("z1",))]
            if np.all(np.isfinite(p)):
                hits += np.asarray(p) < ALPHA
                done += 1
        rate = hits / done
        print(
            "PH  misspecified={} ({} usable): step z2 {}, GLOBAL {}, "
            "z1 {}".format(
                misspecified,
                done,
                *(
                    f"{r:.3f} ({np.sqrt(r * (1 - r) / done):.3f})"
                    for r in rate
                ),
            )
        )
    for misspecified in (False, True):
        rng = np.random.default_rng(20261105)
        hits_aft = 0
        done = 0
        for _ in range(reps):
            T, c, s, g = _aft_times(rng, n, misspecified)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sp.WeibullAFT.fit_tvc(*_rows(T, c, s, g))
                ctx = Context(model, ALPHA, ["g", "z2"], [0, 1])
                rows = ph_aft._shape_rows(ctx)
            p_shape = _p(rows, ("g",))
            if np.isfinite(p_shape):
                hits_aft += p_shape < ALPHA
                done += 1
        r = hits_aft / done
        print(
            "AFT misspecified={} ({} usable): common shape {:.3f} "
            "({:.3f})".format(
                misspecified, done, r, np.sqrt(r * (1 - r) / done)
            )
        )


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    run(*(args + [200, 200][len(args) :]))
