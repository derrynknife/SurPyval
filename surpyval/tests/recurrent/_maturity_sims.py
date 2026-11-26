"""Simulated development tests for the maturity projections' coverage
checks (``test_maturity_projection.py`` and the calibration study
``calibration/test_coverage_maturity_projection.py``).

Each draws a finite but large population of modes (20,000), close to the
models' many-modes limit, each mode with its own fix-effectiveness factor
drawn from :data:`FACTORS` independently of its rate (mean 0.7), and
returns the test's data with the realised projected quantity of that
draw. The ``*_truth`` functions give the model's projected quantity, its
expected value over such tests, which the bounds are for.
"""

import numpy as np
from scipy import integrate

FACTORS = np.array([0.5, 0.7, 0.9])
MU = 0.7


def continuous_test(rng, k=20000, lam_b=0.2, beta=0.02, lam_a=0.02, end=300.0):
    """B modes with gamma rates (total ``lam_b``, scale ``beta``) and A
    failures at ``lam_a``, one system tested to ``end``: ``x``, ``modes``,
    ``c``, the factors of the B modes seen, the A-mode labels, and the
    realised intensity once the B modes seen are fixed."""
    rates = rng.gamma(lam_b / (k * beta), beta, k)
    counts = rng.poisson(rates * end)
    d = rng.choice(FACTORS, k)
    seen = np.flatnonzero(counts)
    x, modes = [], []
    for j in seen:
        x += list(rng.uniform(0, end, counts[j]))
        modes += [f"b{j}"] * int(counts[j])
    n_a = int(rng.poisson(lam_a * end))
    x += list(rng.uniform(0, end, n_a))
    a_modes = [f"a{i}" for i in range(n_a)]
    modes += a_modes
    x.append(end)
    modes.append(None)
    c = [0] * (len(x) - 1) + [1]
    fef = {f"b{j}": float(d[j]) for j in seen}
    realised = (
        lam_a
        + np.sum(rates[counts == 0])
        + np.sum((1 - d[seen]) * rates[seen])
    )
    return x, modes, c, fef, a_modes, float(realised)


def continuous_truth(lam_b=0.2, beta=0.02, lam_a=0.02, end=300.0):
    """The expected intensity once the B modes seen by ``end`` are fixed:
    ``lam_a + lam_b - MU lam_b beta end / (1 + beta end)``."""
    return lam_a + lam_b - MU * lam_b * beta * end / (1 + beta * end)


def discrete_test(rng, k=20000, n=30.0, c=0.5, trials=40):
    """Modes with beta chances (pseudo trials ``n``, total chance ``c``),
    ``trials`` one-shot trials: the failures and factors of the modes seen
    and the realised reliability once they are fixed."""
    a = c * n / k
    p = rng.beta(a, n - a, k)
    failed = rng.binomial(trials, p)
    d = rng.choice(FACTORS, k)
    seen = np.flatnonzero(failed)
    realised = np.prod(1 - p[failed == 0]) * np.prod(
        1 - (1 - d[seen]) * p[seen]
    )
    failures = {int(j): int(failed[j]) for j in seen}
    return failures, {int(j): float(d[j]) for j in seen}, float(realised)


def discrete_truth(n=30.0, c=0.5, trials=40):
    """The expected reliability once the modes seen are fixed, exact for
    the many-modes beta population: modes arrive as a Poisson process of
    chances with intensity ``c n (1 - p)^(n - 1) / p``, so the expected
    product over modes is ``exp(-integral of (1 - E[factor | p]))``."""

    def g(p):
        q = (1 - p) ** trials
        kept = q * (1 - p) + (1 - q) * (1 - (1 - MU) * p)
        return c * n * (1 - p) ** (n - 1) * (1 - kept) / p

    value, _ = integrate.quad(g, 0, 1, limit=200)
    return float(np.exp(-value))
