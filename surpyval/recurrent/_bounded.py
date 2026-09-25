"""Maps between bounded model parameters and an unconstrained search space.

Optimising on the natural scale lets a search land exactly on a bound
(Nelder-Mead and L-BFGS-B clip onto closed bounds), where a positive
parameter is 0 and a likelihood or cumulative intensity divides by it.
"""

from typing import Callable

import numpy as np


def unconstraining_maps(bounds: list) -> tuple[Callable, Callable]:
    """Maps between natural parameters and an unconstrained search space.

    ``(low, None)`` becomes ``low + exp(u)``, ``(None, high)`` becomes
    ``high - exp(u)``, a finite ``(low, high)`` a logistic between them,
    and ``(None, None)`` is left alone. Starting values on or outside a
    bound are nudged inside it.
    """
    lows = [b[0] for b in bounds]
    highs = [b[1] for b in bounds]

    def to_natural(u: np.ndarray) -> np.ndarray:
        out = np.array(u, dtype=float)
        for k, (low, high) in enumerate(zip(lows, highs)):
            if low is not None and high is not None:
                out[k] = low + (high - low) / (1.0 + np.exp(-u[k]))
            elif low is not None:
                out[k] = low + np.exp(u[k])
            elif high is not None:
                out[k] = high - np.exp(u[k])
        return out

    def to_search(v: np.ndarray) -> np.ndarray:
        out = np.array(v, dtype=float)
        for k, (low, high) in enumerate(zip(lows, highs)):
            if low is not None and high is not None:
                frac = np.clip((v[k] - low) / (high - low), 1e-12, 1 - 1e-12)
                out[k] = np.log(frac / (1.0 - frac))
            elif low is not None:
                out[k] = np.log(max(v[k] - low, 1e-12 * max(1.0, abs(low))))
            elif high is not None:
                out[k] = np.log(max(high - v[k], 1e-12 * max(1.0, abs(high))))
        return out

    return to_natural, to_search
