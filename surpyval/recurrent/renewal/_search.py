"""The searches the renewal fits run from each start (#728).

``GradientSearch`` is BFGS on a likelihood's hand-written gradient
(``neg_ll.value_and_grad``, see ``_derivatives``), and ``SimplexSearch``
Nelder-Mead on the likelihood alone, for a lifetime or baseline whose
derivatives are not written by hand. Both run in the unbounded space
``bounds_convert`` maps to the natural parameters, and both give
``RenewalFitMixin._multistart`` the same three steps: ``minimize`` from a
start, ``simplex`` (Nelder-Mead from a point, to restart a capped
search) and ``settle`` (to carry a maximum next to a bound onto it).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import (
    Gradient,
    preconditioned_bfgs,
)


class SearchMap:
    """The map from the search space to the natural parameters,
    ``bounds_convert``'s (``ParameterMap.inverse`` with a unit of 1), with
    its slope, for each parameter's ``(low, high)`` bounds, in Python
    floats: the gradient search needs the slopes, and ``bounds_convert``'s
    map, through autograd's numpy, cost as much as a likelihood on these
    models' small data."""

    def __init__(self, bounds: list) -> None:
        self.bounds = [
            (
                None if low is None else float(low),
                None if high is None else float(high),
            )
            for low, high in bounds
        ]

    def __call__(self, u: np.ndarray) -> tuple:
        natural, slopes = [], []
        values = np.asarray(u, dtype=float).tolist()
        for value, (low, high) in zip(values, self.bounds):
            if low is not None and high is not None:
                width = high - low
                th = math.tanh(value / 10.0)
                natural.append(low + width * (th + 1.0) / 2.0)
                slopes.append(width * (1.0 - th * th) / 20.0)
            elif low is None and high is None:
                natural.append(value)
                slopes.append(1.0)
            else:
                # Linear from 1 away from the bound, and the log of the
                # distance from it within 1 (``adj_relu``)
                if value >= 0:
                    distance, slope = value + 1.0, 1.0
                else:
                    distance = slope = math.exp(value)
                if low is not None:
                    natural.append(low + distance)
                    slopes.append(slope)
                else:
                    assert high is not None
                    natural.append(high - distance)
                    slopes.append(-slope)
        return np.array(natural), np.array(slopes)


class _GivenGradient(Gradient):
    """A ``Gradient`` whose value and gradient come from a function that
    gives them together, for ``preconditioned_bfgs`` (scipy's
    ``jac=True``)."""

    def __init__(self, fun: Callable, value_and_grad: Callable) -> None:
        self.fun = fun
        self._value_and_grad = lambda x, *args: value_and_grad(x)
        self._kept = None


class GradientSearch:
    """BFGS on the hand-written gradient ``neg_ll.value_and_grad``, with
    Nelder-Mead for what BFGS does not settle.

    ``neg_ll`` is the negative log-likelihood of the natural parameters,
    ``bounds`` their natural bounds (restoration parameter first) and
    ``n_obs`` BIC's sample size, the units of BFGS's convergence test.
    ``neg_ll.search_floor``, where the likelihood has one, is the
    smallest unit of each search coordinate (``preconditioned_bfgs``'s
    ``floor``; 1 otherwise): a Cox-Lewis ``beta``'s is one over the
    longest time, a rate per unit time searched as itself.
    """

    #: BFGS's iterations from a start; a regular maximum takes 15 to 50.
    MAX_ITERATIONS = 200

    def __init__(self, neg_ll: Callable, bounds: list, n_obs: float) -> None:
        self.neg_ll = neg_ll
        self.hand = neg_ll.value_and_grad  # type: ignore[attr-defined]
        self.to_natural = SearchMap(bounds)
        self.n_obs = n_obs
        self.floor = getattr(neg_ll, "search_floor", 1.0)
        # One object, so that preconditioned_bfgs sees it is the
        # gradient of its function, and takes both from one pass
        self.gradient = _GivenGradient(self.value, self.value_and_grad)

    def value(self, u: np.ndarray) -> float:
        """The negative log-likelihood at the search point ``u``."""
        natural, _ = self.to_natural(u)
        return self.neg_ll(natural)

    def value_and_grad(self, u: np.ndarray) -> tuple:
        """The negative log-likelihood at ``u`` with its gradient there."""
        natural, slopes = self.to_natural(u)
        value, grad = self.hand(natural)
        if not np.isfinite(value):
            # Outside the model's support: a line search steps back
            return 1e300, np.zeros(slopes.size)
        # A slope of 0 is a parameter whose map has saturated on its
        # bound, where the likelihood's own slope may not be finite
        return value, np.where(slopes == 0, 0.0, grad * slopes)

    def simplex(self, u0: np.ndarray) -> Any:
        """Nelder-Mead from ``u0``."""
        return minimize(self.value, u0, method="Nelder-Mead")

    def bfgs(self, u0: np.ndarray) -> Any:
        """BFGS from ``u0`` (``preconditioned_bfgs``) for at most
        ``MAX_ITERATIONS``; ``usable`` says whether it ended at a finite
        point inside the model's support."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found = preconditioned_bfgs(
                self.gradient.fun,
                np.asarray(u0, dtype=float),
                (),
                self.gradient,
                options={"maxiter": self.MAX_ITERATIONS},
                floor=self.floor,
                obj_scale=self.n_obs,
            )
        found.usable = bool(
            np.all(np.isfinite(found.x))
            and np.isfinite(found.fun)
            and found.fun < 1e300
        )
        return found

    def minimize(self, u0: np.ndarray) -> Any:
        """Minimise from ``u0``: BFGS (``bfgs``), and where that fails,
        BFGS again from where it stopped, then Nelder-Mead from there (or
        from ``u0``), which never ends worse than its start.

        A BFGS that fails mostly ends in a "precision loss": its line
        search lost next to where the likelihood is not finite (a
        Kijima-II ``q`` of 3 ages the systems past the largest float), or
        at the minimum itself. A fresh start of its Hessian then carries
        on from there, for a fraction of what Nelder-Mead takes.

        A search still going after ``MAX_ITERATIONS`` is running off
        towards a supremum at infinity (a Kijima-II ``q`` of 1e60 from a
        start of 2), or creeping up on a maximum on a bound. It is
        returned as it is, unconverged: ``_multistart`` restarts the best
        start's search where it is one of those."""

        def done(res: Any) -> bool:
            return bool(res.success or res.nit >= self.MAX_ITERATIONS)

        found = self.bfgs(u0)
        if found.usable and not done(found):
            again = self.bfgs(found.x)
            if again.usable and not again.fun > found.fun:
                found = again
        if found.usable and done(found):
            return found
        simplex = self.simplex(found.x if found.usable else u0)
        if found.usable and not simplex.fun < found.fun:
            return found
        return simplex

    def settle(self, res: Any, bounds: tuple) -> Any:
        """``res``, carried on to the bound of the restoration parameter
        (natural ``bounds``) where the search stopped next to it, within
        ``1e-3`` of its range. There the maximum is on the bound (an ARA
        ``rho -> 1``, a Kijima ``q -> 0``), at infinity in the search
        space, where the gradient fades away and BFGS stops while the
        likelihood still rises by ~1e-6. Nelder-Mead's expanding steps
        carry on until the likelihood stops changing; the better is
        kept."""
        low, high = bounds
        restoration = float(self.to_natural(res.x)[0][0])
        width = 1.0 if low is None or high is None else high - low
        near = (low is not None and restoration - low < 1e-3 * width) or (
            high is not None and high - restoration < 1e-3 * width
        )
        if not near:
            return res
        settled = self.simplex(res.x)
        if np.isfinite(settled.fun) and settled.fun < res.fun:
            return settled
        return res


class SimplexSearch:
    """Nelder-Mead on the likelihood ``neg_ll`` of the natural parameters
    that ``inv_trans`` maps the search space to: the search where there
    is no hand-written gradient, as all of the renewal fits searched
    before #728."""

    def __init__(self, neg_ll: Callable, inv_trans: Callable) -> None:
        self.neg_ll = neg_ll
        self.inv_trans = inv_trans

    def value(self, u: np.ndarray) -> float:
        return self.neg_ll(self.inv_trans(u))

    def simplex(self, u0: np.ndarray) -> Any:
        return minimize(self.value, u0, method="Nelder-Mead")

    def minimize(self, u0: np.ndarray) -> Any:
        return self.simplex(u0)

    def settle(self, res: Any, bounds: tuple) -> Any:
        return res


def renewal_search(
    neg_ll: Callable, bounds: list, n_obs: float, inv_trans: Callable
) -> "GradientSearch | SimplexSearch":
    """The search for the likelihood ``neg_ll`` of parameters with the
    natural ``bounds``, in the space ``inv_trans`` maps to them
    (``bounds_convert``'s): ``GradientSearch`` where ``neg_ll`` has a
    hand-written gradient (a Weibull, LogNormal, Gamma, LogLogistic,
    Exponential or Rayleigh life; a power-law, Duane, Cox-Lewis or HPP
    baseline), and ``SimplexSearch`` otherwise (any other life, such as
    the ExpoWeibull or Normal)."""
    if getattr(neg_ll, "value_and_grad", None) is not None:
        return GradientSearch(neg_ll, bounds, n_obs)
    return SimplexSearch(neg_ll, inv_trans)
