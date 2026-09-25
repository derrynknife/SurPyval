import inspect
import itertools
from typing import Callable

import numpy as onp
import numpy.typing as npt
from autograd import elementwise_grad
from scipy.integrate import quad

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData


class CustomDistribution(OptimisedFitMixin, ParametricFitter):
    """
    Used to create a custom distribution using only the cumulative hazard
    function. The cumulative hazard function must be a function of x and
    the parameters. The parameters must be named in the param_names and
    the bounds must be specified in the bounds argument. The support
    argument is used to specify the support of the distribution.

    Parameters
    ----------

    name: str
        Name of the distribution

    fun: callable
        Function that returns the cumulative hazard function

    param_names: list
        List of parameter names

    bounds: list
        List of tuples containing the lower and upper bounds of the
        parameters

    support: tuple
        Tuple containing the lower and upper bounds of the support of the
        distribution

    Examples
    --------

    >>> from autograd import numpy as np
    >>> import surpyval as surv
    >>>
    >>> name = 'Gompertz'
    >>>
    >>> def Hf(x, *params):
    ...     # the Gompertz cumulative hazard nu (e^{b x} - 1), zero at x = 0
    ...     return params[0] * (np.exp(params[1] * x) - 1)
    ...
    >>> param_names = ['nu', 'b']
    >>> bounds = ((0, None), (0, None))
    >>> support = (0, np.inf)
    >>> Gompertz = surv.CustomDistribution(
    ...     name, Hf, param_names, bounds, support
    ... )
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = Gompertz.fit(x)
    """

    def __init__(
        self,
        name: str,
        # Validated at runtime to have the signature (x, *params);
        # Callable[..., Boxable] is as close as the type system gets.
        fun: Callable[..., Boxable],
        param_names: list[str],
        bounds: tuple[tuple[int | float | None, int | float | None], ...],
        support: tuple[int | float, int | float],
    ) -> None:
        if str(inspect.signature(fun)) != "(x, *params)":
            detail = "Function must have the signature '(x, *params)'"
            raise ValueError(detail)

        if len(param_names) != len(bounds):
            raise ValueError("param_names and bounds must have same length")

        if "p" in param_names:
            detail = "'p' reserved parameter name for LFP distributions"
            raise ValueError(detail)

        if "gamma" in param_names:
            detail = "'gamma' reserved parameter name for offset distributions"
            raise ValueError(detail)

        if "f0" in param_names:
            detail = (
                "'f0' reserved parameter name for zero"
                "inflated or hurdle models"
            )
            raise ValueError(detail)

        for p_name in param_names:
            if hasattr(self, p_name):
                detail = "Can't name a parameter after a function"
                raise ValueError(detail)

        super().__init__(
            name=name,
            k=len(param_names),
            bounds=bounds,
            support=support,
            param_names=param_names,
            param_map={v: i for i, v in enumerate(param_names)},
            plot_x_scale="linear",
            y_ticks=np.linspace(0, 1, 11),
        )
        # Stored, then exposed through real methods below. Assigning
        # over self.Hf and friends stopped being possible once
        # OptimisedFitMixin declared them for its own use: a subclass
        # inherits those declarations, and assigning to an inherited
        # method is an error. Delegating is equivalent -- the previous
        # ``self.Hf = fun`` was an unbound instance attribute, so
        # ``self.Hf(x, *params)`` called ``fun(x, *params)`` either way.
        self._fun = fun
        # A distribution known only through its cumulative hazard has no
        # linearising transform, so there is no probability-plot
        # regression to fit (the identity transforms below only serve to
        # draw ``plot()``). Advertising MPP support sent ``how='MPP'`` on
        # to an ``unpack_rr`` that does not exist, and it died with an
        # AttributeError instead of the usual refusal.
        self.supports_mpp = False

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Cumulative hazard: the user-supplied function ``fun(x, *params)``.
        """
        return self._fun(x, *params)

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Hazard rate, :math:`h(x) = dH(x)/dx`, differentiated from ``Hf``
        with autograd.
        """
        return elementwise_grad(self.Hf)(x, *params)

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Survival function, :math:`R(x) = e^{-H(x)}`.
        """
        return np.exp(-self.Hf(x, *params))

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Failure (CDF) function, :math:`F(x) = 1 - e^{-H(x)}`.
        """
        return -np.expm1(-self.Hf(x, *params))

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        """
        Density, :math:`f(x) = dF(x)/dx`, differentiated from ``ff`` with
        autograd.
        """
        return elementwise_grad(self.ff)(x, *params)

    def moment(self, m: int, *params: Boxable) -> Boxable:
        r"""
        The ``m``-th raw moment, integrated from the survival function:

        .. math::
            E[X^m] = \int_0^\infty m x^{m-1} R(x)\,dx
                   - \int_{-\infty}^0 m x^{m-1} F(x)\,dx ,

        restricted to the declared support (``R = 1`` below it and
        ``F = 1`` above it contribute in closed form).

        The generic fallback integrates ``x**m * df(x)`` instead, and
        ``df`` here is autograd's derivative of ``ff``: in the far tail the
        cumulative hazard overflows, the derivative of ``exp(-inf)`` is
        ``0 * inf``, and one nan made every moment -- and so ``var()`` --
        nan. ``R`` and ``F`` themselves underflow cleanly to 0 and 1, and
        need no derivative at all.
        """
        if m == 0:
            return 1.0
        theta = [float(p) for p in params]
        lo, hi = float(self.support[0]), float(self.support[1])

        # Evaluated at a length-one array (a user's cumulative hazard may
        # index or reduce its argument) and read back as a float.
        def sf(t: float) -> float:
            value = self.sf(onp.array([t]), *theta)
            return float(onp.asarray(value, dtype=float).ravel()[0])

        def ff(t: float) -> float:
            value = self.ff(onp.array([t]), *theta)
            return float(onp.asarray(value, dtype=float).ravel()[0])

        total = 0.0
        with onp.errstate(all="ignore"):
            if hi > 0:
                start = max(lo, 0.0)
                # R = 1 on (0, lo) for a support starting above zero
                total += start**m
                total += quad(
                    lambda t: m * t ** (m - 1) * sf(t), start, hi, limit=200
                )[0]
            if lo < 0:
                end = min(hi, 0.0)
                # F = 1 on (hi, 0) for a support ending below zero
                total += end**m if hi < 0 else 0.0
                total -= quad(
                    lambda t: m * t ** (m - 1) * ff(t), lo, end, limit=200
                )[0]
        return total

    def mean(self, *params: Boxable) -> Boxable:
        """The mean, the first raw moment (see :meth:`moment`)."""
        return self.moment(1, *params)

    # Returns a list, where Weibull returns a tuple and the discrete
    # distributions return an array. The base contract does not pin
    # this down; callers coerce whichever they get.
    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        """
        A starting point for the fit, chosen by likelihood from a coarse
        grid of magnitudes.

        A custom distribution knows nothing about its parameters beyond
        their bounds, and a fixed start (1 for a positive parameter) can sit
        so far from the data's scale that the likelihood is flat to machine
        precision there -- a mortality rate of 1e-4 started at 1 gives
        ``exp(1 * 70)`` terms -- and the optimiser stops at once, reporting
        success. So each parameter gets a handful of candidate values
        spanning many orders of magnitude (and the data's own scale), and
        the combination with the best log-likelihood is the start (the
        fixed default is tried as a second start, see
        :meth:`_alternative_base_starts`). With an offset the returned
        vector leads with the offset.
        """
        x = np.asarray(data.x, dtype=float)
        finite = np.abs(x[np.isfinite(x)])
        positive = finite[finite > 0]
        scale = float(np.median(positive)) if positive.size else 1.0

        grids = [
            self._start_candidates(low, high, scale)
            for low, high in self.bounds
        ]
        default = [g[0] for g in grids]

        def neg_ll(params: "list[float]") -> float:
            with np.errstate(all="ignore"):
                try:
                    value = float(
                        self._neg_ll_func(data, *params, 0.0, 0.0, 1.0)
                    )
                except (ValueError, FloatingPointError, OverflowError):
                    return np.inf
            return value if np.isfinite(value) else np.inf

        n_combinations = int(np.prod([len(g) for g in grids]))
        best: "list[float]" = list(default)
        best_value = neg_ll(best)
        if n_combinations <= 512:
            for candidate in itertools.product(*grids):
                value = neg_ll(list(candidate))
                if value < best_value:
                    best, best_value = list(candidate), value
        else:
            # too many to enumerate: coordinate-wise sweeps from the default
            for _ in range(3):
                for k, grid in enumerate(grids):
                    for value_k in grid:
                        trial = list(best)
                        trial[k] = value_k
                        value = neg_ll(trial)
                        if value < best_value:
                            best, best_value = trial, value

        out = np.array(best, dtype=float)
        if offset:
            gamma = float(np.min(x[np.isfinite(x)])) - 1.0
            out = np.concatenate([[gamma], out])
        return out

    def _alternative_base_starts(
        self, data: SurpyvalData, offset: bool = False
    ) -> "list[npt.NDArray]":
        """
        The fixed default start (1 above a lower bound, the midpoint of a
        finite interval, 0 when unbounded) is also tried: the grid's best
        starting likelihood can sit on a plateau -- a spline knot below
        every data point, say -- that the optimiser cannot leave.
        """
        fixed = np.array(
            [self._start_candidates(lo, hi, 1.0)[0] for lo, hi in self.bounds],
            dtype=float,
        )
        if offset:
            x = np.asarray(data.x, dtype=float)
            gamma = float(np.min(x[np.isfinite(x)])) - 1.0
            fixed = np.concatenate([[gamma], fixed])
        return [fixed]

    @staticmethod
    def _start_candidates(
        low: "float | None", high: "float | None", scale: float
    ) -> "list[float]":
        """Candidate starting values for one parameter within its bounds;
        the first is the old fixed default (1 above a lower bound, the
        midpoint of a finite interval, 0 when unbounded)."""
        magnitudes = [1.0, 1e-6, 1e-4, 1e-2, 1e2, 1.0 / scale, scale]
        if low is None and high is None:
            return [0.0] + [m for m in (1.0, -1.0, scale, -scale)]
        if high is None:
            assert low is not None
            return [float(low) + m for m in magnitudes]
        if low is None:
            return [float(high) - m for m in magnitudes]
        lo, hi = float(low), float(high)
        return [lo + (hi - lo) * f for f in (0.5, 0.1, 0.9, 0.01, 0.99)]

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return y

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return y

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x
