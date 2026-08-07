import inspect
from typing import Callable

import numpy.typing as npt
from autograd import elementwise_grad

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
    ...     return params[0] * np.exp(params[1] * x - 1)
    ...
    >>> param_names = ['nu', 'b']
    >>> bounds = ((0, None), (0, None))
    >>> support = (-np.inf, np.inf)
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

    def Hf(self, x: Numeric, *params: Boxable) -> Boxable:
        return self._fun(x, *params)

    def hf(self, x: Numeric, *params: Boxable) -> Boxable:
        return elementwise_grad(self.Hf)(x, *params)

    def sf(self, x: Numeric, *params: Boxable) -> Boxable:
        return np.exp(-self.Hf(x, *params))

    def ff(self, x: Numeric, *params: Boxable) -> Boxable:
        return -np.expm1(-self.Hf(x, *params))

    def df(self, x: Numeric, *params: Boxable) -> Boxable:
        return elementwise_grad(self.ff)(x, *params)

    # Returns a list, where Weibull returns a tuple and the discrete
    # distributions return an array. The base contract does not pin
    # this down; callers coerce whichever they get.
    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        out: list[float] = []
        for low, high in self.bounds:
            if low is None:
                out.append(0.0 if high is None else float(high) - 1.0)
            elif high is None:
                out.append(float(low) + 1.0)
            else:
                out.append((float(high) + float(low)) / 2.0)

        return np.array(out, dtype=float)

    def mpp_inv_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_y_transform(self, y: Numeric, *params: Boxable) -> Numeric:
        return y

    def mpp_x_transform(self, x: Numeric) -> Numeric:
        return x
