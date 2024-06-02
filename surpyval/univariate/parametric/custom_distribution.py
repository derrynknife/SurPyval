import inspect

from autograd import elementwise_grad

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


class CustomDistribution(ParametricFitter):
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
    >>>     return params[0] * np.exp(params[1] * x - 1)
    >>>
    >>> param_names = ['nu', 'b']
    >>> bounds = ((0, None), (0, None))
    >>> support = (-np.inf, np.inf)
    >>> Gompertz = surv.CustomDistribution(
        name, Hf, param_names, bounds, support
    )
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> Gompertz.fit(x)
    """

    def __init__(self, name, fun, param_names, bounds, support):
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
                + "inflated or hurdle models"
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
        self.Hf = fun
        self.hf = lambda x, *params: elementwise_grad(self.Hf)(x, *params)
        self.sf = lambda x, *params: np.exp(-self.Hf(x, *params))
        self.ff = lambda x, *params: -np.expm1(-self.Hf(x, *params))
        self.df = lambda x, *params: elementwise_grad(self.ff)(x, *params)

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        out = []
        for low, high in self.bounds:
            if (low is None) & (high is None):
                out.append(0)
            elif high is None:
                out.append(low + 1.0)
            elif low is None:
                out.append(high - 1.0)
            else:
                out.append((high + low) / 2.0)

        return out

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma
