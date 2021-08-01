from surpyval.parametric.parametric_fitter import ParametricFitter
import surpyval
import autograd.numpy as np
from autograd import elementwise_grad
import inspect

class Distribution(ParametricFitter):
    def __init__(self, name, fun, param_names, bounds, support):
        if str(inspect.signature(fun)) != '(x, *params)':
            raise ValueError('Function must have the signature \'(x, *params)\'')

        if len(param_names) != len(bounds):
            raise ValueError('param_names and bounds must have same length')

        if 'p' in param_names:
            raise ValueError("'p' reserved parameter name for LFP distributions")

        if 'gamma' in param_names:
            raise ValueError("'gamma' reserved parameter name for offset distributions")

        if 'f0' in param_names:
            raise ValueError("'f0' reserved parameter name for zero inflated or hurdle models")

        for p_name in param_names:
            if hasattr(self, p_name):
                raise ValueError("Can't name a parameter after a function in the Parametric class.")

        self.name = name
        self.k = len(param_names)
        self.bounds = bounds
        self.support = support
        self.plot_x_scale = 'linear'
        self.y_ticks = np.linspace(0, 1, 11)
        self.param_names = param_names
        self.param_map = {v : i for i, v in enumerate(param_names)}
        self.Hf = fun
        self.hf = lambda x, *params: elementwise_grad(self.Hf)(x, *params)
        self.sf = lambda x, *params: np.exp(-self.Hf(x, *params))
        self.ff = lambda x, *params: 1 - np.exp(-self.Hf(x, *params))
        self.df = lambda x, *params: elementwise_grad(self.ff)(x, *params)

    def _parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
        out = []
        for low, high in self.bounds:
            if (low is None) & (high is None):
                out.append(0)
            elif high is None:
                out.append(low + 1.)
            elif low is None:
                out.append(high - 1.)
            else:
                out.append((high + low)/2.)

        return out

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma


