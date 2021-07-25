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

        # out = self.fit(x, c, n, t, how='MSE', init=out).params
        
        return out

    # def mpp_x_transform(self, x):
    #     return np.log(x)

    # def mpp_inv_x_transform(self, x, gamma=0):
    #     return np.exp(x - gamma)

    # def mpp_y_transform(self, y, *params):
    #     mask = ((y == 0) | (y == 1))
    #     out = np.zeros_like(y)
    #     out[~mask] = np.log(-np.log((1 - y[~mask])))
    #     out[mask] = np.nan
    #     return out

    def mpp_inv_y_transform(self, y, *params):
        return 1 - np.exp(-np.exp(y))

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma



