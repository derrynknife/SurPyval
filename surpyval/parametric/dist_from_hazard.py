from surpyval.parametric.parametric_fitter import ParametricFitter
import autograd.numpy as np
from autograd import grad
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
        self.y_ticks = np.linspace(0, 1, 10)
        self.param_names = param_names
        self.param_map = {v : i for i, v in enumerate(param_names)}
        self.Hf = fun
        self._hf = lambda x, *params: grad(self.Hf)(x, *params)
        # self.hf = np.vectorize(lambda x, *params: grad(self.Hf)(x, *params))
        self.sf = lambda x, *params: np.exp(-self.Hf(x, *params))
        self.ff = lambda x, *params: 1 - np.exp(-self.Hf(x, *params))
        self._df = lambda x, *params: grad(self.Hf)(x, *params) * np.exp(-self.Hf(x, *params))

    def hf(self, x, *params):
        if hasattr(x, '__iter__'):
            out = np.zeros_like(x).astype(float)
            for i, v in enumerate(x):
                hf = self._hf(v, *params)
                if str(type(hf)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
                    out[i] = hf._value
                else:
                    out[i] = hf
            return out
        else:
            return self._hf(x, *params)

    def df(self, x, *params):
        if hasattr(x, '__iter__'):
            out = np.zeros_like(x).astype(float)
            for i, v in enumerate(x):
                df = self._df(v, *params)
                if str(type(df)) == "<class 'autograd.numpy.numpy_boxes.ArrayBox'>":
                    out[i] = df._value
                else:
                    out[i] = df

            return out
        else:
            return self._df(x, *params)

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

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma



