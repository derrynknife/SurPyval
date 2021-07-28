import numpy as np
from surpyval import nonparametric as nonp
from surpyval.utils import xcnt_handler

class NonParametricFitter():
    
    def fit(self, x=None, c=None, n=None, t=None,
            xl=None, xr=None, tl=None, tr=None,
            estimator='Fleming-Harrington'):
        x, c, n, t = xcnt_handler(x=x, c=c, n=n, t=t, 
                                  tl=tl, tr=tr, 
                                  xl=xl, xr=xr)

        data = {}
        data['x'] = x
        data['c'] = c
        data['n'] = n
        data['t'] = t
        if self.how == 'Turnbull':
            data['estimator'] = estimator

        out = nonp.NonParametric()
        out.data = data
        results = nonp.FIT_FUNCS[self.how](**data)
        for k, v in results.items():
            setattr(out, k, v)
        out.model = self.how
        out.F = 1 - out.R

        with np.errstate(all='ignore'):
            out.H = -np.log(out.R)
            var = out.d / (out.r * (out.r - out.d))
            greenwood = np.cumsum(var)
            out.greenwood = greenwood

        return out