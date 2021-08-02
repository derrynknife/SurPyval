from autograd import numpy as np

class DualExponential_():
    def __init__(self):
        self.name = 'DualExponential'
        self.phi_param_map = {'a' : 0, 'b' : 1, 'c' : 2}
        self.phi_bounds = ((None, None), (None, None), (0, None))

    def phi(self, X, *params):
        X = np.atleast_2d(X)
        X1 = X[:, 0]
        X2 = X[:, 1]
        a = params[0]
        b = params[1]
        c = params[2]
        return c * np.exp(a/X1 + b/X2)

    def phi_init(self, life, X):
        A = np.atleast_2d(X)
        A = 1./np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        y = np.log(life)
        c, a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return [a, b, np.exp(c)]

DualExponential = DualExponential_()