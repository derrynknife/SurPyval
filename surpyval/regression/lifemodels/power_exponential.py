from autograd import numpy as np

class PowerExponential_():
    def __init__(self):
        self.name = 'PowerExponential'
        self.phi_param_map = {'c' : 0, 'a' : 1, 'n' : 2}
        self.phi_bounds = ((0, None), (None, None), (None, None))

    def phi(self, X, *params):
        X = np.atleast_2d(X)
        X1 = X[:, 0]
        X2 = X[:, 1]
        c = params[0]
        a = params[1]
        n = params[2]
        return c * np.exp(a/X1) * X2**n

    def phi_init(self, life, X):
        A = np.atleast_2d(X)
        A = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
        A[:, 1] = 1./A[:, 1]
        A[:, 2] = np.log(A[:, 2])
        y = np.log(life)
        c, a, n = np.linalg.lstsq(A, y, rcond=None)[0]
        return [np.exp(c), a, n]

PowerExponential = PowerExponential_()