from autograd import numpy as np

class DualPower_():
    def __init__(self):
        self.name = 'DualPower'
        self.phi_param_map = {'c' : 0, 'm' : 1, 'n' : 2}
        self.phi_bounds = ((0, None), (None, None), (None, None))

    def phi(self, X, *params):
        X = np.atleast_2d(X)
        X1 = X[:, 0]
        X2 = X[:, 1]
        c = params[0]
        m = params[1]
        n = params[2]
        return c * X1**m * X2**n

    def phi_init(self, life, X):
        A = np.atleast_2d(X)
        A = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), np.log(X)])
        y = np.log(life)
        c, m, n = np.linalg.lstsq(A, y, rcond=None)[0]
        return [np.exp(c), m, n]

DualPower = DualPower_()