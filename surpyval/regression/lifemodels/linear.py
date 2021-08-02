from autograd import numpy as np

class Linear_():
    def __init__(self):
        self.name = 'Linear'
        self.phi_param_map = {'a' : 0, 'b' : 1}
        self.phi_bounds = ((None, None), (None, None),)

    def phi(self, X, *params):
        a = params[0]
        b = params[1]
        return a + b * X

    def phi_init(self, life, X):
        X = X.flatten()
        b, a =  np.polyfit(X, life, 1)
        return [a, b]

Linear = Linear_()