from autograd import numpy as np
from scipy.special import lambertw

class Eyring_():
    def __init__(self):
        self.name = 'Eyring'
        self.phi_param_map = {'a' : 0, 'b' : 1}
        self.phi_bounds = ((None, None), (None, None),)

    def phi(self, X, *params):
        a = params[0]
        c = params[1]
        return (1./X) * np.exp(-(c - a/X))

    def phi_init(self, life, X):
        X = X.flatten()
        a, c =  np.polyfit(1./X, np.log(life) + np.log(X), 1)
        return [a, -c]

Eyring = Eyring_()

class InverseEyring_():
    def __init__(self):
        self.name = 'InverseEyring'
        self.phi_param_map = {'a' : 0, 'c' : 1}
        self.phi_bounds = ((None, None), (None, None),)

    def phi(self, X, *params):
        a = params[0]
        c = params[1]
        return 1./((1./X) * np.exp(-(c - a/X)))

    def phi_init(self, life, X):
        X = X.flatten()
        a, c =  np.polyfit(1./X, np.log(1./life) + np.log(X), 1)
        return [a, -c]

InverseEyring = InverseEyring_()