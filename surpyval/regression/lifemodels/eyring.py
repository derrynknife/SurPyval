from autograd import numpy as np
from scipy.special import lambertw

class Eyring_():
    def __init__(self):
        self.name = 'Eyring'
        self.phi_param_map = {'a' : 0, 'b' : 1}
        self.phi_bounds = ((None, None), (None, None),)

    def phi(self, Z, *params):
        a = params[0]
        c = params[1]
        return (1./Z) * np.exp(-(c - a/Z))

    def phi_init(self, life, Z):
        Z = Z.flatten()
        a, c =  np.polyfit(1./Z, np.log(life) + np.log(Z), 1)
        return [a, -c]

Eyring = Eyring_()

class InverseEyring_():
    def __init__(self):
        self.name = 'InverseEyring'
        self.phi_param_map = {'a' : 0, 'c' : 1}
        self.phi_bounds = ((None, None), (None, None),)

    def phi(self, Z, *params):
        a = params[0]
        c = params[1]
        return 1./((1./Z) * np.exp(-(c - a/Z)))

    def phi_init(self, life, Z):
        Z = Z.flatten()
        a, c =  np.polyfit(1./Z, np.log(1./life) + np.log(Z), 1)
        return [a, -c]

InverseEyring = InverseEyring_()