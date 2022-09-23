from autograd import numpy as np

class InversePower_():
    def __init__(self):
        self.name = 'InversePower'
        self.phi_param_map = {'a' : 0, 'n' : 1}
        self.phi_bounds = ((0, None), (None, None),)

    def phi(self, Z, *params):
        return 1./(params[0] * Z**params[1])

    def phi_init(self, life, Z):
        Z = Z.flatten()
        n, a = (np.polyfit(np.log(Z), np.log(1./life), 1))
        return [np.exp(a), n]

InversePower = InversePower_()

class Power_():
    def __init__(self):
        self.name = 'Power'
        self.phi_param_map = {'a' : 0, 'n' : 1}
        self.phi_bounds = ((0, None), (None, None),)

    def phi(self, Z, *params):
        return params[0] * Z**params[1]

    def phi_init(self, life, Z):
        Z = Z.flatten()
        n, a = (np.polyfit(np.log(Z), np.log(life), 1))
        return [np.exp(a), n]

Power = Power_()