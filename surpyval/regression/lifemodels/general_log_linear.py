from autograd import numpy as np

class GeneralLogLinear_():
    def __init__(self):
        self.name = 'GeneralLogLinear'
        self.phi_bounds = lambda Z: (((None, None),) * Z.shape[1])
        self.phi_param_map = lambda Z: {'beta_' + str(i) : i for i in range(Z.shape[1])}

    def phi(self, Z, *params):
        return np.exp(np.dot(Z, np.array(params)))

    def phi_init(self, Z):
        # return np.zeros(Z.shape[1])
        return 1./Z.mean(axis=0)

GeneralLogLinear = GeneralLogLinear_()