from autograd import numpy as np

class GeneralLogLinear_():
    def __init__(self):
        self.name = 'GeneralLogLinear'
        self.phi_bounds = lambda X: (((None, None),) * X.shape[1])
        self.phi_param_map = lambda X: {'beta_' + str(i) : i for i in range(X.shape[1])}

    def phi(self, X, *params):
        return np.exp(np.dot(X, np.array(params)))

    def phi_init(self, X):
        # return np.zeros(X.shape[1])
        return 1./X.mean(axis=0)

GeneralLogLinear = GeneralLogLinear_()