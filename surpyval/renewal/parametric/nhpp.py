from autograd import numpy as np
from inspect import signature
from autograd import elementwise_grad, jacobian
from scipy.optimize import minimize
from surpyval.utils.counting_utils import xicn_to_xrd, handle_xicn
import numpy_indexed as npi

class ParametricCountingModel():
    def __repr__(self):

        return "Parametric Counting Model with {} CIF".format(self.dist.name)

    def cif(self, x):
        return self.dist.cif(x, *self.params)
    
    def iif(self, x):
        return self.dist.iif(x, *self.params)

    def rocof(self, x):
        if hasattr(self.dist, 'rocof'):
            return self.dist.rocof(x, *self.params)
        else:
            raise ValueError("rocof undefined for {}".format(self.dist.name))
        
    def inv_cif(self, x):
        if hasattr(self.dist, 'inv_cif'):
            return self.dist.inv_cif(x, *self.params)
        else:
            raise ValueError("Inverse cif undefined for {}".format(self.dist.name))

    # TODO: random, to T, and to N

class NHPP():
    def __init__(self, name, param_names, param_bounds, support, cif, iif=None,
                 parameter_initialiser=None):

        self.name = name
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.support = support

        if str(signature(cif)) != '(x, *params)':
            raise ValueError("cif must have signature of '(x, *params)'")
        
        self.cif = cif

        if iif is None:
            self.iif = elementwise_grad(cif)
        else:
            if str(signature(iif)) != '(x, *params)':
                raise ValueError("iif must have signature of '(x, *params)'")
            self.iif = iif

        if parameter_initialiser is None:
            self.parameter_initialiser = lambda _: np.ones(len(self.param_names))
        else:
            self.parameter_initialiser = parameter_initialiser

    def create_ll_func(self, x, i, c, n):
        """
        Need to allow for multiple items, i.
        """
        ll_dict = {}

        for ii in set(i):
            T = x[i == ii].max()
            mask = (i == ii) & (c == 0)
            xi = x[mask]
            ni = n[mask]
            ll_dict[ii] = {
                'T' : T,
                'x' : xi,
                'n' : ni
            }


        def ll_func(params):
            rv = 0
            for i in ll_dict.keys():
                rv += self.cif(ll_dict[i]['T'], *params) - (ll_dict[i]['n'] * np.log(self.iif(ll_dict[i]['x'], *params))).sum()
            return rv

        return ll_func

    def fit(self, x, i=None, c=None, n=None, how="MSE", init=None):
        """
        Format for counting processes...
        Thinking x, i, n.
        x = obvious
        i is the identity.

        n is counts ... maybe

        """

        if init is None:
            param_init = self.parameter_initialiser(x)
        else:
            param_init = np.array(init)

        model = ParametricCountingModel()

        # Need an xin checker
        # Need an xin to xrd format wrangler
        x, i, c, n = handle_xicn(x, i, c, n)
        x_unqiue, r, d = xicn_to_xrd(x, i, c, n)
        
        mcf_hat = np.cumsum(d/r)
        fun = lambda params : np.sum((self.cif(x_unqiue, *params) - mcf_hat)**2)
        res = minimize(fun, param_init)
        model.mcf_hat = mcf_hat

        if how == "MSE":
            params = res.x

        elif how == "MLE":
            if self.name == "HPP":
                # Homogeneous Poisson Process has a simple, closed form solution.
                total = npi.group_by(i).max(x)[1].sum()
                d = (n * (c == 0).astype(int)).sum()
                params = np.array([d / total])
                res = None
            else:
                ll_func = self.create_ll_func(x, i, c, n)
                jac = jacobian(ll_func)
                res = minimize(ll_func, res.x, jac=jac, method="TNC")
                params = res.x

        model.res = res
        model.params = params
        model.x = x_unqiue
        model.dist = self
        return model

# Duane Model
# Parameterisation done in accordance with:
# http://reliawiki.org/index.php/Duane_Model

duane_param_names = ['alpha', 'b']
duane_bounds = ((0, None), (0, None))
duane_support = (0.0, np.inf)

def duane_cif(x, *params):
    return params[1] * x**params[0]

def duane_iif(x, *params):
    return params[0] * params[1] * x**(params[0]-1.)

def duane_rocof(x, *params):
    return (1./params[1]) * x**(-params[0])

def duane_inv_cif(N, *params):
    return (N/params[1])**(1./params[0])

Duane = NHPP("Duane",
             duane_param_names,
             duane_bounds,
             duane_support,
             duane_cif,
             duane_iif
             )

Duane.rocof = duane_rocof
Duane.inv_cif = duane_inv_cif

# Cox-Lewis
cox_lewis_param_names = ['alpha', 'beta']
cox_lewis_bounds = ((0, None), (None, None))
cox_lewis_support = (0.0, np.inf)

def cox_lewis_cif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (np.exp(alpha + beta*x) - np.exp(alpha))/beta

def cox_lewis_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (np.exp(alpha + beta*x) - np.exp(alpha))/beta

def cox_lewis_inv_cif(cif, *params):
    alpha = params[0]
    beta = params[1]
    return (np.log((mcf * beta) + np.exp(alpha)) - alpha)/beta

def cox_lewis_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return np.exp(alpha + beta * x)

CoxLewis = NHPP("CoxLewis",
             cox_lewis_param_names,
             cox_lewis_bounds,
             cox_lewis_support,
             cox_lewis_cif,
             cox_lewis_iif
             )

CoxLewis.rocof = cox_lewis_rocof
CoxLewis.inv_cif = cox_lewis_inv_cif

# Crow

crow_param_names = ['alpha', 'beta']
crow_bounds = ((0, None), (0, None))
crow_support = (0.0, np.inf)

def crow_cif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (x**beta)/alpha

def crow_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta / alpha) * (x**(beta - 1))

def crow_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta/alpha) * x**(beta - 1.)

def crow_inv_cif(mcf, *params):
    alpha = params[0]
    beta = params[1]
    return (alpha * mcf) ** (1./beta)

Crow = NHPP("Crow",
             crow_param_names,
             crow_bounds,
             crow_support,
             crow_cif,
             crow_iif
             )

Crow.rocof = crow_rocof
Crow.inv_cif = crow_inv_cif

# Crow

crow_amsaa_param_names = ['alpha', 'beta']
crow_amsaa_bounds = ((0, None), (0, None))
crow_amsaa_support = (0.0, np.inf)

def crow_amsaa_cif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (x/alpha)**beta

def crow_amsaa_iif(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta / alpha**beta) * (x**(beta - 1))

def crow_amsaa_rocof(x, *params):
    alpha = params[0]
    beta = params[1]
    return (beta/alpha) * x**(beta - 1.)

def crow_amsaa_inv_cif(mcf, *params):
    alpha = params[0]
    beta = params[1]
    return (alpha * mcf) ** (1./beta)

CrowAMSAA = NHPP("CrowAMSAA",
             crow_amsaa_param_names,
             crow_amsaa_bounds,
             crow_amsaa_support,
             crow_amsaa_cif,
             crow_amsaa_iif
             )

CrowAMSAA.rocof = crow_amsaa_rocof
CrowAMSAA.inv_cif = crow_amsaa_inv_cif

# Homogeneous Poisson Process
hpp_param_names = ['lambda']
hpp_bounds = ((0, None),)
hpp_support = (0.0, np.inf)

def hpp_iif(x, *params):
    rate = params[0]
    return rate

def hpp_cif(x, *params):
    rate = params[0]
    return rate * x

def hpp_rocof(x, *params):
    rate = params[0]
    return np.ones_like(x) * rate

def hpp_inv_cif(cif, *params):
    rate = params[0]
    return mcf / rate

HPP = NHPP("HPP",
             hpp_param_names,
             hpp_bounds,
             hpp_support,
             hpp_cif,
             hpp_iif
             )

HPP.rocof = hpp_rocof
HPP.inv_cif = hpp_inv_cif