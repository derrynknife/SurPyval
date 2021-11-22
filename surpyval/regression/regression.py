import re
from autograd import jacobian
from autograd import grad
import autograd.numpy as np
from scipy.stats import uniform

from surpyval import round_sig, fsli_to_xcn
from scipy.special import ndtri as z
from surpyval import nonparametric as nonp
from copy import deepcopy, copy

import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime

class Regression():
    """
    Result of ``.fit()`` or ``.from_params()`` method for every parametric surpyval 
    distribution.
    
    Instances of this class are very useful when a user needs the other functions 
    of a distribution for plotting, optimizations, monte carlo analysis and numeric 
    integration.

    """
    def __repr__(self):
        dist_params = self.params[0:self.k_dist]
        reg_model_params = self.params[self.k_dist:]
        dist_param_string = '\n'.join(['{:>10}'.format(name) + ": " 
            + str(p) for p, name in zip(dist_params, self.distribution.param_names) 
            if name not in self.fixed])

        reg_model_param_string = '\n'.join(['{:>10}'.format(name) + ": " 
            + str(p) for p, name in zip(reg_model_params, self.reg_model.phi_param_map) 
            if name not in self.fixed])

        if hasattr(self, 'params'):
            out = ('Parametric Regression SurPyval Model'
                   + '\n===================================='
                   + '\nKind                : {kind}'
                   + '\nDistribution        : {dist}'
                   + '\nRegression Model    : {reg_model}'
                   + '\nFitted by           : MLE'
                   ).format(kind=self.kind, 
                            dist=self.distribution.name, 
                            reg_model=self.reg_model.name)

            out = (out + '\nDistribution        :\n'
                       + '{params}'.format(params=dist_param_string))

            out = (out + '\nRegression Model    :\n'
                       + '{params}'.format(params=reg_model_param_string))

            return out
        else:
            return "Unable to fit values"

    def phi(self, X):
        return self.reg_model.phi(X, *self.phi_params)

    def sf(self, x, X):
        r"""
        Surival (or Reliability) function for a distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the survival function will be calculated 

        Returns
        -------

        sf : scalar or numpy array 
            The scalar value of the survival function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the survival function at each corresponding value in the input array.


        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.sf(2)
        0.9920319148370607
        >>> model.sf([1, 2, 3, 4, 5])
        array([0.9990005 , 0.99203191, 0.97336124, 0.938005  , 0.8824969 ])
        """
        if type(x) == list:
            x = np.array(x)
        return self.model.sf(x, X, *self.params)

    def ff(self, x, X):
        r"""
        The cumulative distribution function, or failure function, for a distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the failure function (CDF) will be calculated

        Returns
        -------

        ff : scalar or numpy array 
            The scalar value of the CDF of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the CDF at each corresponding value in the input array.


        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.ff(2)
        0.007968085162939342
        >>> model.ff([1, 2, 3, 4, 5])
        array([0.0009995 , 0.00796809, 0.02663876, 0.061995  , 0.1175031 ])
        """
        if type(x) == list:
            x = np.array(x)

        return self.model.ff(x, X, *self.params)

    def df(self, x, X):
        r"""
        The density function for a distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the density function will be calculated

        Returns
        -------

        df : scalar or numpy array 
            The scalar value of the density function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the density function at each corresponding value in the input array.


        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.df(2)
        0.01190438297804473
        >>> model.df([1, 2, 3, 4, 5])
        array([0.002997  , 0.01190438, 0.02628075, 0.04502424, 0.06618727])
        """
        if type(x) == list:
            x = np.array(x)
        return self.model.df(x, X, *self.params)

    def hf(self, x, X):
        r"""
        The instantaneous hazard function for a distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the instantaneous hazard function will be calculated

        Returns
        -------

        hf : scalar or numpy array 
            The scalar value of the instantaneous hazard function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the instantaneous hazard function at each corresponding value in the input array.


        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.hf(2)
        0.012000000000000002
        >>> model.hf([1, 2, 3, 4, 5])
        array([0.003, 0.012, 0.027, 0.048, 0.075])
        """
        if type(x) == list:
            x = np.array(x)
        return self.model.hf(x, X, *self.params)

    def Hf(self, x, X):
        r"""

        The cumulative hazard function for a distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the cumulative hazard function will be calculated

        Returns
        -------

        Hf : scalar or numpy array 
            The scalar value of the cumulative hazard function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the cumulative hazard function at each corresponding value in the input array.


        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.Hf(2)
        0.008000000000000002
        >>> model.Hf([1, 2, 3, 4, 5])
        array([0.001, 0.008, 0.027, 0.064, 0.125])
        """
        if type(x) == list:
            x = np.array(x)
        return self.model.hf(x, X, *self.params)

    def random(self, size, X):
        r"""

        A method to draw random samples from the distributions using the parameters found in the ``.params`` attribute.

        Parameters
        ----------
        size : int
            The number of random samples to be drawn from the distribution.

        X : scalar or array like
            The value(s) of the stresses at which the random 

        Returns
        -------
        random : numpy array 
            Returns a numpy array of size ``size`` with random values drawn from the distribution.


        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> np.random.seed(1)
        >>> model.random(1)
        array([8.14127103])
        >>> model.random(10)
        array([10.84103403,  0.48542084,  7.11387062,  5.41420125,  4.59286657,
                5.90703589,  7.5124326 ,  7.96575225,  9.18134126,  8.16000438])
        """
        if (self.p == 1) and (self.f0 == 0):
            return self.dist.qf(uniform.rvs(size=size), *self.params) + self.gamma
        elif (self.p != 1) and (self.f0 == 0):
            n_obs = np.random.binomial(size, self.p)

            f = self.dist.qf(uniform.rvs(size=n_obs), *self.params) + self.gamma
            s = np.ones(np.array(size) - n_obs) * np.max(f) + 1

            return fsli_to_xcn(f, s)

        elif (self.p == 1) and (self.f0 != 0):
            n_doa = np.random.binomial(size, self.f0)

            x0 = np.zeros(n_doa) + self.gamma
            x = self.dist.qf(uniform.rvs(size=size - n_doa), *self.params) + self.gamma
            x = np.concatenate([x, x0])
            np.random.shuffle(x)

            return x
        else:
            N = np.random.multinomial(1, [self.f0, self.p - self.f0, 1. - self.p], size).sum(axis=0)
            N = np.atleast_2d(N)
            n_doa, n_obs, n_cens = N[:, 0], N[:, 1], N[:, 2]
            x0 = np.zeros(n_doa) + self.gamma
            x = self.dist.qf(uniform.rvs(size=n_obs), *self.params) + self.gamma
            f = np.concatenate([x, x0])
            s = np.ones(n_cens) * np.max(f) + 1
            # raise NotImplementedError("Combo zero-inflated and lfp model not yet supported")
            return fsli_to_xcn(f, s)

    def neg_ll(self):
        r"""

        The the negative log-likelihood for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        neg_ll : float
            The negative log-likelihood of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.neg_ll()
        262.52685642385734
        """
        if not hasattr(self, 'data'):
            raise ValueError("Must have been fit with data")

        return self._neg_ll

    def bic(self):
        r"""

        The the Bayesian Information Criterion (BIC) for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        bic : float
            The BIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.bic()
        534.2640532196908

        References:
        -----------

        `Bayesian Information Criterion for Censored Survival Models <https://www.jstor.org/stable/2677130>`_.

        """
        if hasattr(self, '_bic'):
            return self._bic
        else:
            self._bic = self.k  * np.log(self.data['n'][self.data['c'] == 0].sum()) + 2 * self.neg_ll()
            return self._bic

    def aic(self): 
        r"""

        The the Aikake Information Criterion (AIC) for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        aic : float
            The AIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.aic()
        529.0537128477147
        """
        if hasattr(self, '_aic'):
            return self._aic
        else:
            self._aic = 2 * self.k + 2 * self.neg_ll()
            return self._aic

    def aic_c(self):
        r"""

        The the Corrected Aikake Information Criterion (AIC) for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        aic_c : float
            The Corrected AIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.aic()
        529.1774241879209
        """
        if hasattr(self, '_aic_c'):
            return self._aic_c
        else:
            k = len(self.params)
            n = self.data['n'].sum()
            self._aic_c = self.aic() + (2*k**2 + 2*k)/(n - k - 1)
            return self._aic_c
