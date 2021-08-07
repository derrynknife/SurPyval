import re
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import jacobian
from scipy.stats import uniform
from surpyval import round_sig, fs_to_xcn
from scipy.special import ndtri as z
from surpyval import nonparametric as nonp
from copy import deepcopy, copy
from matplotlib.ticker import FixedLocator


def _round_vals(x):
    not_different = True
    i = 1
    while not_different:
        x_ticks = np.array(round_sig(x, i))
        not_different = (np.diff(x_ticks) == 0).any()
        i += 1
    return x_ticks


class Parametric():
    """
    Result of ``.fit()`` or ``.from_params()`` method for every parametric
    surpyval distribution.

    Instances of this class are very useful when a user needs the other
    functions of a distribution for plotting, optimizations, monte carlo
    analysis and numeric integration.
    """
    def __init__(self, dist, method, data, offset, lfp, zi):
        self.dist = dist
        self.k = copy(dist.k)
        self.method = method
        self.data = data
        self.offset = offset
        self.lfp = lfp
        self.zi = zi

        bounds = deepcopy(dist.bounds)
        param_map = dist.param_map.copy()

        if offset:
            if data is not None:
                bounds = ((None, np.min(data['x'])), *bounds)
            else:
                bounds = ((None, None), *bounds)

            param_map = {k: v + 1 for k, v in param_map.items()}
            param_map.update({'gamma': 0})
            self.k += 1
        else:
            self.gamma = 0

        if lfp:
            bounds = (*bounds, (0, 1))
            param_map.update({'p': len(param_map) + 1})
            self.k += 1
        else:
            self.p = 1

        if zi:
            bounds = (*bounds, (0, 1))
            param_map.update({'f0': len(param_map) + 1})
            self.k += 1
        else:
            self.f0 = 0

        self.bounds = bounds
        self.param_map = param_map

    def __repr__(self):
        param_string = '\n'.join(['{:>10}'.format(name)
                                  + ": " + str(p) for p, name in
                                  zip(self.params, self.dist.param_names)])

        if hasattr(self, 'params'):
            out = ('Parametric SurPyval Model'
                   + '\n========================='
                   + '\nDistribution        : {dist}'
                   + '\nFitted by           : {method}'
                   ).format(dist=self.dist.name,
                            method=self.method)
            if self.offset:
                out += '\nOffset (gamma)      : {g}'.format(g=self.gamma)

            if self.lfp:
                out += '\nMax Proportion (p)  : {p}'.format(p=self.p)

            if self.zi:
                out += '\nZero-Inflation (f0) : {f0}'.format(f0=self.f0)

            out = (out + '\nParameters          :\n'
                       + '{params}'.format(params=param_string))

            return out
        else:
            return "Unable to fit values"

    def sf(self, x):
        r"""

        Surival (or Reliability) function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the survival
            function will be calculated.

        Returns
        -------

        sf : scalar or numpy array
            The scalar value of the survival function of the distribution if
            a scalar was passed. If an array like object was passed then a
            numpy array is returned with the value of the survival function at
            each corresponding value in the input array.

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
        if self.p == 1:
            sf = self.dist.sf(x - self.gamma, *self.params)
        else:
            sf = 1 - self.p * self.dist.ff(x - self.gamma, *self.params)
        return (1. - self.f0) * sf

    def ff(self, x):
        r"""

        The cumulative distribution function, or failure function, for a
        distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the failure function
            (CDF) will be calculated.

        Returns
        -------

        ff : scalar or numpy array
            The scalar value of the CDF of the distribution if a scalar was
            passed. If an array like object was passed then a numpy array is
            returned with the value of the CDF at each corresponding value in
            the input array.

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

        return self.f0 + ((1 - self.f0)
                          * self.p
                          * self.dist.ff(x - self.gamma, *self.params))

    def df(self, x):
        r"""

        The density function for a distribution using the parameters found
        in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the density function
            will be calculated.

        Returns
        -------

        df : scalar or numpy array
            The scalar value of the density function of the distribution if
            a scalar was passed. If an array like object was passed then a
            numpy array is returned with the value of the density function at
            each corresponding value in the input array.

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
        if self.f0 == 0:
            df = self.p * self.dist.df(x - self.gamma, *self.params)
        else:
            df = np.where(x == 0, self.f0,
                          ((1 - self.f0)
                           * self.p
                           * self.dist.df(x - self.gamma, *self.params)))
        return df

    def hf(self, x):
        r"""
        The instantaneous hazard function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the instantaneous
            hazard function will be calculated.

        Returns
        -------

        hf : scalar or numpy array
            The scalar value of the instantaneous hazard function of the
            distribution if a scalar was passed. If an array like object was
            passed then a numpy array is returned with the value of the
            instantaneous hazard function at each corresponding value in
            the input array.

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
        if self.p == 1:
            return self.dist.hf(x - self.gamma, *self.params)
        else:
            return 1 - self.p * self.dist.ff(x - self.gamma, *self.params)

    def Hf(self, x):
        r"""
        The cumulative hazard function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the cumulative
            hazard function will be calculated

        Returns
        -------

        Hf : scalar or numpy array
            The scalar value of the cumulative hazard function of the
            distribution if a scalar was passed. If an array like object was
            passed then a numpy array is returned with the value of the
            cumulative hazard function at each corresponding value in the
            input array.

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

        if self.p == 1:
            return self.dist.Hf(x - self.gamma, *self.params)
        else:
            return -np.log(self.sf(x))

    def qf(self, p):
        r"""

        The quantile function for a distribution using the parameters found
        in the ``.params`` attribute.

        Parameters
        ----------
        p : array like or scalar
            The values, which must be between 0 and 1, at which the the
            quantile will be calculated

        Returns
        -------
        qf : scalar or numpy array
            The scalar value of the quantile of the distribution if a
            scalar was passed. If an array like object was passed then a
            numpy array is returned with the value of the quantile at each
            corresponding value in the input array.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.qf(0.2)
        6.06542793124108
        >>> model.qf([.1, .2, .3, .4, .5])
        array([4.72308719, 6.06542793, 7.09181722, 7.99387877, 8.84997045])
        """
        if type(p) == list:
            p = np.array(p)
        if self.p == 1:
            return self.dist.qf(p, *self.params) + self.gamma
        else:
            raise NotImplementedError("Quantile for LFP not implemented.")

    def cs(self, x, X):
        r"""

        The conditional survival of the model.

        Parameters
        ----------

        x : array like or scalar
            The values at which conditional survival is to be calculated.
        X : array like or scalar
            The value(s) at which it is known the item has survived

        Returns
        -------

        cs : array
            The conditional survival probability.

        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.cs(11, 10)
        0.00025840046151723767
        """
        return self.dist.cs(x - self.gamma, X - self.gamma, *self.params)

    def random(self, size):
        r"""

        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------
        size : int
            The number of random samples to be drawn from the distribution.

        Returns
        -------
        random : numpy array
            Returns a numpy array of size ``size`` with random values
            drawn from the distribution.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> np.random.seed(1)
        >>> model.random(1)
        array([8.14127103])
        >>> model.random(10)
        array([10.84103403,  0.48542084,  7.11387062,  5.41420125, 4.59286657,
                5.90703589,  7.5124326 ,  7.96575225,  9.18134126, 8.16000438])
        """
        if (self.p == 1) and (self.f0 == 0):
            return self.dist.qf(uniform.rvs(size=size),
                                *self.params) + self.gamma
        elif (self.p != 1) and (self.f0 == 0):
            n_obs = np.random.binomial(size, self.p)

            f = self.dist.qf(uniform.rvs(size=n_obs),
                             *self.params) + self.gamma
            s = np.ones(np.array(size) - n_obs) * np.max(f) + 1

            return fs_to_xcn(f, s)

        elif (self.p == 1) and (self.f0 != 0):
            n_doa = np.random.binomial(size, self.f0)

            x0 = np.zeros(n_doa) + self.gamma
            x = self.dist.qf(uniform.rvs(size=size - n_doa),
                             *self.params) + self.gamma
            x = np.concatenate([x, x0])
            np.random.shuffle(x)

            return x
        else:
            N = np.random.multinomial(1, [self.f0,
                                          self.p - self.f0,
                                          1. - self.p], size).sum(axis=0)

            N = np.atleast_2d(N)
            n_doa, n_obs, n_cens = N[:, 0], N[:, 1], N[:, 2]
            x0 = np.zeros(n_doa) + self.gamma

            x = self.dist.qf(uniform.rvs(size=n_obs),
                             *self.params) + self.gamma

            f = np.concatenate([x, x0])
            s = np.ones(n_cens) * np.max(f) + 1
            return fs_to_xcn(f, s)

    def mean(self):
        r"""
        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Returns
        -------
        mean : float
            Returns the mean of the distribution.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.mean()
        8.929795115692489
        """
        if not hasattr(self, '_mean'):
            self._mean = self.p * (self.dist.mean(*self.params) + self.gamma)
        return self._mean

    def moment(self, n):
        r"""

        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------
        n : integer
            The degree of the moment to be computed

        Returns
        -------
        moment[n] : float
            Returns the n-th moment of the distribution

        References
        ----------
        INSERT WIKIPEDIA HERE

        Examples
        --------
        >>> from surpyval import Normal
        >>> model = Normal.from_params([10, 3])
        >>> model.moment(1)
        10.0
        >>> model.moment(5)
        202150.0
        """
        if self.p == 1:
            return self.dist.moment(n, *self.params)
        else:
            msg = 'LFP distributions cannot yet have their moment calculated'
            raise NotImplementedError(msg)

    def entropy(self):
        r"""
        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Returns
        -------

        entropy : float
            Returns entropy of the distribution

        Examples
        --------
        >>> from surpyval import Normal
        >>> model = Normal.from_params([10, 3])
        >>> model.entropy()
        2.588783247593625
        """
        if self.p == 1:
            return self.dist.entropy(*self.params)
        else:
            msg = "Entropy not available for LFP distribution"
            raise NotImplementedError(msg)

    def cb(self, t, on='R', alpha_ci=0.05, bound='two-sided'):
        r"""
        Confidence bounds of the ``on`` function at the ``alpa_ci`` level of
        significance. Can be the upper, lower, or two-sided confidence by
        changing value of ``bound``.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the confidence bounds
            will be calculated
        on : ('sf', 'ff', 'Hf'), optional
            The function on which the confidence bound will be calculated.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.

        Returns
        -------

        cb : scalar or numpy array
            The value(s) of the upper, lower, or both confidence bound(s) of
            the selected function at x

        """
        if self.method != 'MLE':
            raise Exception('Only MLE has confidence bounds')

        hess_inv = np.copy(self.hess_inv)

        pvars = hess_inv[np.triu_indices(hess_inv.shape[0])]
        old_err_state = np.seterr(all='ignore')

        if hasattr(self.dist, 'R_cb'):
            def R_cb(x):
                return self.dist.R_cb(x - self.gamma,
                                      *self.params,
                                      hess_inv,
                                      alpha_ci=alpha_ci,
                                      bound=bound)

        else:
            def R_cb(x):
                def sf_func(params):
                    return self.dist.sf(x - self.gamma, *params)
                jac = np.atleast_2d(jacobian(sf_func)(np.array(self.params)))

                # Second-Order Taylor Series Expansion of Variance
                var_R = []
                for i, j in enumerate(jac):
                    j = np.atleast_2d(j).T * j
                    j = j[np.triu_indices(j.shape[0])]
                    var_R.append(np.sum(j * pvars))

                # First-Order Taylor Series Expansion of Variance
                # var_R = (jac**2 * np.diag(hess_inv)).sum(axis=1).T

                R_hat = self.sf(x)
                if bound == 'two-sided':
                    diff = (z(alpha_ci / 2)
                            * np.sqrt(np.array(var_R))
                            * np.array([1., -1.]).reshape(2, 1))
                elif bound == 'upper':
                    diff = z(alpha_ci) * np.sqrt(np.array(var_R))
                else:
                    diff = -z(alpha_ci) * np.sqrt(np.array(var_R))

                exponent = diff / (R_hat * (1 - R_hat))
                R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
                return R_cb.T

        # Default cb is R
        cb = R_cb(t)

        if (on == 'ff') or (on == 'F'):
            cb = 1. - cb

        elif on == 'Hf':
            cb = -np.log(cb)

        elif on == 'hf':
            def cb_hf(x):
                out = []
                for v in x:
                    out.append(jacobian(lambda x: -np.log(R_cb(x)))(v))
                return np.concatenate(out)
            cb = cb_hf(t)

        elif on == 'df':
            def cb_df(x):
                out = []
                for v in x:
                    out.append(jacobian(lambda x: (-np.log(R_cb(x)))(v)
                                        * self.sf(v)))
                return np.concatenate(out)
            cb = cb_df(t)

        np.seterr(**old_err_state)
        return cb

    def neg_ll(self):
        r"""

        The the negative log-likelihood for the model, if it was fit with the
        ``fit()`` method. Not available if fit with the ``from_params()``
        method.

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

        The the Bayesian Information Criterion (BIC) for the model, if it
        was fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

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

        `Bayesian Information Criterion for Censored Survival Models
        <https://www.jstor.org/stable/2677130>`_.

        """
        if hasattr(self, '_bic'):
            return self._bic
        else:
            self._bic = (self.k
                         * np.log(self.data['n'][self.data['c'] == 0].sum())
                         + 2 * self.neg_ll())
            return self._bic

    def aic(self):
        r"""
        The the Aikake Information Criterion (AIC) for the model, if it was
        fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

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
        The the Corrected Aikake Information Criterion (AIC) for the model,
        if it was fit with the ``fit()`` method. Not available if fit with
        the ``from_params()`` method.

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
            self._aic_c = self.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
            return self._aic_c

    def get_plot_data(self, heuristic='Turnbull', alpha_ci=0.05):
        r"""

        A method to gather plot data

        Parameters
        ----------
        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean',
        'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None', 'Tukey',
        'DPW', 'Fleming-Harrington', 'Kaplan-Meier', 'Nelson-Aalen',
        'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probablility plot will
            be calculated.

        alpha_ci : float, optional
            The confidence with which the confidence bounds, if able, will
            be calculated. Defaults to 0.95.

        Returns
        -------
        data : dict
            Returns dictionary containing the data needed to do a plot.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> data = model.get_plot_data()
        """
        x_, r, d, F = nonp.plotting_positions(
            x=self.data['x'],
            c=self.data['c'],
            n=self.data['n'],
            t=self.data['t'],
            heuristic=heuristic)

        mask = np.isfinite(x_)
        x_ = x_[mask] - self.gamma
        r = r[mask]
        d = d[mask]
        F = F[mask]

        if np.isfinite(self.data['t']).any():
            Ftl = self.ff(x_[0])
            Ftr = self.ff(x_[-1])
            F = Ftl + F * (Ftr - Ftl)

        y_scale_min = np.min(F[F > 0]) / 2
        # y_scale_max = np.max(F[F < 1]) + (1 - np.max(F[F < 1]))/2
        y_scale_max = (1 - (1 - np.max(F[F < 1])) / 10)

        # x-axis
        if self.dist.plot_x_scale == 'log':
            log_x = np.log10(x_[x_ > 0])
            x_min = np.min(log_x)
            x_max = np.max(log_x)
            vals_non_sig = 10 ** np.linspace(x_min, x_max, 7)
            x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
            x_minor_ticks = ((10**x_minor_ticks * np.array(np.arange(1, 11))
                             .reshape((10, 1)))
                             .flatten())
            diff = (x_max - x_min) / 10
            x_scale_min = 10**(x_min - diff)
            x_scale_max = 10**(x_max + diff)
            x_model = 10**np.linspace(x_min - diff, x_max + diff, 100)
        elif self.dist.name in ('Beta'):
            x_min = np.min(x_)
            x_max = np.max(x_)
            x_scale_min = 0
            x_scale_max = 1
            vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
            x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
            x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
        elif self.dist.name in ('Uniform'):
            x_min = np.min(self.params)
            x_max = np.max(self.params)
            x_scale_min = x_min
            x_scale_max = x_max
            vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
            x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
            x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
        else:
            x_min = np.min(x_)
            x_max = np.max(x_)
            vals_non_sig = np.linspace(x_min, x_max, 7)
            x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
            diff = (x_max - x_min) / 10
            x_scale_min = x_min - diff
            x_scale_max = x_max + diff
            x_model = np.linspace(x_scale_min, x_scale_max, 100)

        cdf = self.ff(x_model + self.gamma)

        x_ticks = _round_vals(vals_non_sig)
        x_ticks_labels = [str(int(x))
                          if (re.match(r'([0-9]+\.0+)', str(x)) is not None)
                          & (x > 1)
                          else str(x)
                          for x in _round_vals(vals_non_sig + self.gamma)]

        y_ticks = np.array(self.dist.y_ticks)
        y_ticks = y_ticks[np.where((y_ticks > y_scale_min)
                          & (y_ticks < y_scale_max))[0]]

        y_ticks_labels = [str(int(y)) + '%'
                          if (re.match(r'([0-9]+\.0+)', str(y)) is not None)
                          & (y > 1)
                          else str(y) for y in y_ticks * 100]

        if hasattr(self, 'hess_inv') & (self.method == 'MLE'):
            if (self.hess_inv is not None):
                cbs = self.cb(x_model + self.gamma, on='ff', alpha_ci=alpha_ci)
            else:
                cbs = []
        else:
            cbs = []

        return {
            'x_scale_min': x_scale_min,
            'x_scale_max': x_scale_max,
            'y_scale_min': y_scale_min,
            'y_scale_max': y_scale_max,
            'y_ticks': y_ticks,
            'y_ticks_labels': y_ticks_labels,
            'x_ticks': x_ticks,
            'x_ticks_labels': x_ticks_labels,
            'cdf': cdf,
            'x_model': x_model,
            'x_minor_ticks': x_minor_ticks,
            'cbs': cbs,
            'x_scale': self.dist.plot_x_scale,
            'x_': x_,
            'F': F
        }

    def plot(self, heuristic='Turnbull', plot_bounds=True, alpha_ci=0.05):
        r"""
        A method to do a probability plot

        Parameters
        ----------
        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean',
        'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None', 'Tukey',
        'DPW', 'Fleming-Harrington', 'Kaplan-Meier', 'Nelson-Aalen',
        'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probablility plot will
            be calculated.

        plot_bounds : Boolean, optional
            A Boolean value to indicate whehter you want the probability
            bounds to be calculated.

        alpha_ci : float, optional
            The confidence with which the confidence bounds, if able, will
            be calculated. Defaults to 0.95.

        Returns
        -------
        plot : list
            list of a matplotlib plot object

        Examples
        --------
        >>> from surpyval import Weibull
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.plot()
        """
        if not hasattr(self, 'params'):
            print("Can't plot model that failed to fit")
            return None

        if self.method == 'given parameters':
            print("Can't plot model that was given parameters and no data")
            return None

        d = self.get_plot_data(heuristic=heuristic, alpha_ci=alpha_ci)
        # MAKE THE PLOT
        # Set the y limits
        plt.gca().set_ylim([d['y_scale_min'], d['y_scale_max']])

        # Set the x scale
        plt.xscale(d['x_scale'])
        # Set the y scale
        functions = (
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params))

        plt.gca().set_yscale('function', functions=functions)

        # Set Major Y axis ticks
        plt.yticks(d['y_ticks'], labels=d['y_ticks_labels'])
        # Set Minor Y axis ticks
        plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 51)))
        # Set Major X axis ticks
        plt.xticks(d['x_ticks'], labels=d['x_ticks_labels'])
        # Set Minor X axis ticks if log scale.
        if d['x_scale'] == 'log':
            plt.gca().set_xticks(d['x_minor_ticks'], minor=True)
            plt.gca().set_xticklabels([], minor=True)

        # Turn on the grid
        plt.grid(b=True, which='major', color='g', alpha=0.4, linestyle='-')
        plt.grid(b=True, which='minor', color='g', alpha=0.1, linestyle='-')

        # Label it
        plt.title('{} Probability Plot'.format(self.dist.name))
        plt.ylabel('CDF')
        plt.scatter(d['x_'], d['F'])
        plt.gca().set_xlim([d['x_scale_min'], d['x_scale_max']])
        if plot_bounds & (len(d['cbs']) != 0):
            plt.plot(d['x_model'], d['cbs'], color='r')
        return plt.plot(d['x_model'], d['cdf'], color='k', linestyle='--')
