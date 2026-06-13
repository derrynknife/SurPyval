import json
from copy import copy, deepcopy

import matplotlib.pyplot as plt
from autograd import jacobian
from scipy.special import ndtri as z
from scipy.stats import uniform

import surpyval as surv
from surpyval import Distribution, np
from surpyval.utils import fsli_to_xcnt

from .probability_plotting import (
    adjust_heuristic,
    draw_probability_plot,
    probability_plot_data,
)


class Parametric(Distribution):
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
                x_min = np.asarray(data["x"])
                if zi:
                    # Exact zeros belong to the zero-inflation mass, so
                    # they must not cap the offset of the continuous part
                    x_min = x_min[x_min != 0]
                bounds = ((None, np.min(x_min)), *bounds)
            else:
                bounds = ((None, None), *bounds)

            param_map = {k: v + 1 for k, v in param_map.items()}
            param_map.update({"gamma": 0})
            self.k += 1
        else:
            self.gamma = 0

        if lfp:
            bounds = (*bounds, (0, 1))
            param_map.update({"p": len(param_map)})
            self.k += 1
        else:
            self.p = 1

        if zi:
            bounds = (*bounds, (0, 1))
            param_map.update({"f0": len(param_map)})
            self.k += 1
        else:
            self.f0 = 0

        self.bounds = bounds
        self.param_map = param_map

    @classmethod
    def from_json(cls, fp):
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, model_dict):
        # Imported here since parametric_fitter imports this module
        from surpyval.univariate.parametric.parametric_fitter import (
            ParametricFitter,
        )

        if model_dict["parameterization"] != "parametric":
            raise ValueError(
                "Must create parametric model from parametric model dict"
            )

        # Restrict the lookup to known distributions so an untrusted
        # model dict cannot resolve arbitrary surpyval attributes
        dist = getattr(surv, model_dict["distribution"], None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                f"Unknown distribution '{model_dict['distribution']}'"
            )
        how = model_dict["how"]
        if "data" in model_dict:
            data = model_dict["data"]
        else:
            data = None
        offset = model_dict["offset"]
        lfp = model_dict["lfp"]
        zi = model_dict["zi"]
        out = cls(dist, how, data, offset, lfp, zi)

        if offset:
            out.gamma = model_dict["gamma"]

        if lfp:
            out.p = model_dict["p"]

        if zi:
            out.f0 = model_dict["f0"]

        if "hess_inv" in model_dict:
            out.hess_inv = np.array(model_dict["hess_inv"])

        if "cov_matrix" in model_dict:
            out.cov_matrix = np.array(model_dict["cov_matrix"])

        if "_neg_ll" in model_dict:
            out._neg_ll = model_dict["_neg_ll"]

        out.params = np.array(model_dict["params"])

        return out

    def to_dict(self, with_data=False):
        out = {}
        out["parameterization"] = "parametric"
        out["distribution"] = self.dist.name
        out["how"] = self.method
        out["param_names"] = self.dist.param_names

        data_dict = {}
        if with_data:
            if self.data is not None:
                for ch in ["x", "c", "n", "t"]:
                    if self.data[ch] is None:
                        data_dict[ch] = []
                    else:
                        data_dict[ch] = self.data[ch].tolist()
                out["data"] = data_dict

        out["params"] = np.array(self.params).tolist()
        out["lfp"] = self.lfp

        if self.lfp:
            out["p"] = self.p
        else:
            out["p"] = 1.0

        out["zi"] = self.zi
        if self.zi:
            out["f0"] = self.f0
        else:
            out["f0"] = 0.0

        out["offset"] = self.offset
        if self.offset:
            out["gamma"] = self.gamma
        else:
            out["gamma"] = 0.0

        if hasattr(self, "hess_inv"):
            if self.hess_inv is None:
                pass
            else:
                out["hess_inv"] = self.hess_inv.tolist()
        if getattr(self, "cov_matrix", None) is not None:
            out["cov_matrix"] = self.cov_matrix.tolist()
        if hasattr(self, "_neg_ll"):
            out["_neg_ll"] = self._neg_ll

        return out

    def to_json(self, fp):
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    def __repr__(self):
        if hasattr(self, "params"):
            param_string = "\n".join(
                [
                    f"{name:>10}: {p}"
                    for p, name in zip(self.params, self.dist.param_names)
                ]
            )
            out = (
                "Parametric SurPyval Model"
                "\n========================="
                f"\nDistribution        : {self.dist.name}"
                f"\nFitted by           : {self.method}"
            )
            if self.offset:
                out += f"\nOffset (gamma)      : {self.gamma}"

            if self.lfp:
                out += f"\nMax Proportion (p)  : {self.p}"

            if self.zi:
                out += f"\nZero-Inflation (f0) : {self.f0}"

            out = out + "\nParameters          :\n" + param_string

            return out
        else:
            return "Unable to fit values"

    def param_cb(self, name, alpha_ci=0.05, bound="two-sided"):
        """
        Method to calculate the confidence bound on a parameter.
        """
        if name in ("p", "f0"):
            if name == "p" and not self.lfp:
                raise ValueError("'p' is only estimated for lfp models")
            if name == "f0" and not self.zi:
                raise ValueError("'f0' is only estimated for zi models")
            cov = getattr(self, "cov_matrix", None)
            if cov is None:
                raise ValueError(
                    f"Model has no covariance for '{name}'; "
                    "it must be fit with the MLE method"
                )
            idx = len(self.params)
            if name == "f0" and self.lfp:
                idx += 1
            p_hat = self.p if name == "p" else self.f0
            var = cov[idx, idx]
            param_bounds = (0, 1)
        else:
            idx = self.dist.param_map[name]
            p_hat = self.params[idx]
            var = self.hess_inv[idx, idx]
            param_bounds = self.dist.bounds[idx]

        if bound == "two-sided":
            alpha = alpha_ci / 2
            bounds = np.array([-1, 1])
        elif bound == "lower":
            alpha = alpha_ci
            bounds = np.array([-1])
        elif bound == "upper":
            alpha = alpha_ci
            bounds = np.array([1])

        if param_bounds == (0, None):
            exponent = z(alpha) * np.sqrt(var) / p_hat
            bounds = -bounds * exponent
            return p_hat * np.exp(bounds)
        elif param_bounds == (0, 1):
            # Bounds on the logit keep the result within (0, 1)
            u_hat = np.log(p_hat / (1 - p_hat))
            diff = -bounds * z(alpha) * np.sqrt(var) / (p_hat * (1 - p_hat))
            return 1 / (1 + np.exp(-(u_hat + diff)))
        else:
            factor = z(alpha) * np.sqrt(var)
            bounds = -bounds * factor
            return p_hat + bounds

    def sf(self, x):
        r"""

        Survival (or Reliability) function for a distribution using the
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
        if isinstance(x, list):
            x = np.array(x)
        return (
            1
            - self.p
            + (self.p - self.f0) * self.dist.sf(x - self.gamma, *self.params)
        )

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
        if isinstance(x, list):
            x = np.array(x)

        return self.f0 + (self.p - self.f0) * self.dist.ff(
            x - self.gamma, *self.params
        )

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
        if isinstance(x, list):
            x = np.array(x)
        if self.f0 == 0:
            df = self.p * self.dist.df(x - self.gamma, *self.params)
        else:
            df = np.where(
                x == 0,
                self.f0,
                (
                    (1 - self.f0)
                    * self.p
                    * self.dist.df(x - self.gamma, *self.params)
                ),
            )
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
        if isinstance(x, list):
            x = np.array(x)
        if self.p == 1:
            return self.dist.hf(x - self.gamma, *self.params)
        else:
            return self.df(x) / self.sf(x)

    def Hf(self, x):
        """
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
        if isinstance(x, list):
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
        if isinstance(p, list):
            p = np.array(p)
        if self.p == 1:
            return self.dist.qf(p, *self.params) + self.gamma
        else:
            raise NotImplementedError("Quantile for LFP not implemented.")

    def cs(self, x, X):
        r"""

        The conditional survival of the model; that is, the probability
        that an item that has survived to ``X`` survives a further ``x``:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : array like or scalar
            The further durations at which conditional survival is to be
            calculated.
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
        x = np.array(x)
        cs = np.array(self.dist.cs(x, X - self.gamma, *self.params))
        cs[cs > 1.0] = 1
        return cs

    def random(self, size, a=None, b=None):
        r"""

        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------
        size : int
            The number of random samples to be drawn from the distribution.
        a: float or None
            The left truncated value if sampling from a truncated
            distribution
        b: float or None
            The right truncated value if sampling from a truncated
            distribution

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
        if ((a is not None) or (b is not None)) and (
            (self.p != 1) or (self.f0 != 0)
        ):
            raise NotImplementedError(
                "Truncated sampling not supported with LFP or ZI models"
            )
        elif ((a is not None) or (b is not None)) and self.offset:
            raise NotImplementedError(
                "Truncated sampling not supported with offset distributions"
            )

        if (self.p == 1) and (self.f0 == 0):
            if (a is None) and (b is None):
                if hasattr(self.dist, "qf"):
                    return (
                        self.dist.qf(uniform.rvs(size=size), *self.params)
                        + self.gamma
                    )
                else:
                    return self.dist.random(size, *self.params)

            else:
                # Truncated sampling
                # F-1(u) = G-1[u(G(b) - G(a)) + G(a)]
                if a is None:
                    Fa = 0
                else:
                    Fa = self.dist.ff(a, *self.params)
                if b is None:
                    Fb = 1
                else:
                    Fb = self.dist.ff(b, *self.params)
                u = uniform.rvs(size=size)
                return self.dist.qf((u * (Fb - Fa) + Fa), *self.params)

        elif (self.p != 1) and (self.f0 == 0):
            n_obs = np.random.binomial(size, self.p)

            f = (
                self.dist.qf(uniform.rvs(size=n_obs), *self.params)
                + self.gamma
            )
            s = np.ones(np.array(size) - n_obs) * np.max(f) + 1

            return fsli_to_xcnt(f, s)

        elif (self.p == 1) and (self.f0 != 0):
            n_doa = np.random.binomial(size, self.f0)

            x0 = np.zeros(n_doa) + self.gamma
            x = (
                self.dist.qf(uniform.rvs(size=size - n_doa), *self.params)
                + self.gamma
            )
            x = np.concatenate([x, x0])
            np.random.shuffle(x)

            return x
        else:
            N = np.random.multinomial(
                1, [self.f0, self.p - self.f0, 1.0 - self.p], size
            ).sum(axis=0)

            N = np.atleast_2d(N)
            n_doa, n_obs, n_cens = N[:, 0], N[:, 1], N[:, 2]
            x0 = np.zeros(n_doa) + self.gamma

            x = (
                self.dist.qf(uniform.rvs(size=n_obs), *self.params)
                + self.gamma
            )

            f = np.concatenate([x, x0])
            s = np.ones(n_cens) * np.max(f) + 1
            return fsli_to_xcnt(f, s)

    def mean(self):
        r"""
        The mean of the distribution using the parameters found in the
        ``.params`` attribute.

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
        if not hasattr(self, "_mean"):
            self._mean = self.p * (self.dist.mean(*self.params) + self.gamma)
        return self._mean

    def var(self):
        r"""
        The variance of the distribution using the parameters found in the
        ``.params`` attribute.

        Returns
        -------
        var : float
            Returns the variance of the distribution.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.var()
        11.229...
        """
        m1 = self.dist._moment(1, *self.params)
        m2 = self.dist._moment(2, *self.params)
        return m2 - m1**2

    def moment(self, n):
        r"""

        The n-th moment of the distribution using the parameters found
        in the ``.params`` attribute.

        Parameters
        ----------
        n : integer
            The degree of the moment to be computed

        Returns
        -------
        moment[n] : float
            Returns the n-th moment of the distribution

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
            msg = "LFP distributions cannot yet have their moment calculated"
            raise NotImplementedError(msg)

    def entropy(self):
        r"""
        The entropy of the distribution using the parameters found in
        the ``.params`` attribute.

        Returns
        -------

        entropy : float
            Returns entropy of the distribution

        Examples
        --------
        >>> from surpyval import Normal
        >>> model = Normal.from_params([10, 3])
        >>> model.entropy()
        2.5175508218727822
        """
        if self.p == 1:
            return self.dist.entropy(*self.params)
        else:
            msg = "Entropy not available for LFP distribution"
            raise NotImplementedError(msg)

    def cb(self, t, on="R", alpha_ci=0.05, bound="two-sided"):
        r"""
        Confidence bounds of the ``on`` function at the ``alpa_ci`` level of
        significance. Can be the upper, lower, or two-sided confidence by
        changing value of ``bound``.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the confidence bounds
            will be calculated
        on : ('sf', 'ff', 'Hf', 'hf', 'df'), optional
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
        t = np.atleast_1d(t)
        if self.method != "MLE":
            raise ValueError("Only MLE has confidence bounds")

        hess_inv = np.copy(self.hess_inv)

        # The variance is computed over the extended parameter vector
        # (*params, p?, f0?) so that the uncertainty of the LFP and
        # zero-inflation parameters widens the bounds. gamma is held
        # fixed: the threshold parameter is non-regular, so it carries
        # no Wald variance. Models deserialized without a cov_matrix
        # fall back to treating p and f0 as fixed.
        n_core = len(self.params)
        phi_hat = list(self.params)
        if self.lfp:
            phi_hat.append(self.p)
        if self.zi:
            phi_hat.append(self.f0)
        phi_hat = np.array(phi_hat)

        cov = getattr(self, "cov_matrix", None)
        if cov is None:
            cov = np.zeros((len(phi_hat), len(phi_hat)))
            cov[:n_core, :n_core] = hess_inv

        def unpack(phi):
            core = phi[:n_core]
            i = n_core
            if self.lfp:
                p = phi[i]
                i += 1
            else:
                p = 1.0
            f0 = phi[i] if self.zi else 0.0
            return core, p, f0

        def full_sf(x, phi):
            core, p, f0 = unpack(phi)
            return 1 - p + (p - f0) * self.dist.sf(x - self.gamma, *core)

        old_err_state = np.seterr(all="ignore")

        def delta_method_var(func):
            # First-order delta method: Var(g) = J Sigma J^T
            jac = np.atleast_2d(jacobian(func)(phi_hat))
            return np.einsum("ij,jk,ik->i", jac, cov, jac)

        def sf_cb(x, bound=bound):
            def sf_func(phi):
                return full_sf(x, phi)

            var_R = delta_method_var(sf_func)
            R_hat = full_sf(x, phi_hat)
            if bound == "two-sided":
                diff = (
                    z(alpha_ci / 2)
                    * np.sqrt(var_R)
                    * np.array([1.0, -1.0]).reshape(2, 1)
                )
            elif bound == "upper":
                diff = z(alpha_ci) * np.sqrt(var_R)
            else:
                diff = -z(alpha_ci) * np.sqrt(var_R)

            # Bounds on the logit of R keep the result within (0, 1)
            exponent = diff / (R_hat * (1 - R_hat))
            R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
            return R_cb.T

        # ff, F and Hf are decreasing transforms of R; flip one-sided bounds
        if on in ["ff", "F", "Hf"] and bound == "lower":
            bound = "upper"
        elif on in ["ff", "F", "Hf"] and bound == "upper":
            bound = "lower"

        if (on == "ff") or (on == "F"):
            cb = 1.0 - sf_cb(t, bound=bound)
        elif (on == "sf") or (on == "R"):
            cb = sf_cb(t, bound=bound)
            if bound == "two-sided":
                cb = np.fliplr(cb)
        elif on == "Hf":
            cb = -np.log(sf_cb(t, bound=bound))
        elif on in ["hf", "df"]:

            def density(phi):
                core, p, f0 = unpack(phi)
                return (p - f0) * self.dist.df(t - self.gamma, *core)

            if on == "hf":

                def func(phi):
                    return density(phi) / full_sf(t, phi)

            else:
                func = density

            g_hat = func(phi_hat)
            var_g = delta_method_var(func)

            if bound == "two-sided":
                diff = z(alpha_ci / 2) * np.array([1.0, -1.0]).reshape(2, 1)
            elif bound == "upper":
                diff = -z(alpha_ci)
            else:
                diff = z(alpha_ci)

            # Bounds on the log of hf/df keep the result positive
            cb = g_hat * np.exp(diff * np.sqrt(var_g) / g_hat)
            if bound == "two-sided":
                cb = cb.T

        np.seterr(**old_err_state)
        return cb

    def neg_ll(self):
        r"""

        The negative log-likelihood for the model, if it was fit with the
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
        if self.data is None:
            raise ValueError("Must have been fit with data")

        return self._neg_ll

    def bic(self):
        r"""

        The Bayesian Information Criterion (BIC) for the model, if it
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

        References
        ----------

        `Bayesian Information Criterion for Censored Survival Models
        <https://www.jstor.org/stable/2677130>`_.

        """
        if hasattr(self, "_bic"):
            return self._bic
        else:
            self._bic = (
                self.k * np.log(self.data["n"][self.data["c"] == 0].sum())
                + 2 * self.neg_ll()
            )
            return self._bic

    def aic(self):
        r"""
        The Aikake Information Criterion (AIC) for the model, if it was
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
        if hasattr(self, "_aic"):
            return self._aic
        else:
            self._aic = 2 * self.k + 2 * self.neg_ll()
            return self._aic

    def aic_c(self):
        r"""
        The Corrected Aikake Information Criterion (AIC) for the model,
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
        if hasattr(self, "_aic_c"):
            return self._aic_c
        else:
            k = len(self.params)
            n = self.data["n"].sum()
            self._aic_c = self.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
            return self._aic_c

    def get_plot_data(self, heuristic="Nelson-Aalen", alpha_ci=0.05):
        """

        A method to gather plot data

        Parameters
        ----------

        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean',\
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probability plot will
            be calculated. Default is "Nelson-Aalen".

        alpha_ci : float, optional
            The level of significance at which the confidence bounds, if
            able, will be calculated. Defaults to 0.05.

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
        if hasattr(self, "hess_inv") and (self.method == "MLE"):
            if self.hess_inv is not None:

                def cb_func(x_model):
                    return self.cb(x_model, on="ff", alpha_ci=alpha_ci)

            else:
                cb_func = None
        else:
            cb_func = None

        return probability_plot_data(
            dist=self.dist,
            ff=self.ff,
            x=self.data["x"],
            c=self.data["c"],
            n=self.data["n"],
            t=self.data["t"],
            heuristic=heuristic,
            gamma=self.gamma,
            params=self.params,
            cb_func=cb_func,
        )

    def plot(
        self,
        heuristic="Nelson-Aalen",
        plot_bounds=True,
        alpha_ci=0.05,
        ax=None,
    ):
        """
        A method to do a probability plot

        Parameters
        ----------

        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', \
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probability plot will
            be calculated.

        plot_bounds : Boolean, optional
            A Boolean value to indicate whether you want the probability
            bounds to be calculated.

        alpha_ci : float, optional
            The level of significance at which the confidence bounds, if
            able, will be calculated. Defaults to 0.05.

        ax: matplotlib.axes.Axes, optional
            The axis onto which the plot will be created. Optional, if not
            provided a new axes will be created.

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
        if ax is None:
            ax = plt.gcf().gca()

        if not hasattr(self, "params"):
            raise ValueError("Can't plot model that failed to fit")

        if self.method == "given parameters":
            detail = "Can't plot model that was given parameters and no data"
            raise ValueError(detail)

        if not (
            hasattr(self.dist, "mpp_y_transform")
            and hasattr(self.dist, "mpp_inv_y_transform")
        ):
            raise NotImplementedError(
                f"{self.dist.name} does not support probability plotting"
            )

        heuristic = adjust_heuristic(self.data["c"], self.data["t"], heuristic)

        d = self.get_plot_data(heuristic=heuristic, alpha_ci=alpha_ci)

        return draw_probability_plot(
            ax,
            d,
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params),
            title=f"{self.dist.name} Probability Plot",
            plot_bounds=plot_bounds,
        )
