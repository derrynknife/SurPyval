from surpyval import np
from scipy.integrate import quad

from ..nonparametric import plotting_positions as pp

import surpyval
from .parametric import Parametric

import pandas as pd
from copy import copy

from .fitters.mom import mom
from .fitters.mle import mle
from .fitters.mps import mps
from .fitters.mse import mse
from .fitters.mpp import mpp

from .fitters import bounds_convert, fix_idx_and_function

PARA_METHODS = ['MPP', 'MLE', 'MPS', 'MSE', 'MOM']
METHOD_FUNC_DICT = {
    'MPP': mpp,
    'MOM': mom,
    'MLE': mle,
    'MPS': mps,
    'MSE': mse
}


def init_from_bounds(dist):
    out = []
    for low, high in dist.bounds:
        if (low is None) & (high is None):
            out.append(0)
        elif high is None:
            out.append(low + 1.)
        elif low is None:
            out.append(high - 1.)
        else:
            out.append((high + low)/2.)

    return out


class ParametricFitter():
    def log_df(self, x, *params):
        return np.log(self.hf(x, *params)) - self.Hf(x, *params)

    def log_sf(self, x, *params):
        return -self.Hf(x, *params)

    def log_ff(self, x, *params):
        return np.log(self.ff(x, *params))

    def like(self, x, c, n, *params):
        like = np.zeros_like(x).astype(float)
        like = np.where(c == 0, self.df(x, *params), like)
        like = np.where(c == -1, self.ff(x, *params), like)
        like = np.where(c == 1, self.sf(x, *params), like)
        return like

    def log_like(self, x, c, n, p, f0, *params):
        # This is getting a bit much....
        like = np.zeros_like(x).astype(float)
        if f0 == 0:
            like = np.where(c == 0, np.log(p) + self.log_df(x, *params), like)
            like = np.where(c == -1, np.log(p) + self.log_ff(x, *params), like)
            if p == 1:
                like = np.where(c == 1, self.log_sf(x, *params), like)
            else:
                like = np.where(c == 1,
                                np.log(1 - (p * self.ff(x, *params))),
                                like)
        else:
            like = np.where(c == 0,
                            np.log(1 - f0) + np.log(p) +
                            self.log_df(x, *params),
                            like)
            like = np.where(c == -1,
                            np.log(1 - f0) + np.log(p) +
                            self.log_ff(x, *params),
                            like)
            like = np.where((c == 0) & (x == 0), np.log(f0), like)
            if p == 1:
                like = np.where(c == 1,
                                np.log(1 - f0) + self.log_sf(x, *params),
                                like)
            else:
                like = np.where(c == 1,
                                np.log(1 - f0) +
                                np.log(1 - (p * self.ff(x, *params))),
                                like)

        return like

    def log_like_i(self, x, c, n, inf_c_flags, p, f0, *params):
        ir = np.where(inf_c_flags[:, 1] == 1,
                      1,
                      (1 - f0) * p * self.ff(x[:, 1], *params))
        il = np.where(inf_c_flags[:, 0] == 1,
                      0,
                      (1 - f0) * p * self.ff(x[:, 0], *params))
        like_i = ir - il
        like_i = np.where(c != 2, 1., like_i)
        return np.log(like_i)

    def like_i(self, x, c, n, inf_c_flags, *params):
        # This makes sure that any intervals that are at the boundaries
        # of support or are infinite will not cause the autograd functions
        # to fail.
        ir = np.where(inf_c_flags[:, 1] == 1, 1, self.ff(x[:, 1], *params))
        il = np.where(inf_c_flags[:, 0] == 1, 0, self.ff(x[:, 0], *params))
        like_i = ir - il
        return like_i

    def like_t(self, t, t_flags, *params):
        # Needs to be updated to work with zi and ds models.
        # until then, can prevent it working in the `fit` method
        tr_denom = np.where(t_flags[:, 1] == 1, self.ff(t[:, 1], *params), 1.)
        tl_denom = np.where(t_flags[:, 0] == 1, self.ff(t[:, 0], *params), 0.)
        t_denom = tr_denom - tl_denom
        return t_denom

    def neg_ll(self, x, c, n, inf_c_flags, t, t_flags, gamma, p, f0, *params):
        x = copy(x) - gamma

        if 2 in c:
            like_i = self.log_like_i(x, c, n, inf_c_flags, p, f0, *params)
            x_ = copy(x[:, 0])
        else:
            like_i = 0
            x_ = copy(x)

        like = self.log_like(x_, c, n, p, f0, *params)
        like = like + like_i
        like = like - np.log(self.like_t(t, t_flags, *params))
        like = np.multiply(n, like)
        like = -np.sum(like)
        return like

    def neg_mean_D(self, x, c, n, *params):
        mask = c == 0
        x_obs = x[mask]
        n_obs = n[mask]

        gamma = 0

        # Assumes already ordered
        F = self.ff(x_obs - gamma, *params)
        D0 = F[0]
        Dn = 1 - F[-1]
        D = np.diff(F)
        D = np.concatenate([np.array([D0]), D.T, np.array([Dn])]).T

        Dr = self.sf(x[c == 1] - gamma, *params)
        Dl = self.ff(x[c == -1] - gamma, *params)

        if (n_obs > 1).any():
            n_ties = (n_obs - 1).sum()
            Df = self.df(x_obs - gamma, *params)
            # Df = Df[Df != 0]
            LL = np.concatenate([Dl, Df, Dr])
            ll_n = np.concatenate([n[c == -1], (n_obs - 1), n[c == 1]])
        else:
            Df = []
            n_ties = n_obs.sum()
            LL = np.concatenate([Dl, Dr])
            ll_n = np.concatenate([n[c == -1], n[c == 1]])

        M = np.log(D)
        M = -np.sum(M)/(M.shape[0])

        LL = -(np.log(LL) * ll_n).sum()/(n.sum() - n_obs.sum() + n_ties)
        return M + LL

    def _moment(self, n, *params, offset=False):
        if offset:
            gamma = params[0]
            params = params[1::]

            def fun(x):
                return x**n * self.df((x - gamma), *params)

            m = quad(fun, gamma, np.inf)[0]
        else:
            if hasattr(self, 'moment'):
                m = self.moment(n, *params)
            else:

                def fun(x):
                    return x**n * self.df(x, *params)

                m = quad(fun, *self.support)[0]
        return m

    def mom_moment_gen(self, *params, offset=False):
        if offset:
            k = self.k + 1
        else:
            k = self.k
        moments = np.zeros(k)
        for i in range(0, k):
            n = i + 1
            moments[i] = self._moment(n, *params, offset=offset)
        return moments

    def fit(self, x=None, c=None, n=None, t=None, how='MLE',
            offset=False, zi=False, lfp=False, tl=None, tr=None,
            xl=None, xr=None, fixed=None, heuristic='Turnbull',
            init=[], rr='y', on_d_is_0=False,
            turnbull_estimator='Fleming-Harrington'):

        r"""

        The central feature to SurPyval's capability. This function aimed to
        have an API to mimic the simplicity of the scipy API. That is, to use
        a simple :code:`fit()` call, with as many or as few parameters as
        is needed.

        Parameters
        ----------

        x : array like, optional
            Array of observations of the random variables. If x is
            :code:`None`, xl and xr must be provided.
        c : array like, optional
            Array of censoring flag. -1 is left censored, 0 is observed, 1 is
            right censored, and 2 is intervally censored. If not provided
            will assume all values are observed.
        n : array like, optional
            Array of counts for each x. If data is proivded as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.
        t : 2D-array like, optional
            2D array like of the left and right values at which the
            respective observation was truncated. If not provided it assumes
            that no truncation occurs.
        how : {'MLE', 'MPP', 'MOM', 'MSE', 'MPS'}, optional
            Method to estimate parameters, these are:

                - MLE : Maximum Likelihood Estimation
                - MPP : Method of Probability Plotting
                - MOM : Method of Moments
                - MSE : Mean Square Error
                - MPS : Maximum Product Spacing

        offset : boolean, optional
            If :code:`True` finds the shifted distribution. If not provided
            assumes not a shifted distribution. Only works with distributions
            that are supported on the half-real line.

        tl : array like or scalar, optional
            Values of left truncation for observations. If it is a scalar
            value assumes each observation is left truncated at the value.
            If an array, it is the respective 'late entry' of the observation

        tr : array like or scalar, optional
            Values of right truncation for observations. If it is a scalar
            value assumes each observation is right truncated at the value.
            If an array, it is the respective right truncation value for each
            observation

        xl : array like, optional
            Array like of the left array for 2-dimensional input of x. This
            is useful for data that is all intervally censored. Must be used
            with the :code:`xr` input.

        xr : array like, optional
            Array like of the right array for 2-dimensional input of x. This
            is useful for data that is all intervally censored. Must be used
            with the :code:`xl` input.

        fixed : dict, optional
            Dictionary of parameters and their values to fix. Fixes parameter
            by name.

        heuristic : {'"Blom", "Median", "ECDF", "Modal", "Midpoint", "Mean",
                      "Weibull", "Benard", "Beard", "Hazen", "Gringorten",
                      "None", "Tukey", "DPW", "Fleming-Harrington",
                      "Kaplan-Meier", "Nelson-Aalen", "Filliben",
                      "Larsen", "Turnbull"}
            Plotting method to use, if using the probability plotting,
            MPP, method.

        init : array like, optional
            initial guess of parameters. Useful if method is failing.

        rr : ('y', 'x')
            The dimension on which to minimise the spacing between the line
            and the observation. If 'y' the mean square error between the
            line and vertical distance to each point is minimised. If 'x' the
            mean square error between the line and horizontal distance to each
            point is minimised.

        on_d_is_0 : boolean, optional
            For the case when using MPP and the highest value is right
            censored, you can choosed to include this value into the
            regression analysis or not. That is, if :code:`False`, all values
            where there are 0 deaths are excluded from the regression. If
            :code:`True` all values regardless of whether there is a death
            or not are included in the regression.

        turnbull_estimator : ('Nelson-Aalen', 'Kaplan-Meier', or
                              'Fleming-Harrington'), str, optional
            If using the Turnbull heuristic, you can elect to use either the
            KM, NA, or FH estimator with the Turnbull estimates of r, and d.
            Defaults to FH.

        Returns
        -------

        model : Parametric
            A parametric model with the fitted parameters and methods for
            all functions of the distribution using the fitted parameters.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> x = Weibull.random(100, 10, 4)
        >>> model = Weibull.fit(x)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Parameters          :
             alpha: 10.551521182640098
              beta: 3.792549834495306
        >>> Weibull.fit(x, how='MPS', fixed={'alpha' : 10})
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MPS
        Parameters          :
             alpha: 10.0
              beta: 3.4314657446866836
        >>> Weibull.fit(xl=x-1, xr=x+1, how='MPP')
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MPP
        Parameters          :
             alpha: 9.943092756713078
              beta: 8.613016934518258
        >>> c = np.zeros_like(x)
        >>> c[x > 13] = 1
        >>> x[x > 13] = 13
        >>> c = c[x > 6]
        >>> x = x[x > 6]
        >>> Weibull.fit(x=x, c=c, tl=6)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Parameters          :
             alpha: 10.363725328793413
              beta: 4.9886821457305865
        """

        if offset and self.name in ['Normal', 'Beta', 'Uniform',
                                    'Gumbel', 'Logistic']:
            detail = '{} distribution cannot be offset'.format(self.name)
            raise ValueError(detail)

        if how not in PARA_METHODS:
            raise ValueError('"how" must be one of: ' + str(PARA_METHODS))

        if how == 'MPP' and self.name == 'ExpoWeibull':
            detail = 'ExpoWeibull distribution does not work' + \
                     ' with probability plot fitting'
            raise ValueError(detail)

        if t is not None and how == 'MPS':
            detail = 'Maximum product spacing doesn\'t yet support tuncation'
            raise NotImplementedError(detail)

        if t is not None and how == 'MSE':
            detail = 'Mean square error doesn\'t yet support tuncation'
            raise NotImplementedError(detail)

        if t is not None and how == 'MOM':
            detail = 'Maximum product spacing doesn\'t support tuncation'
            raise ValueError(detail)

        if (lfp or zi) & (how != 'MLE'):
            detail = 'Limited failure or zero-inflated models' + \
                     ' can only be made with MLE'
            raise ValueError(detail)

        if (zi & (self.support[0] != 0)):
            detail = "zero-inflated models can only work" + \
                     "with models starting at 0"
            raise ValueError()

        x, c, n, t = surpyval.xcnt_handler(x=x, c=c, n=n, t=t,
                                           tl=tl, tr=tr, xl=xl, xr=xr)

        if (c == 1).all():
            raise ValueError("Cannot have only right censored data")

        if (c == -1).all():
            raise ValueError("Cannot have only left censored data")

        if surpyval.utils.check_no_censoring(c) and (how == 'MOM'):
            raise ValueError('Method of moments doesn\'t support censoring')

        if (surpyval.utils.no_left_or_int(c)) and \
           (how == 'MPP') and \
           (not heuristic == 'Turnbull'):
            detail = 'Probability plotting estimation with left or ' + \
                     'interval censoring only works with Turnbull heuristic'
            raise ValueError()

        if (heuristic == 'Turnbull') and \
           (not ((-1 in c) or (2 in c))) and \
           ((~np.isfinite(t[:, 1])).any()):
            # The Turnbull method is extremely memory intensive.
            # So if no left or interval censoring and no right-truncation
            # then this is equivalent.
            heuristic = turnbull_estimator

        if (not offset) & (not zi):
            detail_template = """
            Some of your data is outside support of distribution, observed
            values must be within [{lower}, {upper}].

            Are some of your observed values 0, -Inf, or Inf?
            """

            if x.ndim == 2:
                if ((x[:, 0] <= self.support[0]) & (c == 0)).any():
                    detail = detail_template.format(lower=self.support[0],
                                                    upper=self.support[1])
                    raise ValueError(detail)
                elif ((x[:, 1] >= self.support[1]) & (c == 0)).any():
                    detail = detail_template.format(lower=self.support[0],
                                                    upper=self.support[1])
                    raise ValueError(detail)
            else:
                if ((x <= self.support[0]) & (c == 0)).any():
                    detail = detail_template.format(lower=self.support[0],
                                                    upper=self.support[1])
                    raise ValueError(detail)
                elif ((x >= self.support[1]) & (c == 0)).any():
                    detail = detail_template.format(lower=self.support[0],
                                                    upper=self.support[1])
                    raise ValueError(detail)

        # Passed checks
        data = {
            'x': x,
            'c': c,
            'n': n,
            't': t
        }

        model = Parametric(self, how, data, offset, lfp, zi)
        fitting_info = {}

        if how != 'MPP':
            transform, inv_trans, funcs, inv_f = bounds_convert(x,
                                                                model.bounds)
            const, fixed_idx, not_fixed = fix_idx_and_function(fixed,
                                                               model.param_map,
                                                               funcs)

            fitting_info['transform'] = transform
            fitting_info['inv_trans'] = inv_trans
            fitting_info['funcs'] = funcs
            fitting_info['inv_f'] = inv_f

            fitting_info['const'] = const
            fitting_info['fixed_idx'] = fixed_idx
            fitting_info['not_fixed'] = not_fixed

            # This initial estimation is just terrible
            if init == []:
                if self.name in ['Gumbel', 'Beta', 'Normal', 'Uniform']:
                    with np.errstate(all='ignore'):
                        init = np.array(self._parameter_initialiser(x, c, n))
                else:
                    with np.errstate(all='ignore'):
                        if x.ndim == 2:
                            x_arr = x[:, 0]
                            init_mask = np.logical_or(x_arr <= self.support[0],
                                                      x_arr >= self.support[1])
                            init_mask = ~np.logical_and(init_mask, c == 0)
                            xl = x[init_mask, 0]
                            xr = x[init_mask, 1]
                            x_init = np.vstack([xl, xr]).T
                        else:
                            init_mask = np.logical_or(x <= self.support[0],
                                                      x >= self.support[1])
                            init_mask = ~np.logical_and(init_mask, c == 0)

                            x_init = x[init_mask]

                        c_init = c[init_mask]
                        n_init = n[init_mask]

                        init = self._parameter_initialiser(x_init,
                                                           c_init,
                                                           n_init,
                                                           offset=offset)
                        init = np.array(init)
                        if offset:
                            init[0] = x.min() - 1.

                if lfp:
                    _, _, _, F = pp(x, c, n, heuristic='Nelson-Aalen')

                    max_F = np.max(F)

                    if max_F > 0.5:
                        init = np.concatenate([init, [0.99]])
                    else:
                        init = np.concatenate([init_from_bounds(self),
                                              [max_F]])

                if zi:
                    init = np.concatenate([init, [(n[x == 0]).sum()/n.sum()]])

            init = transform(init)
            init = init[not_fixed]
            fitting_info['init'] = init
        else:
            # Probability plotting method does not need an initial estimate
            fitting_info['rr'] = rr
            fitting_info['heuristic'] = heuristic
            fitting_info['on_d_is_0'] = on_d_is_0
            fitting_info['turnbull_estimator'] = turnbull_estimator
            fitting_info['init'] = None

        model.fitting_info = fitting_info

        results = METHOD_FUNC_DICT[how](model)

        for k, v in results.items():
            setattr(model, k, v)

        if hasattr(model, 'params'):
            for k, v in zip(self.param_names, model.params):
                setattr(model, k, v)

        return model

    def fit_from_df(self, df, x=None, c=None, n=None,
                    xl=None, xr=None, tl=None, tr=None,
                    **fit_options):
        r"""
        The central feature to SurPyval's capability. This function aimed to
        have an API to mimic the simplicity of the scipy API. That is, to use
        a simple :code:`fit()` call, with as many or as few parameters as
        is needed.

        Parameters
        ----------

        df : DataFrame
            DataFrame of data to be used to create surpyval model

        x : string, optional
            column name for the column in df containing the variable data.
            If not provided must provide both xl and xr.

        c : string, optional
            column name for the column in df containing the censor flag of x.
            If not provided assumes all values of x are observed.

        n : string, optional
            column name in for the column in df containing the counts of x.
            If not provided assumes each x is one observation.

        tl : string or scalar, optional
            If string, column name in for the column in df containing the left
            truncation data. If scalar assumes each x is left truncated by
            that value. If not provided assumes x is not left truncated.

        tr : string or scalar, optional
            If string, column name in for the column in df containing the
            right truncation data. If scalar assumes each x is right truncated
            by that value. If not provided assumes x is not right truncated.

        xl : string, optional
            column name for the column in df containing the left interval for
            interval censored data. If left interval is -Inf, assumes left
            censored. If xl[i] == xr[i] assumes observed. Cannot be provided
            with x, must be provided with xr.

        xr : string, optional
            column name for the column in df containing the right interval
            for interval censored data. If right interval is Inf, assumes
            right censored. If xl[i] == xr[i] assumes observed. Cannot be
            provided with x, must be provided with xl.

        fit_options : dict, optional
            dictionary of fit options that will be passed to the :code:`fit`
            method, see that method for options.

        Returns
        -------

        model : Parametric
            A parametric model with the fitted parameters and methods for
            all functions of the distribution using the fitted parameters.


        Examples
        --------
        >>> import surpyval as surv
        >>> df = surv.datasets.BoforsSteel.df
        >>> model = surv.Weibull.fit_from_df(df, x='x', n='n', offset=True)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Offset (gamma)      : 39.76562962867477
        Parameters          :
             alpha: 7.141925216146524
              beta: 2.6204524040137844
        """

        if not type(df) == pd.DataFrame:
            raise ValueError("df must be a pandas DataFrame")

        if (x is not None) and ((xl is not None) or (xr is not None)):
            raise ValueError("Cannot use `x` and (`xl` and `xr`) together")

        if x is not None:
            x = df[x].astype(float)
        else:
            xl = df[xl].astype(float)
            xr = df[xr].astype(float)
            x = np.vstack([xl, xr]).T

        if c is not None:
            c = df[c].values.astype(int)

        if n is not None:
            n = df[n].values.astype(int)

        if tl is not None:
            if type(tl) == str:
                tl = df[tl].values.astype(float)
            elif np.isscalar(tl):
                tl = (np.ones(df.shape[0]) * tl).astype(float)
            else:
                raise ValueError('`tl` must be scalar or column label string')
        else:
            tl = np.ones(df.shape[0]) * -np.inf

        if tr is not None:
            if type(tr) == str:
                tr = df[tr].values.astype(float)
            elif np.isscalar(tr):
                tr = (np.ones(df.shape[0]) * tr).astype(float)
            else:
                detail = "`tr` must be scalar or a column label string"
                raise ValueError(detail)
        else:
            tr = np.ones(df.shape[0]) * np.inf

        t = np.vstack([tl, tr]).T

        return self.fit(x=x, c=c, n=n, t=t, **fit_options)

    def from_params(self, params, gamma=None, p=None, f0=None):
        r"""

        Creating a SurPyval Parametric class with provided parameters.

        Parameters
        ----------

        params : array like
            array of the parameters of the distribution.

        gamma : scalar, optional
            offset value for the distribution. If not provided will fit a
            regular, unshifted/not offset, distribution.


        Returns
        -------

        model : Parametric
            A parametric model with the fitted parameters and methods for all
            functions of the distribution using the
            fitted parameters.


        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 4])
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : given parameters
        Parameters          :
             alpha: 10
              beta: 4
        >>> model = Weibull.from_params([10, 4], gamma=2)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : given parameters
        Offset (gamma)      : 2
        Parameters          :
             alpha: 10
              beta: 4
        """
        if self.k != len(params):
            msg_base = "Must have {k} params for {dist} distribution"
            detail = msg_base.format(k=self.k, dist=self.name)
            raise ValueError(detail)

        if gamma is not None and self.name in ['Normal', 'Beta', 'Uniform',
                                               'Gumbel', 'Logistic']:
            msg_base = '{dist} distribution cannot be offset'
            detail = msg_base.format(dist=self.name)
            raise ValueError(detail)

        if gamma is not None:
            offset = True
        else:
            offset = False
            gamma = 0

        if p is not None:
            lfp = True
        else:
            lfp = False
            p = 1

        if f0 is not None:
            zi = True
        else:
            zi = False
            f0 = 0

        model = Parametric(self, 'given parameters',
                                None, offset, lfp, zi)
        model.gamma = gamma
        model.p = p
        model.f0 = f0
        model.params = np.array(params)

        for i, (low, upp) in enumerate(self.bounds):
            if low is None:
                lower_limit = -np.inf
            else:
                lower_limit = low
            if upp is None:
                upper_limit = np.inf
            else:
                upper_limit = upp

            if not ((lower_limit < params[i]) & (params[i] < upper_limit)):
                params = ', '.join(self.param_names)
                base = "Params {params} must be in bounds {bounds}"
                detail = base.format(params=params, bounds=self.bounds)
                raise ValueError(detail)
        model.dist = self
        return model
