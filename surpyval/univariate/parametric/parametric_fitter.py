import pandas as pd
from scipy.integrate import quad
from scipy.special import expit

import surpyval
from surpyval import np
from surpyval.utils import _check_x_not_empty
from surpyval.utils.surpyval_data import SurpyvalData

from ..nonparametric import plotting_positions as pp
from .fitters import bounds_convert
from .fitters.mle import mle
from .fitters.mom import mom
from .fitters.mpp import mpp, mpp_from_ecfd
from .fitters.mps import mps
from .fitters.mse import mse
from .parametric import Parametric

PARA_METHODS = ["MPP", "MLE", "MPS", "MSE", "MOM"]
METHOD_FUNC_DICT = {"MPP": mpp, "MOM": mom, "MLE": mle, "MPS": mps, "MSE": mse}


class ParametricFitter:
    def __init__(
        self,
        name: str,
        k: int,
        bounds: tuple[tuple[int | float | None, int | float | None], ...],
        support: tuple[int | float, int | float],
        param_names: list[str],
        param_map: dict[str, int],
        plot_x_scale: str,
        y_ticks: list[float],
    ):
        self.name: str = name
        self.k = k
        self.bounds = bounds
        self.support = support
        self.param_names = param_names
        self.param_map = param_map
        self.plot_x_scale = plot_x_scale
        self.y_ticks = y_ticks

    def log_df(self, x, *params):
        return np.log(self.hf(x, *params)) - self.Hf(x, *params)

    def log_sf(self, x, *params):
        return -self.Hf(x, *params)

    def log_ff(self, x, *params):
        return np.log(-np.expm1(-self.Hf(x, *params)))

    @_check_x_not_empty
    def ll_observed(self, x, n, *params):
        *params, gamma, f0, p = params
        x = x - gamma
        n_zeros = np.sum(n[x == 0])
        zero_weight = n_zeros * np.log(f0) if n_zeros != 0 else 0
        non_zero_mask = x != 0
        N = np.sum(n[non_zero_mask])
        return (
            (n[non_zero_mask] * self.log_df(x[non_zero_mask], *params)).sum()
            + zero_weight
            + N * np.log(p - f0)
        )

    @_check_x_not_empty
    def ll_right_censored(self, x, n, *params):
        *params, gamma, f0, p = params
        x = x - gamma
        if p == 1:
            return np.sum(n * (np.log1p(-f0) + self.log_sf(x, *params)))
        else:
            F = self.ff(x, *params)
            # ALso could be:
            # np.sum(n * np.log((1 - p + (p - f0)*self.sf(x, *params))))
            return np.sum(n * np.log(1 - f0 - (p - f0) * F))

    @_check_x_not_empty
    def ll_left_censored(self, x, n, *params):
        *params, gamma, f0, p = params
        x = x - gamma
        if f0 == 1:
            return np.sum(n * self.log_ff(x, *params)) + n.sum() * np.log(p)
        else:
            return np.sum(n * np.log(f0 + (p - f0) * self.ff(x, *params)))

    @_check_x_not_empty
    def ll_interval_or_truncated(self, xl, xr, n, *params):
        *params, gamma, f0, p = params
        xr = xr - gamma
        xl = xl - gamma
        right = np.where(np.isfinite(xr), self.ff(xr, *params), 1)
        left = np.where(np.isfinite(xl), self.ff(xl, *params), 0)
        return np.sum(n * np.log(right - left)) + n.sum() * np.log(p - f0)

    def parameter_transform(self, x_min, params):
        *params, gamma, f0, p = params
        p = expit(p)
        f0 = expit(f0)
        gamma = x_min - np.exp(gamma) if gamma < 0 else x_min - 1 - gamma
        params = self._parameter_transform(*params)
        return (*params, gamma, f0, p)

    def _log_likelihood(self, data, *params):
        return (
            self.ll_observed(data.x_o, data.n_o, *params)
            + self.ll_right_censored(data.x_r, data.n_r, *params)
            + self.ll_left_censored(data.x_l, data.n_l, *params)
            + self.ll_interval_or_truncated(
                data.x_il, data.x_ir, data.n_i, *params
            )
            - self.ll_interval_or_truncated(
                data.x_tl, data.x_tr, data.n_t, *params
            )
        )

    def _neg_ll_func(self, data, *params):
        return -self._log_likelihood(data, *params)

    def neg_mean_D(self, x, c, n, tl, tr, *params):
        mask = c == 0
        x_obs = x[mask]
        n_obs = n[mask]

        # Assumes already ordered
        if np.isfinite(tl):
            F_tl = self.ff(tl, *params)
        else:
            F_tl = 0.0

        if np.isfinite(tr):
            F_tr = self.ff(tr, *params)
        else:
            F_tr = 1.0

        F = self.ff(x_obs, *params)

        all_F = np.hstack([F_tl, F, F_tr])
        D_0_1_normed = (all_F - F_tl) / (F_tr - F_tl)
        D = np.diff(D_0_1_normed)

        # Censoring
        Dr = self.sf(x[c == 1], *params)
        Dl = self.ff(x[c == -1], *params)

        if (n_obs > 1).any():
            n_ties = (n_obs - 1).sum()
            Df = self.df(x_obs, *params)
            # Df = Df[Df != 0]
            LL = np.concatenate([Dl, Df, Dr])
            ll_n = np.concatenate([n[c == -1], (n_obs - 1), n[c == 1]])
        else:
            Df = []
            n_ties = n_obs.sum()
            LL = np.concatenate([Dl, Dr])
            ll_n = np.concatenate([n[c == -1], n[c == 1]])

        M = np.log(D)
        M = -np.sum(M) / (M.shape[0])

        LL = -(np.log(LL) * ll_n).sum() / (n.sum() - n_obs.sum() + n_ties)
        return M + LL

    def _moment(self, n, *params, offset=False):
        if offset:
            gamma = params[0]
            params = params[1::]

            def fun(x):
                return x**n * self.df((x - gamma), *params)

            m = quad(fun, gamma, np.inf)[0]
        else:
            if hasattr(self, "moment"):
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

    def _validate_fit_inputs(
        self, surv_data, how, offset, lfp, zi, heuristic, turnbull_estimator
    ):
        if offset and (self.support[0] != 0):
            detail = "{} distribution cannot be offset".format(self.name)
            raise ValueError(detail)

        if how not in PARA_METHODS:
            raise ValueError('"how" must be one of: ' + str(PARA_METHODS))

        if how == "MPP" and self.name == "ExpoWeibull":
            detail = (
                "ExpoWeibull distribution does not work"
                + " with probability plot fitting"
            )
            raise ValueError(detail)

        if np.isfinite(surv_data.t).any() and how == "MSE":
            detail = "Mean square error doesn't yet support tuncation"
            raise NotImplementedError(detail)

        if np.isfinite(surv_data.t).any() and how == "MOM":
            detail = "Maximum product spacing doesn't support tuncation"
            raise ValueError(detail)

        if (lfp or zi) & (how != "MLE"):
            detail = (
                "Limited failure or zero-inflated models"
                + " can only be made with MLE"
            )
            raise ValueError(detail)

        if zi & (self.support[0] != 0):
            detail = (
                "zero-inflated models can only work"
                + "with models starting at 0"
            )
            raise ValueError()

        if (surv_data.c == 1).all():
            raise ValueError("Cannot have only right censored data")

        if (surv_data.c == -1).all():
            raise ValueError("Cannot have only left censored data")

        if surpyval.utils.check_no_censoring(surv_data.c) and (how == "MOM"):
            raise ValueError("Method of moments doesn't support censoring")

        if (
            (surpyval.utils.no_left_or_int(surv_data.c))
            and (how == "MPP")
            and (not heuristic == "Turnbull")
        ):
            detail = (
                "Probability plotting estimation with left or "
                + "interval censoring only works with Turnbull heuristic"
            )
            raise ValueError()

        if (
            (heuristic == "Turnbull")
            and (not ((-1 in surv_data.c) or (2 in surv_data.c)))
            and ((~np.isfinite(surv_data.tr)).any())
        ):
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

            if surv_data.x.ndim == 2:
                if (
                    (surv_data.x[:, 0] <= self.support[0]) & (surv_data.c == 0)
                ).any():
                    detail = detail_template.format(
                        lower=self.support[0], upper=self.support[1]
                    )
                    raise ValueError(detail)
                elif (
                    (surv_data.x[:, 1] >= self.support[1]) & (surv_data.c == 0)
                ).any():
                    detail = detail_template.format(
                        lower=self.support[0], upper=self.support[1]
                    )
                    raise ValueError(detail)
            else:
                if (
                    (surv_data.x <= self.support[0]) & (surv_data.c == 0)
                ).any():
                    detail = detail_template.format(
                        lower=self.support[0], upper=self.support[1]
                    )
                    raise ValueError(detail)
                elif (
                    (surv_data.x >= self.support[1]) & (surv_data.c == 0)
                ).any():
                    detail = detail_template.format(
                        lower=self.support[0], upper=self.support[1]
                    )
                    raise ValueError(detail)

        if (surv_data.tl[0] != surv_data.tl).any() and how == "MPS":
            raise ValueError(
                "Left truncated value can only be single number \
                              when using MPS"
            )

        if (surv_data.tr[0] != surv_data.tr).any() and how == "MPS":
            raise ValueError(
                "Right truncated value can only be single number \
                              when using MPS"
            )

        return True

    def fit(
        self,
        x=None,
        c=None,
        n=None,
        t=None,
        how="MLE",
        offset=False,
        zi=False,
        lfp=False,
        tl=None,
        tr=None,
        xl=None,
        xr=None,
        fixed=None,
        heuristic="Nelson-Aalen",
        init=[],
        rr="y",
        on_d_is_0=False,
        turnbull_estimator="Fleming-Harrington",
    ):
        """

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

                - MLE, Maximum Likelihood Estimation
                - MPP, Method of Probability Plotting
                - MOM, Method of Moments
                - MSE, Mean Square Error
                - MPS, Maximum Product Spacing

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

        heuristic : {"Blom", "Median", "ECDF", "Modal", "Midpoint", "Mean",\
            "Weibull", "Benard", "Beard", "Hazen", "Gringorten",\
            "None", "Tukey", "DPW", "Fleming-Harrington",\
            "Kaplan-Meier", "Nelson-Aalen", "Filliben",\
            "Larsen", "Turnbull"}, str, optional.
            Plotting method to use, if using the probability plotting,
            MPP, method.

        init : array like, optional
            initial guess of parameters. Instead of finding an initial guess
            for the optimization you can provide one. Can be useful to see if
            optimization is failing due to poor initial guess.

        rr : {'y', 'x'}, str, optional
            The dimension on which to minimise the spacing between the line
            and the observation. If 'y' the mean square error between the
            line and vertical distance to each point is minimised. If 'x' the
            mean square error between the line and horizontal distance to each
            point is minimised.

        on_d_is_0 : boolean, optional
            For the case when using MPP and the highest value is right
            censored, you can choose to include this value into the
            regression analysis or not. That is, if :code:`False`, all values
            where there are 0 deaths are excluded from the regression. If
            :code:`True` all values regardless of whether there is a death
            or not are included in the regression.

        turnbull_estimator : {'Nelson-Aalen', 'Kaplan-Meier', or\
            'Fleming-Harrington'), str, optional
            If using the Turnbull heuristic, you can elect to use either the
            KM, NA, or FH estimator with the Turnbull estimates of r, and d.
            Defaults to FH.

        Returns
        -------

        Parametric
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

        surv_data = SurpyvalData(
            x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr
        )
        return self.fit_from_surpyval_data(
            surv_data,
            how=how,
            offset=offset,
            zi=zi,
            lfp=lfp,
            fixed=fixed,
            heuristic=heuristic,
            init=init,
            rr=rr,
            on_d_is_0=on_d_is_0,
            turnbull_estimator=turnbull_estimator,
        )

    def fit_from_df(
        self,
        df,
        x=None,
        c=None,
        n=None,
        xl=None,
        xr=None,
        tl=None,
        tr=None,
        **fit_options
    ):
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

        Parametric
            A parametric model with the fitted parameters and methods for
            all functions of the distribution using the fitted parameters.


        Examples
        --------
        >>> import surpyval as surv
        >>> df = surv.datasets.BoforsSteel.data
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
                raise ValueError("`tl` must be scalar or column label string")
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

    def fit_from_ecdf(self, x, F):
        model = Parametric(self, "given ecdf", None, False, False, False)
        res = mpp_from_ecfd(self, x, F)
        model.dist = self
        model.params = np.array(res["params"])
        model.support = self.support

        return model

    def fit_from_non_parametric(self, non_parametric_model):
        x, F = non_parametric_model.x, 1 - non_parametric_model.R
        return self.fit_from_ecdf(x, F)

    def fit_from_surpyval_data(
        self,
        surv_data,
        how="MLE",
        offset=False,
        zi=False,
        lfp=False,
        fixed=None,
        heuristic="Nelson-Aalen",
        init=[],
        rr="y",
        on_d_is_0=False,
        turnbull_estimator="Fleming-Harrington",
    ):
        """

        The central feature to SurPyval's capability. This function aimed to
        have an API to mimic the simplicity of the scipy API. That is, to use
        a simple :code:`fit()` call, with as many or as few parameters as
        is needed.

        Parameters
        ----------

        surv_data : SurpyvalData
            Survival data in the SurpyvalData class.


        For other input options see :code:`fit` method.

        Returns
        -------

        Parametric
            A parametric model with the fitted parameters and methods for
            all functions of the distribution using the fitted parameters.

        """
        x, c, n, t = surv_data.x, surv_data.c, surv_data.n, surv_data.t
        # Unpack the truncation
        tl = t[:, 0]
        tr = t[:, 1]

        # Ensure truncation values move to edge where support is not
        # -np.inf to np.inf
        if np.isfinite(self.support[0]):
            tl = np.where(tl < self.support[0], self.support[0], tl)

        if np.isfinite(self.support[1]):
            tr = np.where(tl > self.support[1], self.support[1], tr)

        # Validate inputs
        self._validate_fit_inputs(
            surv_data, how, offset, lfp, zi, heuristic, turnbull_estimator
        )

        # Passed checks
        data = {"x": x, "c": c, "n": n, "t": t}

        model = Parametric(self, how, data, offset, lfp, zi)
        model.surv_data = surv_data
        fitting_info = {}

        if how == "MPS":
            # Need to set the scalar truncation values
            # if the MPS method is used.
            # since it has already been checked that they are all the same
            # we need only get the first item of each truncation array.
            model.tl = tl[0]
            model.tr = tr[0]

        if how != "MPP":
            transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
                x, model.bounds, fixed, model.param_map
            )

            fitting_info["transform"] = transform
            fitting_info["inv_trans"] = inv_trans
            fitting_info["const"] = const
            fitting_info["fixed_idx"] = fixed_idx
            fitting_info["not_fixed"] = not_fixed

            if init == []:
                if x.ndim == 2:
                    # If x has 2 dims, then there is intervally
                    # censored data. Simply take the midpoint to
                    # get the initial estimate.
                    x_init = x.mean(axis=1)
                    c_init = np.copy(c)
                    c_init[c_init == 2] = 0
                    n_init = np.copy(n)
                else:
                    x_init = np.copy(x)
                    c_init = np.copy(c)
                    n_init = np.copy(n)

                # If there is left censoring, assume that the
                # left censored value is the midpoint between
                # the censored value and the lowest x value
                x_init[c_init == -1] = (x_init[c_init == -1] + x.min()) / 2
                c_init[c_init == -1] = 0

                # check if the one support is -inf or inf and the other is
                # finite. If it isn't, then the distribution cannot be offset.
                # i.e if both finite or both infinite, then cannot be offset,
                # zero-inflated, or limited failure.
                if np.all(np.isinf(self.support)) or np.all(
                    np.isfinite(self.support)
                ):
                    with np.errstate(all="ignore"):
                        init = np.array(
                            self._parameter_initialiser(x_init, c_init, n_init)
                        )
                else:
                    with np.errstate(all="ignore"):
                        # Remove x where x is out of support
                        # This is if data for a zi or lfp model is present
                        if not offset:
                            in_support_mask = (x_init > self.support[0]) & (
                                x_init < self.support[1]
                            )

                            # Reduce x, c, and n to the case where it is in the
                            # support of the distribution
                            x_init = x_init[in_support_mask]
                            c_init = c_init[in_support_mask]
                            n_init = n[in_support_mask]

                        # Create an initial estimate with the new points
                        init = self._parameter_initialiser(
                            x_init, c_init, n_init, offset=offset
                        )
                        init = np.array(init)

                        if offset:
                            init[0] = x.min() - 1.0

                if lfp:
                    _, _, _, F = pp(
                        x_init, c_init, n_init, heuristic="Nelson-Aalen"
                    )

                    max_F = np.max(F)
                    init = np.concatenate([init, [min(0.6, max_F)]])

                if zi:
                    if x.ndim == 2:
                        x_0 = x[c == 0, 0]
                    else:
                        x_0 = x[c == 0]

                    n_0 = n[c == 0]
                    total_failures_at_zero = n_0[x_0 == 0].sum()

                    f_0_init = total_failures_at_zero / n.sum()
                    init = np.concatenate([init, [f_0_init]])

            init = np.atleast_1d(init)
            init = transform(init)
            init = init[not_fixed]
            fitting_info["init"] = init
        else:
            # Probability plotting method does not need an initial estimate
            fitting_info["rr"] = rr
            fitting_info["heuristic"] = heuristic
            fitting_info["on_d_is_0"] = on_d_is_0
            fitting_info["turnbull_estimator"] = turnbull_estimator
            fitting_info["init"] = None

        model.fitting_info = fitting_info

        results = METHOD_FUNC_DICT[how](model)

        for k, v in results.items():
            setattr(model, k, v)

        # Only needed since not all models return the params
        # as a numpy array... which ought to be fixed.
        model.params = np.atleast_1d(model.params)

        if hasattr(model, "params"):
            for k, v in zip(self.param_names, model.params):
                setattr(model, k, v)

        # Set the support of the distribution.
        if offset:
            left = model.gamma
        elif np.isfinite(self.support[0]):
            left = self.support[0]
        elif self.support[0] == -np.inf:
            left = -np.inf
        elif np.isnan(self.support[0]):
            # This only works for the uniform dist
            # TODO: More general support setting. i.e. 4 parameter Beta
            left = model.params[0]

        if np.isfinite(self.support[1]):
            right = self.support[1]
        elif self.support[1] == np.inf:
            right = np.inf
        elif np.isnan(self.support[1]):
            right = model.params[1]

        model.support = np.array([left, right])

        return model

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

        p : scalar, optional
            The proportion of the population that will never die or fail. If
            used it must be a value between 0 and 1. If None will assume 1,
            i.e. no proportion of the population will never die or fail.

        f0 : scalar, optional
            The proportion of the population that will die or fail at time 0.
            If used it must be a value between 0 and 1. If None will assume 0,
            i.e. no proportion of the population will die or fail at time 0.

        Returns
        -------

        Parametric
            A parametric model with the parameters provided.


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

        if gamma is not None and np.isinf(self.support).all():
            msg_base = "{dist} distribution cannot be offset"
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

        model = Parametric(self, "given parameters", None, offset, lfp, zi)
        model.gamma = gamma
        model.p = p
        model.f0 = f0
        model.params = np.array(params)
        model.support = self.support

        if offset:
            model.support = (gamma, model.support[1])
        elif np.isnan(self.support).any():
            model.support = np.array(model.params)

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
                params = ", ".join(self.param_names)
                base = "Params {params} must be in bounds {bounds}"
                detail = base.format(params=params, bounds=self.bounds)
                raise ValueError(detail)
        model.dist = self
        return model
