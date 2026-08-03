from numbers import Number

import numpy.typing as npt
import pandas as pd
from scipy.integrate import quad
from scipy.stats import uniform

import surpyval
from surpyval import np
from surpyval.utils import _check_x_not_empty
from surpyval.utils.surpyval_data import SurpyvalData

from ..nonparametric import plotting_positions as pp
from .fitters import bounds_convert
from .fitters.mle import mle
from .fitters.mom import mom
from .fitters.closed_form import closed_form_results
from .fitters.mpp import mpp, mpp_from_ecfd
from .fitters.mps import mps
from .fitters.mse import mse
from .parametric import Parametric

PARA_METHODS = ["MPP", "MLE", "MPS", "MSE", "MOM"]
METHOD_FUNC_DICT = {"MPP": mpp, "MOM": mom, "MLE": mle, "MPS": mps, "MSE": mse}

DEFAULT_Y_TICKS = [
    0.0001,
    0.0002,
    0.0003,
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
]


class ParametricFitter:
    """
    Base class for all parametric distributions.

    A distribution needs only ``hf`` and ``Hf`` (or ``sf``, ``ff`` and
    ``df``) plus a ``_parameter_initialiser`` with the signature
    ``(self, x, c=None, n=None, t=None, offset=False)`` for fitting to
    work; ``log_df``, ``log_sf``, ``log_ff`` and ``random`` have generic
    implementations here that subclasses can override with closed forms.
    Probability plotting (the MPP fit method and ``Parametric.plot``)
    additionally requires ``mpp_x_transform``, ``mpp_y_transform(y,
    *params)`` and ``mpp_inv_y_transform(y, *params)``.

    A subclass can take over an entire estimation method by defining
    ``mpp(x, c, n, heuristic, rr, on_d_is_0, offset)``, returning a
    results dict with at least a ``params`` numpy array.

    A subclass with an exact analytic MLE may also define
    ``_closed_form_mle(data)``, returning the parameter vector, or
    ``None`` when the closed form does not apply to *that* data. It is
    consulted before any initial guess or optimisation, so an eligible
    fit skips both entirely; ``init`` has no effect on that path because
    the closed form is exact. Structural requests (an offset, limited
    failure population, zero inflation, or fixed parameters) bypass it.

    Implementations must do their math with ``surpyval.np``, which is
    ``autograd.numpy``: maximum likelihood estimation differentiates
    through these functions, and plain numpy silently breaks the
    gradients.
    """

    # Whether the distribution's mass sits on integers rather than a
    # continuum. ``DiscreteParametricFitter`` overrides this; fit-method
    # validation and callers branch on the trait.
    discrete = False

    def __init__(
        self,
        name: str,
        k: int,
        bounds: tuple[tuple[int | float | None, int | float | None], ...],
        support: tuple[int | float, int | float],
        param_names: list[str],
        param_map: dict[str, int],
        plot_x_scale: str,
        y_ticks: list[float] | None = None,
    ):
        self.name: str = name
        self.k = k
        self.bounds = bounds
        self.support = support
        self.param_names = param_names
        self.param_map = param_map
        self.plot_x_scale = plot_x_scale
        self.y_ticks = DEFAULT_Y_TICKS if y_ticks is None else y_ticks
        self.supports_mpp = True
        # For distributions whose support is data-dependent (declared as
        # NaN, e.g. the 4-parameter Beta), these give the indices of the
        # parameters that supply the left and right support bounds once
        # the model is fitted. The default ``(0, 1)`` matches the legacy
        # behaviour used by ``Uniform``.
        self.support_param_index = (0, 1)

    def random(self, size, *params):
        r"""

        Draws random samples from the distribution in shape `size`, using
        the inverse transform method with the distribution's quantile
        function.

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        params : numpy array or scalar
            The parameters of the distribution

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> np.random.seed(1)
        >>> Weibull.random(5, 3, 4)
        array([2.57122697, 3.18730986, 0.31024877, 2.32381059, 1.89352939])
        """
        U = uniform.rvs(size=size)
        return self.qf(U, *params)

    def log_df(self, x, *params):
        return np.log(self.hf(x, *params)) - self.Hf(x, *params)

    def log_sf(self, x, *params):
        return -self.Hf(x, *params)

    def log_ff(self, x, *params):
        return np.log(-np.expm1(-self.Hf(x, *params)))

    def cs(self, x, X, *params):
        # Conditional survival R(x + X) / R(X); distributions override
        # this only to carry a docstring or a simplified closed form.
        # The default also gives discrete distributions a working
        # ``Parametric.cs`` (previously AttributeError).
        return self.sf(x + X, *params) / self.sf(X, *params)

    def _plot_x_bounds(self, x, params):
        """Return (x_scale_min, x_scale_max) for probability plots.

        Returns None to auto-compute the bounds from the data.
        """
        return None

    @_check_x_not_empty
    def ll_observed(self, x, n, *params):
        *params, gamma, f0, p = params
        if f0 == 0:
            # Not zero-inflated; x == 0 is an ordinary observation.
            zero_weight = 0
            non_zero_mask = np.full(x.shape, True)
        else:
            # The zero-inflation mass sits at x == 0 in observed time,
            # so the mask must be taken before the offset shift
            n_zeros = np.sum(n[x == 0])
            zero_weight = n_zeros * np.log(f0) if n_zeros != 0 else 0
            non_zero_mask = x != 0
        x = x - gamma
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
            return np.sum(n * np.log(1 - f0 - (p - f0) * F))

    @_check_x_not_empty
    def ll_left_censored(self, x, n, *params):
        *params, gamma, f0, p = params
        x = x - gamma
        if f0 == 0:
            # No zero-inflation: F_mix = p * F, so the numerically stable
            # log_ff path applies (the branch was inverted as ``f0 == 1``,
            # which never occurs, #256).
            return np.sum(n * self.log_ff(x, *params)) + n.sum() * np.log(p)
        else:
            return np.sum(n * np.log(f0 + (p - f0) * self.ff(x, *params)))

    @_check_x_not_empty
    def ll_interval_or_truncated(self, xl, xr, n, *params):
        *params, gamma, f0, p = params
        xr = xr - gamma
        xl = xl - gamma
        # Probabilities must come from the mixture CDF
        # F_mix(t) = f0 + (p - f0) * F0(t), not (p - f0) * F0(t): with no
        # right bound (tr = inf) the window probability is the mixture
        # survival 1 - F_mix(tl), which includes the never-failing mass
        # (1 - p). The old (p - f0) * (1 - F0(tl)) form made the LFP +
        # left-truncation likelihood unbounded (#269). For finite-bound
        # intervals the f0 terms cancel, so plain fits are unchanged.
        right = np.where(
            np.isfinite(xr), f0 + (p - f0) * self.ff(xr, *params), 1.0
        )
        left = np.where(
            np.isfinite(xl), f0 + (p - f0) * self.ff(xl, *params), 0.0
        )
        return np.sum(n * np.log(np.maximum(right - left, 0.0)))

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
        denom = F_tr - F_tl
        if denom < np.finfo(float).eps:
            return np.inf
        D_0_1_normed = (all_F - F_tl) / denom
        D = np.diff(D_0_1_normed)

        # Censored contributions, conditioned on the truncation window:
        # under truncation the sample comes from the conditional
        # distribution, so survivor/CDF terms are renormalised exactly
        # like the spacings (previously they were left unconditioned,
        # biasing every truncated + censored fit, #268).
        Dr = (F_tr - self.ff(x[c == 1], *params)) / denom
        Dl = (self.ff(x[c == -1], *params) - F_tl) / denom

        # Cheng-Amin sum form: one log-spacing per distinct observed
        # value (plus the two boundary spacings), (n - 1) conditional
        # density terms for ties, and one conditional survivor/CDF term
        # per censored unit -- all in a single sum. The previous form
        # divided the spacings block and the censored/ties block by
        # different counts, which made the estimator inconsistent for
        # censored or tied data (#268); dividing the single sum by the
        # total count only scales the objective.
        obj = np.sum(np.log(D))
        if (n_obs > 1).any():
            # Evaluate the tie densities only at genuinely tied points:
            # untied points contribute 0 * log(0) = NaN when the density
            # underflows, poisoning the objective where a clean inf
            # penalty is wanted (#289).
            tied = n_obs > 1
            Df = self.df(x_obs[tied], *params) / denom
            obj = obj + np.sum((n_obs[tied] - 1) * np.log(Df))
        if (c == 1).any():
            obj = obj + np.sum(n[c == 1] * np.log(Dr))
        if (c == -1).any():
            obj = obj + np.sum(n[c == -1] * np.log(Dl))
        return -obj / n.sum()

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

    def _check_identifiable(self, surv_data, offset, lfp, zi, fixed):
        """
        Reject data that cannot pin down the free parameters.

        A right censored observation says only "later than this", so it
        constrains a fitted curve without locating a point on it. What
        locates a point is an exact observation, a left censored one, or
        an interval. Fewer *distinct* such values than there are free
        parameters and the likelihood has a flat direction: for a
        Weibull on a tied sample it is unbounded, since a spike of
        arbitrary height can sit on the repeated value, and the reported
        answer is wherever the optimiser happened to stop. Three tied
        observations at 10 returned ``beta = 512`` with ``success=True``
        and no warning.

        The count is of *free* parameters, not of the distribution's
        parameters, so fixing one buys back a degree of freedom: a
        Weibull fit to a single observation with ``beta`` fixed is well
        posed and recovers ``alpha = (sum x^beta / n) ** (1 / beta)``.
        That is why this cannot be a per-distribution constant.
        """
        n_free = (
            self.k
            + int(bool(offset))
            + int(bool(lfp))
            + int(bool(zi))
            - len(fixed or {})
        )
        if n_free <= 0:
            return

        x, c = surv_data.x, surv_data.c
        informative = np.asarray(c) != 1
        if not informative.any():
            return
        rows = np.asarray(x)[informative]
        if rows.ndim == 1:
            distinct = np.unique(rows).size
        else:
            distinct = np.unique(rows, axis=0).shape[0]

        if distinct < n_free:
            raise ValueError(
                f"{self.name} has {n_free} free parameter(s) but the data "
                f"contains only {distinct} distinct non-right-censored "
                f"value(s). The likelihood has a flat (or unbounded) "
                f"direction, so no unique fit exists. Provide more "
                f"distinct observations, fix a parameter with "
                f"`fixed=`, or choose a distribution with fewer "
                f"parameters."
            )

    def _validate_fit_inputs(
        self,
        surv_data,
        how,
        offset,
        lfp,
        zi,
        fixed,
        heuristic,
        turnbull_estimator,
    ):
        # Offsetting (a free location/threshold ``gamma``) only makes sense
        # for distributions supported on a half-line ``[0, inf)``. A
        # distribution with a finite upper bound (e.g. Beta on ``[0, 1]``)
        # or a data-dependent support cannot be offset: shifting the lower
        # bound while pinning the upper one is not a member of the family.
        # Use the 4-parameter Beta instead if you need a shifted/scaled
        # Beta on an arbitrary ``[a, b]`` interval.
        offsettable = (self.support[0] == 0) and np.isinf(self.support[1])
        if offset and not offsettable:
            detail = f"{self.name} distribution cannot be offset"
            raise ValueError(detail)

        # Probability plotting is exempt. It is a regression through the
        # plotting positions, not a likelihood maximisation, so it has no
        # unbounded direction to fall into and now always returns finite
        # parameters. It is also how several distributions seed
        # themselves, and that internal call does not carry the caller's
        # ``fixed``, so checking it would reject well posed fits.
        if how != "MPP":
            self._check_identifiable(surv_data, offset, lfp, zi, fixed)

        if fixed and how == "MPP":
            detail = (
                "Probability plotting (MPP) does not support"
                " fixing parameters"
            )
            raise ValueError(detail)

        if how not in PARA_METHODS:
            raise ValueError('"how" must be one of: ' + str(PARA_METHODS))

        if how == "MPP" and not self.supports_mpp:
            detail = (
                f"{self.name} distribution does not work"
                " with probability plot fitting; use how='MLE', 'MSE' or"
                " 'MOM' instead"
            )
            raise ValueError(detail)

        if how == "MPS" and self.discrete:
            detail = (
                f"{self.name} is a discrete distribution; maximum product"
                " of spacings (MPS) is defined by increments of a"
                " continuous CDF, and repeated integer observations make"
                " the spacings degenerate. Use how='MLE' instead."
            )
            raise ValueError(detail)

        if np.isfinite(surv_data.t).any() and how == "MSE":
            detail = "Mean square error doesn't yet support truncation"
            raise NotImplementedError(detail)

        if np.isfinite(surv_data.t).any() and how == "MOM":
            detail = "Method of moments doesn't support truncation"
            raise ValueError(detail)

        if (lfp or zi) and (how != "MLE"):
            detail = (
                "Limited failure or zero-inflated models"
                " can only be made with MLE"
            )
            raise ValueError(detail)

        if zi and (self.support[0] != 0):
            detail = (
                "zero-inflated models can only work with models starting at 0"
            )
            raise ValueError(detail)

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
                "interval censoring only works with Turnbull heuristic"
            )
            raise ValueError(detail)

        if (
            (heuristic == "Turnbull")
            and (not ((-1 in surv_data.c) or (2 in surv_data.c)))
            and ((~np.isfinite(surv_data.tr)).all())
        ):
            # The Turnbull method is extremely memory intensive.
            # So if no left or interval censoring and no right-truncation
            # then this is equivalent.
            heuristic = turnbull_estimator

        if (not offset) and (not zi):
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
                elif (
                    (surv_data.x[:, 0] < self.support[0]) & (surv_data.c == 2)
                ).any():
                    # An interval endpoint strictly below the support makes
                    # the CDF evaluate outside its domain: NaN likelihood
                    # everywhere and a silent initial-guess "fit" (#261).
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
                elif (
                    (surv_data.x <= self.support[0]) & (surv_data.c == -1)
                ).any():
                    # A left-censored point at or below the support start is
                    # a zero-probability observation: the likelihood is
                    # -inf/NaN everywhere and the optimiser silently
                    # returns the initial guess (#261).
                    detail = detail_template.format(
                        lower=self.support[0], upper=self.support[1]
                    )
                    raise ValueError(detail)

        if how == "MPS" and (surv_data.c == 2).any():
            # neg_mean_D has no interval-censored term; without this
            # guard 2-D input dies deep in np.hstack with a cryptic
            # dimensions error (#268).
            raise ValueError(
                "MPS does not support interval-censored observations; "
                "use MLE (or MPP with the Turnbull heuristic) for "
                "interval data."
            )

        if (surv_data.tl[0] != surv_data.tl).any() and how == "MPS":
            raise ValueError("Left truncated value can only be single number \
                              when using MPS")

        if (surv_data.tr[0] != surv_data.tr).any() and how == "MPS":
            raise ValueError("Right truncated value can only be single number \
                              when using MPS")

        return heuristic

    def fit(
        self,
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        how: str = "MLE",
        offset: bool = False,
        zi: bool = False,
        lfp: bool = False,
        tl: npt.ArrayLike | Number | None = None,
        tr: npt.ArrayLike | Number | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        heuristic: str = "Nelson-Aalen",
        init: npt.ArrayLike = [],
        rr: str = "y",
        on_d_is_0: bool = False,
        turnbull_estimator: str = "Fleming-Harrington",
    ) -> Parametric:
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
            Array of counts for each x. If data is provided as counts, then
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
        df: pd.DataFrame,
        x: str | None = None,
        c: str | None = None,
        n: str | None = None,
        xl: str | None = None,
        xr: str | None = None,
        tl: str | float | None = None,
        tr: str | float | None = None,
        **fit_options,
    ) -> Parametric:
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
        >>> from surpyval.datasets import load_bofors_steel
        >>> df = load_bofors_steel()
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

        if not isinstance(df, pd.DataFrame):
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
            if isinstance(tl, str):
                tl = df[tl].values.astype(float)
            elif np.isscalar(tl):
                tl = (np.ones(df.shape[0]) * tl).astype(float)
            else:
                raise ValueError("`tl` must be scalar or column label string")
        else:
            tl = np.ones(df.shape[0]) * -np.inf

        if tr is not None:
            if isinstance(tr, str):
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

    def fit_from_ecdf(self, x: npt.ArrayLike, F: npt.ArrayLike) -> Parametric:
        model = Parametric(self, "given ecdf", None, False, False, False)
        res = mpp_from_ecfd(self, x, F)
        model.params = np.array(res["params"])
        model.support = self.support

        return model

    def fit_from_non_parametric(self, non_parametric_model) -> Parametric:
        x, F = non_parametric_model.x, 1 - non_parametric_model.R
        return self.fit_from_ecdf(x, F)

    def _clamp_truncation_to_support(self, t):
        """Clamp the truncation bounds to the distribution's support.

        Returns the left and right truncation arrays with any value that
        falls outside a *finite* support edge moved onto that edge. An
        infinite support edge leaves the corresponding bound untouched.
        """
        tl = t[:, 0]
        tr = t[:, 1]

        if np.isfinite(self.support[0]):
            tl = np.where(tl < self.support[0], self.support[0], tl)

        if np.isfinite(self.support[1]):
            tr = np.where(tr > self.support[1], self.support[1], tr)

        return tl, tr

    def _initial_guess(self, x, c, n, offset, zi, lfp, heuristic):
        """Derive an initial parameter vector for the iterative fitters.

        Builds a working copy of the data with interval- and
        left-censored points imputed to point observations, asks the
        distribution's ``_parameter_initialiser`` for a seed, and appends
        the limited-failure (``p``) and zero-inflation (``f0``) seeds when
        those models are requested. The returned vector is in the natural
        (untransformed) parameter space.
        """
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
        if (
            np.all(np.isinf(self.support))
            or np.all(np.isfinite(self.support))
            or np.all(np.isnan(self.support))
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
                elif zi:
                    # Exact zeros belong to the zero-inflation
                    # mass; including them would drag the offset
                    # initial guess below zero
                    nonzero_mask = x_init != 0
                    x_init = x_init[nonzero_mask]
                    c_init = c_init[nonzero_mask]
                    n_init = n_init[nonzero_mask]

                # Create an initial estimate with the new points
                init = self._parameter_initialiser(
                    x_init, c_init, n_init, offset=offset
                )
                init = np.array(init)

                if offset:
                    x_nonzero = x[x != 0] if zi else x
                    init[0] = x_nonzero.min() - 1.0

        if lfp:
            _, _, _, F = pp(x_init, c_init, n_init, heuristic="Nelson-Aalen")

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

        return init

    def _set_support(self, model, offset):
        """Resolve and assign the fitted model's support interval.

        For an offset model the left edge is the fitted ``gamma``;
        otherwise each edge comes from the distribution's declared
        support, except a data-dependent (NaN) edge, which is read from
        the fitted parameter the distribution nominates via
        ``support_param_index`` (``a``/``b`` for the uniform and the
        4-parameter Beta).
        """
        if offset:
            left = model.gamma
        elif np.isfinite(self.support[0]):
            left = self.support[0]
        elif self.support[0] == -np.inf:
            left = -np.inf
        elif np.isnan(self.support[0]):
            left = model.params[self.support_param_index[0]]

        if np.isfinite(self.support[1]):
            right = self.support[1]
        elif self.support[1] == np.inf:
            right = np.inf
        elif np.isnan(self.support[1]):
            right = model.params[self.support_param_index[1]]

        model.support = np.array([left, right])

    def fit_from_surpyval_data(
        self,
        surv_data: SurpyvalData,
        how: str = "MLE",
        offset: bool = False,
        zi: bool = False,
        lfp: bool = False,
        fixed: dict[str, float] | None = None,
        heuristic: str = "Nelson-Aalen",
        init: npt.ArrayLike = [],
        rr: str = "y",
        on_d_is_0: bool = False,
        turnbull_estimator: str = "Fleming-Harrington",
    ) -> Parametric:
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
        # Clamp the truncation values to the (possibly finite) support edges
        tl, tr = self._clamp_truncation_to_support(t)

        # Validate inputs
        heuristic = self._validate_fit_inputs(
            surv_data,
            how,
            offset,
            lfp,
            zi,
            fixed,
            heuristic,
            turnbull_estimator,
        )

        # Passed checks
        data = {"x": x, "c": c, "n": n, "t": t}

        model = Parametric(self, how, data, offset, lfp, zi)
        model.surv_data = surv_data
        fitting_info: dict = {}

        # An exact analytic MLE, where one exists for this distribution and
        # this data, is attempted *before* the initial guess and bounds
        # machinery below -- both of which exist only to seed and run the
        # optimiser. Returns None whenever the closed form does not apply,
        # and the numerical path proceeds untouched.
        results = self._try_closed_form_mle(
            surv_data, how, offset, lfp, zi, fixed
        )

        if results is None:
            results = self._fit_numerically(
                model,
                fitting_info,
                x,
                c,
                n,
                tl,
                tr,
                how,
                offset,
                zi,
                lfp,
                fixed,
                heuristic,
                init,
                rr,
                on_d_is_0,
                turnbull_estimator,
            )
        else:
            model.fitting_info = fitting_info

        for k, v in results.items():
            setattr(model, k, v)

        # A fit must never hand back a non-finite parameter. When the
        # optimiser fails, the reported parameters are the initial guess
        # (#261), so any initialiser that produced a nan or an inf had
        # it laundered into what looked like a fitted model: an offset
        # Gamma on a tied sample returned ``(inf, inf)`` in silence. The
        # initialisers that could do that are fixed, but this is the
        # backstop, since a non-finite parameter is never a valid answer
        # whatever produced it.
        _params = np.atleast_1d(np.asarray(model.params, dtype=float))
        _extra = [getattr(model, name, None) for name in ("gamma", "p", "f0")]
        _extra = [float(v) for v in _extra if v is not None]
        if not (np.isfinite(_params).all() and np.isfinite(_extra).all()):
            raise ValueError(
                f"{self.name} fit produced non-finite parameters "
                f"({np.asarray(model.params)}). The optimiser did not "
                f"reach a valid solution; check the data for degenerate "
                f"or extreme values."
            )

        # Only maximum likelihood and the closed forms report a
        # log-likelihood, because only they compute one on the way to
        # the answer. That left ``neg_ll``, ``aic``, ``bic`` and
        # ``aic_c`` raising AttributeError for every MPS, MSE, MOM and
        # MPP fit -- so the usual way of choosing between distributions
        # was unavailable for four of the five methods.
        #
        # The log-likelihood is a property of the parameters and the
        # data, not of the search that found them, so evaluate it here.
        # Guarded by ``hasattr`` so the methods that already report one
        # keep theirs untouched: maximum likelihood's is the optimiser's
        # own final objective, which on its fallback path is deliberately
        # taken at the initial guess rather than at the failed result
        # (#261), and recomputing would quietly undo that.
        if not hasattr(model, "_neg_ll"):
            with np.errstate(all="ignore"):
                model._neg_ll = float(
                    self._neg_ll_func(
                        surv_data,
                        *model.params,
                        model.gamma,
                        model.f0,
                        model.p,
                    )
                )

        # Expose each fitted parameter by name (e.g. ``model.alpha``), but
        # never overwrite the reserved offset / limited-failure /
        # zero-inflation attributes, which the survival functions rely on.
        # A distribution may legitimately name a parameter ``p`` (e.g.
        # ``Geometric``, ``NegativeBinomial``); those remain available via
        # ``model.params``.
        reserved = {"gamma", "p", "f0"}
        for k, v in zip(self.param_names, model.params):
            if k not in reserved:
                setattr(model, k, v)

        self._set_support(model, offset)

        return model

    def _try_closed_form_mle(
        self, surv_data, how, offset, lfp, zi, fixed
    ) -> "dict | None":
        """An exact analytic MLE, or ``None`` to use the optimiser.

        Two conditions have to hold, and they live in different places
        because they are different kinds of question.

        The *structural* ones are checked here: an offset ``gamma``, a
        limited-failure ``p``, zero-inflation ``f0`` or any user-fixed
        parameter each adds structure the analytic solutions do not
        solve for. These are properties of the requested model rather
        than of the data, and they are identical for every distribution.

        The *data-shape* condition is left to the distribution's own
        ``_closed_form_mle``, which alone knows what it can solve -- the
        Exponential accepts right censoring and left truncation, the
        Normal needs complete data -- and which signals inapplicability
        by returning ``None``.
        """
        if how != "MLE":
            return None
        if offset or lfp or zi or fixed:
            return None

        solver = getattr(self, "_closed_form_mle", None)
        if solver is None:
            return None

        params = solver(surv_data)
        if params is None:
            return None

        return closed_form_results(self, surv_data, params)

    def _fit_numerically(
        self,
        model,
        fitting_info,
        x,
        c,
        n,
        tl,
        tr,
        how,
        offset,
        zi,
        lfp,
        fixed,
        heuristic,
        init,
        rr,
        on_d_is_0,
        turnbull_estimator,
    ) -> dict:
        """Seed an initial guess, convert bounds and run the estimator."""
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
            fitting_info["inv_trans"] = inv_trans
            fitting_info["const"] = const
            fitting_info["fixed_idx"] = fixed_idx

            # ``len``-based check: comparing an ndarray to ``[]`` raises a
            # broadcast error (#261).
            if init is None or len(np.atleast_1d(init)) == 0:
                init = self._initial_guess(x, c, n, offset, zi, lfp, heuristic)

            init = np.atleast_1d(init)
            if fixed and len(init) == len(not_fixed):  # type: ignore[arg-type]
                # The initial guess covers only the free parameters;
                # merge it with the fixed values to get the full vector
                full_init = np.zeros(len(model.param_map))
                full_init[not_fixed] = init
                for name, value in fixed.items():
                    full_init[model.param_map[name]] = value
                init = full_init
            init = transform(init)
            init = init[not_fixed]  # type: ignore[index]
            fitting_info["init"] = init
        else:
            # Probability plotting method does not need an initial estimate
            fitting_info["rr"] = rr
            fitting_info["heuristic"] = heuristic
            fitting_info["on_d_is_0"] = on_d_is_0
            fitting_info["turnbull_estimator"] = turnbull_estimator
            fitting_info["init"] = None

        model.fitting_info = fitting_info

        return METHOD_FUNC_DICT[how](model)

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
            detail = f"Must have {self.k} params for {self.name} distribution"
            raise ValueError(detail)

        # Offsetting only makes sense for a half-line support; a fully
        # unbounded support (Normal) or a data-dependent one whose bounds
        # are themselves estimated (Uniform/Beta4, declared NaN) cannot be
        # offset. This mirrors the ``offsettable`` check in ``fit``.
        if gamma is not None and (
            np.isinf(self.support).all() or np.isnan(self.support).any()
        ):
            detail = f"{self.name} distribution cannot be offset"
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
        self._set_support(model, offset)

        for i, (low, upp) in enumerate(self.bounds):
            if low is None:
                lower_limit = -np.inf
            else:
                lower_limit = low
            if upp is None:
                upper_limit = np.inf
            else:
                upper_limit = upp

            if not (lower_limit < params[i] < upper_limit):
                param_names = ", ".join(self.param_names)
                detail = (
                    f"Params {param_names} must be in" f" bounds {self.bounds}"
                )
                raise ValueError(detail)
        return model
