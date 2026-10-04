"""The estimation machinery of the fittable parametric distributions.

:class:`OptimisedFitMixin` holds ``fit`` and everything it needs; the
checks of its inputs and its initial guesses are in ``_fit_inputs``
(``FitInputsMixin``, which it inherits).
"""

from __future__ import annotations

import warnings
from numbers import Number
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.utils.no_maximum import (
    maximum_warnings_quiet,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.surpyval_data import SurpyvalData

from ._fit_inputs import FitInputsMixin, normalise_how
from .fitters import bounds_convert
from .fitters.closed_form import closed_form_results
from .fitters.mle import mle
from .fitters.mom import mom
from .fitters.mpp import mpp, mpp_from_ecfd
from .fitters.mps import mps
from .fitters.mps import offset_start as mps_offset_start
from .fitters.mse import mse
from .parametric import Parametric


def _search_units(
    init: npt.NDArray,
    bounds: "tuple[tuple[float | None, float | None], ...]",
    fixed_idx: "list[int] | tuple[int, ...]" = (),
) -> list[float]:
    """Per-parameter ``units`` for ``bounds_convert``.

    A parameter with one bound is searched as the log of its distance
    from the bound below one unit, and linearly above it. With a unit of
    1 the switch sits at a fixed *value*, so which half a parameter is
    searched in depends on the data's units: a Weibull scale is searched
    as a log for data in thousandths and linearly for data in thousands.
    The search is then a different one at every scale. For most fits
    both routes lead to the same optimum (to about 1e-6 of a quantile on
    40 points), but an offset fit has a ridge
    along which the offset, scale and shape trade off, and there they do
    not: an ExpoWeibull MSE fit to data in ten-thousandths wandered for
    800 iterations and ended 13% of the data's spread from the fit to
    the same data in its own units, and a LogLogistic scale that started
    several times too large was stepped so far into the log half, in
    data units of thousands, that it underflowed.

    Each one-sided parameter's unit is therefore its own starting
    distance from its bound -- for the offset, the step below the
    smallest observation (see ``_offset_start``). Every such parameter
    starts at the switch, a searched value of 0, and the search in them
    is the same whatever the data's units: a scale's start and its unit
    both scale with the data, a shape's are both unchanged. A parameter
    that starts on its bound (or at a non-finite value) keeps a unit of
    1; the other kinds of bound ignore the unit, and so does a fixed
    parameter (``fixed_idx``), which is not searched: in its own unit its
    value went through the map and back as 1.9999999999999998 for a fixed
    2.

    Offset fits used this first; every fit does now (#366), so that the
    search is one search in any units (principle 6). That moved the fits
    without an offset in their last digits only.
    """
    units = [1.0] * len(bounds)
    for i, (low, upp) in enumerate(bounds):
        if (low is None) == (upp is None) or i in fixed_idx:
            continue
        if upp is None:
            assert low is not None
            distance = float(init[i]) - float(low)
        else:
            distance = float(upp) - float(init[i])
        if np.isfinite(distance) and distance > 0:
            units[i] = distance
    return units


def _optimizer_label(how: str, res: Any) -> str:
    """What found a non-MLE fit's answer, for ``model.optimizer``."""
    if how == "MPP":
        return "least squares"
    if res is None:
        # MOM solved in closed form (``_mom``), or with nothing to solve
        return "closed-form"
    return str(getattr(res, "optimizer", "BFGS"))


METHOD_FUNC_DICT = {"MPP": mpp, "MOM": mom, "MLE": mle, "MPS": mps, "MSE": mse}


class OptimisedFitMixin(FitInputsMixin):
    """The estimation machinery: ``fit`` and everything it needs.

    Separated from :class:`ParametricFitter` so that the distributions
    which do *not* have it are not claiming to. ``Bernoulli``,
    ``Binomial`` and ``ExactEventTime`` estimate their parameters in
    closed form; they take ``x`` and at most ``c``, ``n`` and ``t``, and
    have no use for ``how``, ``offset``, ``zi``, ``lfp``, ``fixed`` or
    the truncation arguments. While this lived on the base class those
    three overrode ``fit`` with a narrower signature, which is a Liskov
    violation mypy reports and, more to the point, a real one:
    ``Bernoulli.fit(x, c=...)`` raises TypeError, so code written
    against a ``ParametricFitter`` breaks on exactly those three.

    Every distribution is still a ``ParametricFitter`` -- that is what
    the ``isinstance`` gates in the model, mixture, regression, frailty
    and renewal code check, and what carries the distribution functions
    and the likelihood. This mixin adds the estimation methods on top,
    for the 22 that have them.

    Declare a parameter as ``OptimisedFitMixin`` when it must be
    fittable by a chosen method; declare it as ``ParametricFitter`` when
    only the distribution functions are needed.
    """

    if TYPE_CHECKING:
        # Supplied by ParametricFitter, which every user of this mixin
        # also inherits. Declared rather than defined so the methods
        # below type check without the mixin pretending to own them.
        name: str
        k: int
        bounds: tuple[tuple[int | float | None, int | float | None], ...]
        support: tuple[int | float, int | float]
        parameter_names: list[str]
        param_map: dict[str, int]
        discrete: bool
        supports_mpp: bool
        support_param_index: tuple[int, int]

        # Every implementation returns a 1-D float array. It used to
        # be a tuple in nine, an array in six, a list in one and a
        # fitted model's .params in five -- and a bare scalar in
        # Rayleigh, which made the seed 0-dimensional and broke the
        # lfp and zi paths outright.
        def _parameter_initialiser(
            self, data: SurpyvalData, offset: bool = False
        ) -> npt.NDArray: ...
        def _neg_ll_func(self, data: Any, *params: Any) -> Any: ...
        def _log_likelihood(self, data: Any, *params: Any) -> Any: ...
        def _moment(self, n: Any, *p: Any, offset: bool = False) -> Any: ...
        def _set_support(self, model: Any, offset: Any) -> Any: ...
        def sf(self, x: Any, *params: Any) -> Any: ...
        def ff(self, x: Any, *params: Any) -> Any: ...
        def df(self, x: Any, *params: Any) -> Any: ...
        def hf(self, x: Any, *params: Any) -> Any: ...
        def Hf(self, x: Any, *params: Any) -> Any: ...
        def qf(self, u: Any, *params: Any) -> Any: ...
        def mpp_x_transform(self, x: Any, *args: Any) -> Any: ...
        def mpp_y_transform(self, y: Any, *params: Any) -> Any: ...
        def mpp_inv_y_transform(self, y: Any, *params: Any) -> Any: ...

    def neg_mean_D(
        self, x: npt.NDArray, c: Any, n: Any, tl: Any, tr: Any, *params: Any
    ) -> Any:
        r"""The maximum-product-of-spacings objective that ``how='MPS'``
        minimises: minus the mean log spacing, with the tie, censoring
        and truncation terms described in :doc:`/Parametric Estimation`.

        ``x`` must be sorted, ``c`` and ``n`` are the matching censoring
        flags and counts, and ``tl`` and ``tr`` are the single truncation
        window shared by every observation (``-inf`` and ``inf`` when
        there is none). Returns ``inf`` where the window has no
        probability.
        """
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

    def mom_moment_gen(
        self, *params: Any, offset: bool = False, k: int | None = None
    ) -> Any:
        """The first ``k`` raw moments at ``params`` (leading with the
        offset when ``offset``). ``k`` defaults to one per parameter; the
        method of moments passes the number of *free* parameters, since a
        fixed parameter needs no equation of its own."""
        if k is None:
            k = self.k + 1 if offset else self.k
        moments = np.zeros(k)
        for i in range(0, k):
            n = i + 1
            moments[i] = self._moment(n, *params, offset=offset)
        return moments

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

        Fit the distribution to data and return the fitted model.

        This is the central call of SurPyval. Pass as many or as few of the
        arguments as the data needs: the event times ``x`` (or ``xl`` and
        ``xr``) are the only required input, and any mix of censoring,
        counts and truncation can be added to them.

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

                - MLE, Maximum Likelihood Estimation (the default)
                - MPP, Method of Probability Plotting
                - MOM, Method of Moments
                - MSE, Mean Square Error between the fitted CDF and a
                  non-parametric estimate
                - MPS, Maximum Product Spacing

            Only MLE supports every kind of censoring and truncation, and
            only MLE can fit ``zi`` and ``lfp`` models; MOM and MSE do not
            support truncation.

        offset : boolean, optional
            If :code:`True` finds the shifted distribution. If not provided
            assumes not a shifted distribution. Only works with continuous
            distributions that are supported on the half-real line.

        zi : boolean, optional
            If :code:`True` fits a zero-inflated model: an extra parameter
            ``f0``, the proportion of the population that fails at time 0.
            MLE only, and only for distributions supported from 0. Defaults
            to :code:`False`.

        lfp : boolean, optional
            If :code:`True` fits a limited-failure-population model: an
            extra parameter ``lfp_p``, the proportion of the population that
            will ever fail (``1 - lfp_p`` never fails). MLE only. Defaults to
            :code:`False`.

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

        heuristic : str, optional
            Plotting method to use, if using the probability plotting,
            MPP, method. One of the heuristics accepted by
            ``plotting_positions`` (``"Blom"``, ``"Median"``,
            ``"Kaplan-Meier"``, ``"Turnbull"``, ...). Defaults to
            ``"Nelson-Aalen"``.

        init : array like, optional
            initial guess of parameters. Instead of finding an initial guess
            for the optimization you can provide one. Can be useful to see if
            optimization is failing due to poor initial guess. For MLE the
            default start is tried as well and the better likelihood kept,
            so a poor guess cannot give a worse fit than none.

        rr : {'y', 'x'}, str, optional
            The dimension on which to minimise the spacing between the line
            and the observation. If 'y' the mean square error between the
            line and vertical distance to each point is minimised. If 'x' the
            mean square error between the line and horizontal distance to each
            point is minimised.

        on_d_is_0 : boolean, optional
            For MPP: whether to keep the points at which nothing failed (a
            time with only censored units, such as a right-censored highest
            value) in the regression. If :code:`False` (the default), every
            point where there are 0 deaths is excluded from the regression;
            if :code:`True` all points are included, whether or not there
            was a death there.

        turnbull_estimator : str, optional
            If using the Turnbull heuristic, the estimator used with the
            Turnbull estimates of r and d: ``'Fleming-Harrington'`` (the
            default), ``'Nelson-Aalen'`` or ``'Kaplan-Meier'``.

        Returns
        -------

        Parametric
            A parametric model with the fitted parameters and methods for
            all functions of the distribution using the fitted parameters.

        Examples
        --------
        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 4)
        >>> model = Weibull.fit(x)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Data                : 100 units: 100 events at 100 unique times
        Parameters          :
             alpha: 9.815018791049368
              beta: 3.798740470368033
        >>> Weibull.fit(x, how='MPS', fixed={'alpha' : 10})
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MPS
        Data                : 100 units: 100 events at 100 unique times
        Parameters          :
             alpha: 10.0
              beta: 3.670796510564323
        >>> Weibull.fit(xl=np.floor(x), xr=np.ceil(x), how='MPP',
        ...             heuristic='Turnbull')
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MPP
        Data                : 100 units: 0 events, 100 interval censored
        Parameters          :
             alpha: 9.834445729732789
              beta: 3.2602770099790424
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
        Data                : 86 units: 80 events at 80 unique times,
                              6 right censored; 86 left truncated
        Parameters          :
             alpha: 9.893584496413128
              beta: 3.78688602908912
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

    def fit_from_ecdf(self, x: npt.ArrayLike, F: npt.ArrayLike) -> Parametric:
        r"""
        Fit the distribution to points of an empirical CDF by probability
        plotting.

        The points ``(x, F)`` are transformed to the distribution's
        probability-plot axes and a straight line is fitted through them
        by least squares (the ``how='MPP'`` regression with ``rr='y'``,
        but on the CDF values given rather than on plotting positions
        computed from data). Points with ``F`` equal to 0 or 1 cannot be
        transformed and are left out. Only distributions that support
        ``how='MPP'`` can be fitted this way.

        Parameters
        ----------
        x : array like
            The values at which the CDF is known.
        F : array like
            The CDF at each ``x``, between 0 and 1, of the same length as
            ``x``.

        Returns
        -------
        Parametric
            A model whose ``method`` is ``'given ecdf'``. It holds no
            data, so it has no likelihood, information criteria or
            confidence bounds.

        Raises
        ------
        ValueError
            If ``x`` and ``F`` differ in length, or an ``F`` is NaN or
            outside [0, 1].

        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.fit_from_ecdf([1, 2, 3, 4], [0.1, 0.3, 0.6, 0.9])
        >>> model.params
        array([2.96150944, 2.1761779 ])
        """
        # The regression needs the distribution's linearising transforms
        # and the map from the fitted line back to its parameters
        # (``unpack_rr``); without them the call died with an IndexError,
        # AttributeError or TypeError depending on the distribution.
        if not (self.supports_mpp and hasattr(self, "unpack_rr")):
            raise ValueError(
                f"{self.name} cannot be fitted to an ECDF: it has no "
                "straight-line probability plot to regress the points on. "
                "Fit it to the data instead (how='MLE')."
            )
        # A value outside [0, 1] or NaN was dropped by the transform's
        # NaN without a word (F = [0.1, 0.3, 1.2, 0.9] fitted alpha 2.886
        # to the other three), and unequal lengths died in an IndexError.
        x_arr = np.asarray(x, dtype=float).ravel()
        F_arr = np.asarray(F, dtype=float).ravel()
        if x_arr.size != F_arr.size:
            raise ValueError(
                f"x and F must have the same length: x has {x_arr.size} "
                f"values and F has {F_arr.size}."
            )
        bad = ~((F_arr >= 0) & (F_arr <= 1))
        if bad.any():
            raise ValueError(
                "F must lie in [0, 1]: got "
                f"{F_arr[bad].tolist()} at x = {x_arr[bad].tolist()}."
            )
        model = Parametric(self, "given ecdf", None, False, False, False)
        res = mpp_from_ecfd(self, x_arr, F_arr)
        model.params = np.array(res["params"])
        model.support = self.support

        return model

    def fit_from_non_parametric(self, non_parametric_model: Any) -> Parametric:
        r"""
        Fit the distribution to a fitted non-parametric model by
        probability plotting.

        Equivalent to :meth:`fit_from_ecdf` with ``x`` the model's
        failure times (those with a death, ``d > 0``) and ``F = 1 - R``
        its estimate there, so a Kaplan-Meier model gives the same
        parameters as ``fit(x, c, n, t, how='MPP',
        heuristic='Kaplan-Meier')`` on its data, censored or not.

        Parameters
        ----------
        non_parametric_model : NonParametric
            A fitted ``KaplanMeier``, ``NelsonAalen``,
            ``FlemingHarrington`` or ``Turnbull`` model.

        Returns
        -------
        Parametric
            A model whose ``method`` is ``'given ecdf'`` (see
            :meth:`fit_from_ecdf`).

        Examples
        --------
        >>> from surpyval import KaplanMeier, Weibull
        >>> km = KaplanMeier.fit([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> Weibull.fit_from_non_parametric(km).params
        array([5.9544901 , 1.35505406])
        """
        # Only the times with a failure are plotted, as ``how='MPP'``
        # does by default (``on_d_is_0=False``): the censored times kept
        # the step of R before them and pulled the line (alpha 10.711
        # where the documented equivalent gives 10.597, #438).
        x = np.asarray(non_parametric_model.x, dtype=float)
        F = 1 - np.asarray(non_parametric_model.R, dtype=float)
        keep = (np.asarray(non_parametric_model.d) > 0) & np.isfinite(x)
        return self.fit_from_ecdf(x[keep], F[keep])

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

        Fit the distribution to data already held in a
        :class:`~surpyval.utils.surpyval_data.SurpyvalData` object.

        :meth:`fit` builds a ``SurpyvalData`` from its arrays and calls this
        method; call it directly to reuse one prepared data object across
        several fits.

        Parameters
        ----------

        surv_data : SurpyvalData
            Survival data in the SurpyvalData class.
        how, offset, zi, lfp, fixed, heuristic, init, rr, on_d_is_0, \
turnbull_estimator
            As for :meth:`fit`.

        Returns
        -------

        Parametric
            A parametric model with the fitted parameters and methods for
            all functions of the distribution using the fitted parameters.

        Examples
        --------
        >>> from surpyval import Weibull, SurpyvalData
        >>> data = SurpyvalData(x=[1, 3, 4, 7, 9], c=[0, 0, 0, 0, 1])
        >>> model = Weibull.fit_from_surpyval_data(data)
        >>> model.params.round(3)
        array([6.022, 1.351])
        """
        how = normalise_how(how)
        x, c, n, t = surv_data.x, surv_data.c, surv_data.n, surv_data.t
        # Clamp the truncation values to the (possibly finite) support edges
        tl, tr = self._clamp_truncation_to_support(t, offset)

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
        self._check_fixed_and_init(model, fixed, init, how)
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
                surv_data,
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
            # Some likelihoods have more than one optimum, and the default
            # start can lead to the worse one: a limited failure population
            # (p and the failure distribution trade off), or a custom
            # distribution whose default start is a grid choice. With a
            # default start, the optimiser is also run from the
            # alternatives each offers, and the best likelihood is kept.
            #
            # A start the user gave is followed by the default start
            # (``[]``), so a poor ``init`` cannot hand back a worse model
            # than no ``init`` would have: from a start far from the
            # maximum the search can stall on a plateau where the
            # likelihood only looks level (#427) -- a NegativeBinomial
            # started at r = 4e6 stops in the Poisson limit, 1.2 below the
            # maximum, at a point with a zero gradient. Where that is
            # still not a verified maximum (see ``mle``), the default's
            # alternatives are tried too.
            user_init = init is not None and len(np.atleast_1d(init)) > 0
            starts: list = []
            from surpyval.utils.refits import warm_starts_on

            # A warm start (``utils.refits.warm_starts``) that reached a
            # verified maximum is not searched from the default too.
            warm = warm_starts_on() and results.get("_verified", False)
            if how == "MLE" and user_init and not warm:
                starts = [[]]
            if how == "MLE" and not fixed:
                if not user_init or not results["_verified"]:
                    starts += self._alternative_starts(
                        surv_data, offset, zi, lfp, heuristic
                    )

            def from_start(start: Any) -> tuple[dict, dict]:
                alt_model = Parametric(self, how, data, offset, lfp, zi)
                alt_model.surv_data = surv_data
                alt_info: dict = {}
                alt = self._fit_numerically(
                    alt_model,
                    alt_info,
                    surv_data,
                    tl,
                    tr,
                    how,
                    offset,
                    zi,
                    lfp,
                    fixed,
                    heuristic,
                    start,
                    rr,
                    on_d_is_0,
                    turnbull_estimator,
                )
                return alt, alt_info

            for start in starts:
                results = self._better_fit(model, results, *from_start(start))
            # A parameter left on a bound of its range (a ``p`` of 1) where
            # the likelihood rises off it is searched once more, from the
            # middle of its range (``_OnBounds.off``): in its own units the
            # search cannot leave the bound, where the parameter no longer
            # moves the likelihood (#579).
            off_bound = results.get("_off_bound")
            if how == "MLE" and not fixed and off_bound is not None:
                results = self._better_fit(
                    model, results, *from_start(off_bound)
                )
        else:
            model.fitting_info = fitting_info

        # Only the answer kept speaks: a start that failed and was beaten
        # by another says nothing. A maximum-likelihood answer that is not
        # verifiably a maximum is never returned in silence (principle 13).
        # A family whose only parameters are its support's end points (the
        # Uniform) has its maximum on the data's extremes, an edge where
        # the gradient does not vanish: there is nothing to verify. Its
        # support is declared data-dependent (NaN): ``support_param_index``
        # alone defaults to (0, 1) for every family, and so exempted every
        # two-parameter family (the Weibull, the Gamma, ...) from this
        # warning.
        warning = results.pop("_warning", None)
        reason = results.pop("_unverified_reason", None)
        edges_only = bool(
            np.isnan(np.asarray(self.support, dtype=float)).all()
        ) and getattr(self, "support_param_index", None) == tuple(
            range(self.k)
        )
        unverified = (
            warning is None
            and not results.pop("_verified", True)
            and not edges_only
        )
        results.pop("_verified", None)
        results.pop("_off_bound", None)
        # What the fit reached, recorded as ``model.maximum`` so that a
        # caller (``fit_best``) need not read it from the warnings; it
        # follows them exactly. An answer with nothing to verify (a closed
        # form, a Uniform's extreme observations) is a maximum.
        maximum = (
            "unverified" if warning is not None or unverified else "verified"
        )
        # A family whose likelihood can be highest in a limit of its
        # parameters says so instead (one warning per fit).
        if (
            how == "MLE"
            and not fixed
            and self._warn_if_at_limit(surv_data, results, zi, lfp)
        ):
            warning = None
            unverified = False
            maximum = "no finite maximum"
        # So does an offset fit that ran its offset onto the first
        # failure, whatever the family (#487).
        if (
            how == "MLE"
            and offset
            and self._warn_if_offset_at_limit(
                surv_data, results, model.fitting_info, zi, lfp
            )
        ):
            warning = None
            unverified = False
            maximum = "no finite maximum"
        # And any search that found a parameter running off (``mle``,
        # #584), where the family has not said so in its own words above.
        runaway = results.pop("_runaway", [])
        by_limit = results.pop("_runaway_by_limit", False)
        if runaway and maximum != "no finite maximum":
            self._warn_runaway(surv_data, runaway, results, offset, by_limit)
            warning = None
            unverified = False
            maximum = "no finite maximum"
        if warning is not None and not maximum_warnings_quiet():
            warnings.warn(warning, stacklevel=3)
        if unverified:
            warn_unverified("The maximum-likelihood search", reason)
        # Only maximum likelihood seeks a maximum of the likelihood
        model.maximum = maximum if how == "MLE" else "not applicable"

        for k, v in results.items():
            setattr(model, k, v)

        # Every fit says how its answer was found, not only MLE and the
        # closed forms: ``optimizer`` was missing after MPP, MOM, MPS and
        # MSE fits.
        if not hasattr(model, "optimizer"):
            model.optimizer = _optimizer_label(how, results.get("res"))

        # A fit must never hand back a non-finite parameter. When the
        # optimiser fails, the reported parameters are the initial guess
        # (#261), so any initialiser that produced a nan or an inf had
        # it laundered into what looked like a fitted model: an offset
        # Gamma on a tied sample returned ``(inf, inf)`` in silence. The
        # initialisers that could do that are fixed, but this is the
        # backstop, since a non-finite parameter is never a valid answer
        # whatever produced it.
        _params = np.atleast_1d(np.asarray(model.params, dtype=float))
        _extra = [
            getattr(model, name, None) for name in ("gamma", "lfp_p", "f0")
        ]
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
                        model.lfp_p,
                    )
                )

        # Expose each fitted parameter by name (e.g. ``model.alpha``), but
        # never overwrite the reserved offset / limited-failure /
        # zero-inflation attributes, which the survival functions rely on.
        # A distribution may legitimately name a parameter ``p`` (e.g.
        # ``Geometric``, ``NegativeBinomial``), which the model's ``p``
        # property gives (#608).
        reserved = {"gamma", "p", "lfp_p", "f0"}
        for k, v in zip(self.parameter_names, model.params):
            if k not in reserved:
                setattr(model, k, v)

        self._set_support(model, offset)

        return model

    @staticmethod
    def _better_fit(
        model: Parametric, results: dict, alt: dict, alt_info: dict
    ) -> dict:
        """``alt``, the results of a fit from another start (with its
        ``fitting_info``, ``alt_info``), where its likelihood is higher
        than that of ``results`` beyond rounding; else ``results``. The
        model takes the ``fitting_info`` of the results kept.

        A verified maximum is kept over a search that found a parameter
        running off (``_runaway``), whatever their likelihoods: a runaway's
        likelihood is a value on the way to a supremum, which can be
        infinite (an offset run onto the first failure, #622), and the
        maximum-likelihood estimate is the maximum where there is one."""
        if bool(results.get("_runaway")) != bool(alt.get("_runaway")):
            verified = results if not results.get("_runaway") else alt
            if verified.get("_verified", False):
                if verified is alt:
                    model.fitting_info = alt_info
                return verified
        best = results.get("_neg_ll", np.inf)
        value = alt.get("_neg_ll", np.inf)
        if np.isfinite(value) and value < best - 1e-9 * max(1.0, abs(value)):
            model.fitting_info = alt_info
            return alt
        return results

    def _warn_if_at_limit(
        self,
        surv_data: SurpyvalData,
        results: dict,
        zi: bool,
        lfp: bool,
    ) -> bool:
        """Warn, and return ``True``, when a maximum-likelihood fit's
        ``results`` sit in a limit of the family where its likelihood has
        no finite maximum (#392): here, data that bound no failure from
        one side (``_warn_if_one_sided``, #559). A family that contains
        another as a limit overrides this (see ``BetaGeometric``)."""
        return self._warn_if_one_sided(surv_data, results, zi, lfp)

    def _at_unbounded_edge(
        self, surv_data: SurpyvalData, values: dict
    ) -> "list[str]":
        """The names of the parameters that, at ``values`` (each
        parameter's value, by name), sit on an edge where the likelihood
        is unbounded nearby, for a maximum-likelihood search to stop at
        (``fitters.mle``, #584); none by default. A family that has such
        edges (``Beta4``) says where. (An offset run onto the first
        failure is such an edge for every family; the search checks it
        itself, with ``_offset_corner``.)"""
        return []

    @staticmethod
    def _first_failure_gap(
        surv_data: SurpyvalData,
    ) -> "tuple[float, float] | None":
        """``(x1, close)``: the smallest exact observation, and how close
        to it an offset is on it (``sqrt(eps)`` of the data's spread, half
        the digits; see ``_warn_if_offset_at_limit``); ``None`` for data
        with no exact observation, or with intervals."""
        x = np.asarray(surv_data.x, dtype=float)
        c = np.asarray(surv_data.c)
        if x.ndim != 1 or not np.any(c == 0):
            return None
        x1 = float(x[c == 0].min())
        finite = x[np.isfinite(x)]
        spread = float(np.ptp(finite)) if finite.size else 0.0
        if spread <= 0:
            spread = max(abs(x1), 1.0)
        return x1, float(np.sqrt(np.finfo(float).eps) * spread)

    def _offset_corner(
        self, surv_data: SurpyvalData, gamma: float, core: npt.NDArray
    ) -> "str | None":
        """Whether an offset ``gamma`` (with the distribution's parameters
        ``core``) has run onto the smallest exact observation, where the
        density at its origin is not finite and positive: ``"infinite"``
        or ``"zero"``, as that density is, else ``None`` (see
        ``_warn_if_offset_at_limit``).

        The likelihood is unbounded there, and a maximum-likelihood search
        that reaches it stops (``fitters.mle``, #622), as the end of the
        fit then says (``_warn_if_offset_at_limit``), rather than run the
        rest of its ladder into the corner: 15,000 likelihood evaluations
        and 7 s for a Weibull of shape 0.8 on ten points."""
        gap = self._first_failure_gap(surv_data)
        if gap is None:
            return None
        if not (np.isfinite(gamma) and np.all(np.isfinite(core))):
            return None
        x1, close = gap
        if not 0 <= x1 - gamma <= close:
            return None
        with np.errstate(all="ignore"):
            f0 = float(np.asarray(self.df(np.array([0.0]), *core))[0])
        if np.isfinite(f0) and f0 > 0:
            return None
        return "infinite" if f0 > 0 else "zero"

    def _warn_runaway(
        self,
        surv_data: SurpyvalData,
        runaway: "list[str]",
        results: dict,
        offset: bool = False,
        by_limit: bool = False,
    ) -> None:
        """Warn that the maximum-likelihood search found the parameters
        ``runaway`` running off (``fitters.mle._runaway``, #584): the
        likelihood keeps increasing towards a limit of the family that
        none of its members reaches, so it has no finite maximum.
        ``offset`` says whether the fit has one, and ``by_limit`` whether
        the runaway was found by the family's limit fitting the data at
        least as well as anything the search reached, rather than by
        Newton's test (#616)."""
        values = dict(
            zip(self.parameter_names, np.atleast_1d(results["params"]))
        )
        values.update(
            gamma=results["gamma"], lfp_p=results["lfp_p"], f0=results["f0"]
        )
        named = ", ".join(f"{name} ({values[name]:.4g})" for name in runaway)
        one = len(runaway) == 1
        how_found = (
            f"no {self.name} the search reached fits the data better than "
            "that limit"
            if by_limit
            else "Newton's method cannot "
            f"converge along {'its' if one else 'their'} "
            f"profile{'' if one else 's'} where the search stopped"
        )
        warn_no_maximum(
            f"the {self.name} likelihood keeps increasing as {named} "
            f"run{'s' if one else ''} on, towards a limit of the family "
            f"that none of its members reaches: {how_found}",
            "The reported parameters are where the search stopped, and "
            "their standard errors and bounds are meaningless",
            self._runaway_advice(runaway, values, offset),
        )

    def _runaway_advice(
        self, runaway: "list[str]", values: dict, offset: bool = False
    ) -> str:
        """What to do instead of a fit whose parameters ``runaway`` run
        off (their ``values`` where the search stopped; ``offset`` whether
        the fit has one), for :meth:`_warn_runaway`; a family that knows
        its limit says so."""
        limit = self._offset_limit_family()
        if offset and limit is not None:
            # An offset fit runs off only towards the family's limit: the
            # offset towards -inf (towards the first failure its range
            # ends; see ``fitters.mle._Judge.keep``) or the shape that
            # makes up for it
            return (
                f"as gamma runs to -inf the {self.name} approaches a "
                f"{limit.name} distribution, which fits these data at least "
                f"as well as any {self.name} the search reached: fit "
                f"surpyval.{limit.name} instead"
            )
        return (
            "a simpler family, or one that contains the limit, may describe "
            "the data: compare their fits (surpyval.fit_best)"
        )

    def _offset_limit_family(self) -> "Any":
        """The family this distribution tends to as its offset runs to
        -inf, with its shape making up for it (``None`` by default): a
        fit whose offset runs that way is running off towards it where
        the limit fits the data at least as well (``fitters.mle``,
        #599). The LogNormal and the Gamma tend to the Normal, the
        Weibull to the smallest extreme value distribution (``Gumbel``)
        and the LogLogistic to the ``Logistic``."""
        return None

    def _warn_if_offset_at_limit(
        self,
        surv_data: SurpyvalData,
        results: dict,
        fitting_info: dict,
        zi: bool,
        lfp: bool,
    ) -> bool:
        """Warn, and return ``True``, when an offset maximum-likelihood fit
        ran its offset onto the smallest exact observation, where the
        likelihood has no finite maximum (#487, #392).

        As the offset ``gamma`` approaches the first failure ``x(1)``, that
        failure's density term is the base density at ``x(1) - gamma -> 0``.
        Where the density is infinite at its origin -- a Weibull, Gamma,
        LogLogistic or ExpoWeibull shape below 1 -- the likelihood grows
        without bound (Smith, 1985); where it is 0 at the origin (the
        LogNormal, a shape above 1) the fit only gets there along a path on
        which the likelihood also grows without bound (Hill, 1963). Either
        way the "estimate" is where the search stopped. A density finite and
        positive at its origin (the Exponential, a Weibull shape of exactly
        1) gives a genuine maximum at ``gamma = x(1)``, which is not
        flagged.

        The criterion is the offset resting on ``x(1)`` to half the digits
        of the data's spread (``sqrt(eps)`` of it, as ``Beta4`` uses for its
        support ends), which an interior maximum -- where the density of
        the first failure is finite and positive -- does not reach, and a
        base density at its origin that is not finite and positive. It is
        checked at the answer and, when the search failed and the answer is
        its start, at the point the search reached; that point is then the
        answer, as it is whenever the search stops in that corner (with no
        covariance: its Hessian there is not computed, and would be
        meaningless). Maximum product of spacings has no such corner: a
        spacing of zero scores minus infinity (Cheng and Amin, 1983).
        """
        from surpyval.utils.no_maximum import warn_no_maximum

        x = np.asarray(surv_data.x, dtype=float)
        c = np.asarray(surv_data.c)
        if x.ndim != 1 or not np.any(c == 0):
            return False
        x1 = float(x[c == 0].min())

        def at_corner(gamma: float, core: npt.NDArray) -> str | None:
            return self._offset_corner(surv_data, gamma, core)

        k = len(np.atleast_1d(results.get("params", [])))
        core = np.asarray(results.get("params", []), dtype=float)
        gamma = float(results.get("gamma", np.nan))
        origin = at_corner(gamma, core)
        res = results.get("res")
        if origin is None and res is not None and "inv_trans" in fitting_info:
            # The search failed and the answer is its start: check where
            # the search itself went.
            with np.errstate(all="ignore"):
                try:
                    reached = np.asarray(
                        fitting_info["inv_trans"](
                            fitting_info["const"](np.asarray(res.x))
                        ),
                        dtype=float,
                    )
                    neg_ll = float(res.fun)
                except Exception:
                    return False
            origin = at_corner(float(reached[0]), reached[1 : 1 + k])
            if origin is None:
                return False
            gamma, core = float(reached[0]), reached[1 : 1 + k]
            rest = list(reached[1 + k :])
            results["gamma"] = gamma
            results["params"] = core
            if zi:
                results["f0"] = rest.pop()
            if lfp:
                results["lfp_p"] = rest.pop()
            results["_neg_ll"] = neg_ll
            results["log_likelihood"] = -neg_ll
            results["_covariance"] = None
            results["hess_inv"] = None
        if origin is None:
            return False
        shape = ", ".join(
            f"{name} = {value:.4g}"
            for name, value in zip(self.parameter_names, core)
        )
        warn_no_maximum(
            f"the offset gamma = {gamma:.6g} ran onto the smallest "
            f"observation {x1:.6g} ({shape}), where the {self.name} "
            f"density is {origin} at its origin: the likelihood of an "
            "offset fit grows without bound as gamma approaches the first "
            "failure, and this data has no interior maximum short of it",
            "The reported gamma and parameters are where the search stopped, "
            "and their standard errors and bounds are meaningless",
            "fit with how='MPS' (maximum product of spacings, the standard "
            "remedy for an offset fit), or without an offset",
        )
        return True

    def _try_closed_form_mle(
        self,
        surv_data: SurpyvalData,
        how: str,
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float] | None,
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

        # A distribution whose "closed form" is, for some data, a
        # dedicated search reports that search as the optimiser.
        label = getattr(self, "_closed_form_optimizer", None)
        optimizer = "closed-form" if label is None else label(surv_data)
        return closed_form_results(self, surv_data, params, optimizer)

    def _fit_numerically(
        self,
        model: Any,
        fitting_info: Any,
        surv_data: SurpyvalData,
        tl: Any,
        tr: Any,
        how: str,
        offset: bool,
        zi: bool,
        lfp: bool,
        fixed: dict[str, float] | None,
        heuristic: str,
        init: Any,
        rr: str,
        on_d_is_0: bool,
        turnbull_estimator: str,
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
            _, _, _, _, not_fixed = bounds_convert(
                surv_data.x, model.bounds, fixed, model.param_map
            )
            # ``len``-based check: comparing an ndarray to ``[]`` raises a
            # broadcast error (#261).
            if init is None or len(np.atleast_1d(init)) == 0:
                init = self._initial_guess(
                    surv_data, offset, zi, lfp, heuristic
                )
                if how == "MPS" and offset and not fixed:
                    # One at which the spacings are not all 0 (#616)
                    init = mps_offset_start(
                        self, surv_data, tl[0], tr[0], init
                    )

            init = np.atleast_1d(init)
            if fixed and len(init) == len(not_fixed):
                # The initial guess covers only the free parameters;
                # merge it with the fixed values to get the full vector
                full_init = np.zeros(len(model.param_map))
                full_init[not_fixed] = init
                for name, value in fixed.items():
                    full_init[model.param_map[name]] = value
                init = full_init

            # Every parameter with one bound is searched in units of its
            # own starting distance from that bound (see ``_search_units``;
            # #366: fits without an offset used units of 1).
            units = _search_units(
                init,
                model.bounds,
                [model.param_map[name] for name in (fixed or {})],
            )
            transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
                surv_data.x, model.bounds, fixed, model.param_map, units
            )
            fitting_info["transform"] = transform
            fitting_info["inv_trans"] = inv_trans
            fitting_info["const"] = const
            fitting_info["fixed_idx"] = fixed_idx

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

        return METHOD_FUNC_DICT[how](model)
