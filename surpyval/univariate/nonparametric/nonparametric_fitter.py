from numbers import Number
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.univariate import nonparametric as nonp
from surpyval.univariate.nonparametric.nonparametric import NonParametric
from surpyval.utils import xcnt_handler, xcnt_to_xrd, xrd_handler


class NonParametricFitter:
    how: str
    # Provided by the Turnbull estimator subclass; only called on the
    # ``how == "Turnbull"`` path.
    _fit: Callable[..., dict[str, Any]]

    def _create_non_p_model(
        self,
        x: npt.ArrayLike,
        r: npt.ArrayLike,
        d: npt.ArrayLike,
        estimator: str,
        data: dict | None = None,
    ) -> NonParametric:
        out = NonParametric()
        if data is not None:
            out.data = data
        out.x = np.asarray(x)
        out.r = np.asarray(r)
        out.d = np.asarray(d)
        out.R = nonp.FIT_FUNCS[estimator](r, d)
        out.model = self.how
        out.F = 1 - out.R
        with np.errstate(all="ignore"):
            out.H = -np.log(out.R)

        out.greenwood = self._compute_var(estimator, r, d)
        return out

    def _compute_var(
        self, estimator: str, r: npt.ArrayLike, d: npt.ArrayLike
    ) -> npt.NDArray:
        # Variance of the cumulative hazard estimate using the formula
        # appropriate to the estimator, e.g. Greenwood's formula for
        # Kaplan-Meier. See VAR_FUNCS for each.
        return nonp.VAR_FUNCS[estimator](
            np.asarray(r, dtype=float), np.asarray(d, dtype=float)
        )

    def fit(
        self,
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | Number | None = None,
        tr: npt.ArrayLike | Number | None = None,
        turnbull_estimator: str = "Fleming-Harrington",
        set_lower_limit: float | None = None,
        tol: float = 1e-10,
        max_iter: int = 1000,
    ) -> NonParametric:
        r"""

        Estimate the survival function of the data non-parametrically.

        The estimator is the one this instance was created for
        (``KaplanMeier``, ``NelsonAalen``, ``FlemingHarrington`` or
        ``Turnbull``). Pass as many or as few of the arguments as the data
        needs; only ``x`` (or ``xl`` and ``xr``) is required.

        Parameters
        ----------

        x : array like, optional
            Array of observations of the random variables, or (Turnbull
            only) a 2-D array of ``[left, right]`` intervals. If x is
            :code:`None`, xl and xr must be provided.

        c : array like, optional
            Array of censoring flag. -1 is left censored, 0 is observed, 1 is
            right censored, and 2 is intervally censored. If not provided will
            assume all values are observed.

        n : array like, optional
            Array of counts for each x. If data is provided as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.

        t : 2D-array like, optional
            2D array like of the left and right values at which the
            respective observation was truncated. If not provided it assumes
            that no truncation occurs.

        xl : array like, optional
            Array like of the left array for 2-dimensional input of x. This is
            useful for data that is all intervally censored. Must be used with
            the :code:`xr` input.

        xr : array like, optional
            Array like of the right array for 2-dimensional input of x. This is
            useful for data that is all intervally censored. Must be used with
            the :code:`xl` input.

        tl : array like or scalar, optional
            Values of left truncation for observations. If it is a scalar
            value assumes each observation is left truncated at the value.
            If an array, it is the respective 'late entry' of the observation.
            An item is at risk at times in ``(tl, x]``, so each ``tl`` must be
            strictly less than its value.

        tr : array like or scalar, optional
            Values of right truncation for observations. If it is a scalar
            value assumes each observation is right truncated at the value.
            If an array, it is the respective right truncation value for each
            observation. Turnbull only.

        turnbull_estimator : str, optional
            Turnbull only: one of ``'Fleming-Harrington'`` (the default),
            ``'Nelson-Aalen'`` or ``'Kaplan-Meier'``, the estimator used with
            the Turnbull estimates of r and d; any other value raises a
            ``ValueError``. Ignored by the other estimators.

            **This default is why a Turnbull fit does not equal a
            KaplanMeier fit on data both can handle.** Where the option
            acts depends on whether the data are truncated:

            - Without truncation it is used inside the EM as well as at the
              end: each self-consistency step redistributes the uncertain
              observations with the chosen estimator's survival curve, so
              the expected ``r`` and ``d`` the EM converges to depend on
              the option too. A Turnbull fit with the NA or FH option is
              therefore *not* the same as ``NelsonAalen`` or
              ``FlemingHarrington`` on right-censored data: on
              ``x=[2,3,3,4,5,6], c=[0,1,0,0,1,0]`` the survival at 4 is
              0.501 with the NA option against 0.497 from ``NelsonAalen``.
            - With truncation the EM always iterates with the Kaplan-Meier
              (self-consistency) update, whatever the option, and the
              chosen estimator is applied only to the converged ``r`` and
              ``d``. On ``x=[2,3,3,4,5,6], tl=[0,0,1,1,2,2]`` the survival
              at 2 is 0.750 under KM, 0.765 under FH and 0.779 under NA.

            Pass ``turnbull_estimator='Kaplan-Meier'`` to compare like with
            like -- it then agrees with :code:`KaplanMeier` to around 1e-9
            on both ``sf`` and ``cb``, on right-censored and left-truncated
            data alike. (The agreement is to the EM's tolerance, not to the
            last digit.)

            Only the KM option is the non-parametric MLE. NA and FH are
            ``exp(-H)`` constructions and are not trying to maximise the
            likelihood: on the data above the direct NPMLE of the truncated
            likelihood is 0.750, and FH's 0.765 scores worse on it by
            design. FH is the default because it behaves better than KM in
            the far tails and on zero-inflated data (see the v0.8.0 notes),
            not because it is the maximum likelihood answer.

        set_lower_limit : float, optional
            Not used by Turnbull. If given, a point is prepended at this
            value with no deaths (and the risk set of the first time), so
            the estimate starts at ``R = 1`` from this value, typically 0,
            rather than from the first observed time. It must be below
            the smallest observed value (this is not checked).

        tol : float, optional
            Turnbull only. The EM stops once the largest change in any
            interval's probability mass falls below this. Defaults to 1e-10.

        max_iter : int, optional
            Turnbull only. Cap on EM iterations; a warning is raised if it
            is reached before ``tol`` is. Defaults to 1000. Both ``tol`` and
            ``max_iter`` are kept with the model, and ``bootstrap_cb``
            refits every resample with them.

        Returns
        -------

        model : NonParametric
            The fitted non-parametric model, with the survival, hazard and
            quantile functions, confidence bounds and plotting. A Turnbull
            model also carries ``bounds``, ``R_upper``, ``R_lower``,
            ``turnbull_estimator``, ``converged``, ``iters``, ``degenerate``
            and ``exploitable_mass`` (see
            :class:`~surpyval.univariate.nonparametric.turnbull.Turnbull_`).

        Raises
        ------

        ValueError
            If the data has left- (``c=-1``) or interval- (``c=2``) censored
            or right truncated observations and the estimator is not
            ``Turnbull``, or if a ``Turnbull`` fit is given an unknown
            ``turnbull_estimator``.

        Examples
        --------
        >>> from surpyval import KaplanMeier, NelsonAalen, Turnbull
        >>> model = KaplanMeier.fit([2, 3, 3, 4, 5, 6], c=[0, 1, 0, 0, 1, 0])
        >>> model.r
        array([6, 5, 3, 2, 1])
        >>> model.d
        array([1, 1, 1, 0, 1])
        >>> model.R
        array([0.83333333, 0.66666667, 0.44444444, 0.44444444, 0.        ])

        With delayed entry the risk set can grow:

        >>> model = KaplanMeier.fit([2, 3, 3, 4, 5, 6], tl=[0, 0, 1, 1, 2, 2])
        >>> model.r
        array([4, 5, 3, 2, 1])
        >>> model.sf([2, 4])
        array([0.75, 0.3 ])
        >>> print(NelsonAalen.fit([2, 3, 3, 4, 5, 6]))
        Non-Parametric SurPyval Model
        =============================
        Model            : Nelson-Aalen
        >>> Turnbull.fit([2, 3, 3, 4, 5, 6], turnbull_estimator='Kaplan-Meier')
        Non-Parametric SurPyval Model
        =============================
        Model            : Turnbull
        Estimator        : Kaplan-Meier
        """
        if self.how == "Turnbull":
            # Imported here as this module is imported by the package
            # __init__ before ``turnbull`` is.
            from surpyval.univariate.nonparametric.turnbull import (
                check_turnbull_estimator,
            )

            check_turnbull_estimator(turnbull_estimator)

        x, c, n, t = xcnt_handler(
            x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr
        )

        data: dict[str, Any] = {}
        data["x"] = x
        data["c"] = c
        data["n"] = n
        data["t"] = t

        if self.how == "Turnbull":
            data["estimator"] = turnbull_estimator
            # Kept with the estimator so that ``bootstrap_cb`` refits every
            # resample with the settings this fit used, not the defaults.
            data["tol"] = tol
            data["max_iter"] = max_iter
            out = NonParametric()
            t_obj = self._fit(x, c, n, t, turnbull_estimator, tol, max_iter)

            # Truncated fits supply a separate observed-information ladder
            # for the variance (the estimation ladder's ghost events would
            # understate it); untruncated fits use the estimation ladder.
            var_r = t_obj.pop("var_r", t_obj["r"])
            var_d = t_obj.pop("var_d", t_obj["d"])
            out.greenwood = self._compute_var(turnbull_estimator, var_r, var_d)
            for k, v in t_obj.items():
                setattr(out, k, v)
            # The cumulative hazard, defined as for every other estimator
            # (and as ``Hf`` evaluates it): -log of the reported survival.
            # For the NA and FH options that is exactly their summed
            # hazard, since they report R = exp(-H).
            with np.errstate(all="ignore"):
                out.H = -np.log(out.R)

            out.data = data
            return out

        else:
            x, r, d = xcnt_to_xrd(x, c, n, t)
            estimator = self.how

        if set_lower_limit is not None:
            x = np.hstack([[set_lower_limit], x])
            r = np.hstack([[r[0]], r])
            d = np.hstack([[0], d])

        return self._create_non_p_model(
            x, r, d, estimator=estimator, data=data
        )

    def from_xrd(
        self, x: npt.ArrayLike, r: npt.ArrayLike, d: npt.ArrayLike
    ) -> NonParametric:
        r"""
        Build the estimate from data already reduced to risk and death
        sets, the ``xrd`` format: the distinct times, the number at risk
        just before each and the number of deaths at each.

        Not available for ``Turnbull``, which needs the full ``xcnt`` data
        to redistribute censored observations.

        Parameters
        ----------

        x : array like
            The distinct event times, in increasing order (the order is
            not checked, and ``r`` and ``d`` are paired with ``x`` as
            given).

        r : array like
            Array of at risk items. For each value of x the r array is
            the number of at risk items immediately prior to the failures
            at x.

        d : array like
            Array of counts of deaths/failures at each x. For each value of x
            the d array is the number of deaths at x (can be zero).

        Returns
        -------

        model : NonParametric
            A non-parametric model with the survival curve estimated
            using the selected method.

        Examples
        --------
        >>> from surpyval import NelsonAalen, Weibull, Turnbull
        >>> import numpy as np
        >>> x = [1, 2, 3, 4, 5, 6]
        >>> r = [10, 8, 6, 4, 3, 2]
        >>> d = [2, 1, 1, 1, 1, 1]
        >>> model = NelsonAalen.from_xrd(x, r, d)
        >>> print(model)
        Non-Parametric SurPyval Model
        =============================
        Model            : Nelson-Aalen
        >>> model.R
        array([0.81873075, 0.72252735, 0.6116062 , 0.47631939, 0.34129776,
               0.20700755])
        """
        if self.how == "Turnbull":
            raise ValueError("Can't use from_xrd with Turnbull estimator")

        x, r, d = xrd_handler(x, r, d)

        return self._create_non_p_model(x, r, d, estimator=self.how)
