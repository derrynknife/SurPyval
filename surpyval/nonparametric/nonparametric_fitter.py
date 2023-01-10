import numpy as np

from surpyval import nonparametric as nonp
from surpyval.utils import xcnt_handler, xcnt_to_xrd, xrd_handler


class NonParametricFitter:
    def _create_non_p_model(self, x, r, d, estimator, data=None):
        out = nonp.NonParametric()
        if data is not None:
            out.data = data
        out.x = x
        out.r = r
        out.d = d
        out.R = nonp.FIT_FUNCS[estimator](r, d)
        out.model = self.how
        out.F = 1 - out.R
        with np.errstate(all="ignore"):
            out.H = -np.log(out.R)

        out.greenwood = self._compute_var(out.R, r, d)
        return out

    def _compute_var(self, R, r, d):
        with np.errstate(all="ignore"):
            var = d / (r * (r - d))
            var = np.where(np.isfinite(var), var, np.nan)
            greenwood = np.cumsum(var)
        return greenwood

    def fit(
        self,
        x=None,
        c=None,
        n=None,
        t=None,
        xl=None,
        xr=None,
        tl=None,
        tr=None,
        turnbull_estimator="Fleming-Harrington",
        set_lower_limit=None,
    ):
        r"""

        The central feature to SurPyval's capability. This function aimed to
        have an API to mimic the simplicity of the scipy API. That is, to use a
        simple :code:`fit()` call, with as many or as few parameters as are
        needed.

        Parameters
        ----------

        x : array like, optional
            Array of observations of the random variables. If x is
            :code:`None`, xl and xr must be provided.

        c : array like, optional
            Array of censoring flag. -1 is left censored, 0 is observed, 1 is
            right censored, and 2 is intervally censored. If not provided will
            assume all values are observed.

        n : array like, optional
            Array of counts for each x. If data is proivded as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.

        t : 2D-array like, optional
            2D array like of the left and right values at which the
            respective observation was truncated. If not provided it assumes
            that no truncation occurs.

        tl : array like or scalar, optional
            Values of left truncation for observations. If it is a scalar
            value assumes each observation is left truncated at the value.
            If an array, it is the respective 'late entry' of the observation.

        tr : array like or scalar, optional
            Values of right truncation for observations. If it is a scalar
            value assumes each observation is right truncated at the value.
            If an array, it is the respective right truncation value for each
            observation.

        xl : array like, optional
            Array like of the left array for 2-dimensional input of x. This is
            useful for data that is all intervally censored. Must be used with
            the :code:`xr` input.

        xr : array like, optional
            Array like of the right array for 2-dimensional input of x. This is
            useful for data that is all intervally censored. Must be used with
            the :code:`xl` input.

        turnbull_estimator : ('Nelson-Aalen', 'Kaplan-Meier', or
        'Fleming-Harrington'), str, optional
            If using the Turnbull heuristic, you can elect to use either the
            KM, NA, or FH estimator with the Turnbull estimates of r, and d.
            Defaults to FH.

        Returns
        -------

        model : NonParametric
            A parametric model with the fitted parameters and methods for all
            functions of the distribution using the fitted parameters.

        Examples
        --------
        >>> from surpyval import NelsonAalen, Weibull, Turnbull
        >>> import numpy as np
        >>> x = Weibull.random(100, 10, 4)
        >>> model = NelsonAalen.fit(x)
        >>> print(model)
        Non-Parametric SurPyval Model
        =============================
        Model            : Nelson-Aalen
        >>> Turnbull.fit(x, turnbull_estimator='Kaplan-Meier')
        Non-Parametric SurPyval Model
        =============================
        Model            : Turnbull
        Estimator        : Kaplan-Meier
        """
        x, c, n, t = xcnt_handler(
            x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr
        )

        data = {}
        data["x"] = x
        data["c"] = c
        data["n"] = n
        data["t"] = t

        if self.how == "Turnbull":
            data["estimator"] = turnbull_estimator
            out = nonp.NonParametric()
            t_obj = nonp.turnbull(x, c, n, t, turnbull_estimator)
            for k, v in t_obj.items():
                setattr(out, k, v)

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

    def from_xrd(self, x, r, d):
        r"""
        The central feature to SurPyval's capability. This function aimed to
        have an API to mimic the simplicity of the scipy API. That is, to use a
        simple :code:`fit()` call, with as many or as few parameters as are
        needed.

        Parameters
        ----------

        x : array like, optional
            Array of observations of the random variables. If x is
            :code:`None`, xl and xr must be provided.

        r : array like, optional
            Array of at risk items. For each value of x the r array is
            the number of at risk items immediately prior to the failures
            at x.

        d : array like, optional
            Array of counts of deaths/failures at each x. For each value of x
            the d array is the number of deaths at x (can be zero).

        Returns
        -------

        model : NonParametric
            A non-parametric model with the survival curved estimated
            using the selected method.

        Examples
        --------
        >>> from surpyval import NelsonAalen, Weibull, Turnbull
        >>> import numpy as np
        >>> x = [1, 2, 3, 4, 5, 6]
        >>> r = [10, 8, 6, 4, 3, 2]
        >>> d = [2, 1, 1, 1, 1, 1]
        >>> model = NelsonAalen.from_xrd(x)
        >>> print(model)
        Non-Parametric SurPyval Model
        =============================
        Model            : Nelson-Aalen
        """
        if self.how == "Turnbull":
            raise ValueError("Can't use from_xrd with Turnbull estimator")

        x, r, d = xrd_handler(x, r, d)

        return self._create_non_p_model(x, r, d, estimator=self.how)
