from __future__ import annotations

from typing import Any

import autograd.numpy as np
import numpy as onp
import numpy.typing as npt
from autograd.tracer import getval
from scipy.special import betaincinv

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.autograd_gamma_compat import betaincln as abetaincln
from surpyval.utils.autograd_gamma_compat import betaln_accurate
from surpyval.utils.surpyval_data import SurpyvalData

from ._quantile import solve_log_tails
from ._stable import (
    log1mexp,
    on_support,
    positive_or_one,
    power_at_zero,
    softplus,
)

#: Past this |v| = |z + ln(s1 / s2)| the beta variable (or its complement)
#: is below e^-700, where it underflows; the tails are then the first
#: term of the incomplete beta's series, exact to e^-700.
_FAR_V = 700.0


def _shapes(Q: Boxable, P: Boxable) -> tuple:
    r"""``(delta, s1, s2)``: :math:`\delta = \sqrt{Q^2 + 2P}`, :math:`s_1 =
    2 / (Q^2 + 2P + Q\delta)` and :math:`s_2 = 2 / (Q^2 + 2P - Q\delta)`.
    One of the two denominators is a difference that cancels when
    :math:`P \ll Q^2`; it is taken from :math:`s_1 s_2 = 2 / (P (Q^2 +
    2P))` instead."""
    tmp = Q * Q + 2.0 * P
    delta = np.sqrt(tmp)
    product = 2.0 / (P * tmp)
    big = tmp + np.abs(Q) * delta  # no cancellation
    s_small = 2.0 / big  # the shape on Q's side
    s_large = product / s_small
    positive = Q >= 0
    s1 = np.where(positive, s_small, s_large)
    s2 = np.where(positive, s_large, s_small)
    return delta, s1, s2


class GeneralizedF_(OptimisedFitMixin, ParametricFitter):
    r"""
    The generalised F distribution in Prentice's (1975)
    parameterisation, the one R's ``flexsurv`` calls ``genf``: the
    generalised gamma with a fourth parameter, ``P``, that lets the hazard
    take still more shapes and the tail be heavier. It holds the
    generalised gamma (``P = 0``, as a limit) and so the Weibull, the
    lognormal and the gamma, and the log-logistic (``Q = 0, P = 1``).

    The parameters are ``mu``, the location of the log time, ``sigma``,
    the scale of the log time (positive), and the shapes ``Q``, any real
    number, and ``P``, positive. With :math:`\delta = \sqrt{Q^2 + 2P}`,
    :math:`s_1 = 2 / (Q^2 + 2P + Q\delta)`, :math:`s_2 = 2 / (Q^2 + 2P -
    Q\delta)` and :math:`z = \delta (\ln x - \mu) / \sigma`, the variable
    :math:`e^{z}` has an F distribution on :math:`2 s_1` and :math:`2 s_2`
    degrees of freedom, so

    .. math::
        R(x) = 1 - I_{B}(s_1, s_2), \qquad
        B = \frac{s_1 e^{z}}{s_1 e^{z} + s_2},

    with :math:`I` the regularised incomplete beta function.

    Four parameters ask a lot of the data: fit the generalised gamma and
    compare by likelihood ratio before relying on ``P``: where the
    generalised gamma describes the data, the likelihood is highest at
    the edge ``P = 0``, and the fit stops near it with a warning that its
    maximum is not verified.

    ``fit`` estimates the parameters from data (which may be censored
    and truncated); ``from_params`` builds the model from known values.

    Examples
    --------
    >>> from surpyval import GeneralizedF
    >>> model = GeneralizedF.from_params([2.0, 0.5, 0.5, 1.0])
    >>> model.sf([2, 5, 10]).round(4)
    array([0.9432, 0.6809, 0.2349])
    """

    # The Parameters entries the method docstrings below repeat. Edit them
    # here and run scripts/sync_docstrings.py --write to update the copies.
    _param_docs = """
        mu : numpy array or scalar
            location of the log time for the GeneralizedF distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedF distribution
        Q : numpy array or scalar
            first shape parameter for the GeneralizedF distribution
        P : numpy array or scalar
            second shape parameter for the GeneralizedF distribution
        """

    # mu is a log time; sigma, Q and P have no unit.
    _time_roles = ("log_time", "shape", "shape", "shape")

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=4,
            bounds=((None, None), (0, None), (None, None), (0, None)),
            support=(0, np.inf),
            parameter_names=["mu", "sigma", "Q", "P"],
            param_map={"mu": 0, "sigma": 1, "Q": 2, "P": 3},
            plot_x_scale="log",
        )
        # No linearising probability plot (the axis depends on Q and P).
        self.supports_mpp = False

    def _offset_limit_family(self) -> Any:
        from surpyval.univariate.parametric import Normal

        return Normal

    def _check_method(
        self,
        surv_data: SurpyvalData,
        how: str,
        offset: bool,
        lfp: bool,
        zi: bool,
        fixed: dict[str, float] | None,
    ) -> None:
        """Refuse the method of moments, then the checks of every family.
        The upper tail falls as a power of x, so only the moments below
        ``s2 delta / sigma`` exist: matching the first four can pick a
        member whose fourth moment is infinite (on 400 heavy-tailed
        quantiles it put sf(1) at 0.55 where the maximum likelihood fit
        has 0.92)."""
        if how == "MOM":
            raise ValueError(
                "GeneralizedF cannot be fitted by the method of moments "
                "(how='MOM'): its upper tail falls as a power of x, so its "
                "higher moments may not exist; use how='MLE', 'MPS' or 'MSE'"
            )
        super()._check_method(surv_data, how, offset, lfp, zi, fixed)

    def _refuse_mpp_without_plot(self) -> None:
        raise ValueError(
            "GeneralizedF distribution does not work with probability plot "
            "fitting: its straight-line axis depends on Q and P; use "
            "how='MLE', 'MPS' or 'MSE' instead"
        )

    def _runaway_advice(
        self, runaway: list[str], values: dict, offset: bool = False
    ) -> str:
        if "P" in runaway and values["P"] < 1:
            return (
                "as P falls to 0 the generalised F approaches the "
                "generalised gamma with the same mu, sigma and Q: fit "
                "surpyval.GeneralizedGamma instead"
            )
        return super()._runaway_advice(runaway, values, offset)

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        if offset:
            return self._offset_seed(data)
        # The generalised gamma fitted to the data, with a small P: the
        # member of the family nearest to it.
        from surpyval.univariate.parametric import GeneralizedGamma

        with onp.errstate(all="ignore"):
            gg = GeneralizedGamma._fit_from_surpyval_data(data, how="MLE")
        mu, sigma, Q = (float(v) for v in gg.params)
        if not (onp.isfinite(mu) and sigma > 0 and onp.isfinite(Q)):
            log_x = onp.log(onp.asarray(data.x, dtype=float))
            log_x = log_x[onp.isfinite(log_x)]
            mu = float(onp.mean(log_x)) if log_x.size else 0.0
            sigma = float(onp.std(log_x)) if log_x.size > 1 else 1.0
            sigma, Q = (sigma if sigma > 0 else 1.0), 0.0
        return onp.array([mu, sigma, Q, 0.1])

    @staticmethod
    def _z(x: Numeric, mu: Boxable, sigma: Boxable, delta: Boxable) -> tuple:
        """``z = delta (ln x - mu) / sigma`` and ``x``, with the points at
        and below 0, and at infinity, replaced by 1 (for ``_edges``)."""
        x_pos = positive_or_one(x)
        if onp.any(onp.isposinf(getval(x_pos))):
            x_pos = np.where(np.isinf(x_pos), 1.0, x_pos)
        return delta * (np.log(x_pos) - mu) / sigma, x_pos

    @staticmethod
    def _edges(x: Numeric, inside: Boxable, at_zero: Any, at_inf: Any) -> Any:
        """``inside`` in the support, its limits at x = 0 and x = inf."""
        out = on_support(x, inside, at_zero)
        if onp.any(onp.isposinf(onp.asarray(getval(x), dtype=float))):
            out = np.where(np.isposinf(x), at_inf + np.zeros_like(out), out)
        return out

    def _tails(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> tuple:
        r"""``(ln S, ln F)``: the logs of the two tails of the beta
        variable :math:`B`, at :math:`B` and its complement :math:`1 - B =
        s_2 / (s_2 + s_1 e^z)`, each computed from logs so neither rounds
        (:math:`\ln B = -\operatorname{softplus}(-(z + \ln(s_1/s_2)))`)."""
        delta, s1, s2 = _shapes(Q, P)
        z, _ = self._z(x, mu, sigma, delta)
        v = z + np.log(s1) - np.log(s2)
        with onp.errstate(over="ignore", under="ignore"):
            b = np.exp(-softplus(-v))
            one_minus_b = np.exp(-softplus(v))
        # F = I_B(s1, s2) and S = I_{1-B}(s2, s1), each from whichever of
        # B and 1 - B is at most 1/2: the incomplete beta near an argument
        # of 1 works from its complement, which has rounded (F at 0.999
        # was 7e-6 off where s2 = 0.2), so the other tail is the
        # complement of the one computed directly.
        small_b = v <= 0.0
        # Where the series below takes over (|v| > _FAR_V), the direct
        # forms are evaluated at a harmless 0.25 instead: there B or 1 - B
        # underflows, and their gradient would be nan, which the masks
        # below would carry into every derivative.
        far = np.abs(v) > _FAR_V
        b_in = np.where(small_b & ~far, b, 0.25)
        c_in = np.where(small_b | far, 0.25, one_minus_b)
        direct_f = abetaincln(s1, s2, b_in)
        direct_s = abetaincln(s2, s1, c_in)
        if np.any(far):
            # and their complements from a log of -1, not of a tail that
            # may round to 0 at extreme shapes
            direct_f = np.where(far, -1.0, direct_f)
            direct_s = np.where(far, -1.0, direct_s)
        from_f = log1mexp(-direct_f, np.log(-direct_f))[0]
        from_s = log1mexp(-direct_s, np.log(-direct_s))[0]
        log_f = np.where(small_b, direct_f, from_s)
        log_s = np.where(small_b, from_f, direct_s)
        # Far out, where B (or 1 - B) underflows, the series of the
        # incomplete beta to its first term in logs, I_y(a, b) = y^a /
        # (a B(a, b)) (1 + O(y)), with ln y = -softplus(-v) (or
        # -softplus(v)) itself finite: the lower and the power-law upper
        # tail keep falling instead of stopping at -inf.
        if np.any(far):
            v_far = np.where(far, v, 2.0 * _FAR_V)
            lead = -betaln_accurate(s1, s2)
            tail_f = s1 * -softplus(-v_far) - np.log(s1) + lead
            tail_s = s2 * -softplus(v_far) - np.log(s2) + lead
            low = far & (v < 0)
            high = far & (v > 0)
            # and the other tail is 1 minus it, in logs: about -F (or
            # -S), not 0 (log_sf was 0 where F was 1e-300); each formed
            # only on its own rows (from -1 elsewhere), so that no nan
            # reaches a gradient
            r_s = -np.where(high, tail_s, -1.0)
            r_f = -np.where(low, tail_f, -1.0)
            near_f = log1mexp(r_s, np.log(r_s))[0]
            near_s = log1mexp(r_f, np.log(r_f))[0]
            log_f = np.where(low, tail_f, np.where(high, near_f, log_f))
            log_s = np.where(high, tail_s, np.where(low, near_s, log_s))
        return log_s, log_f

    def log_sf(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        return self._edges(x, self._tails(x, mu, sigma, Q, P)[0], 0.0, -np.inf)

    def log_ff(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        return self._edges(x, self._tails(x, mu, sigma, Q, P)[1], -np.inf, 0.0)

    def log_df(
        self, x: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        # ln delta - ln(sigma x) + s1 v - (s1 + s2) softplus(v) - ln B(s1,s2)
        # with v = z + ln(s1 / s2), the middle as -s1 softplus(-v) - s2
        # softplus(v): the same, since softplus(v) - v = softplus(-v), but
        # without s1 v and s1 softplus(v) cancelling far out (they lost 3
        # digits at s1 = 200, v = 24)
        delta, s1, s2 = _shapes(Q, P)
        z, x_pos = self._z(x, mu, sigma, delta)
        v = z + np.log(s1) - np.log(s2)
        inside = (
            np.log(delta)
            - np.log(sigma)
            - np.log(x_pos)
            - s1 * softplus(-v)
            - s2 * softplus(v)
            - betaln_accurate(s1, s2)
        )
        return self._edges(
            x,
            inside,
            lambda: self._log_df_at_zero(sigma, delta, s1, mu, s2),
            -np.inf,
        )

    def sf(
        self,
        x: Numeric,
        mu: Boxable,
        sigma: Boxable,
        Q: Boxable,
        P: Boxable,
    ) -> Boxable:
        r"""

        Survival (or reliability) function for the GeneralizedF
        distribution, :math:`R(x) = I_{1-B}(s_2, s_1)` with :math:`1 - B =
        s_2 / (s_2 + s_1 e^{z})`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedF distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedF distribution
        Q : numpy array or scalar
            first shape parameter for the GeneralizedF distribution
        P : numpy array or scalar
            second shape parameter for the GeneralizedF distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedF
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedF.sf(x, 1.5, 0.6, 0.5, 1.0)
        array([0.93645073, 0.80655108, 0.36929061, 0.08189491])
        """
        return np.exp(self.log_sf(x, mu, sigma, Q, P))

    def ff(
        self,
        x: Numeric,
        mu: Boxable,
        sigma: Boxable,
        Q: Boxable,
        P: Boxable,
    ) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the GeneralizedF
        distribution, :math:`F(x) = I_{B}(s_1, s_2)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedF distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedF distribution
        Q : numpy array or scalar
            first shape parameter for the GeneralizedF distribution
        P : numpy array or scalar
            second shape parameter for the GeneralizedF distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedF
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedF.ff(x, 1.5, 0.6, 0.5, 1.0)
        array([0.06354927, 0.19344892, 0.63070939, 0.91810509])
        """
        return np.exp(self.log_ff(x, mu, sigma, Q, P))

    def df(
        self,
        x: Numeric,
        mu: Boxable,
        sigma: Boxable,
        Q: Boxable,
        P: Boxable,
    ) -> Boxable:
        r"""

        Density function for the GeneralizedF distribution:

        .. math::
            f(x) = \frac{\delta\, (s_1/s_2)^{s_1} e^{s_1 z}}{\sigma x\,
            B(s_1, s_2) \left(1 + s_1 e^{z} / s_2\right)^{s_1 + s_2}}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedF distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedF distribution
        Q : numpy array or scalar
            first shape parameter for the GeneralizedF distribution
        P : numpy array or scalar
            second shape parameter for the GeneralizedF distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedF
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedF.df(x, 1.5, 0.6, 0.5, 1.0)
        array([0.10443958, 0.14920015, 0.11380869, 0.02228852])
        """
        return np.exp(self.log_df(x, mu, sigma, Q, P))

    def hf(
        self,
        x: Numeric,
        mu: Boxable,
        sigma: Boxable,
        Q: Boxable,
        P: Boxable,
    ) -> Boxable:
        r"""

        Instantaneous hazard rate for the GeneralizedF distribution,
        :math:`h(x) = f(x)/R(x)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedF distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedF distribution
        Q : numpy array or scalar
            first shape parameter for the GeneralizedF distribution
        P : numpy array or scalar
            second shape parameter for the GeneralizedF distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedF
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedF.hf(x, 1.5, 0.6, 0.5, 1.0)
        array([0.11152704, 0.18498537, 0.30818191, 0.27215999])
        """
        # inside the support ln f - ln S; at 0 the density, at inf 0 (the
        # upper tail is a power of x, so the hazard falls like 1 / x)
        delta, s1, s2 = _shapes(Q, P)
        z, x_pos = self._z(x, mu, sigma, delta)
        inside = np.exp(
            self.log_df(x_pos, mu, sigma, Q, P)
            - self.log_sf(x_pos, mu, sigma, Q, P)
        )
        at_zero = np.exp(self._log_df_at_zero(sigma, delta, s1, mu, s2))
        return self._edges(x, inside, at_zero, 0.0)

    @staticmethod
    def _log_df_at_zero(
        sigma: Boxable,
        delta: Boxable,
        s1: Boxable,
        mu: Boxable = 0.0,
        s2: Boxable = 1.0,
    ) -> Boxable:
        r"""The log density's limit at x = 0, where it behaves like
        :math:`c\,x^{\delta s_1/\sigma - 1}`: -inf, :math:`\ln c` or inf
        as the power is above, at or below 0."""
        log_c = (
            np.log(delta)
            - np.log(sigma)
            + s1 * (np.log(s1) - np.log(s2) - delta * mu / sigma)
            - betaln_accurate(s1, s2)
        )
        return power_at_zero(delta * s1 / sigma - 1.0, log_c)[1]

    def Hf(
        self,
        x: Numeric,
        mu: Boxable,
        sigma: Boxable,
        Q: Boxable,
        P: Boxable,
    ) -> Boxable:
        r"""

        Cumulative hazard rate for the GeneralizedF distribution,
        :math:`H(x) = -\ln R(x)`.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        mu : numpy array or scalar
            location of the log time for the GeneralizedF distribution
        sigma : numpy array or scalar
            scale of the log time for the GeneralizedF distribution
        Q : numpy array or scalar
            first shape parameter for the GeneralizedF distribution
        P : numpy array or scalar
            second shape parameter for the GeneralizedF distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedF
        >>> x = np.array([1, 2, 5, 10])
        >>> GeneralizedF.Hf(x, 1.5, 0.6, 0.5, 1.0)
        array([0.06565837, 0.21498805, 0.99617137, 2.50231844])
        """
        return -self.log_sf(x, mu, sigma, Q, P)

    def _from_beta(
        self, b: Any, one_minus_b: Any, mu: Any, sigma: Any, Q: Any, P: Any
    ) -> Any:
        delta, s1, s2 = (float(v) for v in _shapes(Q, P))
        with onp.errstate(divide="ignore", over="ignore"):
            # e^z = s2 B / (s1 (1 - B))
            z = onp.log(b) - onp.log(one_minus_b) + onp.log(s2 / s1)
            return onp.exp(float(mu) + float(sigma) * z / delta)

    def _start(
        self, p: Any, s: Any, mu: Any, sigma: Any, Q: Any, P: Any
    ) -> Any:
        """A start for the solver: the inverse incomplete beta."""
        delta, s1, s2 = (float(v) for v in _shapes(Q, P))
        with onp.errstate(all="ignore"):
            lower = p <= s
            # B = s1 e^z / (s1 e^z + s2) is Beta(s1, s2); its upper tail is
            # 1 - B, Beta(s2, s1)
            b = onp.where(lower, betaincinv(s1, s2, p), 1.0)
            c = onp.where(lower, 1.0 - b, betaincinv(s2, s1, s))
            b = onp.where(lower, b, 1.0 - c)
            return self._from_beta(b, c, mu, sigma, Q, P)

    def qf(
        self, u: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        r"""

        Quantile function for the GeneralizedF distribution: the inverse
        incomplete beta function for a start, refined by Newton's method
        on the log of the smaller tail, so that it inverts ``ff`` (and
        ``isf`` inverts ``sf``) to about 1e-14 in both tails.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import GeneralizedF
        >>> GeneralizedF.qf(np.array([.1, .5, .9]), 1.5, 0.6, 0.5, 1.0)
        array([1.31990576, 3.98399262, 9.2803683 ])
        """
        u = onp.asarray(u, dtype=float)
        params = tuple(float(getval(v)) for v in (mu, sigma, Q, P))
        start = self._start(u, 1.0 - u, *params)
        return solve_log_tails(self, u, 1.0 - u, params, start)

    def isf(
        self, s: Numeric, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        r"""The inverse survival function: the time at which the survival
        function is ``s``, from ``s`` itself.

        Examples
        --------
        >>> from surpyval import GeneralizedF
        >>> GeneralizedF.isf([0.5, 1e-300], 1.5, 0.6, 0.5, 1.0)
        array([3.98399262e+00, 5.12416026e+90])
        """
        s = onp.asarray(s, dtype=float)
        params = tuple(float(getval(v)) for v in (mu, sigma, Q, P))
        start = self._start(1.0 - s, s, *params)
        return solve_log_tails(self, 1.0 - s, s, params, start)

    def moment(
        self, m: int, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        r"""
        The ``m``-th raw moment, with :math:`k = m\sigma/\delta`:

        .. math::
            E[X^m] = e^{m\mu} \left(\frac{s_2}{s_1}\right)^{k}
            \frac{B(s_1 + k, s_2 - k)}{B(s_1, s_2)},

        infinite unless :math:`s_2 > k`: the upper tail falls as a power of
        x, so only the moments below it exist.

        Examples
        --------
        >>> from surpyval import GeneralizedF
        >>> round(float(GeneralizedF.moment(2, 1.5, 0.6, 0.5, 1.0)), 6)
        42.633576
        """
        delta, s1, s2 = _shapes(Q, P)
        k = m * sigma / delta
        exists = s2 > k
        k_ok = np.where(exists, k, 0.0)
        log_e = (
            m * mu
            + k_ok * (np.log(s2) - np.log(s1))
            + betaln_accurate(s1 + k_ok, s2 - k_ok)
            - betaln_accurate(s1, s2)
        )
        return np.where(exists, np.exp(log_e), np.inf)

    def mean(
        self, mu: Boxable, sigma: Boxable, Q: Boxable, P: Boxable
    ) -> Boxable:
        r"""The mean, the first moment (see :meth:`moment`); infinite where
        the upper tail is too heavy.

        Examples
        --------
        >>> from surpyval import GeneralizedF
        >>> round(float(GeneralizedF.mean(1.5, 0.6, 0.5, 1.0)), 6)
        4.926465
        """
        return self.moment(1, mu, sigma, Q, P)

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return np.log(x)

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mu, sigma, Q, P = (float(getval(p)) for p in params[-4:])
        q = self.qf(onp.asarray(y, dtype=float), 0.0, 1.0, Q, P)
        with onp.errstate(divide="ignore"):
            return onp.log(q)

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        mu, sigma, Q, P = (float(getval(p)) for p in params[-4:])
        return self.ff(onp.exp(onp.asarray(y, dtype=float)), 0.0, 1.0, Q, P)


GeneralizedF: GeneralizedF_ = GeneralizedF_("GeneralizedF")
