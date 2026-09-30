"""The hypoexponential (generalised Erlang) distribution.

The lifetime of a unit that passes through ``m`` independent
exponential *stages* in series -- with rates ``lambda_1, ...,
lambda_m`` -- is the sum of the stage durations, and that sum is
hypoexponential. With every rate equal it is the Erlang (a Gamma with
integer shape); with distinct rates the survival function is a signed
mixture of exponentials found by partial fractions,

.. math::
    R(x) = \\sum_{j=1}^{m} C_j e^{-\\lambda_j x}, \\qquad
    C_j = \\prod_{l \\neq j} \\frac{\\lambda_l}{\\lambda_l - \\lambda_j},

with :math:`\\sum_j C_j = 1` so that :math:`R(0) = 1`. Load-sharing
groups (the group rate changes as members fail), warm and hot standby
and any other "k stages, each memoryless" model produce this
distribution.

The number of stages is not fixed in advance, so the distribution has no
fixed parameter count: ``Hypoexponential.from_params([r1, r2, ...])``
accepts any number of rates and the fitted model carries that many. The
distribution functions can be called directly on the singleton with the
rates as the parameters, ``Hypoexponential.sf(x, r1, r2, ...)``, exactly
as for every other distribution. There is no ``fit``: build the model
from known stage rates.

Only *distinct* rates are supported. The partial-fraction coefficients
grow without bound (with alternating signs) as two rates approach each
other, so rates closer than a small fraction of the largest rate are
refused with a clear error rather than returning a survival function
poisoned by cancellation; equal rates are the Erlang / Gamma case.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
from scipy import integrate
from scipy.special import factorial, gammaln, xlogy

from surpyval import np
from surpyval.univariate.parametric.parametric import draw_state
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    ParametricFitter,
)

from ..parametric import Parametric

#: Two rates closer than this fraction of the largest rate are treated as
#: equal, and refused: the partial-fraction coefficients scale like
#: ``rate / gap`` with alternating signs, so at this separation the
#: survival function has lost about six of its sixteen digits.
DISTINCT_RATES_TOL = 1e-6

# the smallest positive double, the floor of the quantile's bracket
_SMALLEST = float(np.nextafter(0.0, 1.0))


def _validate_rates(rates: Any) -> npt.NDArray:
    """The stage rates as a 1-D float array: finite, positive, distinct."""
    r = np.atleast_1d(np.asarray(rates, dtype=float))
    if r.ndim != 1 or r.size == 0:
        raise ValueError(
            "Hypoexponential needs a one-dimensional, non-empty vector of "
            "stage rates; got shape {}".format(r.shape)
        )
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError(
            "Hypoexponential stage rates must all be finite and strictly "
            "positive; got {}".format(r.tolist())
        )
    if r.size > 1:
        ordered = np.sort(r)
        gap = float(np.min(np.diff(ordered)))
        if gap <= DISTINCT_RATES_TOL * float(ordered[-1]):
            raise ValueError(
                "Hypoexponential stage rates must be distinct (separated "
                "by more than {:g} of the largest rate); got {}. The "
                "partial-fraction form is ill-conditioned for near-equal "
                "rates. For equal rates the sum of exponentials is the "
                "Erlang distribution: use Gamma with an integer shape.".format(
                    DISTINCT_RATES_TOL, r.tolist()
                )
            )
    return r


def _coefficients(rates: npt.NDArray) -> npt.NDArray:
    """The partial-fraction coefficients ``C_j`` of the survival function,
    ``prod_{l != j} r_l / (r_l - r_j)``; they sum to one."""
    m = len(rates)
    index = np.arange(m)
    return np.array(
        [
            np.prod(rates[index != j] / (rates[index != j] - rates[j]))
            for j in range(m)
        ]
    )


#: Where ``lambda_max * x`` is at most this, the CDF and density are
#: summed as their power series (``_series``) rather than from the partial
#: fractions, which cancel there: two stages at ``x = 1e-15`` have
#: ``F = 1e-30`` as a difference of numbers of size ``1e-15``.
_SERIES_MAX = 1.0
#: Terms of that series; the n-th is below ``1 / n!`` of the first.
_SERIES_TERMS = 40


def _series(x: npt.NDArray, rates: npt.NDArray) -> tuple:
    r"""
    :math:`\ln F(x)` and :math:`\ln f(x)` from the power series

    .. math::
        F(x) = \prod_j (\lambda_j x) \sum_{n \ge 0}
               \frac{(-1)^n h_n(\lambda x)}{(n + m)!}, \qquad
        f(x) = \frac{\prod_j (\lambda_j x)}{x} \sum_{n \ge 0}
               \frac{(-1)^n h_n(\lambda x)}{(n + m - 1)!},

    with :math:`h_n` the complete homogeneous symmetric polynomial of
    degree :math:`n` (the Taylor series of the divided difference of
    :math:`e^{-\lambda x}` over the rates). For :math:`\lambda_j x \le
    1` the terms fall in size and alternate in sign, so neither sum
    cancels, and the logs stay finite where :math:`F` and :math:`f`
    underflow (#442, #443). ``x`` must be positive.
    """
    m = len(rates)
    mu = x[..., None] * rates
    h = np.zeros(x.shape + (_SERIES_TERMS,))
    h[..., 0] = 1.0
    for j in range(m):
        for n in range(1, _SERIES_TERMS):
            h[..., n] += mu[..., j] * h[..., n - 1]
    n = np.arange(_SERIES_TERMS)
    signed = np.where(n % 2 == 0, 1.0, -1.0) * h
    sum_ff = np.sum(signed * np.exp(-gammaln(n + m + 1.0)), axis=-1)
    sum_df = np.sum(signed * np.exp(-gammaln(n + m)), axis=-1)
    log_prod = np.sum(np.log(rates)) + m * np.log(x)
    return log_prod + np.log(sum_ff), log_prod - np.log(x) + np.log(sum_df)


class Hypoexponential_(ParametricFitter):
    r"""

    Class used to generate the Hypoexponential class: the sum of
    independent Exponential stages with distinct rates.

    The singleton has no fixed parameter count -- ``from_params`` takes
    any number of rates and the model it returns has that many
    parameters, named ``lambda_1 ... lambda_m``. The distribution
    functions take the rates as the parameters:
    ``Hypoexponential.sf(x, 0.5, 1.5, 3.0)``. It is not fitted from data
    (``fit`` raises); build it from known stage rates.

    Examples
    --------
    Two stages in series, with mean times 2 and 2/3:

    >>> from surpyval import Hypoexponential
    >>> model = Hypoexponential.from_params([0.5, 1.5])
    >>> model.sf([1, 2, 4]).round(4)
    array([0.7982, 0.5269, 0.2018])
    >>> round(float(model.mean()), 4)
    2.6667
    """

    def __init__(self, name: str, m: int = 0) -> None:
        # ``m == 0`` is the exported singleton, whose arity is only fixed
        # when ``from_params`` is called; that call (and deserialisation)
        # builds an ``m``-stage instance with concrete parameter names
        # and bounds, so the resulting model reports the right ``k``.
        super().__init__(
            name=name,
            k=m,
            bounds=((0, None),) * m,
            support=(0, np.inf),
            param_names=["lambda_{}".format(j + 1) for j in range(m)],
            param_map={"lambda_{}".format(j + 1): j for j in range(m)},
            plot_x_scale="linear",
        )
        # No probability-plotting transform: the survival function is
        # a sum of exponentials with no single linearising scale.
        self.supports_mpp = False

    def _for_params(self, params: Any) -> "Hypoexponential_":
        """The ``m``-stage fitter that models ``params``."""
        m = len(np.atleast_1d(np.asarray(params, dtype=float)))
        if m == self.k:
            return self
        return Hypoexponential_(self.name, m=m)

    def from_params(
        self,
        params: npt.ArrayLike,
        gamma: Boxable | None = None,
        p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        r"""

        Create a hypoexponential model from its stage rates.

        Parameters
        ----------

        params : array like
            The stage rates ``lambda_1, ..., lambda_m``, all strictly
            positive and distinct. Any number of them.
        gamma : scalar, optional
            An offset (shift) of the distribution.
        p : scalar, optional
            The proportion of the population that ever fails
            (limited-failure population).
        f0 : scalar, optional
            The proportion of the population that fails at time zero
            (zero inflation).

        Returns
        -------

        Parametric
            A parametric model with ``len(params)`` parameters.

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> model = Hypoexponential.from_params([0.5, 1.5, 3.0])
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Hypoexponential
        Fitted by           : given parameters
        Parameters          :
          lambda_1: 0.5
          lambda_2: 1.5
          lambda_3: 3.0
        >>> model.mean()
        np.float64(3.0)
        """
        rates = _validate_rates(params)
        return ParametricFitter.from_params(
            self._for_params(rates), rates, gamma, p, f0
        )

    def fit(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
    ) -> Parametric:
        """Not available: build the model from its stage rates instead.

        The hypoexponential is the lifetime of a chain of memoryless
        stages, and its use in surpyval is to describe such a chain
        whose stage rates are already known (a load-sharing group, a
        standby system). Estimating the number of stages and their
        rates from lifetimes alone is a phase-type fitting problem this
        release does not attempt; use :meth:`from_params`.
        """
        raise NotImplementedError(
            "Hypoexponential is not fitted from data; construct it from "
            "its stage rates with Hypoexponential.from_params([r1, r2, ...])"
        )

    @staticmethod
    def _pieces(x: Numeric, rates: tuple) -> dict[str, npt.NDArray]:
        """
        Every function at ``x``, each from the form that is exact where
        it is used:

        - near the origin (``lambda_max x <= 1``) the power series of
          ``_series`` for ``F`` and ``f``;
        - elsewhere ``F`` from the partial fractions in ``expm1`` form,
          and, with the slowest rate factored out,
          ``R = e^(-lambda_min x) sum_j C_j e^(-(lambda_j - lambda_min) x)``
          and ``f`` likewise, whose logs stay finite after ``R`` and
          ``f`` underflow (#443);
        - the survival function as ``1 - F`` (and its log as
          ``log1p(-F)``) while ``F < 1/2``, and the hazard as ``f / R``
          from the scaled sums beyond, which is not 0 / 0 where both
          underflow (#444).
        """
        r = _validate_rates(rates)
        x_arr = np.asarray(x, dtype=float)
        x_flat = x_arr.ravel()
        m, r_min, r_max = len(r), float(np.min(r)), float(np.max(r))
        # the support edges; a NaN stays NaN
        at_zero_df = float(r[0]) if m == 1 else 0.0
        edges = {
            "ff": (0.0, 1.0),
            "sf": (1.0, 0.0),
            "log_ff": (-np.inf, 0.0),
            "log_sf": (0.0, -np.inf),
            "log_df": (np.log(at_zero_df) if m == 1 else -np.inf, -np.inf),
            "df": (at_zero_df, 0.0),
            "hf": (at_zero_df, r_min),
        }
        out = {}
        for name, (at_zero, at_inf) in edges.items():
            value = np.full(x_flat.shape, np.nan)
            value[x_flat == 0] = at_zero
            value[x_flat == np.inf] = at_inf
            out[name] = value
        finite = (x_flat > 0) & (x_flat < np.inf)
        series = finite & (x_flat * r_max <= _SERIES_MAX)
        direct = finite & ~series
        if np.any(series):
            log_ff, log_df = _series(x_flat[series], r)
            ff = np.exp(log_ff)
            # F is at most 1 - 1/e here (one stage), so 1 - F does not
            # cancel
            log_sf = np.log1p(-ff)
            parts = {
                "ff": ff,
                "sf": 1.0 - ff,
                "log_ff": log_ff,
                "log_sf": log_sf,
                "log_df": log_df,
                "df": np.exp(log_df),
                "hf": np.exp(log_df - log_sf),
            }
            for name, value in parts.items():
                out[name][series] = value
        if np.any(direct):
            x_d = x_flat[direct]
            coef = _coefficients(r)
            e = np.exp(-x_d[:, None] * (r - r_min))
            scaled_sf = np.sum(coef * e, axis=-1)
            scaled_df = np.sum(coef * r * e, axis=-1)
            ff = np.clip(
                -np.sum(coef * np.expm1(-x_d[:, None] * r), axis=-1), 0.0, 1.0
            )
            low = ff < 0.5
            with np.errstate(divide="ignore", invalid="ignore"):
                # the sums are clipped at 0 against the cancellation of
                # their terms (near-equal rates), and their logs are
                # then -inf
                log_sf_d = -r_min * x_d + np.log(np.maximum(scaled_sf, 0.0))
                log_df = -r_min * x_d + np.log(np.maximum(scaled_df, 0.0))
                log_sf = np.where(
                    low, np.log1p(-np.minimum(ff, 0.5)), log_sf_d
                )
                log_ff = np.where(
                    low,
                    np.log(ff),
                    np.log1p(-np.exp(np.minimum(log_sf_d, np.log(0.5)))),
                )
                hf = np.where(
                    low, np.exp(log_df - log_sf), scaled_df / scaled_sf
                )
            parts = {
                "ff": ff,
                "sf": np.where(
                    low, 1.0 - ff, np.minimum(np.exp(log_sf_d), 1.0)
                ),
                "log_ff": log_ff,
                "log_sf": log_sf,
                "log_df": log_df,
                "df": np.exp(log_df),
                "hf": hf,
            }
            for name, value in parts.items():
                out[name][direct] = value
        return {
            name: value.reshape(x_arr.shape)[()] for name, value in out.items()
        }

    def sf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the Hypoexponential
        Distribution:

        .. math::
            R(x) = \sum_{j=1}^{m} C_j e^{-\lambda_j x}, \qquad
            C_j = \prod_{l \neq j} \frac{\lambda_l}{\lambda_l - \lambda_j}

        Evaluated as ``1 - F(x)`` while ``F(x) < 1/2`` -- ``F`` is exact
        at the origin, so ``R(0) = 1`` exactly -- and beyond that as the
        signed sum above with the slowest stage's exponential factored
        out, which keeps its relative precision in the tail; the sum is
        clipped at 0 against the floating-point cancellation of its terms
        (see ``_pieces``).

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.sf(x, 0.5, 1.5, 3.0)
        array([0.87858244, 0.61289168, 0.39054997, 0.24112599, 0.14719997])
        """
        return self._pieces(x, rates)["sf"]

    def ff(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Hypoexponential
        Distribution:

        .. math::
            F(x) = 1 - \sum_{j=1}^{m} C_j e^{-\lambda_j x}
                 = -\sum_{j=1}^{m} C_j \left(e^{-\lambda_j x} - 1\right)

        Near the origin (``lambda_max x <= 1``), where the partial
        fractions cancel, it is summed as its power series, which keeps
        its relative precision down to underflow; beyond that it is
        evaluated in the second form (``expm1``).

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.ff(x, 0.5, 1.5, 3.0)
        array([0.12141756, 0.38710832, 0.60945003, 0.75887401, 0.85280003])
        """
        return self._pieces(x, rates)["ff"]

    def df(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Density function for the Hypoexponential Distribution:

        .. math::
            f(x) = \sum_{j=1}^{m} C_j \lambda_j e^{-\lambda_j x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.df(x, 0.5, 1.5, 3.0)
        array([0.24105459, 0.25789815, 0.1842277 , 0.11808731, 0.07304706])
        """
        return self._pieces(x, rates)["df"]

    def hf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Hypoexponential Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        It rises from ``0`` at the origin (two or more stages must all
        complete) towards the smallest stage rate, which dominates the
        tail.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.hf(x, 0.5, 1.5, 3.0)
        array([0.27436764, 0.42078911, 0.4717135 , 0.48973284, 0.49624367])
        """
        return self._pieces(x, rates)["hf"]

    def Hf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Hypoexponential Distribution:

        .. math::
            H(x) = -\ln R(x)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Hypoexponential.Hf(x, 0.5, 1.5, 3.0)
        array([0.12944553, 0.48956707, 0.94019934, 1.42243572, 1.91596325])
        """
        return 0.0 - self._pieces(x, rates)["log_sf"]

    def log_sf(self, x: Numeric, *rates: Boxable) -> Boxable:
        """Log of the survival function (see ``_pieces``)."""
        return self._pieces(x, rates)["log_sf"]

    def log_ff(self, x: Numeric, *rates: Boxable) -> Boxable:
        """Log of the CDF (see ``_pieces``)."""
        return self._pieces(x, rates)["log_ff"]

    def log_df(self, x: Numeric, *rates: Boxable) -> Boxable:
        """Log of the density (see ``_pieces``)."""
        return self._pieces(x, rates)["log_df"]

    def qf(self, u: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Quantile function for the Hypoexponential distribution, the
        inverse of ``ff``. There is no closed form; the failure function
        is inverted by bisection -- on ``log F`` against ``log u`` below
        ``u = 1/2`` and on ``log R`` against ``log(1 - u)`` above it, so a
        tiny probability keeps its digits -- between the bounds

        .. math::
            \frac{-\ln(1 - u)}{\lambda_{\min}} \le q(u) \le
            \frac{m}{\lambda_{\min}} \ln \frac{m}{1 - u},

        the left from the slowest stage alone (the sum exceeds it) and
        the right from a union bound over the stages. ``u = 0`` gives
        ``0`` and ``u = 1`` gives ``inf``.

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Hypoexponential distribution at each
            value u

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Hypoexponential.qf(u, 0.5, 1.5, 3.0)
        array([0.90854377, 1.3054212 , 1.67232635, 2.05027738, 2.46566683])
        """
        r = _validate_rates(rates)
        u_arr = np.asarray(u, dtype=float)
        scalar = u_arr.ndim == 0
        u_flat = np.atleast_1d(u_arr)
        out = np.full(u_flat.shape, np.nan)
        out[u_flat == 0.0] = 0.0
        out[u_flat == 1.0] = np.inf
        inside = (u_flat > 0.0) & (u_flat < 1.0)
        if inside.any():
            u_in = u_flat[inside]
            # u enters as log u below 1/2 and as log(1 - u) above it, and
            # is compared with log F or log R, so a small u keeps its
            # digits: 1 - u rounds to 1 below 1e-16, and the quantile
            # came out at the bisection's floor of about 1e-60 (#447).
            low = u_in <= 0.5
            log_u = np.log(u_in)
            log_1mu = np.log1p(-u_in)
            m, r_min = len(r), float(np.min(r))
            lo = np.maximum(-log_1mu / r_min, _SMALLEST)
            hi = (m / r_min) * (np.log(m) - log_1mu)
            for _ in range(400):
                # a geometric mid-point until the bracket is within a
                # factor of 2, so a quantile of 1e-150 takes a few dozen
                # steps, not a few hundred
                wide = hi > 2.0 * lo
                mid = np.where(
                    wide,
                    np.exp(0.5 * (np.log(lo) + np.log(hi))),
                    lo + 0.5 * (hi - lo),
                )
                p = self._pieces(mid, tuple(r))
                too_small = np.where(
                    low, p["log_ff"] < log_u, p["log_sf"] > log_1mu
                )
                lo = np.where(too_small, mid, lo)
                hi = np.where(too_small, hi, mid)
                if np.all(hi - lo <= 4.0 * np.finfo(float).eps * hi):
                    break
            out[inside] = lo + 0.5 * (hi - lo)
        return out[0] if scalar else out

    def mean(self, *rates: Boxable) -> Boxable:
        r"""

        Mean of the Hypoexponential distribution: the stage means add.

        .. math::
            E = \sum_{j=1}^{m} \frac{1}{\lambda_j}

        Parameters
        ----------

        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        mean : scalar
            The mean of the Hypoexponential distribution

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> Hypoexponential.mean(0.5, 1.5, 3.0)
        np.float64(3.0)
        """
        return np.sum(1.0 / _validate_rates(rates))

    def moment(self, m: int, *rates: Boxable) -> Boxable:
        r"""

        m-th raw moment of the Hypoexponential distribution. The density
        is a signed mixture of exponentials, so its moments are the
        mixture of theirs:

        .. math::
            M(m) = m! \sum_{j=1}^{m} \frac{C_j}{\lambda_j^{m}}

        (so ``moment(1)`` is ``mean`` and the variance
        ``moment(2) - moment(1)**2`` is
        :math:`\sum_j \lambda_j^{-2}`, the stage variances adding).

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        moment : scalar
            The moment of the Hypoexponential distribution

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> Hypoexponential.moment(2, 0.5, 1.5, 3.0)
        np.float64(13.555555555555554)
        """
        r = _validate_rates(rates)
        return factorial(m) * np.sum(_coefficients(r) / r**m)

    def entropy(self, *rates: Boxable) -> Boxable:
        r"""

        Entropy of the Hypoexponential distribution. There is no closed
        form, so it is computed by numerical integration of

        .. math::
            S = -\int_{0}^{\infty} f(x) \ln f(x) \, dx

        Parameters
        ----------

        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``

        Returns
        -------

        entropy : scalar
            The entropy of the Hypoexponential distribution

        Examples
        --------
        >>> from surpyval import Hypoexponential
        >>> Hypoexponential.entropy(0.5, 1.5, 3.0)
        1.9455443600389024
        """
        r = _validate_rates(rates)

        def func(x: float) -> float:
            f = self.df(x, *r)
            return float(xlogy(f, f))

        return -integrate.quad(func, 0, np.inf)[0]

    def random(
        self,
        size: int | tuple[int, ...],
        *rates: Boxable,
        random_state: Any = None,
    ) -> npt.NDArray:
        r"""

        Draws random samples from the Hypoexponential distribution: the
        sum of one Exponential draw per stage, which is exact for any
        rates. (A fitted model's ``random`` goes through ``qf`` instead,
        like every other distribution's.)

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        rates : numpy array or scalars
            The stage rates ``lambda_1, ..., lambda_m``
        random_state : int or numpy.random.Generator, optional
            Seed or generator for a draw of its own; ``None`` (the
            default) draws from numpy's global stream (see
            :meth:`ParametricFitter.random`).

        Returns
        -------

        random : numpy array
            Random values drawn from the distribution in shape ``size``

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Hypoexponential
        >>> np.random.seed(1)
        >>> Hypoexponential.random(5, 0.5, 1.5, 3.0)
        array([1.92866664, 0.85812653, 0.86336444, 2.29543903, 1.86983688])
        """
        r = _validate_rates(rates)
        shape = (size,) if isinstance(size, int) else tuple(size)
        state = draw_state(random_state)
        source = np.random if state is None else state
        stages = source.exponential(1.0 / r, size=shape + (len(r),))
        return np.sum(stages, axis=-1)


Hypoexponential: Hypoexponential_ = Hypoexponential_("Hypoexponential")
