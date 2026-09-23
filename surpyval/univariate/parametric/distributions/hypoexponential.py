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

from typing import Any

import numpy.typing as npt
from scipy import integrate
from scipy.special import factorial, xlogy

from surpyval import np
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


class Hypoexponential_(ParametricFitter):
    r"""

    Class used to generate the Hypoexponential class: the sum of
    independent Exponential stages with distinct rates.

    .. code:: python

        from surpyval import Hypoexponential

        model = Hypoexponential.from_params([0.5, 1.5, 3.0])

    The singleton has no fixed parameter count -- ``from_params`` takes
    any number of rates and the model it returns has that many
    parameters, named ``lambda_1 ... lambda_m``. The distribution
    functions take the rates as the parameters:
    ``Hypoexponential.sf(x, 0.5, 1.5, 3.0)``.
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
    def _terms(x: Numeric, rates: tuple) -> tuple[npt.NDArray, npt.NDArray]:
        """``(C_j, exp(-lambda_j x))`` broadcast to shape ``(..., m)``."""
        r = _validate_rates(rates)
        x_arr = np.asarray(x, dtype=float)
        return _coefficients(r), np.exp(-x_arr[..., None] * r)

    def sf(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Survival (or reliability) function for the Hypoexponential
        Distribution:

        .. math::
            R(x) = \sum_{j=1}^{m} C_j e^{-\lambda_j x}, \qquad
            C_j = \prod_{l \neq j} \frac{\lambda_l}{\lambda_l - \lambda_j}

        Evaluated as ``1 - F(x)`` while ``F(x) < 1/2`` -- ``F`` is exact
        at the origin, so ``R(0) = 1`` exactly -- and as the signed sum
        above beyond that, where it keeps its relative precision in
        the tail; the sum is clipped to ``[0, 1]`` against the
        floating-point cancellation of its terms.

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
        coef, e = self._terms(x, rates)
        direct = np.clip(np.sum(coef * e, axis=-1), 0.0, 1.0)
        ff = self.ff(x, *rates)
        return np.where(ff < 0.5, 1.0 - ff, direct)

    def ff(self, x: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Hypoexponential
        Distribution:

        .. math::
            F(x) = 1 - \sum_{j=1}^{m} C_j e^{-\lambda_j x}
                 = -\sum_{j=1}^{m} C_j \left(e^{-\lambda_j x} - 1\right)

        Evaluated in the second form (``expm1``) so it keeps its
        precision for small ``x``, where ``1 - R(x)`` would cancel.

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
        r = _validate_rates(rates)
        x_arr = np.asarray(x, dtype=float)
        coef = _coefficients(r)
        terms = coef * np.expm1(-x_arr[..., None] * r)
        return np.clip(-np.sum(terms, axis=-1), 0.0, 1.0)

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
        coef, e = self._terms(x, rates)
        r = _validate_rates(rates)
        return np.maximum(np.sum(coef * r * e, axis=-1), 0.0)

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
        return self.df(x, *rates) / self.sf(x, *rates)

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
        return -np.log(self.sf(x, *rates))

    def qf(self, u: Numeric, *rates: Boxable) -> Boxable:
        r"""

        Quantile function for the Hypoexponential distribution, the
        inverse of ``ff``. There is no closed form; the failure function
        is inverted by bisection between the bounds

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
            survival = 1.0 - u_in
            m, r_min = len(r), float(np.min(r))
            lo = -np.log(survival) / r_min
            hi = (m / r_min) * np.log(m / survival)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                too_small = self.sf(mid, *r) > survival
                lo = np.where(too_small, mid, lo)
                hi = np.where(too_small, hi, mid)
                if np.all(hi - lo <= 4.0 * np.finfo(float).eps * hi):
                    break
            out[inside] = 0.5 * (lo + hi)
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
        self, size: int | tuple[int, ...], *rates: Boxable
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
        stages = np.random.exponential(1.0 / r, size=shape + (len(r),))
        return np.sum(stages, axis=-1)


Hypoexponential: Hypoexponential_ = Hypoexponential_("Hypoexponential")
