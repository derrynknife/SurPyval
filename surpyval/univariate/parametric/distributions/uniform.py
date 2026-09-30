from typing import Any

import numpy.typing as npt

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    OptimisedFitMixin,
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData


class Uniform_(OptimisedFitMixin, ParametricFitter):
    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((None, None), (None, None)),
            # The support of a uniform is its fitted [a, b] interval, so it
            # is data-dependent (undefined until the model is set), not the
            # whole real line. Declare it NaN and let support_param_index
            # (default (0, 1) == a, b) resolve it once the params are known.
            support=(np.nan, np.nan),
            param_names=["a", "b"],
            param_map={"a": 0, "b": 1},
            plot_x_scale="linear",
            y_ticks=np.linspace(0, 1, 21)[1:-1],
        )

    def _check_params(self, params: Any) -> None:
        # Each parameter is unbounded on its own, so from_params used to
        # accept a > b -- a model whose sf was 0 everywhere.
        if not params[0] < params[1]:
            raise ValueError(
                f"{self.name} needs a < b; got a = {params[0]}, "
                f"b = {params[1]}"
            )

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        x = np.asarray(data.x, dtype=float)
        x = x[np.isfinite(x)]
        lo, hi = float(np.min(x)), float(np.max(x))
        # The start must lie strictly outside the data (the likelihood is
        # zero otherwise), and the margin must scale with the data. It was
        # a fixed 1.0: on data in thousandths the start was a thousand
        # times wider than the sample, and the search never recovered from
        # it. (max - min) / (n - 1) is the complete-sample MPS estimate
        # of the margin, so the start is usually close to the answer too.
        spread = hi - lo
        if spread > 0:
            pad = spread / max(x.size - 1, 1)
        else:
            pad = abs(hi) if hi != 0 else 1.0
        return np.array([lo - pad, hi + pad], dtype=float)

    def sf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Survival (or Reliability) function for the Uniform Distribution:

        .. math::
            R(x) = \frac{b - x}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.sf(x, 0, 6)
        array([0.83333333, 0.66666667, 0.5       , 0.33333333, 0.16666667])
        """
        return 1 - self.ff(x, a, b)

    def ff(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Uniform Distribution:

        .. math::
            F(x) = \frac{x - a}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.ff(x, 0, 6)
        array([0.16666667, 0.33333333, 0.5       , 0.66666667, 0.83333333])
        """
        f = np.zeros_like(x)
        f = np.where(x < a, 0, f)
        f = np.where(x > b, 1, f)
        f = np.where(((x <= b) & (x >= a)), (x - a) / (b - a), f)
        return f

    def df(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the Uniform Distribution:

        .. math::
            f(x) = \frac{1}{b - a}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.df(x, 0, 6)
        array([0.16666667, 0.16666667, 0.16666667, 0.16666667, 0.16666667])
        """
        d = np.zeros_like(x)
        d = np.where(x < a, 0, d)
        d = np.where(x > b, 0, d)
        d = np.where(((x <= b) & (x >= a)), 1.0 / (b - a), d)
        return d

    def hf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Uniform Distribution:

        .. math::
            h(x) = \frac{1}{b - x}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.hf(x, 0, 6)
        array([0.2       , 0.25      , 0.33333333, 0.5       , 1.        ])
        """
        # inf at x = b, where the survival function is 0: the true limit
        with np.errstate(divide="ignore"):
            return self.df(x, a, b) / self.sf(x, a, b)

    def log_df(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""Log density, :math:`-\ln(b - a)` on the support.

        Defined directly rather than through the generic
        :math:`\ln h(x) - H(x)` identity, which is ``nan`` at the upper
        support edge: there ``sf`` is 0, so the identity evaluates
        ``log(inf) - inf``. The MLE puts ``b`` exactly at the largest
        observation, so that edge is always hit and the whole
        log-likelihood came out ``nan`` -- taking ``neg_ll``, ``aic``,
        ``bic`` and ``aic_c`` with it.
        """
        x = np.asarray(x, dtype=float)
        inside = (x >= a) & (x <= b)
        return np.where(inside, -np.log(b - a), -np.inf)

    def Hf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Instantaneous hazard rate for the Uniform Distribution:

        .. math::
            H(x) = \ln \left ( b - a \right ) - \ln \left ( b - x \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Uniform.Hf(x, 0, 6)
        array([0.18232156, 0.40546511, 0.69314718, 1.09861229, 1.79175947])
        """
        return 0.0 - self.log_sf(x, a, b)

    @staticmethod
    def _log_ff_sf(x: Numeric, a: Boxable, b: Boxable) -> tuple:
        r"""
        :math:`\ln F` and :math:`\ln R`, each from the distance to the
        nearer edge of the support: :math:`\ln(x - a) - \ln(b - a)` and
        ``log1p`` of minus its ratio, and the mirror image above the
        middle. ``-log(1 - F)`` rounded to -0.0, and ``log F`` to -inf,
        near the lower edge (#442, #443).
        """
        x = np.asarray(x, dtype=float)
        # full-shaped: autograd's ``where`` does not unbroadcast its
        # gradient
        width = b - a + np.zeros_like(x)
        inside = (x > a) & (x < b)
        # the points outside (a, b) are evaluated at stand-ins inside it
        x_in = np.where(inside, x, a + 0.5 * width)
        lower = x_in - a <= b - x_in
        x_lo = np.where(lower, x_in, a + 0.25 * width)
        x_hi = np.where(lower, a + 0.75 * width, x_in)
        log_width = np.log(width)
        log_ff = np.where(
            lower,
            np.log(x_lo - a) - log_width,
            np.log1p(-(b - x_hi) / width),
        )
        log_sf = np.where(
            lower,
            np.log1p(-(x_lo - a) / width),
            np.log(b - x_hi) - log_width,
        )
        nan = np.isnan(x)
        log_ff = np.where(
            inside,
            log_ff,
            np.where(nan, np.nan, np.where(x <= a, -np.inf, 0.0)),
        )
        log_sf = np.where(
            inside,
            log_sf,
            np.where(nan, np.nan, np.where(x <= a, 0.0, -np.inf)),
        )
        return log_ff[()], log_sf[()]

    def log_ff(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        return self._log_ff_sf(x, a, b)[0]

    def log_sf(self, x: Numeric, a: Boxable, b: Boxable) -> Boxable:
        return self._log_ff_sf(x, a, b)[1]

    def qf(self, u: Numeric, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Quantile function for the Uniform Distribution:

        .. math::
            q(u) = a + u(b - a)

        Parameters
        ----------

        u : numpy array or scalar
            The percentiles at which the quantile will be calculated
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        q : scalar or numpy array
            The quantiles for the Uniform distribution at each value u.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Uniform
        >>> u = np.array([.1, .2, .3, .4, .5])
        >>> Uniform.qf(u, 0, 6)
        array([0.6, 1.2, 1.8, 2.4, 3. ])
        """
        return a + u * (b - a)

    def mean(self, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Mean of the Uniform distribution

        .. math::
            E = \frac{1}{2} \left ( a + b \right )

        Parameters
        ----------

        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        mean : scalar or numpy array
            The mean(s) of the Uniform distribution

        Examples
        --------
        >>> from surpyval import Uniform
        >>> Uniform.mean(0, 6)
        3.0
        """
        return 0.5 * (a + b)

    def moment(self, m: int, a: Boxable, b: Boxable) -> Boxable:
        r"""

        m-th (non central) moment of the Uniform distribution

        .. math::
            M(m) = \frac{1}{m +1} \sum_{i=0}^{m}a^ib^{m-i}

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        moment : scalar or numpy array
            The moment(s) of the Uniform distribution

        Examples
        --------
        >>> from surpyval import Uniform
        >>> Uniform.moment(2, 0, 6)
        np.float64(12.0)
        """
        if m == 0:
            return 1
        else:
            out = np.zeros(m + 1)
            for i in range(m + 1):
                out[i] = a**i * b ** (m - i)
            return np.sum(out) / (m + 1)

    def entropy(self, a: Boxable, b: Boxable) -> Boxable:
        r"""

        Calculates the entropy of the Uniform distribution.

        .. math::
            S = \ln \left ( b - a \right )

        Parameters
        ----------

        a : numpy array or scalar
            The lower parameter for the Uniform distribution
        b : numpy array or scalar
            The upper parameter for the Uniform distribution

        Returns
        -------

        entropy : scalar or numpy array
            The entropy(ies) of the Uniform distribution

        Examples
        --------
        >>> from surpyval import Uniform
        >>> Uniform.entropy(0, 6)
        np.float64(1.791759469228055)
        """
        return np.log(b - a)

    def _closed_form_mle(self, data: SurpyvalData) -> npt.NDArray | None:
        # Only exactly observed values: the MLE is then (min, max), and
        # truncation does not change that (each term
        # 1 / (min(b, tr) - max(a, tl)) only improves as the range shrinks
        # onto the data). With censored values the MLE still exists but
        # sits on a wall of the likelihood (the smallest or largest value),
        # where the Hessian says nothing about its uncertainty: a censored
        # fit carried a covariance that was not positive definite, and its
        # Wald bounds were NaN or silently several times too wide (#460).
        # So censored data are refused; MPS, MPP and MSE take them.
        if np.asarray(data.x).ndim == 2 or (data.c != 0).any():
            raise ValueError(
                "Uniform distribution MLE does not support censored "
                "observations: its estimates sit on the edge of the data, "
                "where the likelihood gives no usable measure of their "
                "uncertainty. Fit with how='MPS' (or 'MPP' or 'MSE') "
                "instead."
            )
        return np.array([np.min(data.x), np.max(data.x)])

    def _closed_form_optimizer(self, data: SurpyvalData) -> str:
        """How ``_closed_form_mle`` solved this data, for ``optimizer``."""
        return "closed-form"

    def mpp_x_transform(self, x: npt.NDArray) -> Boxable:
        return x

    def mpp_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return y

    def mpp_inv_y_transform(self, y: npt.NDArray, *params: Boxable) -> Boxable:
        return y

    def unpack_rr(
        self, params: npt.NDArray, rr: str
    ) -> tuple[Boxable, Boxable]:
        if rr == "y":
            a = -params[1] / params[0]
            b = (1 - params[1]) / params[0]
        if rr == "x":
            a = params[1]
            b = params[0] + params[1]

        return a, b

    def _mom(self, x: npt.NDArray) -> tuple[float, float]:
        mu_1 = np.mean(x)
        mu_2 = np.mean(x**2)

        d = np.sqrt(3 * (mu_2 - mu_1**2))
        a = mu_1 - d
        b = mu_1 + d
        return a, b

    def _plot_x_bounds(
        self, x: npt.NDArray, params: npt.NDArray
    ) -> tuple[float, float] | None:
        return float(np.min(params)), float(np.max(params))


Uniform: Uniform_ = Uniform_("Uniform")
