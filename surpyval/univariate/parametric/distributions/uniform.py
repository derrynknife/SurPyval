import numpy as onp
import numpy.typing as npt
from autograd import grad
from scipy.optimize import minimize

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

    def _parameter_initialiser(
        self, data: SurpyvalData, offset: bool = False
    ) -> npt.NDArray:
        x = data.x
        return np.array([np.min(x) - 1.0, np.max(x) + 1.0], dtype=float)

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
        return -np.log(self.sf(x, a, b))

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
        if np.asarray(data.x).ndim == 2 or (data.c == 2).any():
            # The closed-form min/max estimator is not the MLE with
            # interval-censored rows (an interval term favours shrinking
            # the range), and the masks below assume 1-D x -- interval
            # input used to die with a cryptic IndexError (#280).
            raise ValueError(
                "Uniform distribution MLE does not support "
                "interval-censored observations."
            )

        if (data.c[data.x == data.x.max()] == 1).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                " the highest value is right censored"
            )

        if (data.c[data.x == data.x.min()] == -1).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                " the lowest value is left censored"
            )

        tl = data.t[:, 0]
        tr = data.t[:, 1]

        if np.isfinite(tr[data.x == data.x.max()]).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                " the highest value is right truncated"
            )

        if np.isfinite(tl[data.x == data.x.min()]).all():
            raise ValueError(
                "Uniform distribution cannot be estimated using MLE when"
                " the lowest value is left truncated"
            )

        if (data.c != 0).any():
            return self._censored_mle(data)

        # With every observation exact, (min, max) is the MLE. Truncation
        # does not change that: each term 1 / (min(b, tr) - max(a, tl))
        # only improves as the range shrinks onto the data.
        return np.array([np.min(data.x), np.max(data.x)])

    def _censored_mle(self, data: SurpyvalData) -> npt.NDArray | None:
        """The MLE with right- and/or left-censored observations.

        (min, max) is *not* the MLE here, and returning it was wrong: a
        right-censored value ``r`` contributes ``(b - r) / (b - a)``, which
        grows with ``b``, so it pulls ``b`` beyond the largest value --
        three units censored at 9.9 next to failures at 0 and 10 put the
        MLE at ``b = 24.75``, not 10 (neg_ll 7.95 against 18.42). A
        left-censored value pulls ``a`` below the smallest the same way.

        There is no closed form, but nor is the generic optimiser the right
        tool: the likelihood drops to zero the moment ``a`` passes the
        smallest exact (or left-censored) value or ``b`` the largest exact
        (or right-censored) one, and the MLE usually sits on one of those
        walls. The generic path searches an unbounded transform of
        ``(a, b)``, so its gradient methods fail at the wall and it ends on
        Nelder-Mead, which stopped up to 0.5% short. The walls are simple
        bounds, though, so a bounded quasi-Newton search over the region
        the data allow lands on them exactly and converges tightly inside.
        Returns ``None`` (use the generic optimiser) if it does not
        converge.
        """
        x = np.asarray(data.x, dtype=float)
        c = np.asarray(data.c)
        a_max = float(np.min(x[(c == 0) | (c == -1)]))
        b_min = float(np.max(x[(c == 0) | (c == 1)]))
        span = max(b_min - a_max, 1.0)

        def neg_ll(theta: npt.NDArray) -> Boxable:
            return self._neg_ll_func(data, theta[0], theta[1], 0.0, 0.0, 1.0)

        gradient = grad(neg_ll)
        # A censored value tied with the extreme exact one has zero
        # probability on the wall itself, so start just inside.
        x0 = onp.array(
            [
                a_max - (1e-3 * span if (c == -1).any() else 0.0),
                b_min + (1e-3 * span if (c == 1).any() else 0.0),
            ]
        )
        with onp.errstate(all="ignore"):
            res = minimize(
                lambda theta: float(neg_ll(theta)),
                x0,
                jac=lambda theta: onp.asarray(gradient(theta), dtype=float),
                method="L-BFGS-B",
                bounds=[(None, a_max), (b_min, None)],
                options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000},
            )
        if not (res.success and onp.isfinite(res.fun)):
            return None
        return onp.asarray(res.x, dtype=float)

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
