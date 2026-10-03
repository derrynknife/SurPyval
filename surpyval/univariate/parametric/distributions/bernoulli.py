import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)

from ._single_probability import SingleProbabilityMixin


class Bernoulli_(  # type: ignore[misc]
    SingleProbabilityMixin, DiscreteParametricFitter
):
    r"""A single weighted coin flip: ``X`` is 0 or 1 with ``P(X = 1) = p``.

    ``x`` is the outcome, not a time, so 0 and 1 are the only values any
    of these functions accept and anything else raises. The survival
    function follows the package's discrete convention,
    :math:`R(x) = P(X > x)`, giving ``R(0) = p`` and ``R(1) = 0``: read as
    a one-shot device, ``p`` is the probability it works when demanded
    (``df(1)``, or the parameter itself), and ``R(0)`` the probability it
    survives that demand.

    .. note::
       ``p`` is the probability of the ``1`` outcome. Before 0.20.0 this
       distribution had ``F(x) = p`` at every ``x`` -- a flat curve with
       no time axis, where ``p`` was documented as the probability of
       *failure*. The parameter has therefore changed direction: code
       that coded failures as 1 now fits the survival probability, and
       wants ``1 - p``. The flat model itself is unchanged and still
       available as :data:`FixedEventProbability`.

    .. versionchanged:: 0.22
       ``R(x)`` was :math:`P(X \geq x)` (``R(0) = 1``, ``R(1) = p``),
       with ``F(x) = P(X < x)``, which never reached 1, so ``qf`` could
       not invert ``ff`` (#344). Every function now equals ``Binomial``
       with ``n = 1`` and ``scipy.stats.bernoulli``: code that read the
       probability of the ``1`` outcome as ``sf(1)`` wants ``sf(0)``, or
       ``p``.

    Examples
    --------
    Ten demands on a one-shot device, which worked (1) on eight:

    >>> from surpyval import Bernoulli
    >>> model = Bernoulli.fit([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
    >>> model.params
    array([0.8])
    >>> model.sf([0, 1])
    array([0.8, 0. ])

    ``param_cb("p")`` bounds ``p`` from the counts, by default with the
    exact (Clopper-Pearson) interval. Three failures (coded 1 here) in
    1200 demands, and a 90% interval on the failure probability:

    >>> demands = Bernoulli.fit([1, 0], n=[3, 1197])
    >>> demands.param_cb("p", alpha_ci=0.1).round(5)
    array([0.00068, 0.00645])

    With no failures at all the upper bound is the whole answer, and equals
    ``1 - alpha_ci ** (1 / n)``, the complement of ``success_run``:

    >>> clean = Bernoulli.fit([0], n=[1200])
    >>> clean.param_cb("p", alpha_ci=0.1, bound="upper").round(6)
    array([0.001917])
    """

    @staticmethod
    def _check_x(x: Numeric) -> npt.NDArray:
        """Reject anything that is not a Bernoulli outcome."""
        x_arr = np.asarray(x, dtype=float)
        if not np.isin(x_arr, (0.0, 1.0)).all():
            raise ValueError(
                "Bernoulli is defined at x = 0 and x = 1 only; x is the "
                "outcome of the flip, not a time. For a model whose event "
                "probability is p at every x, use FixedEventProbability."
            )
        return x_arr

    @staticmethod
    def _shaped(out: Boxable) -> Boxable:
        """``out`` with the shape of ``x``: a scalar for a scalar ``x``."""
        return np.asarray(out)[()] if np.ndim(out) == 0 else out

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=1,
            bounds=((0, 1),),
            support=(0, 1),
            parameter_names=["p"],
            param_map={"p": 0},
            plot_x_scale="linear",
        )

    def sf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Survival function for the Bernoulli Distribution:

        .. math::
            R(x) = P(X > x)

        which is ``p`` at ``x = 0`` and 0 at ``x = 1``.

        Parameters
        ----------

        x : numpy array or scalar
            The outcome(s) at which the function will be calculated.
            Must be 0 or 1.
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> Bernoulli.sf(np.array([0, 1]), 0.3)
        array([0.3, 0. ])
        """
        x_arr = self._check_x(x)
        return self._shaped(np.where(x_arr == 0.0, p, 0.0))

    def ff(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Failure (CDF) function for the Bernoulli Distribution:

        .. math::
            F(x) = P(X \leq x)

        which is ``1 - p`` at ``x = 0`` and 1 at ``x = 1``.

        Parameters
        ----------

        x : numpy array or scalar
            The outcome(s) at which the function will be calculated.
            Must be 0 or 1.
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> Bernoulli.ff(np.array([0, 1]), 0.3)
        array([0.7, 1. ])
        """
        return 1.0 - self.sf(x, p)

    def df(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Probability mass function for the Bernoulli Distribution:

        .. math::
            f(0) = 1 - p, \quad f(1) = p

        Parameters
        ----------

        x : numpy array or scalar
            The outcome(s) at which the function will be calculated.
            Must be 0 or 1.
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        df : scalar or numpy array
            The probability of the outcome(s) in x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> Bernoulli.df(np.array([0, 1]), 0.3)
        array([0.7, 0.3])
        """
        x_arr = self._check_x(x)
        return self._shaped(np.where(x_arr == 0.0, 1.0 - p, p))

    def hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Hazard rate for the Bernoulli Distribution:

        .. math::
            h(x) = \frac{f(x)}{P(X \geq x)} = \frac{f(x)}{R(x - 1)}

        which is ``1 - p`` at ``x = 0`` and 1 at ``x = 1``: everything
        still at risk at the last outcome fails there.

        Parameters
        ----------

        x : numpy array or scalar
            The outcome(s) at which the function will be calculated.
            Must be 0 or 1.
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> Bernoulli.hf(np.array([0, 1]), 0.3)
        array([0.7, 1. ])
        """
        x_arr = self._check_x(x)
        return self._shaped(np.where(x_arr == 0.0, 1.0 - p, 1.0))

    def Hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Bernoulli Distribution:

        .. math::
            H(x) = -\ln R(x)

        which is :math:`-\ln p` at ``x = 0`` and infinite at ``x = 1``,
        where nothing survives.

        Parameters
        ----------

        x : numpy array or scalar
            The outcome(s) at which the function will be calculated.
            Must be 0 or 1.
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> Bernoulli.Hf(np.array([0, 1]), 0.3)
        array([1.2039728,       inf])
        """
        x_arr = self._check_x(x)
        return self._shaped(np.where(x_arr == 0.0, -np.log(p), np.inf))

    def qf(self, u: Numeric, p: Boxable) -> Boxable:
        r"""

        Quantile function for the Bernoulli Distribution:

        .. math::
            q(u) = \begin{cases}
                0 & u \leq 1 - p \\
                1 & u > 1 - p
            \end{cases}

        This inverts ``ff``, :math:`P(X \leq x)`: the smallest outcome
        with ``ff(x) >= u``, so ``qf(U)`` for uniform ``U`` is 1 with
        probability ``p``. On the open interval it agrees exactly with
        ``Binomial.qf(u, 1, p)`` and with ``scipy.stats.binom.ppf``. At
        ``u = 0`` those return ``-1``, one below the support, where this
        returns 0 -- the smallest outcome there is.

        Parameters
        ----------

        u : numpy array or scalar
            The probability or probabilities at which the quantile will
            be calculated
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        qf : scalar or numpy array
            The outcome(s) at the given probabilities.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> Bernoulli.qf(np.array([0.1, 0.7, 0.75, 0.99]), 0.3)
        array([0., 0., 1., 1.])
        """
        u_arr = np.asarray(u, dtype=float)
        return np.where(u_arr <= 1.0 - p, 0.0, 1.0)

    def log_df(self, x: Numeric, p: Boxable) -> Boxable:
        # The log of the pmf directly: exact at p near 0 or 1, where
        # log(h(k) R(k - 1)) would lose digits.
        x_arr = self._check_x(x)
        return self._shaped(np.where(x_arr == 0.0, np.log1p(-p), np.log(p)))

    def mean(self, p: Boxable) -> Boxable:
        r"""

        Mean of the Bernoulli distribution:

        .. math::
            E = p

        Parameters
        ----------

        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        mean : scalar or numpy array
            The mean of the Bernoulli distribution

        Examples
        --------
        >>> from surpyval import Bernoulli
        >>> Bernoulli.mean(0.3)
        0.3
        """
        return p

    def moment(self, m: int, p: Boxable) -> Boxable:
        r"""

        m-th moment of the Bernoulli distribution

        .. math::
            E[X^{m}] = p

        The same for every ``m``, because ``X`` is 0 or 1 and so
        ``X**m == X``.

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        mean : scalar or numpy array
            The moment(s) of the Bernoulli distribution

        Examples
        --------
        >>> from surpyval import Bernoulli
        >>> Bernoulli.moment(2, 0.5)
        0.5
        """
        return p


Bernoulli: Bernoulli_ = Bernoulli_("Bernoulli")
