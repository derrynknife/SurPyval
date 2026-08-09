import numpy.typing as npt
from scipy.stats import uniform

from surpyval import np
from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
    reject_structural_params,
)

from ..parametric import Parametric


class Bernoulli_(DiscreteParametricFitter):
    r"""A single weighted coin flip: ``X`` is 0 or 1 with ``P(X = 1) = p``.

    ``x`` is the outcome, not a time, so 0 and 1 are the only values any
    of these functions accept and anything else raises. The survival
    function is :math:`R(x) = P(X \geq x)`, giving ``R(0) = 1`` and
    ``R(1) = p``: read as a one-shot device, ``p`` is the probability it
    works when demanded.

    .. note::
       ``p`` is the probability of the ``1`` outcome. Before 0.19.1 this
       distribution had ``F(x) = p`` at every ``x`` -- a flat curve with
       no time axis, where ``p`` was documented as the probability of
       *failure*. The parameter has therefore changed direction: code
       that coded failures as 1 now fits the survival probability, and
       wants ``1 - p``. The flat model itself is unchanged and still
       available as :data:`FixedEventProbability`.

    Note also that this is not ``Binomial`` with ``n = 1`` evaluated at
    the same points. Binomial follows the package's discrete convention
    :math:`R(k) = P(X > k)`; this one uses :math:`P(X \geq x)`, so the
    two are offset by one: ``Bernoulli.sf(x, p) == Binomial.sf(x - 1,
    1, p)``.
    """

    @staticmethod
    def _check_x(x: Numeric) -> npt.NDArray:
        """Reject anything that is not a Bernoulli outcome."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        if not np.isin(x_arr, (0.0, 1.0)).all():
            raise ValueError(
                "Bernoulli is defined at x = 0 and x = 1 only; x is the "
                "outcome of the flip, not a time. For a model whose event "
                "probability is p at every x, use FixedEventProbability."
            )
        return x_arr

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=1,
            bounds=((0, 1),),
            support=(0, 1),
            param_names=["p"],
            param_map={"p": 0},
            plot_x_scale="linear",
        )

    def sf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Survival function for the Bernoulli Distribution:

        .. math::
            R(x) = P(X \geq x)

        which is 1 at ``x = 0`` and ``p`` at ``x = 1``.

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
        array([1. , 0.3])
        """
        x_arr = self._check_x(x)
        return np.where(x_arr == 0.0, 1.0, p)

    def ff(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Failure (CDF) function for the Bernoulli Distribution:

        .. math::
            F(x) = P(X < x)

        which is 0 at ``x = 0`` and ``1 - p`` at ``x = 1``. This is
        ``P(X < x)`` rather than the more usual ``P(X \leq x)`` because
        the package's survival and failure functions sum to one, and
        ``R(x)`` here is ``P(X \geq x)``.

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
        array([0. , 0.7])
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
        return np.where(x_arr == 0.0, 1.0 - p, p)

    def hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Hazard rate for the Bernoulli Distribution:

        .. math::
            h(x) = \frac{f(x)}{P(X \geq x)}

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
        return np.where(x_arr == 0.0, 1.0 - p, 1.0)

    def Hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Cumulative hazard rate for the Bernoulli Distribution:

        .. math::
            H(x) = -\ln R(x)

        which is 0 at ``x = 0`` and :math:`-\ln p` at ``x = 1``.

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
        array([0.       , 1.2039728])
        """
        x_arr = self._check_x(x)
        return np.where(x_arr == 0.0, 0.0, -np.log(p))

    def log_df(self, x: Numeric, p: Boxable) -> Boxable:
        # Neither inherited relation fits. DiscreteParametricFitter uses
        # f(k) = h(k) R(k - 1), which assumes R(k) = P(X > k); here R is
        # P(X >= x), so the at-risk set at x is R(x) itself and the
        # mass is f(x) = h(x) R(x) -- the continuous form. Taking the
        # log of the pmf directly sidesteps the choice.
        x_arr = self._check_x(x)
        return np.where(x_arr == 0.0, np.log1p(-p), np.log(p))

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

    def entropy(self, p: Boxable) -> Boxable:
        return -(1 - p) * np.log1p(-p) - p * np.log(p)

    def random(self, size: int | tuple[int, ...], p: Boxable) -> npt.NDArray:
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        p : float
            The probability of the ``1`` outcome

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        """
        U = uniform.rvs(size=size)
        return (U <= p).astype(int)

    def fit(self, x: Numeric, n: npt.NDArray | None = None) -> Parametric:
        x_arr = np.atleast_1d(x)
        # Each observation must be a 0 or a 1 — elementwise, for any length
        # (the previous check broadcast x against the literal [0, 1], so any
        # input of length != 2 crashed and [1, 1] was rejected, #257).
        if not np.isin(x_arr, (0, 1)).all():
            raise ValueError("'x' must be either 0 or 1")
        n_arr = np.ones_like(x_arr) if n is None else np.atleast_1d(n)
        if n_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("'n' must be the same length as 'x'")

        model = Parametric(self, "MLE", None, False, False, False)
        p = (x_arr * n_arr).sum() / n_arr.sum()
        model.params = np.array([p])
        return model

    # Narrower than ParametricFitter.from_params, which takes
    # (params, gamma, p, f0). Unlike `fit`, this one is not resolved
    # by the OptimisedFitMixin split: every distribution has a
    # from_params. It is a parameter *rename* -- the base's `params`
    # became `p` -- so positional calls work and keyword calls
    # raise. Worse here: the base's `p` means the
    # limited-failure proportion, so the same keyword means two
    # unrelated things across sibling classes. Fixing it means renaming
    # back, with a deprecation alias, and is tracked separately.
    def from_params(
        self,
        params: Boxable,
        gamma: Boxable | None = None,
        p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        """Create a Bernoulli model from its event probability.

        Parameters
        ----------
        params : scalar
            The event probability, between 0 and 1.
        gamma, p, f0 : None
            Accepted so the signature matches
            :meth:`ParametricFitter.from_params`, and rejected: a
            Bernoulli has no offset, limited failure population or zero
            inflation. Note that the base's ``p`` is the *never-fails*
            proportion, not this distribution's parameter -- which is why
            the parameter is ``params`` and not ``p``.
        """
        reject_structural_params(self.name, gamma, p, f0)
        prob = float(np.squeeze(np.asarray(params)))

        if prob > 1:
            raise ValueError("'params' must be less than 1")

        if prob < 0:
            raise ValueError("'params' must be greater than 0")

        model = Parametric(self, "given parameters", None, False, False, False)
        model.params = np.atleast_1d(prob)
        return model


Bernoulli: Bernoulli_ = Bernoulli_("Bernoulli")
