"""The fixed-event-probability model.

``F(x) = p`` at every ``x``: a fraction ``p`` of units fail and the rest
never do, with nothing said about *when*. It is the two-point mixture of
:class:`InstantlyOccurs` (weight ``p``) and :class:`NeverOccurs` (weight
``1 - p``), which is why ``degenerate.py`` describes those two as this
model's limits at ``p = 1`` and ``p = 0``.

This was exported as ``Bernoulli`` as well until 0.19.1, when
``Bernoulli`` became a true Bernoulli -- a coin flip over ``{0, 1}``
whose survival steps at the outcome. The two are different models and
now different classes; this one is unchanged.

``df``, ``hf``, ``qf`` and ``mean`` are absent by construction: ``F`` is
constant, so there is no density, no invertible quantile, and no time to
average.
"""

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


class FixedEventProbability_(DiscreteParametricFitter):
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

        Survival (or reliability) function for the
        FixedEventProbability model:

        .. math::
            R(x) = 1 - p

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        p : float
            The probability of failure of the thing

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x. Which for this
            distribution is constant

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import FixedEventProbability
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> FixedEventProbability.sf(x, 0.5)
        array([0.5, 0.5, 0.5, 0.5, 0.5])
        """
        return 1.0 - self.ff(x, p)

    def ff(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Failure (CDF or unreliability) function for the
        FixedEventProbability model:

        .. math::
            F(x) = p

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        p : float
            The probability of failure of the thing

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import FixedEventProbability
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> FixedEventProbability.ff(x, 0.5)
        array([0.5, 0.5, 0.5, 0.5, 0.5])
        """
        return np.ones_like(x).astype(float) * p

    def moment(self, m: int, p: Boxable) -> Boxable:
        r"""

        m-th moment of the FixedEventProbability model

        .. math::
            M(m) = p

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        p : float
            The probability of failure of the thing

        Returns
        -------

        mean : scalar or numpy array
            The moment(s) of the FixedEventProbability model

        Examples
        --------
        >>> from surpyval import FixedEventProbability
        >>> FixedEventProbability.moment(2, 0.5)
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
            The probability of failure of the thing

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`

        """
        U = uniform.rvs(size=size)
        return (U <= p).astype(int)

    def fit(
        self, x: npt.ArrayLike, n: npt.NDArray | None = None
    ) -> Parametric:
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
        params: npt.ArrayLike,
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


FixedEventProbability: FixedEventProbability_ = FixedEventProbability_(
    "FixedEventProbability"
)
