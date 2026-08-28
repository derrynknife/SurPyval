"""
The machinery shared by the two one-parameter probability models.

``Bernoulli`` and ``FixedEventProbability`` are different models -- a
coin flip over ``{0, 1}`` against a flat ``F(x) = p`` -- but their
*estimation* is the same problem: one probability ``p`` in ``(0, 1)``,
fitted from 0/1 observations by a weighted mean, with no offset,
limited-failure or zero-inflation structure. When the classes were split
in 0.19.1 that machinery was copied into both files verbatim; this mixin
is the single copy. Everything distributional -- ``sf``, ``ff``, the
supports, the docstrings that state each model's own convention -- stays
on the classes themselves.
"""

import numpy.typing as npt
from scipy.stats import uniform

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    reject_structural_params,
)

from ..parametric import Parametric


class SingleProbabilityMixin:
    """``fit``/``from_params``/``entropy``/``random`` for a model whose
    single parameter is an event probability and whose data are 0/1."""

    # Provided by the host class's ParametricFitter initialisation.
    name: str

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
        """Create a model from its event probability.

        Parameters
        ----------
        params : scalar
            The event probability, between 0 and 1.
        gamma, p, f0 : None
            Accepted so the signature matches
            :meth:`ParametricFitter.from_params`, and rejected: neither
            model has an offset, limited failure population or zero
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
