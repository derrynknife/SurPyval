"""The fixed-event-probability model.

``F(x) = p`` at every ``x``: a fraction ``p`` of units fail and the rest
never do, with nothing said about *when*. It is the two-point mixture of
:class:`InstantlyOccurs` (weight ``p``) and :class:`NeverOccurs` (weight
``1 - p``), which is why ``degenerate.py`` describes those two as this
model's limits at ``p = 1`` and ``p = 0``.

This was exported as ``Bernoulli`` as well until 0.20.0, when
``Bernoulli`` became a true Bernoulli -- a coin flip over ``{0, 1}``
whose survival steps at the outcome. The two are different models and
now different classes; this one is unchanged.

``df`` and ``hf`` are absent by construction: ``F`` is constant, so
there is no density. ``qf`` is the two-point mixture's: the smallest time
with ``F(x) >= u``, 0 for ``u <= p`` and infinite above it. There is no
failure *time* to average either, so ``moment`` and ``mean`` are those of
the 0/1 event indicator: ``p`` for every order.
"""

import autograd.numpy as np

from surpyval.univariate.parametric.discrete_fitter import (
    DiscreteParametricFitter,
)
from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)

from ._single_probability import SingleProbabilityMixin


class FixedEventProbability_(  # type: ignore[misc]
    SingleProbabilityMixin, DiscreteParametricFitter
):
    """``F(x) = p`` at every ``x``: a fraction ``p`` of units fail and the
    rest never do, with nothing said about *when* (see the module
    docstring). It is fitted from 0/1 event indicators, 1 for a unit that
    failed.

    Examples
    --------
    Two of ten units failed:

    >>> from surpyval import FixedEventProbability
    >>> model = FixedEventProbability.fit([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    >>> model.params
    array([0.2])
    >>> model.ff([10, 100])
    array([0.2, 0.2])
    """

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

    def Hf(self, x: Numeric, p: Boxable) -> Boxable:
        r"""

        Cumulative hazard function for the FixedEventProbability model:

        .. math::
            H(x) = -\ln R(x) = -\ln (1 - p)

        Constant in ``x``, like the survival it comes from. There is no
        hazard *rate* -- ``hf`` is absent because ``F`` is flat, so the
        mass is an atom rather than a density -- but the cumulative
        hazard is still well defined, exactly as for
        :class:`ExactEventTime`, whose ``Hf`` exists while its ``hf``
        does not.

        Without it ``log_sf`` and ``log_ff``, which the base class writes
        in terms of ``Hf``, raised ``AttributeError`` rather than
        returning the constants they should.

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        p : float
            The probability of failure of the thing

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard function at x

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import FixedEventProbability
        >>> x = np.array([1, 2, 3])
        >>> FixedEventProbability.Hf(x, 0.5)
        array([0.69314718, 0.69314718, 0.69314718])
        """
        return -np.log(self.sf(x, p))

    def qf(self, u: Numeric, p: Boxable) -> Boxable:
        r"""

        Quantile function for the FixedEventProbability model:

        .. math::
            q(u) = \begin{cases}
                0 & u \leq p \\
                \infty & u > p
            \end{cases}

        The smallest ``x`` with :math:`F(x) \geq u`: the model is the
        mixture of :class:`InstantlyOccurs` (weight ``p``, all its mass at
        0) and :class:`NeverOccurs` (weight ``1 - p``, at infinity), so a
        fraction up to ``p`` has failed from the start and no more ever
        does. A probability outside [0, 1] gives NaN, with one warning, as
        every model's ``qf`` (#611).

        Parameters
        ----------

        u : numpy array or scalar
            The probability or probabilities at which the quantile will
            be calculated
        p : float
            The probability of failure of the thing

        Returns
        -------

        qf : scalar or numpy array
            The quantile(s) at the given probabilities: 0 or ``inf``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import FixedEventProbability
        >>> FixedEventProbability.qf(np.array([0.0, 0.1, 0.2, 0.5]), 0.2)
        array([ 0.,  0.,  0., inf])
        """
        u_arr = np.asarray(u, dtype=float)
        return np.where(u_arr <= p, 0.0, np.inf)

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

    def mean(self, p: Boxable) -> Boxable:
        r"""

        Mean of the event indicator, :math:`E = p` -- the first
        :meth:`moment` (``mean`` used to be missing while ``moment``
        existed, so a model's ``mean()`` raised AttributeError).

        Examples
        --------
        >>> from surpyval import FixedEventProbability
        >>> FixedEventProbability.mean(0.3)
        0.3
        """
        return self.moment(1, p)


FixedEventProbability: FixedEventProbability_ = FixedEventProbability_(
    "FixedEventProbability"
)
