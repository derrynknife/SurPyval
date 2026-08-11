import numpy.typing as npt
from scipy.stats import binom

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


class Binomial_(DiscreteParametricFitter):
    r"""
    The Binomial distribution: the number of events (failures) ``k`` in a
    fixed number ``n`` of independent pass/fail trials, each with event
    probability ``p``.

    It is the recurrent (repeated-trials) counterpart of the
    :class:`Bernoulli` distribution, which is the special case ``n = 1``.
    The two agree exactly on the probability mass there. Their survival
    functions are offset by one, which is a convention rather than a
    disagreement: this class follows the package's discrete rule
    :math:`R(k) = P(K > k)`, while Bernoulli uses :math:`P(X \geq x)` so
    that ``R(0) = 1`` and ``R(1) = p``. Hence
    ``Bernoulli.sf(x, p) == Binomial.sf(x - 1, 1, p)``.

    The distribution is parameterised by ``n`` (the number of trials, a
    positive integer) and ``p`` (the per-trial event probability). Because
    ``n`` is an integer structural parameter, the distribution does not use
    the gradient-based MLE machinery; instead ``fit`` uses the closed-form
    maximum likelihood estimate of ``p`` for a known number of trials, in the
    same spirit as :class:`Bernoulli`.
    """

    def __init__(self, name: str) -> None:
        super().__init__(
            name=name,
            k=2,
            bounds=((1, None), (0, 1)),
            # ``support`` is a pair of *exclusive* bounds: the shared
            # ``_validate_fit_inputs`` rejects data with
            # ``x <= support[0]`` or ``x >= support[1]``, so a distribution
            # declares the bound one step outside its first and last mass
            # points. The first mass point here is k = 0 -- zero events in
            # n trials is an ordinary outcome, P = 0.168 at n = 5, p = 0.3
            # -- so the lower bound is -1, as for ``Poisson``. It read 0,
            # which is ``Geometric``'s value and says zero events lie
            # outside the distribution. Nothing observed it because
            # ``Binomial`` does not inherit ``OptimisedFitMixin``, where
            # that check lives, and validates its own inputs instead.
            #
            # The upper bound stays infinite here because n is not known
            # until the model is built; ``fit`` and ``from_params`` set it
            # to n + 1 for the same reason.
            support=(-1, np.inf),
            param_names=["n", "p"],
            param_map={"n": 0, "p": 1},
            plot_x_scale="linear",
        )

    def df(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Probability mass function for the Binomial distribution:

        .. math::
            P(X = x) = \binom{n}{x} p^{x} (1 - p)^{n - x}

        Parameters
        ----------

        x : numpy array or scalar
            The number of events at which the mass function is evaluated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the mass function at x

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.df(2, 5, 0.3)
        np.float64(0.3086999999999998)
        """
        return binom.pmf(x, n, p)

    def ff(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Failure (CDF) function for the Binomial distribution:

        .. math::
            F(x) = P(X \leq x) = \sum_{i=0}^{\lfloor x \rfloor}
            \binom{n}{i} p^{i} (1 - p)^{n - i}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.ff(2, 5, 0.3)
        np.float64(0.83692)
        """
        return binom.cdf(x, n, p)

    def sf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Survival (reliability) function for the Binomial distribution:

        .. math::
            R(x) = P(X > x) = 1 - F(x)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at x

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.sf(2, 5, 0.3)
        np.float64(0.16308)
        """
        return binom.sf(x, n, p)

    def hf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Discrete hazard rate for the Binomial distribution; the conditional
        probability of exactly ``x`` events given at least ``x``:

        .. math::
            h(x) = \frac{P(X = x)}{P(X \geq x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the discrete hazard rate at x
        """
        d = self.df(x, n, p)
        # P(X >= x) = P(X > x) + P(X = x)
        denom = self.sf(x, n, p) + d
        return np.where(denom > 0, d / denom, 0.0)

    def Hf(self, x: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Cumulative hazard function for the Binomial distribution:

        .. math::
            H(x) = -\ln R(x)

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard function at x
        """
        return -np.log(self.sf(x, n, p))

    def qf(self, u: Numeric, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Quantile (inverse CDF) function for the Binomial distribution; the
        smallest number of events ``x`` such that :math:`F(x) \geq u`.

        Parameters
        ----------

        u : numpy array or scalar
            The values, between 0 and 1, at which the quantile is evaluated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        qf : scalar or numpy array
            The quantile(s) at u

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.qf(0.5, 5, 0.3)
        np.float64(1.0)
        """
        return binom.ppf(u, n, p)

    def mean(self, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Mean of the Binomial distribution:

        .. math::
            E[X] = n p

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.mean(5, 0.3)
        1.5
        """
        return n * p

    def moment(self, m: int, n: Boxable, p: Boxable) -> Boxable:
        r"""

        m-th (raw) moment of the Binomial distribution.

        Parameters
        ----------

        m : integer
            The ordinal of the moment to calculate
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        moment : scalar
            The m-th raw moment of the Binomial distribution

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.moment(1, 5, 0.3)
        np.float64(1.5)
        """
        return binom.moment(m, n, p)

    def entropy(self, n: Boxable, p: Boxable) -> Boxable:
        r"""

        Entropy of the Binomial distribution (in nats).

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.entropy(5, 0.3)
        np.float64(1.413614855283445)
        """
        return binom.entropy(n, p)

    def random(
        self, size: int | tuple[int, ...], n: Boxable, p: Boxable
    ) -> npt.NDArray:
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        random : scalar or numpy array
            Random values drawn from the distribution in shape `size`
        """
        return binom.rvs(n, p, size=size)

    def fit(
        self,
        x: npt.ArrayLike,
        n_trials: int,
        c: npt.NDArray | None = None,
        n: npt.NDArray | None = None,
    ) -> Parametric:
        r"""

        Fit the Binomial distribution for a known number of trials,
        ``n_trials``, using the closed-form maximum likelihood estimate of
        the per-trial event probability ``p``.

        Parameters
        ----------

        x : array like
            The observed number of events for each experiment. Every value
            must be an integer in ``[0, n_trials]``.
        n_trials : integer
            The (known) number of trials in each experiment.
        c : array like, optional
            Censoring flags. Censoring is not supported for the Binomial
            distribution and any non-zero flag raises a ``ValueError``.
        n : array like, optional
            The count (multiplicity) of each observation in ``x``. If
            ``None`` each observation is assumed to have occurred once.

        Returns
        -------

        model : Parametric
            A parametric model with the fitted ``[n_trials, p]`` parameters.

        Examples
        --------
        >>> from surpyval import Binomial
        >>> model = Binomial.fit([2, 3, 1, 4], n_trials=5)
        >>> model.params
        array([5. , 0.5])
        """
        x_arr = np.atleast_1d(np.asarray(x))

        if not np.equal(np.mod(x_arr, 1), 0).all():
            raise ValueError("'x' must contain only integer counts")

        n_trials = int(n_trials)
        if n_trials < 1:
            raise ValueError("'n_trials' must be a positive integer")

        if ((x_arr < 0) | (x_arr > n_trials)).any():
            raise ValueError("'x' must be between 0 and 'n_trials'")

        if c is not None and (np.atleast_1d(np.asarray(c)) != 0).any():
            raise ValueError(
                "Binomial distribution does not support censored data"
            )

        if n is None:
            n = np.ones_like(x_arr)
        n = np.atleast_1d(np.asarray(n))

        model = Parametric(self, "MLE", None, False, False, False)
        p = (x_arr * n).sum() / (n_trials * n.sum())
        model.params = np.array([float(n_trials), p])
        # Exclusive bounds either side of the outcomes {0, ..., n_trials};
        # see the note in __init__.
        model.support = np.array([-1, n_trials + 1])
        return model

    # Narrower than ParametricFitter.from_params, which takes
    # (params, gamma, p, f0). Unlike `fit`, this one is not resolved
    # by the OptimisedFitMixin split: every distribution has a
    # from_params. It is a parameter *rename* -- the base's `params`
    # became `params` -- so positional calls work and keyword calls
    # raise. Fixing it means renaming
    # back, with a deprecation alias, and is tracked separately.
    def from_params(
        self,
        params: npt.ArrayLike,
        gamma: Boxable | None = None,
        p: Boxable | None = None,
        f0: Boxable | None = None,
    ) -> Parametric:
        r"""

        Create a Binomial model from the parameters ``[n, p]``.

        Parameters
        ----------

        params : array like
            The two parameters ``[n, p]``; ``n`` the (integer) number of
            trials and ``p`` the per-trial event probability.
        gamma, p, f0 : None
            Accepted so the signature matches
            :meth:`ParametricFitter.from_params`, and rejected: a
            Binomial has no offset, limited failure population or zero
            inflation. The base's ``p`` is the *never-fails* proportion,
            not the per-trial probability, which lives in ``params``.

        Returns
        -------

        model : Parametric
            A parametric model with the provided parameters.

        Examples
        --------
        >>> from surpyval import Binomial
        >>> model = Binomial.from_params([5, 0.3])
        >>> model.mean()
        np.float64(1.5)
        """
        reject_structural_params(self.name, gamma, p, f0)
        params_arr = np.atleast_1d(np.asarray(params, dtype=float))

        if params_arr.shape[0] != 2:
            raise ValueError("Binomial distribution requires '[n, p]' params")

        n, prob = params_arr

        if np.mod(n, 1) != 0:
            raise ValueError("'n' must be an integer number of trials")

        if n < 1:
            raise ValueError("'n' must be a positive integer")

        if not (0 <= prob <= 1):
            raise ValueError("'p' must be between 0 and 1")

        model = Parametric(self, "given parameters", None, False, False, False)
        model.params = np.array([float(n), prob])
        # Exclusive bounds either side of the outcomes {0, ..., n}; see the
        # note in __init__.
        model.support = np.array([-1, n + 1])
        return model


Binomial = Binomial_("Binomial")
