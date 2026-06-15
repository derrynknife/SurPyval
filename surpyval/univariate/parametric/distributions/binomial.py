from scipy.stats import binom

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter

from ..parametric import Parametric


class Binomial_(ParametricFitter):
    r"""
    The Binomial distribution: the number of events (failures) ``k`` in a
    fixed number ``n`` of independent pass/fail trials, each with event
    probability ``p``.

    It is the recurrent (repeated-trials) counterpart of the
    :class:`Bernoulli` distribution, which is the special case ``n = 1``.

    The distribution is parameterised by ``n`` (the number of trials, a
    positive integer) and ``p`` (the per-trial event probability). Because
    ``n`` is an integer structural parameter, the distribution does not use
    the gradient-based MLE machinery; instead ``fit`` uses the closed-form
    maximum likelihood estimate of ``p`` for a known number of trials, in the
    same spirit as :class:`Bernoulli`.
    """

    def __init__(self, name):
        super().__init__(
            name=name,
            k=2,
            bounds=((1, None), (0, 1)),
            support=(0, np.inf),
            param_names=["n", "p"],
            param_map={"n": 0, "p": 1},
            plot_x_scale="linear",
        )
        # A binomial is a discrete count distribution with a fixed number
        # of trials; probability plotting (which assumes a continuous,
        # invertible CDF) does not apply.
        self.supports_mpp = False

    def df(self, x, n, p):
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
        0.3086999999999998
        """
        return binom.pmf(x, n, p)

    def ff(self, x, n, p):
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
        0.83692
        """
        return binom.cdf(x, n, p)

    def sf(self, x, n, p):
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
        0.16308
        """
        return binom.sf(x, n, p)

    def hf(self, x, n, p):
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

    def Hf(self, x, n, p):
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

    def qf(self, q, n, p):
        r"""

        Quantile (inverse CDF) function for the Binomial distribution; the
        smallest number of events ``x`` such that :math:`F(x) \geq q`.

        Parameters
        ----------

        q : numpy array or scalar
            The values, between 0 and 1, at which the quantile is evaluated
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        qf : scalar or numpy array
            The quantile(s) at q

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.qf(0.5, 5, 0.3)
        1.0
        """
        return binom.ppf(q, n, p)

    def cs(self, x, X, n, p):
        r"""

        Conditional survival; the probability of surviving a further ``x``
        events having already survived ``X``:

        .. math::
            R(x \mid X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The further number of events
        X : numpy array or scalar
            The number of events already survived
        n : integer
            The number of trials
        p : float
            The per-trial probability of an event

        Returns
        -------

        cs : scalar or numpy array
            The conditional survival probability
        """
        return self.sf(x + X, n, p) / self.sf(X, n, p)

    def mean(self, n, p):
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

    def moment(self, m, n, p):
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
        1.5
        """
        return binom.moment(m, n, p)

    def entropy(self, n, p):
        r"""

        Entropy of the Binomial distribution (in nats).

        Examples
        --------
        >>> from surpyval import Binomial
        >>> Binomial.entropy(5, 0.3)
        1.413614855283445
        """
        return binom.entropy(n, p)

    def random(self, size, n, p):
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

    def fit(self, x, n_trials, c=None, n=None):
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
        x = np.atleast_1d(np.asarray(x))

        if not np.equal(np.mod(x, 1), 0).all():
            raise ValueError("'x' must contain only integer counts")

        n_trials = int(n_trials)
        if n_trials < 1:
            raise ValueError("'n_trials' must be a positive integer")

        if ((x < 0) | (x > n_trials)).any():
            raise ValueError("'x' must be between 0 and 'n_trials'")

        if c is not None and (np.atleast_1d(np.asarray(c)) != 0).any():
            raise ValueError(
                "Binomial distribution does not support censored data"
            )

        if n is None:
            n = np.ones_like(x)
        n = np.atleast_1d(np.asarray(n))

        model = Parametric(self, "MLE", None, False, False, False)
        p = (x * n).sum() / (n_trials * n.sum())
        model.params = np.array([float(n_trials), p])
        model.support = np.array([0, n_trials])
        return model

    def from_params(self, params):
        r"""

        Create a Binomial model from the parameters ``[n, p]``.

        Parameters
        ----------

        params : array like
            The two parameters ``[n, p]``; ``n`` the (integer) number of
            trials and ``p`` the per-trial event probability.

        Returns
        -------

        model : Parametric
            A parametric model with the provided parameters.

        Examples
        --------
        >>> from surpyval import Binomial
        >>> model = Binomial.from_params([5, 0.3])
        >>> model.mean()
        1.5
        """
        params = np.atleast_1d(np.asarray(params, dtype=float))

        if params.shape[0] != 2:
            raise ValueError("Binomial distribution requires '[n, p]' params")

        n, p = params

        if np.mod(n, 1) != 0:
            raise ValueError("'n' must be an integer number of trials")

        if n < 1:
            raise ValueError("'n' must be a positive integer")

        if not (0 <= p <= 1):
            raise ValueError("'p' must be between 0 and 1")

        model = Parametric(self, "given parameters", None, False, False, False)
        model.params = np.array([float(n), p])
        model.support = np.array([0, n])
        return model


Binomial = Binomial_("Binomial")
