from scipy.stats import uniform

import surpyval
from surpyval import np

from .parametric import Parametric


class Bernoulli_:
    def __init__(self, name):
        super().__init__(name)
        # Set 'k', the number of parameters
        self.k = 1
        self.bounds = ((0, 1),)
        self.support = (0, 1)
        self.param_names = ["p"]
        self.param_map = {
            "p": 0,
        }

    def sf(self, x, p):
        r"""

        Survival (or reliability) function for the Bernoulli Distribution:

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
        >>> from surpyval import Bernoulli
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Bernoulli.sf(x, 0.5)
        array([0.5, 0.5, 0.5, 0.5, 0.5])
        """
        return 1.0 - self.ff(x, p)

    def ff(self, x, p):
        r"""

        Failure (CDF or unreliability) function for the Bernoulli Distribution:

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

        sf : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Bernoulli
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Bernoulli.sf(x, 0.5)
        array([0.5, 0.5, 0.5, 0.5, 0.5])
        """
        return np.ones_like(x).astype(float) * p

    def moment(self, n, p):
        r"""

        n-th moment of the Bernoulli distribution

        .. math::
            M(n) = p

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        p : float
            The probability of failure of the thing

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

    def entropy(self, p):
        return -(1 - p) * np.log(1 - p) - p * np.log(p)

    def random(self, size, p):
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

    def fit(self, x, n=None):
        x, _, n = surpyval.xcn_handler(x=x, c=None, n=n)

        if not np.equal(x, np.array([0, 1])).all():
            raise ValueError("'x' must be either 0 or 1")

        model = Parametric(self, "MLE", {}, False, False, False)
        p = (x * n).sum() / n.sum()
        model.params = [p]
        return model

    def from_params(self, p):
        if type(p) == list:
            p = p[0]

        if p > 1:
            raise ValueError("'p' must be less than 1")

        if p < 0:
            raise ValueError("'p' must be greater than 0")

        model = Parametric(self, "given parameters", {}, False, False, False)
        model.params = [p]
        return model


Bernoulli = Bernoulli_("Bernoulli")
FixedEventProbability = Bernoulli_("FixedEventProbability")
