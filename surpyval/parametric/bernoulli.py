from surpyval import np
from scipy.stats import uniform
from scipy.special import ndtri as z

from surpyval import parametric as para
from .parametric import Parametric

from .fitters.mpp import mpp
import surpyval

class Bernoulli_():
    def __init__(self, name):
        self.name = name
        # Set 'k', the number of parameters
        self.k = 1
        self.bounds = ((0, 1),)
        self.support = (0, 1)
        self.param_names = ['p']
        self.param_map = {
            'p' : 0,
        }

    def sf(self, x, p):
        r"""

        Survival (or reliability) function for the Weibull Distribution:

        .. math::
            R(x) = e^{-\left ( \frac{x}{\alpha} \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        sf : scalar or numpy array 
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.sf(x, 3, 4)
        array([9.87730216e-01, 8.20754808e-01, 3.67879441e-01, 4.24047953e-02,
               4.45617596e-04])
        """
        return 1. - self.ff(x, p)

    def ff(self, x, p):
        r"""

        Failure (CDF or unreliability) function for the Weibull Distribution:

        .. math::
            F(x) = 1 - e^{-\left ( \frac{x}{\alpha} \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated 
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        sf : scalar or numpy array 
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> Weibull.ff(x, 3, 4)
        array([0.01226978, 0.17924519, 0.63212056, 0.9575952 , 0.99955438])
        """
        return np.ones_like(x).astype(float) *  p

    def moment(self, n, p):
        r"""

        n-th moment of the Weibull distribution

        .. math::
            M(n) = \alpha^n \Gamma \left ( 1 + \frac{n}{\beta} \right )

        Parameters
        ----------

        n : integer or numpy array of integers
            The ordinal of the moment to calculate
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        mean : scalar or numpy array 
            The moment(s) of the Weibull distribution

        Examples
        --------
        >>> from surpyval import Weibull
        >>> Weibull.moment(2, 3, 4)
        7.976042329074821
        """
        return p

    def entropy(self, p):
        return -(1 - p)*np.log(1 - p) - p * np.log(p)

    def random(self, size, p):
        r"""

        Draws random samples from the distribution in shape `size`

        Parameters
        ----------

        size : integer or tuple of positive integers
            Shape or size of the random draw
        alpha : numpy array or scalar
            scale parameter for the Weibull distribution
        beta : numpy array or scalar
            shape parameter for the Weibull distribution

        Returns
        -------

        random : scalar or numpy array 
            Random values drawn from the distribution in shape `size`

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull
        >>> Weibull.random(10, 3, 4)
        array([1.79782451, 1.7143211 , 2.84778674, 3.12226231, 2.61000839,
               3.05456332, 3.00280851, 2.61910071, 1.37991527, 4.17488394])
        >>> Weibull.random((5, 5), 3, 4)
        array([[1.64782514, 2.79157632, 1.85500681, 2.91908736, 2.46089933],
               [1.85880127, 0.96787742, 2.29677031, 2.42394129, 2.63889601],
               [2.14351859, 3.90677225, 2.24013855, 2.49467774, 3.43755278],
               [3.24417396, 1.40775181, 2.49584969, 3.07603353, 2.54679499],
               [1.98330076, 2.95002633, 3.35402601, 3.11429283, 3.45706789]])
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
        if p > 1:
            raise ValueError("'p' must be less than 1")

        if p < 0:
            raise ValueError("'p' must be greater than 0")

        model = Parametric(self, "from_params", {}, False, False, False)
        model.params = [p]
        return model


Bernoulli = Bernoulli_('Bernoulli')
FixedEventProbability = Bernoulli_('Bernoulli')