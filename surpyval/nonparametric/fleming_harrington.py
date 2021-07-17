import numpy as np
import surpyval
from surpyval import nonparametric as nonp
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

def fleming_harrington(x, c=None, n=None, **kwargs):
	x, r, d = surpyval.xcnt_to_xrd(x, c, n, **kwargs)

	h = [np.sum([1./(r[i]-j) for j in range(d[i])]) for i in range(len(x))]
	H = np.cumsum(h)
	R = np.exp(-H)
	return x, r, d, R

class FlemingHarrington_(NonParametricFitter):
	r"""
	Fleming-Harrington estimation of survival distribution.  Returns a `NonParametric` object from method :code:`fit()` Calculates the Non-Parametric estimate of the survival function using:

	.. math:: 

		R = e^{-\sum_{i:x_{i} \leq x} \sum_{i=0}^{d_x-1} \frac{1}{r_x - i}}

	See 'NonParametric section for detailed estimate of how H is computed.'

	Examples
    --------
    >>> import numpy as np
    >>> from surpyval import FlemingHarrington
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = FlemingHarrington.fit(x)
    >>> model.R
    array([0.81873075, 0.63762815, 0.45688054, 0.27711205, 0.10194383])
    """
	def __init__(self):
		self.how = 'Fleming-Harrington'

FlemingHarrington = FlemingHarrington_()