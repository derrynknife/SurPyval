import surpyval
import numpy as np
from surpyval import nonparametric as nonp
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

def kaplan_meier(x, c=None, n=None, **kwargs):
	"""
	Kaplan-Meier estimate of survival
	Good explanation of K-M reason can be found at:
	http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis#Kaplan-Meier_Example
	Data given not necessarily in order
	"""
	x, r, d = surpyval.xcnt_to_xrd(x, c, n, **kwargs)
	
	R = np.cumprod(1 - d/r)
	return x, r, d, R

class KaplanMeier_(NonParametricFitter):
	def __init__(self):
		self.how = 'Kaplan-Meier'

KaplanMeier = KaplanMeier_()


