import numpy as np
import surpyval.nonparametric as nonp

def kaplan_meier(x, c=None, n=None):
	"""
	Kaplan-Meier estimate of survival
	Good explanation of K-M reason can be found at:
	http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis#Kaplan-Meier_Example
	Data given not necessarily in order
	"""
	x, r, d = nonp.get_x_r_d(x, c, n)
	
	R = np.cumprod(1 - d/r)
	return x, r, d, R