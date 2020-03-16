import numpy as np
import surpyval.nonparametric as nonp

def nelson_aalen(x, c=None, n=None):
	"""
	Nelson-Aalen estimation of Reliability function
    Nelson, W.: Theory and Applications of Hazard Plotting for Censored Failure Data. 
    Technometrics, Vol. 14, #4, 1972
    Technically the NA estimate is for the Cumulative Hazard Function,
    The reliability (survival) curve that is output is also known as the Breslow estimate.
    I will leave it as Nelson-Aalen for this library.

    return_all is called by the fit method to ensure h, x, c, d are all saved

    Hazard Rate
	h = d/r
	Cumulative Hazard Function
	H = cumsum(h)
	Reliability Function
	R = exp(-H)
	"""	
	x, r, d = nonp.get_x_r_d(x, c, n)

	h = d/r
	H = np.cumsum(h)
	R = np.exp(-H)
	return x, r, d, R