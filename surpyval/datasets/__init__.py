import pandas as pd
from pkg_resources import resource_filename

class BoforsSteel_():
	r"""
	A Class with a Pandas DataFrame containing the data of the tensile strenght of Bofors Steel from the Weibull paper [1]_.

	Attributes
	----------
	df : DataFrame
		A Pandas DataFrame containing the data of the tensile strenght of Bofors Steel from the Weibull paper.

	Examples
	--------
	>>> from surpyval import BoforsSteel
	>>> df = BoforsSteel.df
	>>> df.head(5)

	.. list-table::
		:header-rows: 1

		* - 
		  - x
		  - n
		* - 0
		  - 40.800
		  - 10
		* - 1
		  - 42.075
		  - 23
		* - 2
		  - 43.350
		  - 48
		* - 3
		  - 44.625
		  - 80
		* - 4
		  - 45.900
		  - 63

	References
	----------
	.. [1] Weibull, W., A statistical distribution function of wide applicability, Journal of applied mechanics, Vol. 18, No. 3, pp 293-297 (1951).

	"""
	def __init__(self):
		#self.df = pd.read_csv('surpyval/datasets/bofors_steel.csv')
		self.df = pd.read_csv(resource_filename("surpyval", "datasets/bofors_steel.csv"), engine="python")

class Bearing_():
	def __init__(self):
		x = [17.88, 28.92, 33, 41.52, 42.12, 45.6, 48.4, 51.84, 
			51.96, 54.12, 55.56, 67.8, 68.64, 68.64, 68.88, 84.12, 
			93.12, 98.64, 105.12, 105.84, 127.92, 128.04, 173.4]
		self.df = pd.DataFrame({'Cycles to Failure (millions)' : x})

	def __str__(self):
		return """
		Data from:
		Lieblein, J. and Zelen, M. (1956) Statistical Investigation of the Fatigue Life of Deep-Groove Ball Bearings. Journal of Research of the National Bureau of Standards, 57, 273-315. 
		http://dx.doi.org/10.6028/jres.057.033
		"""

BoforsSteel = BoforsSteel_()
Bearing = Bearing_()