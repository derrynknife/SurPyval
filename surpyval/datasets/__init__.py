import pandas as pd
from pkg_resources import resource_filename

class BoforsSteel_():
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