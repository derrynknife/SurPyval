import pandas as pd

class BoforsSteel_():
	def __init__(self):
		self.df = pd.read_csv('surpyval/datasets/bofors_steel.csv')

BoforsSteel = BoforsSteel_()