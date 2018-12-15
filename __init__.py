from .nonparametric import plotting_positions

DISTRIBUTIONS = {
	"Weibull" : {
		'x' : 'log',
		'y' : 'weibull'
	},
	"Normal" : {
		'x' : 'linear',
		'y' : 'qnorm'
	},
	"Lognormal" : {
		'x' : 'log',
		'y' : 'qnorm'
	},
	"Uniform" : {
		'x' : 'linear',
		'y' : 'linear'
	},
	"Pareto" : {
		'x' : 'log',
		'y' : 'log'
	}
}
PLOTTING_METHODS = [ "Blom", "Median", "ECDF", "Modal", "Midpoint", 
"Mean", "Weibull", "Benard", "Beard", "Hazen", "Gringorten", 
"None", "Tukey", "DPW", "Fleming-Harrington", "Kaplan-Meier",
"Nelson-Aalen", "Filiben"]