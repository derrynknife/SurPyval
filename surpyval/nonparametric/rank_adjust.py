import numpy as np

def rank_adjust(t, c=None):
	"""
	Currently limited to only Mean Order Number
	Room to expand to:
	Modal Order Number, and
	Median Order Number
	Uses mean order statistic to conduct rank adjustment
	For further reading see:
	http://reliawiki.org/index.php/Parameter_Estimation
	Above reference provides excellent explanation of how this method is derived
	This function currently assumes good input - Use if you know how
	15 Mar 2015
	"""
	# Total items in test/population
	N = len(t)
	# Preallocate adjusted ranks array
	ranks = np.zeros(N)

	if c is None:
	    c = np.zeros(N)

	# Rank adjustment for [right] censored data
	# PMON - "Previous Mean Order Number"
	# NIPBSS - "Number of Items Before Present Suspended Set"
	PMON = 0
	for i in range(0, N):
	    if c[i] == 0:
	        NIBPSS = N - i
	        ranks[i] = PMON + (N + 1 - PMON)/(1 + NIBPSS)
	        PMON = ranks[i]
	    elif c[i] == 1:
	        ranks[i] = np.nan
	    else:
	        # ERROR
	        pass
	# Return adjusted ranks
	return ranks