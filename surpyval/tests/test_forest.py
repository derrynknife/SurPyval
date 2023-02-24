"""
Tests surpyval's implementation of Random Survival Forests. Since it uses
scikit-survival's RandomSurvivalForest, which is already reasonably well
tested, here we test the new component: the Weibull parametric models on the
leaves.
"""


import numpy as np
import pytest

from surpyval import RandomSurvivalForest, Weibull
from surpyval.regression.tree import SurvivalTree

# Can import RandomSurvivalForest - TICK

# Tree does the same as scikit-survival's, EXCEPT we now have, for each
# terminal node, a Weibull distribution (with parameters) that can be checked!

# Next we need to test the next level: the Forest. Namely, the
# averaging/summing/?how is this done? of all the selected terminal nodes
# across trees.


def test_tree_single_binary_covariate():
    """
    Using a single binary covariate dataset (from Wikipedia's 'Proportional
    hazards model' page:
    https://en.wikipedia.org/wiki/Proportional_hazards_model), test that
    the survival tree returns the correct Weibull distributions.
    """
    # Dataset:
    # X = Hospital (1=A or 2=B)
    # T = period of time measure before death in month
    # T=60 => end of 5 year study period reached before death (right-censored)
    # C = censoring (1=right-censored)
    # Note: I added a few more data points so a Weibull would converge!
    X = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    T = [60, 32, 60, 60, 60, 40, 4, 18, 60, 9, 31, 53, 17]
    C = [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    # Now we'll be making a depth=1 decision tree, so just a decision stump
    # Which since the dataset covariant is binary, the feature split would
    # just be =0 and =1, so we know what samples will end up in each of the
    # terminal nodes. i.e.:
    T_zero = T[:6]
    C_zero = C[:6]
    T_one = T[6:]
    C_one = C[6:]

    # So by fitting these sets with surpyval Weibull models, we can get the
    # Weibull parameters, which should be identical to those returned by the
    # tree
    weibull_zero_params = Weibull.fit(x=T_zero, c=C_zero).params
    weibull_one_params = Weibull.fit(x=T_one, c=C_one).params

    y = list(zip(np.logical_not(C), T))
    y = np.array(y, dtype=[("Censored", np.bool_), ("Time", np.float64)])

    tree = SurvivalTree(max_depth=1)
    tree.fit(np.array(X).reshape(-1, 1), y)

    assert pytest.approx(weibull_zero_params) == tree.leaf_models[0].params
    assert pytest.approx(weibull_one_params) == tree.leaf_models[1].params
