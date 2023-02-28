import pytest

from surpyval import Weibull
from surpyval.regression.forest.tree import Tree

"""
NOTES

Things to test:
- One split - feature split
- One split - Weibull fit + prediction
- One split - all functions
- Bootstrapping - one split

Convention:
Z = Covariant Matrix
x = Event-time vector
c = Censor vector


"""


def test_tree_one_split_one_feature():
    """
    One split. One feature. Ten samples.
    Tests depth=1, feature-value split, and leaf Weibull models
    """
    # Samples
    Z = [0] * 5 + [1] * 5
    x = [10, 15, 12, 5, 11, 20, 30, 35, 30, 25]
    c = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]

    # Expected
    exp_left_weibull = Weibull.fit(x=x[:5], c=c[:5])
    exp_right_weibull = Weibull.fit(x=x[5:], c=c[5:])

    # Actual
    tree = Tree.fit(x=x, Z=Z, c=c, max_depth=1)
    left_weibull = tree._root.left_node.model
    right_weibull = tree._root.right_node.model

    # Assert
    assert pytest.approx(exp_left_weibull.params) == left_weibull.params
    assert pytest.approx(exp_right_weibull.params) == right_weibull.params
