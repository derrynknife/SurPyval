from sksurv.tree.tree import SurvivalTree as sksurv_SurvivalTree


class SurvivalTree(sksurv_SurvivalTree):
    """
    Inherits scikit-survival's SurvivalTree, but which replaces the
    non-parametric terminal node models with the parametric Weibull model.

    Only differences:
    - After fitting the tree, the Weibull model for each terminal node is
      fitted, and saved
    - Predict functions:
        - `predict_cumulative_hazard_function(X)`just traces X down to the
          terminal node, then returns the cumulative hazard function of that
          terminal node Weibull function
    """

    # tree.apply(np.array([0], dtype=np.float32).reshape(-1, 1))
    # gets the terminal node for X=[0]

    def fit(self, X, y, *args, **kwargs):
        super().fit(X, y)
        self.leaf_models = 0
