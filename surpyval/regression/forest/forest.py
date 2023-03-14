import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.regression.forest.tree import Tree


class RandomSurvivalForest:
    """Random Survival Forest

    Specs:
    - n_trees `Tree`'s trained, each given independently bootstrapped samples
    - Each tree is trained
    - Each predicition is evaluated for each tree, the Weibull models are
      collected, then averaged
    """

    def __init__(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        n_trees: int = 100,
        max_depth: int | float = float("inf"),
        min_leaf_failures: int = 6,
        n_features_split: int | float | str = "sqrt",
        bootstrap: bool = True,
    ):
        # Parse data
        self.x = np.array(x)
        self.Z = np.array(Z)
        if self.Z.ndim == 1:
            self.Z = np.reshape(Z, (1, -1)).transpose()
        self.c = np.array(c)
        self.n_trees = n_trees
        self.bootstrap = bootstrap

        # Create trees
        self.trees = []
        bootstrap_indices = np.array(range(len(self.x)))  # Defaults to normal
        for _ in range(self.n_trees):
            if self.bootstrap:
                bootstrap_indices = np.random.choice(
                    len(self.x), len(self.x), replace=True
                )
            self.trees.append(
                Tree(
                    x=self.x[bootstrap_indices],
                    Z=self.Z[bootstrap_indices],
                    c=self.c[bootstrap_indices],
                    max_depth=max_depth,
                    min_leaf_failures=min_leaf_failures,
                    n_features_split=n_features_split,
                )
            )

    def sf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("sf", x, Z)

    def ff(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("ff", x, Z)

    def df(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("df", x, Z)

    def hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("hf", x, Z)

    def Hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("Hf", x, Z)

    def _apply_model_function_to_trees(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
    ) -> NDArray:
        # Prep input - make sure numpy array
        x = np.array(x, ndmin=1)
        Z = np.array(Z, ndmin=1)

        res = np.zeros_like(x, dtype=float)
        for tree in self.trees:
            res += tree.apply_model_function(function_name, x, Z)
        return res / self.n_trees
