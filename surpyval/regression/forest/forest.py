import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from surpyval.regression.forest.tree import Tree
from surpyval.utils import xcnt_handler
from surpyval.utils.score import score


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
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        n_trees: int = 100,
        max_depth: int | float = float("inf"),
        min_leaf_failures: int = 6,
        n_features_split: int | float | str = "sqrt",
        bootstrap: bool = True,
    ):
        # Parse data
        x_, c_, n_, t_ = xcnt_handler(x, c, n, t, group_and_sort=False)
        self.x: NDArray = x_
        self.Z: NDArray = np.array(Z, ndmin=2)
        self.c: NDArray = c_
        self.n: NDArray = n_
        self.t: NDArray = t_
        self.n_trees = n_trees
        self.bootstrap = bootstrap

        # Create Trees
        if self.bootstrap:
            bootstrap_indices = [
                np.random.choice(len(self.x), len(self.x), replace=True)
                for _ in range(self.n_trees)
            ]
        else:
            bootstrap_indices = [np.array(range(len(self.x)))] * self.n_trees

        self.trees: list[Tree] = Parallel(prefer="threads", verbose=1)(
            delayed(Tree)(
                x=self.x[bootstrap_indices[i]],
                Z=self.Z[bootstrap_indices[i]],
                c=self.c[bootstrap_indices[i]],
                max_depth=max_depth,
                min_leaf_failures=min_leaf_failures,
                n_features_split=n_features_split,
            )
            for i in range(self.n_trees)
        )

    def sf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        ensemble_method: str = "sf",
    ) -> NDArray:
        """Returns the ensemble survival function

        Parameters
        ----------
        x : int | float | ArrayLike
            Time samples
        Z : ArrayLike | NDArray
            Covariant matrix
        ensemble_method : str, optional
            Determines whether to average across terminal nodes the terminal
            node survival functions or cumulative hazard functions.
            For these respectively, ensemble_method must be "sf" or
            "Hf". Defaults to "sf".

        Returns
        -------
        NDArray
            Survival function of x as 1D array
        """
        if ensemble_method == "Hf":
            Hf = self._apply_model_function_to_trees("Hf", x, Z)
            return np.exp(-Hf)
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

    def mortality(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> ArrayLike:
        mortality = self.Hf(x, Z).sum(1)
        return np.clip(mortality, 0, np.finfo(np.float64).max)

    def _apply_model_function_to_trees(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
    ) -> NDArray:
        # Prep input - make sure numpy array
        x = np.array(x, ndmin=1)
        Z = np.array(Z, ndmin=2)

        # If there are multiple covariant vector samples, ?
        # if Z.shape[0] > 1 and x.ndim == 1:
        #     x = np.array(x, ndmin=2).transpose()

        res = np.zeros((Z.shape[0], x.size)).astype(np.float64)
        for i_covariant_vector in range(Z.shape[0]):
            for tree in self.trees:
                values = tree.apply_model_function(
                    function_name, x, Z[i_covariant_vector, :]
                )
                res[i_covariant_vector, :] += values
        return res / self.n_trees

    def score(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        tie_tol: float = 1e-8,
    ) -> float:
        scores: ArrayLike = self.mortality(x, Z)
        tol: float = 1e-8
        return score(x, c, scores, tol)
