import math
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
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
        if Z.shape[0] > 1 and x.ndim == 1:
            x = np.array(x, ndmin=2).transpose()

        res = np.zeros_like(x, dtype=float)
        for i_covariant_vector in range(Z.shape[0]):
            for tree in self.trees:
                res[i_covariant_vector] += tree.apply_model_function(
                    function_name, x[i_covariant_vector], Z[i_covariant_vector]
                )
        return res / self.n_trees

    def score(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        tie_tol: float = 1e-8,
    ) -> dict:
        """Returns the concordance index of the model

        Parameters
        ----------
        x : ArrayLike
            Time samples
        Z : ArrayLike | NDArray
            Covariant matrix

        Returns
        -------
        float
            Concordance index (c-index)
        """
        # Steps:
        # 1. Form all pairs of samples
        # 2. Omit pairs where earlier time sample is censored
        #   (the number of permissible pairs, n_permissible_pairs, is the
        #    number of pairs after the above omission)
        # 3. If x_1 < x_2 and x_hat_1 < x_hat_2 => concordance += 1
        # 4. If x_1 < x_2 and x_hat_1 == x_hat_2 => concordance += 0.5
        # 4. If x_1 == x_2 and both are deaths,
        #   if x_hat_1 == x_hat_2 => concordance += 1
        #   else concordance += 0.5
        # c-index = concordance / n_permissible_pairs

        # Correct input
        x = np.array(x, ndmin=1)
        c = np.array(c, ndmin=1)
        Z = np.array(Z, ndmin=2)

        # Package xcZ together
        ixcZ = []
        for i in range(len(x)):
            ixcZ.append((i, x[i], c[i], Z[i]))

        pairs = combinations(ixcZ, 2)

        def predict(i, x, Z):
            """Inner function to get memoised prediction if available,
            otherwise compute, memoise, and return it."""
            # Already memoised
            if memoised_predictions[i] is not None:
                return memoised_predictions[i]

            # Need to calculate it
            memoised_predictions[i] = self.sf(x, Z)
            return memoised_predictions[i]

        memoised_predictions = {i: None for i in range(len(x))}
        concordance = 0.0
        n_permissible_pairs = 0
        n_concordant_pairs = 0
        n_tied_predictions = 0
        n_discordant_pairs = 0
        n_tied_time_samples = 0

        for tup_1, tup_2 in pairs:
            # Get right ordering
            if tup_1[1] > tup_2[1]:
                tup_1, tup_2 = tup_2, tup_1

            # Unpack tuple
            i_1, x_1, c_1, Z_1 = tup_1
            i_2, x_2, c_2, Z_2 = tup_2

            # Omit pair if x_1 is censored
            if c_1 == 1 and x_1 < x_2:
                continue

            # Omit pair if x_1 == x_2 is censored
            if x_1 == x_2 and c_1 == c_2 == 1:
                continue

            n_permissible_pairs += 1

            x_hat_1 = predict(i_1, x_1, Z_1)
            x_hat_2 = predict(i_2, x_2, Z_2)

            if x_1 < x_2:
                if x_hat_1 < x_hat_2:
                    concordance += 1
                    n_concordant_pairs += 1
                elif math.isclose(x_hat_1, x_hat_2, abs_tol=tie_tol):
                    concordance += 0.5
                    n_tied_predictions += 1
                else:
                    n_discordant_pairs += 1
            elif c_1 == c_2 == 0:
                n_tied_time_samples += 1
                if math.isclose(x_hat_1, x_hat_2, abs_tol=tie_tol):
                    concordance += 1
                else:
                    concordance += 0.5
            else:
                # x_1 == x_2 andone is a death
                if (c_1 == 0 and x_hat_1 < x_hat_2) or (
                    c_2 == 0 and x_hat_2 < x_hat_1
                ):
                    concordance += 1
                else:
                    concordance += 0.5

        return {
            "c_index": concordance / n_permissible_pairs,
            "n_concordant_pairs": n_concordant_pairs,
            "n_discordant_pairs": n_discordant_pairs,
            "n_tied_predictions": n_tied_predictions,
            "n_tied_time_samples": n_tied_time_samples,
            "concordance": concordance,
            "n_permissible_pairs": n_permissible_pairs,
        }
