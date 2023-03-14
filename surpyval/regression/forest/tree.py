from math import log2, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.regression.forest.node import build_tree


class Tree:
    """
    A Survival Tree, for use in `RandomSurvivalForest`.

    The Tree is built on initialisation.
    """

    def __init__(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        max_depth: int | float = float("inf"),
        min_leaf_failures: int = 6,
        n_features_split: int | float | str = "sqrt",
    ):
        self.x = np.array(x)
        self.Z = np.array(Z)
        if self.Z.ndim == 1:
            self.Z = np.reshape(Z, (1, -1)).transpose()
        self.c = np.array(c)

        self.n_features_split = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self._root = build_tree(
            x=self.x,
            Z=self.Z,
            c=self.c,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_failures=min_leaf_failures,
            n_features_split=self.n_features_split,
        )

    def apply_model_function(
        self,
        function_name: str,
        x: NDArray,
        Z: NDArray,
    ) -> NDArray:
        return self._root.apply_model_function(function_name, x, Z)


def parse_n_features_split(
    n_features_split: int | float | str, n_features: int
) -> int:
    if isinstance(n_features_split, int):
        return n_features_split
    if isinstance(n_features_split, float):
        return int(n_features_split * n_features)
    if n_features_split == "sqrt":
        return int(sqrt(n_features))
    if n_features_split == "log2":
        return int(log2(n_features))
    if n_features_split == "all":
        return n_features
    else:
        raise ValueError(
            f"n_features_split={n_features_split} is invalid. See\
                         `Tree` docstring for valid values."
        )
