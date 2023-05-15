from math import log2, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.regression.forest.node import build_tree
from surpyval.utils import xcnt_handler
from surpyval.utils.surpyval_data import SurpyvalData


class SurvivalTree:
    """
    A Survival Tree, for use in `RandomSurvivalForest`.

    The Tree is built on initialisation.
    """

    def __init__(
        self,
        data: SurpyvalData,
        Z: NDArray,
        max_depth: int | float = float("inf"),
        min_leaf_failures: int = 6,
        n_features_split: int | float | str = "sqrt",
    ):
        self.data = data
        self.Z = Z

        self.n_features_split = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self._root = build_tree(
            data=self.data,
            Z=self.Z,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_failures=min_leaf_failures,
            n_features_split=self.n_features_split,
        )

    @classmethod
    def fit(
        cls,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        max_depth: int | float = float("inf"),
        min_leaf_failures: int = 6,
        n_features_split: int | float | str = "sqrt",
    ):
        data = xcnt_handler(
            x, c, n, t, group_and_sort=False, as_surpyval_dataset=True
        )
        Z = np.array(Z, ndmin=2)
        return cls(data, Z, max_depth, min_leaf_failures, n_features_split)

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
