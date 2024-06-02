from math import log2, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.regression.forest.node import build_tree
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
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        parametric: str = "non-parametric",
    ):
        self.data = data
        self.Z = Z

        n_features: int = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self.n_features_split = n_features

        is_parametric: bool = parse_leaf_type(parametric)

        self._root = build_tree(
            data=self.data,
            Z=self.Z,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features,
            parametric=is_parametric,
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
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        leaf_type: str = "parametric",
    ):
        data = SurpyvalData(x, c, n, t, group_and_sort=False)
        Z = np.array(Z, ndmin=2)
        n_features: int = parse_n_features_split(n_features_split, Z.shape[1])
        return cls(
            data, Z, max_depth, min_leaf_failures, n_features, leaf_type
        )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
    ) -> NDArray:
        # Prep input - make sure numpy array
        x = np.array(x, ndmin=1)
        Z = np.array(Z, ndmin=1)

        return self._root.apply_model_function(function_name, x, Z)

    def sf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self.apply_model_function("ff", x, Z)

    def ff(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self.apply_model_function("ff", x, Z)

    def df(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self.apply_model_function("df", x, Z)

    def hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self.apply_model_function("hf", x, Z)

    def Hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self.apply_model_function("Hf", x, Z)


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


def parse_leaf_type(leaf_type: str):
    if leaf_type.lower() == "non-parametric":
        return False
    elif leaf_type.lower() == "parametric":
        return True
    else:
        raise ValueError(
            f"leaf_type={leaf_type} is invalid. Must\
            'parametric' or 'non-parametric'"
        )
