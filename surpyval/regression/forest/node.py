from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval import Exponential, NelsonAalen, Weibull
from surpyval.parametric import NeverOccurs
from surpyval.regression.forest.log_rank_split import log_rank_split
from surpyval.utils.surpyval_data import SurpyvalData


class Node(ABC):
    """The common methods between IntermediateNode and LeafNode."""

    @abstractmethod
    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
    ) -> NDArray:
        ...


class IntermediateNode(Node):
    def __init__(
        self,
        data: SurpyvalData,
        Z: NDArray,
        curr_depth: int,
        max_depth: int | float,
        min_leaf_samples: int,
        min_leaf_failures: int,
        n_features_split: int,
        split_feature_index: int,
        split_feature_value: float,
        feature_indices_in: NDArray,
        parametric: bool = True,
    ):
        # Set split attributes
        self.split_feature_index = split_feature_index
        self.split_feature_value = split_feature_value
        self.feature_indices_in = feature_indices_in

        # Get left/right indices
        left_indices = (
            Z[:, self.split_feature_index] <= self.split_feature_value
        )
        right_indices = np.logical_not(left_indices)

        # Build left and right nodes
        self.left_child = build_tree(
            data[left_indices],
            Z[left_indices, :],
            curr_depth=curr_depth + 1,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features_split,
            parametric=parametric,
        )
        self.right_child = build_tree(
            data[right_indices],
            Z[right_indices, :],
            curr_depth=curr_depth + 1,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features_split,
            parametric=parametric,
        )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
    ) -> NDArray:
        # Determine which node, left/right, to call sf() on
        if Z[self.split_feature_index] <= self.split_feature_value:
            return self.left_child.apply_model_function(function_name, x, Z)
        return self.right_child.apply_model_function(function_name, x, Z)


class TerminalNode(Node):
    def __init__(self, data: SurpyvalData, parametric: bool = True):
        self.data = deepcopy(data)
        self.paramettric = parametric

    @cached_property
    def model(self):
        if self.paramettric:
            if self.data.to_xrd()[2].sum() == 0:
                return NeverOccurs
            elif self.data.to_xrd()[2].sum() <= 1:
                return Exponential.fit_from_surpyval_data(self.data)
            else:
                return Weibull.fit_from_surpyval_data(self.data)
        else:
            return NelsonAalen.fit(
                self.data.x, self.data.c, self.data.n, self.data.t
            )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        _: NDArray,
    ) -> NDArray:
        return getattr(self.model, function_name)(x)


def build_tree(
    data: SurpyvalData,
    Z: NDArray,
    curr_depth: int,
    max_depth: int | float,
    min_leaf_samples: int,
    min_leaf_failures: int,
    n_features_split: int,
    parametric: bool = True,
) -> Node:
    """
    Node factory. Decides to return IntermediateNode object, or its
    sibling TerminalNode.
    """
    # If max_depth has been reached, return a TerminalNode
    if curr_depth == max_depth:
        return TerminalNode(data)

    # Choose the random n_features_split subset of features, without
    # replacement
    feature_indices_in = np.unique(
        np.random.choice(Z.shape[1], size=n_features_split, replace=False)
    )

    # Figure out best feature-value split
    split_feature_index, split_feature_value = log_rank_split(
        data, Z, min_leaf_samples, min_leaf_failures, feature_indices_in
    )

    # If log_rank_split() can't suggest a feature-value split, return a
    # TerminalNode
    if split_feature_index == -1 and split_feature_value == float("-Inf"):
        return TerminalNode(data, parametric)

    # Else, return an IntermediateNode, with the best feature-value split
    return IntermediateNode(
        data=data,
        Z=Z,
        curr_depth=curr_depth,
        max_depth=max_depth,
        min_leaf_samples=min_leaf_samples,
        min_leaf_failures=min_leaf_failures,
        n_features_split=n_features_split,
        split_feature_index=split_feature_index,
        split_feature_value=split_feature_value,
        feature_indices_in=feature_indices_in,
        parametric=parametric,
    )
