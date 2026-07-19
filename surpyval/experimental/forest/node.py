from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval import Exponential, NelsonAalen, Turnbull, Weibull
from surpyval.experimental.forest.deviance_split import (
    _exp_theta0,
    deviance_split,
    needs_full_likelihood_split,
)
from surpyval.experimental.forest.log_rank_split import log_rank_split
from surpyval.univariate.parametric import NeverOccurs
from surpyval.utils.surpyval_data import SurpyvalData


class Node(ABC):
    """The common methods between IntermediateNode and LeafNode."""

    @abstractmethod
    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
    ) -> NDArray: ...


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
        kind: str = "weibull",
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
            kind=kind,
        )
        self.right_child = build_tree(
            data[right_indices],
            Z[right_indices, :],
            curr_depth=curr_depth + 1,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features_split,
            kind=kind,
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
    def __init__(self, data: SurpyvalData, kind: str = "weibull"):
        self.data = deepcopy(data)
        self.kind = kind

    def _nonparametric_model(self):
        # Nelson-Aalen is a risk-set estimator, so it is only defined for
        # observed / right-censored (optionally left-truncated) data; the
        # Turnbull NPMLE covers the full data model. The tree entry point
        # already raises for a non-parametric kind on such data, so the
        # Turnbull branch is defence in depth for direct build_tree use.
        if needs_full_likelihood_split(self.data):
            return Turnbull.fit(
                self.data.x, self.data.c, self.data.n, self.data.t
            )
        return NelsonAalen.fit(
            self.data.x, self.data.c, self.data.n, self.data.t
        )

    def _crude_exponential(self):
        # Last-resort parametric leaf: the crude event-weight / exposure
        # rate. Always computable when the leaf carries any event
        # information, so a parametric tree stays parametric all the way
        # down (a leaf must never crash the forest, but it must not
        # silently become nonparametric either).
        theta0 = _exp_theta0(self.data)
        if theta0 is None:
            return NeverOccurs
        return Exponential.from_params([float(np.exp(theta0))])

    @cached_property
    def model(self):
        if self.kind == "non-parametric":
            return self._nonparametric_model()

        # n-weighted count of event-informative observations (any
        # observation that is not purely right censored).
        n_failures = self.data.n[self.data.c != 1].sum()
        if n_failures == 0:
            return NeverOccurs

        # A degenerate bootstrap sample (e.g. heavily tied event times)
        # can make an MLE's covariance/Hessian step fail. A single
        # terminal node must not crash the whole forest, so fall back to
        # progressively simpler fits -- staying within the parametric
        # family: Weibull -> Exponential (a Weibull with shape fixed at
        # 1) -> the crude rate.
        if self.kind == "weibull" and n_failures > 1:
            try:
                return Weibull.fit_from_surpyval_data(self.data)
            except Exception:
                pass
        try:
            return Exponential.fit_from_surpyval_data(self.data)
        except Exception:
            return self._crude_exponential()

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
    kind: str = "weibull",
) -> Node:
    """
    Node factory. Decides to return IntermediateNode object, or its
    sibling TerminalNode.

    ``kind`` couples the split criterion with the matching leaf model:
    ``"weibull"`` (Weibull deviance split, Weibull leaves),
    ``"exponential"`` (exponential deviance split, Exponential leaves)
    or ``"non-parametric"`` (risk-set log-rank split, Nelson-Aalen
    leaves; observed / right-censored data, optionally left truncated).
    """
    # If max_depth has been reached, return a TerminalNode
    if curr_depth == max_depth:
        return TerminalNode(data, kind)

    # Choose the random n_features_split subset of features, without
    # replacement
    feature_indices_in = np.unique(
        np.random.choice(Z.shape[1], size=n_features_split, replace=False)
    )

    # Figure out best feature-value split
    if kind == "non-parametric":
        split_feature_index, split_feature_value = log_rank_split(
            data, Z, min_leaf_samples, min_leaf_failures, feature_indices_in
        )
    else:
        split_feature_index, split_feature_value = deviance_split(
            data,
            Z,
            min_leaf_samples,
            min_leaf_failures,
            feature_indices_in,
            model=kind,
        )

    # If the split rule can't suggest a feature-value split, return a
    # TerminalNode
    if split_feature_index == -1 and split_feature_value == float("-Inf"):
        return TerminalNode(data, kind)

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
        kind=kind,
    )
