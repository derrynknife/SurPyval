from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval import Exponential, NelsonAalen, Turnbull, Weibull
from surpyval.beta.ml.forest.conditional_inference import ctree_select
from surpyval.beta.ml.forest.deviance_split import (
    _exp_theta0,
    deviance_split,
    needs_full_likelihood_split,
)
from surpyval.beta.ml.forest.log_rank_split import log_rank_split
from surpyval.beta.ml.forest.turnbull_score_split import turnbull_score_split
from surpyval.serialisation import to_native
from surpyval.univariate.parametric import NeverOccurs
from surpyval.utils.surpyval_data import SurpyvalData


class Node(ABC):
    """The common methods between IntermediateNode and TerminalNode."""

    @abstractmethod
    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
    ) -> NDArray:
        """
        Evaluate ``function_name`` (``"sf"``, ``"Hf"``, ...) of the leaf
        model(s) reached by ``Z`` at the times ``x``.

        A 1-D ``Z`` is one covariate vector and returns that leaf's values
        at ``x``. A 2-D ``Z`` holds one covariate vector per row; each row
        is routed on its own and the result is an ``(n_rows, x.size)``
        grid, row ``i`` being the values for ``Z[i]``.
        """

    @abstractmethod
    def to_dict(self) -> dict: ...


class IntermediateNode(Node):
    """
    A split in a survival tree: observations whose feature
    ``split_feature_index`` is at most ``split_feature_value`` go to
    ``left_child``, the rest to ``right_child``. Building one grows the
    subtree below it. In a tree grown with ``selection="ctree"``,
    ``p_value`` is the Bonferroni-adjusted p-value that chose the split's
    feature (``None`` otherwise).
    """

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
        rng: Any = None,
        selection: str = "greedy",
        alpha_split: float = 0.05,
        p_value: float | None = None,
        min_split_gain: float | str = 0.0,
    ) -> None:
        # Set split attributes
        self.split_feature_index = split_feature_index
        self.split_feature_value = split_feature_value
        self.feature_indices_in = feature_indices_in
        self.p_value = p_value

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
            rng=rng,
            selection=selection,
            alpha_split=alpha_split,
            min_split_gain=min_split_gain,
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
            rng=rng,
            selection=selection,
            alpha_split=alpha_split,
            min_split_gain=min_split_gain,
        )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
    ) -> NDArray:
        if np.ndim(Z) < 2:
            # One covariate vector: follow its side of the split
            if Z[self.split_feature_index] <= self.split_feature_value:
                return self.left_child.apply_model_function(
                    function_name, x, Z
                )
            return self.right_child.apply_model_function(function_name, x, Z)

        # One covariate vector per row: split the rows on the feature
        # *column* and send each group down its own branch, so every leaf
        # is evaluated once for all the rows that reach it.
        goes_left = Z[:, self.split_feature_index] <= self.split_feature_value
        res = np.empty((Z.shape[0], np.size(x)), dtype=np.float64)
        if goes_left.any():
            res[goes_left] = self.left_child.apply_model_function(
                function_name, x, Z[goes_left]
            )
        if not goes_left.all():
            res[~goes_left] = self.right_child.apply_model_function(
                function_name, x, Z[~goes_left]
            )
        return res

    def describe(
        self, feature_names: "list[str] | None" = None, right: bool = False
    ) -> str:
        """The split rule as text, ``"temp <= 42"`` (``"temp >  42"`` for
        the right branch with ``right=True``), naming the feature by
        ``feature_names`` (``Z3`` for column 3 without them)."""
        j = int(self.split_feature_index)
        name = (
            feature_names[j]
            if feature_names is not None and j < len(feature_names)
            else f"Z{j}"
        )
        op = ">  " if right else "<= "
        return f"{name} {op}{float(self.split_feature_value):.6g}"

    def to_dict(self) -> dict:
        """Serialise the split rule and both child subtrees. The training
        data is deliberately not stored -- a restored tree is a predictor,
        rebuilt from its structure and its leaf models, not re-fitted."""
        out = {
            "node": "intermediate",
            "split_feature_index": int(self.split_feature_index),
            "split_feature_value": float(self.split_feature_value),
            "feature_indices_in": to_native(self.feature_indices_in),
            "left": self.left_child.to_dict(),
            "right": self.right_child.to_dict(),
        }
        if self.p_value is not None:
            out["p_value"] = float(self.p_value)
        return out

    @classmethod
    def from_dict(cls, node_dict: dict) -> "IntermediateNode":
        # Rebuild without re-running ``build_tree`` (which needs the data):
        # the split rule and children fully determine prediction.
        node = cls.__new__(cls)
        node.split_feature_index = node_dict["split_feature_index"]
        node.split_feature_value = node_dict["split_feature_value"]
        node.feature_indices_in = np.asarray(node_dict["feature_indices_in"])
        node.p_value = node_dict.get("p_value")
        node.left_child = node_from_dict(node_dict["left"])
        node.right_child = node_from_dict(node_dict["right"])
        return node


class TerminalNode(Node):
    """
    A leaf of a survival tree. It holds the observations that reach it
    and fits, on first use, the leaf model given by the tree's ``kind``
    (``model``): a Weibull or Exponential fit, or for a non-parametric
    tree a Nelson-Aalen estimate (a Turnbull estimate if the leaf holds
    left- or interval-censored rows); ``NeverOccurs`` for a parametric
    leaf with no failures.
    """

    def __init__(self, data: SurpyvalData, kind: str = "weibull") -> None:
        self.data = deepcopy(data)
        self.kind = kind

    def _nonparametric_model(self) -> Any:
        # Nelson-Aalen is a risk-set estimator, so it is only defined for
        # observed / right-censored (optionally left-truncated) data; the
        # Turnbull estimate covers left and interval censoring, the data
        # the Turnbull-score split is used on.
        if needs_full_likelihood_split(self.data):
            return Turnbull.fit(
                self.data.x, self.data.c, self.data.n, self.data.t
            )
        return NelsonAalen.fit(
            self.data.x, self.data.c, self.data.n, self.data.t
        )

    def _crude_exponential(self) -> Any:
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
    def model(self) -> Any:
        """The leaf's fitted model, fitted when first used."""
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

    def describe(self) -> str:
        """The leaf as text: its model (with the parameters of a
        parametric one) and, on a fitted tree, the number of units that
        reached it."""
        model = self.model
        if model is NeverOccurs:
            text = "never occurs (no failures)"
        elif hasattr(model, "params") and hasattr(model, "parameter_names"):
            params = ", ".join(
                f"{name}={float(value):.4g}"
                for name, value in zip(model.parameter_names, model.params)
            )
            text = f"{model.dist.name}({params})"
        else:
            text = str(getattr(model, "model", type(model).__name__))
        if self.data is not None:
            text += f", {float(np.sum(self.data.n)):g} units"
        return text

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
    ) -> NDArray:
        values = getattr(self.model, function_name)(x)
        if np.ndim(Z) < 2:
            return values
        # Every row that reached this leaf shares its curve
        values = np.asarray(values, dtype=np.float64).reshape(1, -1)
        return np.repeat(values, np.shape(Z)[0], axis=0)

    def to_dict(self) -> dict:
        """Serialise the leaf as its *fitted* model rather than its data, so
        the restored leaf predicts without re-fitting. ``NeverOccurs`` (the
        empty / all-censored leaf) is a parameterless class, stored as a
        string sentinel; every other leaf is a ``Parametric`` or
        ``NonParametric`` model with its own ``to_dict``."""
        model = self.model
        if model is NeverOccurs:
            leaf: str | dict = "NeverOccurs"
        else:
            leaf = model.to_dict()
        return {"node": "terminal", "kind": self.kind, "leaf": leaf}

    @classmethod
    def from_dict(cls, node_dict: dict) -> "TerminalNode":
        # Local import avoids any import cycle through the package-level
        # serialisation dispatcher.
        from surpyval.serialisation import from_dict as _restore_model

        node = cls.__new__(cls)
        node.kind = node_dict["kind"]
        # A restored predictor carries no training data.
        node.data = None  # type: ignore[assignment]
        leaf = node_dict["leaf"]
        model = NeverOccurs if leaf == "NeverOccurs" else _restore_model(leaf)
        # ``model`` is a cached_property; seed the instance ``__dict__`` slot
        # so the getter (which needs ``data``) never runs on a restored node.
        node.__dict__["model"] = model
        return node


def route_to_leaves(
    node: Node, Z: NDArray
) -> list[tuple["TerminalNode", NDArray]]:
    """The leaf each row of the covariate matrix ``Z`` reaches, grouped:
    a list of ``(leaf, row indices)`` pairs, one per leaf reached. Rows
    are routed as :meth:`Node.apply_model_function` routes them."""
    out: list[tuple[TerminalNode, NDArray]] = []
    stack: list[tuple[Node, NDArray]] = [(node, np.arange(Z.shape[0]))]
    while stack:
        current, idx = stack.pop()
        if idx.size == 0:
            continue
        if isinstance(current, TerminalNode):
            out.append((current, idx))
            continue
        assert isinstance(current, IntermediateNode)
        goes_left = (
            Z[idx, current.split_feature_index] <= current.split_feature_value
        )
        stack.append((current.left_child, idx[goes_left]))
        stack.append((current.right_child, idx[~goes_left]))
    return out


def tree_lines(
    node: Node, feature_names: "list[str] | None", depth: int = 0
) -> list[str]:
    """The subtree under ``node`` as lines of text, in the layout of
    scikit-learn's ``export_text``: each split as its left rule, the left
    subtree indented under it, then its right rule and the right
    subtree; each leaf by :meth:`TerminalNode.describe`."""
    pad = "|   " * depth
    if isinstance(node, TerminalNode):
        return [f"{pad}|--- leaf: {node.describe()}"]
    assert isinstance(node, IntermediateNode)
    p_value = (
        "" if node.p_value is None else f"  (p = {float(node.p_value):.3g})"
    )
    return (
        [f"{pad}|--- {node.describe(feature_names)}{p_value}"]
        + tree_lines(node.left_child, feature_names, depth + 1)
        + [f"{pad}|--- {node.describe(feature_names, right=True)}"]
        + tree_lines(node.right_child, feature_names, depth + 1)
    )


def node_from_dict(node_dict: dict) -> Node:
    """Restore an ``IntermediateNode`` or ``TerminalNode`` from its dict."""
    kind = node_dict.get("node")
    if kind == "intermediate":
        return IntermediateNode.from_dict(node_dict)
    if kind == "terminal":
        return TerminalNode.from_dict(node_dict)
    raise ValueError(
        f"Unrecognised tree node dict (node={kind!r}); expected "
        "'intermediate' or 'terminal'."
    )


def build_tree(
    data: SurpyvalData,
    Z: NDArray,
    curr_depth: int,
    max_depth: int | float,
    min_leaf_samples: int,
    min_leaf_failures: int,
    n_features_split: int,
    kind: str = "weibull",
    rng: Any = None,
    selection: str = "greedy",
    alpha_split: float = 0.05,
    min_split_gain: float | str = 0.0,
) -> Node:
    """
    Node factory. Decides to return IntermediateNode object, or its
    sibling TerminalNode.

    ``kind`` couples the split criterion with the matching leaf model:
    ``"weibull"`` (Weibull deviance split, Weibull leaves),
    ``"exponential"`` (exponential deviance split, Exponential leaves)
    or ``"non-parametric"``: the risk-set log-rank split with
    Nelson-Aalen leaves at a node of observed / right-censored data
    (optionally left truncated), and the Turnbull-score split with
    Turnbull leaves at a node with left- or interval-censored rows
    (untruncated).

    ``rng`` draws the features considered at each split: numpy's global
    generator when ``None`` (see
    :func:`~surpyval.beta.ml.forest.tree.resolve_random_state`).

    ``selection="greedy"`` takes the best cut of the kind's criterion over
    every drawn feature. ``selection="ctree"`` first chooses the feature
    by conditional inference (see
    :mod:`~surpyval.beta.ml.forest.conditional_inference`), stops if its
    Bonferroni-adjusted p-value is not below ``alpha_split``, and
    otherwise cuts that feature by the kind's criterion.

    ``min_split_gain`` is the least log-likelihood gain a deviance split
    must make (see
    :func:`~surpyval.beta.ml.forest.deviance_split.deviance_split`).
    """
    if rng is None:
        rng = np.random.mtrand._rand

    # If max_depth has been reached, return a TerminalNode
    if curr_depth == max_depth:
        return TerminalNode(data, kind)

    # Choose the random n_features_split subset of features, without
    # replacement
    feature_indices_in = np.unique(
        rng.choice(Z.shape[1], size=n_features_split, replace=False)
    )

    # Conditional inference picks the feature, or stops the tree here;
    # greedy search lets the split criterion consider every drawn feature.
    candidates = feature_indices_in
    p_value = None
    if selection == "ctree":
        chosen, p_value = ctree_select(
            data, Z, kind, min_leaf_samples, min_leaf_failures, candidates
        )
        if chosen == -1 or not p_value < alpha_split:
            return TerminalNode(data, kind)
        candidates = np.array([chosen])

    # Figure out best feature-value split
    if kind == "non-parametric" and needs_full_likelihood_split(data):
        # Left or interval censoring (the tree has refused truncation
        # with it): the log-rank scores of the pooled Turnbull estimate.
        split_feature_index, split_feature_value = turnbull_score_split(
            data, Z, min_leaf_samples, min_leaf_failures, candidates
        )
    elif kind == "non-parametric":
        split_feature_index, split_feature_value = log_rank_split(
            data, Z, min_leaf_samples, min_leaf_failures, candidates
        )
    else:
        split_feature_index, split_feature_value = deviance_split(
            data,
            Z,
            min_leaf_samples,
            min_leaf_failures,
            candidates,
            model=kind,
            min_split_gain=min_split_gain,
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
        rng=rng,
        selection=selection,
        alpha_split=alpha_split,
        p_value=p_value,
        min_split_gain=min_split_gain,
    )
