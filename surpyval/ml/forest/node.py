from abc import ABC, abstractmethod
from copy import deepcopy
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval import Exponential, NelsonAalen, Turnbull, Weibull
from surpyval.ml.forest.cause_specific_split import CauseSplit, cause_codes
from surpyval.ml.forest.conditional_inference import ctree_select
from surpyval.ml.forest.deviance_split import (
    _exp_theta0,
    deviance_split,
    leaf_mle,
    leaf_mles,
    needs_full_likelihood_split,
)
from surpyval.ml.forest.log_rank_split import log_rank_split
from surpyval.ml.forest.random_cut import random_cuts
from surpyval.ml.forest.turnbull_score_split import turnbull_score_split
from surpyval.serialisation import to_native
from surpyval.univariate.competing_risks.labels import label_from_native
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
        event: Any = None,
    ) -> NDArray:
        """
        Evaluate ``function_name`` (``"sf"``, ``"Hf"``, ...; ``"cif"`` of
        cause ``event`` in a tree grown with causes) of the leaf model(s)
        reached by ``Z`` at the times ``x``.

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
        competing: CauseSplit | None = None,
        splitrule: str = "searched",
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
            competing=competing,
            splitrule=splitrule,
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
            competing=competing,
            splitrule=splitrule,
        )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
        event: Any = None,
    ) -> NDArray:
        if np.ndim(Z) < 2:
            # One covariate vector: follow its side of the split
            if Z[self.split_feature_index] <= self.split_feature_value:
                return self.left_child.apply_model_function(
                    function_name, x, Z, event
                )
            return self.right_child.apply_model_function(
                function_name, x, Z, event
            )

        # One covariate vector per row: split the rows on the feature
        # *column* and send each group down its own branch, so every leaf
        # is evaluated once for all the rows that reach it.
        goes_left = Z[:, self.split_feature_index] <= self.split_feature_value
        res = np.empty((Z.shape[0], np.size(x)), dtype=np.float64)
        if goes_left.any():
            res[goes_left] = self.left_child.apply_model_function(
                function_name, x, Z[goes_left], event
            )
        if not goes_left.all():
            res[~goes_left] = self.right_child.apply_model_function(
                function_name, x, Z[~goes_left], event
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
    left- or interval-censored or right-truncated rows); ``NeverOccurs``
    for a parametric
    leaf with no failures. On observed and right-censored data a
    parametric leaf is the maximum found as the split search finds a
    child's, built from its parameters (so it has no ``cb()`` of its own);
    a tree finds those for all its leaves together as it is grown
    (:func:`fit_leaves`).
    """

    def __init__(
        self, data: SurpyvalData, kind: str = "weibull", copy: bool = True
    ) -> None:
        # A tree's leaves are given rows the tree has already copied
        # (``copy=False``): a deep copy per leaf was an eighth of growing
        # a non-parametric forest (#549).
        self.data = deepcopy(data) if copy else data
        self.kind = kind

    def _nonparametric_model(self) -> Any:
        # Nelson-Aalen is a risk-set estimator, so it is only defined for
        # observed / right-censored (optionally left-truncated) data; the
        # Turnbull estimate covers left and interval censoring and right
        # truncation, the data the Turnbull-score split is used on.
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

        # On observed and right-censored data the leaf is the maximum the
        # split search finds, built from its parameters (leaf_mle) rather
        # than re-fitted: a full fit per leaf made a forest's first
        # prediction several times slower than growing it. Such a leaf
        # has no covariance, so no cb() of its own; the forest uses none.
        #
        # Otherwise the leaf is fitted. A degenerate bootstrap sample
        # (e.g. heavily tied event times) can make an MLE's
        # covariance/Hessian step fail. A single terminal node must not
        # crash the whole forest, so fall back to progressively simpler
        # fits -- staying within the parametric family: Weibull ->
        # Exponential (a Weibull with shape fixed at 1) -> the crude rate.
        if self.kind == "weibull" and n_failures > 1:
            params = leaf_mle(self.data, "weibull")
            if params is not None:
                return Weibull.from_params(params)
            try:
                return Weibull._fit_from_surpyval_data(self.data)
            except Exception:
                pass
        params = leaf_mle(self.data, "exponential")
        if params is not None:
            return Exponential.from_params(params)
        try:
            return Exponential._fit_from_surpyval_data(self.data)
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
        units = self.units
        if units is not None:
            text += f", {units:g} units"
        return text

    @property
    def units(self) -> float | None:
        """The number of units (``n``-weighted rows) that reached the leaf
        when the tree was grown (kept by a restored tree; ``None`` for one
        saved before it was stored)."""
        if self.data is not None:
            return float(np.sum(self.data.n))
        return self.__dict__.get("_units")

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: NDArray,
        event: Any = None,
    ) -> NDArray:
        values = self.evaluate(function_name, x, event)
        if np.ndim(Z) < 2:
            return values
        # Every row that reached this leaf shares its curve
        values = np.asarray(values, dtype=np.float64).reshape(1, -1)
        return np.repeat(values, np.shape(Z)[0], axis=0)

    def evaluate(
        self, function_name: str, x: int | float | ArrayLike, event: Any = None
    ) -> Any:
        """The leaf model's ``function_name`` at the times ``x`` (``event``
        is for a leaf with causes; a plain leaf has none)."""
        return getattr(self.model, function_name)(x)

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
        out: dict = {"node": "terminal", "kind": self.kind, "leaf": leaf}
        if self.units is not None:
            out["units"] = self.units
        return out

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
        node.__dict__["_units"] = node_dict.get("units")
        return node


class CompetingRisksLeaf(TerminalNode):
    """
    A leaf of a survival tree grown with the cause of each failure. Its
    ``model`` is a competing-risks model of the rows that reach it: for a
    ``"non-parametric"`` tree the Aalen-Johansen estimate
    (:class:`~surpyval.univariate.competing_risks.CompetingRisks`, with the
    Kaplan-Meier all-cause survival, so ``sf`` is one less the sum of the
    causes' cumulative incidences); for a ``"weibull"`` or
    ``"exponential"`` tree a
    :class:`~surpyval.univariate.competing_risks.ParametricCompetingRisks`
    of one distribution per cause, each fitted to the cause-specific view
    of the leaf's rows (that cause's failures observed, the rest censored)
    as a plain leaf of the kind is fitted. ``None`` for a leaf with no
    failures.

    A cause of the tree with no failure in the leaf (``absent``) has zero
    incidence there, and :meth:`describe` says so.
    """

    def __init__(self, data: SurpyvalData, kind: str, causes: list) -> None:
        super().__init__(data, kind, copy=False)
        self.causes = list(causes)

    @cached_property
    def present(self) -> list:
        """The tree's causes with at least one failure in the leaf."""
        codes = cause_codes(self.data.e, self.causes)
        failed = np.asarray(self.data.c) == 0
        return [
            label
            for k, label in enumerate(self.causes)
            if (failed & (codes == k)).any()
        ]

    @property
    def absent(self) -> list:
        """The tree's causes with no failure in the leaf."""
        return [k for k in self.causes if k not in self.present]

    @cached_property
    def model(self) -> Any:
        """The leaf's competing-risks model (``None`` with no failures)."""
        # Local imports: the competing-risks package imports the top-level
        # surpyval namespace, which imports this module.
        from surpyval.univariate.competing_risks import (
            CompetingRisks,
            ParametricCompetingRisks,
        )

        if not self.present:
            return None
        data = self.data
        tl = np.asarray(data.t[:, 0], dtype=float)
        entry = {"tl": tl} if np.isfinite(tl).any() else {}
        if self.kind == "non-parametric":
            return CompetingRisks.fit(
                np.asarray(data.x, dtype=float),
                data.e,
                data.c,
                data.n,
                how="Kaplan-Meier",
                **entry,
            )
        codes = cause_codes(data.e, self.causes)
        models = {}
        for label in self.present:
            failed = codes == self.causes.index(label)
            view = SurpyvalData(
                np.asarray(data.x),
                np.where(failed, 0, 1),
                np.asarray(data.n),
                data.t,
                handle=False,
            )
            models[label] = TerminalNode(view, self.kind, copy=False).model
        return ParametricCompetingRisks.from_fitted(models)

    def evaluate(
        self, function_name: str, x: int | float | ArrayLike, event: Any = None
    ) -> Any:
        """``function_name`` at the times ``x``: the all-cause ``sf``,
        ``ff``, ``Hf``, ``hf`` or ``df``, or ``"cif"`` of cause ``event``
        (zero for a cause absent from the leaf)."""
        x = np.asarray(x, dtype=float)
        model = self.model
        if function_name == "cif":
            if model is None or event not in self.present:
                return np.where(np.isnan(x), np.nan, 0.0)
            return np.asarray(model.cif(x, event), dtype=float)
        if model is None:
            # No failures: nothing happens
            value = 1.0 if function_name == "sf" else 0.0
            return np.where(np.isnan(x), np.nan, value)
        return np.asarray(getattr(model, function_name)(x), dtype=float)

    def describe(self) -> str:
        """The leaf as text: its causes' models, the causes with zero
        incidence here, and the number of units that reached it."""
        model = self.model
        if model is None:
            text = "no failures: every cause has zero incidence"
        else:
            if hasattr(model, "models"):
                parts = []
                for label in self.present:
                    sub = model.models[label]
                    params = ", ".join(
                        f"{name}={float(value):.4g}"
                        for name, value in zip(sub.parameter_names, sub.params)
                    )
                    parts.append(f"{label!r}: {sub.dist.name}({params})")
                text = "; ".join(parts)
            else:
                text = "Aalen-Johansen, causes " + ", ".join(
                    repr(k) for k in self.present
                )
            if self.absent:
                text += "; zero incidence (no failures here): " + ", ".join(
                    repr(k) for k in self.absent
                )
        units = self.units
        if units is not None:
            text += f", {units:g} units"
        return text

    def to_dict(self) -> dict:
        """The leaf's fitted competing-risks model (``None`` with no
        failures), the causes present and the units that reached it."""
        model = self.model
        out: dict = {
            "node": "terminal-causes",
            "kind": self.kind,
            "causes": to_native(list(self.causes)),
            "present": to_native(list(self.present)),
            "leaf": None if model is None else model.to_dict(),
        }
        if self.units is not None:
            out["units"] = self.units
        return out

    @classmethod
    def from_dict(cls, node_dict: dict) -> "CompetingRisksLeaf":
        from surpyval.serialisation import from_dict as _restore_model

        node = cls.__new__(cls)
        node.kind = node_dict["kind"]
        node.data = None  # type: ignore[assignment]
        node.causes = [label_from_native(k) for k in node_dict["causes"]]
        leaf = node_dict["leaf"]
        node.__dict__["model"] = None if leaf is None else _restore_model(leaf)
        node.__dict__["present"] = [
            label_from_native(k) for k in node_dict["present"]
        ]
        node.__dict__["_units"] = node_dict.get("units")
        return node


def fit_leaves(root: Node) -> None:
    """Fit the parametric leaves of the tree under ``root`` together.

    Each leaf's model is what :attr:`TerminalNode.model` would fit on
    first use; the leaves whose maximum :func:`leaf_mle` finds (observed
    and right-censored rows) are found in one pass for the whole tree,
    rather than one leaf at a time (#549). Any other leaf stays to be
    fitted on first use, as before.
    """
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, IntermediateNode):
            stack += [node.right_child, node.left_child]
        elif (
            isinstance(node, TerminalNode)
            and not isinstance(node, CompetingRisksLeaf)
            and node.kind != "non-parametric"
            and node.data is not None
            and "model" not in node.__dict__
        ):
            leaves.append(node)
    batches: dict = {"weibull": [], "exponential": []}
    for leaf in leaves:
        n_failures = leaf.data.n[leaf.data.c != 1].sum()
        if n_failures == 0:
            leaf.__dict__["model"] = NeverOccurs
        elif leaf.kind == "weibull" and n_failures > 1:
            batches["weibull"].append(leaf)
        else:
            batches["exponential"].append(leaf)
    for model, dist in (("weibull", Weibull), ("exponential", Exponential)):
        batch = batches[model]
        # Quiet: a leaf the closed forms cannot fit is left to be fitted
        # on first use, as before, with whatever that says.
        with np.errstate(all="ignore"):
            fitted = leaf_mles([leaf.data for leaf in batch], model)
        for leaf, params in zip(batch, fitted):
            if params is not None:
                leaf.__dict__["model"] = dist.from_params(params)


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
    if kind == "terminal-causes":
        return CompetingRisksLeaf.from_dict(node_dict)
    raise ValueError(
        f"Unrecognised tree node dict (node={kind!r}); expected "
        "'intermediate', 'terminal' or 'terminal-causes'. A dict from a "
        "newer SurPyval may need it: upgrade surpyval to load it."
    )


def point_time_data(data: SurpyvalData) -> SurpyvalData:
    """``data`` with its event times as a 1-D array when no row needs
    two of them.

    Interval-censored data stores ``x`` as ``[left, right]`` rows. A node
    split off such data can hold only observed and right-censored rows
    (``left == right``), yet keep the 2-D layout, which the risk-set
    code (the log-rank split, the Nelson-Aalen leaf) reads as 1-D times
    (#543). Such a node is returned with ``x`` as its rows' times, as if
    it had been given on its own; any other ``data`` is returned as is.
    """
    if np.ndim(data.x) != 2 or needs_full_likelihood_split(data):
        return data
    out = SurpyvalData(
        np.asarray(data.x)[:, 0], data.c, data.n, data.t, handle=False
    )
    Z = getattr(data, "Z", None)
    if Z is not None:
        out.add_covariates(Z)
    if data.e is not None:
        out.e = data.e
    return out


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
    competing: CauseSplit | None = None,
    splitrule: str = "searched",
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
    Turnbull leaves at a node with left- or interval-censored or
    right-truncated rows (with any truncation).

    ``rng`` draws the features considered at each split: numpy's global
    generator when ``None`` (see
    :func:`~surpyval.ml.forest.tree.resolve_random_state`).

    ``selection="greedy"`` takes the best cut of the kind's criterion over
    every drawn feature. ``selection="ctree"`` first chooses the feature
    by conditional inference (see
    :mod:`~surpyval.ml.forest.conditional_inference`), stops if its
    Bonferroni-adjusted p-value is not below ``alpha_split``, and
    otherwise cuts that feature by the kind's criterion.

    ``splitrule="searched"`` tries every value of a feature as a cut;
    ``"random"`` draws one cut a drawn feature, uniformly between its
    smallest and largest value at the node, and the split rule takes the
    best of those (see :mod:`~surpyval.ml.forest.random_cut`).

    ``min_split_gain`` is the least log-likelihood gain a deviance split
    must make (see
    :func:`~surpyval.ml.forest.deviance_split.deviance_split`).

    ``competing``, for a tree grown with the cause of each failure, gives
    the causes and the split rule (see
    :mod:`~surpyval.ml.forest.cause_specific_split`); its leaves are
    :class:`CompetingRisksLeaf`, a competing-risks model of the ``kind``.
    """
    if rng is None:
        rng = np.random.mtrand._rand

    if kind == "non-parametric":
        # A node of observed / right-censored rows split off interval
        # data takes the log-rank split and a Nelson-Aalen leaf, which
        # read 1-D times (#543).
        data = point_time_data(data)

    def leaf() -> TerminalNode:
        if competing is not None:
            return CompetingRisksLeaf(data, kind, competing.causes)
        return TerminalNode(data, kind, copy=False)

    # If max_depth has been reached, return a TerminalNode
    if curr_depth == max_depth:
        return leaf()

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
            return leaf()
        candidates = np.array([chosen])

    # Figure out best feature-value split. With random cuts each candidate
    # feature is its one cut's indicator, and the rule scores those.
    cuts: dict[int, float] | None = None
    Z_split = Z
    if splitrule == "random":
        Z_split, cuts = random_cuts(Z, candidates, rng)
    if competing is not None:
        split_feature_index, split_feature_value = competing.best_cut(
            data, Z_split, min_leaf_samples, min_leaf_failures, candidates
        )
    elif kind == "non-parametric" and needs_full_likelihood_split(data):
        # Left or interval censoring, or right truncation: the log-rank
        # scores of the pooled Turnbull estimate, less each truncated
        # row's window score.
        split_feature_index, split_feature_value = turnbull_score_split(
            data, Z_split, min_leaf_samples, min_leaf_failures, candidates
        )
    elif kind == "non-parametric":
        split_feature_index, split_feature_value = log_rank_split(
            data, Z_split, min_leaf_samples, min_leaf_failures, candidates
        )
    else:
        split_feature_index, split_feature_value = deviance_split(
            data,
            Z_split,
            min_leaf_samples,
            min_leaf_failures,
            candidates,
            model=kind,
            min_split_gain=min_split_gain,
        )

    # If the split rule can't suggest a feature-value split, return a
    # TerminalNode
    if split_feature_index == -1 and split_feature_value == float("-Inf"):
        return leaf()
    if cuts is not None:
        # The rule saw an indicator; the split is the feature's random cut
        split_feature_value = cuts[int(split_feature_index)]

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
        competing=competing,
        splitrule=splitrule,
    )
