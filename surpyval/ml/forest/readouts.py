"""What a fitted survival tree or forest says: read-outs of the model.

* :func:`partial_dependence` -- individual conditional expectation (ICE)
  and partial-dependence curves: how the predicted survival of each unit
  changes as one covariate is moved over a range of values, the others
  kept as they are, and the average of those curves.
* :func:`minimal_depth` -- how near the root a covariate first splits
  each tree, and the threshold below which it is selected (Ishwaran,
  Kogalur, Gorodeski, Minn and Lauer, 2010).
* :func:`horizon_importance` -- permutation importance of each covariate
  at each of several horizons, by the out-of-bag Brier score or AUC.

These are quantities of the fitted model; none of them refits it. The
``SurvivalTree`` and ``RandomSurvivalForest`` methods of the same names
call these functions.

References
----------
Friedman, J. H. (2001), "Greedy function approximation: a gradient
boosting machine", Annals of Statistics 29, 1189-1232.

Goldstein, A., Kapelner, A., Bleich, J. and Pitkin, E. (2015), "Peeking
inside the black box: visualizing statistical learning with plots of
individual conditional expectation", Journal of Computational and
Graphical Statistics 24, 44-65.

Ishwaran, H., Kogalur, U. B., Gorodeski, E. Z., Minn, A. J. and Lauer,
M. S. (2010), "High-dimensional variable selection for survival data",
JASA 105, 205-217.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from surpyval import plotting
from surpyval.univariate.regression.regression_data import prepare_Z
from surpyval.utils import validate_1d
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import caller_stacklevel

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: The functions a partial-dependence curve can be of.
_ON = ("sf", "ff", "Hf", "cif")
_YLABELS = {
    "sf": "Survival probability",
    "ff": "Failure probability",
    "Hf": "Cumulative hazard",
    "cif": "Cumulative incidence",
}


def _is_forest(model: Any) -> bool:
    return hasattr(model, "trees")


def _grid(model: Any, on: str, x: NDArray, Z: NDArray, event: Any) -> NDArray:
    """``on`` of ``model`` at every time ``x`` for every row of ``Z``:
    shape ``(len(Z), len(x))``."""
    if on == "cif":
        model._check_event(event)
    elif event is not None:
        raise ValueError(
            f"event is for on='cif' only; on={on!r} is of every cause"
        )
    if _is_forest(model):
        out = model._apply_model_function_to_trees(
            on, x, Z, grid=True, event=event
        )
    else:
        out = model.apply_model_function(on, x, Z, grid=True, event=event)
    return np.asarray(out, dtype=float).reshape(Z.shape[0], x.size)


def _column(model: Any, covariate: Any) -> int:
    """The column of the model's ``Z`` that ``covariate`` names."""
    labels = list(model.feature_labels)
    if isinstance(covariate, (bool, np.bool_)):
        raise ValueError(
            f"covariate must be a name or a column, not {covariate!r}"
        )
    if isinstance(covariate, (int, np.integer)):
        if not -len(labels) <= covariate < len(labels):
            raise ValueError(
                f"covariate {covariate} is not a column of the model's Z, "
                f"which has {len(labels)} columns ({labels})"
            )
        return int(covariate) % len(labels)
    if covariate in labels:
        return labels.index(covariate)
    raise ValueError(
        f"covariate {covariate!r} is not one of the model's covariates: "
        f"pass one of {labels} or a column position"
    )


def _rows(model: Any, Z: Any, weights: Any) -> tuple[NDArray, NDArray]:
    """The covariate rows the curves are for, and their weights."""
    if Z is None:
        Z_fit = getattr(model, "Z", None)
        if Z_fit is None:
            raise ValueError(
                "This model keeps no training covariates (it was restored "
                "with from_dict): pass Z, the rows to draw the curves for"
            )
        rows = np.asarray(Z_fit, dtype=float)
        if weights is None:
            weights = np.asarray(model.data.n, dtype=float)
    else:
        if isinstance(Z, pd.DataFrame):
            rows = prepare_Z(Z, model.feature_names, model._model_spec)
        else:
            rows = np.array(Z, dtype=float, ndmin=2)
        rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != len(model.feature_labels):
        raise ValueError(
            f"Z must have one column per covariate the model was grown on "
            f"({len(model.feature_labels)}: {list(model.feature_labels)})"
        )
    if weights is None:
        w = np.ones(rows.shape[0])
    else:
        w = validate_1d(weights, "weights").astype(float)
        if w.shape[0] != rows.shape[0] or (w < 0).any() or w.sum() <= 0:
            raise ValueError(
                "weights must be one non-negative number per row of Z, "
                "not all zero"
            )
    # A row with a missing covariate has no prediction; it is left out.
    keep = ~np.isnan(rows).any(axis=1)
    return rows[keep], w[keep]


@dataclass
class PartialDependence:
    """ICE and partial-dependence curves of one covariate.

    Attributes
    ----------
    covariate : str
        The covariate varied, by the model's name for it.
    values : ndarray, shape ``(n_values,)``
        The values it was set to.
    x : ndarray, shape ``(n_times,)``
        The times the model's function was evaluated at.
    on : str
        The function: ``"sf"``, ``"ff"``, ``"Hf"`` or ``"cif"``.
    event : object
        The cause, for ``on="cif"``; ``None`` otherwise.
    ice : ndarray, shape ``(n_rows, n_values, n_times)``
        ``ice[i, k, j]`` is the function at ``x[j]`` for row ``i`` with
        the covariate set to ``values[k]`` and its other covariates as
        they are: one ICE curve per row and time.
    weights : ndarray, shape ``(n_rows,)``
        The weight of each row in the average.
    average : ndarray, shape ``(n_values, n_times)``
        The partial dependence: the weighted mean of the ICE curves,
        ``average[k, j] = sum_i w_i ice[i, k, j] / sum_i w_i``.
    """

    covariate: str
    values: NDArray
    x: NDArray
    on: str
    event: Any
    ice: NDArray
    weights: NDArray
    average: NDArray

    def to_frame(self) -> pd.DataFrame:
        """The partial dependence as a table: one row per value of the
        covariate, one column per time."""
        return pd.DataFrame(
            self.average,
            index=pd.Index(self.values, name=self.covariate),
            columns=pd.Index(self.x, name="x"),
        )

    def plot(
        self,
        at: float | None = None,
        *,
        ice: bool = True,
        centred: bool = False,
        ax: Axes | None = None,
    ) -> Axes:
        """Draw the curves at one time against the covariate.

        The thick line is the partial dependence; with ``ice`` (the
        default) a thin pale line is drawn for each row, so a spread of
        shapes (an interaction with another covariate) shows. With
        ``centred``, every curve is shifted to start at 0 at the first
        value (the c-ICE plot of Goldstein et al.), which shows the
        differences in shape more clearly than the levels.

        Parameters
        ----------
        at : float, optional
            The time, one of ``x``; it may be left out when there is only
            one.
        ice, centred : bool, optional
            See above.
        ax : matplotlib axes, optional
            The axes to draw on; default ``plt.gca()``.
        """
        if at is None:
            if self.x.size != 1:
                raise ValueError(
                    f"pass at=, one of the times {self.x.tolist()}"
                )
            j = 0
        else:
            hits = np.flatnonzero(np.isclose(self.x, float(at)))
            if hits.size == 0:
                raise ValueError(
                    f"at={at!r} is not one of the times {self.x.tolist()}"
                )
            j = int(hits[0])
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()
        curves = self.ice[:, :, j]
        mean = self.average[:, j]
        if centred:
            curves = curves - curves[:, :1]
            mean = mean - mean[0]
        color = plotting.next_color(ax)
        themed = plotting.theme_enabled()
        if ice:
            for row in curves:
                ax.plot(
                    self.values, row, color=color, alpha=0.15, linewidth=0.8
                )
        ax.plot(
            self.values,
            mean,
            color=color,
            linewidth=plotting.LINE_WIDTH if themed else 2.5,
            label="Partial dependence",
        )
        ylabel = _YLABELS[self.on]
        ax.set_xlabel(self.covariate)
        ax.set_ylabel(
            ("Change in " + ylabel.lower() if centred else ylabel)
            + f" at {self.x[j]:g}"
        )
        plotting.style_axes(ax)
        plotting.style_legend(ax)
        return ax


def partial_dependence(
    model: Any,
    covariate: Any,
    values: ArrayLike,
    x: ArrayLike,
    *,
    Z: Any = None,
    weights: ArrayLike | None = None,
    on: str = "sf",
    event: Any = None,
) -> PartialDependence:
    """ICE and partial-dependence curves of ``model`` for one covariate.

    See
    :meth:`~surpyval.ml.forest.forest.RandomSurvivalForest.partial_dependence`.
    """
    check_option("on", on, _ON)
    j = _column(model, covariate)
    vals = validate_1d(values, "values").astype(float)
    if vals.size == 0 or np.isnan(vals).any():
        raise ValueError("values must be one or more numbers, none missing")
    times = validate_1d(x, "x").astype(float)
    rows, w = _rows(model, Z, weights)
    ice = np.empty((rows.shape[0], vals.size, times.size))
    for k, v in enumerate(vals):
        Zk = rows.copy()
        Zk[:, j] = v
        ice[:, k, :] = _grid(model, on, times, Zk, event)
    average = np.tensordot(w, ice, axes=(0, 0)) / w.sum()
    return PartialDependence(
        covariate=str(model.feature_labels[j]),
        values=vals,
        x=times,
        on=on,
        event=event,
        ice=ice,
        weights=w,
        average=average,
    )


def _split_depths(root: Any, n_features: int) -> tuple[NDArray, int]:
    """The depth of the shallowest split on each feature (``nan`` for a
    feature the tree never splits on) and the depth of the tree (its
    deepest leaf), the root at depth 0."""
    first = np.full(n_features, np.nan)
    deepest = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if hasattr(node, "left_child"):
            f = int(node.split_feature_index)
            if np.isnan(first[f]) or depth < first[f]:
                first[f] = depth
            stack.append((node.left_child, depth + 1))
            stack.append((node.right_child, depth + 1))
        else:
            deepest = max(deepest, depth)
    return first, deepest


def tree_minimal_depths(model: Any) -> tuple[NDArray, NDArray]:
    """Per tree, the minimal depth of each feature (with an unused one
    at the tree's depth) and whether the tree splits on it; both shape
    ``(n_trees, n_features)``."""
    trees = model.trees if _is_forest(model) else [model]
    n_features = len(model.feature_labels)
    depths = np.empty((len(trees), n_features))
    used = np.zeros((len(trees), n_features), dtype=bool)
    for b, tree in enumerate(trees):
        first, deepest = _split_depths(tree._root, n_features)
        used[b] = ~np.isnan(first)
        depths[b] = np.where(used[b], first, deepest)
    return depths, used


def _split_nodes_by_depth(root: Any) -> tuple[list[int], int]:
    """The number of split nodes at each depth, the root at depth 0, and
    the depth of the tree (its deepest leaf)."""
    counts: dict[int, int] = {}
    deepest = 0
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if hasattr(node, "left_child"):
            counts[depth] = counts.get(depth, 0) + 1
            stack.append((node.left_child, depth + 1))
            stack.append((node.right_child, depth + 1))
        else:
            deepest = max(deepest, depth)
    return [counts.get(d, 0) for d in range(deepest)], deepest


def null_minimal_depth_distribution(
    split_nodes: Any, tree_depth: int, p: Any
) -> list[Any]:
    """The distribution of the minimal depth of a covariate with no effect.

    ``split_nodes[d]`` is the number of split nodes at depth ``d`` of a
    tree, ``d = 0, ..., tree_depth - 1``; ``p`` is the number of
    covariates. Returns ``tree_depth + 1`` probabilities, for the depths
    ``0, ..., tree_depth`` (the last is "never split on", at the tree's
    depth), after Ishwaran et al. (2010):

    ``P(D = d) = (1 - 1/p)**L_d * (1 - (1 - 1/p)**l_d)`` with
    ``l_d = split_nodes[d]`` and ``L_d = l_0 + ... + l_(d-1)``.

    The arithmetic is plain Python, so a ``fractions.Fraction`` ``p`` gives
    exact probabilities.
    """
    q = 1 - 1 / p
    probs = []
    above = 0
    for d in range(tree_depth):
        probs.append(q**above * (1 - q ** int(split_nodes[d])))
        above += int(split_nodes[d])
    probs.append(1 - sum(probs))
    return probs


def null_minimal_depth_mean(split_nodes: Any, tree_depth: int, p: Any) -> Any:
    """The mean of :func:`null_minimal_depth_distribution`."""
    probs = null_minimal_depth_distribution(split_nodes, tree_depth, p)
    return sum(d * pr for d, pr in enumerate(probs))


def minimal_depth_threshold(model: Any) -> float:
    """The selection threshold for :func:`minimal_depth`.

    See :meth:`~surpyval.ml.forest.forest.RandomSurvivalForest.minimal_depth`.
    """
    trees = model.trees if _is_forest(model) else [model]
    p = len(model.feature_labels)
    means = []
    for tree in trees:
        split_nodes, deepest = _split_nodes_by_depth(tree._root)
        means.append(float(null_minimal_depth_mean(split_nodes, deepest, p)))
    return float(np.mean(means))


def minimal_depth(model: Any) -> pd.DataFrame:
    """The minimal depth of each feature.

    See :meth:`~surpyval.ml.forest.forest.RandomSurvivalForest.minimal_depth`.
    """
    depths, used = tree_minimal_depths(model)
    mean_depth = depths.mean(axis=0)
    threshold = minimal_depth_threshold(model)
    out = pd.DataFrame(
        {
            "minimal_depth": mean_depth,
            "split_fraction": used.mean(axis=0),
            "selected": mean_depth < threshold,
        },
        index=pd.Index(model.feature_labels, name="feature"),
    )
    out.attrs["threshold"] = threshold
    return out


_METRICS = ("brier", "auc")


def horizon_importance(
    forest: Any,
    times: ArrayLike,
    metric: str,
    n_repeats: int,
    rng: np.random.Generator,
    event: Any,
) -> pd.DataFrame:
    """Permutation importance by horizon, by the out-of-bag Brier score
    or AUC. See
    :meth:`~surpyval.ml.forest.forest.RandomSurvivalForest.feature_importances`.
    """
    from surpyval.metrics.validation import brier_score_at, time_dependent_auc

    check_option("metric", metric, _METRICS)
    horizons = validate_1d(times, "times").astype(float)
    if horizons.size == 0 or not np.isfinite(horizons).all():
        raise ValueError("times must be one or more finite horizons")
    data = forest.data
    c = np.asarray(data.c)
    if (
        not np.isin(c, (0, 1)).all()
        or np.isfinite(data.tl).any()
        or (np.isfinite(data.tr).any())
    ):
        raise ValueError(
            "Importance by horizon (times=) scores the out-of-bag Brier "
            "score or AUC, which need observed and right-censored times "
            "without truncation; this forest's data have left or interval "
            "censoring or truncation. Leave times out for the out-of-bag "
            "log-likelihood importance, which takes every kind of data."
        )
    on = "sf" if event is None else "cif"
    if event is not None:
        forest._check_event(event)
    with warnings.catch_warnings():
        # Its warning speaks of the log-likelihood; this one is ours.
        warnings.simplefilter("ignore", UserWarning)
        oob, _, _, n_oob = forest._oob_setup()
    scored = n_oob > 0
    if not scored.all():
        warnings.warn(
            f"{int((~scored).sum())} of {scored.size} rows were in the "
            "sample of every tree, so no tree can score them out of bag; "
            "they are left out of the out-of-bag Brier score or AUC (grow "
            "more trees to cover them).",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    reps = np.asarray(data.n, dtype=int)[scored]
    x_rows = np.repeat(np.asarray(data.x, dtype=float)[scored], reps)
    c_rows = np.repeat(c[scored], reps)
    e_rows = None if event is None else np.repeat(data.e[scored], reps)
    cause = event

    def predictions(permute: int | None) -> NDArray:
        # The out-of-bag ensemble prediction of every row at every
        # horizon, with feature ``permute`` shuffled among each tree's
        # out-of-bag rows as for the log-likelihood importance.
        total = np.zeros((len(data), horizons.size))
        for tree, rows in zip(forest.trees, oob):
            if rows.size == 0:
                continue
            Z = forest.Z[rows]
            if permute is not None:
                Z = Z.copy()
                Z[:, permute] = Z[rng.permutation(rows.size), permute]
            total[rows] += _grid(tree, on, horizons, Z, event)
        pred = total[scored] / n_oob[scored, None]
        return np.repeat(pred, reps, axis=0)

    def score(pred: NDArray) -> NDArray:
        if metric == "brier":
            return np.asarray(
                brier_score_at(
                    x_rows, c_rows, pred, horizons, None, None, e_rows, cause
                )[1],
                dtype=float,
            )
        risk = pred if event is not None else 1.0 - pred
        return np.asarray(
            time_dependent_auc(
                x_rows,
                c_rows,
                risk,
                horizons,
                None,
                None,
                e_rows,
                cause,
                "cause_free",
            )[1],
            dtype=float,
        )

    base = score(predictions(None))
    out = np.zeros((forest.Z.shape[1], horizons.size))
    for j in range(forest.Z.shape[1]):
        drops = []
        for _ in range(int(n_repeats)):
            shuffled = score(predictions(j))
            # Positive for a feature the forest relies on, for both
            # metrics: the Brier score rises, the AUC falls.
            drops.append(
                shuffled - base if metric == "brier" else base - shuffled
            )
        out[j] = np.mean(drops, axis=0)
    return pd.DataFrame(
        out,
        index=pd.Index(forest.feature_labels, name="feature"),
        columns=pd.Index(horizons, name="time"),
    )
