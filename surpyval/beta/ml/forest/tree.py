from copy import deepcopy
from math import log2, sqrt
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.conditional_inference import parse_selection
from surpyval.beta.ml.forest.deviance_split import parse_min_split_gain
from surpyval.beta.ml.forest.node import (
    IntermediateNode,
    Node,
    build_tree,
    fit_leaves,
    node_from_dict,
    route_to_leaves,
    tree_lines,
)
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.parametric import Exponential, Weibull
from surpyval.univariate.regression.regression_data import (
    check_finite_event_times,
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from surpyval.utils import check_covariate_rows, finite_covariate_mask
from surpyval.utils.dataframe import RegressionDataFrameMixin
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import check_paired_rows, flatten_query
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import option_error

Random = np.random.Generator | np.random.RandomState


def query_layout(n_x: int, Z_ndim: int, n_rows: int, grid: bool) -> str:
    """How a tree's or forest's ``sf(x, Z)`` (and ``ff``, ``df``, ``hf``,
    ``Hf``) lays out its result (#666): ``"grid"``, every time for every
    row; ``"single"``, one covariate vector (a 1-D ``Z``) at every time;
    ``"paired"``, row ``i`` of ``Z`` with ``x[i]``; ``"row"``, a single
    row at every time; ``"time"``, a single time for every row.

    By default rows are paired with times as every regression model
    pairs them, and counts that cannot be paired are refused with the
    regression models' message; ``grid=True`` is the grid (a 1-D ``Z``
    its one row).
    """
    if grid:
        return "grid"
    if Z_ndim < 2:
        return "single"
    if n_rows == n_x:
        return "paired"
    if n_rows == 1:
        return "row"
    if n_x == 1:
        return "time"
    check_paired_rows(n_x, n_rows)
    return "grid"  # not reached: check_paired_rows raised


def resolve_random_state(random_state: Any = None) -> Random:
    """The random stream a tree or forest draws from.

    ``None`` is numpy's global generator itself, drawn from directly, so
    ``np.random.seed`` reproduces a fit exactly as it did before trees and
    forests took a ``random_state``. Anything else goes through
    :func:`~surpyval.utils.rng.as_generator`: a seed or ``Generator``
    gives a stream of its own, which neither depends on nor advances the
    global one.
    """
    if random_state is None:
        return np.random.mtrand._rand
    return as_generator(random_state)


def covariate_matrix(
    Z: "ArrayLike | NDArray | pd.DataFrame",
    feature_names: "list[str] | None" = None,
) -> "tuple[ArrayLike | NDArray, list[str] | None]":
    """The covariate matrix and its feature names.

    A DataFrame ``Z`` gives its values and its column names (unless
    ``feature_names`` is given); any other ``Z`` is returned as it is,
    with ``feature_names`` (``None`` when not given: a tree fitted from
    an array has no feature names, as a regression model fitted from one
    has none).
    """
    if isinstance(Z, pd.DataFrame):
        if feature_names is None:
            feature_names = [str(column) for column in Z.columns]
        Z = Z.to_numpy(dtype=float)
    if feature_names is not None:
        feature_names = [str(name) for name in feature_names]
        n_columns = np.shape(Z)[1] if np.ndim(Z) == 2 else 1
        if len(feature_names) != n_columns:
            raise ValueError(
                f"feature_names has {len(feature_names)} names but Z has "
                f"{n_columns} columns"
            )
    return Z, feature_names


def feature_labels(
    feature_names: "list[str] | None", n_features: int
) -> list[str]:
    """The name of each feature for display: its ``feature_names`` where
    the model has them, else ``Z0``, ``Z1``, ... (the column of ``Z``)."""
    if feature_names is not None:
        return list(feature_names)
    return [f"Z{j}" for j in range(n_features)]


def check_covariate_count(
    Z: NDArray,
    n_fitted: "int | None",
    labels: "list[str]",
    what: str,
    at_least: int = 0,
) -> None:
    """Refuse covariate vectors whose length is not the number the model
    was fitted with (#657): an extra column was ignored, and a missing one
    raised numpy's IndexError, so a column-order or width mistake gave
    plausible, wrong predictions. ``Z`` is one vector (1-D) or one per row
    (2-D); ``n_fitted`` is the fitted count (``None`` for a model restored
    from a dict saved without it, which is then checked only to have the
    ``at_least`` columns its splits read)."""
    if Z.ndim not in (1, 2):
        return
    got = Z.shape[-1]
    if n_fitted is None:
        if got >= at_least:
            return
        expected = f"at least {at_least} covariates"
    elif got == n_fitted:
        return
    else:
        names = f" ({', '.join(labels)})" if labels else ""
        expected = f"{n_fitted} covariate{'s' * (n_fitted != 1)}{names}"
    unit = "value" if Z.ndim == 1 else "column"
    raise ValueError(
        f"The {what} has {expected}; got {got} {unit}{'s' * (got != 1)} "
        "in Z. Pass one value per covariate, in the order the model was "
        "fitted with (or a DataFrame with those columns)."
    )


def drop_missing_covariate_rows(
    data: SurpyvalData, Z: ArrayLike | NDArray
) -> tuple[SurpyvalData, NDArray]:
    """Pair ``Z`` with the data and drop the rows with a missing (NaN) or
    infinite covariate from both, with one warning giving the count.

    A NaN compares false with every split value, so such rows used to be
    sent down the right-hand branch of every split on their missing feature
    and kept in the fit without a word.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        # A 1-d Z is a single feature, one value per sample
        Z = Z.reshape(-1, 1)
    check_covariate_rows(Z, len(data))
    mask = finite_covariate_mask(Z)
    if mask.all():
        return data, Z
    return data[mask], Z[mask]


class SurvivalTree(RegressionDataFrameMixin, SerialisableMixin):
    """
    A Survival Tree, for use in `RandomSurvivalForest`.

    The Tree is built on initialisation. Supports the full SurPyval data
    model: observed, left-, right- and interval-censored observations
    with optional left and/or right truncation.

    The tree's ``kind`` couples the split criterion with the matching
    leaf model, so every split greedily improves the model the tree
    predicts with:

    - ``"weibull"`` (default): full-likelihood Weibull deviance split
      (a 2-d.f. likelihood-ratio gain, with power against scale *and*
      shape differences) with Weibull MLE leaves. Supports the full
      data model.
    - ``"exponential"``: exponential deviance split (Davis & Anderson,
      1989; 1-d.f., splits on rate) with Exponential MLE leaves.
      Supports the full data model.
    - ``"non-parametric"``: for observed / right-censored data
      (optionally left-truncated), the risk-set log-rank split with
      Nelson-Aalen leaves. For data with left or interval censoring or
      right truncation, the Turnbull-score split -- the standardised
      sum of each child's log-rank scores under the node's pooled
      Turnbull estimate (Finkelstein, 1986), which reduces to the
      log-rank scores on right-censored data; a truncated row's score
      is that of its truncation-conditioned likelihood -- with
      Turnbull leaves. Supports the full data model.

    ``selection`` decides how a node chooses the feature it splits on:

    - ``"greedy"`` (default): the best cut of the kind's criterion over
      every feature drawn for the split. A feature with many distinct
      values offers more cuts, so it is favoured even when it carries no
      information, and a node always splits if some cut is allowed.
    - ``"ctree"``: conditional inference (Hothorn, Hornik and Zeileis,
      2006). Each feature is tested for association with the scores of
      the kind's split statistic (the log-rank scores for
      ``"non-parametric"``; the working model's score contributions for
      ``"exponential"`` and ``"weibull"``), by its maximally selected
      statistic over its cuts, whose p-value allows for the number of
      cuts. The feature with the smallest p-value is chosen, and the node
      splits only if that p-value, Bonferroni-adjusted for the number of
      features tested, is below ``alpha_split``; its cut is then chosen by
      the kind's criterion. This removes the preference for features with
      many values and stops the tree where the data show no effect. See
      :mod:`~surpyval.beta.ml.forest.conditional_inference`.

    Predictions take one value per covariate the tree was grown on, in
    its column order (or a DataFrame with its columns); another number
    raises a ``ValueError`` (#657).
    """

    #: The covariate count a restored tree's :meth:`to_dict` saved.
    _saved_n_covariates: "int | None" = None

    def __init__(
        self,
        data: SurpyvalData,
        Z: NDArray,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        kind: str = "weibull",
        selection: str = "greedy",
        alpha_split: float = 0.05,
        random_state: Any = None,
        feature_names: list[str] | None = None,
        min_split_gain: float | str = 0.0,
    ) -> None:
        self.selection = parse_selection(selection, alpha_split)
        self.alpha_split = float(alpha_split)
        Z_in, self.feature_names = covariate_matrix(Z, feature_names)
        # Set by ``fit_from_df(formula=...)``: the formula and its
        # design-matrix transformer, to expand a DataFrame at prediction.
        self.formula: str | None = None
        self._model_spec: Any = None
        self.data, self.Z = drop_missing_covariate_rows(data, Z_in)
        if self.data is data:
            # The leaves keep the rows they are given without copying
            # them, so the tree holds its own copy of the caller's data.
            self.data = deepcopy(data)

        n_features: int = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self.n_features_split = n_features

        self.kind = parse_kind(kind, self.data)
        self.min_split_gain = parse_min_split_gain(min_split_gain, self.kind)

        self._root = build_tree(
            data=self.data,
            Z=self.Z,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features,
            kind=self.kind,
            rng=resolve_random_state(random_state),
            selection=self.selection,
            alpha_split=self.alpha_split,
            min_split_gain=self.min_split_gain,
        )
        # The parametric leaves all at once, rather than one by one on
        # first use (#549)
        fit_leaves(self._root)

    @classmethod
    def fit(
        cls,
        x: ArrayLike | None = None,
        Z: ArrayLike | NDArray | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        t: ArrayLike | None = None,
        xl: ArrayLike | None = None,
        xr: ArrayLike | None = None,
        tl: ArrayLike | None = None,
        tr: ArrayLike | None = None,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        kind: str = "weibull",
        selection: str = "greedy",
        alpha_split: float = 0.05,
        min_split_gain: float | str = 0.0,
        random_state: Any = None,
    ) -> "SurvivalTree":
        """
        Fit a survival tree from data in the full xcnt(+truncation) data
        model.

        ``x``/``c``/``n``/``t`` follow the standard SurPyval conventions
        (``c`` in ``{-1, 0, 1, 2}``; interval-censored entries of ``x``
        are ``[left, right]`` pairs). Interval bounds can alternatively
        be given as ``xl``/``xr``, and truncation as ``tl``/``tr``
        instead of the two-column ``t``. ``kind`` selects the tree type
        (see the class docstring).

        Parameters
        ----------
        x : array_like, optional
            Event times (``[left, right]`` rows for interval-censored
            observations). A ``"weibull"`` or ``"exponential"`` tree
            refuses a time outside its leaves' support ``(0, inf)``, as
            the parametric regression fits do; a ``"non-parametric"`` one
            refuses a failure (``c=0``) at infinity.
        Z : array_like
            Covariate (feature) matrix, one row per observation. Required.
            Rows with a missing (NaN) or infinite covariate are dropped,
            with a warning giving the count.
        c : array_like, optional
            Censoring flags: 0 observed, 1 right, -1 left, 2 interval
            censored. Defaults to all observed.
        n : array_like, optional
            Counts. Defaults to 1.
        t : array_like, optional
            (N, 2) truncation bounds.
        xl, xr : array_like, optional
            Interval bounds, instead of 2-D ``x``.
        tl, tr : array_like, optional
            Left and right truncation, instead of ``t``.
        max_depth : int, optional
            Maximum depth of a tree. Defaults to unlimited.
        min_leaf_samples : int, optional
            A split is only made if each child keeps at least this many
            observations. Defaults to 5.
        min_leaf_failures : int, optional
            ... and at least this many failures (rows that are not
            right censored, each counted ``n`` times). Defaults to 2.
        n_features_split : int, float or str, optional
            The number of features considered at each split: an int, a
            fraction of the features (float), ``"sqrt"`` (the default),
            ``"log2"`` or ``"all"``.
        kind : str, optional
            ``"weibull"`` (the default), ``"exponential"`` or
            ``"non-parametric"``; see the class docstring.
        selection : str, optional
            How a node chooses its feature: ``"greedy"`` (the default),
            the best cut over every feature, or ``"ctree"``, conditional
            inference; see the class docstring.
        alpha_split : float, optional
            With ``selection="ctree"``, a node splits only if the
            Bonferroni-adjusted p-value of its chosen feature is below
            ``alpha_split``, the size of the test of no association.
            Defaults to 0.05. Ignored by ``"greedy"``.
        min_split_gain : float, "aic" or "bic", optional
            The least gain in log-likelihood a split of a ``"weibull"`` or
            ``"exponential"`` tree must make: a node splits only if its
            best cut raises the maximised log-likelihood of its working
            model by more than this (the two children's against the
            node's). ``"aic"`` is the kind's degrees of freedom ``k`` (1
            for ``"exponential"``, 2 for ``"weibull"``): the split must
            lower Akaike's criterion. ``"bic"`` is ``k log(d) / 2``, with
            ``d`` the node's failures (rows not right censored, counted
            ``n`` times; its units if it has none), as every BIC in
            SurPyval counts them: the split must lower the Bayesian
            criterion. Defaults to 0: any gain, as a forest of deep trees
            wants. ``"aic"`` is the recommended setting for a tree used
            on its own, which otherwise splits on noise until
            ``min_leaf_samples`` or ``min_leaf_failures`` stops it. Not
            used by ``"non-parametric"`` trees, whose splits are not
            likelihoods; stop those with ``selection="ctree"``.
        random_state : None, int or numpy.random.Generator, optional
            Seeds the features drawn for each split (when
            ``n_features_split`` is less than the number of features).
            ``None`` (the default) draws from NumPy's global random state,
            so ``np.random.seed`` reproduces the tree; a seed or
            ``Generator`` gives a stream of its own and leaves the global
            one alone.

        Returns
        -------
        SurvivalTree
            The fitted tree. Its ``sf(x, Z)`` (and ``ff``, ``df``, ``hf``,
            ``Hf``) evaluate the model of the leaf that a covariate vector
            ``Z`` falls in; a matrix ``Z`` pairs row ``i`` with time
            ``x[i]``, as every regression model does, and ``grid=True``
            gives one row per covariate vector and one column per time.
            A covariate vector with a missing (NaN) value gives NaN.

        Examples
        --------
        Life halves when the first feature exceeds 0.5; a single split
        finds it:

        >>> import numpy as np
        >>> from surpyval.beta.ml import SurvivalTree
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 2))
        >>> x = rng.weibull(2.0, 200) * np.where(Z[:, 0] > 0.5, 5.0, 10.0)
        >>> c = (x > 12).astype(int)
        >>> x = np.minimum(x, 12)
        >>> tree = SurvivalTree.fit(x, Z, c=c, max_depth=1, n_features_split=2)
        >>> tree.sf(5, [0.2, 0.5]).round(4), tree.sf(5, [0.8, 0.5]).round(4)
        (np.float64(0.8831), np.float64(0.3168))

        A matrix routes each row to its own leaf and pairs row ``i`` with
        time ``i``, as a regression model does:

        >>> tree.sf([2, 5], [[0.2, 0.5], [0.8, 0.5]]).round(4)
        array([0.9897, 0.3168])

        A single time is used for every row, and a single row at every
        time:

        >>> tree.sf(5, [[0.2, 0.5], [0.8, 0.5]]).round(4)
        array([0.8831, 0.3168])
        >>> tree.sf([2, 5], [[0.8, 0.5]]).round(4)
        array([0.8062, 0.3168])

        ``grid=True`` gives every time for every row, a survival curve per
        covariate vector:

        >>> tree.sf([2, 5], [[0.2, 0.5], [0.8, 0.5]], grid=True).round(4)
        array([[0.9897, 0.8831],
               [0.8062, 0.3168]])

        With conditional-inference selection, a tree grown on the same
        data without the effect does not split at all, where greedy
        search always does:

        >>> x0 = rng.weibull(2.0, 200) * 10.0
        >>> ctree = SurvivalTree.fit(
        ...     x0, Z, kind="non-parametric", n_features_split="all",
        ...     selection="ctree",
        ... )
        >>> type(ctree._root).__name__
        'TerminalNode'
        >>> ctree = SurvivalTree.fit(
        ...     x, Z, c=c, kind="non-parametric", n_features_split="all",
        ...     selection="ctree", max_depth=1,
        ... )
        >>> root = ctree._root
        >>> int(root.split_feature_index), bool(root.p_value < 1e-10)
        (0, True)
        """
        if Z is None:
            raise ValueError("The covariate matrix Z is required")
        data = SurpyvalData(
            x, c, n, t, xl=xl, xr=xr, tl=tl, tr=tr, group_and_sort=False
        )
        Z, feature_names = covariate_matrix(Z)
        Z = np.asarray(Z)
        if Z.ndim == 1:
            # A 1-d Z is a single feature, one value per sample
            Z = Z.reshape(-1, 1)
        return cls(
            data,
            Z,
            max_depth,
            min_leaf_samples,
            min_leaf_failures,
            n_features_split,
            kind,
            selection,
            alpha_split,
            random_state,
            feature_names,
            min_split_gain,
        )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Evaluate ``function_name`` (``"sf"``, ``"ff"``, ``"df"``, ``"hf"``
        or ``"Hf"``) of the leaf model that each covariate vector falls in.

        Parameters
        ----------
        function_name : str
            The name of the leaf model's function to evaluate.
        x : int, float or array_like
            Times.
        Z : array_like
            One covariate vector (1-D), or a matrix with one covariate
            vector per row (2-D).
        grid : bool, optional
            ``False`` (the default) pairs row ``i`` of ``Z`` with ``x[i]``
            (a single row is used at every time, a single time for every
            row), as every regression model does (#666); other counts of
            rows and times are refused with a ``ValueError``. ``True``
            evaluates every time for every row of ``Z`` (a 1-D ``Z`` is
            one row).

        Returns
        -------
        ndarray
            For a 1-D ``Z`` (and no ``grid``), the values at ``x``, shaped
            like ``x`` (a scalar for a scalar ``x``). Paired, the shape of
            ``x`` (or ``(n_rows,)`` for a single time). On the grid, shape
            ``(n_rows,) + x.shape``, row ``i`` the values for ``Z[i]``, as
            for :class:`~surpyval.beta.ml.forest.forest.RandomSurvivalForest`.
            A covariate vector with a missing (NaN) value gives NaN, and
            leaves the other rows unaffected.
        """
        # The times flat; the result gets their shape back (on its last
        # axis for a grid), so a scalar time gives a scalar.
        x, restore = flatten_query(x)
        Z = self._covariates(Z)
        layout = query_layout(
            x.size, Z.ndim, Z.shape[0] if Z.ndim == 2 else 1, grid
        )
        if layout == "paired":
            return restore(self._apply_flat(function_name, x, Z, True))
        if layout == "grid" and Z.ndim == 1:
            Z = Z[None, :]
        res = self._apply_flat(function_name, x, Z)
        if layout == "row":
            return restore(res[0])
        if layout == "time":
            return res[:, 0]
        return restore(res, axis=-1)

    def _covariates(self, Z: "ArrayLike | NDArray | pd.DataFrame") -> NDArray:
        # A DataFrame is read by the names the tree was fitted with (or
        # expanded by its formula), so its columns may be in any order;
        # anything else is read in the fitted column order.
        if isinstance(Z, pd.DataFrame):
            return prepare_Z(Z, self.feature_names, self._model_spec)
        return np.array(Z, ndmin=1, dtype=float)

    def _n_covariates(self) -> "int | None":
        """The number of covariates the tree was grown on: the columns of
        its ``Z``, or as saved by :meth:`to_dict`; ``None`` for a tree
        restored from a dict saved without it and without names."""
        if getattr(self, "Z", None) is not None:
            return int(np.shape(self.Z)[1])
        saved = getattr(self, "_saved_n_covariates", None)
        if saved is not None:
            return int(saved)
        if self.feature_names is not None:
            return len(self.feature_names)
        return None

    def _apply_flat(
        self,
        function_name: str,
        x: NDArray,
        Z: ArrayLike | NDArray,
        paired: bool = False,
    ) -> NDArray:
        # ``apply_model_function`` at a 1-D array of times: on the grid of
        # every time for every row, or, ``paired``, row i at time x[i].
        Z = self._covariates(Z)
        if Z.ndim > 2:
            raise ValueError(
                f"Z must be one covariate vector (1-D) or one per row "
                f"(2-D), got {Z.ndim} dimensions"
            )
        n_fitted = self._n_covariates()
        check_covariate_count(
            Z,
            n_fitted,
            self.feature_labels if n_fitted is not None else [],
            "tree",
            at_least=_n_features(self._root, None),
        )

        # A NaN compares false with every split value, so it used to be
        # routed right at every split on its feature and given a number.
        # A covariate vector with a missing value has no leaf: NaN out.
        if Z.ndim == 1:
            if np.isnan(Z).any():
                return np.full(x.shape, np.nan)
            return self._root.apply_model_function(function_name, x, Z)
        missing = np.isnan(Z).any(axis=1)
        if paired:
            # Each row's leaf at its own time (#666): the rows grouped by
            # the leaf they reach, each leaf evaluated once.
            out = np.full(x.size, np.nan)
            rows = np.flatnonzero(~missing)
            for leaf, idx in route_to_leaves(self._root, Z[rows]):
                sel = rows[idx]
                out[sel] = np.asarray(
                    getattr(leaf.model, function_name)(x[sel]), dtype=float
                )
            return out
        if not missing.any():
            return self._root.apply_model_function(function_name, x, Z)
        res = np.full((Z.shape[0], x.size), np.nan)
        if not missing.all():
            res[~missing] = self._root.apply_model_function(
                function_name, x, Z[~missing]
            )
        return res

    def sf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Survival function at ``x`` of the leaf model each covariate vector
        falls in; ``Z``, ``grid`` and the result are as for
        :meth:`apply_model_function` (a 2-D ``Z`` pairs rows with times;
        ``grid=True`` gives one row per covariate vector).
        """
        return self.apply_model_function("sf", x, Z, grid=grid)

    def ff(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Failure (CDF) function at ``x`` of the leaf model each covariate
        vector falls in, as for :meth:`sf`.
        """
        return self.apply_model_function("ff", x, Z, grid=grid)

    def df(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Density at ``x`` of the leaf model each covariate vector falls in,
        as for :meth:`sf`.
        """
        return self.apply_model_function("df", x, Z, grid=grid)

    def hf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Hazard rate at ``x`` of the leaf model each covariate vector falls
        in, as for :meth:`sf`.
        """
        return self.apply_model_function("hf", x, Z, grid=grid)

    def Hf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Cumulative hazard at ``x`` of the leaf model each covariate vector
        falls in, as for :meth:`sf`.
        """
        return self.apply_model_function("Hf", x, Z, grid=grid)

    def to_dict(self) -> dict:
        """Serialise the fitted tree to a plain, JSON/BSON-safe dictionary.

        Only what prediction needs is stored -- the tree ``kind``, the
        resolved ``n_features_split``, the ``selection`` and
        ``alpha_split`` it was grown with, and the recursive node structure
        with its fitted leaf models (and, for ``selection="ctree"``, each
        split's p-value). The training data and covariate matrix are not
        persisted: a restored tree is a predictor, not a re-fittable
        object.
        """
        out = {
            "model": "SurvivalTree",
            "kind": self.kind,
            "n_features_split": int(self.n_features_split),
            "selection": self.selection,
            "alpha_split": float(self.alpha_split),
            "min_split_gain": self.min_split_gain,
            "root": self._root.to_dict(),
        }
        n_covariates = self._n_covariates()
        if n_covariates is not None:
            out["n_covariates"] = n_covariates
        serialise_covariate_meta(self, out)
        # The leaves are finished model dictionaries already (#549)
        return stamp_schema(out, stamped=True)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "SurvivalTree":
        """Reconstruct a fitted tree from a :meth:`to_dict` dictionary."""
        require_model_tag(model_dict, "SurvivalTree", "a survival tree")
        tree = cls.__new__(cls)
        tree.kind = model_dict["kind"]
        tree.n_features_split = model_dict["n_features_split"]
        # Trees saved before selection existed were grown greedily.
        tree.selection = model_dict.get("selection", "greedy")
        tree.alpha_split = model_dict.get("alpha_split", 0.05)
        tree.min_split_gain = model_dict.get("min_split_gain", 0.0)
        # A restored tree predicts but is not re-fittable; it holds no data.
        tree.data = None  # type: ignore[assignment]
        tree.Z = None  # type: ignore[assignment]
        tree._model_spec = None
        # The covariate count predictions are checked against (#657);
        # trees saved before it was stored have none.
        tree._saved_n_covariates = model_dict.get("n_covariates")
        # Trees saved before feature names existed have none.
        restore_covariate_meta(tree, model_dict)
        tree._root = node_from_dict(model_dict["root"])
        return tree

    @property
    def feature_labels(self) -> list[str]:
        """The name of each feature: ``feature_names`` for a tree fitted
        from a DataFrame (``fit_from_df``, or ``fit`` with a DataFrame
        ``Z``), else ``Z0``, ``Z1``, ... by column of ``Z``. Split
        descriptions and the printout use them."""
        if self.feature_names is not None:
            return list(self.feature_names)
        return feature_labels(None, _n_features(self._root, self.Z))

    def describe(self) -> str:
        """The tree as text: one line per split, ``name <= value`` for
        the left branch and ``name >  value`` for the right one (with the
        adjusted p-value of a ``selection="ctree"`` split), each branch's
        subtree indented under it, and each leaf's model."""
        lines = [
            f"SurvivalTree(kind={self.kind!r}, "
            f"selection={self.selection!r})"
        ]
        lines += tree_lines(self._root, self.feature_labels)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()


def _n_features(root: Node, Z: NDArray | None) -> int:
    # The number of features a tree was grown on: the columns of its Z,
    # or for a restored tree (which keeps no Z) one past the largest
    # feature its splits use or drew.
    if Z is not None:
        return int(np.shape(Z)[1])
    largest = -1
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, IntermediateNode):
            drawn = np.asarray(node.feature_indices_in, dtype=int)
            largest = max(
                largest,
                int(node.split_feature_index),
                int(drawn.max()) if drawn.size else -1,
            )
            stack += [node.left_child, node.right_child]
    return largest + 1


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
        raise ValueError(f"n_features_split={n_features_split} is invalid. See\
                         `Tree` docstring for valid values.")


def parse_kind(kind: str, data: SurpyvalData) -> str:
    """
    Resolve and validate the tree ``kind`` against the data.

    Every kind supports the full data model. The non-parametric kind
    splits observed and right-censored data (optionally left truncated)
    by the risk-set log-rank, and data with left or interval censoring or
    right truncation by the Turnbull scores, which allow for truncation
    through the truncation-conditioned likelihood (issue #188).

    The times are checked as the fits the kind's leaves are (#618): a
    ``"weibull"`` or ``"exponential"`` tree refuses a time outside its
    distribution's support, with the parametric regression fits' check
    and wording (``OutsideSupportError``: a failure at 0 or at infinity,
    a time below 0 whatever its censoring); a ``"non-parametric"`` tree
    refuses a failure at infinity, as the non-parametric and Cox fits do.
    An infinite time used to reach the leaf fits: a Weibull tree gave
    hundreds of numpy warnings, an Exponential one a scipy error.
    """
    resolved = kind.lower().replace("_", "-")
    if resolved not in ("weibull", "exponential", "non-parametric"):
        raise option_error(
            "kind",
            kind,
            ("weibull", "exponential", "non-parametric"),
            "Case does not matter, and '_' may stand for '-'.",
        )
    if resolved == "non-parametric":
        check_finite_event_times(data.x, data.c)
    else:
        leaf = Weibull if resolved == "weibull" else Exponential
        leaf._check_inside_support(data, every_row=True)
    return resolved
