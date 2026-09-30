from math import log2, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.node import build_tree, node_from_dict
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils import check_covariate_rows, finite_covariate_mask
from surpyval.utils.shapes import flatten_query
from surpyval.utils.surpyval_data import SurpyvalData


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


class SurvivalTree(SerialisableMixin):
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
      Nelson-Aalen leaves. For data with left or interval censoring,
      the Turnbull-score split -- the standardised sum of each child's
      log-rank scores under the node's pooled Turnbull estimate
      (Finkelstein, 1986), which reduces to the log-rank scores on
      right-censored data -- with Turnbull leaves. Raises
      ``ValueError`` for right truncation, or for truncation together
      with left or interval censoring: the scores would have to come
      from the truncation-conditioned likelihood (issue #188).
    """

    def __init__(
        self,
        data: SurpyvalData,
        Z: NDArray,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        kind: str = "weibull",
    ) -> None:
        self.data, self.Z = drop_missing_covariate_rows(data, Z)

        n_features: int = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self.n_features_split = n_features

        self.kind = parse_kind(kind, self.data)

        self._root = build_tree(
            data=self.data,
            Z=self.Z,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features,
            kind=self.kind,
        )

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
            observations).
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
            ... and at least this many failures. Defaults to 2.
        n_features_split : int, float or str, optional
            The number of features considered at each split: an int, a
            fraction of the features (float), ``"sqrt"`` (the default),
            ``"log2"`` or ``"all"``.
        kind : str, optional
            ``"weibull"`` (the default), ``"exponential"`` or
            ``"non-parametric"``; see the class docstring.

        Returns
        -------
        SurvivalTree
            The fitted tree. Its ``sf(x, Z)`` (and ``ff``, ``df``, ``hf``,
            ``Hf``) evaluate the model of the leaf that a covariate vector
            ``Z`` falls in; a matrix ``Z`` gives one row per covariate
            vector and one column per time. A covariate vector with a
            missing (NaN) value gives NaN.

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

        A matrix routes each row to its own leaf:

        >>> tree.sf([2, 5], [[0.2, 0.5], [0.8, 0.5]]).round(4)
        array([[0.9897, 0.8831],
               [0.8062, 0.3168]])
        """
        if Z is None:
            raise ValueError("The covariate matrix Z is required")
        data = SurpyvalData(
            x, c, n, t, xl=xl, xr=xr, tl=tl, tr=tr, group_and_sort=False
        )
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
        )

    def apply_model_function(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
    ) -> NDArray:
        """
        Evaluate ``function_name`` (``"sf"``, ``"ff"``, ``"df"``, ``"hf"``
        or ``"Hf"``) of the leaf model that each covariate vector falls in.

        Parameters
        ----------
        function_name : str
            The name of the leaf model's function to evaluate.
        x : int, float or array_like
            Times, the same for every covariate vector.
        Z : array_like
            One covariate vector (1-D), or a matrix with one covariate
            vector per row (2-D).

        Returns
        -------
        ndarray
            For a 1-D ``Z``, the values at ``x``, shaped like ``x`` (a
            scalar for a scalar ``x``). For a 2-D ``Z``, a grid of shape
            ``(n_rows,) + x.shape`` whose row ``i`` is the values for
            ``Z[i]`` -- every row at every time, the one documented
            exception to pairing rows with times -- as for
            :class:`~surpyval.beta.ml.forest.forest.RandomSurvivalForest`.
            A covariate vector with a missing (NaN) value gives NaN, and
            leaves the other rows unaffected.
        """
        # The times flat; the result gets their shape back (on its last
        # axis for a grid), so a scalar time gives a scalar.
        x, restore = flatten_query(x)
        return restore(self._apply_flat(function_name, x, Z), axis=-1)

    def _apply_flat(
        self, function_name: str, x: NDArray, Z: ArrayLike | NDArray
    ) -> NDArray:
        # ``apply_model_function`` at a 1-D array of times.
        Z = np.array(Z, ndmin=1, dtype=float)
        if Z.ndim > 2:
            raise ValueError(
                f"Z must be one covariate vector (1-D) or one per row "
                f"(2-D), got {Z.ndim} dimensions"
            )

        # A NaN compares false with every split value, so it used to be
        # routed right at every split on its feature and given a number.
        # A covariate vector with a missing value has no leaf: NaN out.
        if Z.ndim == 1:
            if np.isnan(Z).any():
                return np.full(x.shape, np.nan)
            return self._root.apply_model_function(function_name, x, Z)
        missing = np.isnan(Z).any(axis=1)
        if not missing.any():
            return self._root.apply_model_function(function_name, x, Z)
        res = np.full((Z.shape[0], x.size), np.nan)
        if not missing.all():
            res[~missing] = self._root.apply_model_function(
                function_name, x, Z[~missing]
            )
        return res

    def sf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Survival function at ``x`` of the leaf model each covariate vector
        falls in; ``Z`` and the result are as for
        :meth:`apply_model_function` (a 2-D ``Z`` gives one row per
        covariate vector).
        """
        return self.apply_model_function("sf", x, Z)

    def ff(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Failure (CDF) function at ``x`` of the leaf model each covariate
        vector falls in, as for :meth:`sf`.
        """
        return self.apply_model_function("ff", x, Z)

    def df(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Density at ``x`` of the leaf model each covariate vector falls in,
        as for :meth:`sf`.
        """
        return self.apply_model_function("df", x, Z)

    def hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Hazard rate at ``x`` of the leaf model each covariate vector falls
        in, as for :meth:`sf`.
        """
        return self.apply_model_function("hf", x, Z)

    def Hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Cumulative hazard at ``x`` of the leaf model each covariate vector
        falls in, as for :meth:`sf`.
        """
        return self.apply_model_function("Hf", x, Z)

    def to_dict(self) -> dict:
        """Serialise the fitted tree to a plain, JSON/BSON-safe dictionary.

        Only what prediction needs is stored -- the tree ``kind``, the
        resolved ``n_features_split`` and the recursive node structure with
        its fitted leaf models. The training data and covariate matrix are
        not persisted: a restored tree is a predictor, not a re-fittable
        object.
        """
        return stamp_schema(
            {
                "model": "SurvivalTree",
                "kind": self.kind,
                "n_features_split": int(self.n_features_split),
                "root": self._root.to_dict(),
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> "SurvivalTree":
        """Reconstruct a fitted tree from a :meth:`to_dict` dictionary."""
        require_model_tag(model_dict, "SurvivalTree", "a survival tree")
        tree = cls.__new__(cls)
        tree.kind = model_dict["kind"]
        tree.n_features_split = model_dict["n_features_split"]
        # A restored tree predicts but is not re-fittable; it holds no data.
        tree.data = None  # type: ignore[assignment]
        tree.Z = None  # type: ignore[assignment]
        tree._root = node_from_dict(model_dict["root"])
        return tree


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

    The parametric kinds (``"weibull"``, ``"exponential"``) support the
    full data model. The non-parametric kind splits observed and
    right-censored data (optionally left truncated) by the risk-set
    log-rank, and left- and interval-censored data by the Turnbull
    scores; neither is defined for right truncation, or for truncation
    with left or interval censoring, so such data are rejected.
    """
    resolved = kind.lower().replace("_", "-")
    if resolved in ("weibull", "exponential"):
        return resolved
    if resolved == "non-parametric":
        right_truncated = bool(np.isfinite(data.t[:, 1]).any())
        left_truncated = bool(np.isfinite(data.t[:, 0]).any())
        interval_like = bool(((data.c == 2) | (data.c == -1)).any())
        if right_truncated or (left_truncated and interval_like):
            raise ValueError(
                "kind='non-parametric' does not support right truncation, "
                "or truncation together with left or interval censoring: "
                "its splits (the risk-set log-rank, and the Turnbull "
                "scores for left and interval censoring) would need "
                "scores from the truncation-conditioned likelihood, which "
                "are not implemented yet (issue #188). Use kind='weibull' "
                "or kind='exponential' for this data."
            )
        return "non-parametric"
    raise ValueError(
        f"kind={kind!r} is invalid. Must be 'weibull', 'exponential' or "
        "'non-parametric'."
    )
