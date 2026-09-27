from math import log2, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.deviance_split import (
    needs_full_likelihood_split,
)
from surpyval.beta.ml.forest.node import build_tree, node_from_dict
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.surpyval_data import SurpyvalData


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
    - ``"non-parametric"``: risk-set log-rank split with Nelson-Aalen
      leaves. Only defined for observed / right-censored data
      (optionally left-truncated); raises ``ValueError`` otherwise --
      left/interval censoring and right truncation carry their event
      information as interval probabilities, for which no risk-set
      statistic exists.
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
        self.data = data
        self.Z = Z

        n_features: int = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self.n_features_split = n_features

        self.kind = parse_kind(kind, data)

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
            ``Hf``) evaluate the model of the leaf that one covariate
            vector ``Z`` falls in.

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
        (array([0.8831]), array([0.3168]))
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
        # Prep input - make sure numpy array
        x = np.array(x, ndmin=1)
        Z = np.array(Z, ndmin=1)

        return self._root.apply_model_function(function_name, x, Z)

    def sf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Survival function at ``x`` of the leaf model for one covariate
        vector ``Z``.
        """
        return self.apply_model_function("sf", x, Z)

    def ff(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Failure (CDF) function at ``x`` of the leaf model for one covariate
        vector ``Z``.
        """
        return self.apply_model_function("ff", x, Z)

    def df(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Density at ``x`` of the leaf model for one covariate
        vector ``Z``.
        """
        return self.apply_model_function("df", x, Z)

    def hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Hazard rate at ``x`` of the leaf model for one covariate
        vector ``Z``.
        """
        return self.apply_model_function("hf", x, Z)

    def Hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """
        Cumulative hazard at ``x`` of the leaf model for one covariate
        vector ``Z``.
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
    full data model. The non-parametric kind's split (the risk-set
    log-rank) is undefined for left/interval censoring and right
    truncation, so it is rejected for such data.
    """
    resolved = kind.lower().replace("_", "-")
    if resolved in ("weibull", "exponential"):
        return resolved
    if resolved == "non-parametric":
        if needs_full_likelihood_split(data):
            raise ValueError(
                "kind='non-parametric' is undefined for data with left "
                "censoring, interval censoring, or right truncation: its "
                "risk-set log-rank split has no risk-set formulation for "
                "interval-probability observations. Use kind='weibull' or "
                "kind='exponential' for this data (a Turnbull-score split "
                "is planned; see issue #188)."
            )
        return "non-parametric"
    raise ValueError(
        f"kind={kind!r} is invalid. Must be 'weibull', 'exponential' or "
        "'non-parametric'."
    )
