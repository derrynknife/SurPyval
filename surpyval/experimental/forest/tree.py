from math import log2, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from surpyval.experimental.forest.deviance_split import (
    needs_full_likelihood_split,
)
from surpyval.experimental.forest.node import build_tree
from surpyval.utils.surpyval_data import SurpyvalData


class SurvivalTree:
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
    ):
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
    ):
        """
        Fit a survival tree from data in the full xcnt(+truncation) data
        model.

        ``x``/``c``/``n``/``t`` follow the standard SurPyval conventions
        (``c`` in ``{-1, 0, 1, 2}``; interval-censored entries of ``x``
        are ``[left, right]`` pairs). Interval bounds can alternatively
        be given as ``xl``/``xr``, and truncation as ``tl``/``tr``
        instead of the two-column ``t``. ``kind`` selects the tree type
        (see the class docstring).
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
        return self.apply_model_function("sf", x, Z)

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
