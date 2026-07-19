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

    The split criterion is controlled by ``split_rule``:

    - ``"auto"`` (default): the risk-set log-rank split when the data is
      expressible in the xrd format (observed / right-censored,
      optionally left-truncated), and the full-likelihood exponential
      deviance split (Davis & Anderson, 1989) otherwise -- left or
      interval censoring and right truncation carry their event
      information as interval probabilities, for which no risk-set
      statistic exists.
    - ``"log-rank"``: force the risk-set log-rank split. Raises
      ``ValueError`` if the data contains left/interval censoring or
      right truncation.
    - ``"deviance"``: force the full-likelihood deviance split (valid
      for every data type).
    """

    def __init__(
        self,
        data: SurpyvalData,
        Z: NDArray,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        parametric: bool | str = "non-parametric",
        split_rule: str = "auto",
    ):
        self.data = data
        self.Z = Z

        n_features: int = parse_n_features_split(
            n_features_split, self.Z.shape[1]
        )

        self.n_features_split = n_features

        is_parametric: bool = (
            parametric
            if isinstance(parametric, bool)
            else parse_leaf_type(parametric)
        )

        self.split_rule = parse_split_rule(split_rule, data)

        self._root = build_tree(
            data=self.data,
            Z=self.Z,
            curr_depth=0,
            max_depth=max_depth,
            min_leaf_samples=min_leaf_samples,
            min_leaf_failures=min_leaf_failures,
            n_features_split=n_features,
            parametric=is_parametric,
            split_rule=self.split_rule,
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
        leaf_type: str = "parametric",
        split_rule: str = "auto",
    ):
        """
        Fit a survival tree from data in the full xcnt(+truncation) data
        model.

        ``x``/``c``/``n``/``t`` follow the standard SurPyval conventions
        (``c`` in ``{-1, 0, 1, 2}``; interval-censored entries of ``x``
        are ``[left, right]`` pairs). Interval bounds can alternatively
        be given as ``xl``/``xr``, and truncation as ``tl``/``tr``
        instead of the two-column ``t``.
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
            parse_leaf_type(leaf_type),
            split_rule,
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


def parse_leaf_type(leaf_type: str):
    if leaf_type.lower() == "non-parametric":
        return False
    elif leaf_type.lower() == "parametric":
        return True
    else:
        raise ValueError(f"leaf_type={leaf_type} is invalid. Must\
            'parametric' or 'non-parametric'")


def parse_split_rule(split_rule: str, data: SurpyvalData) -> str:
    """
    Resolve the requested split rule against the data.

    ``"auto"`` picks the risk-set log-rank when the data can be expressed
    in the xrd format and the full-likelihood deviance split otherwise.
    An explicit ``"log-rank"`` is validated against the data, since the
    risk-set statistic is undefined for left/interval censoring and
    right truncation.
    """
    rule = split_rule.lower().replace("-", "_")
    if rule == "auto":
        if needs_full_likelihood_split(data):
            return "deviance"
        return "log_rank"
    if rule in ("log_rank", "logrank"):
        if needs_full_likelihood_split(data):
            raise ValueError(
                "The risk-set log-rank split is undefined for data with "
                "left censoring, interval censoring, or right truncation; "
                "use split_rule='deviance' (or 'auto') for this data."
            )
        return "log_rank"
    if rule in ("deviance", "exponential"):
        return "deviance"
    raise ValueError(
        f"split_rule={split_rule!r} is invalid. Must be 'auto', "
        "'log-rank' or 'deviance'."
    )
