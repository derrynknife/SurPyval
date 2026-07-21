import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.tree import SurvivalTree
from surpyval.serialisation import stamp_schema
from surpyval.utils.score import score
from surpyval.utils.surpyval_data import SurpyvalData


class RandomSurvivalForest:
    """Random Survival Forest

    Specs:
    - n_trees `Tree`'s trained, each given independently bootstrapped samples
    - Each tree is trained
    - Each predicition is evaluated for each tree, the Weibull models are
      collected, then averaged
    """

    def __init__(
        self,
        data: SurpyvalData,
        Z: ArrayLike | NDArray,
        n_trees: int = 100,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        bootstrap: bool = True,
        kind: str = "weibull",
    ):
        self.data: SurpyvalData = data
        Z = np.asarray(Z)
        if Z.ndim == 1:
            # A 1-d Z is a single feature, one value per sample
            Z = Z.reshape(-1, 1)
        self.Z: NDArray = Z
        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.kind = kind

        # Create Trees
        if self.bootstrap:
            bootstrap_indices = [
                np.random.choice(
                    len(self.data.x), len(self.data.x), replace=True
                )
                for _ in range(self.n_trees)
            ]
        else:
            bootstrap_indices = [
                np.array(range(len(self.data.x)))
            ] * self.n_trees

        self.trees: list[SurvivalTree] = Parallel(prefer="threads", verbose=1)(
            delayed(SurvivalTree)(
                data=self.data[bootstrap_indices[i]],
                Z=self.Z[bootstrap_indices[i]],
                max_depth=max_depth,
                min_leaf_samples=min_leaf_samples,
                min_leaf_failures=min_leaf_failures,
                n_features_split=n_features_split,
                kind=kind,
            )
            for i in range(self.n_trees)
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
        n_trees: int = 100,
        max_depth: int | float = float("inf"),
        min_leaf_samples: int = 5,
        min_leaf_failures: int = 2,
        n_features_split: int | float | str = "sqrt",
        bootstrap: bool = True,
        kind: str = "weibull",
    ):
        if Z is None:
            raise ValueError("The covariate matrix Z is required")
        data = SurpyvalData(
            x, c, n, t, xl=xl, xr=xr, tl=tl, tr=tr, group_and_sort=False
        )
        return cls(
            data,
            Z,
            n_trees,
            max_depth,
            min_leaf_samples,
            min_leaf_failures,
            n_features_split,
            bootstrap,
            kind,
        )

    def sf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        ensemble_method: str = "sf",
    ) -> NDArray:
        """Returns the ensemble survival function

        Parameters
        ----------
        x : int | float | ArrayLike
            Time samples
        Z : ArrayLike | NDArray
            Covariant matrix
        ensemble_method : str, optional
            Determines whether to average across terminal nodes the terminal
            node survival functions or cumulative hazard functions.
            For these respectively, ensemble_method must be "sf" or
            "Hf". Defaults to "sf".

        Returns
        -------
        NDArray
            Survival function of x as 1D array
        """
        if ensemble_method == "Hf":
            Hf = self._apply_model_function_to_trees("Hf", x, Z)
            return np.exp(-Hf)
        return self._apply_model_function_to_trees("sf", x, Z)

    def ff(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("ff", x, Z)

    def df(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("df", x, Z)

    def hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("hf", x, Z)

    def Hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        return self._apply_model_function_to_trees("Hf", x, Z)

    def mortality(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> ArrayLike:
        mortality = np.atleast_2d(self.Hf(x, Z)).sum(1)
        return np.clip(mortality, 0, np.finfo(np.float64).max)

    def _apply_model_function_to_trees(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
    ) -> NDArray:
        # Prep input - make sure numpy array
        x = np.array(x, ndmin=1)
        single_covariant_vector = np.ndim(Z) < 2
        Z = np.array(Z, ndmin=2)

        res = np.zeros((Z.shape[0], x.size)).astype(np.float64)
        for i_covariant_vector in range(Z.shape[0]):
            for tree in self.trees:
                values = tree.apply_model_function(
                    function_name, x, Z[i_covariant_vector, :]
                )
                res[i_covariant_vector, :] += values
        res = res / self.n_trees
        if single_covariant_vector:
            return res[0]
        return res

    def score(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        tie_tol: float = 1e-8,
    ) -> float:
        """Harrell's concordance index of the forest's mortality scores."""
        scores: ArrayLike = self.mortality(x, Z)
        return score(x, c, scores, tie_tol)

    def to_dict(self) -> dict:
        """Serialise the fitted forest to a plain, JSON/BSON-safe dictionary:
        the ensemble settings and every fitted tree. The training data is not
        persisted -- a restored forest is a predictor, not re-fittable."""
        return stamp_schema(
            {
                "model": "RandomSurvivalForest",
                "kind": self.kind,
                "n_trees": int(self.n_trees),
                "bootstrap": bool(self.bootstrap),
                "trees": [tree.to_dict() for tree in self.trees],
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> "RandomSurvivalForest":
        """Reconstruct a fitted forest from a :meth:`to_dict` dictionary."""
        if model_dict.get("model") != "RandomSurvivalForest":
            raise ValueError(
                "Must create a RandomSurvivalForest from a "
                "RandomSurvivalForest model dict"
            )
        forest = cls.__new__(cls)
        forest.kind = model_dict["kind"]
        forest.n_trees = model_dict["n_trees"]
        forest.bootstrap = model_dict["bootstrap"]
        # A restored forest predicts but is not re-fittable; it holds no data.
        forest.data = None  # type: ignore[assignment]
        forest.Z = None  # type: ignore[assignment]
        forest.trees = [
            SurvivalTree.from_dict(tree_dict)
            for tree_dict in model_dict["trees"]
        ]
        return forest

    def to_json(self, fp: str | Path) -> None:
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_json(cls, fp: str | Path) -> "RandomSurvivalForest":
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))
