import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.tree import SurvivalTree
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.score import score
from surpyval.utils.surpyval_data import SurpyvalData


class RandomSurvivalForest(SerialisableMixin):
    """Random survival forest: an ensemble of survival trees.

    ``n_trees`` instances of
    :class:`~surpyval.beta.ml.forest.tree.SurvivalTree` are fitted, each
    to an independently bootstrapped sample of the data and each
    considering a random subset of the features at every split. A
    prediction evaluates all of them and averages the fitted models
    their leaves return, which trades the variance of one deep tree for
    the bias of an average.

    Constructed by :meth:`fit` rather than directly.
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
    ) -> None:
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
    ) -> "RandomSurvivalForest":
        """
        Fit a random survival forest.

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
        n_trees : int, optional
            The number of trees. Defaults to 100.
        bootstrap : bool, optional
            Fit each tree to a bootstrap resample of the data (the
            default); otherwise every tree sees all of it. Resampling uses
            NumPy's global random state, so seed it with
            ``np.random.seed`` for a reproducible forest.
        kind : str, optional
            The tree type, ``"weibull"`` (the default), ``"exponential"``
            or ``"non-parametric"``; see
            :class:`~surpyval.beta.ml.forest.tree.SurvivalTree`.

        Returns
        -------
        RandomSurvivalForest
            The fitted forest.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.beta.ml import RandomSurvivalForest
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 2))
        >>> x = rng.weibull(2.0, 200) * np.where(Z[:, 0] > 0.5, 5.0, 10.0)
        >>> c = (x > 12).astype(int)
        >>> x = np.minimum(x, 12)
        >>> np.random.seed(0)
        >>> forest = RandomSurvivalForest.fit(
        ...     x, Z, c=c, n_trees=5, max_depth=1, kind="exponential"
        ... )
        >>> forest.sf(5, [[0.2, 0.5], [0.8, 0.5]]).round(3)
        array([[0.561],
               [0.396]])
        """
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
        """Failure (CDF) function averaged over the trees, as for
        :meth:`sf`."""
        return self._apply_model_function_to_trees("ff", x, Z)

    def df(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """Density averaged over the trees, as for :meth:`sf`."""
        return self._apply_model_function_to_trees("df", x, Z)

    def hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """Hazard rate averaged over the trees, as for :meth:`sf`."""
        return self._apply_model_function_to_trees("hf", x, Z)

    def Hf(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> NDArray:
        """Cumulative hazard averaged over the trees, as for :meth:`sf`."""
        return self._apply_model_function_to_trees("Hf", x, Z)

    def mortality(
        self, x: int | float | ArrayLike, Z: ArrayLike | NDArray
    ) -> ArrayLike:
        """
        The ensemble mortality of each covariate vector: its cumulative
        hazard summed over the times ``x`` (the risk score used by
        :meth:`score`).
        """
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
        require_model_tag(
            model_dict, "RandomSurvivalForest", "a random survival forest"
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
