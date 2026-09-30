import warnings
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.oob import (
    RowTerms,
    add_tree_terms,
    row_log_likelihood,
    time_origin,
    weighted_mean,
)
from surpyval.beta.ml.forest.tree import (
    SurvivalTree,
    drop_missing_covariate_rows,
    resolve_random_state,
)
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils import _caller_stacklevel
from surpyval.utils.rng import as_generator
from surpyval.utils.score import score
from surpyval.utils.shapes import flatten_query
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

    Constructed by :meth:`fit` rather than directly. A fitted forest keeps
    its training data (``data`` and ``Z``, after dropping rows with a
    missing covariate) and ``bootstrap_indices``, the rows of ``data`` each
    tree was grown on (with repeats); :meth:`oob_log_likelihood` and
    :meth:`feature_importances` use them to score every row with the trees
    that did not see it.
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
        random_state: Any = None,
    ) -> None:
        # Rows with a missing covariate are dropped once, here, with the
        # standard warning, so no bootstrap sample can draw one.
        self.data: SurpyvalData
        self.Z: NDArray
        self.data, self.Z = drop_missing_covariate_rows(data, Z)
        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.kind = kind

        # With random_state=None every draw is from numpy's global stream,
        # in the order it always was: the bootstraps, then each tree's
        # feature draws in turn. A seed gives the forest its own stream,
        # and each tree a child stream of it, so a tree's draws do not
        # depend on the order the trees are grown in.
        rng = resolve_random_state(random_state)
        tree_states: list[Any]
        if random_state is None:
            tree_states = [None] * self.n_trees
        else:
            assert isinstance(rng, np.random.Generator)
            tree_states = list(rng.spawn(self.n_trees))

        # Create Trees
        bootstrap_indices: list[NDArray]
        if self.bootstrap:
            bootstrap_indices = [
                rng.choice(len(self.data.x), len(self.data.x), replace=True)
                for _ in range(self.n_trees)
            ]
        else:
            bootstrap_indices = [
                np.array(range(len(self.data.x)))
            ] * self.n_trees
        # Kept for the out-of-bag methods: the rows each tree did not see.
        self.bootstrap_indices: list[NDArray] | None = bootstrap_indices

        self.trees: list[SurvivalTree] = Parallel(prefer="threads", verbose=1)(
            delayed(SurvivalTree)(
                data=self.data[bootstrap_indices[i]],
                Z=self.Z[bootstrap_indices[i]],
                max_depth=max_depth,
                min_leaf_samples=min_leaf_samples,
                min_leaf_failures=min_leaf_failures,
                n_features_split=n_features_split,
                kind=kind,
                random_state=tree_states[i],
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
        random_state: Any = None,
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
        n_trees : int, optional
            The number of trees. Defaults to 100.
        bootstrap : bool, optional
            Fit each tree to a bootstrap resample of the data (the
            default); otherwise every tree sees all of it.
        kind : str, optional
            The tree type, ``"weibull"`` (the default), ``"exponential"``
            or ``"non-parametric"``; see
            :class:`~surpyval.beta.ml.forest.tree.SurvivalTree`.
        random_state : None, int or numpy.random.Generator, optional
            Seeds the bootstrap resamples and the features drawn for each
            split. ``None`` (the default) draws from NumPy's global random
            state, so ``np.random.seed`` reproduces the forest; a seed or
            ``Generator`` gives the forest a stream of its own (and each
            tree a child stream of it) and leaves the global one alone.

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
        array([0.561, 0.396])

        A seed of its own reproduces the forest without touching NumPy's
        global state:

        >>> a = RandomSurvivalForest.fit(
        ...     x, Z, c=c, n_trees=5, max_depth=1, kind="exponential",
        ...     random_state=1,
        ... )
        >>> b = RandomSurvivalForest.fit(
        ...     x, Z, c=c, n_trees=5, max_depth=1, kind="exponential",
        ...     random_state=1,
        ... )
        >>> bool(np.array_equal(a.sf(5, Z[:3]), b.sf(5, Z[:3])))
        True
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
            random_state,
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
            Times, the same for every covariate vector.
        Z : ArrayLike | NDArray
            One covariate vector (1-D), or a matrix with one covariate
            vector per row (2-D).
        ensemble_method : str, optional
            Determines whether to average across terminal nodes the terminal
            node survival functions or cumulative hazard functions.
            For these respectively, ensemble_method must be "sf" or
            "Hf". Defaults to "sf".

        Returns
        -------
        NDArray
            For a 1-D ``Z``, the survival function at ``x``, shaped like
            ``x`` (a scalar for a scalar ``x``). For a 2-D ``Z``, a grid of
            shape ``(n_rows,) + x.shape`` whose row ``i`` is the survival
            function for ``Z[i]`` (every row at every time). A covariate
            vector with a missing (NaN) value gives NaN, and leaves the
            other rows unaffected.
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
        # The times flat; the result gets their shape back (on its last
        # axis for a grid), so a scalar time gives a scalar.
        x, restore = flatten_query(x)
        single_covariant_vector = np.ndim(Z) < 2
        Z = np.array(Z, ndmin=2)

        # Each tree routes every row to its own leaf and returns an
        # (n_rows, x.size) grid
        res = np.zeros((Z.shape[0], x.size), dtype=np.float64)
        for tree in self.trees:
            res += tree.apply_model_function(function_name, x, Z)
        res = res / self.n_trees
        if single_covariant_vector:
            return restore(res[0])
        return restore(res, axis=-1)

    def score(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        tie_tol: float = 1e-8,
    ) -> float:
        """Harrell's concordance index of the forest's mortality scores.

        A missing (NaN) covariate or time leaves a subject's score, and so
        the index, undefined: the index is NaN, not a number computed by
        comparing the NaN score as though it were one.
        """
        scores: ArrayLike = self.mortality(x, Z)
        if np.isnan(scores).any():
            return float("nan")
        return score(x, c, scores, tie_tol)

    def _oob_setup(self) -> tuple[list[NDArray], RowTerms, float, NDArray]:
        # The rows each tree left out, the likelihood's view of every row,
        # the time origin of the step-function leaves, and how many trees
        # left each row out. A restored forest has no data to score.
        if self.data is None or self.bootstrap_indices is None:
            raise ValueError(
                "The out-of-bag methods need the training data and the "
                "bootstrap samples, which a forest restored with from_dict "
                "does not keep; call them on the fitted forest."
            )
        n_rows = len(self.data)
        oob = [
            np.setdiff1d(np.arange(n_rows), idx)
            for idx in self.bootstrap_indices
        ]
        n_oob = np.zeros(n_rows)
        for rows in oob:
            n_oob[rows] += 1
        missing = int((n_oob == 0).sum())
        if missing:
            reason = (
                "bootstrap=False grows every tree on every row"
                if not self.bootstrap
                else "grow more trees to cover them"
            )
            warnings.warn(
                f"{missing} of {n_rows} rows were in the sample of every "
                f"tree, so no tree can score them out of bag; they are "
                f"left out of the out-of-bag log-likelihood ({reason}).",
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
        return oob, RowTerms(self.data), time_origin(self.data), n_oob

    def _oob_rows_log_likelihood(
        self,
        oob: list[NDArray],
        terms: RowTerms,
        origin: float,
        n_oob: NDArray,
        curves: dict,
        permute: tuple[int, Any] | None = None,
    ) -> NDArray:
        # Each row's log-likelihood under the trees that left it out. With
        # ``permute=(j, rng)``, feature j is first shuffled among each
        # tree's out-of-bag rows.
        numerator = np.zeros(len(terms))
        denominator = np.zeros(len(terms))
        for tree, rows in zip(self.trees, oob):
            if rows.size == 0:
                continue
            Z = self.Z[rows]
            if permute is not None:
                j, rng = permute
                Z = Z.copy()
                Z[:, j] = Z[rng.permutation(rows.size), j]
            add_tree_terms(
                tree._root,
                Z,
                rows,
                terms,
                curves,
                origin,
                numerator,
                denominator,
            )
        return row_log_likelihood(numerator, denominator, n_oob)

    def oob_log_likelihood(self) -> float:
        r"""The mean out-of-bag log-likelihood per observation.

        Each row of the training data is scored by the ensemble of the
        trees whose bootstrap sample left it out, so by trees that never
        saw it: its contribution is the log of

        - the density :math:`f(x)` if it was observed at :math:`x`,
        - :math:`S(x)` if right censored at :math:`x`,
        - :math:`1 - S(x)` if left censored at :math:`x`,
        - :math:`S(x_l) - S(x_r)` if interval censored in
          :math:`(x_l, x_r]`,

        divided by :math:`S(t_l) - S(t_r)` if it is truncated to
        :math:`(t_l, t_r]`, where :math:`S` and :math:`f` are the averages
        of the out-of-bag trees' leaf survival functions and densities. A
        censored row's interval is first cut to its truncation window, as
        in the fitters' likelihoods. The result is the count-weighted mean
        over the rows, so higher is better and it estimates the expected
        log-likelihood of a new observation; it works for every censoring
        type and truncation, where the concordance of :meth:`score` needs
        orderable event times.

        A non-parametric leaf is a step function, which puts no
        probability at an out-of-bag event time unless the tree saw a
        tied one, so for this score it is read as a continuous
        distribution: its survival curve is joined linearly between the
        points where it drops, from 1 at time 0 (or at the smallest time,
        if it is negative), and continued past its last drop with the
        constant hazard it averaged up to there (Brown, Hollander and
        Korwar's exponential tail). Its density is then per unit of time,
        on the same scale as a parametric leaf's, so forests of different
        ``kind`` can be compared by this score.

        A row that is in the bootstrap sample of every tree has no
        out-of-bag prediction; it is left out of the mean, with one
        warning giving the count (every row, and a NaN result, with
        ``bootstrap=False``). A row the ensemble gives zero probability
        makes the mean ``-inf``.

        Returns
        -------
        float
            The mean out-of-bag log-likelihood per observation.

        Raises
        ------
        ValueError
            On a forest restored with ``from_dict``, which keeps neither
            the training data nor the bootstrap samples.

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
        ...     x, Z, c=c, n_trees=20, max_depth=1, kind="exponential"
        ... )
        >>> round(forest.oob_log_likelihood(), 3)
        -2.518
        """
        oob, terms, origin, n_oob = self._oob_setup()
        ll = self._oob_rows_log_likelihood(oob, terms, origin, n_oob, {})
        return weighted_mean(ll, terms.n)

    def feature_importances(
        self, n_repeats: int = 5, random_state: Any = None
    ) -> NDArray:
        r"""Permutation importance of each feature, by out-of-bag
        log-likelihood.

        For each feature, its values are shuffled among each tree's
        out-of-bag rows (Breiman, 2001), which breaks its link with the
        outcome while keeping its distribution, and the out-of-bag
        log-likelihood (see :meth:`oob_log_likelihood`) is computed again.
        A feature's importance is the drop from the unshuffled value,
        averaged over ``n_repeats`` shuffles: about zero for a feature the
        forest does not use, positive for one it relies on, and on the
        scale of the log-likelihood per observation.

        Parameters
        ----------
        n_repeats : int, optional
            The number of shuffles averaged for each feature. Defaults
            to 5.
        random_state : None, int or numpy.random.Generator, optional
            Seeds the shuffles. ``None`` (the default) draws from NumPy's
            global random state, so ``np.random.seed`` reproduces it; a
            seed or ``Generator`` gives its own stream.

        Returns
        -------
        numpy.ndarray
            One importance per column of ``Z``.

        Raises
        ------
        ValueError
            If ``n_repeats`` is not a positive integer, or on a forest
            restored with ``from_dict`` (see :meth:`oob_log_likelihood`).

        Examples
        --------
        Only the first feature matters; the second is noise:

        >>> import numpy as np
        >>> from surpyval.beta.ml import RandomSurvivalForest
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 2))
        >>> x = rng.weibull(2.0, 200) * np.where(Z[:, 0] > 0.5, 5.0, 10.0)
        >>> np.random.seed(0)
        >>> forest = RandomSurvivalForest.fit(
        ...     x, Z, n_trees=20, max_depth=1, kind="exponential"
        ... )
        >>> forest.feature_importances(random_state=1).round(3)
        array([ 0.093, -0.001])
        """
        if (
            isinstance(n_repeats, bool)
            or not isinstance(n_repeats, (int, np.integer))
            or n_repeats < 1
        ):
            raise ValueError(
                f"n_repeats must be a positive integer, got {n_repeats!r}"
            )
        rng = as_generator(random_state)
        oob, terms, origin, n_oob = self._oob_setup()
        curves: dict = {}
        baseline = weighted_mean(
            self._oob_rows_log_likelihood(oob, terms, origin, n_oob, curves),
            terms.n,
        )
        importances = np.zeros(self.Z.shape[1])
        for j in range(self.Z.shape[1]):
            drops = [
                baseline
                - weighted_mean(
                    self._oob_rows_log_likelihood(
                        oob, terms, origin, n_oob, curves, permute=(j, rng)
                    ),
                    terms.n,
                )
                for _ in range(int(n_repeats))
            ]
            importances[j] = np.mean(drops)
        return importances

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
        forest.bootstrap_indices = None
        forest.trees = [
            SurvivalTree.from_dict(tree_dict)
            for tree_dict in model_dict["trees"]
        ]
        return forest
