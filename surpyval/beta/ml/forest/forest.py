import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from surpyval.beta.ml.forest.conditional_inference import parse_selection
from surpyval.beta.ml.forest.deviance_split import parse_min_split_gain
from surpyval.beta.ml.forest.oob import (
    RowTerms,
    add_tree_terms,
    row_log_likelihood,
    time_origin,
    weighted_mean,
)
from surpyval.beta.ml.forest.tree import (
    SurvivalTree,
    covariate_matrix,
    drop_missing_covariate_rows,
    feature_labels,
    resolve_random_state,
)
from surpyval.metrics.concordance import concordance_index
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression.regression_data import (
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from surpyval.utils.dataframe import RegressionDataFrameMixin
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import flatten_query
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import check_option
from surpyval.utils.warnings import caller_stacklevel


def _warn_zero_probability(
    ll: NDArray, consequence: str, lost: int = 0, shuffles: int = 0
) -> None:
    """One warning (#533) giving the number of out-of-bag rows the
    ensemble gives zero probability (a log-likelihood of ``-inf``; NaN is
    a row no tree left out, warned of by ``_oob_setup``), what that does
    to the result, and, for the permutation importance, the number of a
    shuffle's row scorings that a shuffle took to zero probability."""
    scored = ~np.isnan(ll)
    zero = int((scored & ~np.isfinite(ll)).sum())
    if not zero and not lost:
        return
    parts = []
    if zero:
        parts.append(
            f"{zero} of {int(scored.sum())} out-of-bag rows have zero "
            "probability under the trees that left them out (their leaves "
            f"put no density at their times), so {consequence}"
        )
    if lost:
        parts.append(
            f"{lost} scorings of a row became zero probability when a "
            f"feature was shuffled ({shuffles} shuffles), and are left out "
            "of that shuffle's drop"
        )
    warnings.warn(
        "; ".join(parts) + ". With few trees a row can land only in "
        "leaves too steep or narrow to cover it: grow more trees "
        "(n_trees), or use kind='exponential', whose leaves give every "
        "time a density.",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


def _check_n_jobs(n_jobs: Any) -> None:
    """``n_jobs`` is joblib's: a non-zero integer, -1 for every core."""
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
    ):
        raise ValueError(
            "n_jobs must be a non-zero integer (-1 for every core), got "
            f"{n_jobs!r}"
        )


class RandomSurvivalForest(RegressionDataFrameMixin, SerialisableMixin):
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

    A forest fitted from a DataFrame (``fit_from_df``, or ``fit`` with a
    DataFrame ``Z``) keeps the covariate names as ``feature_names``
    (``None`` when fitted from an array); its trees' split descriptions
    and :meth:`feature_importances` use them, and its predictions accept
    a DataFrame ``Z``, read by those names.
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
        selection: str = "greedy",
        alpha_split: float = 0.05,
        random_state: Any = None,
        feature_names: list[str] | None = None,
        min_split_gain: float | str = 0.0,
        n_jobs: int = 1,
    ) -> None:
        _check_n_jobs(n_jobs)
        self.selection = parse_selection(selection, alpha_split)
        self.alpha_split = float(alpha_split)
        Z, self.feature_names = covariate_matrix(Z, feature_names)
        # Set by ``fit_from_df(formula=...)``: the formula and its
        # design-matrix transformer, to expand a DataFrame at prediction.
        self.formula: str | None = None
        self._model_spec: Any = None
        # Rows with a missing covariate are dropped once, here, with the
        # standard warning, so no bootstrap sample can draw one.
        self.data: SurpyvalData
        self.Z: NDArray
        self.data, self.Z = drop_missing_covariate_rows(data, Z)
        self.n_trees = n_trees
        self.bootstrap = bootstrap
        self.kind = kind
        # Validated against the kind once, before any tree is grown
        self.min_split_gain = parse_min_split_gain(
            min_split_gain, kind.lower().replace("_", "-")
        )

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

        if random_state is None and n_jobs != 1:
            # Worker processes do not share numpy's global stream, so the
            # trees draw from streams of their own, seeded by one draw
            # from it: ``np.random.seed`` still reproduces the forest.
            seed = int(np.random.randint(0, 2**63 - 1, dtype=np.int64))
            tree_states = list(np.random.default_rng(seed).spawn(n_trees))

        # Quiet (principle 22): joblib reports progress only if asked.
        # With n_jobs=1 the trees are grown one after another in this
        # process; otherwise in joblib's worker processes.
        self.trees: list[SurvivalTree] = Parallel(n_jobs=n_jobs)(
            delayed(SurvivalTree)(
                data=self.data[bootstrap_indices[i]],
                Z=self.Z[bootstrap_indices[i]],
                max_depth=max_depth,
                min_leaf_samples=min_leaf_samples,
                min_leaf_failures=min_leaf_failures,
                n_features_split=n_features_split,
                kind=kind,
                selection=selection,
                alpha_split=alpha_split,
                random_state=tree_states[i],
                feature_names=self.feature_names,
                min_split_gain=self.min_split_gain,
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
        selection: str = "greedy",
        alpha_split: float = 0.05,
        min_split_gain: float | str = 0.0,
        random_state: Any = None,
        n_jobs: int = 1,
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
            ... and at least this many failures (rows that are not
            right censored, each counted ``n`` times). Defaults to 2.
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
        selection : str, optional
            How each node chooses its feature: ``"greedy"`` (the default)
            or ``"ctree"`` (conditional inference, which also stops a
            tree where the data show no effect); see
            :class:`~surpyval.beta.ml.forest.tree.SurvivalTree`.
        alpha_split : float, optional
            With ``selection="ctree"``, a node splits only if the
            Bonferroni-adjusted p-value of its chosen feature is below
            ``alpha_split``. Defaults to 0.05.
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
            wants. (``"aic"`` is the recommended setting for a single
            :class:`~surpyval.beta.ml.forest.tree.SurvivalTree`.) Not
            used by ``"non-parametric"`` trees, whose splits are not
            likelihoods; stop those with ``selection="ctree"``.
        random_state : None, int or numpy.random.Generator, optional
            Seeds the bootstrap resamples and the features drawn for each
            split. ``None`` (the default) draws from NumPy's global random
            state, so ``np.random.seed`` reproduces the forest; a seed or
            ``Generator`` gives the forest a stream of its own (and each
            tree a child stream of it) and leaves the global one alone.
            The forest is the same whatever ``n_jobs`` is, except with
            ``None`` (see ``n_jobs``).
        n_jobs : int, optional
            The number of worker processes the trees are grown in, as in
            joblib and scikit-learn: 1 (the default) grows them one after
            another in this process, -1 uses every core. The trees and
            their predictions do not depend on it given a seed. With
            ``random_state=None`` and ``n_jobs`` other than 1, the trees'
            feature draws come from streams seeded by one draw from
            NumPy's global state (worker processes do not share it), so
            ``np.random.seed`` still reproduces the forest for that
            ``n_jobs``, but not the ``n_jobs=1`` forest.

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
        Z, feature_names = covariate_matrix(Z)
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
            selection,
            alpha_split,
            random_state,
            feature_names,
            min_split_gain,
            n_jobs,
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
        # Anything but 'Hf' used to be taken silently as 'sf'.
        check_option("ensemble_method", ensemble_method, ("sf", "Hf"))
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
        if isinstance(Z, pd.DataFrame):
            # Read by the fitted names (or expanded by the formula)
            Z = prepare_Z(Z, self.feature_names, self._model_spec)
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
        ties: str = "therneau",
    ) -> float:
        """Harrell's concordance index of the forest's mortality scores.

        The index is :func:`surpyval.metrics.concordance_index`, with its
        tie conventions: by default (``ties="therneau"``, as R's
        ``survival::concordance``, lifelines and every model's
        ``concordance``) two events at the same time are not a usable
        pair; ``ties="harrell"`` counts them, as this method did before
        v0.22.

        A missing (NaN) covariate or time leaves a subject's score, and so
        the index, undefined: the index is NaN, not a number computed by
        comparing the NaN score as though it were one.
        """
        scores: ArrayLike = self.mortality(x, Z)
        if np.isnan(scores).any():
            return float("nan")
        return concordance_index(x, c, scores, tie_tol, ties)

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
                stacklevel=caller_stacklevel(),
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
        makes the mean ``-inf``, with one warning giving the count of such
        rows: with few trees a row can land only in leaves that put no
        density at its time (a steep Weibull leaf grown on bunched
        failures, or a step leaf); grow more trees, or use
        ``kind="exponential"``, whose leaves give every time a density.

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
        _warn_zero_probability(ll, "the mean is -inf")
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

        Each drop is the mean over the same rows before and after the
        shuffle, the rows whose out-of-bag log-likelihood is finite in
        both. A row the ensemble gives zero probability (see
        :meth:`oob_log_likelihood`) has a log-likelihood of ``-inf``, and
        a drop from or to ``-inf`` is no number at all: such a row is left
        out (of every feature's importance if it is ``-inf`` unshuffled,
        of that shuffle's drop if it becomes ``-inf`` only when shuffled),
        with one warning giving the counts. Grow more trees, or use
        ``kind="exponential"``, so that every row is scored.

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
        pandas.Series
            One importance per column of ``Z``, indexed by the feature
            names (:attr:`feature_labels`: ``feature_names`` for a forest
            fitted from a DataFrame, ``Z0``, ``Z1``, ... otherwise).

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
        Z0    0.093
        Z1   -0.001
        Name: importance, dtype: float64
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
        ll0 = self._oob_rows_log_likelihood(oob, terms, origin, n_oob, curves)
        # The rows scored before the shuffle, and the shuffles' scorings
        # of them lost to a zero probability (#533).
        finite0 = np.isfinite(ll0)
        lost = 0
        importances = np.zeros(self.Z.shape[1])
        for j in range(self.Z.shape[1]):
            drops = []
            for _ in range(int(n_repeats)):
                ll = self._oob_rows_log_likelihood(
                    oob, terms, origin, n_oob, curves, permute=(j, rng)
                )
                keep = finite0 & np.isfinite(ll)
                lost += int((finite0 & ~keep).sum())
                # The drop over the same rows before and after: with every
                # row finite, the drop of the whole out-of-bag mean.
                drops.append(
                    weighted_mean(np.where(keep, ll0, np.nan), terms.n)
                    - weighted_mean(np.where(keep, ll, np.nan), terms.n)
                )
            importances[j] = np.mean(drops)
        _warn_zero_probability(
            ll0,
            "they are left out of every feature's importance",
            lost,
            self.Z.shape[1] * int(n_repeats),
        )
        return pd.Series(
            importances, index=self.feature_labels, name="importance"
        )

    @property
    def feature_labels(self) -> list[str]:
        """The name of each feature: ``feature_names`` for a forest
        fitted from a DataFrame, else ``Z0``, ``Z1``, ... by column of
        ``Z``."""
        if self.feature_names is not None or self.Z is not None:
            n_features = 0 if self.Z is None else self.Z.shape[1]
            return feature_labels(self.feature_names, n_features)
        return self.trees[0].feature_labels if self.trees else []

    def __repr__(self) -> str:
        return (
            f"RandomSurvivalForest(kind={self.kind!r}, "
            f"n_trees={self.n_trees}, selection={self.selection!r}, "
            f"features={self.feature_labels})"
        )

    def to_dict(self) -> dict:
        """Serialise the fitted forest to a plain, JSON/BSON-safe dictionary:
        the ensemble settings and every fitted tree. The training data is not
        persisted -- a restored forest is a predictor, not re-fittable."""
        out = {
            "model": "RandomSurvivalForest",
            "kind": self.kind,
            "n_trees": int(self.n_trees),
            "bootstrap": bool(self.bootstrap),
            "selection": self.selection,
            "alpha_split": float(self.alpha_split),
            "min_split_gain": self.min_split_gain,
            "trees": [tree.to_dict() for tree in self.trees],
        }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

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
        # Forests saved before selection existed were grown greedily.
        forest.selection = model_dict.get("selection", "greedy")
        forest.alpha_split = model_dict.get("alpha_split", 0.05)
        forest.min_split_gain = model_dict.get("min_split_gain", 0.0)
        # A restored forest predicts but is not re-fittable; it holds no data.
        forest.data = None  # type: ignore[assignment]
        forest.Z = None  # type: ignore[assignment]
        forest.bootstrap_indices = None
        forest._model_spec = None
        # Forests saved before feature names existed have none.
        restore_covariate_meta(forest, model_dict)
        forest.trees = [
            SurvivalTree.from_dict(tree_dict)
            for tree_dict in model_dict["trees"]
        ]
        return forest
