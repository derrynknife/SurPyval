import warnings
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray

from surpyval.metrics.concordance import concordance_index
from surpyval.ml.forest.conditional_inference import parse_selection
from surpyval.ml.forest.deviance_split import parse_min_split_gain
from surpyval.ml.forest.oob import (
    RowTerms,
    add_tree_terms,
    row_log_likelihood,
    time_origin,
    weighted_mean,
)
from surpyval.ml.forest.tree import (
    SurvivalTree,
    check_covariate_count,
    covariate_matrix,
    drop_missing_covariate_rows,
    feature_labels,
    parse_causes,
    parse_kind,
    query_layout,
    resolve_random_state,
)
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import label_from_native
from surpyval.univariate.regression.regression_data import (
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from surpyval.utils.dataframe import RegressionDataFrameMixin
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import flatten_query
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.units import takes_unit
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
        "leaves too steep or narrow to cover it, or, with causes, only in "
        "leaves with no failure from its cause (a rare cause): grow more "
        "trees (n_trees), or use kind='exponential', whose leaves give "
        "every time a density.",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


def _cause_outcomes(
    x: ArrayLike, c: ArrayLike, e: Any, event: Any
) -> tuple[NDArray, NDArray]:
    """The outcomes of one cause for Harrell's index (Wolbers et al.,
    2014): a failure from ``event`` is the event; a failure from another
    cause never fails from it, so it is censored at infinity (later than
    every time); a censored row stays censored where it was."""
    x = np.asarray(x, dtype=float)
    c = np.asarray(c)
    labels = list(e)
    if len(labels) != x.size or c.size != x.size:
        raise ValueError(
            f"x, c and e must have one value per row; got {x.size}, "
            f"{c.size} and {len(labels)}."
        )
    mine = np.array([v == event for v in labels], dtype=bool) & (c == 0)
    other = (c == 0) & ~mine
    return np.where(other, np.inf, x), np.where(mine, 0, 1)


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
    :class:`~surpyval.ml.forest.tree.SurvivalTree` are fitted, each
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

    Grown with the cause of each failure (``e``), every tree is a
    competing-risks tree (see
    :class:`~surpyval.ml.forest.tree.SurvivalTree`), grown for every cause
    of the forest's data whether or not its bootstrap sample has it:
    :meth:`cif` is the average of the trees' leaf incidences of a cause,
    :meth:`total_cif` their sum, ``sf`` the average all-cause survival, and
    the out-of-bag score and importances judge the cause of each failure
    (:meth:`oob_log_likelihood`).
    """

    #: The causes of failure of a forest grown with ``e``, in their sorted
    #: order; ``None`` for a forest grown without causes.
    causes: "list | None" = None
    #: The split rule of a forest grown with causes, and the cause it was
    #: nominated for (``None``: every cause); ``None`` without causes.
    split: "str | None" = None
    event: Any = None

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
        split: str | None = None,
        event: Any = None,
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
        # The kind and the times are checked once, on every row, before
        # any tree is grown (#618): a bootstrap sample may miss a bad row
        parse_kind(kind, self.data)
        self.kind = kind
        # Validated against the kind once, before any tree is grown
        self.min_split_gain = parse_min_split_gain(
            min_split_gain, kind.lower().replace("_", "-")
        )
        # The causes are found and checked once, on every row: each tree is
        # grown for all of them, though its bootstrap sample may miss one.
        parsed = parse_causes(
            self.data,
            self.selection,
            self.min_split_gain,
            split,
            event,
            model="forest",
        )
        if parsed is not None:
            self.causes, self.split, self.event = parsed

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
                split=self.split,
                event=self.event,
                causes=self.causes,
            )
            for i in range(self.n_trees)
        )

    @classmethod
    @takes_unit
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
        e: ArrayLike | None = None,
        split: str | None = None,
        event: Any = None,
    ) -> "RandomSurvivalForest":
        """
        Fit a random survival forest.

        Parameters
        ----------
        x : array_like, optional
            Event times (``[left, right]`` rows for interval-censored
            observations), checked as ``SurvivalTree.fit`` checks
            them for the ``kind``.
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
            :class:`~surpyval.ml.forest.tree.SurvivalTree`.
        selection : str, optional
            How each node chooses its feature: ``"greedy"`` (the default)
            or ``"ctree"`` (conditional inference, which also stops a
            tree where the data show no effect); see
            :class:`~surpyval.ml.forest.tree.SurvivalTree`.
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
            :class:`~surpyval.ml.forest.tree.SurvivalTree`.) Not
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
        e : array_like, optional
            The cause of each failure, with a missing value (``None``,
            ``NaN``) for a right-censored row, as
            ``SurvivalTree.fit`` takes it: every tree is then a
            competing-risks tree, grown for every cause in ``e``, and the
            same data and options are refused as there (left and interval
            censoring, right truncation, ``selection="ctree"``, a non-zero
            ``min_split_gain``). ``c`` may be left out: a row with a cause
            failed, one without is right censored.
        split : str, optional
            With ``e``, the trees' split rule: ``"cause-specific"`` (the
            default) or ``"gray"``; see ``SurvivalTree.fit``.
        event : optional
            With ``e``, the cause the splits target: required by
            ``split="gray"``; ``None`` (the default) is every cause.

        Returns
        -------
        RandomSurvivalForest
            The fitted forest.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.ml import RandomSurvivalForest
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

        With the cause of each failure, each cause's cumulative incidence
        at given covariates, and an out-of-bag score that judges the
        causes:

        >>> t_a = rng.exponential(np.where(Z[:, 0] > 0.5, 2.0, 10.0))
        >>> t_b = rng.exponential(10.0, 200)
        >>> xe = np.minimum(np.minimum(t_a, t_b), 8.0)
        >>> e = np.where(t_a < t_b, "a", "b").astype(object)
        >>> e[xe == 8.0] = None
        >>> cr = RandomSurvivalForest.fit(
        ...     xe, Z, e=e, n_trees=20, max_depth=2, kind="exponential",
        ...     random_state=0,
        ... )
        >>> cr.causes
        ['a', 'b']
        >>> cr.cif(5.0, [[0.2, 0.5], [0.8, 0.5]], "a").round(2)
        array([0.39, 0.78])

        Only the first feature changes the risk of cause ``"a"``, and
        neither changes that of ``"b"``; the importances restricted to
        each cause say so:

        >>> cr.feature_importances(event="a", random_state=0).round(2)
        Z0    0.17
        Z1   -0.01
        Name: importance, dtype: float64
        >>> cr.feature_importances(event="b", random_state=0).round(2)
        Z0   -0.01
        Z1   -0.01
        Name: importance, dtype: float64
        """
        if Z is None:
            raise ValueError("The covariate matrix Z is required")
        if e is not None and c is None and xl is None and xr is None:
            # A row with a cause failed; one without is right censored
            from surpyval.utils.data_formats import is_missing_event

            labels: Any = e
            c = [1 if is_missing_event(v) else 0 for v in labels]
        data = SurpyvalData(
            x,
            c,
            n,
            t,
            xl=xl,
            xr=xr,
            tl=tl,
            tr=tr,
            group_and_sort=False,
            e=e,
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
            split=split,
            event=event,
        )

    def sf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        ensemble_method: str = "sf",
        *,
        grid: bool = False,
    ) -> NDArray:
        """Returns the ensemble survival function

        Parameters
        ----------
        x : int | float | ArrayLike
            Times, the same for every covariate vector.
        Z : ArrayLike | NDArray
            One covariate vector (1-D), or a matrix with one covariate
            vector per row (2-D), with one value per covariate the forest
            was grown on, in its column order: another number raises a
            ``ValueError`` (#657). A DataFrame is read by column name.
        ensemble_method : str, optional
            Determines whether to average across terminal nodes the terminal
            node survival functions or cumulative hazard functions.
            For these respectively, ensemble_method must be "sf" or
            "Hf". Defaults to "sf".
        grid : bool, optional
            ``False`` (the default) pairs row ``i`` of ``Z`` with ``x[i]``
            (a single row is used at every time, a single time for every
            row), as every regression model does (#666); other counts of
            rows and times are refused with a ``ValueError``. ``True``
            evaluates every time for every row of ``Z`` (a 1-D ``Z`` is
            one row), a survival curve per covariate vector.

        Returns
        -------
        NDArray
            For a 1-D ``Z`` (and no ``grid``), the survival function at
            ``x``, shaped like ``x`` (a scalar for a scalar ``x``).
            Paired, the shape of ``x`` (or ``(n_rows,)`` for a single
            time). On the grid, shape ``(n_rows,) + x.shape``, row ``i``
            the survival function for ``Z[i]``. A covariate vector with a
            missing (NaN) value gives NaN, and leaves the other rows
            unaffected.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.ml import RandomSurvivalForest
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 2))
        >>> x = rng.weibull(2.0, 200) * np.where(Z[:, 0] > 0.5, 5.0, 10.0)
        >>> forest = RandomSurvivalForest.fit(
        ...     x, Z, n_trees=5, max_depth=1, kind="exponential",
        ...     random_state=0,
        ... )
        >>> rows = [[0.2, 0.5], [0.8, 0.5]]
        >>> forest.sf([2.0, 5.0], rows).shape  # row i at time i
        (2,)
        >>> forest.sf(5.0, rows).shape  # one time for every row
        (2,)
        >>> forest.sf([2.0, 5.0, 8.0], rows[:1]).shape  # one row, every time
        (3,)
        >>> forest.sf([2.0, 5.0, 8.0], rows, grid=True).shape
        (2, 3)
        >>> forest.sf([2.0, 5.0, 8.0], rows)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Z has 2 covariate rows for 3 times: ... pass grid=True.
        """
        # Anything but 'Hf' used to be taken silently as 'sf'.
        check_option("ensemble_method", ensemble_method, ("sf", "Hf"))
        if ensemble_method == "Hf":
            Hf = self._apply_model_function_to_trees("Hf", x, Z, grid)
            return np.exp(-Hf)
        return self._apply_model_function_to_trees("sf", x, Z, grid)

    def ff(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """Failure (CDF) function averaged over the trees, as for
        :meth:`sf`."""
        return self._apply_model_function_to_trees("ff", x, Z, grid)

    def df(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """Density averaged over the trees, as for :meth:`sf`."""
        return self._apply_model_function_to_trees("df", x, Z, grid)

    def hf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """Hazard rate averaged over the trees, as for :meth:`sf`."""
        return self._apply_model_function_to_trees("hf", x, Z, grid)

    def Hf(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """Cumulative hazard averaged over the trees, as for :meth:`sf`."""
        return self._apply_model_function_to_trees("Hf", x, Z, grid)

    def _check_event(self, event: Any, method: str = "cif") -> None:
        if self.causes is None:
            raise ValueError(
                f"This forest was grown without causes, so {method} has no "
                "cause to take: grow it with e (fit(..., e=...)), or leave "
                "event out."
            )
        if event is None:
            raise ValueError(
                f"cif needs the cause: pass event, one of {self.causes} "
                "(total_cif gives every cause's together)."
            )
        if event not in self.causes:
            from surpyval.utils.validation import unknown_cause_error

            raise unknown_cause_error(event, self.causes)

    def cif(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        event: Any,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        Cumulative incidence of cause ``event`` at ``x``, averaged over the
        trees: the probability of having failed from that cause by ``x``,
        with the other causes acting. ``Z``, ``grid`` and the result are as
        for :meth:`sf`. Only for a forest grown with causes (``e``).
        """
        self._check_event(event)
        return self._apply_model_function_to_trees(
            "cif", x, Z, grid, event=event
        )

    def total_cif(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        *,
        grid: bool = False,
    ) -> NDArray:
        """
        The sum of every cause's cumulative incidence at ``x`` (the
        probability of having failed from any cause), as for :meth:`cif`.
        """
        if self.causes is None:
            self._check_event(None, "total_cif")
        assert self.causes is not None
        out = self.cif(x, Z, self.causes[0], grid=grid)
        for label in self.causes[1:]:
            out = out + self.cif(x, Z, label, grid=grid)
        return out

    def mortality(
        self,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        event: Any = None,
    ) -> ArrayLike:
        """
        The ensemble mortality of each covariate vector: its all-cause
        cumulative hazard summed over the times ``x`` (the risk score used
        by :meth:`score`). With ``event`` (a forest grown with causes),
        the cause's cumulative incidence summed over ``x`` instead, Ishwaran
        et al.'s (2014) cause-specific mortality: the expected time lost
        to the cause, in steps of the grid.
        """
        if event is not None:
            self._check_event(event, "mortality")
            values = self.cif(x, Z, event, grid=True)
            return np.atleast_2d(values).sum(1)
        mortality = np.atleast_2d(self.Hf(x, Z, grid=True)).sum(1)
        return np.clip(mortality, 0, np.finfo(np.float64).max)

    def _apply_model_function_to_trees(
        self,
        function_name: str,
        x: int | float | ArrayLike,
        Z: ArrayLike | NDArray,
        grid: bool = False,
        event: Any = None,
    ) -> NDArray:
        # ``event``: the cause, for "cif". The times flat; the result gets
        # their shape back (on its last axis for a grid), so a scalar time
        # gives a scalar.
        x, restore = flatten_query(x)
        if isinstance(Z, pd.DataFrame):
            # Read by the fitted names (or expanded by the formula)
            Z = prepare_Z(Z, self.feature_names, self._model_spec)
        single_covariant_vector = np.ndim(Z) < 2
        Z = np.array(Z, ndmin=2)
        # One value per covariate the forest was grown on (#657); each tree
        # would say so in its own terms.
        if self.trees:
            n_fitted = (
                self.Z.shape[1]
                if self.Z is not None
                else self.trees[0]._n_covariates()
            )
            if n_fitted is not None:
                check_covariate_count(
                    Z[0] if single_covariant_vector else Z,
                    n_fitted,
                    self.feature_labels,
                    "forest",
                )

        layout = query_layout(
            x.size, 1 if single_covariant_vector else 2, Z.shape[0], grid
        )
        if layout == "paired":
            # Each tree evaluates row i at time x[i] (#666)
            paired = np.zeros(x.size, dtype=np.float64)
            for tree in self.trees:
                paired += tree.apply_model_function(
                    function_name, x, Z, grid=False, event=event
                )
            return restore(paired / self.n_trees)

        # Each tree routes every row to its own leaf and returns an
        # (n_rows, x.size) grid
        res = np.zeros((Z.shape[0], x.size), dtype=np.float64)
        for tree in self.trees:
            res += tree.apply_model_function(
                function_name, x, Z, grid=True, event=event
            )
        res = res / self.n_trees
        if layout == "single" or layout == "row":
            return restore(res[0])
        if layout == "time":
            return res[:, 0]
        return restore(res, axis=-1)

    def score(
        self,
        x: ArrayLike,
        Z: ArrayLike | NDArray,
        c: ArrayLike,
        tie_tol: float = 1e-8,
        ties: str = "therneau",
        *,
        e: ArrayLike | None = None,
        event: Any = None,
    ) -> float:
        """Harrell's concordance index of the forest's mortality scores.

        The index is of the all-cause event: ``c`` is 0 for a failure from
        any cause. With ``e`` (each row's cause, as at fitting) and
        ``event``, it is the concordance of cause ``event``'s mortality
        (its cumulative incidence summed over ``x``, see :meth:`mortality`)
        with the failures from that cause, a failure from another cause
        counting as later than every failure from ``event`` (Wolbers et
        al., 2014): it never fails from it.

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
        if (e is None) != (event is None):
            raise ValueError(
                "A concordance of one cause needs both e (each row's cause) "
                "and event (the cause scored); pass both, or neither for "
                "the all-cause index."
            )
        if event is not None:
            self._check_event(event, "score")
            scores: ArrayLike = self.mortality(x, Z, event)
            if np.isnan(scores).any():
                return float("nan")
            x_k, c_k = _cause_outcomes(x, c, e, event)
            return concordance_index(x_k, c_k, scores, tie_tol, ties)
        scores = self.mortality(x, Z)
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
        terms = RowTerms(self.data, self.causes)
        return oob, terms, time_origin(self.data), n_oob

    def _oob_rows_log_likelihood(
        self,
        oob: list[NDArray],
        terms: RowTerms,
        origin: float,
        n_oob: NDArray,
        curves: dict,
        permute: tuple[int, Any] | None = None,
        event: int | None = None,
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
                event,
            )
        return row_log_likelihood(numerator, denominator, n_oob)

    def _event_code(self, event: Any) -> int | None:
        """The position of ``event`` among the causes (``None`` for the
        all-cause score), checked."""
        if event is None:
            return None
        self._check_event(event, "the out-of-bag score")
        assert self.causes is not None
        return self.causes.index(event)

    def oob_log_likelihood(self, event: Any = None) -> float:
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

        A forest grown with causes (``e``) is scored by the competing-risks
        likelihood: a failure from cause :math:`k` at :math:`x` contributes
        the cause's sub-density :math:`f_k(x)`, the slope of the ensemble's
        cumulative incidence :math:`F_k` of that cause there (the average
        of the trees' leaf sub-densities), so the score judges the cause of
        each failure as well as its time; a censored row contributes the
        all-cause :math:`S(x)`, and delayed entry divides by
        :math:`S(t_l)`. An Aalen-Johansen leaf is read continuously as
        above, each cause's incidence joined linearly between the
        all-cause drops. With ``event``, the score is restricted to that
        cause: the cause-specific likelihood, in which a failure from
        ``event`` contributes the cause's net density
        :math:`h_k(x) S_k(x)` and every other row (a censoring, or a
        failure from another cause) its net survival :math:`S_k(x)`, with
        delayed entry dividing by :math:`S_k(t_l)`; it judges the
        predictions of that cause's hazard only, and for one parametric
        tree the causes' restricted scores add up to the full one. See
        :mod:`~surpyval.ml.forest.oob`.

        Parameters
        ----------
        event : optional
            For a forest grown with causes, the cause to restrict the score
            to; ``None`` (the default) is the full competing-risks
            likelihood (or, without causes, the plain one).

        Returns
        -------
        float
            The mean out-of-bag log-likelihood per observation.

        Raises
        ------
        ValueError
            On a forest restored with ``from_dict``, which keeps neither
            the training data nor the bootstrap samples; or for an
            ``event`` the forest was not grown with.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval.ml import RandomSurvivalForest
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
        code = self._event_code(event)
        oob, terms, origin, n_oob = self._oob_setup()
        ll = self._oob_rows_log_likelihood(
            oob, terms, origin, n_oob, {}, event=code
        )
        _warn_zero_probability(ll, "the mean is -inf")
        return weighted_mean(ll, terms.n)

    def feature_importances(
        self, n_repeats: int = 5, random_state: Any = None, event: Any = None
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

        For a forest grown with causes the score is the competing-risks
        out-of-bag log-likelihood, so a feature's importance is what it
        adds to predicting the time and the cause of failure; with
        ``event``, the score restricted to that cause (see
        :meth:`oob_log_likelihood`), so the importance is what the feature
        adds to predicting that cause's hazard.

        Parameters
        ----------
        n_repeats : int, optional
            The number of shuffles averaged for each feature. Defaults
            to 5.
        random_state : None, int or numpy.random.Generator, optional
            Seeds the shuffles. ``None`` (the default) draws from NumPy's
            global random state, so ``np.random.seed`` reproduces it; a
            seed or ``Generator`` gives its own stream.
        event : optional
            For a forest grown with causes, the cause whose restricted
            score is used; ``None`` (the default) is the full
            competing-risks score (or, without causes, the plain one).

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
        >>> from surpyval.ml import RandomSurvivalForest
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
        code = self._event_code(event)
        rng = as_generator(random_state)
        oob, terms, origin, n_oob = self._oob_setup()
        curves: dict = {}
        ll0 = self._oob_rows_log_likelihood(
            oob, terms, origin, n_oob, curves, event=code
        )
        # The rows scored before the shuffle, and the shuffles' scorings
        # of them lost to a zero probability (#533).
        finite0 = np.isfinite(ll0)
        lost = 0
        importances = np.zeros(self.Z.shape[1])
        for j in range(self.Z.shape[1]):
            drops = []
            for _ in range(int(n_repeats)):
                ll = self._oob_rows_log_likelihood(
                    oob,
                    terms,
                    origin,
                    n_oob,
                    curves,
                    permute=(j, rng),
                    event=code,
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
            + (
                ""
                if self.causes is None
                else f"causes={self.causes!r}, split={self.split!r}, "
                + ("" if self.event is None else f"event={self.event!r}, ")
            )
            + f"features={self.feature_labels})"
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
        if self.causes is not None:
            out["causes"] = to_native(list(self.causes))
            out["split"] = self.split
            if self.event is not None:
                out["event"] = to_native(self.event)
        serialise_covariate_meta(self, out)
        # The trees are finished dictionaries already (#549)
        return stamp_schema(out, stamped=True)

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
        if "causes" in model_dict:
            forest.causes = [
                label_from_native(k) for k in model_dict["causes"]
            ]
            forest.split = model_dict["split"]
            forest.event = label_from_native(model_dict.get("event"))
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
