"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.univariate.regression import CoxPH
from surpyval.univariate.regression.regression_data import (
    check_finite_event_times,
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from surpyval.utils import (
    is_missing_event,
    validate_fine_gray_inputs,
    wrangle_and_check_form_and_Z_cols,
)
from surpyval.utils.deprecation import REMOVED_IN, renamed_arguments
from surpyval.utils.ipcw import step_at as _step
from surpyval.utils.shapes import keeps_query_shape

from .fine_gray import FineGray, FineGrayModel, paired_covariate_rows


class CompetingRisksProportionalHazards(SerialisableMixin):
    """
    Competing-risks proportional-hazards regression.

    Fits either a cause-specific proportional-hazards model (``model="Cox"``,
    one Cox model per cause with the other causes treated as censored) or a
    Fine-Gray subdistribution-hazards model (``model="Fine-Gray"``). The
    naming follows the package convention (compare ``CompetingRisks`` and
    ``ProportionalHazards``).

    Call the class method ``CompetingRisksProportionalHazards.fit`` (or
    ``fit_from_df``); it returns a fitted instance. Every prediction takes
    the covariates ``Z`` and, for one cause, its label ``event``: an array
    in the fitted column order or, for a model fitted with ``fit_from_df``,
    a DataFrame of the raw covariate columns (a ``formula`` is applied to
    it, as for ``CoxPH``). A fitted model can be saved with
    ``to_dict``/``to_json`` and restored with ``from_dict``/``from_json``
    (or ``surpyval.from_dict``).
    """

    # Populated by ``fit``; declared for the type checker. ``model`` is
    # ``"Cox"`` or ``"Fine-Gray"``, the ``model`` argument of ``fit``.
    model: str
    x: "npt.NDArray"
    #: Each cause's optimiser result, in ``event_idx_map`` order (``None``
    #: for a model restored from a dict: the optimiser objects are not
    #: serialised).
    results: "list | None"
    betas: "npt.NDArray"
    beta: "npt.NDArray"
    event_idx_map: dict
    n_event_types: int
    h0_e: "npt.NDArray"
    H0_e: "npt.NDArray"
    phi: Any
    phi_e: Any
    _fg_models: dict
    # Covariate metadata, set by ``fit_from_df``; ``None`` after ``fit``.
    feature_names: "list | None" = None
    formula: Any = None
    _model_spec: Any = None

    @property
    def how(self) -> str:
        """Deprecated: ``model``, the model fitted (``"Cox"`` or
        ``"Fine-Gray"``), under its old name."""
        warnings.warn(
            "CompetingRisksProportionalHazards.how is deprecated and will "
            "be removed in v{}; use .model.".format(REMOVED_IN),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.model

    def __dir__(self) -> list[str]:
        # The deprecated alias is left out of listings (tab completion,
        # anything that walks ``dir``), which would otherwise warn.
        return [name for name in super().__dir__() if name != "how"]

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted model to a plain, JSON-serialisable dict.

        Stores the causes (``event_idx_map``), the per-cause coefficients
        ``betas`` and the per-cause baseline step arrays on the shared time
        grid ``x``; for ``model="Fine-Gray"`` also each cause's
        :class:`FineGrayModel` (its own ``to_dict``), from which the
        Fine-Gray predictions come. The reloaded model reproduces every
        prediction (``cif``, ``sf``, ``ff``, ``Hf``, ``hf``, ``df``) for any
        ``Z``. The optimiser results (``results``) are not stored.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval
        >>> from surpyval.univariate.competing_risks import (
        ...     CompetingRisksProportionalHazards,
        ... )
        >>> x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> Z = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
        >>> e = ["a", "b", "a", None, "b", "a", "a", None, "b", "a"]
        >>> model = CompetingRisksProportionalHazards.fit(x, Z, e)
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.cif([5, 9], [1], "a"),
        ...                  model.cif([5, 9], [1], "a")))
        True
        """
        out: dict = {
            "model": "CompetingRisksProportionalHazards",
            # Stored under "how", the argument's old name, so files written
            # before the rename still load.
            "how": self.model,
            # list of [event, index] pairs to preserve the event key types
            "event_idx_map": [
                [to_native(k), int(v)] for k, v in self.event_idx_map.items()
            ],
            "n_event_types": int(self.n_event_types),
            "x": np.asarray(self.x, dtype=float).tolist(),
            "betas": np.asarray(self.betas, dtype=float).tolist(),
            "h0_e": np.asarray(self.h0_e, dtype=float).tolist(),
        }
        if self.model == "Fine-Gray":
            # The Fine-Gray predictions come from the per-cause models (the
            # shared grid only mirrors their baselines), so store them whole,
            # in ``event_idx_map`` order.
            out["fg_models"] = [
                self._fg_models[event].to_dict()
                for event in self.event_idx_map
            ]
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(
        cls, model_dict: dict
    ) -> "CompetingRisksProportionalHazards":
        """Rebuild a competing-risks proportional-hazards model from a
        :meth:`to_dict` dictionary."""
        require_model_tag(
            model_dict,
            "CompetingRisksProportionalHazards",
            "a competing-risks proportional-hazards model",
        )
        model = cls()
        model.model = model_dict["how"]
        model.event_idx_map = {
            label_from_native(k): int(v)
            for k, v in model_dict["event_idx_map"]
        }
        model.n_event_types = int(model_dict["n_event_types"])
        model.x = np.array(model_dict["x"], dtype=float)
        if model.model == "Fine-Gray":
            model._fg_models = {
                event: FineGrayModel.from_dict(fg)
                for event, fg in zip(
                    model.event_idx_map, model_dict["fg_models"]
                )
            }
        model.results = None
        model._finish(
            np.array(model_dict["betas"], dtype=float),
            np.array(model_dict["h0_e"], dtype=float),
        )
        restore_covariate_meta(model, model_dict)
        return model

    def _finish(self, betas: npt.NDArray, baselines: npt.NDArray) -> None:
        # The attributes derived from the per-cause coefficients and baseline
        # increments, shared by ``fit`` and ``from_dict`` so a reloaded model
        # is rebuilt exactly as the fitted one was.
        self.betas = betas
        self.beta = betas.sum(axis=0)
        self.phi_e = lambda Z, e_i: np.exp(
            self._prepare_Z(Z) @ self.betas[e_i, :]
        )
        self.phi = lambda Z: np.exp(self._prepare_Z(Z) @ self.beta)
        self.h0_e = baselines
        self.H0_e = baselines.cumsum(axis=1)

    def _prepare_Z(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        Convert ``Z`` to a numeric design matrix: a DataFrame is read by
        the covariate names (or expanded by the formula) recorded by
        ``fit_from_df`` -- it used to be read by column position, and a
        formula's raw columns were not expanded at all (#370); an array is
        taken as it is, in the fitted column order.
        """
        return prepare_Z(Z, self.feature_names, self._model_spec)

    def _fg_model(self, event: Any) -> Any:
        # Resolve the per-cause Fine-Gray subdistribution model, requiring an
        # explicit cause (the Fine-Gray CIF is defined one cause at a time).
        if event is None:
            raise ValueError(
                "A Fine-Gray model predicts one cause at a time; pass `event`."
            )
        if event not in self._fg_models:
            raise ValueError("Unrecognised event type for this model")
        return self._fg_models[event]

    def _f(
        self,
        arr: npt.NDArray,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        # The baseline step at each time, in the order the times were given,
        # so that one covariate row per time pairs row ``i`` with ``x[i]``
        # (the steps used to be read at the *sorted* times and multiplied by
        # the rows in the given order, mismatching them for unsorted ``x``).
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(self.x, x_arr, side="right") - 1
        # Query times before the first event have index -1, which would
        # otherwise wrap to the last step value; the step functions are all
        # zero there (#253). A missing (NaN) time is nan: searchsorted puts
        # it after the last event, the value at t = inf.
        base = np.where(idx[None, :] < 0, 0.0, arr[:, np.maximum(idx, 0)])
        base = np.where(np.isnan(x_arr), np.nan, base)

        if event is not None:
            if event not in self.event_idx_map:
                raise ValueError("Unrecognised event type for this model")
            e_i = self.event_idx_map[event]
            return base[e_i] * self.phi_e(Z, e_i)
        # All causes combined: each cause contributes with its OWN
        # coefficients, so the all-cause (cumulative) hazard is the sum of
        # H0_e(t) * exp(beta_e'Z), not a single summed-coefficient term.
        return sum(
            base[e_i] * self.phi_e(Z, e_i)
            for e_i in self.event_idx_map.values()
        )

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        """
        Cause-specific hazard increments at ``x`` for covariates ``Z``: one
        cause's (``event``) or the sum over causes (``event=None``). Not
        available for a Fine-Gray model.
        """
        if self.model == "Fine-Gray":
            raise ValueError(
                "The Fine-Gray subdistribution hazard has no pointwise "
                "density from the step baseline; use `cif` or `Hf`."
            )
        Z = self._prepare_Z(Z)
        return self._f(self.h0_e, x, Z, event=event, interp=interp)

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        """
        Cumulative hazard at ``x`` for covariates ``Z``: one cause's
        cause-specific cumulative hazard (``event``) or the all-cause sum
        (``event=None``). For a Fine-Gray model, the cumulative
        subdistribution hazard of ``event``.
        """
        Z = self._prepare_Z(Z)
        if self.model == "Fine-Gray":
            # Cumulative subdistribution hazard H0_k(x) * exp(beta'Z) = -log S.
            return -np.log(self.sf(x, Z, event=event))
        return self._f(self.H0_e, x, Z, event=event, interp=interp)

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        """
        :math:`e^{-H}` at ``x`` for covariates ``Z``: the all-cause survival
        (``event=None``), which is one minus the sum of the causes'
        :meth:`cif`, or one cause's net survival (the other causes
        treated as censoring). For a Fine-Gray model, ``1 - cif``.
        """
        Z = self._prepare_Z(Z)
        if self.model == "Fine-Gray":
            return self._fg_model(event).sf(x, Z)
        return np.exp(-self.Hf(x, Z, event=event, interp=interp))

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        """
        ``1 - sf`` at ``x`` for covariates ``Z``. For a Fine-Gray model,
        the cumulative incidence of ``event``.
        """
        Z = self._prepare_Z(Z)
        if self.model == "Fine-Gray":
            return self.cif(x, Z, event)
        return 1 - self.sf(x, Z, event=event, interp=interp)

    @keeps_query_shape
    def df(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        """
        ``hf * sf`` at ``x`` for covariates ``Z``. Not available for a
        Fine-Gray model.
        """
        if self.model == "Fine-Gray":
            raise ValueError(
                "The Fine-Gray subdistribution density has no pointwise form "
                "from the step baseline; use `cif`."
            )
        Z = self._prepare_Z(Z)
        return self.hf(x, Z, event=event, interp=interp) * self.sf(
            x, Z, event=event, interp=interp
        )

    @keeps_query_shape
    def cif(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, event: Any
    ) -> npt.NDArray:
        """
        Cumulative incidence of cause ``event`` at ``x`` for covariates
        ``Z``: the probability of failing from that cause by ``x`` with the
        other causes acting. The cause-specific (``model="Cox"``) model
        builds it step by step from the causes' hazard increments, as R's
        ``survfit`` does for a multi-state ``coxph`` (the Aalen-Johansen
        estimate with each step's matrix exponential), so the causes'
        incidences sum to ``ff = 1 - exp(-H)``; the Fine-Gray model
        evaluates the subdistribution directly.

        ``Z`` is one covariate vector (a 1-D array or a single row), used
        at every time, or one row per time in ``x`` (row ``i`` with
        ``x[i]``), as for :meth:`sf` and :meth:`Hf`; a DataFrame of raw
        covariates for a model fitted with ``fit_from_df``. ``event`` must
        be one of the fitted causes.
        """
        if event is None or event not in self.event_idx_map:
            causes = list(self.event_idx_map)
            raise ValueError(
                f"`event` must be one of the fitted causes {causes}, got "
                f"{event!r}."
            )
        Z = self._prepare_Z(Z)
        if self.model == "Fine-Gray":
            # Direct subdistribution CIF: 1 - exp(-H0_k(x) exp(beta'Z)).
            return self._fg_model(event).cif(x, Z)

        e_i = self.event_idx_map[event]

        def incidence(z: npt.NDArray) -> npt.NDArray:
            return self._incidence_steps(z)[e_i].cumsum()

        return self._per_covariate_row(x, Z, incidence, 0.0)

    def _per_covariate_row(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, curve: Any, start: float
    ) -> npt.NDArray:
        """A step function of the shared time grid ``self.x`` that depends
        on the covariates, ``curve(z)``, read at each time in ``x`` with
        its paired covariate row: ``Z`` is one row for every time or one
        row per time. ``start`` is its value before the first time."""
        x_flat = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        rows = paired_covariate_rows(Z, x_flat.size, self.betas.shape[1])
        out = np.empty(x_flat.size)
        # One curve per distinct covariate row, read at the times paired
        # with that row.
        uniq, inverse = np.unique(rows, axis=0, return_inverse=True)
        inverse = np.ravel(inverse)
        for u, z in enumerate(uniq):
            at = np.flatnonzero(inverse == u)
            values = curve(z)
            idx = np.searchsorted(self.x, x_flat[at], side="right") - 1
            # Times before the first event would wrap to the last value
            # (#253).
            out[at] = np.where(idx < 0, start, values[np.maximum(idx, 0)])
        # A missing (NaN) time is nan, not the value at t = inf.
        return np.where(np.isnan(x_flat), np.nan, out)

    def _incidence_steps(self, Z: npt.ArrayLike) -> npt.NDArray:
        """
        Each cause's step in cumulative incidence at the event times, for
        one covariate row: an array of shape ``(n_causes, len(self.x))``.

        Over a step the causes' hazards add ``dH_k`` each and ``dH`` in
        all, and the transition probabilities are those of the matrix
        exponential of the step's hazards, as R's ``survfit`` computes
        them for a multi-state ``coxph``: a unit still event-free before
        the step, with probability ``S(t-) = exp(-H(t-))``, fails from
        cause ``k`` over it with probability
        ``S(t-) (dH_k / dH) (1 - exp(-dH))``. The incidences then sum to
        ``1 - exp(-H) = ff`` exactly, and a step stays a probability
        however large the increment (a Breslow increment times a large
        multiplier can exceed 1). The incidence used to be built on the
        product-limit survival ``prod (1 - dH)`` while ``sf`` is
        ``exp(-H)``, so the two disagreed (1.0 against 0.975, #384); the
        product ``prod (1 - dH)`` also needs a clip where ``dH > 1``.
        """
        increments = np.array(
            [
                np.broadcast_to(
                    self.h0_e[e_i] * self.phi_e(Z, e_i), self.x.shape
                )
                for e_i in range(self.n_event_types)
            ],
            dtype=float,
        )
        total = increments.sum(axis=0)
        # exp(-H(t-)): the survival just before each step.
        before = np.exp(-np.concatenate([[0.0], np.cumsum(total)[:-1]]))
        positive = total > 0
        share = np.where(
            positive, increments / np.where(positive, total, 1.0), 0.0
        )
        return before * -np.expm1(-total) * share

    @classmethod
    @renamed_arguments(how="model")
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        formula: "str | None" = None,
        model: str = "Cox",
        tie_method: str = "efron",
    ) -> "CompetingRisksProportionalHazards":
        """
        Fit a competing-risks proportional-hazards model from a pandas
        DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            Column of observed times.
        e_col : str
            Column of event-type (cause) labels. Use ``None`` (or a blank/NaN
            cell) for a censored observation.
        Z_cols : str or list of str, optional
            Covariate columns. Either ``Z_cols`` or ``formula`` must be given.
        c_col : str, optional
            Column of censoring flags (0 observed, 1 right-censored).
        n_col : str, optional
            Column of counts per row.
        formula : str, optional
            A patsy/formulaic formula for the covariates, as an alternative to
            ``Z_cols``.
        model : {'Cox', 'Fine-Gray'}, optional
            Cause-specific proportional hazards or Fine-Gray subdistribution
            hazards. Default 'Cox'.
        tie_method : str, optional
            Tie handling for the ``model='Cox'`` path, passed to
            :meth:`CoxPH.fit`: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (alias ``'kp'``).

        Returns
        -------
        CompetingRisksProportionalHazards
            The fitted model. Its prediction methods take a DataFrame of
            the raw covariate columns (the ``formula`` is applied to it) or
            a covariate array in the fitted column order.

        Examples
        --------
        A categorical covariate through a formula; the model predicts
        from a DataFrame of raw covariates, before and after saving:

        >>> import numpy as np
        >>> import pandas as pd
        >>> import surpyval
        >>> from surpyval.univariate.competing_risks import (
        ...     CompetingRisksProportionalHazards,
        ... )
        >>> rng = np.random.default_rng(1)
        >>> g = rng.choice(["a", "b", "c"], 300)
        >>> rate = 0.1 * np.exp(np.select([g == "b", g == "c"], [0.8, -0.5]))
        >>> t_a = rng.exponential(1 / rate)
        >>> t_b = rng.exponential(1 / 0.05, 300)
        >>> df = pd.DataFrame({
        ...     "time": np.minimum(t_a, t_b).round(3),
        ...     "cause": np.where(t_a < t_b, "a", "b"),
        ...     "g": g,
        ... })
        >>> model = CompetingRisksProportionalHazards.fit_from_df(
        ...     df, "time", "cause", formula="g"
        ... )
        >>> model.feature_names
        ['g[T.b]', 'g[T.c]']
        >>> new = pd.DataFrame({"g": ["a", "b", "c"]})
        >>> model.cif(np.full(3, 5.0), new, "a").round(4)
        array([0.3752, 0.6449, 0.2551])
        >>> restored = surpyval.from_dict(model.to_dict())
        >>> bool(np.allclose(restored.cif(np.full(3, 5.0), new, "a"),
        ...                  model.cif(np.full(3, 5.0), new, "a")))
        True
        """
        Z, mask, form, feature_names, model_spec = (
            wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)
        )
        sub = df.loc[mask]
        x = sub[x_col].values
        # A censored row's cause is ``None``; accept a blank/NaN cell for it.
        e = sub[e_col].to_numpy(dtype=object).copy()
        e[[is_missing_event(v) for v in e]] = None
        c = sub[c_col].values if c_col is not None else None
        n = sub[n_col].values if n_col is not None else None

        fitted = cls.fit(x, Z, e, c=c, n=n, model=model, tie_method=tie_method)
        fitted.formula = form
        fitted.feature_names = feature_names
        fitted._model_spec = model_spec
        return fitted

    @classmethod
    @renamed_arguments(how="model")
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        model: str = "Cox",
        tie_method: str = "efron",
    ) -> "CompetingRisksProportionalHazards":
        r"""
        Fit the competing-risks proportional-hazards model.

        Parameters
        ----------

        x : array like
            Failure or censoring times.

        Z : ndarray like
            Covariate matrix, one row per observation. Rows with a missing
            (``NaN``) or infinite covariate are dropped, with a warning.

        e : array like
            The cause of each failure; ``None`` (or ``NaN``) for a
            right-censored observation.

        c : array like, optional
            Censoring flags: 0 a failure, 1 right-censored. Derived from
            ``e`` if not given (a missing cause is censored). Left and
            interval censoring are not supported.

        n : array like, optional
            Array of counts for each x. If data is provided as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.

        model : {'Cox', 'Fine-Gray'}, optional
            ``'Cox'`` (default) fits cause-specific proportional hazards --
            one Cox model per cause, the other causes treated as censored;
            ``'Fine-Gray'`` fits one subdistribution-hazards model per cause.

        tie_method : str, optional
            Tie handling for the ``'Cox'`` path, passed to
            :meth:`CoxPH.fit`. Default ``'efron'``.

        Returns
        -------

        model : CompetingRisksProportionalHazards
            A competing-risks proportional-hazards model. ``betas`` holds one
            row of coefficients per cause, in the order of ``event_idx_map``
            (causes sorted); ``phi_e(Z, i)`` is cause ``i``'s hazard
            multiplier. ``beta`` and ``phi`` (the sum of the per-cause
            coefficients and its multiplier) are kept for backward
            compatibility but are not a model quantity: every prediction
            uses the per-cause coefficients.

        Examples
        --------
        Two causes; the covariate doubles cause ``a``'s hazard and leaves
        cause ``b``'s alone:

        >>> from surpyval.univariate.competing_risks import (
        ...     CompetingRisksProportionalHazards,
        ... )
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
        >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * Z[:, 0])))
        >>> t_b = rng.exponential(1 / 0.05, 200)
        >>> t_c = rng.uniform(0, 20, 200)  # censoring times
        >>> x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
        >>> first = np.where(t_a < t_b, "a", "b")
        >>> e = np.where(t_c < np.minimum(t_a, t_b), None, first)
        >>> model = CompetingRisksProportionalHazards.fit(x, Z, e)
        >>> model.betas.round(3)
        array([[0.985],
               [0.005]])
        >>> model.cif([5, 10], [[1]], "a").round(4)
        array([0.59  , 0.7369])
        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)
        check_finite_event_times(x, c)

        # A fixed order for the causes (a set's iteration order depends on
        # the hash seed for strings), so ``betas`` rows are reproducible.
        causes = ordered_labels(e)
        if not causes:
            raise ValueError("No observed events: every row is censored.")

        n_event_types = len(causes)

        event_idx_map = {state: i for i, state in enumerate(causes)}

        betas = np.zeros((n_event_types, Z.shape[1]))
        unique_x = np.unique(x)

        baselines = np.zeros((n_event_types, len(unique_x)))
        # Best initial assumption is to assume there is no risk
        # beta_init = np.zeros(Z.shape[1])

        out = cls()
        out.n_event_types = n_event_types
        out.event_idx_map = event_idx_map
        out.model = model

        if model == "Cox":
            # Cause-specific proportional hazards: one Cox model per cause,
            # treating every other cause (and censoring) as right-censored.
            results = []
            for i, event in enumerate(causes):
                c_e = np.where(label_mask(e, event), 0, 1)
                cox_model = CoxPH.fit(x, Z, c_e, n, tie_method=tie_method)

                results.append(cox_model.res)
                betas[i, :] = cox_model.res.x
                # Cause-specific baseline hazard: reuse the fitted Cox model's
                # own baseline (Efron's after an Efron fit, else Breslow's),
                # which is built from c_e (the cause-specific event
                # indicator) and the standard risk set.
                # Map its cumulative hazard onto the shared unique_x grid and
                # store increments so H0_e = baselines.cumsum stays coherent.
                H_grid = _step(cox_model.x, cox_model.H0, unique_x, before=0.0)
                baselines[i, :] = np.diff(H_grid, prepend=0.0)

        elif model == "Fine-Gray":
            # Delegate to the IPCW Fine-Gray fitter, one subdistribution model
            # per cause. The authoritative predictions come from these models
            # (see ``_fg_models`` and the ``cif``/``sf`` branches below); the
            # ``baselines`` grid is filled with each cause's cumulative
            # subdistribution hazard for a coherent ``H0_e``.
            fg_models = {}
            results = []
            for i, event in enumerate(causes):
                fg = FineGray.fit(x, Z, e, c=c, n=n, event=event)
                fg_models[event] = fg
                results.append(fg.res)
                betas[i, :] = fg.beta
                # Store increments so the shared ``H0_e = baselines.cumsum``
                # equals this cause's cumulative subdistribution hazard.
                H_grid = _step(fg._times, fg._cumhaz, unique_x, before=0.0)
                baselines[i, :] = np.diff(H_grid, prepend=0.0)
            out._fg_models = fg_models
        else:
            raise ValueError("`model` must be either 'Cox' or 'Fine-Gray'")

        out.results = results
        out._finish(betas, baselines)
        out.x = unique_x
        return out
