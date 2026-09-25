"""
This code was created for and sponsored by Cartiga (www.cartiga.com).
Cartiga makes no representations or warranties in connection with the code
and waives any and all liability in connection therewith. Your use of the
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_iif,
)
from surpyval.univariate.regression import CoxPH
from surpyval.univariate.regression.regression_data import (
    restore_covariate_meta,
    serialise_covariate_meta,
)
from surpyval.utils import (
    _get_idx,
    validate_fine_gray_inputs,
    wrangle_and_check_form_and_Z_cols,
)
from surpyval.utils.ipcw import step_at as _step

from .fine_gray import FineGray, FineGrayModel


class CompetingRisksProportionalHazards(SerialisableMixin):
    """
    Competing-risks proportional-hazards regression.

    Fits either a cause-specific proportional-hazards model (``how="Cox"``,
    one Cox model per cause with the other causes treated as censored) or a
    Fine-Gray subdistribution-hazards model (``how="Fine-Gray"``). The naming
    follows the package convention (compare ``CompetingRisks`` and
    ``ProportionalHazards``).

    Call the class method ``CompetingRisksProportionalHazards.fit`` (or
    ``fit_from_df``); it returns a fitted instance. Every prediction takes
    the covariates ``Z`` and, for one cause, its label ``event``. A fitted
    model can be saved with ``to_dict``/``to_json`` and restored with
    ``from_dict``/``from_json`` (or ``surpyval.from_dict``).
    """

    # Populated by ``fit``; declared for the type checker.
    how: str
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

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted model to a plain, JSON-serialisable dict.

        Stores the causes (``event_idx_map``), the per-cause coefficients
        ``betas`` and the per-cause baseline step arrays on the shared time
        grid ``x``; for ``how="Fine-Gray"`` also each cause's
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
            "how": self.how,
            # list of [event, index] pairs to preserve the event key types
            "event_idx_map": [
                [to_native(k), int(v)] for k, v in self.event_idx_map.items()
            ],
            "n_event_types": int(self.n_event_types),
            "x": np.asarray(self.x, dtype=float).tolist(),
            "betas": np.asarray(self.betas, dtype=float).tolist(),
            "h0_e": np.asarray(self.h0_e, dtype=float).tolist(),
        }
        if self.how == "Fine-Gray":
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
        model.how = model_dict["how"]
        model.event_idx_map = {
            k: int(v) for k, v in model_dict["event_idx_map"]
        }
        model.n_event_types = int(model_dict["n_event_types"])
        model.x = np.array(model_dict["x"], dtype=float)
        if model.how == "Fine-Gray":
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
        self.phi_e = lambda Z, e_i: np.exp(Z @ self.betas[e_i, :])
        self.phi = lambda Z: np.exp(Z @ self.beta)
        self.h0_e = baselines
        self.H0_e = baselines.cumsum(axis=1)

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
        idx, rev = _get_idx(self.x, x)

        if event is not None:
            if event not in self.event_idx_map:
                raise ValueError("Unrecognised event type for this model")
            e_i = self.event_idx_map[event]
            out = (arr[e_i, idx] * self.phi_e(Z, e_i))[rev]
        else:
            # All causes combined: each cause contributes with its OWN
            # coefficients, so the all-cause (cumulative) hazard is the sum
            # of H0_e(t) * exp(beta_e'Z), not a single summed-coefficient
            # term.
            total = sum(
                arr[e_i, idx] * self.phi_e(Z, e_i)
                for e_i in self.event_idx_map.values()
            )
            out = total[rev]
        # Query times before the first event have index -1, which would
        # otherwise wrap to the last step value; the step functions are all
        # zero there (#253).
        return np.where(idx[rev] < 0, 0.0, out)

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
        if self.how == "Fine-Gray":
            raise ValueError(
                "The Fine-Gray subdistribution hazard has no pointwise "
                "density from the step baseline; use `cif` or `Hf`."
            )
        return self._f(self.h0_e, x, Z, event=event, interp=interp)

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
        if self.how == "Fine-Gray":
            # Cumulative subdistribution hazard H0_k(x) * exp(beta'Z) = -log S.
            return -np.log(self.sf(x, Z, event=event))
        return self._f(self.H0_e, x, Z, event=event, interp=interp)

    def sf(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        event: Any = None,
        interp: str = "step",
    ) -> npt.NDArray:
        """
        :math:`e^{-H}` at ``x`` for covariates ``Z``: the all-cause survival
        (``event=None``) or one cause's net survival (the other causes
        treated as censoring). For a Fine-Gray model, ``1 - cif``.
        """
        if self.how == "Fine-Gray":
            return self._fg_model(event).sf(x, Z)
        return np.exp(-self.Hf(x, Z, event=event, interp=interp))

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
        if self.how == "Fine-Gray":
            return self.cif(x, Z, event)
        return 1 - self.sf(x, Z, event=event, interp=interp)

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
        if self.how == "Fine-Gray":
            raise ValueError(
                "The Fine-Gray subdistribution density has no pointwise form "
                "from the step baseline; use `cif`."
            )
        return self.hf(x, Z, event=event, interp=interp) * self.sf(
            x, Z, event=event, interp=interp
        )

    def cif(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, event: Any
    ) -> npt.NDArray:
        """
        Cumulative incidence of cause ``event`` at ``x`` for covariates
        ``Z``: the probability of failing from that cause by ``x`` with the
        other causes acting. The cause-specific (``how="Cox"``) model
        integrates the cause's hazard against the all-cause product-limit
        survival; the Fine-Gray model evaluates the subdistribution
        directly.
        """
        if self.how == "Fine-Gray":
            # Direct subdistribution CIF: 1 - exp(-H0_k(x) exp(beta'Z)).
            return self._fg_model(event).cif(x, Z)

        # Cause-specific CIF: integrate this cause's hazard against the
        # all-cause survival. Index and reverse index in case x is unordered.
        idx, rev = _get_idx(self.x, x)

        S, shares = self._product_limit_survival(Z)
        e_i = self.event_idx_map[event]
        cif = aalen_johansen_iif(S, shares[e_i]).cumsum()

        # Times before the first event would wrap to the last value (#253).
        return np.where(idx[rev] < 0, 0.0, cif[idx][rev])

    def _product_limit_survival(
        self, Z: npt.ArrayLike
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        All-cause survival at the event times as a product limit, and each
        cause's share of the hazard increment, for the incidence weights.

        Only the product-limit survival ``prod (1 - dH(t_j))`` satisfies the
        telescoping identity ``sum_j S(t_j-) dH(t_j) = 1 - S(t)``, so weighting
        the cause-specific increments with ``exp(-H)`` inflated the incidence
        and let the causes sum past 1 (#278). A Breslow increment can also
        exceed 1 at a covariate value far from the data (a small risk set
        times a large multiplier); such a step exhausts the survivors, and
        each cause takes its proportional share of them. The causes'
        incidences then sum to exactly ``1 - S``.

        Returns ``(S, shares)`` with ``shares[e]`` cause ``e``'s effective
        hazard increments.
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
        scale = np.where(total > 1.0, 1.0 / np.where(total > 0, total, 1), 1.0)
        shares = increments * scale
        S = np.cumprod(1.0 - shares.sum(axis=0))
        return np.clip(S, 0.0, 1.0), shares

    @classmethod
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        formula: "str | None" = None,
        how: str = "Cox",
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
        how : {'Cox', 'Fine-Gray'}, optional
            Cause-specific proportional hazards or Fine-Gray subdistribution
            hazards. Default 'Cox'.
        tie_method : str, optional
            Tie handling for the ``how='Cox'`` path, passed to
            :meth:`CoxPH.fit`: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (alias ``'kp'``).

        Returns
        -------
        CompetingRisksProportionalHazards
            The fitted model. Predictions still take a covariate array ``Z``.
        """
        import pandas as pd

        Z, mask, form, feature_names, model_spec = (
            wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)
        )
        sub = df.loc[mask]
        x = sub[x_col].values
        # A censored row's cause is ``None``; accept a blank/NaN cell for it.
        e = np.array(
            [None if pd.isna(v) else v for v in sub[e_col].values],
            dtype=object,
        )
        c = sub[c_col].values if c_col is not None else None
        n = sub[n_col].values if n_col is not None else None

        model = cls.fit(x, Z, e, c=c, n=n, how=how, tie_method=tie_method)
        model.formula = form
        model.feature_names = feature_names
        model._model_spec = model_spec
        return model

    @classmethod
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        e: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        how: str = "Cox",
        tie_method: str = "efron",
    ) -> "CompetingRisksProportionalHazards":
        r"""
        Fit the competing-risks proportional-hazards model.

        Parameters
        ----------

        x : array like
            Failure or censoring times.

        Z : ndarray like
            Covariate matrix, one row per observation.

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

        how : {'Cox', 'Fine-Gray'}, optional
            ``'Cox'`` (default) fits cause-specific proportional hazards --
            one Cox model per cause, the other causes treated as censored;
            ``'Fine-Gray'`` fits one subdistribution-hazards model per cause.

        tie_method : str, optional
            Tie handling for the ``'Cox'`` path, passed to
            :meth:`CoxPH.fit` as its ``method``. Default ``'efron'``.

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
        array([0.5922, 0.7401])
        """
        x, Z, e, c, n = validate_fine_gray_inputs(x, Z, e, c, n)

        unique_e = set(e)
        if None in unique_e:
            unique_e.remove(None)
        # A fixed order for the causes (a set's iteration order depends on
        # the hash seed for strings), so ``betas`` rows are reproducible.
        try:
            causes = sorted(unique_e)
        except TypeError:
            causes = sorted(unique_e, key=lambda v: (type(v).__name__, str(v)))

        n_event_types = len(causes)

        event_idx_map = {state: i for i, state in enumerate(causes)}

        betas = np.zeros((len(unique_e), Z.shape[1]))
        unique_x = np.unique(x)

        baselines = np.zeros((len(unique_e), len(unique_x)))
        # Best initial assumption is to assume there is no risk
        # beta_init = np.zeros(Z.shape[1])

        model = cls()
        model.n_event_types = n_event_types
        model.event_idx_map = event_idx_map
        model.how = how

        if how == "Cox":
            # Cause-specific proportional hazards: one Cox model per cause,
            # treating every other cause (and censoring) as right-censored.
            results = []
            for i, event in enumerate(causes):
                c_e = np.where(e == event, 0, 1)
                cox_model = CoxPH.fit(x, Z, c_e, n, method=tie_method)

                results.append(cox_model.res)
                betas[i, :] = cox_model.res.x
                # Cause-specific baseline hazard: reuse the fitted Cox model's
                # own Breslow baseline, which is built from c_e (the
                # cause-specific event indicator) and the standard risk set.
                # Map its cumulative hazard onto the shared unique_x grid and
                # store increments so H0_e = baselines.cumsum stays coherent.
                H_grid = _step(cox_model.x, cox_model.H0, unique_x, before=0.0)
                baselines[i, :] = np.diff(H_grid, prepend=0.0)

        elif how == "Fine-Gray":
            # Delegate to the IPCW Fine-Gray fitter, one subdistribution model
            # per cause. The authoritative predictions come from these models
            # (see ``_fg_models`` and the ``cif``/``sf`` branches below); the
            # ``baselines`` grid is filled with each cause's cumulative
            # subdistribution hazard for a coherent ``H0_e``.
            fg_models = {}
            results = []
            for i, event in enumerate(causes):
                fg = FineGray.fit(x, Z, e, c=c, n=n, cause=event)
                fg_models[event] = fg
                results.append(fg.res)
                betas[i, :] = fg.beta
                # Store increments so the shared ``H0_e = baselines.cumsum``
                # equals this cause's cumulative subdistribution hazard.
                H_grid = _step(fg._times, fg._cumhaz, unique_x, before=0.0)
                baselines[i, :] = np.diff(H_grid, prepend=0.0)
            model._fg_models = fg_models
        else:
            raise ValueError("`how` must be either 'Cox' or 'Fine-Gray")

        model.results = results
        model._finish(betas, baselines)
        model.x = unique_x
        return model
