from __future__ import annotations

import types
import warnings
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.serialisation import SerialisableMixin, stamp_schema
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils.data_summary import data_summary
from surpyval.utils.deprecation import CallableList, RenamedAttribute
from surpyval.utils.linalg import (
    cb_link,
    delta_method_se,
    link_band,
    log_transformed_cb,
    numerical_hessian,
    sf_link_from_H,
    wald_bound_on_support,
)
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)

from ._bounds import logit_sf_bound
from ._concordance import ConcordanceMixin
from .regression_data import (
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


# Regression families whose fitted model round-trips through ``to_dict`` /
# ``from_dict``: each has a fixed-form covariate link (a log-linear multiplier
# ``exp(beta'Z)`` or an additive ``beta'Z`` term) that is fully determined by
# the ``kind`` plus the distribution and coefficients, so the fitter -- and
# therefore every prediction -- can be rebuilt from the distribution's name.
# Maps kind -> (public fitter factory name, covariate-link form).
_SERIALISABLE_KINDS: "dict[str, tuple[str, str]]" = {
    "Accelerated Failure Time": ("AFT", "exp"),
    "Proportional Hazard": ("PH", "exp"),
    "Proportional Odds": ("PO", "exp"),
    "Additive Hazard": ("AH", "additive"),
}

# The covariate-link (``reg_model``) names those families produce. A model
# carrying any other link is a bespoke/custom link that cannot be rebuilt from
# a name alone, so serialisation refuses it rather than round-trip it wrongly.
_SERIALISABLE_REG_NAMES = {
    "Log Linear [exp(beta'Z)]",  # AFT, PO
    "Log Linear [e^(beta'Z)]",  # PH
    "Additive [beta'Z]",  # AH
}


class ParametricRegressionModel(
    ConcordanceMixin, InformationCriteriaMixin, SerialisableMixin
):
    """
    The fitted model returned by every parametric regression fitter: the
    proportional hazards (``WeibullPH``, ``PH(dist)``), accelerated failure
    time (``AFT``), proportional odds (``PO``), parametric additive hazards
    (``AH``) and accelerated life (``AcceleratedLife``) families.

    ``params`` holds the distribution parameters followed by the covariate
    coefficients (``dist_params`` and ``phi_params`` split them), named in
    order by ``parameter_names``. In an accelerated life model the life
    parameter (``life_parameter``, e.g. the Weibull's ``alpha``) is not
    estimated: the life model gives it at each stress, and its slot in
    ``params`` holds a placeholder 1, which the printed model does not show
    as a value. The
    survival functions take the covariates as a second argument,
    ``sf(x, Z)``; ``sf_tvc`` / ``Hf_tvc`` evaluate them along a
    time-varying covariate path. The model also provides parameter
    standard errors and confidence bounds, information criteria, plotting
    and serialisation.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Weibull, WeibullPH
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> model = WeibullPH.fit(x, Z)
    >>> model.params.round(3)
    array([9.629, 1.751, 0.829])
    >>> model.sf(5, [[0], [1]]).round(4)
    array([0.728 , 0.4833])
    """

    # Covariate metadata populated when the model is fit from a pandas
    # DataFrame (see ``DataFrameRegressionMixin.fit_from_df``). These defaults
    # keep the array based interface working unchanged.
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None
    #: Set only on models rebuilt by :meth:`from_dict` that carried a stored
    #: parameter covariance; lets them produce confidence bounds without the
    #: original data. ``None`` on freshly fitted models.
    _restored_covariance: "npt.NDArray | None" = None
    #: True on models rebuilt by :meth:`from_dict`, which carry no data.
    _restored: bool = False
    #: The printout's "Data" line of a model rebuilt by :meth:`from_dict`
    #: (#508).
    _data_summary: "str | None" = None
    #: The covariate point the baseline parameters are at: zeros (or
    #: ``None``, for an accelerated life model) when they are those of a
    #: unit with ``Z = 0``, the default. A fit with ``center=True`` keeps
    #: its baseline at the ``n``-weighted covariate means (#463), stored
    #: here, and every prediction uses ``Z - center``.
    center: "npt.NDArray | None" = None
    #: ``(params, center, jacobian)`` of the centred fit behind a model
    #: that reports its baseline at 0 (the log-linear families whose
    #: baseline maps exactly between the two, #463): the covariance and
    #: the confidence bounds are computed there, where the parameters are
    #: well conditioned, and carried to ``params`` by the jacobian of the
    #: map.
    _fit_centring: "tuple | None" = None
    #: ``(point, H)``: the exact (autograd) Hessian of the negative
    #: log-likelihood in the free natural parameters at ``point`` (see
    #: ``_covariance_point``), kept by the fit
    #: (``_fit_skeleton.keep_information``) for ``_observed_covariance``.
    _information: "tuple | None" = None
    #: ``(point, covariance)`` of the last covariance computed.
    _covariance_cache: "tuple | None" = None

    # Attributes populated after construction (by ``fit`` / ``from_params``).
    # Declared here so static type checkers know their types.
    params: npt.NDArray
    dist_params: npt.NDArray
    phi_params: npt.NDArray
    k: int
    k_dist: int
    gamma: float
    p: float
    f0: float
    kind: str
    fixed: dict[str, float]
    dist: Any
    distribution: Any
    distribution_param_map: Any
    phi_param_map: Any
    reg_model: Any
    model: Any
    data: Any
    res: Any
    fun: Any
    _neg_ll: float
    _bic: float
    #: Set by the AFT time-varying-covariate fit; absent otherwise.
    is_tvc: bool
    n_subjects: int
    _aic: float
    _aic_c: float

    # -- serialisation -----------------------------------------------------

    def _serialise_link(self) -> "dict[str, Any]":
        """The link-identity head of :meth:`to_dict`.

        Encodes just enough to rebuild the covariate link: for the fixed-form
        families the link name; for Accelerated Life the built-in life-model
        name. Raises ``NotImplementedError`` for any link that cannot be
        reconstructed from a name.
        """
        phi_param_map = getattr(self.reg_model, "phi_param_map", None)
        if not isinstance(phi_param_map, dict):
            raise NotImplementedError(
                "This model's covariate coefficients are not a fixed name map "
                "and cannot be serialised."
            )
        reg_name = getattr(self.reg_model, "name", None)
        base: dict[str, Any] = {
            "parameterization": "parametric-regression",
            "kind": self.kind,
            "distribution": self.distribution.name,
            "phi_param_map": {
                str(k): int(v) for k, v in phi_param_map.items()
            },
        }

        if self.kind == "Accelerated Life":
            from surpyval.univariate.regression.accelerated_life import (
                LIFE_MODELS,
            )

            if reg_name not in LIFE_MODELS:
                raise NotImplementedError(
                    "Serialisation of an Accelerated Life model requires a "
                    "built-in life model (one of {}); the {!r} life model "
                    "cannot be rebuilt from a name.".format(
                        sorted(LIFE_MODELS), reg_name
                    )
                )
            base["life_model_name"] = reg_name
            return base

        if self.kind not in _SERIALISABLE_KINDS:
            raise NotImplementedError(
                "Serialisation is implemented for the fixed-form regression "
                "families (Accelerated Failure Time, Proportional Hazard, "
                "Proportional Odds, Additive Hazard) and Accelerated Life; "
                "the {!r} model's covariate link cannot be rebuilt from a "
                "name.".format(self.kind)
            )
        if reg_name not in _SERIALISABLE_REG_NAMES:
            raise NotImplementedError(
                "This {} model carries a non-standard covariate link ({!r}) "
                "that cannot be serialised; only the built-in log-linear / "
                "additive links round-trip.".format(self.kind, reg_name)
            )
        base["reg_model_name"] = reg_name
        return base

    def to_dict(self) -> dict:
        """
        Serialise this fitted regression model to a plain ``dict``.

        The returned dictionary is JSON-serialisable and captures everything
        needed to rebuild the model for prediction (``sf``/``ff``/``df``/
        ``hf``/``Hf``/``phi``/``random``): the ``kind``, the distribution's
        name, the covariate-link identity, the fitted parameters, and the
        covariate-coefficient names. If the model was fit from data and its
        parameter covariance can be computed, that is stored too, so the
        restored model can also produce confidence bounds
        (``cb``/``param_cb``/``standard_errors``).

        Two link forms round-trip. The fixed-form parametric families --
        Accelerated Failure Time, Proportional Hazards, Proportional Odds and
        (parametric) Additive Hazards -- whose covariate link is fully
        determined by the ``kind`` and coefficients; and Accelerated Life
        parameter-substitution models built on a built-in life model
        (``Power``, ``Eyring``, ``Arrhenius``-style ``Exponential``, ...),
        which are rebuilt from the distribution and life-model names. A model
        with a genuinely bespoke covariate link (e.g. a custom life model whose
        parameterisation is not a fixed name map) cannot be rebuilt from a name
        and raises ``NotImplementedError``.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict[str, Any] = self._serialise_link()
        out["params"] = np.asarray(self.params, dtype=float).tolist()
        out["k"] = int(self.k)
        out["k_dist"] = int(self.k_dist)
        out["fixed"] = {str(k): float(v) for k, v in self.fixed.items()}
        out["gamma"] = float(getattr(self, "gamma", 0.0))
        out["p"] = float(getattr(self, "p", 1.0))
        out["f0"] = float(getattr(self, "f0", 0.0))
        if self._has_center():
            # Only a baseline at the covariate means (center=True, #463) is
            # stored, which makes the dict schema 2: a schema-1 reader
            # would take it for the baseline at Z = 0.
            out["center"] = np.asarray(self.center, dtype=float).tolist()
        serialise_covariate_meta(self, out)

        # Store the parameter covariance so the restored model can produce
        # confidence bounds without the original data: from the fit when
        # available, else the covariance restored from a previous dict so
        # repeated save/load cycles do not silently lose it (#261).
        if hasattr(self, "data") and getattr(self, "res", None) is not None:
            try:
                cov = self.covariance()
            except Exception:
                cov = None
        else:
            cov = getattr(self, "_restored_covariance", None)
        if cov is not None and np.all(np.isfinite(cov)):
            out["covariance"] = np.asarray(cov, dtype=float).tolist()
        if hasattr(self, "_neg_ll"):
            out["_neg_ll"] = float(self._neg_ll)
        # The sample size of bic() and aic_c(), which the restored model,
        # having no data, could not otherwise compute.
        ic_n = self._ic_sample_size_or_none()
        if ic_n is not None:
            out["ic_n"] = ic_n
        # The printout's "Data" line (#508), so the restored model prints
        # the same; the data themselves are not stored.
        if self._data_repr():
            out["data_summary"] = self._data_repr()
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "ParametricRegressionModel":
        """
        Rebuild a regression model from a :meth:`to_dict` dictionary.

        The distribution and fitter factory are resolved from the public
        ``surpyval`` namespace by name (restricted to the known distributions
        and regression families, so an untrusted dict cannot resolve arbitrary
        attributes), then the fitted parameters are restored. The result
        predicts identically to the original model; if a covariance was stored
        it also produces confidence bounds.

        See Also
        --------
        to_dict, to_json, from_json
        """
        import surpyval
        from surpyval.univariate.parametric.parametric_fitter import (
            OptimisedFitMixin,
            ParametricFitter,
        )

        if model_dict.get("parameterization") != "parametric-regression":
            raise ValueError(
                "Must create a regression model from a parametric-regression "
                "model dict"
            )
        kind = model_dict["kind"]
        dist = getattr(surpyval, model_dict["distribution"], None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                "Unknown distribution {!r}".format(model_dict["distribution"])
            )

        params = np.array(model_dict["params"], dtype=float)
        k_dist = int(model_dict["k_dist"])

        reg_model: Any
        if kind == "Accelerated Life":
            # Rebuild the parameter-substitution fitter from the distribution
            # and the built-in life model; the fitter carries the life-model's
            # phi and the distribution's life-parameter transforms, so it
            # predicts identically. The reg_model is the life-model singleton
            # itself (its phi and phi_param_map drive phi() and __repr__).
            from surpyval.univariate.regression.accelerated_life import (
                LIFE_MODELS,
                AcceleratedLife,
            )

            life_name = model_dict.get("life_model_name")
            if life_name not in LIFE_MODELS:
                raise ValueError(
                    "Cannot deserialise Accelerated Life model with life "
                    "model {!r}".format(life_name)
                )
            # The guard above only establishes a ParametricFitter, which
            # admits Bernoulli, Binomial and ExactEventTime -- none of
            # them fittable, and an accelerated life model needs a
            # distribution it can fit. The dict is untrusted input, so a
            # name like that would otherwise get this far and fail deep
            # inside the fitter on a missing attribute.
            if not isinstance(dist, OptimisedFitMixin):
                raise ValueError(
                    "Cannot deserialise Accelerated Life model with "
                    "distribution {!r}: it has no fitting machinery.".format(
                        model_dict["distribution"]
                    )
                )
            reg_model = LIFE_MODELS[life_name]
            fitter = AcceleratedLife(dist, reg_model)
        elif kind in _SERIALISABLE_KINDS:
            factory_name, phi_kind = _SERIALISABLE_KINDS[kind]
            factory = getattr(surpyval, factory_name)
            fitter = factory(dist)
            phi_param_map = {
                k: int(v) for k, v in model_dict["phi_param_map"].items()
            }
            reg_model = types.SimpleNamespace(
                name=model_dict["reg_model_name"],
                phi_param_map=phi_param_map,
            )
            if phi_kind == "exp":
                # The log-linear multiplier exp(beta'Z), matching the
                # fitters. Imported here because _fit_skeleton imports
                # this module at load time.
                from ._fit_skeleton import LogLinearPhi

                reg_model.phi = LogLinearPhi.phi
        else:
            raise ValueError(
                "Cannot deserialise regression kind {!r}".format(kind)
            )

        out = cls()
        out.model = fitter
        out.distribution = dist
        out.dist = dist
        out.reg_model = reg_model
        out.kind = kind
        out.params = params
        out.dist_params = params[:k_dist]
        out.phi_params = params[k_dist:]
        out.k_dist = k_dist
        out.fixed = {
            k: float(v) for k, v in model_dict.get("fixed", {}).items()
        }
        # The number of estimated parameters, recomputed rather than read
        # from the stored ``k``: dicts written before ``k`` excluded the
        # fixed parameters (and the accelerated-life placeholder) stored the
        # full parameter-vector length.
        out.k = len(params) - len(out.fixed)
        out._restored = True
        out._data_summary = model_dict.get("data_summary")
        out.gamma = float(model_dict.get("gamma", 0.0))
        out.p = float(model_dict.get("p", 1.0))
        out.f0 = float(model_dict.get("f0", 0.0))
        if kind != "Accelerated Life":
            # A dict without one has its baseline at Z = 0 (#463).
            out.center = np.array(
                model_dict.get("center", np.zeros(len(params) - k_dist)),
                dtype=float,
            )
            if out.center.shape != (len(params) - k_dist,):
                raise ValueError(
                    "The model dict's 'center' has {} value(s) for {} "
                    "covariate coefficient(s).".format(
                        out.center.size, len(params) - k_dist
                    )
                )
        restore_covariate_meta(out, model_dict)

        if "covariance" in model_dict:
            out._restored_covariance = np.array(
                model_dict["covariance"], dtype=float
            )
        if "_neg_ll" in model_dict:
            out._neg_ll = float(model_dict["_neg_ll"])
        # Dicts written before "ic_n" existed carry no sample size, and
        # bic() / aic_c() then say they need the data.
        out._ic_n = cls._restored_ic_n(model_dict)
        return out

    def _prepare_Z(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        Convert ``Z`` to a numeric design matrix.

        If a pandas DataFrame is passed and the model was fit from a DataFrame,
        the covariate columns (or formula) recorded at fit time are used to
        select and encode the correct columns. Otherwise ``Z`` is returned
        unchanged.
        """
        return prepare_Z(Z, self.feature_names, self._model_spec)

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine (#476): a constant column where the family has an
        intercept, or a linear combination of the others. Their
        coefficients are ``nan`` in ``params`` (R's ``NA``), as are their
        standard errors, and predictions take them as 0. For an
        accelerated-life model, whose parameters are not one per column,
        they are positions in ``phi_params``: a stress effect the data
        cannot determine (#503), such as the second of two equal stress
        columns of ``DualPower``."""
        phi = np.asarray(self.params, dtype=float)[self.k_dist :]
        return np.flatnonzero(np.isnan(phi))

    def _eval_params(self) -> npt.NDArray:
        """``params`` with an aliased coefficient as 0, as the model
        predicts with it."""
        if not self.aliased.size:
            return self.params
        params = np.array(self.params, dtype=float)
        params[self.k_dist + self.aliased] = 0.0
        return params

    def _held(self) -> set:
        """The names of the parameters that were not estimated: the
        ``fixed`` ones and the aliased coefficients."""
        names = self.parameter_names
        return set(self.fixed) | {
            names[self.k_dist + j] for j in self.aliased.tolist()
        }

    def _n_covariates(self) -> int:
        """The number of columns of ``Z``: that of the fitted data where
        the model has it, else one per coefficient (an accelerated-life
        model's life-model parameters are not one per column)."""
        data = getattr(self, "data", None)
        Z = getattr(data, "Z", None)
        if Z is not None and np.ndim(Z) == 2:
            return int(np.shape(Z)[1])
        return len(self.params) - self.k_dist

    def _has_center(self) -> bool:
        """Whether the baseline is at a nonzero covariate ``center``."""
        return self.center is not None and bool(np.any(self.center))

    def _centred(
        self, Z: npt.ArrayLike, center: "npt.NDArray | None" = None
    ) -> Any:
        """The covariate rows ``Z`` (already prepared) relative to
        ``center`` (default: the model's), where the baseline is."""
        center = self.center if center is None else center
        if center is None or not np.any(center):
            return Z
        return np.asarray(Z, dtype=float) - center

    #: What ``exp(coef)`` is, for a log-linear link, by kind.
    _EXP_MEANING = {
        "Proportional Hazard": "the hazard ratio",
        "Accelerated Failure Time": "the acceleration factor",
        "Proportional Odds": "the survival odds ratio",
    }

    @property
    def life_parameter(self) -> "str | None":
        """The distribution parameter an accelerated life model replaces by
        its life model (``None`` for the other families)."""
        if self.kind != "Accelerated Life":
            return None
        return getattr(getattr(self, "model", None), "life_parameter", None)

    def _life_relation(self) -> str:
        # How the life parameter follows from the life model, e.g.
        # "L(Z) of the Power life model", for the printed model.
        relation = getattr(self.model, "life_relation", "L(Z)")
        return "{} of the {} life model".format(relation, self.reg_model.name)

    def _is_linear_predictor(self) -> bool:
        """Whether the covariate parameters are coefficients of a linear
        predictor ``beta'Z`` (one per column of ``Z``), which the
        coefficient table is for; an accelerated-life model's are the
        parameters of its life model."""
        n_phi = len(self.params) - self.k_dist
        pmap = dict(getattr(self.reg_model, "phi_param_map", {}) or {})
        return self.kind != "Accelerated Life" and pmap == {
            "beta_{}".format(i): i for i in range(n_phi)
        }

    def _exp_meaning(self) -> "str | None":
        """What ``exp(coef)`` means, or ``None`` where the link is not
        log-linear (``exp(coef)`` is then not a ratio)."""
        from ._fit_skeleton import LogLinearPhi

        name = getattr(self.reg_model, "name", "")
        if name not in (LogLinearPhi.NAME_E, LogLinearPhi.NAME_EXP):
            return None
        return self._EXP_MEANING.get(self.kind)

    def _summary_se(self) -> npt.NDArray:
        """The standard errors for the summary: ``nan`` where there are
        none (a model built from parameters, or one whose information
        cannot be inverted), and for a parameter held fixed."""
        n = len(self.params)
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                se = np.array(self.standard_errors(), dtype=float)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):
            se = np.full(n, np.nan)
        if se.shape != (n,):
            se = np.full(n, np.nan)
        names = self.parameter_names
        se[[i for i, name in enumerate(names) if name in self.fixed]] = np.nan
        return se

    def summary(self, alpha_ci: float = 0.05) -> "pd.DataFrame":
        """
        The parameter table (#484), in lifelines' layout: the baseline
        distribution's parameters, then the regression coefficients (or,
        for an accelerated-life model, the life model's parameters), each
        with its standard error and a two-sided ``1 - alpha_ci`` Wald
        interval; for the coefficients also ``exp(coef)`` (where the link
        is log-linear: the hazard ratio for proportional hazards, the
        acceleration factor for AFT, the survival odds ratio for
        proportional odds), the Wald statistic ``z`` and its two-sided
        p-value. The coefficients are named by ``feature_names`` for a
        model fitted with ``fit_from_df``.

        The baseline parameters' intervals are those of :meth:`param_cb`,
        which stay in the parameter's support (a positive scale's is
        computed on the log scale). A fixed parameter has no standard
        error or interval (``nan``), nor does an aliased coefficient
        (#476), whose value is ``nan`` too.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(part, name)``, ``part`` one of ``"baseline"``,
            ``"coefficients"`` or ``"life model"``, with the columns of
            ``CoxPH``'s :meth:`summary`: ``coef`` (the estimate),
            ``exp(coef)``, ``se(coef)``, ``coef lower 95%``, ``coef upper
            95%``, ``exp(coef) lower 95%``, ``exp(coef) upper 95%``, ``z``
            and ``p``.

        Examples
        --------
        >>> from surpyval import WeibullPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["censored"] = 1 - df["arrest"]  # arrest is 1 for an arrest
        >>> model = WeibullPH.fit_from_df(
        ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age"]
        ... )
        >>> model.summary()[["coef", "se(coef)", "p"]].round(4)
                               coef  se(coef)       p
        part         name
        baseline     alpha  32.2365   11.4172     NaN
                     beta    1.3801    0.1241     NaN
        coefficients fin    -0.3296    0.1898  0.0826
                     age    -0.0713    0.0209  0.0006
        """
        import pandas as pd

        from ._summary import coefficient_names, coefficient_table

        params = np.asarray(self.params, dtype=float)
        se = self._summary_se()
        names = self.parameter_names
        k = self.k_dist
        level = "{:g}%".format(100 * (1 - alpha_ci))
        if self._is_linear_predictor():
            part = "coefficients"
            rows = coefficient_table(
                coefficient_names(self, len(params) - k),
                params[k:],
                se[k:],
                alpha_ci,
                exp=self._exp_meaning() is not None,
            )
            first = k
        else:
            part, rows, first = "life model", None, len(params)
        # The other parameters: their estimate, standard error and the
        # support-respecting interval of ``param_cb``.
        # The life parameter an accelerated-life model replaces by its life
        # model is a placeholder, not a parameter (#489): no row.
        kept = [i for i in range(first) if names[i] != self.life_parameter]
        others = []
        for i in kept:
            bounds = np.full(2, np.nan)
            if np.isfinite(se[i]):
                try:
                    with warnings.catch_warnings(), np.errstate(all="ignore"):
                        warnings.simplefilter("ignore")
                        bounds = np.asarray(
                            self.param_cb(names[i], alpha_ci), dtype=float
                        ).ravel()
                except (ValueError, ArithmeticError):
                    pass
            others.append(
                {
                    "coef": params[i],
                    "se(coef)": se[i],
                    "coef lower " + level: bounds[0],
                    "coef upper " + level: bounds[-1],
                }
            )
        table = pd.DataFrame(
            others,
            columns=list(coefficient_table([], [], [], alpha_ci).columns),
        )
        parts = ["baseline"] * min(k, first) + [part] * (first - k)
        index = [(parts[i], names[i]) for i in kept]
        if rows is not None:
            table = pd.concat([table, rows.reset_index(drop=True)])
            index += [(part, name) for name in rows.index]
        table.index = pd.MultiIndex.from_tuples(index, names=["part", "name"])
        return table

    def _data_repr(self) -> str:
        """The data the model was fitted to, in one line, for the printout
        (#508): units weighted by ``n``, by kind of censoring and
        truncation. Empty for a model built from parameters; a restored
        model gives the line it was saved with."""
        data = getattr(self, "data", None)
        if data is None:
            return self._data_summary or ""
        if isinstance(data, dict):
            c, n, t = data.get("c"), data.get("n"), data.get("t")
            x = data.get("x")
        else:
            c = getattr(data, "c", None)
            n = getattr(data, "n", None)
            t = getattr(data, "t", None)
            x = getattr(data, "x", None)
        if c is None:
            return ""
        lower, upper = getattr(self.distribution, "support", (-np.inf, np.inf))
        t = None if t is None else np.asarray(t, dtype=float)
        if t is None or t.ndim != 2 or len(t) != len(np.asarray(c)):
            return data_summary(c, n, x=x)
        return data_summary(c, n, t[:, 0], t[:, 1], lower, upper, x=x)

    def __repr__(self) -> str:
        if not hasattr(self, "params"):
            return "Unable to fit values"
        from ._summary import coefficient_repr, format_table

        out = (
            "Parametric Regression SurPyval Model"
            + "\n===================================="
            + "\nKind                : {kind}"
            + "\nDistribution        : {dist}"
            + "\nRegression Model    : {reg_model}"
            + "\nFitted by           : MLE"
        ).format(
            kind=self.kind,
            dist=self.distribution.name,
            reg_model=self.reg_model.name,
        )
        data_line = self._data_repr()
        if data_line:
            out += "\nData                : " + data_line
        if self._has_center():
            # A fit with center=True (#463): say where the baseline
            # parameters are.
            out += (
                "\nBaseline at         : the covariate means, "
                "Z = center = {}".format(
                    np.array2string(
                        np.asarray(self.center, dtype=float),
                        separator=", ",
                    )
                )
            )
        # The life parameter an accelerated-life model substitutes is held
        # at a placeholder value, not a parameter of the model.
        placeholder = set()
        if self.kind == "Accelerated Life":
            placeholder = set(getattr(self.model, "fixed", None) or {})
        fixed = {k: v for k, v in self.fixed.items() if k not in placeholder}
        if fixed:
            out += "\nFixed               : {}".format(
                ", ".join("{} = {:.6g}".format(k, v) for k, v in fixed.items())
            )
        table = self.summary()
        estimates = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }

        def block(part: str) -> str:
            rows = table.loc[part].rename(columns=estimates)
            rows = rows.loc[[n for n in rows.index if n not in placeholder]]
            rows.index.name = None
            return format_table(rows, list(estimates.values()))

        parts = table.index.get_level_values(0)
        out += "\nBaseline            : {} parameters".format(
            self.distribution.name
        )
        if "baseline" in parts:
            out += "; Wald 95% intervals\n" + block("baseline")
        if self.life_parameter is not None:
            # Replaced by the life model, not fitted (#489).
            out += "\n    {}: {}".format(
                self.life_parameter, self._life_relation()
            )
        if "life model" in parts:
            out += "\nLife model          : Wald 95% intervals\n" + block(
                "life model"
            )
        if "coefficients" in parts:
            meaning = self._exp_meaning()
            out += "\nCoefficients        : {}Wald 95% intervals\n".format(
                "exp(coef) is {}; ".format(meaning) if meaning else ""
            ) + coefficient_repr(table.loc["coefficients"])
        return out

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        # H(t* | Z) at the median time scored: for a family acting through
        # a linear predictor it ranks the rows as that predictor does, with
        # the sign of a higher risk, at every t* (see ``concordance``).
        t = np.full(x.size, np.nanmedian(x))
        with warnings.catch_warnings():
            # An additive model's negative hazard at t* does not change
            # the ranking.
            warnings.filterwarnings("ignore", message="The additive hazard")
            return np.asarray(self.Hf(t, Z), dtype=float)

    def _concordance_data(self) -> "tuple | None":
        data = getattr(self, "data", None)
        if data is None or getattr(self, "is_tvc", False):
            return None
        return data.x, data.c, data.n, data.Z

    def phi(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        Z = self._prepare_Z(Z)
        if not hasattr(self.reg_model, "phi"):
            # Additive-hazards reg models have no multiplier: the
            # covariate effect enters as beta'Z added to the hazard, so
            # phi() is undefined rather than an AttributeError (#277).
            raise NotImplementedError(
                "phi() is not defined for additive-hazards models: the "
                "covariate effect is additive (beta'Z on the hazard), "
                "not a multiplier."
            )
        # Relative to the centre for a baseline kept there (#463).
        return self.reg_model.phi(
            self._centred(Z), *self._eval_params()[self.k_dist :]
        )

    def _eval(
        self,
        fn: Any,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        below_support: float,
        grid: bool = False,
    ) -> npt.NDArray:
        # The shared body of the five distribution functions below: coerce
        # ``x``, resolve DataFrame covariates against the fit-time design,
        # and evaluate the family's function at the fitted parameters. Each
        # named method carried this verbatim.
        if isinstance(x, list):
            x = np.array(x)
        Z = self._prepare_Z(Z)
        shape = None
        if grid:
            # Every time for every row (#488): the pairs, then reshaped.
            rows = covariate_rows(Z, self._n_covariates())
            shape = (rows.shape[0], np.size(x))
            x = np.tile(np.asarray(x, dtype=float).reshape(-1), shape[0])
            Z = np.repeat(rows, shape[1], axis=0)
        elif np.ndim(Z) == 2:
            check_paired_rows(np.size(x), np.shape(Z)[0])
        Z = self._centred(Z)
        # Below the support (a negative time for a positive distribution)
        # nothing has happened yet: survival 1, and 0 for the others. The
        # distribution functions gave nan there, with a RuntimeWarning, and
        # warned "divide by zero" at 0 itself for the log-based ones, where
        # the value is already right.
        lower = self.distribution.support[0]
        below = np.asarray(x) < lower
        if np.any(below):
            inside = lower + 1.0 if np.isfinite(lower) else 0.0
            x = np.where(below, inside, x)
        with np.errstate(divide="ignore"):
            out = fn(x, Z, *self._eval_params())
        if self.kind == "Additive Hazard":
            self._warn_if_hazard_negative(x, Z, ~below, stacklevel=5)
        if np.any(below):
            out = np.where(below, below_support, out)
        if shape is not None:
            out = np.asarray(out, dtype=float).reshape(shape)
        return out

    def _warn_if_hazard_negative(
        self,
        x: npt.ArrayLike,
        Z: npt.NDArray,
        valid: Any = True,
        stacklevel: int = 4,
    ) -> None:
        """Warn (once) when the additive hazard ``h_0(x) + beta'Z`` or its
        integral is negative at a queried point (#376).

        Nothing in the additive model keeps the hazard positive: the fit
        keeps it positive at the observed failures only, so for a
        protective covariate row, or far from the data, ``h`` can be
        negative. The cumulative hazard then falls, and the predictions
        stop being those of a distribution -- ``sf`` above 1, ``ff`` and
        ``df`` negative. They are returned as the model defines them, with
        this warning.
        """
        with np.errstate(all="ignore"):
            params = self._eval_params()
            h = np.asarray(self.model.hf(x, Z, *params), dtype=float)
            H = np.asarray(self.model.Hf(x, Z, *params), dtype=float)
        valid = np.broadcast_to(valid, h.shape)
        neg_h = valid & (h < 0)
        neg_H = valid & (H < 0)
        if not (neg_h.any() or neg_H.any()):
            return
        self._warn_negative_hazard(
            int((neg_h | neg_H).sum()),
            h.size,
            float(np.exp(-np.min(H[valid]))) if neg_H.any() else None,
            stacklevel + 1,
        )

    def _warn_negative_hazard(
        self, count: int, size: int, max_sf: "float | None", stacklevel: int
    ) -> None:
        above = (
            ", so sf exceeds 1 (up to {:.4g}) and ff is negative".format(
                max_sf
            )
            if max_sf is not None
            else ""
        )
        warnings.warn(
            "The additive hazard h_0(x) + beta'Z is negative at {} of the "
            "{} queried points: the model's cumulative hazard falls there"
            "{}. The additive model does not keep the hazard positive "
            "(the fit does so only at the observed failures); these "
            "values are the model's, not a distribution's. A proportional "
            "hazards model (e.g. {}PH) keeps the hazard positive by "
            "construction.".format(count, size, above, self.distribution.name),
            RuntimeWarning,
            stacklevel=stacklevel,
        )

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        r"""
        Survival (or Reliability) function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the survival function
            will be calculated

        Z : array like or DataFrame
            The covariates: one row per value of ``x`` (or a single row,
            broadcast to every ``x``), in the column order used in the fit. A
            model fitted with ``fit_from_df`` also accepts a DataFrame with
            the named (or formula) columns. Other row counts are refused.

        grid : bool, optional
            ``True`` evaluates every ``x`` for every row of ``Z`` (a curve
            per subject, lifelines' ``predict_survival_function``), with
            shape ``(len(Z),) + x.shape``, row ``i`` for row ``i`` of ``Z``
            (#488). Default ``False``: rows and times paired.

        Returns
        -------

        sf : scalar or numpy array
            The scalar value of the survival function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the survival function at each
            corresponding value in the input array.


        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> model.sf([1, 2, 3], [[0], [0], [1]]).round(4)
        array([0.9812, 0.9382, 0.7429])
        """
        return self._eval(self.model.sf, x, Z, 1.0, grid)

    # Families whose survival along a step-valued covariate path has an exact
    # closed form. Proportional hazards, additive hazards and proportional
    # odds have a hazard that depends only on the time and the *current*
    # covariate, so the cumulative hazard is a sum of per-segment increments
    # of the constant-covariate ``Hf``; accelerated failure time instead
    # accumulates an *accelerated age* over the segments and then evaluates the
    # baseline once. Accelerated life does the same (cumulative exposure,
    # with the rate 1 / L(Z)) where its life parameter scales time; a
    # location life parameter is refused (see ``_check_tvc_evaluable``).
    _TVC_ADDITIVE_KINDS = (
        "Proportional Hazard",
        "Additive Hazard",
        "Proportional Odds",
    )
    _TVC_EVALUABLE_KINDS = (
        "Proportional Hazard",
        "Additive Hazard",
        "Proportional Odds",
        "Accelerated Failure Time",
        "Accelerated Life",
    )
    #: The accelerated-life distributions whose life parameter is a scale
    #: of time (Weibull ``alpha``, the Exponential and Gamma rates
    #: ``1 / L``, LogNormal's ``exp(mu)``): S(t | V) = S_1(t / L(V)), with
    #: S_1 the distribution at unit life, so a changing stress accumulates
    #: an age ``int du / L(V(u))`` (#172).
    _TVC_SCALE_LIFE = ("Weibull", "Exponential", "Gamma", "LogNormal")

    def _check_tvc_evaluable(self) -> None:
        """Refuse a family with no form along a covariate path."""
        if self.kind not in self._TVC_EVALUABLE_KINDS:
            raise NotImplementedError(
                "time-varying-covariate evaluation is defined for the "
                "proportional-hazards, additive-hazards, proportional-odds, "
                "accelerated-failure-time and accelerated-life families "
                "(this model is '{}').".format(self.kind)
            )
        name = self.distribution.name
        if self.kind == "Accelerated Life" and name not in (
            self._TVC_SCALE_LIFE
        ):
            raise NotImplementedError(
                "An Accelerated Life model is evaluated along a changing "
                "stress by cumulative exposure, S(t) = S_1(int_0^t du / "
                "L(V(u))), which needs a life parameter that scales time "
                "({} only). The {} life parameter '{}' is a location: a "
                "change of stress shifts the distribution rather than "
                "rescaling time, so there is no accumulated age to carry "
                "from one stress to the next. Fit a scale-life "
                "distribution (e.g. AcceleratedLife(Weibull, ...)) or an "
                "accelerated failure time model for this.".format(
                    ", ".join(self._TVC_SCALE_LIFE),
                    name,
                    self.life_parameter,
                )
            )

    def _tvc_scales_time(self) -> bool:
        """Whether the covariate rescales time along a path (AFT, and
        accelerated life), rather than setting the current hazard."""
        return self.kind in ("Accelerated Failure Time", "Accelerated Life")

    def _tvc_theta(
        self, theta: "tuple | None"
    ) -> "tuple[npt.NDArray, npt.NDArray | None]":
        """``(params, center)`` to evaluate a path at: ``theta``, or the
        model's own (``cb_tvc`` passes those of ``_inference_state``)."""
        if theta is None:
            return self._eval_params(), self.center
        return theta

    def _tvc_rate(self, Zc: npt.NDArray, params: npt.NDArray) -> npt.NDArray:
        """The rate at which a covariate row (already centred) ages a unit:
        AFT's ``phi = exp(beta'z)``, and an accelerated life model's
        ``1 / L(z)``. One value per row."""
        Zc = np.atleast_2d(np.asarray(Zc, dtype=float))
        phi_params = params[self.k_dist :]
        with np.errstate(all="ignore"):
            if self.kind == "Accelerated Life":
                rate = 1.0 / np.asarray(
                    self.model.phi(Zc, *phi_params), dtype=float
                )
            else:
                rate = np.asarray(
                    self.model._phi(Zc, *phi_params), dtype=float
                )
        rate = rate.ravel()
        if rate.size == 1 and Zc.shape[0] != 1:
            rate = np.full(Zc.shape[0], float(rate[0]))
        return rate

    def _tvc_segments(
        self, schedule: Any, t_max: float
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Materialise ``schedule`` to ``t_max`` with the first segment held back
        to the time origin (survival measured from ``0``).
        """
        from .tvc_schedule import segments_from_origin

        return segments_from_origin(schedule, t_max)

    def _to_schedule(self, Z: Any, xl: "npt.ArrayLike | None") -> Any:
        """
        Coerce the ``sf_tvc`` covariate argument into a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` and
        check its covariate count against the fitted model.
        """
        from .tvc_schedule import as_covariate_path

        schedule = as_covariate_path(Z, xl)
        # Columns of Z (an accelerated life model's life-model parameters
        # are not one per column).
        n_cov = self._n_covariates()
        if schedule.p != n_cov:
            raise ValueError(
                "the {} has {} covariate(s) but the model was fit with "
                "{}".format(
                    "schedule" if hasattr(schedule, "segments") else "path",
                    schedule.p,
                    n_cov,
                )
            )
        return schedule

    @keeps_query_shape
    def Hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard for a covariate following a path ``Z(t)``: a step
        schedule, or a continuously varying path.

        For the proportional-hazards, additive-hazards and proportional-odds
        families the hazard at time :math:`t` depends only on :math:`t` and
        the covariate value *at* :math:`t`, so along a piecewise constant path
        the cumulative hazard is exactly the sum of the per-segment increments
        of the constant-covariate cumulative hazard

        .. math::
            H\bigl(x \mid Z(\cdot)\bigr)
            = \sum_{\text{seg } (a, b]} \bigl[\,H(b, z) - H(a, z)\,\bigr] .

        For proportional odds, with :math:`\phi = e^{\beta' z}` multiplying the
        survival odds, the hazard is
        :math:`h(t \mid z) = h_0(t) / (F_0(t) + \phi S_0(t))` and its integral
        at constant :math:`z` is
        :math:`H(t, z) = H_0(t) - \ln\phi + \ln(F_0(t) + \phi S_0(t))
        = -\ln S(t \mid z)`, so each segment contributes
        :math:`\ln[S(a \mid z) / S(b \mid z)]`. On entering a segment the
        hazard switches to the new covariate's PO hazard; the survival does
        not jump to the new covariate's PO curve. The first segment is held
        back to the bottom of the baseline's support (for a baseline defined
        below zero, such as ``Logistic``, the value in force at time zero is
        taken to apply before it too), so the result is the unconditional
        survival.

        For accelerated failure time the covariate rescales time, so the path
        accumulates an *accelerated age*
        :math:`\psi(x) = \sum_{\text{seg}} e^{\beta' z}\,(b - a)` and the
        cumulative hazard is the baseline evaluated there,
        :math:`H(x \mid Z(\cdot)) = H_0(\psi(x))`. An accelerated life model
        whose life parameter scales time (Weibull ``alpha``, the Exponential
        and Gamma rates, LogNormal's ``exp(mu)``) does the same with the rate
        :math:`1 / L(z)` and the distribution at unit life, Nelson's
        cumulative exposure; one whose life parameter is a location (Normal,
        Gumbel, Logistic) raises ``NotImplementedError``. Either way a
        single constant segment reduces exactly to ``Hf(x, Z)``.

        Along a
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`, a
        covariate that changes continuously, the same hazards are
        integrated: :math:`H(x) = \int_0^x h(u \mid Z(u))\, du`, and for
        accelerated failure time
        :math:`\psi(x) = \int_0^x e^{\beta' Z(u)}\, du` (Nelson's cumulative
        exposure). For proportional odds the hazard is that of the current
        covariate, :math:`h_0(t) / (F_0(t) + \phi(Z(t)) S_0(t))`: the limit
        of the step sum above, and the model ``fit_tvc`` fits. The integral
        is by adaptive Gauss-Kronrod quadrature to a relative error of about
        ``1e-10``; where that is not reached, one ``RuntimeWarning`` says at
        how many query times. The type of ``Z`` picks the method: a
        ``StepSchedule`` is always summed exactly.

        The path is measured from time zero: a schedule starting after zero
        has its first value held back to zero, and the part of a schedule
        before zero is ignored (the value in force at zero applies from
        there). Any time is a valid query, zero and below included: a
        constant path gives ``Hf(x, Z)`` there too.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the cumulative hazard.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path -- a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            a :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`,
            or an array of per-segment covariate rows (with ``xl`` giving the
            segment start times).
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.

        Returns
        -------
        ndarray
            The cumulative hazard at each ``x``.

        Examples
        --------
        An exponential proportional hazards model along a stress ramp
        ``Z(t) = 0.1 t``, whose cumulative hazard is
        :math:`\lambda (e^{0.1 \beta t} - 1) / (0.1 \beta)`:

        >>> import numpy as np
        >>> from surpyval import CovariatePath, ExponentialPH
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 2, (300, 1))
        >>> x = rng.exponential(10 * np.exp(-0.7 * Z[:, 0]))
        >>> model = ExponentialPH.fit(x, Z)
        >>> lam, beta = model.params
        >>> ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
        >>> t = np.array([2.0, 5.0, 10.0])
        >>> H = model.Hf_tvc(t, ramp)
        >>> exact = lam * np.expm1(0.1 * beta * t) / (0.1 * beta)
        >>> bool(np.allclose(H, exact, rtol=1e-12, atol=0))
        True
        """
        H, falls, accuracy = self._hf_tvc(x, Z, xl)
        self._warn_tvc(H, falls, accuracy, stacklevel=5)
        return H

    def _warn_tvc(
        self,
        H: npt.NDArray,
        falls: int,
        accuracy: "tuple[int, int, float, str] | None",
        stacklevel: int,
    ) -> None:
        """The warnings of ``sf_tvc`` / ``Hf_tvc``: a falling additive
        hazard (#376), and a quadrature that missed its target (#172);
        ``stacklevel`` counts from here to the caller of the public
        method."""
        if falls:
            self._warn_negative_hazard(
                falls, H.size, self._max_sf(H), stacklevel=stacklevel
            )
        if accuracy is not None:
            from .tvc_path import warn_missed_target

            missed, total, worst, limit = accuracy
            warn_missed_target(
                missed, total, worst, limit, self._tvc_rtol, stacklevel
            )

    @staticmethod
    def _max_sf(H: npt.NDArray) -> "float | None":
        finite = H[np.isfinite(H)]
        if finite.size and finite.min() < 0:
            return float(np.exp(-finite.min()))
        return None

    def _hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None",
        given: "float | None" = None,
        theta: "tuple | None" = None,
        frozen: "dict | None" = None,
    ) -> "tuple[npt.NDArray, int, tuple | None]":
        """The cumulative hazard along the path, the number of query
        times at which an additive hazard fell (a negative ``H`` or a
        negative segment increment, #376) -- 0 for the other families --
        and, for a ``CovariatePath`` whose quadrature missed its target,
        ``(missed, total, worst)`` (else ``None``). ``given`` is used only
        for a ``CovariatePath``: ``H`` is then integrated from ``given``,
        ``H(x) - H(given)``.

        ``theta`` is ``(params, center)`` to evaluate at instead of the
        model's own. ``frozen`` is a dict that keeps a path's quadrature
        mesh: an empty one gets the mesh of this call (under
        ``"edges"``), and one holding a mesh is integrated on it with no
        refinement, so that a function of the parameters is smooth in
        them (``cb_tvc``'s delta method, #172)."""
        from .tvc_path import CovariatePath

        self._check_tvc_evaluable()
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        schedule = self._to_schedule(Z, xl)
        # A missing query time has no value (NaN); the others are
        # evaluated as usual.
        missing = np.isnan(xq)
        if missing.all():
            return np.full(xq.shape, np.nan), 0, None
        if isinstance(schedule, CovariatePath):
            # Integrated, not summed.
            return self._tvc_hf_path(xq, schedule, given, theta, frozen)
        # A horizon at or below 0 materialises the one segment in force at
        # 0: H is then 0, or the baseline's value for a time below 0.
        t_max = float(np.max(xq[~missing]))
        starts, ends, Zseg = self._tvc_segments(schedule, t_max)
        xq_eval = np.where(missing, t_max, xq)

        falls = np.zeros(xq.shape[0], dtype=bool)
        if self.kind in self._TVC_ADDITIVE_KINDS:
            H = self._tvc_hf_additive(
                xq_eval, starts, ends, Zseg, falls, theta
            )
        else:
            H = self._tvc_hf_aft(xq_eval, starts, ends, Zseg, theta)
        if self.kind == "Additive Hazard":
            falls |= H < 0
        falls &= ~missing
        return np.where(missing, np.nan, H), int(falls.sum()), None

    #: The relative accuracy the quadrature along a ``CovariatePath``
    #: aims for on the cumulative hazard (private: tests change it).
    _tvc_rtol: float = 1e-10

    def _tvc_hf_path(
        self,
        xq: npt.NDArray,
        path: Any,
        given: "float | None",
        theta: "tuple | None" = None,
        frozen: "dict | None" = None,
    ) -> "tuple[npt.NDArray, int, tuple | None]":
        r"""
        The cumulative hazard along a continuously varying ``path`` (#172),
        less its value at ``given`` when that is supplied; the values and
        counts, ``theta`` and ``frozen`` are as for :meth:`_hf_tvc`.

        At and before time 0 the value in force at 0 applies, exactly as
        for a step schedule, so there the one-segment step sum gives ``H``.
        After 0 the hazard (for AFT and accelerated life, the rate at which
        the unit ages) is integrated over panels by
        :func:`~.tvc_path.integrate_panels`, and summed outward from
        ``given`` (or 0): nothing is subtracted for a baseline that starts
        at 0.

        Along a periodic path the age a time-scaling family accumulates
        over a whole period is the same every period, so only one period
        is integrated: :math:`\psi(t) = k\,\Psi_P + \psi(t - kP)` with
        :math:`k = \lfloor t / P \rfloor` (#172, the periodic shortcut). A
        hazard family has no such shortcut: its baseline ages.
        """
        from .tvc_path import (
            integrate_panels,
            missed_target,
            path_mesh,
            sum_between,
        )

        aft = self._tvc_scales_time()
        missing = np.isnan(xq)
        xe = np.where(missing, 0.0, xq)
        n = xq.shape[0]
        g_pos = given is not None and given > 0

        # H at min(x, 0), at 0 and at min(given, 0): the one segment in
        # force at 0, [0, 0], of the step sum.
        z0 = np.asarray(path._values(np.zeros(1), left=False), dtype=float)
        g_low = min(given, 0.0) if given is not None else 0.0
        low_t = np.concatenate([np.minimum(xe, 0.0), [0.0, g_low]])
        falls_low = np.zeros(low_t.shape, dtype=bool)
        seg = (np.zeros(1), np.zeros(1), z0.reshape(1, -1))
        if aft:
            H_low = self._tvc_hf_aft(low_t, *seg, theta)
        else:
            H_low = self._tvc_hf_additive(low_t, *seg, falls_low, theta)
        H_x_low, H_at0, H_g_low = H_low[:n], H_low[n], H_low[n + 1]
        falls = falls_low[:n].copy()

        points = xe[xe > 0]
        if g_pos:
            points = np.append(points, given)
        if points.size == 0:
            # Every time is at or before 0.
            H_full = H_x_low
            H = H_full if given is None else H_full - H_g_low
            accuracy = None
        else:
            ex = np.maximum(xe, 0.0)
            period = path.period
            # The periodic shortcut: integrate one period only.
            whole = aft and period is not None and np.max(points) > period

            def split(t: npt.NDArray) -> tuple:
                # Whole periods, and the time into the last one.
                k = np.floor(t / period)
                return k, np.clip(t - k * period, 0.0, period)

            if whole:
                k_x, r_x = split(ex)
                inner = np.append(r_x[r_x > 0], period)
                if g_pos:
                    inner = np.append(inner, split(np.array([given]))[1])
                inner = inner[inner > 0]
            else:
                inner = points
            if frozen is not None and "edges" in frozen:
                mesh, rounds = frozen["edges"], 0
            else:
                # The model's own time scale, where the hazard can sit
                # however far out the query is.
                params, center = self._tvc_theta(theta)
                scale = self._tvc_time_scale(
                    0.0, self._centred(z0, center), params
                )
                mesh, rounds = path_mesh(path, np.unique(inner), scale), None
            res = integrate_panels(
                self._path_panel_terms(path, theta),
                mesh,
                self._tvc_rtol,
                max_rounds=rounds,
            )
            if frozen is not None and "edges" not in frozen:
                frozen["edges"] = res["edges"]
            edges, value = res["edges"], res["value"]

            def age(t: npt.NDArray) -> npt.NDArray:
                # The integral from 0 to each t (the accelerated age, for
                # a family that scales time).
                if not whole:
                    return sum_between(edges, value, 0.0, t)
                k, r = split(t)
                cycle = sum_between(edges, value, 0.0, np.array([period]))
                return k * cycle[0] + sum_between(edges, value, 0.0, r)

            from_0 = age(ex)
            if aft:
                # The accelerated age, through the baseline once.
                H_full = np.where(xe > 0, self._aft_H0(from_0, theta), H_x_low)
                H = H_full
                if given is not None:
                    if g_pos:
                        psi_g = age(np.array([float(given)]))
                        H = H_full - self._aft_H0(psi_g, theta)[0]
                    else:
                        H = H_full - H_g_low
                origin = 0.0
                if whole:
                    # Each value carries a whole period's integral (but
                    # for those in the first period, where this is
                    # conservative).
                    reach = np.full(ex.shape, float(period))
                else:
                    reach = np.maximum(ex, given) if g_pos else ex
            else:
                # H(x) = A(x) + int_0^max(x, 0) h, with A(x) the step
                # value at min(x, 0) (0 for a baseline that starts at 0).
                A_x = np.where(xe > 0, H_at0, H_x_low)
                H_full = A_x + from_0
                origin = float(given) if given is not None and g_pos else 0.0
                H = H_full
                if given is not None:
                    A_g = H_at0 if g_pos else H_g_low
                    H = (A_x - A_g) + sum_between(edges, value, origin, ex)
                reach = ex
                if self.kind == "Additive Hazard":
                    # A negative hazard at a node before x (#376).
                    fell = sum_between(
                        edges, res["flag"], 0.0, ex, signed=False
                    )
                    falls |= fell > 0
            accuracy = missed_target(
                res, origin, reach, self._tvc_rtol, missing
            )
        if self.kind == "Additive Hazard":
            falls |= H_full < 0
        falls &= ~missing
        return np.where(missing, np.nan, H), int(falls.sum()), accuracy

    def _aft_H0(
        self, psi: npt.NDArray, theta: "tuple | None" = None
    ) -> npt.NDArray:
        """The baseline cumulative hazard at accelerated ages ``psi``: the
        AFT baseline, or an accelerated life model's distribution at unit
        life (its life parameter set to ``L = 1``). (An age of 0 makes a
        log-time baseline evaluate log(0) = -inf on its way to the correct
        H = 0.)"""
        params = self._tvc_theta(theta)[0]
        dist = np.array(params[: self.k_dist], dtype=float)
        if self.kind == "Accelerated Life":
            slot = self.model.param_map[self.model.life_parameter]
            dist[slot] = self.model.param_transform(1.0)
        with np.errstate(divide="ignore"):
            return np.asarray(
                self.model.Hf_dist(np.asarray(psi, dtype=float), *dist),
                dtype=float,
            ).ravel()

    def _path_panel_terms(
        self, path: Any, theta: "tuple | None" = None
    ) -> Any:
        """
        The family's ``panel_terms(a, b)`` for
        :func:`~.tvc_path.integrate_panels`: on each panel ``[a, b]`` the
        exact increment with the covariate frozen at the panel's midpoint
        value ``zbar``, and at the 15 Kronrod nodes the correction
        integrand -- the hazard along the path less the hazard at ``zbar``
        (for AFT and accelerated life, the ageing rate ``phi(Z(u)) -
        phi(zbar)``) -- its size, for the rounding floor, and (for AH)
        whether the hazard is negative at a node. The correction is 0
        where the path is flat.
        """
        from .tvc_path import _NODES

        M = self.model
        params, center = self._tvc_theta(theta)
        aft = self._tvc_scales_time()
        additive = self.kind == "Additive Hazard"
        starts_at_0 = float(self.distribution.support[0]) >= 0

        def flat(values: Any, n: int) -> npt.NDArray:
            arr = np.asarray(values, dtype=float)
            if arr.size == 1 and n != 1:
                return np.full(n, float(arr.ravel()[0]))
            return arr.reshape(n)

        def terms(a: npt.NDArray, b: npt.NDArray) -> tuple:
            m = a.shape[0]
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            u = (mid[:, None] + half[:, None] * _NODES[None, :]).ravel()
            n = u.shape[0]
            zu = self._centred(path._values(u), center)
            zbar = self._centred(path._values(mid), center)
            zrep = np.repeat(zbar, _NODES.shape[0], axis=0)
            with np.errstate(all="ignore"):
                if aft:
                    along = self._tvc_rate(zu, params)
                    frozen = self._tvc_rate(zrep, params)
                    exact = self._tvc_rate(zbar, params) * (b - a)
                else:
                    along = flat(M.hf(u, zu, *params), n)
                    frozen = flat(M.hf(u, zrep, *params), n)
                    # The first panel of a baseline that starts at 0 has
                    # H(0) = 0 (a log-time baseline would say log(0)).
                    first = (a == 0) & starts_at_0
                    hi = flat(M.Hf(b, zbar, *params), m)
                    lo = flat(M.Hf(np.where(first, b, a), zbar, *params), m)
                    exact = hi - np.where(first, 0.0, lo)
                g = (along - frozen).reshape(m, -1)
            scale = np.abs(along).reshape(m, -1)
            if additive:
                flag = (along < 0).reshape(m, -1).any(axis=1)
            else:
                flag = np.zeros(m, dtype=bool)
            return exact, g, scale, flag

        return terms

    def _tvc_hf_additive(
        self,
        xq: npt.NDArray,
        starts: npt.NDArray,
        ends: npt.NDArray,
        Zseg: npt.NDArray,
        falls: "npt.NDArray | None" = None,
        theta: "tuple | None" = None,
    ) -> npt.NDArray:
        """
        Cumulative hazard along a step path for the families whose hazard
        depends only on the time and the current covariate (PH, AH, PO):
        telescoping sum of the model's ``Hf`` increment on each segment, the
        last clipped at the query time.
        """
        params, center = self._tvc_theta(theta)
        H = np.zeros(xq.shape[0], dtype=float)
        support_lo = float(self.distribution.support[0])
        for i, (a, b, z) in enumerate(zip(starts, ends, Zseg)):
            zrow = self._centred(
                np.asarray(z, dtype=float).reshape(1, -1), center
            )
            # Query times before 0 fall in the first segment when the
            # baseline is defined there.
            upper = np.clip(xq, min(a, support_lo) if i == 0 else a, b)
            # A query time of 0 makes a log-time baseline (LogNormal,
            # LogLogistic) evaluate log(0) = -inf on its way to the correct
            # H = 0; that is not worth a warning.
            with np.errstate(divide="ignore"):
                hi = np.asarray(
                    self.model.Hf(upper, zrow, *params), dtype=float
                ).ravel()
                # The first segment runs from the bottom of the support,
                # where H = 0. Subtracting H(0, z) instead would, for a
                # baseline defined below zero (Normal, Gumbel, Logistic),
                # give the survival conditional on reaching 0, not sf(x, Z).
                if i == 0:
                    lo = np.zeros(1)
                else:
                    lo = np.asarray(
                        self.model.Hf(np.array([a]), zrow, *params),
                        dtype=float,
                    ).ravel()
            if falls is not None and self.kind == "Additive Hazard":
                # A negative increment: the additive hazard fell in this
                # segment before the query time (#376).
                falls |= (hi - lo) < 0
            H = H + (hi - lo)
        return H

    def _tvc_hf_aft(
        self,
        xq: npt.NDArray,
        starts: npt.NDArray,
        ends: npt.NDArray,
        Zseg: npt.NDArray,
        theta: "tuple | None" = None,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard along a step path for accelerated failure time
        and accelerated life.

        The covariate rescales time by ``phi(z) = exp(beta'z)`` (for
        accelerated life ``1 / L(z)``), so each segment contributes
        ``phi(z) * (width)`` of *accelerated age*. The accumulated age
        ``psi(x)`` is then fed once through the baseline cumulative hazard
        ``H0`` (for accelerated life, the distribution at unit life). This
        is exact for a step covariate and reduces to ``Hf(x, Z)`` for a
        single constant segment.
        """
        params, center = self._tvc_theta(theta)
        psi = np.zeros(xq.shape[0], dtype=float)
        support_lo = float(self.distribution.support[0])
        for i, (a, b, z) in enumerate(zip(starts, ends, Zseg)):
            zrow = self._centred(
                np.asarray(z, dtype=float).reshape(1, -1), center
            )
            rate = float(self._tvc_rate(zrow, params)[0])
            # Query times before 0 fall in the first segment when the
            # baseline is defined there (a negative age, as sf(x, Z)).
            width = np.clip(xq, min(a, support_lo) if i == 0 else a, b) - a
            psi = psi + rate * width
        return self._aft_H0(psi, theta)

    @keeps_query_shape
    def sf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
    ) -> npt.NDArray:
        r"""
        Survival for a covariate that follows a path ``Z(t)``: a step
        (piecewise-constant) schedule, or a continuously varying path.

        With a time-varying covariate the survival depends on the whole
        covariate path, not one fixed vector. This is exact along a step path
        for the proportional-hazards, additive-hazards, proportional-odds and
        accelerated-failure-time families: ``S(x) = exp(-H(x))`` with ``H`` the
        per-segment accumulation in :meth:`Hf_tvc` (a cumulative-hazard sum for
        PH/AH/PO, an accelerated-age sum fed through the baseline for AFT).
        Along a
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath` the
        same quantities are integrated by quadrature, to a relative error of
        about ``1e-10`` on ``H`` (see :meth:`Hf_tvc`). A constant path gives
        ``sf(x, Z)``. An accelerated life model follows cumulative exposure
        where its life parameter scales time, and raises
        ``NotImplementedError`` where it is a location (see :meth:`Hf_tvc`).
        :meth:`cb_tvc` bounds this survival and :meth:`mean_tvc` integrates
        it.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate survival.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path. A
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`
            (built from change-points, intervals, a cyclic pattern, or a
            step-valued expression), a
            :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
            (a covariate that changes continuously), or an array of
            per-segment covariate rows with ``xl`` giving the segment start
            times.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            If supplied, return the *conditional* survival given the item has
            survived to age ``given``:
            ``S(x | given) = exp(-(H(x) - H(given)))`` for ``x > given``,
            and 1 for ``x <= given`` (survival to those times is certain).
            Along a ``CovariatePath`` the hazard is integrated from
            ``given`` on, so nothing is subtracted. A ``nan`` ``given``
            gives ``nan``.

        Returns
        -------
        ndarray
            Survival at each ``x`` (conditional on ``given`` when supplied).

        Examples
        --------
        A proportional-odds model whose covariate switches from 0 to 1 at
        ``t = 6``: before the switch the survival is that of ``Z = 0``, after
        it the hazard is that of ``Z = 1``.

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPO
        >>> from surpyval.univariate.regression import StepSchedule
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(0.5 * Z[:, 0])
        >>> model = WeibullPO.fit(x, Z)
        >>> sched = StepSchedule.from_changepoints([0, 6], [[0.0], [1.0]])
        >>> model.sf_tvc([4, 8, 12], sched).round(4)
        array([0.7721, 0.5698, 0.4292])
        >>> model.sf([4, 8, 12], [[0]]).round(4)
        array([0.7721, 0.4937, 0.2809])

        Along a covariate ramped from 0 to 1 over the first 10 time units,
        and conditional on survival to 4:

        >>> from surpyval import CovariatePath
        >>> ramp = CovariatePath.from_points([0, 10], [0.0, 1.0])
        >>> model.sf_tvc([4, 8, 12], ramp).round(4)
        array([0.8195, 0.6331, 0.471 ])
        >>> model.sf_tvc([2, 4, 8, 12], ramp, given=4).round(4)
        array([1.    , 1.    , 0.7725, 0.5747])
        """
        from .tvc_path import CovariatePath

        g = None if given is None else float(given)
        # Along a path the hazard is integrated from given on; a step
        # schedule subtracts H(given).
        from_given = (
            isinstance(Z, CovariatePath) and g is not None and not np.isnan(g)
        )
        H, falls, accuracy = self._hf_tvc(x, Z, xl, g if from_given else None)
        self._warn_tvc(H, falls, accuracy, stacklevel=5)
        if g is not None and not from_given:
            if np.isnan(g):
                # A missing conditioning age: nothing is known (as Cox).
                H = np.full(np.shape(H), np.nan)
            else:
                # H(given) is 0 at or below 0, unless the baseline has
                # mass below 0 (then it is -log of the survival to given).
                H = H - self._hf_tvc(g, Z, xl)[0]
        if g is not None and not np.isnan(g):
            # Given survival to g, survival to any x <= g is certain: the
            # difference H(x) - H(g) is not a cumulative hazard there, and
            # gave a "survival" above 1 (#523).
            xq = np.atleast_1d(np.asarray(x, dtype=float))
            H = np.where(xq <= g, 0.0, H)
        return np.exp(-H)

    def mean_tvc(
        self,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
    ) -> float:
        r"""
        The mean life along a covariate path ``Z(t)`` (a step schedule or
        a continuously varying path, as for :meth:`sf_tvc`), or, with
        ``given``, the mean *residual* life of a unit that has survived to
        that age along it.

        .. math::
            \text{mean} = \int_0^\infty S(t)\, dt, \qquad
            \text{mrl}(g) = \int_g^\infty S(t \mid g)\, dt .

        The outer integral is adaptive Gauss-Kronrod on panels graded
        geometrically from the start, and its nodes are simply more query
        times of :meth:`sf_tvc`: each round of refinement is one pass along
        the path, not an integral per node. A step schedule is integrated
        as the matching piecewise-constant path, whose survival is its step
        sum to rounding. For a baseline defined below zero (``Normal``,
        ``Gumbel``, ``Logistic``) the mean without ``given`` also takes off
        :math:`\int_{-\infty}^0 F(t)\, dt`, the covariate held at its value
        at 0 before it, as :meth:`sf_tvc` does.

        A path can stop units from failing: a hazard that dies away (a
        stress driven to a level with no failures, an additive hazard
        driven to 0) leaves the survival levelling off above 0. A fraction
        of units then never fails and the mean is infinite: ``inf`` is
        returned, with a warning that gives the survival where the
        integration stopped, as a univariate model's ``mean()`` returns
        ``inf`` for a limited-failure population. An additive hazard can
        also turn negative, and survival rise above 1 (#376): the mean is
        then infinite as well, or undefined (``nan``, with a warning)
        where the failure probability before time 0 grows without
        limit.

        Parameters
        ----------
        Z : StepSchedule, CovariatePath or array_like
            The covariate path, as for :meth:`sf_tvc`.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            An age survived to: the mean remaining life from it. A ``nan``
            ``given`` gives ``nan``.

        Returns
        -------
        float
            The mean (residual) life along the path.

        Examples
        --------
        At a constant covariate the mean of a Weibull proportional hazards
        model is that of a Weibull with scale
        :math:`\alpha e^{-\beta z / \gamma}` (shape :math:`\gamma`):

        >>> import numpy as np
        >>> from scipy.special import gamma
        >>> from surpyval import CovariatePath, StepSchedule, WeibullPH
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 1))
        >>> x = 100 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> alpha, shape, beta = model.params
        >>> exact = alpha * np.exp(-beta * 0.5 / shape) * gamma(1 + 1 / shape)
        >>> mean = model.mean_tvc(StepSchedule.constant([0.5]))
        >>> bool(np.isclose(mean, exact, rtol=1e-9))
        True

        Along a stress ramped from 0 to 1 over 50 hours the mean lies
        between those at the two ends, and a unit that has survived the
        ramp has less left:

        >>> ramp = CovariatePath.from_points([0, 50], [0.0, 1.0])
        >>> round(model.mean_tvc(ramp), 2)
        58.92
        >>> round(model.mean_tvc(StepSchedule.constant([0.0])), 2)
        101.32
        >>> round(model.mean_tvc(StepSchedule.constant([1.0])), 2)
        51.42
        >>> round(model.mean_tvc(ramp, given=50), 2)
        24.07
        """
        from .tvc_path import (
            _TAIL_KNOTS,
            CovariatePath,
            integrate_to_infinity,
            step_path,
        )

        self._check_tvc_evaluable()
        path = self._to_schedule(Z, xl)
        if not isinstance(path, CovariatePath):
            path = step_path(path)
        g = None if given is None else float(given)
        if g is not None and np.isnan(g):
            return np.nan
        origin = 0.0 if g is None else g
        # What the passes along the path warn of, gathered into one
        # warning each.
        seen = {"falls": 0, "points": 0, "missed": 0, "total": 0}
        worst: list = []

        def sf_at(t: npt.NDArray) -> npt.NDArray:
            H, falls, accuracy = self._tvc_hf_path(
                np.asarray(t, dtype=float), path, g
            )
            seen["falls"] += falls
            seen["points"] += H.size
            if accuracy is not None:
                seen["missed"] += accuracy[0]
                seen["total"] += accuracy[1]
                worst.append(accuracy[2:])
            with np.errstate(over="ignore"):
                # A falling additive hazard can send sf above 1 without
                # limit: the mean is then infinite (below).
                return np.exp(-H)

        params, center = self._tvc_theta(None)
        zc = self._centred(
            path._values(np.array([max(origin, 0.0)]), left=False), center
        )
        scale = self._tvc_time_scale(origin, zc, params)

        def knots(t_max: float) -> npt.NDArray:
            # The path's kinks and jumps as outer panel edges, unless it
            # repeats too often for that to help.
            if path._n_breakpoints(t_max) > _TAIL_KNOTS:
                return np.empty(0)
            return path.breakpoints(t_max)

        value, tail = integrate_to_infinity(
            sf_at, origin, scale, self._tvc_rtol, knots
        )
        if g is None and float(self.distribution.support[0]) < 0:
            # Less the area under F before 0, where the value at 0 holds.
            def ff_below(t: npt.NDArray) -> npt.NDArray:
                with np.errstate(all="ignore"):
                    return np.asarray(
                        self.model.ff(-t, zc, *params), dtype=float
                    ).ravel()

            below, below_tail = integrate_to_infinity(
                ff_below, 0.0, scale, self._tvc_rtol
            )
            value -= below
            if below_tail is not None:
                # F does not fall away before 0: an additive hazard that
                # is negative there (#376) makes it grow without limit.
                value = np.nan
        if seen["falls"]:
            self._warn_negative_hazard(
                seen["falls"], seen["points"], None, stacklevel=3
            )
        if seen["missed"]:
            from .tvc_path import warn_missed_target

            rel = max(w[0] for w in worst)
            limit = (
                "panels"
                if any(w[1] == "panels" for w in worst)
                else ("rounds")
            )
            warn_missed_target(
                seen["missed"],
                seen["total"],
                rel,
                limit,
                self._tvc_rtol,
                stacklevel=3,
            )
        if tail is not None:
            at, sf_end = tail
            warnings.warn(
                "The survival along this path has not fallen to 0: it is "
                "still {:.4g} at t = {:.4g}, so a fraction of units never "
                "fails along it (the hazard dies away, or an additive "
                "hazard turns negative) and the mean {}life is infinite; "
                "inf is returned, as a univariate model's mean() does for "
                "a limited-failure population. sf_tvc gives the survival "
                "along the path.".format(
                    sf_end, at, "residual " if g is not None else ""
                ),
                stacklevel=2,
            )
            return np.inf
        if np.isnan(value):
            warnings.warn(
                "The mean is undefined: before time 0, where the "
                "covariate's value at 0 holds, the failure probability of "
                "this additive hazards model grows without limit (its "
                "hazard is negative there), so the area it takes off the "
                "mean does not converge; nan is returned. The mean "
                "residual life (given=) avoids the times before 0.",
                stacklevel=2,
            )
        return float(value)

    def _tvc_time_scale(
        self, origin: float, zc: npt.NDArray, params: npt.NDArray
    ) -> float:
        """A time scale for the integral to infinity from ``origin``: how
        long the cumulative hazard takes to grow by 1 with the covariate
        held at ``zc`` (a power of 2; 1 where it never does)."""
        u = 2.0 ** np.arange(-60, 61)
        with np.errstate(all="ignore"):
            t = np.append(origin + u, origin)
            H = np.asarray(self.model.Hf(t, zc, *params), dtype=float)
            H = np.broadcast_to(H.ravel(), t.shape)
            grown = H[:-1] - H[-1]
        ok = np.isfinite(grown) & (grown >= 1.0)
        return float(u[np.argmax(ok)]) if ok.any() else 1.0

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        r"""
        The cumulative distribution function, or failure function, for a
        distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the failure function
            (CDF) will be calculated

        Z : array like or DataFrame
            The covariates: one row per value of ``x`` (or a single row,
            broadcast to every ``x``), in the column order used in the fit. A
            model fitted with ``fit_from_df`` also accepts a DataFrame with
            the named (or formula) columns. Other row counts are refused.

        grid : bool, optional
            ``True`` evaluates every ``x`` for every row of ``Z`` (a curve
            per subject, lifelines' ``predict_survival_function``), with
            shape ``(len(Z),) + x.shape``, row ``i`` for row ``i`` of ``Z``
            (#488). Default ``False``: rows and times paired.

        Returns
        -------

        ff : scalar or numpy array
            The scalar value of the CDF of the distribution if a scalar was
            passed. If an array like object was passed then a numpy array is
            returned with the value of the CDF at each corresponding value in
            the input array.


        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> model.ff([1, 2, 3], [[0], [0], [1]]).round(4)
        array([0.0188, 0.0618, 0.2571])
        """
        return self._eval(self.model.ff, x, Z, 0.0, grid)

    @keeps_query_shape
    def df(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        r"""
        The density function for a distribution using the parameters found in
        the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the density function
            will be calculated

        Z : array like or DataFrame
            The covariates: one row per value of ``x`` (or a single row,
            broadcast to every ``x``), in the column order used in the fit. A
            model fitted with ``fit_from_df`` also accepts a DataFrame with
            the named (or formula) columns. Other row counts are refused.

        grid : bool, optional
            ``True`` evaluates every ``x`` for every row of ``Z`` (a curve
            per subject, lifelines' ``predict_survival_function``), with
            shape ``(len(Z),) + x.shape``, row ``i`` for row ``i`` of ``Z``
            (#488). Default ``False``: rows and times paired.

        Returns
        -------

        df : scalar or numpy array
            The scalar value of the density function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the density function at each
            corresponding value in the input array.


        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> model.df([1, 2, 3], [[0], [0], [1]]).round(4)
        array([0.0326, 0.0524, 0.1289])
        """
        return self._eval(self.model.df, x, Z, 0.0, grid)

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        r"""
        The instantaneous hazard function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the instantaneous
            hazard function will be calculated

        Z : array like or DataFrame
            The covariates: one row per value of ``x`` (or a single row,
            broadcast to every ``x``), in the column order used in the fit. A
            model fitted with ``fit_from_df`` also accepts a DataFrame with
            the named (or formula) columns. Other row counts are refused.

        grid : bool, optional
            ``True`` evaluates every ``x`` for every row of ``Z`` (a curve
            per subject, lifelines' ``predict_survival_function``), with
            shape ``(len(Z),) + x.shape``, row ``i`` for row ``i`` of ``Z``
            (#488). Default ``False``: rows and times paired.

        Returns
        -------

        hf : scalar or numpy array
            The scalar value of the instantaneous hazard function of the
            distribution if a scalar was passed. If an array like object was
            passed then a numpy array is returned with the value of the
            instantaneous hazard function at each corresponding value in the
            input array.


        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> model.hf([1, 2, 3], [[0], [0], [1]]).round(4)
        array([0.0332, 0.0559, 0.1735])
        """
        return self._eval(self.model.hf, x, Z, 0.0, grid)

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        r"""

        The cumulative hazard function for a distribution using the parameters
        found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the cumulative hazard
            function will be calculated

        Z : array like or DataFrame
            The covariates: one row per value of ``x`` (or a single row,
            broadcast to every ``x``), in the column order used in the fit. A
            model fitted with ``fit_from_df`` also accepts a DataFrame with
            the named (or formula) columns. Other row counts are refused.

        grid : bool, optional
            ``True`` evaluates every ``x`` for every row of ``Z`` (a curve
            per subject, lifelines' ``predict_survival_function``), with
            shape ``(len(Z),) + x.shape``, row ``i`` for row ``i`` of ``Z``
            (#488). Default ``False``: rows and times paired.

        Returns
        -------

        Hf : scalar or numpy array
            The scalar value of the cumulative hazard function of the
            distribution if a scalar was passed. If an array like object was
            passed then a numpy array is returned with the value of the
            cumulative hazard function at each corresponding value in the input
            array.


        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> model.Hf([1, 2, 3], [[0], [0], [1]]).round(4)
        array([0.0189, 0.0638, 0.2972])
        """
        return self._eval(self.model.Hf, x, Z, 0.0, grid)

    def random(
        self,
        size: int,
        Z: "npt.ArrayLike | pd.DataFrame",
        random_state: Any = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        r"""

        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------
        size : int
            The number of random samples to draw for each covariate row
            (for each distinct stress, for an accelerated life model).

        Z : scalar or array like
            The covariate row(s) (or stress value(s)) at which to draw: one
            row per covariate vector, or a scalar / 1-D array of stresses
            for a single-stress accelerated life model.

        random_state : None, int or numpy.random.Generator, optional
            The seed of the draw. ``None`` (the default) draws from numpy's
            global generator, so ``np.random.seed`` reproduces it; an int
            or a ``Generator`` gives a stream of its own, which neither
            depends on nor advances the global one.

        Returns
        -------
        x : numpy array
            The ``size`` draws for each row, concatenated row by row.
        Z : numpy array
            A 2-D array giving the covariate row each draw was made at.


        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> np.random.seed(1)
        >>> x_rand, Z_rand = model.random(5, Z[:1])
        >>> x_rand.round(3)
        array([ 8.919,  5.095, 33.929, 10.666, 13.97 ])
        >>> model.random(3, Z[:1], random_state=0)[0].round(3)
        array([ 6.111, 11.235, 18.691])
        >>> Z_rand
        array([[0.],
               [0.],
               [0.],
               [0.],
               [0.]])
        """
        # Dispatch to the regression fitter's own covariate-aware sampler
        # (#261): the previous implementation ignored ``Z`` entirely and
        # read LFP/ZI attributes regression fits never set, so it crashed
        # on every path.
        Z = self._prepare_Z(Z)
        if hasattr(self.model, "random"):
            if not self._has_center():
                return self.model.random(
                    size, Z, *self._eval_params(), random_state=random_state
                )
            # A baseline at the covariate means (#463): draw at Z - center
            # and report the rows as given.
            x, Z_out = self.model.random(
                size,
                self._centred(Z),
                *self._eval_params(),
                random_state=random_state,
            )
            return x, Z_out + self.center
        raise NotImplementedError(
            f"random() is not implemented for {self.kind} models."
        )

    def _require_data(self, what: str) -> None:
        """Refuse, by name, an operation that needs the fitted data.

        A model rebuilt by :meth:`from_dict` keeps its parameters, stored
        covariance and log-likelihood but not the data it was fitted to,
        and used to fail with ``AttributeError: no attribute 'data'``.
        """
        if getattr(self, "data", None) is None:
            raise ValueError(
                "{} needs the data the model was fitted to, which a model "
                "restored with from_dict / from_json does not carry. Call it "
                "on the fitted model, or refit.".format(what)
            )

    # neg_ll/aic/bic/aic_c come from InformationCriteriaMixin.
    def _ic_sample_size_from_data(self) -> float:
        self._require_data("bic() / aic_c()")
        # The observed failures (exact, left- or interval-censored), as for
        # every model's BIC and AIC_c (ic_sample_size); only exact failures
        # were counted here, unlike the univariate models. A
        # time-varying-covariate fit has one row per interval, but only a
        # subject's last interval can end in a failure, so the count is
        # unchanged by splitting its time into more intervals; the fallback
        # for data with no failure counts subjects, not interval rows.
        return ic_sample_size(
            self.data.c,
            self.data.n,
            n_rows=getattr(self, "_ic_n_total", None),
        )

    # ``self.k`` is the number of estimated parameters, so the AIC/BIC
    # penalties and the AIC_c correction all use it (the mixin's defaults).
    # Fixed parameters -- and the accelerated-life placeholder for the life
    # parameter -- used to be counted as well.

    # -- confidence bounds -------------------------------------------------

    def _check_inference(self) -> None:
        # A model deserialised with a stored covariance can produce bounds
        # without the original data.
        if getattr(self, "_restored_covariance", None) is not None:
            return
        if getattr(self, "_restored", False):
            # Restored without a covariance: to_dict stores one only when
            # it was finite at fit time, and the data are not stored.
            raise ValueError(
                "Confidence bounds are unavailable: this model was restored "
                "from a dict that carries no parameter covariance (it could "
                "not be computed when the model was saved), and a restored "
                "model does not keep the data to recompute it."
            )
        if not hasattr(self, "data") or getattr(self, "res", None) is None:
            raise ValueError(
                "Confidence bounds are only available for models fit from "
                "data; from_params models carry no likelihood."
            )

    @property
    def parameter_names(self) -> CallableList:
        """
        Names of ``params``, in order: the distribution's parameters, then
        the covariate coefficients (or life-model parameters). The list
        lines up with ``params``, :meth:`covariance` and
        :meth:`standard_errors` entry by entry, fixed parameters included.
        In an accelerated life model the slot named by ``life_parameter``
        is a placeholder, not a fitted value, and is named too.

        Until v0.22 this was a method; calling it,
        ``model.parameter_names()``, still returns the list, with a
        ``DeprecationWarning``, until v0.23.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullPH
        >>> x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
        >>> Z = np.array([0.0, 1, 0, 1, 0, 1, 1, 0])
        >>> model = WeibullPH.fit(x=x, Z=Z)
        >>> model.parameter_names
        ['alpha', 'beta', 'beta_0']
        """
        dist_names = list(self.distribution.parameter_names)
        phi_map = self.reg_model.phi_param_map
        phi_names = [
            k for k, _ in sorted(phi_map.items(), key=lambda kv: kv[1])
        ]
        return CallableList(
            dist_names + phi_names,
            "ParametricRegressionModel.parameter_names",
        )

    # ``param_names``, the name the model had before v0.22, reads
    # ``parameter_names`` for one release, with a DeprecationWarning.
    param_names = RenamedAttribute("parameter_names")

    def covariance(self) -> npt.NDArray:
        """
        Approximate covariance matrix of the fitted parameters, ordered to
        match :attr:`parameter_names`. Computed as the inverse of the Hessian
        of the negative log-likelihood at the MLE (the observed
        information): the exact one the fit computed with autograd to check
        its answer. It falls back to a numerical Hessian where there is
        none at the fitted parameters: an accelerated-life fit (which makes
        no such check), an AFT time-varying fit (whose likelihood autograd
        cannot differentiate), a fit with no finite maximum, one whose
        Hessian there is not positive definite, or a model whose
        parameters or data have changed since the fit. Fixed parameters get
        a zero row/column. It is computed once, and kept while the
        parameters stay as they are.

        A parameter driven to a boundary breaks the Wald approximation; the
        covariance is then returned filled with ``nan`` (with a warning).

        For a fit on centred covariates (#463) the information is that of
        the centred fit, carried to the reported parameters (the baseline
        at ``Z = 0``) by the jacobian of the map between them.
        """
        restored = getattr(self, "_restored_covariance", None)
        if restored is not None:
            return restored
        self._check_inference()
        _, _, cov = self._inference_state()
        if self._fit_centring is not None:
            J = self._fit_centring[2]
            cov = J @ cov @ J.T
        if self.aliased.size:
            # No variance for a coefficient that was not estimated (#476).
            cov = np.array(cov, dtype=float)
            cov[self.k_dist + self.aliased, :] = np.nan
            cov[:, self.k_dist + self.aliased] = np.nan
        return cov

    def _inference_state(
        self,
    ) -> "tuple[npt.NDArray, npt.NDArray | None, npt.NDArray]":
        """``(params, center, covariance)`` of the parameterisation the
        confidence bounds are computed in: that of the centred fit behind
        a model that reports its baseline at 0 (#463), where the
        coefficients and the baseline are not nearly collinear, else the
        model's own."""
        restored = getattr(self, "_restored_covariance", None)
        if restored is not None:
            return (
                np.asarray(self._eval_params(), dtype=float),
                self.center,
                restored,
            )
        if self._fit_centring is not None:
            params, center = self._fit_centring[:2]
        else:
            params, center = self._eval_params(), self.center
        p_hat = np.asarray(params, dtype=float)
        return p_hat, center, self._observed_covariance(p_hat, center)

    def _observed_covariance(
        self, p_hat: npt.NDArray, center: "npt.NDArray | None"
    ) -> npt.NDArray:
        """The inverse of the Hessian of the negative log-likelihood at
        ``p_hat``, the baseline at ``center``: the exact one the fit kept
        (``_information``) if it was computed at that point, else a
        numerical one. A covariance computed is kept, and returned (as a
        copy) while the point is the same."""
        point = self._covariance_point(p_hat, center)
        cached = self._covariance_cache
        if cached is not None and _same_point(cached[0], point):
            return cached[1].copy()
        names = self.parameter_names
        held = self._held()
        free = [i for i, nm in enumerate(names) if nm not in held]
        n = len(names)
        cov = np.zeros((n, n))
        if not free:
            return cov
        step = self._hessian_step(p_hat)[free]
        info = self._information
        if info is not None and _same_point(info[0], point):
            H = info[1]
        else:
            H = self._numerical_information(p_hat, center, free, step)
        bad = not np.all(np.isfinite(H))
        if not bad:
            # Invert in step-scaled coordinates: with a parameter many
            # orders of magnitude from the others the raw information
            # matrix is too ill-conditioned to invert directly.
            try:
                cov_free = np.linalg.inv(H * np.outer(step, step)) * np.outer(
                    step, step
                )
            except np.linalg.LinAlgError:
                bad = True
        if bad:

            warnings.warn(
                "The information matrix could not be inverted (the optimum "
                "may be at a parameter boundary); covariance is unavailable."
            )
            return np.full((n, n), np.nan)
        cov[np.ix_(free, free)] = cov_free
        self._covariance_cache = (point, cov.copy())
        return cov

    def _covariance_point(
        self, p_hat: npt.ArrayLike, center: "npt.ArrayLike | None"
    ) -> tuple:
        """What the covariance at ``p_hat``, the baseline at ``center``,
        is a function of: those two (copied, and a centre of zeros read as
        none), the data and the names of the fixed parameters. Two points
        are the same by :func:`_same_point`."""
        centre = None
        if center is not None and np.any(center):
            centre = np.array(center, dtype=float)
        return (
            np.array(p_hat, dtype=float),
            centre,
            getattr(self, "data", None),
            tuple(sorted(self.fixed)),
        )

    def _numerical_information(
        self,
        p_hat: npt.NDArray,
        center: "npt.NDArray | None",
        free: list,
        step: npt.NDArray,
    ) -> npt.NDArray:
        """The numerical Hessian of the negative log-likelihood in the free
        parameters at ``p_hat``, the baseline at ``center``."""
        data = self.data
        if center is not None and np.any(center):
            from ._fit_skeleton import centred_copy

            data = centred_copy(data, center)

        def neg_ll_free(free_vals: npt.NDArray) -> float:
            full = p_hat.copy()
            full[free] = free_vals
            return self.model.neg_ll(data, *full)

        return numerical_hessian(neg_ll_free, p_hat[free], step)

    def _parameter_bounds(self) -> list:
        """``(lower, upper)`` for every entry of ``params``: the
        distribution's support bounds, then the life model's parameter
        bounds for an accelerated-life model (the other families'
        coefficients are unbounded)."""
        n_phi = len(self.params) - self.k_dist
        phi_bounds: Any = ((None, None),) * n_phi
        if self.kind == "Accelerated Life":
            declared = getattr(self.reg_model, "phi_bounds", phi_bounds)
            if callable(declared):
                declared = declared(np.asarray(self.data.Z))
            phi_bounds = declared
        return [*self.distribution.bounds, *phi_bounds]

    def _hessian_step(self, p_hat: npt.NDArray) -> npt.NDArray:
        """Finite-difference step for the covariance Hessian, and the
        scale its inversion (numerical or exact) is done in.

        The usual ``eps**(1/3) * max(|p|, 1e-2)``, except that a parameter
        closer to one of its bounds than a few steps gets a step relative
        to that distance. The absolute floor is far larger than, say, an
        accelerated-life coefficient of 5.6e-22 (``InversePower``'s ``a``
        for lives in the thousands), so the difference stepped outside the
        support and the covariance came back nan.
        """
        h = np.finfo(float).eps ** (1.0 / 3.0)
        step = h * np.maximum(np.abs(p_hat), 1e-2)
        for i, (lower, upper) in enumerate(self._parameter_bounds()):
            gaps = [
                p_hat[i] - lower if lower is not None else np.inf,
                upper - p_hat[i] if upper is not None else np.inf,
            ]
            gap = min(gaps)
            if 0 < gap < 10 * step[i]:
                step[i] = h * gap
        return step

    def standard_errors(self) -> npt.NDArray:
        """
        Standard errors of the fitted parameters (square roots of the diagonal
        of :meth:`covariance`), ordered to match :attr:`parameter_names`.
        """
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(self.covariance()))

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Confidence bound(s) on a single fitted parameter.

        Wald bounds from the observed information, computed on a scale chosen
        from the parameter's support so the result stays inside it: log for a
        one-sided-bounded distribution parameter (e.g. a positive scale), the
        natural scale for the unbounded covariate coefficients.

        Parameters
        ----------
        name : str
            The parameter to bound; one of :attr:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.
        """
        self._check_inference()
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        if name == self.life_parameter:
            raise ValueError(
                "{!r} is not a parameter of this accelerated life model: it "
                "is {} at each stress. Bound the life-model parameters "
                "({}) instead, or the predictions with cb().".format(
                    name,
                    self._life_relation(),
                    ", ".join(names[self.k_dist :]),
                )
            )
        idx = names.index(name)
        p_hat = float(self.params[idx])
        var = float(self.covariance()[idx, idx])

        # Distribution parameters carry the distribution's support bounds; the
        # covariate coefficients are unbounded.
        dist_bounds = list(self.distribution.bounds)
        n_phi = len(names) - self.k_dist
        all_bounds = dist_bounds + [(None, None)] * n_phi
        lower, upper = all_bounds[idx]
        return wald_bound_on_support(
            p_hat, var, lower, upper, alpha_ci, bound, name=name
        )

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        r"""
        Confidence bounds on a predicted function at covariate vector ``Z``.

        The bounds propagate the fitted parameter covariance through the
        requested function by the delta method. ``sf``/``ff``/``Hf`` are
        derived from one bound on the baseline family's probability-plot
        scale, as for the univariate models: ``log H`` for a Weibull,
        Exponential, Rayleigh or Gumbel baseline, the normal quantile of
        ``F`` for a Normal or LogNormal one, the logit of ``F`` for the rest
        (#504; every band was on the logit before v0.22). Each keeps ``sf``
        in ``(0, 1)``, and is formed from the cumulative hazard so the ``Hf``
        bound has no ceiling where ``sf`` underflows. ``hf``/``df`` use a
        log-scale bound (so they stay positive).

        Parameters
        ----------
        x : array like or scalar
            Times at which to evaluate the bound(s).
        Z : array like
            A single covariate vector, used at every ``x`` (one row per
            ``x`` is paired element-wise, as for :meth:`sf`).
        on : {'sf', 'ff', 'Hf', 'hf', 'df'}, optional
            The function to bound. Default ``'sf'``.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.

        Returns
        -------
        numpy array
            The confidence bound(s) on ``on`` at each ``x``.
        """
        self._check_inference()
        valid = ("sf", "R", "ff", "F", "Hf", "hf", "df")
        if on not in valid:
            raise ValueError("`on` must be one of {}".format(valid))
        if bound not in ("two-sided", "lower", "upper"):
            raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # In the parameterisation of the centred fit when there is one
        # (#463): the bounds are the same function of the data, and there
        # the coefficients are not nearly collinear with the baseline.
        if np.ndim(self._prepare_Z(Z)) == 2:
            # Rows and times paired, as for sf (#488).
            check_paired_rows(
                np.size(x), np.shape(self._prepare_Z(Z))[0], grid=False
            )
        params, center, cov = self._inference_state()
        Zp = self._centred(self._prepare_Z(Z), center)
        if self.kind == "Additive Hazard":
            # Not below the support, where nothing has happened yet.
            self._warn_if_hazard_negative(
                x,
                self._centred(self._prepare_Z(Z)),
                x >= self.distribution.support[0],
                stacklevel=4,
            )

        if on in ("hf", "df"):
            fn = self.model.hf if on == "hf" else self.model.df
            est = np.asarray(fn(x, Zp, *params), dtype=float)
            se = delta_method_se(lambda p: fn(x, Zp, *p), params, cov)
            return log_transformed_cb(est, se, alpha_ci, bound)

        # Below the support nothing has happened yet: H is 0 there, as
        # for sf (the bound is then the estimate, sf = 1). An additive
        # hazard's beta'Z x is not 0 at a negative x, and gave a band
        # around a survival of about 0.9 where sf is 1.
        lower = self.distribution.support[0]
        below = np.asarray(x) < lower
        x_in = np.where(below, lower + 1.0 if np.isfinite(lower) else 0.0, x)

        def H_of(p: npt.NDArray) -> npt.NDArray:
            H = np.asarray(self.model.Hf(x_in, Zp, *p), dtype=float)
            return np.where(below, 0.0, H)

        def sf_of(p: npt.NDArray) -> npt.NDArray:
            S = np.asarray(self.model.sf(x_in, Zp, *p), dtype=float)
            return np.where(below, 1.0, S)

        return self._sf_bounds(
            H_of,
            sf_of,
            params,
            cov,
            np.shape(x),
            on,
            alpha_ci,
            bound,
        )

    @property
    def _cb_link(self) -> str:
        """The scale of the Wald bands on ``sf``/``ff``/``Hf``: the baseline
        family's probability-plot scale, as for the univariate models
        (#477, #504)."""
        return cb_link(self.distribution)

    def _sf_bounds(
        self,
        H_of: Any,
        sf_of: Any,
        params: npt.NDArray,
        cov: npt.NDArray,
        shape: tuple,
        on: str,
        alpha_ci: float,
        bound: str,
    ) -> npt.NDArray:
        """The bounds of :meth:`cb` and :meth:`cb_tvc` on ``sf``, ``ff`` or
        ``Hf``: a Wald bound on the baseline family's scale (``log H``,
        the normal quantile of ``F`` or the logit of ``F``, see
        :attr:`_cb_link`), formed from the cumulative hazard ``H_of(p)``
        and propagated by the delta method, carried to the scale of
        ``on``. ``sf_of(p)`` gives the survival where ``H`` is negative (an
        additive hazard's ``sf`` above 1)."""
        link = self._cb_link
        name = {"R": "sf", "F": "ff"}.get(on, on)

        # Formed from the cumulative hazard, so the Hf bound has no
        # ceiling where sf underflows (#418).
        def u_of(p: npt.NDArray) -> npt.NDArray:
            return sf_link_from_H(H_of(p), link)

        H_hat = np.asarray(H_of(params), dtype=float)
        H_hat = np.broadcast_to(H_hat, np.broadcast_shapes(H_hat.shape, shape))
        u_hat = np.broadcast_to(u_of(params), H_hat.shape)
        with np.errstate(invalid="ignore"):
            # inf - inf where sf is 1 or 0: the bounds are the estimate
            # there (link_band)
            se_u = delta_method_se(u_of, params, cov)
        cb = link_band(u_hat, se_u, alpha_ci, bound, link, name)
        # A negative H (an additive hazards sf above 1, documented) has no
        # point on these scales: it keeps the clipped-sf logit bound.
        negative = H_hat < 0
        if not negative.any():
            return cb
        sf_hat = np.asarray(sf_of(params), dtype=float)
        se = delta_method_se(sf_of, params, cov)

        def end(sign: float, tail: float) -> npt.NDArray:
            # One end on the scale of ``on``; sign +1 is sf's upper.
            sf_c = logit_sf_bound(sf_hat, se, sign, tail)
            if name == "sf":
                return sf_c
            with np.errstate(divide="ignore"):
                return 1.0 - sf_c if name == "ff" else -np.log(sf_c)

        # ff and Hf decrease in sf: their lower end is sf's upper.
        flip = -1.0 if name == "sf" else 1.0
        if bound == "two-sided":
            fallback = np.stack(
                [end(flip, alpha_ci / 2.0), end(-flip, alpha_ci / 2.0)],
                axis=-1,
            )
            negative = negative[..., None]
        else:
            sign = flip if bound == "lower" else -flip
            fallback = end(sign, alpha_ci)
        return np.where(negative, fallback, cb)

    @keeps_query_shape
    def cb_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        r"""
        Confidence bounds on the survival, failure probability or
        cumulative hazard along a covariate path ``Z(t)``: a step schedule
        or a continuously varying path, as for :meth:`sf_tvc`.

        The bounds are those of :meth:`cb`, carried along the path: a Wald
        bound on the baseline family's probability-plot scale, formed from
        the cumulative hazard of :meth:`Hf_tvc`, its standard error
        propagated from the fitted parameter covariance by the delta method.
        ``ff`` and ``Hf`` follow from the same bound, so the three agree with
        each other, and a constant path gives :meth:`cb`.

        Along a
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
        the integral is taken on a quadrature mesh adapted at the fitted
        parameters and then held fixed, so the function differentiated is
        smooth in the parameters. The cost is ``2k + 1`` evaluations along
        the path for ``k`` parameters.

        With ``given`` the bounds are on the conditional survival
        :math:`S(x \mid \text{survived to } g)` of :meth:`sf_tvc`: 1, with
        no width, at and before ``given``.

        Parameters
        ----------
        x : array_like
            Times at which to bound the function.
        Z : StepSchedule, CovariatePath or array_like
            The covariate path, as for :meth:`sf_tvc`.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            Condition on survival to this age, as for :meth:`sf_tvc`. A
            ``nan`` ``given`` gives ``nan``.
        on : {'sf', 'ff', 'Hf'}, optional
            The function to bound (``'R'`` and ``'F'`` are accepted for
            ``'sf'`` and ``'ff'``). Default ``'sf'``. The hazard and the
            density along a path are not bounded.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds put ``[lower, upper]`` on the last axis.

        Returns
        -------
        numpy array
            The confidence bound(s) on ``on`` at each ``x``: the query's
            shape, with ``[lower, upper]`` on a last axis for two-sided
            bounds.

        Examples
        --------
        A proportional hazards model along a stress ramped from 0 to 1 over
        50 hours:

        >>> import numpy as np
        >>> from surpyval import CovariatePath, WeibullPH
        >>> rng = np.random.default_rng(0)
        >>> Z = rng.uniform(0, 1, (200, 1))
        >>> x = 100 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> ramp = CovariatePath.from_points([0, 50], [0.0, 1.0])
        >>> model.sf_tvc([40, 80], ramp).round(4)
        array([0.771 , 0.1911])
        >>> model.cb_tvc([40, 80], ramp).round(4)
        array([[0.7243, 0.8109],
               [0.1256, 0.267 ]])

        A constant path gives the ordinary bounds:

        >>> flat = CovariatePath.from_points([0], [0.5])
        >>> bool(np.allclose(model.cb_tvc([40, 80], flat),
        ...                  model.cb(np.array([40, 80]), [0.5])))
        True
        """
        from .tvc_path import CovariatePath

        self._check_inference()
        valid = ("sf", "R", "ff", "F", "Hf")
        if on not in valid:
            raise ValueError(
                "`on` must be one of {} for cb_tvc: the survival, failure "
                "probability and cumulative hazard along a path are "
                "bounded (the hazard and density along a path are "
                "not)".format(valid)
            )
        if bound not in ("two-sided", "lower", "upper"):
            raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")
        self._check_tvc_evaluable()
        xq: npt.NDArray = np.atleast_1d(np.asarray(x, dtype=float))
        shape = xq.shape + ((2,) if bound == "two-sided" else ())
        g = None if given is None else float(given)
        if (g is not None and np.isnan(g)) or xq.size == 0:
            # A missing conditioning age: nothing is known (as sf_tvc).
            self._to_schedule(Z, xl)
            return np.full(shape, np.nan)
        on_path = isinstance(Z, CovariatePath)
        # In the parameterisation of the centred fit when there is one, as
        # for cb (#463).
        params, center, cov = self._inference_state()
        # The path's mesh, adapted at the fitted parameters and then held.
        frozen: dict = {}

        def H_of(p: npt.NDArray) -> npt.NDArray:
            theta = (p, center)
            if on_path or g is None:
                H = self._hf_tvc(xq, Z, xl, g, theta, frozen)[0]
            else:
                H = (
                    self._hf_tvc(xq, Z, xl, None, theta)[0]
                    - self._hf_tvc(g, Z, xl, None, theta)[0]
                )
            if g is not None:
                # Survival to x <= g is certain (as sf_tvc, #523).
                H = np.where(xq <= g, 0.0, H)
            return H

        # The estimate first: it adapts the mesh, and gives sf_tvc's
        # warnings (a falling additive hazard, a missed target).
        H, falls, accuracy = self._hf_tvc(
            xq, Z, xl, g if on_path else None, (params, center), frozen
        )
        self._warn_tvc(H, falls, accuracy, stacklevel=5)
        return self._sf_bounds(
            H_of,
            lambda p: np.exp(-H_of(p)),
            params,
            cov,
            xq.shape,
            on,
            alpha_ci,
            bound,
        )

    def plot(
        self,
        ax: "Axes | None" = None,
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
    ) -> "Axes":
        r"""

        A method to plot the survival function of the distribution at the mean
        covariate vector against a non-parametric estimate of the pooled
        fitted data (the exponentiated Nelson-Aalen estimate, which ignores
        the covariates), with a delta-method confidence band. It needs the
        fitted data, so it is not available on a restored model.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to draw on; a new one is created if not provided.
        plot_bounds : bool, optional
            Whether to draw the confidence band around the fitted survival
            curve. Default True.
        alpha_ci : float, optional
            Total tail probability of the band. Default 0.05.
        """

        self._require_data("plot()")
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gca()

        x, r, d = self.data.to_xrd()
        x_plot = np.linspace(self.data.x.min(), self.data.x.max(), 1000)

        Z_mean = self.data.Z.mean(axis=0)
        ax.step(x, np.exp(-(d / r).cumsum()), color="r", where="post")
        sf = self.sf(x_plot, Z_mean)
        ax.plot(x_plot, sf, color="b")
        if plot_bounds:
            cb = self.cb(x_plot, Z_mean, on="sf", alpha_ci=alpha_ci)
            ax.fill_between(
                x_plot,
                cb[:, 0],
                cb[:, 1],
                color="b",
                alpha=0.2,
                label=f"{(1 - alpha_ci) * 100:g}% Confidence Band",
            )
        return ax


def _same_point(a: tuple, b: tuple) -> bool:
    """Whether ``a`` and ``b``, as ``_covariance_point`` gives them, are
    the same point: equal parameters and centres, the same data object and
    the same fixed parameters."""
    params_a, centre_a, data_a, fixed_a = a
    params_b, centre_b, data_b, fixed_b = b
    if data_a is not data_b or fixed_a != fixed_b:
        return False
    if not np.array_equal(params_a, params_b):
        return False
    if centre_a is None or centre_b is None:
        return centre_a is None and centre_b is None
    return bool(np.array_equal(centre_a, centre_b))
