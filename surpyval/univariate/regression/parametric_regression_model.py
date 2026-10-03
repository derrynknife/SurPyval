from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, cast

import autograd.numpy as np
import numpy.typing as npt

from surpyval.serialisation import SerialisableMixin, stamp_schema
from surpyval.univariate.information_criteria import (
    InformationCriteriaMixin,
    ic_sample_size,
)
from surpyval.utils.covariates import loaded_coefficient_names
from surpyval.utils.data_summary import data_summary
from surpyval.utils.deprecation import CallableList, RenamedAttribute
from surpyval.utils.no_maximum import maximum_entry, restored_maximum
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.validation import warn_outside_unit_interval

from ._concordance import ConcordanceMixin
from ._covariate_link import CovariateLink
from ._inference import InferenceMixin
from ._kinds import (
    ACCELERATED_FAILURE_TIME,
    ACCELERATED_LIFE,
    ADDITIVE_HAZARD,
    PROPORTIONAL_HAZARD,
    PROPORTIONAL_ODDS,
)
from ._prediction import ConditionalSurvivalMixin, quantiles_by_inversion
from ._tvc_evaluation import TVCEvaluationMixin
from .regression_data import (
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

    from surpyval.univariate.parametric.parametric_fitter import (
        ParametricFitter,
    )
    from surpyval.utils.surpyval_data import SurpyvalData

    from .accelerated_life.lifemodel import LifeModel


# Regression families whose fitted model round-trips through ``to_dict`` /
# ``from_dict``: each has a fixed-form covariate link (a log-linear multiplier
# ``exp(beta'Z)`` or an additive ``beta'Z`` term) that is fully determined by
# the ``kind`` plus the distribution and coefficients, so the fitter -- and
# therefore every prediction -- can be rebuilt from the distribution's name.
# Maps kind -> (public fitter factory name, covariate-link form).
_SERIALISABLE_KINDS: "dict[str, tuple[str, str]]" = {
    ACCELERATED_FAILURE_TIME: ("AFT", "exp"),
    PROPORTIONAL_HAZARD: ("PH", "exp"),
    PROPORTIONAL_ODDS: ("PO", "exp"),
    ADDITIVE_HAZARD: ("AH", "additive"),
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
    ConditionalSurvivalMixin,
    TVCEvaluationMixin,
    InferenceMixin,
    ConcordanceMixin,
    InformationCriteriaMixin,
    SerialisableMixin,
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

    # Every attribute a fitted model carries. The model is created empty
    # and filled in by its builder: ``assemble_regression_model`` (every
    # ``fit``, the time-varying-covariate fits and ``AcceleratedLife``),
    # then ``fit_from_df`` / ``fit_tvc`` add theirs, and ``from_dict``
    # (for a model restored without its data). An attribute with a value
    # here is optional: the builders that have nothing to say leave the
    # default. ``conformance/test_attributes.py`` checks that every
    # builder gives the same set, and nothing undeclared.

    # -- set by every builder ---------------------------------------------
    #: The fitted parameters: the distribution's, then the covariate
    #: coefficients (or life-model parameters); an aliased one is nan.
    params: npt.NDArray
    #: ``params[:k_dist]``, the baseline distribution's parameters.
    dist_params: npt.NDArray
    #: ``params[k_dist:]``, the covariate coefficients (or life-model
    #: parameters).
    phi_params: npt.NDArray
    #: The number of estimated parameters (fixed and aliased ones are not
    #: counted), the ``k`` of the information criteria.
    k: int
    #: The number of baseline distribution parameters.
    k_dist: int
    #: The family, one of the names in ``_kinds``: ``"Proportional
    #: Hazard"``, ``"Accelerated Failure Time"``, ``"Proportional Odds"``,
    #: ``"Additive Hazard"`` or ``"Accelerated Life"``.
    kind: str
    #: ``{name: value}`` of the parameters held fixed in the fit (an
    #: accelerated life model's placeholder for its life parameter
    #: included; the aliased coefficients are not).
    fixed: dict[str, float]
    #: The baseline distribution (``Weibull``, ...).
    distribution: ParametricFitter
    #: The same distribution, under the name the fitters use.
    dist: ParametricFitter
    #: ``{name: position}`` of the baseline distribution's parameters.
    distribution_param_map: dict[str, int]
    #: ``{name: position}`` of the covariate coefficients (life-model
    #: parameters), counted from the first of them.
    phi_param_map: dict[str, int]
    #: How the covariates act: a :class:`CovariateLink` (its ``name``,
    #: ``phi_param_map`` and ``phi``), or the life model of an
    #: accelerated life model.
    reg_model: "CovariateLink | LifeModel"
    #: The regression fitter that built the model; its ``sf(x, Z,
    #: *params)`` and the others (and ``neg_ll``) are the model's.
    model: Any
    #: The fitted negative log-likelihood.
    _neg_ll: float

    # -- set by the fits from data (absent on a model from ``from_dict``) --
    #: The data fitted to, with its covariates ``Z``.
    data: SurpyvalData
    #: The optimiser's result (``scipy.optimize.OptimizeResult``).
    res: Any
    #: The objective the search minimised, in the transformed search
    #: space; set by the accelerated life fit only.
    fun: Callable[[npt.NDArray], Any]

    # -- optional ----------------------------------------------------------
    #: Not parameters of a regression model; kept at the univariate
    #: models' neutral values (no offset, no defective fraction, no zero
    #: inflation), which ``to_dict`` stores.
    gamma: float = 0.0
    p: float = 1.0
    f0: float = 0.0
    #: The covariate point the baseline parameters are at: zeros (or
    #: ``None``, for an accelerated life model) when they are those of a
    #: unit with ``Z = 0``, the default. A fit with ``center=True`` keeps
    #: its baseline at the ``n``-weighted covariate means (#463), stored
    #: here, and every prediction uses ``Z - center``.
    center: "npt.NDArray | None" = None
    #: Covariate metadata of a model fitted from a pandas DataFrame (see
    #: ``DataFrameRegressionMixin.fit_from_df``): the coefficients'
    #: column names, the formula, and the formulaic model spec that
    #: encodes a DataFrame's columns. ``None`` for a fit from arrays.
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None
    #: What the fit reached, one of ``MAXIMUM_STATES``
    #: (``surpyval.utils.no_maximum``): ``"verified"`` (a zero gradient and
    #: a positive-definite Hessian), ``"unverified"`` or ``"no finite
    #: maximum"``, each as the fit's warnings say (principles 12 and 13);
    #: ``"unknown"`` for a model restored from a dict saved without it.
    maximum: str = "unknown"
    #: Whether the model was fitted to time-varying covariates
    #: (``fit_tvc``, ``fit_tvc_timeline``), one data row per interval.
    is_tvc: bool = False
    #: The number of subjects of a time-varying-covariate fit.
    n_subjects: "int | None" = None
    #: The weighted number of subjects of a time-varying-covariate fit,
    #: the sample size of ``bic`` / ``aic_c`` when none failed.
    _ic_n_total: "float | None" = None
    #: Set only on models rebuilt by :meth:`from_dict` that carried a stored
    #: parameter covariance; lets them produce confidence bounds without the
    #: original data. ``None`` on freshly fitted models.
    _restored_covariance: "npt.NDArray | None" = None
    #: True on models rebuilt by :meth:`from_dict`, which carry no data.
    _restored: bool = False
    #: The printout's "Data" line of a model rebuilt by :meth:`from_dict`
    #: (#508).
    _data_summary: "str | None" = None
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
    #: The likelihood-ratio searches of ``cb`` / ``param_cb`` with
    #: ``method="lr"`` (``_likelihood_ratio.lr_search``), with what they
    #: have found, kept while the parameters and data stay as they are;
    #: not pickled (``__getstate__``).
    _lr_searches: "list | None" = None
    # The information criteria's sample size ``_ic_n`` and their caches
    # ``_aic``, ``_bic``, ``_aic_c`` are InformationCriteriaMixin's.

    def __getstate__(self) -> dict:
        # The likelihood-ratio searches, with the regions and bounds they
        # have found, are rebuilt where a bound is asked for again (#617).
        state = dict(self.__dict__)
        state.pop("_lr_searches", None)
        return state

    # -- serialisation -----------------------------------------------------

    def _serialise_link(self) -> "dict[str, Any]":
        """The link-identity head of :meth:`to_dict`.

        Encodes just enough to rebuild the covariate link: for the fixed-form
        families the link name; for Accelerated Life the built-in life-model
        name. Raises ``NotImplementedError`` for any link that cannot be
        reconstructed from a name.
        """
        phi_param_map = self.reg_model.phi_param_map
        if not isinstance(phi_param_map, dict):
            raise NotImplementedError(
                "This model's covariate coefficients are not a fixed name map "
                "and cannot be serialised."
            )
        reg_name = self.reg_model.name
        base: dict[str, Any] = {
            "parameterization": "parametric-regression",
            "kind": self.kind,
            "distribution": self.distribution.name,
            "phi_param_map": {
                str(k): int(v) for k, v in phi_param_map.items()
            },
        }

        if self._is_accelerated_life():
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
            if LIFE_MODELS[reg_name].n_stresses is None:
                # A parameter per stress column (GeneralLogLinear): the
                # reader resolves the model for this many columns.
                life_model = cast("LifeModel", self.reg_model)
                base["n_stresses"] = int(cast(int, life_model.n_stresses))
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
        out["gamma"] = float(self.gamma)
        out["p"] = float(self.p)
        out["f0"] = float(self.f0)
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
            cov = self._restored_covariance
        if cov is not None and np.all(np.isfinite(cov)):
            out["covariance"] = np.asarray(cov, dtype=float).tolist()
        if hasattr(self, "_neg_ll"):
            out["_neg_ll"] = float(self._neg_ll)
        out.update(maximum_entry(self.maximum))
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
        # Saved coefficient names -> their names now (#614)
        renamed: dict[str, str] = {}

        reg_model: "CovariateLink | LifeModel"
        if kind == ACCELERATED_LIFE:
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
            if reg_model.n_stresses is None:
                n_stresses = model_dict.get("n_stresses")
                if not isinstance(n_stresses, int) or n_stresses < 1:
                    raise ValueError(
                        "Cannot deserialise the {} life model without its "
                        "number of stress columns ('n_stresses', a positive "
                        "integer).".format(life_name)
                    )
                reg_model = reg_model.resolve(n_stresses)
            columns = list(reg_model.coefficient_columns())
            if columns:
                # Its column coefficients by the names saved, those of a
                # dict saved before v0.23 (beta_j) as named now (#614)
                saved = model_dict.get("phi_param_map", {})
                saved_names = sorted(saved, key=saved.__getitem__)
                others = [
                    *dist.parameter_names,
                    *(k for k in reg_model.phi_param_map if k not in columns),
                ]
                names = loaded_coefficient_names(
                    [*others, *saved_names[-len(columns) :]],
                    len(others),
                    len(columns),
                    model_dict.get("feature_names"),
                )[len(others) :]
                if len(saved_names) == len(reg_model.phi_param_map):
                    renamed.update(zip(saved_names[-len(columns) :], names))
                    reg_model = reg_model.named(names)
            fitter = AcceleratedLife(dist, reg_model)
            phi_param_map = dict(reg_model.phi_param_map)
        elif kind in _SERIALISABLE_KINDS:
            factory_name, phi_kind = _SERIALISABLE_KINDS[kind]
            factory = getattr(surpyval, factory_name)
            fitter = factory(dist)
            phi_param_map = {
                k: int(v) for k, v in model_dict["phi_param_map"].items()
            }
            # A dict saved before v0.23 named the coefficients beta_j: they
            # load with the names the model has now (#614).
            saved_names = sorted(phi_param_map, key=phi_param_map.__getitem__)
            names = loaded_coefficient_names(
                [*dist.parameter_names, *saved_names],
                len(dist.parameter_names),
                len(saved_names),
                model_dict.get("feature_names"),
            )[len(dist.parameter_names) :]
            renamed.update(zip(saved_names, names))
            phi_param_map = {
                renamed.get(k, k): v for k, v in phi_param_map.items()
            }
            if phi_kind == "exp":
                # The log-linear multiplier exp(beta'Z), matching the
                # fitters. Imported here because _fit_skeleton imports
                # this module at load time.
                from ._fit_skeleton import LogLinearPhi

                reg_model = LogLinearPhi(
                    model_dict["reg_model_name"], phi_param_map
                )
            else:
                # Additive: beta'Z is added to the hazard, no multiplier.
                reg_model = CovariateLink(
                    model_dict["reg_model_name"], phi_param_map
                )
        else:
            raise ValueError(
                "Cannot deserialise regression kind {!r}".format(kind)
            )

        out = cls()
        out.model = fitter
        out.distribution = dist
        out.dist = dist
        out.distribution_param_map = fitter.param_map
        out.phi_param_map = phi_param_map
        out.reg_model = reg_model
        out.kind = kind
        out.params = params
        out.dist_params = params[:k_dist]
        out.phi_params = params[k_dist:]
        out.k_dist = k_dist
        out.fixed = {
            renamed.get(k, k): float(v)
            for k, v in model_dict.get("fixed", {}).items()
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
        if kind != ACCELERATED_LIFE:
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
        out.maximum = restored_maximum(model_dict)
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

    def _is_accelerated_life(self) -> bool:
        """Whether this is an accelerated life model: a life model gives
        the distribution's life parameter, and the covariate parameters
        are the life model's, not one coefficient per column."""
        return self.kind == ACCELERATED_LIFE

    def _is_additive(self) -> bool:
        """Whether the covariates add ``beta'Z`` to the hazard (additive
        hazards), which nothing keeps positive, rather than act through
        a multiplier."""
        return self.kind == ADDITIVE_HAZARD

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
        PROPORTIONAL_HAZARD: "the hazard ratio",
        ACCELERATED_FAILURE_TIME: "the acceleration factor",
        PROPORTIONAL_ODDS: "the survival odds ratio",
    }

    @property
    def life_parameter(self) -> "str | None":
        """The distribution parameter an accelerated life model replaces by
        its life model (``None`` for the other families)."""
        if not self._is_accelerated_life():
            return None
        return getattr(self.model, "life_parameter", None)

    def _life_relation(self) -> str:
        # How the life parameter follows from the life model, e.g.
        # "L(Z) of the Power life model", for the printed model.
        relation = getattr(self.model, "life_relation", "L(Z)")
        return "{} of the {} life model".format(relation, self.reg_model.name)

    def _is_linear_predictor(self) -> bool:
        """Whether the covariate parameters are coefficients of a linear
        predictor ``beta'Z`` (one per column of ``Z``), which the
        coefficient table is for: a built-in link's, or a custom link's
        that names them as coefficients were named before v0.23
        (``beta_j``); an accelerated-life model's are the parameters of
        its life model."""
        if self._is_accelerated_life():
            return False
        if self.reg_model.name in _SERIALISABLE_REG_NAMES:
            return True
        n_phi = len(self.params) - self.k_dist
        pmap = dict(self.reg_model.phi_param_map or {})
        return pmap == {"beta_{}".format(i): i for i in range(n_phi)}

    def _coefficient_names(self) -> "list[str]":
        """The names of the coefficients of the columns of ``Z``, in
        column order (#614): the linear predictor's, or an accelerated
        life model's (``GeneralLogLinear``'s); none for another life
        model."""
        if self._is_linear_predictor():
            return list(self.parameter_names[self.k_dist :])
        columns = getattr(self.reg_model, "coefficient_columns", None)
        return [] if columns is None else list(columns())

    def _exp_meaning(self) -> "str | None":
        """What ``exp(coef)`` means, or ``None`` where the link is not
        log-linear (``exp(coef)`` is then not a ratio)."""
        from ._fit_skeleton import LogLinearPhi

        name = self.reg_model.name
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
        p-value. A coefficient is named by its covariate's column (a
        formula, ``fit_from_df`` or a DataFrame ``Z``), else ``coef_j``
        (:attr:`parameter_names`).

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

        from ._summary import coefficient_table

        params = np.asarray(self.params, dtype=float)
        se = self._summary_se()
        names = self.parameter_names
        k = self.k_dist
        level = "{:g}%".format(100 * (1 - alpha_ci))
        if self._is_linear_predictor():
            part = "coefficients"
            rows = coefficient_table(
                names[k:],
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
        if self._is_accelerated_life():
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
        if data is None or self.is_tvc:
            return None
        return data.x, data.c, data.n, data.Z

    def phi(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        Z = self._prepare_Z(Z)
        phi = self.reg_model.phi
        if phi is None:
            # Additive-hazards reg models have no multiplier: the
            # covariate effect enters as beta'Z added to the hazard, so
            # phi() is undefined rather than an AttributeError (#277).
            raise NotImplementedError(
                "phi() is not defined for additive-hazards models: the "
                "covariate effect is additive (beta'Z on the hazard), "
                "not a multiplier."
            )
        # Relative to the centre for a baseline kept there (#463).
        return phi(self._centred(Z), *self._eval_params()[self.k_dist :])

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
        if self._is_additive():
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

    @keeps_query_shape
    def qf(
        self,
        p: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        r"""
        The quantile function: the time by which a proportion ``p`` of the
        units with covariates ``Z`` have failed, ``ff(qf(p, Z), Z) = p``
        (the B10 life of a unit is ``qf(0.1, Z)``; #571).

        Parameters
        ----------

        p : array like or scalar
            The probabilities, in [0, 1].

        Z : array like or DataFrame
            The covariates, paired with ``p`` as :meth:`sf` pairs them with
            ``x``: one row per probability (or a single row for every
            probability, or a single probability for every row). A model
            fitted with ``fit_from_df`` also accepts a DataFrame.

        grid : bool, optional
            ``True`` gives every ``p`` for every row of ``Z``, with shape
            ``(len(Z),) + p.shape``. Default ``False``: rows and
            probabilities paired.

        Returns
        -------

        qf : scalar or numpy array
            The quantiles.

        Notes
        -----
        Every family is inverted the same way, from the model's own
        cumulative hazard: the time at which :math:`H(t \mid Z)` reaches
        :math:`-\log(1 - p)`, solved to a relative ``1e-12`` for every
        probability at once. ``qf(0, Z)`` is the start of the
        distribution's support and ``qf(1, Z)`` its end; a probability
        outside [0, 1] gives NaN with a warning, as for the univariate
        models, and NaN gives NaN. Where the hazard of an additive model
        turns negative the cumulative hazard is not monotone, and the
        quantile is one of its crossings (``hf`` warns there).

        Examples
        --------

        >>> import numpy as np
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
        >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullPH.fit(x, Z)
        >>> b10 = model.qf(0.1, [[0], [1]])
        >>> b10.round(4)
        array([2.6636, 1.6593])
        >>> model.ff(b10, [[0], [1]]).round(12)
        array([0.1, 0.1])
        """
        u = np.asarray(p, dtype=float)
        Z = self._prepare_Z(Z)
        rows = covariate_rows(Z, self._n_covariates())
        shape = None
        if grid:
            shape = (rows.shape[0], u.size)
            u = np.tile(u, shape[0])
            rows = np.repeat(rows, shape[1], axis=0)
        else:
            check_paired_rows(u.size, rows.shape[0])
            size = max(u.size, rows.shape[0])
            u = np.broadcast_to(u, (size,))
            rows = np.broadcast_to(rows, (size, rows.shape[1]))
        rows = np.asarray(self._centred(rows), dtype=float)
        outside = warn_outside_unit_interval(u)
        u = np.where(outside, np.nan, u)
        params = self._eval_params()
        dist_params = params[: self.k_dist]
        with np.errstate(all="ignore"):
            # The baseline's quantile, a start for each search.
            start = np.asarray(
                self.distribution.qf(np.clip(u, 0.0, 1.0), *dist_params),
                dtype=float,
            ) * np.ones(u.shape)
        out = quantiles_by_inversion(
            lambda t, k: self.model.Hf(t, rows[k], *params),
            u,
            self.distribution.support,
            start,
        )
        if self._is_additive():
            finite = np.isfinite(out)
            self._warn_if_hazard_negative(
                np.where(finite, out, 0.0), rows, finite, stacklevel=5
            )
        if shape is not None:
            out = out.reshape(shape)
        return out

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
            n_rows=self._ic_n_total,
        )

    # ``self.k`` is the number of estimated parameters, so the AIC/BIC
    # penalties and the AIC_c correction all use it (the mixin's defaults).
    # Fixed parameters -- and the accelerated-life placeholder for the life
    # parameter -- used to be counted as well.

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
        ['alpha', 'beta', 'coef_0']
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

        Z_mean = np.asarray(self.data.Z).mean(axis=0)
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
