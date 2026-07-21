import json
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.stats import norm, uniform

from surpyval.serialisation import stamp_schema
from surpyval.utils import fsli_to_xcnt

from ._bounds import (
    bound_signs,
    delta_method_se,
    log_transformed_cb,
    logit_sf_bound,
    numerical_hessian,
)
from .regression_data import prepare_Z

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


class ParametricRegressionModel:
    """
    Result of ``.fit()`` or ``.from_params()`` method for parametric
    regression modelling.

    Instances of this class are very useful when a user needs the other
    functions of a distribution for plotting, optimizations, monte carlo
    analysis and numeric integration.

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
        if self.feature_names is not None:
            out["feature_names"] = list(self.feature_names)
        if self.formula is not None:
            out["formula"] = self.formula

        # Store the parameter covariance so the restored model can produce
        # confidence bounds without the original data. Only for data-fit
        # models whose covariance is finite (a boundary optimum gives nan).
        if hasattr(self, "data") and getattr(self, "res", None) is not None:
            try:
                cov = self.covariance()
            except Exception:
                cov = None
            if cov is not None and np.all(np.isfinite(cov)):
                out["covariance"] = np.asarray(cov, dtype=float).tolist()
            if hasattr(self, "_neg_ll"):
                out["_neg_ll"] = float(self._neg_ll)
        return stamp_schema(out)

    def to_json(self, fp: "str | Path") -> None:
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

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
                # the log-linear multiplier exp(beta'Z), matching the fitters
                reg_model.phi = lambda Z, *p: np.exp(np.dot(Z, np.array(p)))
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
        out.k = int(model_dict["k"])
        out.k_dist = k_dist
        out.fixed = {
            k: float(v) for k, v in model_dict.get("fixed", {}).items()
        }
        out.gamma = float(model_dict.get("gamma", 0.0))
        out.p = float(model_dict.get("p", 1.0))
        out.f0 = float(model_dict.get("f0", 0.0))
        out.feature_names = model_dict.get("feature_names")
        out.formula = model_dict.get("formula")

        if "covariance" in model_dict:
            out._restored_covariance = np.array(
                model_dict["covariance"], dtype=float
            )
        if "_neg_ll" in model_dict:
            out._neg_ll = float(model_dict["_neg_ll"])
        return out

    @classmethod
    def from_json(cls, fp: "str | Path") -> "ParametricRegressionModel":
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def _prepare_Z(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        Convert ``Z`` to a numeric design matrix.

        If a pandas DataFrame is passed and the model was fit from a DataFrame,
        the covariate columns (or formula) recorded at fit time are used to
        select and encode the correct columns. Otherwise ``Z`` is returned
        unchanged.
        """
        return prepare_Z(Z, self.feature_names, self._model_spec)

    def __repr__(self) -> str:
        dist_params = self.params[0 : self.k_dist]
        reg_model_params = self.params[self.k_dist :]
        dist_param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(dist_params, self.distribution.param_names)
            ]
        )

        reg_model_param_string = "\n".join(
            [
                "{:>10}".format(name) + ": " + str(p)
                for p, name in zip(
                    reg_model_params, self.reg_model.phi_param_map
                )
                if name not in self.fixed
            ]
        )

        if hasattr(self, "params"):
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

            out = (
                out
                + "\nDistribution        :\n"
                + "{params}".format(params=dist_param_string)
            )

            out = (
                out
                + "\nRegression Model    :\n"
                + "{params}".format(params=reg_model_param_string)
            )

            return out
        else:
            return "Unable to fit values"

    def phi(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        Z = self._prepare_Z(Z)
        return self.reg_model.phi(Z, *self.phi_params)

    def sf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        r"""
        Surival (or Reliability) function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the survival function
            will be calculated

        Returns
        -------

        sf : scalar or numpy array
            The scalar value of the survival function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the survival function at each
            corresponding value in the input array.


        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.sf(2)
        0.9920319148370607
        >>> model.sf([1, 2, 3, 4, 5])
        array([0.9990005 , 0.99203191, 0.97336124, 0.938005  , 0.8824969 ])
        """
        if isinstance(x, list):
            x = np.array(x)
        Z = self._prepare_Z(Z)
        return self.model.sf(x, Z, *self.params)

    # Families whose cumulative hazard is additive over disjoint time
    # intervals, so a step-valued covariate path factorises exactly into a sum
    # over its segments (the same structure that lets ``fit_tvc`` reshape TVC
    # data). Accelerated failure time (covariate rescales time -> needs an
    # accumulated accelerated age) and proportional odds (no additive form) do
    # not compose this way and are refused below.
    _TVC_EVALUABLE_KINDS = ("Proportional Hazard", "Additive Hazard")

    def _tvc_segments(self, schedule, t_max):
        """
        Materialise ``schedule`` to ``t_max`` and return ``(starts, ends, Z)``
        with the first segment held back to the time origin, so the cumulative
        hazard is measured from ``0`` (unconditional survival).
        """
        starts, ends, Z = schedule.segments(t_max)
        if starts[0] > 0:
            # Hold the first covariate back to the origin (matches the Cox
            # ``predict_tvc`` clamp): survival is measured from t = 0.
            starts = starts.copy()
            starts[0] = 0.0
        return starts, ends, Z

    def _to_schedule(self, Z, xl):
        """
        Coerce the ``sf_tvc`` covariate argument into a
        :class:`~...tvc_schedule.StepSchedule`.

        ``Z`` is either a ready-made schedule, or an array of per-segment
        covariate rows whose segment start times are given in ``xl``.
        """
        from .tvc_schedule import StepSchedule

        if isinstance(Z, StepSchedule):
            if xl is not None:
                raise ValueError(
                    "xl must not be given when Z is already a StepSchedule"
                )
            schedule = Z
        else:
            if xl is None:
                raise ValueError(
                    "for the array form pass xl (the segment start times) "
                    "alongside Z (one covariate row per segment); or pass a "
                    "StepSchedule"
                )
            schedule = StepSchedule.from_changepoints(xl, Z)

        n_cov = self.params.shape[0] - self.k_dist
        if schedule.p != n_cov:
            raise ValueError(
                "the schedule has {} covariate(s) but the model was fit with "
                "{}".format(schedule.p, n_cov)
            )
        return schedule

    def Hf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
    ) -> npt.NDArray:
        r"""
        Cumulative hazard for a covariate following a step schedule ``Z(t)``.

        For the proportional- and additive-hazards families the cumulative
        hazard is additive over disjoint intervals, so along a piecewise
        constant path it is exactly the sum of the per-segment increments

        .. math::
            H\bigl(x \mid Z(\cdot)\bigr)
            = \sum_{\text{seg } (a, b]} \bigl[\,H(b, z) - H(a, z)\,\bigr],

        each increment evaluated with the model's own ``Hf`` at the covariate
        ``z`` active on that segment (the last segment clipped at ``x``). With
        a single constant segment this reduces exactly to ``Hf(x, Z)``.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate the cumulative hazard.
        Z : StepSchedule or array_like
            The covariate path -- either a
            :class:`~...tvc_schedule.StepSchedule`, or an array of per-segment
            covariate rows (with ``xl`` giving the segment start times).
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.

        Returns
        -------
        ndarray
            The cumulative hazard at each ``x``.
        """
        if self.kind not in self._TVC_EVALUABLE_KINDS:
            raise NotImplementedError(
                "time-varying-covariate evaluation is only defined for the "
                "proportional-hazards and additive-hazards families (this "
                "model is '{}'). Accelerated failure time needs an "
                "accumulated accelerated age and proportional odds has no "
                "additive hazard "
                "form; neither factorises over covariate segments.".format(
                    self.kind
                )
            )
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        schedule = self._to_schedule(Z, xl)
        t_max = float(np.max(xq))
        if t_max <= 0:
            raise ValueError("x must contain a positive time")
        starts, ends, Zseg = self._tvc_segments(schedule, t_max)

        H = np.zeros(xq.shape[0], dtype=float)
        for a, b, z in zip(starts, ends, Zseg):
            zrow = np.asarray(z, dtype=float).reshape(1, -1)
            upper = np.clip(xq, a, b)
            hi = np.asarray(
                self.model.Hf(upper, zrow, *self.params), dtype=float
            ).ravel()
            lo = np.asarray(
                self.model.Hf(np.array([a]), zrow, *self.params), dtype=float
            ).ravel()
            H = H + (hi - lo)
        return H

    def sf_tvc(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | Any",
        xl: "npt.ArrayLike | None" = None,
        given: "float | None" = None,
    ) -> npt.NDArray:
        r"""
        Survival for a covariate that follows a step (piecewise-constant)
        schedule ``Z(t)``.

        With a time-varying covariate the survival depends on the whole
        covariate path, not one fixed vector. For the proportional- and
        additive-hazards families this is exact along a step path:
        ``S(x) = exp(-H(x))`` where ``H`` is the segment sum in
        :meth:`Hf_tvc`. Only these families compose this way -- accelerated
        failure time and proportional odds raise ``NotImplementedError``.

        Parameters
        ----------
        x : array_like
            Times at which to evaluate survival.
        Z : StepSchedule or array_like
            The covariate path. Either a
            :class:`~...tvc_schedule.StepSchedule` (built from change-points,
            intervals, a cyclic pattern, or a step-valued expression) or an
            array of per-segment covariate rows with ``xl`` giving the segment
            start times.
        xl : array_like, optional
            Segment start times, required only when ``Z`` is an array.
        given : float, optional
            If supplied, return the *conditional* survival given the item has
            survived to age ``given``:
            ``S(x | given) = exp(-(H(x) - H(given)))``.

        Returns
        -------
        ndarray
            Survival at each ``x`` (conditional on ``given`` when supplied).

        Examples
        --------
        >>> from surpyval import WeibullPH
        >>> from surpyval.univariate.regression import StepSchedule
        >>> # ... model = WeibullPH.fit(...)
        >>> sched = StepSchedule.from_changepoints([0, 500], [[0.0], [1.0]])
        >>> model.sf_tvc([250, 750], sched)          # doctest: +SKIP
        """
        H = self.Hf_tvc(x, Z, xl)
        if given is not None:
            given = float(given)
            if given > 0:
                H = H - self.Hf_tvc(given, Z, xl)[0]
        return np.exp(-H)

    def ff(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        r"""
        The cumulative distribution function, or failure function, for a
        distribution using the parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the failure function
            (CDF) will be calculated

        Returns
        -------

        ff : scalar or numpy array
            The scalar value of the CDF of the distribution if a scalar was
            passed. If an array like object was passed then a numpy array is
            returned with the value of the CDF at each corresponding value in
            the input array.


        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.ff(2)
        0.007968085162939342
        >>> model.ff([1, 2, 3, 4, 5])
        array([0.0009995 , 0.00796809, 0.02663876, 0.061995  , 0.1175031 ])
        """
        if isinstance(x, list):
            x = np.array(x)
        Z = self._prepare_Z(Z)
        return self.model.ff(x, Z, *self.params)

    def df(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        r"""
        The density function for a distribution using the parameters found in
        the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the density function
            will be calculated

        Returns
        -------

        df : scalar or numpy array
            The scalar value of the density function of the distribution if a
            scalar was passed. If an array like object was passed then a numpy
            array is returned with the value of the density function at each
            corresponding value in the input array.


        Examples
        --------

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.df(2)
        0.01190438297804473
        >>> model.df([1, 2, 3, 4, 5])
        array([0.002997  , 0.01190438, 0.02628075, 0.04502424, 0.06618727])
        """
        if isinstance(x, list):
            x = np.array(x)
        Z = self._prepare_Z(Z)
        return self.model.df(x, Z, *self.params)

    def hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        r"""
        The instantaneous hazard function for a distribution using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the instantaneous
            hazard function will be calculated

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

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.hf(2)
        0.012000000000000002
        >>> model.hf([1, 2, 3, 4, 5])
        array([0.003, 0.012, 0.027, 0.048, 0.075])
        """
        if isinstance(x, list):
            x = np.array(x)
        Z = self._prepare_Z(Z)
        return self.model.hf(x, Z, *self.params)

    def Hf(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        r"""

        The cumulative hazard function for a distribution using the parameters
        found in the ``.params`` attribute.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the cumulative hazard
            function will be calculated

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

        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> model.Hf(2)
        0.008000000000000002
        >>> model.Hf([1, 2, 3, 4, 5])
        array([0.001, 0.008, 0.027, 0.064, 0.125])
        """
        if isinstance(x, list):
            x = np.array(x)
        Z = self._prepare_Z(Z)
        return self.model.Hf(x, Z, *self.params)

    def random(
        self, size: int, Z: "npt.ArrayLike | pd.DataFrame"
    ) -> npt.NDArray:
        r"""

        A method to draw random samples from the distributions using the
        parameters found in the ``.params`` attribute.

        Parameters
        ----------
        size : int
            The number of random samples to be drawn from the distribution.

        Z : scalar or array like
            The value(s) of the stresses at which the random

        Returns
        -------
        random : numpy array
            Returns a numpy array of size ``size`` with random values drawn
            from the distribution.


        Examples
        --------
        >>> from surpyval import Weibull
        >>> model = Weibull.from_params([10, 3])
        >>> np.random.seed(1)
        >>> model.random(1)
        array([8.14127103])
        >>> model.random(10)
        array([10.84103403,  0.48542084,  7.11387062,  5.41420125,  4.59286657,
                5.90703589,  7.5124326 ,  7.96575225,  9.18134126,
                8.16000438])
        """
        if (self.p == 1) and (self.f0 == 0):
            return (
                self.dist.qf(uniform.rvs(size=size), *self.params) + self.gamma
            )
        elif (self.p != 1) and (self.f0 == 0):
            n_obs = np.random.binomial(size, self.p)

            f = (
                self.dist.qf(uniform.rvs(size=n_obs), *self.params)
                + self.gamma
            )
            s = np.ones(np.array(size) - n_obs) * np.max(f) + 1

            return fsli_to_xcnt(f, s)

        elif (self.p == 1) and (self.f0 != 0):
            n_doa = np.random.binomial(size, self.f0)

            x0 = np.zeros(n_doa) + self.gamma
            x = (
                self.dist.qf(uniform.rvs(size=size - n_doa), *self.params)
                + self.gamma
            )
            x = np.concatenate([x, x0])
            np.random.shuffle(x)

            return x
        else:
            N = np.random.multinomial(
                1, [self.f0, self.p - self.f0, 1.0 - self.p], size
            ).sum(axis=0)
            N = np.atleast_2d(N)
            n_doa, n_obs, n_cens = N[:, 0], N[:, 1], N[:, 2]
            x0 = np.zeros(n_doa) + self.gamma
            x = (
                self.dist.qf(uniform.rvs(size=n_obs), *self.params)
                + self.gamma
            )
            f = np.concatenate([x, x0])
            s = np.ones(n_cens) * np.max(f) + 1
            # raise NotImplementedError("Combo zero-inflated and lfp model not
            # yet supported")
            return fsli_to_xcnt(f, s)

    def neg_ll(self) -> float:
        r"""

        The the negative log-likelihood for the model, if it was fit with the
        ``fit()`` method. Not available if fit with the ``from_params()``
        method.

        Parameters
        ----------

        None

        Returns
        -------

        neg_ll : float
            The negative log-likelihood of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.neg_ll()
        262.52685642385734
        """
        if not hasattr(self, "data"):
            raise ValueError("Must have been fit with data")

        return self._neg_ll

    def bic(self) -> float:
        r"""

        The the Bayesian Information Criterion (BIC) for the model, if it was
        fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        bic : float
            The BIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.bic()
        534.2640532196908

        References:
        -----------

        `Bayesian Information Criterion for Censored Survival Models
        <https://www.jstor.org/stable/2677130>`_.

        """
        if hasattr(self, "_bic"):
            return self._bic
        else:
            self._bic = (
                self.k * np.log(self.data.n[self.data.c == 0].sum())
                + 2 * self.neg_ll()
            )
            return self._bic

    def aic(self) -> float:
        r"""

        The the Aikake Information Criterion (AIC) for the model, if it was
        fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        aic : float
            The AIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.aic()
        529.0537128477147
        """
        if hasattr(self, "_aic"):
            return self._aic
        else:
            self._aic = 2 * self.k + 2 * self.neg_ll()
            return self._aic

    def aic_c(self) -> float:
        r"""

        The the Corrected Aikake Information Criterion (AIC) for the model, if
        it was fit with the ``fit()`` method. Not available if fit with the
        ``from_params()`` method.

        Parameters
        ----------

        None

        Returns
        -------

        aic_c : float
            The Corrected AIC of the model

        Examples
        --------

        >>> from surpyval import Weibull
        >>> import numpy as np
        >>> np.random.seed(1)
        >>> x = Weibull.random(100, 10, 3)
        >>> model = Weibull.fit(x)
        >>> model.aic()
        529.1774241879209
        """
        if hasattr(self, "_aic_c"):
            return self._aic_c
        else:
            k = len(self.params)
            n = self.data.n.sum()
            self._aic_c = self.aic() + (2 * k**2 + 2 * k) / (n - k - 1)
            return self._aic_c

    # -- confidence bounds -------------------------------------------------

    def _check_inference(self) -> None:
        # A model deserialised with a stored covariance can produce bounds
        # without the original data.
        if getattr(self, "_restored_covariance", None) is not None:
            return
        if not hasattr(self, "data") or getattr(self, "res", None) is None:
            raise ValueError(
                "Confidence bounds are only available for models fit from "
                "data; from_params models carry no likelihood."
            )

    def parameter_names(self) -> list[str]:
        """
        Names of the fitted parameters in ``.params`` order: the distribution's
        parameters followed by the covariate coefficients.
        """
        dist_names = list(self.distribution.param_names)
        phi_map = self.reg_model.phi_param_map
        phi_names = [
            k for k, _ in sorted(phi_map.items(), key=lambda kv: kv[1])
        ]
        return dist_names + phi_names

    def covariance(self) -> npt.NDArray:
        """
        Approximate covariance matrix of the fitted parameters, ordered to
        match :meth:`parameter_names`. Computed as the inverse of the numerical
        Hessian of the negative log-likelihood at the MLE (the observed
        information). Fixed parameters get a zero row/column.

        A parameter driven to a boundary breaks the Wald approximation; the
        covariance is then returned filled with ``nan`` (with a warning).
        """
        restored = getattr(self, "_restored_covariance", None)
        if restored is not None:
            return restored
        self._check_inference()
        names = self.parameter_names()
        p_hat = np.asarray(self.params, dtype=float)
        free = [i for i, nm in enumerate(names) if nm not in self.fixed]
        n = len(names)
        cov = np.zeros((n, n))
        if not free:
            return cov

        def neg_ll_free(free_vals: npt.NDArray) -> float:
            full = p_hat.copy()
            full[free] = free_vals
            return self.model.neg_ll(self.data, *full)

        H = numerical_hessian(neg_ll_free, p_hat[free])
        bad = not np.all(np.isfinite(H))
        if not bad:
            try:
                cov_free = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                bad = True
        if bad:
            import warnings

            warnings.warn(
                "The information matrix could not be inverted (the optimum "
                "may be at a parameter boundary); covariance is unavailable."
            )
            return np.full((n, n), np.nan)
        cov[np.ix_(free, free)] = cov_free
        return cov

    def standard_errors(self) -> npt.NDArray:
        """
        Standard errors of the fitted parameters (square roots of the diagonal
        of :meth:`covariance`), ordered to match :meth:`parameter_names`.
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
            The parameter to bound; one of :meth:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.
        """
        self._check_inference()
        names = self.parameter_names()
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
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

        alpha, signs = bound_signs(alpha_ci, bound)
        offsets = signs * norm.ppf(1.0 - alpha) * np.sqrt(var)

        if lower is not None and upper is not None:
            width = upper - lower
            frac = (p_hat - lower) / width
            u_hat = np.log(frac / (1.0 - frac))
            du = offsets / (width * frac * (1.0 - frac))
            return lower + width / (1.0 + np.exp(-(u_hat + du)))
        elif lower is not None:
            return lower + (p_hat - lower) * np.exp(offsets / (p_hat - lower))
        elif upper is not None:
            return upper - (upper - p_hat) * np.exp(-offsets / (upper - p_hat))
        return p_hat + offsets

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
        derived from a survival-function bound taken on the logit scale (so it
        stays in ``(0, 1)``); ``hf``/``df`` use a log-scale bound (so they stay
        positive).

        Parameters
        ----------
        x : array like or scalar
            Times at which to evaluate the bound(s).
        Z : array like
            A single covariate vector.
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
        Zp = self._prepare_Z(Z)
        params = np.asarray(self.params, dtype=float)
        cov = self.covariance()

        if on in ("hf", "df"):
            fn = self.model.hf if on == "hf" else self.model.df
            est = np.asarray(fn(x, Zp, *params), dtype=float)
            se = delta_method_se(lambda p: fn(x, Zp, *p), params, cov)
            return log_transformed_cb(est, se, alpha_ci, bound)

        # sf, ff and Hf all derive from a survival-function bound.
        sf_hat = np.asarray(self.model.sf(x, Zp, *params), dtype=float)
        se = delta_method_se(lambda p: self.model.sf(x, Zp, *p), params, cov)

        if bound == "two-sided":
            sf_lo = logit_sf_bound(sf_hat, se, -1.0, alpha_ci / 2.0)
            sf_hi = logit_sf_bound(sf_hat, se, +1.0, alpha_ci / 2.0)
            if on in ("sf", "R"):
                return np.stack([sf_lo, sf_hi], axis=-1)
            elif on in ("ff", "F"):
                return np.stack([1.0 - sf_hi, 1.0 - sf_lo], axis=-1)
            else:  # Hf: -log(sf) is decreasing in sf
                return np.stack([-np.log(sf_hi), -np.log(sf_lo)], axis=-1)

        # One-sided. ff and Hf decrease in sf, so their bound uses the
        # opposite survival-function tail.
        if on in ("sf", "R"):
            sign = -1.0 if bound == "lower" else 1.0
            return logit_sf_bound(sf_hat, se, sign, alpha_ci)
        sign = 1.0 if bound == "lower" else -1.0
        sf_b = logit_sf_bound(sf_hat, se, sign, alpha_ci)
        if on in ("ff", "F"):
            return 1.0 - sf_b
        return -np.log(sf_b)

    def plot(
        self,
        ax: "Axes | None" = None,
        plot_bounds: bool = True,
        alpha_ci: float = 0.05,
    ) -> "Axes":
        r"""

        A method to plot the survival function of the distribution at the mean
        covariate vector against the empirical (Kaplan-Meier) survival of the
        fitted data, with a delta-method confidence band.

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

        if ax is None:
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
