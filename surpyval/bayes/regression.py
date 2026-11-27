"""Bayesian fits of the parametric regressions: priors on the coefficients
and on the baseline's parameters.

``WeibullPH.fit_bayes(x, Z, c, priors=...)``, ``AFT(dist).fit_bayes`` and
``AcceleratedLife(dist, life_model).fit_bayes`` come here. The posterior
is the fitter's own likelihood (every censoring and truncation ``fit``
takes) times the priors, approximated by the Laplace engine
(:mod:`surpyval.bayes.engines.laplace`): a normal at the posterior's peak
on the parameters' unconstrained scale, checked against the exact
posterior by importance sampling. For the exact posterior, sample it with
PyMC through :mod:`surpyval.pymc`, whose regression likelihoods are this
fitter's.

How it is computed
------------------
- The maximum-likelihood fit is made first, as a user would call it. It
  gives the data as the fitter reads them, the parameters' names and
  ranges, the comparison in the summary, and the search's start.
- The posterior's peak is found on the parameters' unconstrained scale
  (the log of a positive parameter), and its curvature there gives the
  normal's covariance, correlations included (a baseline's scale and a
  coefficient are strongly correlated whenever a covariate's mean is far
  from 0, a temperature in kelvin), so the covariates are used as given;
  nothing is standardised in the model.
- A coefficient of ``exp(beta'Z)`` with no prior gets
  ``Normal(0, 2.5 / sd)``, ``sd`` its covariate's standard deviation in
  the data (:data:`COEFFICIENT_SCALE`): weakly informative, and the same
  whatever the covariate's unit. Every other parameter with no prior is
  flat on its unconstrained scale, as for ``Weibull.fit_bayes``. The
  defaults used are listed on the result (``default_priors``) and in its
  summary.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.engines.grid import weighted_quantile
from surpyval.bayes.posterior import Posterior
from surpyval.bayes.prior import (
    BoundPriors,
    NormalOnScale,
    PowerPrior,
    PriorSpec,
)
from surpyval.bayes.result import (
    BayesianResult,
    fit_target,
    interval_levels,
)
from surpyval.bayes.target import Target, register_adapter, register_target
from surpyval.serialisation import require_model_tag, stamp_schema
from surpyval.utils.shapes import covariate_rows, keeps_query_shape
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.warnings import ignoring_warnings

#: The default prior on a coefficient of ``exp(beta'Z)`` is
#: ``Normal(0, COEFFICIENT_SCALE / sd)``, ``sd`` the covariate's standard
#: deviation: a change of 2.5 in the log hazard (or log life) for one
#: standard deviation of the covariate is one prior standard deviation.
COEFFICIENT_SCALE = 2.5
#: The families ``fit_bayes`` covers, by the fitted model's ``kind``.
COVERED = (
    "Proportional Hazard",
    "Accelerated Failure Time",
    "Accelerated Life",
)
#: The Laplace engine's draws for a regression: each predictive value is
#: a sum over them, through the fitted model's own functions.
DRAWS = 1024
_ON = ("sf", "ff", "Hf", "hf", "df")

_NOT_HERE = {
    "w": "weights (the posterior of a weighted pseudo-likelihood is not a "
    "posterior; use fit(..., w=) for the weighted estimate, or counts n "
    "for identical units)",
    "penalizer": "a penalty (a prior on the coefficients is the Bayesian "
    "penalty: priors={'coef_0': surpyval.Normal.from_params([0, s])}; "
    "fit_penalised gives the penalised maximum-likelihood fit)",
    "l1_ratio": "a penalty (use fit_penalised, or a prior on the "
    "coefficients)",
    "init": "a starting point (the search for the posterior's peak "
    "starts from the maximum-likelihood fit)",
    "center": "centring (the normal's covariance holds the correlations "
    "between the parameters, so the covariates are used as given)",
}


def refuse_options(what: str, options: dict) -> None:
    """Refuse the arguments of ``fit`` that a Bayesian regression does
    not take, each with what to do instead."""
    for k in options:
        if k in _NOT_HERE:
            raise ValueError(f"{what} does not take {k}: {_NOT_HERE[k]}.")
    if options:
        raise ValueError(
            f"{what} does not take {', '.join(sorted(options))}. It takes "
            "the data (x, Z, c, n, t, tl, tr), priors, fixed, method and "
            "importance."
        )


def refuse_time_varying(Z: Any) -> None:
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if isinstance(Z, (CovariatePath, StepSchedule, dict)):
        raise ValueError(
            "fit_bayes does not take time-varying covariates in this "
            "version: it covers covariates fixed for each unit, with the "
            "proportional hazards, accelerated failure time and accelerated "
            "life fitters. Use fit_tvc for the maximum-likelihood fit."
        )


def _is_mixture_cure(mle: Any) -> bool:
    """Whether ``mle`` is a mixture-cure regression's fit."""
    from surpyval.univariate.regression.distributional.mixture_cure import (
        MixtureCureModel,
    )

    return isinstance(mle, MixtureCureModel)


def _coefficient_columns(mle: Any) -> dict[str, int]:
    """``{name: its column of Z}`` for the parameters that are
    coefficients of a covariate: a proportional hazards or accelerated
    failure time model's with the log-linear link, and every coefficient
    of a mixture-cure regression (on the incidence's link, or on a
    latency parameter's)."""
    if _is_mixture_cure(mle):
        return dict(mle._coefficient_columns())
    return {name: k for k, name in enumerate(_coefficient_names(mle))}


def _coefficient_names(mle: Any) -> list[str]:
    """The parameters that are coefficients of ``exp(beta'Z)``: those of a
    proportional hazards or accelerated failure time model with the
    log-linear link. (An accelerated life model's are its life model's
    parameters, and a custom ``phi``'s need not be one per covariate.)"""
    from surpyval.univariate.regression._fit_skeleton import LogLinearPhi

    if mle.kind == "Accelerated Life":
        return []
    phi = getattr(mle.model, "phi", LogLinearPhi.phi)
    if phi is not LogLinearPhi.phi:
        return []
    names = list(mle.parameter_names)[mle.k_dist :]
    return names if len(names) == np.shape(mle.data.Z)[1] else []


def _covariate_sds(data: Any) -> npt.NDArray:
    """Each covariate's standard deviation in the data, the rows weighted
    by their counts."""
    Z = np.asarray(data.Z, dtype=float)
    n = np.asarray(data.n, dtype=float)
    mean = np.sum(n[:, None] * Z, axis=0) / n.sum()
    out: npt.NDArray = np.sqrt(
        np.sum(n[:, None] * (Z - mean) ** 2, axis=0) / n.sum()
    )
    return out


def _default_priors(mle: Any, spec: PriorSpec, fixed: dict) -> dict:
    """``Normal(0, 2.5 / sd)`` for each coefficient with no prior."""
    covered = set(spec.single) | set(fixed)
    for j in spec.joint:
        if j.is_likelihood:
            continue
        if j.names is None:
            # a joint prior on every free parameter (on_quantities with a
            # statement for each)
            return {}
        covered |= set(j.names)
    sds = _covariate_sds(mle.data)
    out = {}
    for name, k in _coefficient_columns(mle).items():
        if name in covered:
            continue
        sd = float(sds[k])
        if not sd > 0:
            raise ValueError(
                f"The covariate of {name} is the same for every row, so "
                "the data say nothing about its coefficient and it has no "
                "scale for a default prior. Leave the column out, or give "
                f"the coefficient a prior (priors={{{name!r}: ...}})."
            )
        scale = COEFFICIENT_SCALE / sd
        out[name] = NormalOnScale(
            loc=0.0,
            scale=scale,
            kind="identity",
            said=f"default: Normal(0, {scale:.4g}), which is "
            f"{COEFFICIENT_SCALE:g} over the covariate's standard "
            f"deviation ({sd:.4g})",
        )
    return out


def _mle(fitter: Any, how: Any) -> Any:
    """The maximum-likelihood fit, quietly but for the rows it drops (a
    missing covariate value), which this fit drops too and says so, and
    (a mixture-cure regression) for its finding that the data show no
    cured fraction, said again here; its refusal is this fit's, with the
    reason."""
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.warnings import caller_stacklevel, collecting_warnings

    try:
        with collecting_warnings() as caught:
            with quiet_maximum_warnings(), np.errstate(all="ignore"):
                model = how()
    except ValueError as e:
        raise ValueError(
            "fit_bayes starts from the maximum-likelihood fit, which "
            f"refuses these data: {e}"
        ) from None
    for w in caught:
        if str(w.message).startswith("Dropped "):
            warnings.warn(
                str(w.message), w.category, stacklevel=caller_stacklevel()
            )
    if _is_mixture_cure(model) and model.maximum == "no finite maximum":
        _warn_no_cure(model)
    return model


def _warn_no_cure(mle: Any) -> None:
    """Say, as the maximum-likelihood fit says it, that a mixture-cure
    fit's likelihood has no finite maximum, so that the data do not
    identify the cure fraction."""
    from surpyval.utils.no_maximum import warn_no_maximum

    p = float(mle.params[list(mle.parameter_names).index("lfp_p")])
    warn_no_maximum(
        f"the maximum-likelihood fit's lfp_p ({p:.4g}), the probability "
        "of ever failing at the reference row, runs on towards 1 (or a "
        "coefficient without bound): the data show no sign of units that "
        "never fail there (no plateau at the end of follow-up), so they "
        "do not identify the cure fraction",
        "The posterior of lfp_p and of the cure fraction is then its "
        "prior's, shaped by the latency",
        "give lfp_p a prior from what is known about the cure fraction, "
        "fit the latency alone (an AFT model or Distributional(dist)), or "
        "follow the units for longer",
    )


def fit(
    fitter: Any,
    x: Any,
    Z: Any,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    *,
    priors: Any = None,
    fixed: dict[str, float] | None = None,
    tl: Any = None,
    tr: Any = None,
    importance: bool = False,
    method: str = "auto",
    **options: Any,
) -> BayesianRegression:
    """The Bayesian fit of a parametric regression; see the fitters'
    ``fit_bayes``."""
    refuse_options("fit_bayes", options)
    refuse_time_varying(Z)
    mle = _mle(
        fitter,
        lambda: fitter.fit(x, Z, c, n, t, fixed=fixed, tl=tl, tr=tr),
    )
    return from_mle(mle, priors, importance, method)


def fit_from_df(
    fitter: Any,
    df: Any,
    *,
    priors: Any = None,
    importance: bool = False,
    method: str = "auto",
    **columns: Any,
) -> BayesianRegression:
    """``fit_bayes`` on the columns of a DataFrame, named as
    ``fit_from_df`` names them."""
    from surpyval.utils.dataframe import as_pandas as _as_pandas

    df = _as_pandas(df)
    for k in ("w_col", "penalizer", "l1_ratio", "init", "center"):
        if columns.get(k) is not None:
            refuse_options("fit_bayes_from_df", {k.replace("_col", ""): 1})
        columns.pop(k, None)
    mle = _mle(fitter, lambda: fitter.fit_from_df(df, **columns))
    return from_mle(mle, priors, importance, method)


def _check_covered(mle: Any) -> None:
    """Refuse the families ``fit_bayes`` does not cover, and a design
    the data do not determine."""
    if mle.kind not in COVERED and not _is_mixture_cure(mle):
        raise ValueError(
            f"fit_bayes covers the proportional hazards, accelerated "
            f"failure time, accelerated life and mixture-cure fitters; "
            f"this is a "
            f"{mle.kind} model. Use fit for its maximum-likelihood fit."
        )
    aliased = np.atleast_1d(getattr(mle, "aliased", ())).astype(int)
    if aliased.size:
        names = list(mle.parameter_names)
        which = ", ".join(names[mle.k_dist + j] for j in aliased.tolist())
        raise ValueError(
            f"The data do not determine {which} (the coefficient of a "
            "column that is constant or a combination of the others). "
            "Leave the column out."
        )


@register_target
class RegressionTarget(Target):
    """The parametric regressions' side of a Bayesian fit
    (:mod:`surpyval.bayes.target`): the fitter's own likelihood, the
    maximum-likelihood fit's names, ranges and start, and the default
    prior on a coefficient.

    ``method="auto"`` is the Laplace engine for a regression, whatever
    the number of parameters: its likelihood is evaluated one parameter
    point at a time, which the grid's tens of thousands of points make
    slow. The grid is there on request (``method="grid"``, up to three
    free parameters).
    """

    tag = "regression"
    unit_name = "row"
    default_method = "laplace"
    laplace_draws = DRAWS
    laplace_only = True

    def __init__(self, mle: Any, data: SurpyvalData | None = None) -> None:
        self.mle = mle
        self.fitter = mle.model
        self.dist = mle.dist
        self.kind = mle.kind
        # (a loaded maximum-likelihood fit holds no data: it comes apart)
        self.data = mle.data if data is None else data
        self.names = list(mle.parameter_names)
        self.bounds = list(mle.parameter_bounds)
        self.fixed = {k: float(v) for k, v in dict(mle.fixed).items()}
        self.family = f"{self.dist.name} {self.kind}"

    def neg_ll(self, theta: npt.NDArray) -> float:
        return float(self.fitter.neg_ll(self.data, *theta))

    def coordinate_scales(self, bound: Any) -> npt.NDArray | None:
        """A coefficient of ``exp(beta'Z)`` is searched in units of one
        over its covariate's standard deviation (as if the covariate
        were standardised), so that the search is the same whatever the
        covariate's unit; the other parameters (on log or identity
        scales) in units of 1. The prior is on the coefficient as given,
        per unit of the covariate: only the search's steps change."""
        columns = _coefficient_columns(self.mle)
        if not columns:
            return None
        sds = _covariate_sds(self.data)
        out = np.ones(bound.d)
        for j, name in enumerate(bound.free_names):
            if name in columns:
                sd = float(sds[columns[name]])
                if np.isfinite(sd) and sd > 0:
                    out[j] = 1.0 / sd
        return out

    def with_params(self, theta: npt.NDArray) -> Any:
        return self.mle.with_params(theta)

    def default_priors(self, spec: Any, fixed: dict) -> dict:
        out: dict = _default_priors(self.mle, spec, fixed)
        return out

    def check(self, bound: BoundPriors) -> None:
        for j in bound.spec.joint:
            if isinstance(j, PowerPrior):
                raise ValueError(
                    "prior.from_data holds the lives of a life "
                    "distribution, without covariates. For an earlier "
                    "regression's data use prior.from_fit(earlier_model, "
                    "weight=...), or prior.from_model(earlier_model) for "
                    "its estimates and their covariance."
                )

    # -- units: one row -------------------------------------------------------
    def unit_log_lik(self, theta: npt.NDArray) -> npt.NDArray:
        from surpyval.univariate.regression._likelihood import (
            regression_row_ll,
        )

        theta = np.atleast_2d(theta)
        out = np.empty((len(theta), len(self.data.c)))
        with ignoring_warnings(), np.errstate(all="ignore"):
            for k, row in enumerate(theta):
                out[k] = regression_row_ll(self.fitter, self.data, *row)
        return out

    def unit_counts(self) -> npt.NDArray:
        return np.asarray(self.data.n, dtype=float)

    # -- replicates under the observed design ---------------------------------
    def _design(self) -> dict:
        """What a replicate keeps from the data: each unit's covariates,
        its entry time (left truncation) and when it would be censored.

        A unit that was right-censored keeps its own censoring time. A
        unit that failed was never seen censored, so its censoring time
        is not in the data: it is taken to be the latest time in the
        data where the data have any censoring (exact for a test ended
        at one time, the usual case), and none otherwise.
        """
        d = self.data
        c = np.asarray(d.c)
        if np.any(np.isin(c, (-1, 2))):
            raise ValueError(
                "A posterior predictive check simulates lives and censors "
                "them as the data were; left- and interval-censored rows "
                "come from an inspection scheme the data do not record, so "
                "it is not available for them. The Laplace checks are in "
                "model.diagnostics, and the family's residual checks on "
                "the maximum-likelihood fit: model.mle.diagnose()."
            )
        t = np.asarray(d.t, dtype=float)
        if np.isfinite(t[:, 1]).any():
            raise ValueError(
                "A posterior predictive check is not available for "
                "right-truncated data: a unit is in them because it "
                "failed early, which a replicate cannot be made to do. "
                "Use model.mle.diagnose()."
            )
        n = np.asarray(d.n)
        if np.any(n != np.round(n)):
            raise ValueError(
                "A posterior predictive check simulates each unit, so the "
                "counts n must be whole numbers."
            )
        x = np.asarray(d.x, dtype=float)
        x = x if x.ndim == 1 else x[:, 0]
        latest = float(np.max(x)) if np.any(c == 1) else np.inf
        rep = n.astype(int)
        return {
            "Z": np.repeat(np.asarray(d.Z, dtype=float), rep, axis=0),
            "censor": np.repeat(np.where(c == 1, x, latest), rep),
            "entry": np.repeat(
                np.where(np.isfinite(t[:, 0]), t[:, 0], 0.0), rep
            ),
            "x": np.repeat(x, rep),
            "c": np.repeat(c, rep),
        }

    def simulate(self, theta: npt.NDArray, rng: Any) -> dict:
        """Lives for every unit at its covariates, from the regression
        at ``theta``, each entering at its truncation time and censored
        as :meth:`_design` says: ``{"x", "c"}``."""
        design = self._design()
        model = self.mle.with_params(theta)
        Z = design["Z"]
        entry = design["entry"]
        lo = np.zeros(len(Z))
        late = entry > 0
        if late.any():
            lo[late] = np.asarray(model.ff(entry[late], Z[late]), dtype=float)
        u = lo + (1.0 - lo) * rng.random(len(Z))
        life = np.asarray(model.qf(u, Z), dtype=float)
        censored = life > design["censor"]
        return {
            "x": np.where(censored, design["censor"], life),
            "c": censored.astype(int),
        }

    def observed(self) -> dict:
        design = self._design()
        return {"x": design["x"], "c": design["c"]}

    def statistics(self) -> dict:
        def failures(d: dict) -> float:
            return float(np.sum(d["c"] == 0))

        def time_on_test(d: dict) -> float:
            return float(np.sum(d["x"]))

        def shortest(d: dict) -> float:
            return float(np.min(d["x"]))

        out = {}
        if np.any(np.asarray(self.data.c) == 1):
            out["number of failures"] = failures
        out["total time on test"] = time_on_test
        out["shortest time"] = shortest
        return out

    def check_flat(self, bound: BoundPriors) -> None:
        _check_flat(self.mle, self.data, bound)

    def describe_data(self) -> str:
        n = np.asarray(self.data.n)
        c = np.asarray(self.data.c)
        total, failed = int(n.sum()), int(n[c == 0].sum())
        k = int(np.asarray(self.data.Z).shape[1])
        return (
            f"{total} items ({failed} failed), {k} "
            f"covariate{'s' if k != 1 else ''}"
        )

    def same_data(self, other: Any) -> bool:
        a, b = self.data, other.data
        for name in ("x", "c", "n", "t", "Z"):
            u = np.asarray(getattr(a, name))
            v = np.asarray(getattr(b, name))
            if u.shape != v.shape or not np.array_equal(u, v):
                return False
        return True

    def to_dict(self) -> dict:
        return {
            "target": self.tag,
            "mle": self.mle.to_dict(),
            "data": self.data.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegressionTarget:
        import surpyval

        with ignoring_warnings():
            mle = surpyval.from_dict(d["mle"])
        return cls(mle, SurpyvalData.from_dict(d["data"]))


def _is_regression_fit(model: Any) -> bool:
    return (
        getattr(model, "kind", None) in COVERED
        and hasattr(model, "k_dist")
        and getattr(model, "data", None) is not None
        and not isinstance(model, BayesianResult)
    )


register_adapter(_is_regression_fit, lambda model: RegressionTarget(model))


def _check_flat(mle: Any, data: Any, bound: BoundPriors) -> None:
    """A flat prior is allowed only where the data determine the
    parameter: refused with no failure information, and, where every
    free parameter is flat, unless the maximum-likelihood fit's maximum
    is verified; and a mixture-cure regression's ``lfp_p`` is refused a
    flat prior where the data show no cured fraction (its likelihood
    then rises all the way to 1)."""
    improper = bound.improper_names
    if not improper:
        return
    if (
        _is_mixture_cure(mle)
        and "lfp_p" in improper
        and getattr(mle, "maximum", None) == "no finite maximum"
    ):
        raise ValueError(
            "The data show no cured fraction (no plateau at the end of "
            "follow-up): the likelihood keeps rising as lfp_p, the "
            "probability of ever failing at the reference row, goes to 1, "
            "so with no prior on it the cure fraction is not identified "
            "and the posterior has no finite mass. Give lfp_p a prior "
            "from what is known about the cure fraction (for example "
            "priors={'lfp_p': surpyval.prior.between(0.6, 0.95)}), fit "
            "the latency alone without a cure fraction, or follow the "
            "units for longer."
        )
    if not np.any(np.asarray(data.c) != 1):
        raise ValueError(
            f"With no failures, the data do not determine "
            f"{', '.join(improper)}; give each a prior (for example "
            "prior.around(value, between=(low, high)))."
        )
    if set(improper) == set(bound.free_names) and mle.maximum != "verified":
        raise ValueError(
            "Every parameter has no prior (flat), and the data alone do "
            "not determine them: the maximum-likelihood fit's maximum is "
            f"{mle.maximum!r}. Give a prior for {', '.join(improper)}."
        )


#: The follow-up check's level: above it, follow-up was not shown to be
#: long enough to tell a cured fraction from failures still to come.
FOLLOW_UP_LEVEL = 0.05


def _warn_follow_up(mle: Any) -> None:
    """Warn where a mixture-cure fit's follow-up was not shown to be
    long enough to identify the cured fraction (Maller and Zhou's check,
    ``MixtureCureModel.follow_up``), for data it applies to (exact and
    right-censored times)."""
    from surpyval.utils.warnings import caller_stacklevel

    try:
        check = mle.follow_up()
    except ValueError:
        return
    if check["p_value"] < FOLLOW_UP_LEVEL:
        return
    if check["plateau"] > 0 and check["n_plateau"] > 0:
        found = (
            f"{check['n_failures_in_window']:g} of the {check['n']:g} "
            "units failed in the last stretch of the failure times as long "
            f"as the plateau ({check['plateau']:.4g}), so the failures were "
            "still arriving when follow-up ended (Maller and Zhou's check, "
            f"p = {check['p_value']:.3g})"
        )
    else:
        found = (
            "no unit was still under observation after the last failure "
            "(there is no plateau at the end of follow-up)"
        )
    warnings.warn(
        f"The follow-up may be too short to identify the cure fraction: "
        f"{found}. The data cannot then tell a cured unit "
        "from one that has not failed yet, and the posterior of lfp_p (and "
        "of the cure fraction) rests on its prior and on the latency's "
        "shape. Give lfp_p a prior from what is known about the cure "
        "fraction, or follow the units for longer.",
        stacklevel=caller_stacklevel(),
    )


def from_mle(
    mle: Any,
    priors: Any,
    importance: bool = False,
    method: str = "auto",
) -> BayesianRegression:
    """The posterior of the regression ``mle`` was fitted for, with
    ``priors``: by the Laplace engine unless ``method`` says otherwise."""
    _check_covered(mle)
    if _is_mixture_cure(mle):
        _warn_follow_up(mle)
    result = BayesianMixtureCure if _is_mixture_cure(mle) else None
    model: BayesianRegression = fit_target(
        RegressionTarget(mle),
        priors,
        method=method,
        importance=importance,
        result=result or BayesianRegression,
    )
    return model


class BayesianRegression(BayesianResult):
    """
    A Bayesian fit of a parametric regression (proportional hazards,
    accelerated failure time or accelerated life): the posterior of the
    baseline's parameters and the coefficients, and the posterior
    predictive distribution of a new unit's life at given covariates.

    Built by a regression fitter's ``fit_bayes`` (or
    ``fit_bayes_from_df``).

    Attributes
    ----------
    params : numpy.ndarray
        The posterior medians of every parameter, named by
        ``parameter_names`` (the baseline's, then the coefficients, as
        the maximum-likelihood fit names them).
    posterior : Posterior
        The posterior's draws: ``draws``, ``to_dataframe``, ``quantile``.
    priors : PriorSpec
        The priors used, the defaults among them.
    default_priors : dict
        The priors SurPyval chose (none was given), by parameter name,
        in words.
    mle : ParametricRegressionModel
        The maximum-likelihood fit to the same data.
    diagnostics : dict
        How the posterior was computed and its checks against the exact
        posterior (``pareto_k``, ``shift``, ``approximation``; see
        :mod:`surpyval.bayes.engines.laplace`), and ``prior_conflict``:
        the parameters whose prior the data disagree with.
    method : str
        The engine: ``"laplace"`` unless ``fit_bayes`` was asked for
        another.
    kind : str
        The family, as the maximum-likelihood model names it.

    Examples
    --------
    A proportional hazards fit with a prior belief that the covariate
    raises the hazard (a log hazard ratio near 0.5):

    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(1)
    >>> Z = rng.normal(size=(40, 1))
    >>> x = sp.Weibull.random(40, 50.0, 2.0, random_state=2)
    >>> x = x * np.exp(-0.25 * Z[:, 0])
    >>> model = sp.WeibullPH.fit_bayes(
    ...     x, Z, priors={"coef_0": sp.Normal.from_params([0.5, 0.2])})
    >>> model.parameter_names
    ['alpha', 'beta', 'coef_0']
    >>> bool(0.2 < model.params[2] < 0.8)
    True
    >>> lo, hi = model.param_cri("coef_0")
    >>> bool(lo < model.params[2] < hi)
    True
    >>> bool(0 < model.sf(40.0, [[1.0]]) < model.sf(40.0, [[0.0]]) < 1)
    True
    """

    model_tag = "BayesianRegression"

    def __init__(
        self,
        mle: Any,
        posterior: Posterior,
        priors: PriorSpec,
        default_priors: dict[str, str],
        diagnostics: dict,
        data: SurpyvalData | None = None,
    ) -> None:
        target = (
            mle if isinstance(mle, Target) else RegressionTarget(mle, data)
        )
        diagnostics = {"method": "laplace", **dict(diagnostics)}
        super().__init__(
            target, posterior, priors, default_priors, diagnostics
        )
        mle = target.mle
        self.model = mle.model
        self.kind = mle.kind
        self.fixed = dict(mle.fixed)
        self.k_dist = mle.k_dist
        self.k = mle.k
        self.unit = getattr(mle, "unit", None)
        self._point = mle.with_params(self.params)

    def at_params(self) -> Any:
        """The plug-in regression model at the posterior medians (a
        ``ParametricRegressionModel`` built from parameters)."""
        return self._point

    # ------------------------------------------------------------------
    # The posterior predictive at covariates
    # ------------------------------------------------------------------
    def _draws(self) -> npt.NDArray:
        """The posterior's draws, which the predictive functions average
        over with :meth:`_weights`."""
        return self.posterior.points

    def _weights(self) -> npt.NDArray:
        """The draws' weights: equal, or the importance weights where the
        fit was reweighted."""
        return self.posterior.weights

    def _per_draw(self, fname: str) -> Any:
        """The family's function ``fname`` at every predictive draw, as a
        function of prepared ``(x, Z)``: one row per draw."""
        fn = getattr(self.model, fname)
        draws = self._draws()

        def values(x: Any, Z: Any, *_: Any) -> npt.NDArray:
            with np.errstate(all="ignore"):
                return np.stack(
                    [np.asarray(fn(x, Z, *d), dtype=float) for d in draws]
                )

        return values

    def _predict(self, fname: str, x: Any, Z: Any, grid: bool) -> Any:
        per_draw = self._per_draw("sf" if fname in ("Hf", "hf") else fname)
        below = 1.0 if fname == "sf" else 0.0

        def mean(x: Any, Z: Any, *p: Any) -> npt.NDArray:
            return self.posterior.expectation(per_draw(x, Z))

        if fname in ("sf", "ff", "df"):
            return self._point._eval(mean, x, Z, below, grid)
        sf = self._point._eval(mean, x, Z, 1.0, grid)
        if fname == "Hf":
            with np.errstate(divide="ignore"):
                return -np.log(sf)
        density = self._per_draw("df")

        def mean_df(x: Any, Z: Any, *p: Any) -> npt.NDArray:
            return self.posterior.expectation(density(x, Z))

        df = self._point._eval(mean_df, x, Z, 0.0, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            return df / sf

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """
        The posterior predictive survival function at covariates ``Z``:
        the probability that a new unit with those covariates survives
        past ``x``, the uncertainty in the parameters averaged in.

        Parameters
        ----------
        x : array like or scalar
            The times.
        Z : array like or DataFrame
            The covariates: one row per time, or one row for every time,
            as for the maximum-likelihood model's ``sf``.
        grid : bool, optional
            ``True`` gives every time for every row,
            ``(len(Z),) + x.shape``.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> model.sf([20.0, 40.0], [[0.0]]).shape
        (2,)
        """
        return self._predict("sf", x, Z, grid)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The posterior predictive CDF at covariates ``Z`` (see
        :meth:`sf`).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> bool(abs(model.ff(40.0, [[0.0]])
        ...          + model.sf(40.0, [[0.0]]) - 1) < 1e-12)
        True
        """
        return self._predict("ff", x, Z, grid)

    @keeps_query_shape
    def df(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The posterior predictive density at covariates ``Z`` (see
        :meth:`sf`).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> bool(model.df(40.0, [[0.0]]) > 0)
        True
        """
        return self._predict("df", x, Z, grid)

    @keeps_query_shape
    def hf(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The hazard of the posterior predictive distribution at
        covariates ``Z``, ``df / sf`` (not the posterior mean of the
        hazard: the units that survive are those with low hazards).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> bool(model.hf(40.0, [[0.0]]) > 0)
        True
        """
        return self._predict("hf", x, Z, grid)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any, *, grid: bool = False) -> Any:
        """The cumulative hazard of the posterior predictive distribution
        at covariates ``Z``, ``-log sf``.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> bool(model.Hf(40.0, [[0.0]]) > 0)
        True
        """
        return self._predict("Hf", x, Z, grid)

    def _draw_quantiles(self, p: npt.NDArray, Z: Any) -> npt.NDArray:
        """Each predictive draw's own quantile function at ``(p, Z)``:
        one row per draw."""
        with ignoring_warnings(), np.errstate(all="ignore"):
            return np.stack(
                [
                    np.atleast_1d(
                        np.asarray(
                            self.mle.with_params(d).qf(p, Z), dtype=float
                        )
                    )
                    for d in self._draws()
                ]
            )

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> Any:
        """
        The quantile function of the posterior predictive distribution
        at covariates ``Z``: the time by which a new unit with those
        covariates has failed with probability ``p``.

        It inverts :meth:`ff` numerically, between the smallest and the
        largest of the draws' own quantiles.

        Parameters
        ----------
        p : array like or scalar
            Probabilities; outside [0, 1] gives ``nan``.
        Z : array like or DataFrame
            One row per probability, or one row for all of them.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> b10 = model.qf(0.1, [[0.0]])
        >>> bool(abs(model.ff(b10, [[0.0]]) - 0.1) < 1e-8)
        True
        """
        from scipy.optimize import brentq

        from surpyval.utils.validation import warn_outside_unit_interval

        p = np.atleast_1d(np.asarray(p, dtype=float))
        warn_outside_unit_interval(p)
        per_draw = self._draw_quantiles(p, Z)
        rows = covariate_rows(self._point._prepare_Z(Z), self._n_covariates())
        rows = np.broadcast_to(rows, (len(p), rows.shape[1]))
        out = np.full(len(p), np.nan)
        for i, pi in enumerate(p):
            q = per_draw[:, i]
            if not np.isfinite(pi) or not 0 <= pi <= 1:
                continue
            if pi in (0.0, 1.0) or not np.all(np.isfinite(q)):
                out[i] = np.nanmin(q) if pi == 0 else np.nanmax(q)
                continue
            lo, hi = float(q.min()), float(q.max())
            if lo == hi:
                out[i] = lo
                continue
            z = rows[i : i + 1]

            def gap(t: float, pi: float = pi, z: Any = z) -> float:
                return float(self.ff(t, z)) - pi

            out[i] = brentq(gap, lo, hi, xtol=1e-300, rtol=1e-13)
        return out

    def random(
        self, size: int, Z: Any, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Lives drawn from the posterior predictive distribution at
        covariates ``Z``: for each, a posterior draw (of those
        :meth:`sf` averages over) at random, and a life from it, so the
        draws follow :meth:`sf`.

        Parameters
        ----------
        size : int
            The number of lives at each covariate row.
        Z : array like or DataFrame
            The covariate rows.
        random_state : None, int or numpy.random.Generator, optional
            The seed, by the one seed rule (``None``: numpy's global
            stream).

        Returns
        -------
        x : numpy.ndarray
            The ``size`` lives of each row, row after row.
        Z : numpy.ndarray
            The covariate row of each life.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> lives, rows = model.random(5, [[0.0], [1.0]], random_state=3)
        >>> lives.shape, rows.shape
        ((10,), (10, 1))
        """
        from surpyval.utils.rng import as_generator

        size = int(size)
        rows = covariate_rows(self._point._prepare_Z(Z), self._n_covariates())
        rows = np.repeat(rows, size, axis=0)
        rng = as_generator(random_state)
        draws = self._draws()
        which = rng.choice(len(draws), size=len(rows), p=self._weights())
        u = rng.random(len(rows))
        out = np.empty(len(rows))
        with ignoring_warnings(), np.errstate(all="ignore"):
            for k in np.unique(which):
                at = which == k
                model = self.mle.with_params(draws[k])
                out[at] = np.asarray(model.qf(u[at], rows[at]), dtype=float)
        return out, rows

    def _n_covariates(self) -> int:
        return int(np.asarray(self.data.Z).shape[1])

    # ------------------------------------------------------------------
    # Credible intervals
    # ------------------------------------------------------------------
    def cri(
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for a function of the regression at
        covariates ``Z``: the posterior's ``alpha_ci / 2`` and
        ``1 - alpha_ci / 2`` points of the function's value over the
        parameters' draws.

        Parameters
        ----------
        x : array like or scalar
            The times.
        Z : array like or DataFrame
            The covariates: one row per time, or one row for all.
        on : {'sf', 'ff', 'Hf', 'hf', 'df'}, optional
            The function (``'R'`` and ``'F'`` are ``'sf'`` and ``'ff'``).
            Default ``'sf'``.
        alpha_ci : float, optional
            1 minus the credible level. Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided adds a last ``[lower, upper]`` axis.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> lo, hi = model.cri(40.0, [[0.0]])
        >>> bool(lo < model.sf(40.0, [[0.0]]) < hi)
        True
        """
        # ``'R'`` and ``'F'`` are ``'sf'`` and ``'ff'``, as for every model
        on = {"R": "sf", "F": "ff"}.get(on, on)
        if on not in _ON:
            raise ValueError(
                f"on must be one of {', '.join(_ON)}; got {on!r}. "
                "(quantile_cri gives the interval of a quantile, param_cri "
                "of a parameter.)"
            )
        levels = interval_levels(alpha_ci, bound)
        shape = np.shape(x)
        # ff and Hf are falling functions of sf: their interval is sf's,
        # transformed, so the three agree exactly (as the predictive's do)
        # rather than by the interpolation between draws.
        via_sf = on in ("ff", "Hf")
        per_draw = self._per_draw("sf" if via_sf else on)
        below = 1.0 if on == "sf" or via_sf else 0.0
        out = []
        for level in levels:

            # (sf's point at 1 - level, where the function falls in sf)
            at = 1.0 - level if via_sf else level

            def at_level(
                x: Any, Z: Any, *p: Any, at: float = at
            ) -> npt.NDArray:
                return weighted_quantile(per_draw(x, Z), self._weights(), at)

            flat = np.reshape(np.asarray(x, dtype=float), -1)
            if via_sf:
                sf = self._point._eval(at_level, flat, Z, 1.0)
                with np.errstate(divide="ignore"):
                    value = 1.0 - sf if on == "ff" else -np.log(sf)
            else:
                value = self._point._eval(at_level, flat, Z, below)
            out.append(np.reshape(value, shape))
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def quantile_cri(
        self,
        p: npt.ArrayLike,
        Z: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        A credible interval for the ``p`` quantile of life at covariates
        ``Z`` (a B10 life at a use condition): the posterior's points of
        each draw's own quantile.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> lo, hi = model.quantile_cri(0.1, [[0.0]])
        >>> bool(lo < hi)
        True
        """
        from surpyval.utils.validation import warn_outside_unit_interval

        levels = interval_levels(alpha_ci, bound)
        shape = np.shape(p)
        flat = np.reshape(np.asarray(p, dtype=float), -1)
        warn_outside_unit_interval(flat, "quantile_cri")
        per_draw = self._draw_quantiles(flat, Z)
        out = [
            np.reshape(
                weighted_quantile(per_draw, self._weights(), level), shape
            )
            for level in levels
        ]
        return np.stack(out, axis=-1) if len(out) == 2 else out[0]

    def param_cri(
        self, name: str, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        """
        A credible interval for a parameter (a coefficient by its name,
        ``coef_0`` or its column's).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> model.param_cri("coef_0").shape
        (2,)
        """
        return super().param_cri(name, alpha_ci, bound)

    def _refuse_cb(self, what: str, instead: str) -> Any:
        raise ValueError(
            f"A Bayesian fit has credible intervals, not confidence "
            f"bounds: use {instead} in place of {what}."
        )

    def cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: a Bayesian fit's interval is :meth:`cri`."""
        return self._refuse_cb("cb", "cri(x, Z, on=...)")

    def quantile_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`quantile_cri`."""
        return self._refuse_cb("quantile_cb", "quantile_cri(p, Z)")

    def param_cb(self, *args: Any, **kwargs: Any) -> Any:
        """Refuses: use :meth:`param_cri`."""
        return self._refuse_cb("param_cb", "param_cri(name)")

    def diagnose(self, replicates: int = 200, random_state: Any = 0) -> Any:
        """
        The fit's checks in one report: the Laplace approximation's
        checks against the exact posterior, where the data disagree with
        the priors, and posterior predictive checks.

        A predictive check, in words: simulate ``replicates`` data sets
        from the fitted regression, each from one posterior draw, for
        the same units at the same covariates and censored as the data
        were, and see whether the data's number of failures, total time
        on test and shortest time sit among the simulated ones. A value
        in the extreme 2.5% at either end is flagged: the model does not
        reproduce that feature of the data. (A unit that was censored
        keeps its censoring time; one that failed is censored at the
        latest time in the data, which is exact for a test ended at one
        time. Left- and interval-censored and right-truncated data are
        refused, saying so.)

        The family's residual checks are those of the maximum-likelihood
        fit, ``model.mle.diagnose()``.

        Parameters
        ----------
        replicates : int, optional
            The data sets simulated for each check. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed. Default 0, so the report is the same on each call.

        Returns
        -------
        surpyval.bayes.BayesianDiagnosis
            Printable; ``checks`` holds ``(name, ok, detail)`` rows,
            ``predictive`` each check by its statistic, and ``ok``
            whether all passed.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> report = model.diagnose()
        >>> [name for name, ok, detail in report.checks]  # doctest: +ELLIPSIS
        ['posterior', 'priors against the data', 'predictive check, ...]
        >>> report.ok
        True
        """
        return super().diagnose(replicates, random_state)

    def waic(self) -> Any:
        """
        WAIC of the regression: the expected log predictive density of a
        new unit, summed over the data's rows (see
        :class:`~surpyval.bayes.PredictiveCriterion`). Compare two
        regressions fitted to the same data with
        :func:`surpyval.bayes.compare`.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> res = model.waic()
        >>> res.method, bool(res.p > 0)
        ('draws', True)
        """
        return super().waic()

    def loo(self) -> Any:
        """
        Leave-one-out cross-validation of the regression, by
        Pareto-smoothed importance sampling over the posterior's draws:
        each row's log predictive density from the posterior of the
        others, summed (see
        :class:`~surpyval.bayes.PredictiveCriterion`; a row whose Pareto
        k is high is flagged).

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> res = model.loo()
        >>> res.method, int(res.units)
        ('psis', 30)
        """
        return super().loo()

    # ------------------------------------------------------------------
    # Comparison and summary
    # ------------------------------------------------------------------
    def compare(self, alpha_ci: float = 0.05) -> Any:
        """
        The prior, the maximum-likelihood estimate (with its Wald
        confidence interval) and the posterior of each free parameter,
        side by side.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> sorted(model.compare().table)
        ['alpha', 'beta', 'coef_0']
        """
        return super().compare(alpha_ci)

    def summary(self, alpha_ci: float = 0.05) -> str:
        """The fit in words: the data, the priors (the defaults marked),
        the posterior of each parameter and how it was computed.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> "coef_0: default: Normal(0," in model.summary()
        True
        """
        return super().summary(alpha_ci)

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        A strict-JSON dictionary of the model: the maximum-likelihood fit
        (which holds the family and the covariates' names), the data, the
        priors, the posterior's draws and the diagnostics.
        ``surpyval.from_dict`` reads it.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as sp
        >>> Z = np.random.default_rng(1).normal(size=(30, 1))
        >>> x = sp.Weibull.random(30, 50.0, 2.0, random_state=2)
        >>> model = sp.WeibullPH.fit_bayes(
        ...     x, Z)
        >>> again = sp.from_dict(model.to_dict())
        >>> bool(again.sf(40.0, [[0.0]]) == model.sf(40.0, [[0.0]]))
        True
        """
        import surpyval

        out = {
            "model": self.model_tag,
            "kind": self.kind,
            "mle": self.mle.to_dict(),
            "priors": self.priors.to_dict(),
            "default_priors": dict(self.default_priors),
            "posterior": self.posterior.to_dict(),
            "data": self.data.to_dict(),
            "diagnostics": self._diagnostics_dict(),
            "surpyval_version": surpyval.__version__,
        }
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> BayesianRegression:
        """Rebuild a model from :meth:`to_dict`."""
        import surpyval

        require_model_tag(model_dict, cls.model_tag, "a Bayesian regression")
        with ignoring_warnings():
            mle = surpyval.from_dict(model_dict["mle"])
        model = cls(
            mle,
            Posterior.from_dict(model_dict["posterior"]),
            PriorSpec.from_dict(model_dict["priors"]),
            model_dict.get("default_priors", {}),
            model_dict.get("diagnostics", {}),
            data=SurpyvalData.from_dict(model_dict["data"]),
        )
        model.unit = model_dict.get("unit", model.unit)
        return model


class BayesianMixtureCure(BayesianRegression):
    """
    A Bayesian fit of a mixture-cure regression (``MixtureCure(dist)
    .fit_bayes``): the posterior of the latency's parameters, ``lfp_p``
    (the probability of ever failing at the reference row) and the
    coefficients, by the Laplace approximation with its checks. Beside a
    regression's predictive functions, :meth:`cure_probability` and
    :meth:`cure_cri` give the posterior of the cured fraction at given
    covariates.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> rng = np.random.default_rng(3)
    >>> Z = rng.binomial(1, 0.5, size=(150, 1)).astype(float)
    >>> cured = rng.uniform(size=150) < np.where(Z[:, 0] > 0, 0.5, 0.25)
    >>> life = sp.Weibull.random(150, 10.0, 2.0, random_state=4)
    >>> x = np.where(cured, 40.0, np.minimum(life, 40.0))
    >>> c = np.where(cured | (life > 40.0), 1, 0)
    >>> model = sp.MixtureCure(sp.Weibull).fit_bayes(x, Z, c)
    >>> type(model).__name__, model.method
    ('BayesianMixtureCure', 'laplace')
    >>> lo, hi = model.cure_cri([[1.0]])
    >>> bool(0 < lo < model.cure_probability([[1.0]]) < hi < 1)
    True
    """

    model_tag = "BayesianMixtureCure"

    def _cure_draws(self, Z: Any) -> npt.NDArray:
        """The cured fraction at ``Z`` (one row) at every posterior
        draw, ``(draws,)``."""
        points = self._draws()
        with ignoring_warnings(), np.errstate(all="ignore"):
            values = [
                float(np.asarray(self.mle.with_params(p).cure_probability(Z)))
                for p in points
            ]
        return np.asarray(values, dtype=float)

    def cure_probability(self, Z: Any) -> float:
        """The posterior mean of the cured fraction (the probability of
        never failing) of units with covariates ``Z`` (one row)."""
        return float(self.posterior.expectation(self._cure_draws(Z)))

    def cure_cri(
        self, Z: Any, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> npt.NDArray:
        """A credible interval for the cured fraction at covariates ``Z``
        (one row): the posterior's quantiles of it."""
        levels = interval_levels(alpha_ci, bound)
        values = self._cure_draws(Z)
        out = np.array(
            [
                float(weighted_quantile(values, self._weights(), level))
                for level in levels
            ]
        )
        return out if len(out) == 2 else out[0]
