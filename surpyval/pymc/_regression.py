"""SurPyval's regression likelihoods (proportional hazards, accelerated
failure time and accelerated life, with covariates) as pytensor terms.

Each family's covariates enter the baseline distribution's functions in
SurPyval's own way, written here with the baseline's pytensor functions
(:mod:`surpyval.pymc._logp`), so the term equals the fitter's
``neg_ll`` (what ``fit`` minimises):

- proportional hazards, ``H(x | Z) = e^{beta'Z} H_0(x)``:
  ``log S = e^{beta'Z} log S_0`` and ``log f = beta'Z + log h_0 + log S``;
- accelerated failure time, ``S(x | Z) = S_0(e^{beta'Z} x)``: the
  baseline's functions at the rescaled time, the density times
  ``e^{beta'Z}``;
- accelerated life, the baseline's life parameter replaced row by row by
  the life model's ``L(Z)`` (``1 / L`` for a rate, ``log L`` for the
  LogNormal's ``mu``).

pytensor is imported by the caller, never here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.pymc._logp import FAMILIES, Family, log1mexp
from surpyval.utils.aliases import canonical_name

#: The families of regression fitters the bridge takes.
REGRESSION_KINDS = (
    "proportional hazards",
    "accelerated failure time",
    "accelerated life",
)


def regression_kind(fitter: Any) -> str | None:
    """Which of :data:`REGRESSION_KINDS` ``fitter`` is, or None for a
    distribution (or anything else)."""
    from surpyval.univariate.regression.accelerated_failure_time.aft_fitter import (  # noqa: E501
        AFTFitter,
    )
    from surpyval.univariate.regression.accelerated_life.parameter_substitution import (  # noqa: E501
        ParameterSubstitutionFitter,
    )
    from surpyval.univariate.regression.proportional_hazards.proportional_hazards_fitter import (  # noqa: E501
        ProportionalHazardsFitter,
    )

    if isinstance(fitter, ProportionalHazardsFitter):
        return "proportional hazards"
    if isinstance(fitter, AFTFitter):
        return "accelerated failure time"
    if isinstance(fitter, ParameterSubstitutionFitter):
        return "accelerated life"
    return None


def _baseline(fitter: Any, what: str) -> Family:
    name = canonical_name(fitter.dist)
    if name not in FAMILIES:
        raise ValueError(
            f"{what} has no PyMC likelihood for a {fitter.dist.name} "
            f"baseline. Regression baselines it takes: "
            f"{', '.join(FAMILIES)}."
        )
    return FAMILIES[name]


def proportional_hazards(base: Family) -> Family:
    """``base`` with a last parameter, ``log phi`` (one value per row or
    a scalar), multiplying its cumulative hazard by ``phi``."""

    def log_sf(pt: Any, x: Any, *p: Any) -> Any:
        *b, lp = p
        return pt.exp(lp) * base.log_sf(pt, x, *b)

    def log_df(pt: Any, x: Any, *p: Any) -> Any:
        *b, lp = p
        ls0 = base.log_sf(pt, x, *b)
        return lp + base.log_df(pt, x, *b) - ls0 + pt.exp(lp) * ls0

    def log_ff(pt: Any, x: Any, *p: Any) -> Any:
        return log1mexp(pt, log_sf(pt, x, *p))

    def centre(pt: Any, *p: Any) -> Any:
        return base.centre(pt, *p[:-1])

    return Family(log_df, log_sf, log_ff, centre)


def accelerated_failure_time(base: Family) -> Family:
    """``base`` with a last parameter, ``log phi``, rescaling time: the
    baseline's functions at ``phi x``."""

    def log_sf(pt: Any, x: Any, *p: Any) -> Any:
        *b, lp = p
        return base.log_sf(pt, pt.exp(lp) * x, *b)

    def log_ff(pt: Any, x: Any, *p: Any) -> Any:
        *b, lp = p
        return base.log_ff(pt, pt.exp(lp) * x, *b)

    def log_df(pt: Any, x: Any, *p: Any) -> Any:
        *b, lp = p
        return lp + base.log_df(pt, pt.exp(lp) * x, *b)

    def centre(pt: Any, *p: Any) -> Any:
        *b, lp = p
        return base.centre(pt, *b) / pt.exp(lp)

    return Family(log_df, log_sf, log_ff, centre)


def _coefficients(
    pt: Any, names: list, params: dict, p: int, what: str
) -> Any:
    """The coefficients as one vector: from ``coefs`` (a vector of
    ``p``), or each by its name."""
    given = [k for k in names if k in params]
    if "coefs" in params:
        if given:
            raise TypeError(
                f"{what}: give the coefficients as coefs (one vector) or "
                f"by name ({', '.join(names)}), not both."
            )
        vec = params.pop("coefs")
        if not hasattr(vec, "type"):
            vec = np.asarray(vec, dtype=float)
        vec = pt.as_tensor_variable(vec)
        if vec.ndim != 1 or (
            vec.type.shape[0] is not None and vec.type.shape[0] != p
        ):
            raise ValueError(
                f"{what}: coefs must be a vector of {p} coefficients, one "
                "per column of Z."
            )
        return vec
    missing = [k for k in names if k not in params]
    if missing:
        raise TypeError(
            f"{what}: give every coefficient ({', '.join(names)}) by name, "
            f"or all of them as coefs=; missing {', '.join(missing)}."
        )
    return pt.stack([pt.as_tensor_variable(params.pop(k)) for k in names])


def regression_theta(
    pt: Any,
    fitter: Any,
    kind: str,
    Z: np.ndarray,
    params: dict,
    what: str,
) -> tuple[Family, list, list]:
    """The family the rows follow, its parameters (tensors: scalars or one
    value per row) and the names of the baseline's parameters taken from
    ``params``."""
    base = _baseline(fitter, what)
    params = dict(params)
    dist_names = list(fitter.dist.parameter_names)
    if kind == "accelerated life":
        return (
            base,
            _life_substituted(pt, fitter, Z, params, dist_names, what),
            dist_names,
        )
    from surpyval.univariate.regression._fit_skeleton import LogLinearPhi

    if kind == "proportional hazards" and fitter.phi is not LogLinearPhi.phi:
        raise ValueError(
            f"{what} takes proportional hazards models with the log-linear "
            "link exp(beta'Z); this fitter has its own phi. Give the "
            "baseline's parameters per row yourself, with the distribution "
            f"(log_likelihood(surpyval.{fitter.dist.name}, ...))."
        )
    p = Z.shape[1]
    names = [f"coef_{j}" for j in range(p)]
    beta = _coefficients(pt, names, params, p, what)
    theta = _baseline_parameters(pt, dist_names, params, what)
    log_phi = pt.dot(pt.as_tensor_variable(Z), beta)
    family = (
        proportional_hazards(base)
        if kind == "proportional hazards"
        else accelerated_failure_time(base)
    )
    return family, theta + [log_phi], dist_names


def _baseline_parameters(
    pt: Any, names: list, params: dict, what: str, skip: str | None = None
) -> list:
    out: list = []
    for name in names:
        if name == skip:
            params.pop(name, None)
            out.append(None)
            continue
        if name not in params:
            raise TypeError(
                f"{what}: give the baseline's {name} (a PyMC variable, an "
                "expression or a number)."
            )
        value = params.pop(name)
        if not hasattr(value, "type"):
            value = np.asarray(value, dtype=float)
        tensor = pt.as_tensor_variable(value)
        if tensor.ndim != 0:
            raise ValueError(
                f"{what}: the baseline's {name} is one value (a scalar); "
                "the covariates are what vary by row."
            )
        out.append(tensor)
    if params:
        raise TypeError(
            f"{what}: unknown parameters {', '.join(sorted(params))}; this "
            f"model takes {', '.join(names)} and its coefficients."
        )
    return out


def _life_substituted(
    pt: Any,
    fitter: Any,
    Z: np.ndarray,
    params: dict,
    dist_names: list,
    what: str,
) -> list:
    """The baseline's parameters with the life parameter, row by row,
    from the life model's ``L(Z)`` (by its log where the model gives its
    log-life, as the fit computes it)."""
    life = fitter.life_model
    life_names = sorted(life.phi_param_map, key=life.phi_param_map.get)
    given = params.pop("life", None)
    if not isinstance(given, dict):
        shape = ", ".join(repr(k) + ": ..." for k in life_names)
        raise TypeError(
            f"{what}: give the {life.name} life model's parameters as a "
            f"dict, life={{{shape}}} (a dict, as their names can be the "
            "data's, such as n and c)."
        )
    given = dict(given)
    missing = [k for k in life_names if k not in given]
    unknown = sorted(set(given) - set(life_names))
    if missing or unknown:
        raise TypeError(
            f"{what}: the {life.name} life model's parameters are "
            f"{', '.join(life_names)}; life= "
            + (f"is missing {', '.join(missing)}" if missing else "")
            + ("; " if missing and unknown else "")
            + (f"has unknown {', '.join(unknown)}" if unknown else "")
            + "."
        )
    values = []
    for k in life_names:
        v = given.pop(k)
        if not hasattr(v, "type"):
            v = np.asarray(v, dtype=float)
        values.append(pt.as_tensor_variable(v))
    slot = fitter.life_parameter
    theta = _baseline_parameters(pt, dist_names, params, what, skip=slot)
    Zm = fitter._stress_matrix(Z)
    stress = Zm[:, 0] if Zm.shape[1] == 1 else Zm
    log_scale = tuple(getattr(life, "log_scale_parameters", ()) or ())
    try:
        if log_scale:
            logged = [
                pt.log(v) if k in log_scale else v
                for k, v in zip(life_names, values)
            ]
            log_l = pt.as_tensor_variable(life.log_life(stress, *logged))
        else:
            log_l = pt.log(pt.as_tensor_variable(life.phi(stress, *values)))
    except (TypeError, ValueError, AttributeError) as err:
        raise ValueError(
            f"{what} cannot write the {life.name} life model as a pytensor "
            "expression. Give the baseline's life parameter per row "
            "yourself, with the distribution "
            f"(log_likelihood(surpyval.{fitter.dist.name}, ..., "
            f"{slot}=...))."
        ) from err
    family = canonical_name(fitter.dist)
    if family in ("Exponential", "Gamma"):
        row = pt.exp(-log_l)
    elif family == "LogNormal":
        row = log_l
    else:
        row = pt.exp(log_l)
    theta[dist_names.index(slot)] = row
    return theta
