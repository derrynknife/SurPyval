"""
The fitted shared-frailty model returned by :class:`FrailtyFitter`.

A shared-frailty model is a proportional-hazards model with an extra random
multiplier ``u`` on the hazard that is *shared* by every observation in a
group (a lot, a site, a repairable unit):

.. math::
    h(t \\mid Z, u) = u \\, h_0(t) \\, e^{\\beta' Z},

with the frailties drawn once per group from a Gamma distribution of mean 1 and
variance :math:`\\theta` (``theta``). ``theta`` measures the unexplained
between-group variability; ``theta = 0`` recovers an ordinary parametric PH
model.

Because the frailty enters multiplicatively on the *cumulative* hazard, a Gamma
frailty integrates out of a group's likelihood in closed form, and the same
conjugacy gives each group's posterior frailty in closed form -- so prediction
comes in two flavours:

* **marginal** (population-averaged), integrating the frailty out --
  ``S(t \\mid Z) = (1 + \\theta\\, e^{\\beta'Z} H_0(t))^{-1/\\theta}`` -- the
  right curve for a new unit from an unknown group; and
* **conditional**, on a supplied frailty value or on an *observed* group's
  posterior frailty ``S(t \\mid Z, u) = e^{-u e^{\\beta'Z} H_0(t)}``.
"""

from typing import Any

import numpy as np
from scipy.special import ndtri as _z

from surpyval.serialisation import SerialisableMixin, stamp_schema, to_native

from ..regression_data import (
    model_spec_to_meta,
    prepare_Z,
    rebuild_model_spec,
)


class FrailtyModel(SerialisableMixin):
    """A fitted shared-frailty proportional-hazards model.

    See :class:`FrailtyFitter` for how one is produced. Prediction methods
    (:meth:`sf`, :meth:`ff`, :meth:`hf`, :meth:`Hf`, :meth:`df`) return the
    *marginal* (population) curve by default; pass ``group=`` to condition on
    an observed group's posterior frailty, or ``frailty=`` to condition on a
    supplied frailty value.
    """

    def __init__(self) -> None:
        self.kind = "Frailty"
        self.family = "gamma"
        self.dist: Any = None
        self.dist_params: np.ndarray = np.array([])
        self.beta: np.ndarray = np.array([])
        self.theta: float = 0.0
        self.k_dist: int = 0
        self.feature_names: "list[str] | None" = None
        self.formula: "str | None" = None
        self._model_spec: Any = None
        self.group_labels: list = []
        self.frailties: dict = {}
        self.covariance: "np.ndarray | None" = None
        self.param_names: "list[str]" = []
        self.n_obs: int = 0
        self.n_events: int = 0
        self.n_groups: int = 0
        self._neg_ll: float = 0.0

    # -- covariate / frailty resolution ------------------------------------

    def _eta(self, Z: Any) -> np.ndarray:
        """The hazard multiplier ``exp(beta'Z)`` for a covariate setting."""
        if self.beta.size == 0:
            return np.array(1.0)
        if Z is None:
            raise ValueError(
                "This model was fit with covariates; 'Z' is required."
            )
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        Zp = np.atleast_2d(np.asarray(Zp, dtype=float))
        eta = np.exp(Zp @ self.beta)
        return eta[0] if eta.shape[0] == 1 else eta

    def _resolve_frailty(self, group: Any, frailty: Any) -> "float | None":
        """Return the frailty value to condition on, or ``None`` (marginal)."""
        if group is not None and frailty is not None:
            raise ValueError("Pass at most one of 'group' or 'frailty'.")
        if frailty is not None:
            return float(frailty)
        if group is not None:
            key = group
            if key not in self.frailties:
                key = str(group)
            if key not in self.frailties:
                raise KeyError(
                    f"Unknown group {group!r}; known groups are "
                    f"{list(self.frailties)[:8]}..."
                )
            return float(self.frailties[key])
        return None

    # -- prediction --------------------------------------------------------

    def Hf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Cumulative hazard (marginal, or conditional on a frailty)."""
        x = np.asarray(x, dtype=float)
        eta = self._eta(Z)
        H0 = self.dist.Hf(x, *self.dist_params)
        s = eta * H0
        u = self._resolve_frailty(group, frailty)
        if u is None:
            if self.theta < 1e-12:
                # theta -> 0 is the no-frailty PH limit
                # log(1 + theta s)/theta -> s; dividing by a zero theta
                # (e.g. frailty-free data, or a restored model) gave NaN
                # (#262).
                out = s
            else:
                out = np.log1p(self.theta * s) / self.theta
        else:
            out = u * s
        return out

    def sf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Survival function (marginal by default)."""
        return np.exp(-self.Hf(x, Z, group=group, frailty=frailty))

    def ff(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """CDF / failure function."""
        return 1.0 - self.sf(x, Z, group=group, frailty=frailty)

    def hf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Hazard function (marginal by default)."""
        x = np.asarray(x, dtype=float)
        eta = self._eta(Z)
        H0 = self.dist.Hf(x, *self.dist_params)
        h0 = self.dist.hf(x, *self.dist_params)
        u = self._resolve_frailty(group, frailty)
        if u is None:
            return eta * h0 / (1.0 + self.theta * eta * H0)
        return u * eta * h0

    def df(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """Density function."""
        return self.hf(x, Z, group=group, frailty=frailty) * self.sf(
            x, Z, group=group, frailty=frailty
        )

    # -- inference ---------------------------------------------------------

    @property
    def frailty_variance(self) -> float:
        """The estimated frailty variance :math:`\\hat\\theta`."""
        return float(self.theta)

    def standard_errors(self) -> "dict[str, float]":
        """Wald standard errors for each parameter, keyed by name."""
        if self.covariance is None:
            raise ValueError("No covariance was stored for this model.")
        se = np.sqrt(np.diag(self.covariance))
        return {name: float(s) for name, s in zip(self.param_names, se)}

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """Wald confidence bound on a named parameter.

        The bound is formed on a scale chosen from the parameter's support (log
        for the positive baseline parameters and ``theta``, natural for the
        unbounded coefficients) so the interval stays valid.
        """
        if self.covariance is None:
            raise ValueError("No covariance was stored for this model.")
        idx = self.param_names.index(name)
        est = self._param_vector()[idx]
        se = float(np.sqrt(self.covariance[idx, idx]))
        positive = name == "theta" or (
            idx < self.k_dist and self.dist.bounds[idx][0] == 0
        )
        if bound == "two-sided":
            q = _z(1 - alpha_ci / 2)
            signs = np.array([-1.0, 1.0])
        elif bound == "lower":
            q = _z(1 - alpha_ci)
            signs = np.array([-1.0])
        elif bound == "upper":
            q = _z(1 - alpha_ci)
            signs = np.array([1.0])
        else:
            raise ValueError("bound must be 'two-sided', 'lower' or 'upper'")
        if positive:
            if est <= 0:
                # A boundary estimate (theta -> 0: no detectable frailty)
                # has no log-scale Wald interval; dividing by zero gave
                # NaN/ZeroDivision (#262).
                return np.zeros_like(signs, dtype=float)
            return est * np.exp(signs * q * se / est)
        return est + signs * q * se

    def _param_vector(self) -> np.ndarray:
        return np.concatenate([self.dist_params, self.beta, [self.theta]])

    def summary(self) -> str:
        """A short text summary of the fit."""
        lines = [
            "Shared-Frailty Regression SurPyval Model",
            "========================================",
            f"Distribution        : {self.dist.name}",
            f"Frailty             : {self.family}",
            f"Groups              : {self.n_groups}"
            f"  (observations {self.n_obs}, events {self.n_events})",
            "Baseline            :",
        ]
        for name, val in zip(self.dist.param_names, self.dist_params):
            lines.append(f"    {name:>8} : {val:.6g}")
        if self.beta.size:
            lines.append("Coefficients        :")
            for i, b in enumerate(self.beta):
                lines.append(f"    beta_{i:<4}: {b:.6g}")
        lines.append(f"Frailty variance    : theta = {self.theta:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain, JSON-serialisable ``dict``."""
        out: dict[str, Any] = {
            "model": "FrailtyModel",
            "kind": self.kind,
            "family": self.family,
            "distribution": self.dist.name,
            "dist_params": np.asarray(self.dist_params, float).tolist(),
            "beta": np.asarray(self.beta, float).tolist(),
            "theta": float(self.theta),
            "k_dist": int(self.k_dist),
            "param_names": list(self.param_names),
            "group_labels": [str(g) for g in self.group_labels],
            "frailties": {str(k): float(v) for k, v in self.frailties.items()},
            "n_obs": int(self.n_obs),
            "n_events": int(self.n_events),
            "n_groups": int(self.n_groups),
            "_neg_ll": to_native(self._neg_ll),
        }
        if self.covariance is not None:
            out["covariance"] = np.asarray(self.covariance, float).tolist()
        if self.feature_names is not None:
            out["feature_names"] = list(self.feature_names)
        if self.formula is not None:
            out["formula"] = str(self.formula)
            if self._model_spec is not None:
                out["formula_meta"] = model_spec_to_meta(self._model_spec)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "FrailtyModel":
        """Rebuild a model from a :meth:`to_dict` dictionary."""
        import surpyval as surv
        from surpyval.univariate.parametric.parametric_fitter import (
            ParametricFitter,
        )

        if model_dict.get("model") != "FrailtyModel":
            raise ValueError("Must create a FrailtyModel from its own dict")

        dist = getattr(surv, model_dict["distribution"], None)
        if not isinstance(dist, ParametricFitter):
            raise ValueError(
                f"Unknown distribution '{model_dict['distribution']}'"
            )

        out = cls()
        out.family = model_dict.get("family", "gamma")
        out.dist = dist
        out.dist_params = np.array(model_dict["dist_params"], dtype=float)
        out.beta = np.array(model_dict["beta"], dtype=float)
        out.theta = float(model_dict["theta"])
        out.k_dist = int(model_dict["k_dist"])
        out.param_names = list(model_dict["param_names"])
        out.group_labels = list(model_dict.get("group_labels", []))
        out.frailties = {
            k: float(v) for k, v in model_dict.get("frailties", {}).items()
        }
        out.n_obs = int(model_dict.get("n_obs", 0))
        out.n_events = int(model_dict.get("n_events", 0))
        out.n_groups = int(model_dict.get("n_groups", 0))
        out._neg_ll = float(model_dict.get("_neg_ll", 0.0))
        if "covariance" in model_dict:
            out.covariance = np.array(model_dict["covariance"], dtype=float)
        out.feature_names = model_dict.get("feature_names")
        out.formula = model_dict.get("formula")
        formula_meta = model_dict.get("formula_meta")
        if out.formula is not None and formula_meta is not None:
            out._model_spec = rebuild_model_spec(out.formula, formula_meta)
        return out
