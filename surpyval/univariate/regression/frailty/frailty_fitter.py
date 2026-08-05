"""
Fitter for shared-frailty proportional-hazards models (Gamma frailty).

The frailty enters as a random multiplier on the hazard shared within a group,
so the conditional cumulative hazard of an observation is ``u * eta * H0(t)``
with ``eta = exp(beta'Z)``. A Gamma frailty of mean 1 and variance ``theta``
integrates out of a group's likelihood in closed form. With ``D`` the number
of events in a group and ``H`` the sum of ``eta * H0`` over its observations,
the group log-likelihood is

.. math::
    \\sum_{\\text{events}} \\log(h_0 \\, \\eta)
    - \\tfrac{1}{\\theta}\\log\\theta - \\log\\Gamma(\\tfrac{1}{\\theta})
    + \\log\\Gamma(D + \\tfrac{1}{\\theta})
    - (D + \\tfrac{1}{\\theta}) \\log(H + \\tfrac{1}{\\theta}),

and the posterior frailty is ``(D + 1/theta) / (H + 1/theta)``. Only observed
(``c = 0``) and right-censored (``c = 1``) data are supported -- the closed
form relies on that split. The ordinary parametric MLE is not touched; this
module
maximises its own marginal likelihood on a fresh optimiser.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from ..regression_data import design_matrix_from_df
from .frailty_model import FrailtyModel


def _make_transforms(dist: Any, k_dist: int) -> tuple[
    Callable[[npt.NDArray, int], npt.NDArray],
    Callable[[npt.NDArray, int], npt.NDArray],
]:
    """Per-parameter (natural <-> unconstrained) maps for the optimiser.

    Baseline parameters follow the distribution's bounds (log for a positive
    parameter, logit for a ``(0, 1)`` parameter, identity otherwise); the
    coefficients are unconstrained; ``theta`` is positive, so log.
    """
    forms = []
    for low, high in dist.bounds[:k_dist]:
        if low == 0 and high is None:
            forms.append("log")
        elif (low, high) == (0, 1):
            forms.append("logit")
        else:
            forms.append("id")

    def to_unc(nat: npt.NDArray, n_beta: int) -> npt.NDArray:
        out = []
        for i, f in enumerate(forms):
            v = nat[i]
            if f == "log":
                out.append(np.log(v))
            elif f == "logit":
                out.append(np.log(v / (1 - v)))
            else:
                out.append(v)
        out.extend(nat[k_dist : k_dist + n_beta])  # beta: identity
        out.append(np.log(nat[-1]))  # theta: log
        return np.array(out, dtype=float)

    def to_nat(unc: npt.NDArray, n_beta: int) -> npt.NDArray:
        out = []
        for i, f in enumerate(forms):
            v = unc[i]
            if f == "log":
                out.append(np.exp(v))
            elif f == "logit":
                out.append(1.0 / (1.0 + np.exp(-v)))
            else:
                out.append(v)
        out.extend(unc[k_dist : k_dist + n_beta])
        out.append(np.exp(unc[-1]))
        return np.array(out, dtype=float)

    return to_unc, to_nat


def _numerical_hessian(
    f: Callable[[npt.NDArray], float],
    x: npt.NDArray,
    eps: float = 1e-5,
) -> npt.NDArray:
    """Finite-difference Hessian of scalar ``f`` at ``x``."""
    n = len(x)
    H = np.zeros((n, n))
    steps = np.maximum(np.abs(x), 1.0) * eps
    for i in range(n):
        for j in range(i, n):
            xi = x.copy()
            xi[i] += steps[i]
            xi[j] += steps[j]
            fpp = f(xi)
            xi = x.copy()
            xi[i] += steps[i]
            xi[j] -= steps[j]
            fpm = f(xi)
            xi = x.copy()
            xi[i] -= steps[i]
            xi[j] += steps[j]
            fmp = f(xi)
            xi = x.copy()
            xi[i] -= steps[i]
            xi[j] -= steps[j]
            fmm = f(xi)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (
                4 * steps[i] * steps[j]
            )
    return H


class FrailtyFitter:
    """Configured fitter for a shared-frailty PH model on one distribution."""

    def __init__(self, name: str, dist: Any, family: str = "gamma") -> None:
        if family != "gamma":
            raise NotImplementedError(
                "Only the 'gamma' frailty family is available; "
                f"got {family!r}."
            )
        self.name = name
        self.dist = dist
        self.family = family
        self.k_dist = len(dist.param_names)

    @staticmethod
    def create(distribution: Any, family: str = "gamma") -> "FrailtyFitter":
        return FrailtyFitter(
            f"{distribution.name}Frailty", distribution, family
        )

    # -- likelihood --------------------------------------------------------

    def _neg_ll_natural(
        self,
        nat: npt.NDArray,
        x: npt.NDArray,
        c: npt.NDArray,
        w: npt.NDArray,
        eta_Z: npt.NDArray,
        inv: npt.NDArray,
        n_beta: int,
    ) -> float:
        """Marginal negative log-likelihood in natural parameters.

        ``eta_Z`` is the covariate matrix (n_obs x n_beta); ``inv`` maps each
        observation to its group index; ``w`` are observation weights.
        """
        dist_params = nat[: self.k_dist]
        beta = nat[self.k_dist : self.k_dist + n_beta]
        theta = nat[-1]

        H0 = self.dist.Hf(x, *dist_params)
        h0 = self.dist.hf(x, *dist_params)
        eta = np.exp(eta_Z @ beta) if n_beta else np.ones_like(x)

        event = c == 0
        it = 1.0 / theta
        ll = np.sum(w[event] * (np.log(h0[event]) + np.log(eta[event])))

        n_groups = inv.max() + 1
        D = np.bincount(inv, weights=w * event, minlength=n_groups)
        H = np.bincount(inv, weights=w * eta * H0, minlength=n_groups)
        ll += np.sum(
            -it * np.log(theta)
            - gammaln(it)
            + gammaln(D + it)
            - (D + it) * np.log(H + it)
        )
        return -ll

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        x: Any,
        Z: Any = None,
        c: Any = None,
        n: Any = None,
        groups: Any = None,
        init: Any = None,
    ) -> FrailtyModel:
        """Fit the shared-frailty model.

        Parameters
        ----------
        x : array_like
            Observed times.
        Z : array_like, optional
            Covariates ``(n_obs, p)``. Omit for a frailty model with no
            covariates (a pure random-effects survival model).
        c : array_like, optional
            Censoring flags: ``0`` event, ``1`` right-censored (the only two
            supported). Defaults to all events.
        n : array_like, optional
            Observation weights / counts. Defaults to 1.
        groups : array_like
            The group (cluster) label of each observation. Required.
        init : array_like, optional
            Optional initial natural parameters ``[*dist, *beta, theta]``.
        """
        x = np.asarray(x, dtype=float).ravel()
        n_obs = x.shape[0]
        c = (
            np.zeros(n_obs, dtype=int)
            if c is None
            else np.asarray(c, dtype=int).ravel()
        )
        w = np.ones(n_obs) if n is None else np.asarray(n, dtype=float).ravel()
        if groups is None:
            raise ValueError("'groups' (a cluster label per row) is required.")
        groups = np.asarray(groups).ravel()

        if not np.all(np.isin(c, (0, 1))):
            raise ValueError(
                "Frailty fitting supports only observed (c=0) and "
                "right-censored (c=1) data."
            )
        if int((c == 0).sum()) == 0:
            raise ValueError("At least one event (c=0) is required.")

        labels, inv = np.unique(groups, return_inverse=True)
        n_groups = labels.shape[0]
        if n_groups < 2:
            raise ValueError(
                "The frailty variance is not identifiable from a single "
                "group; at least two groups are required."
            )

        if Z is None:
            Zc = np.zeros((n_obs, 0))
            n_beta = 0
            feature_names = None
        else:
            Zc = np.atleast_2d(np.asarray(Z, dtype=float))
            if Zc.shape[0] != n_obs:
                Zc = Zc.T
            n_beta = Zc.shape[1]
            feature_names = None

        to_unc, to_nat = _make_transforms(self.dist, self.k_dist)

        if init is None:
            base = self.dist.fit(x=x, c=c, n=w).params
            init_nat = np.concatenate(
                [np.asarray(base, float), np.zeros(n_beta), [0.5]]
            )
        else:
            init_nat = np.asarray(init, dtype=float)

        def obj_unc(u: npt.NDArray) -> float:
            nat = to_nat(u, n_beta)
            return self._neg_ll_natural(nat, x, c, w, Zc, inv, n_beta)

        u0 = to_unc(init_nat, n_beta)
        with np.errstate(all="ignore"):
            res = minimize(
                obj_unc,
                u0,
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
            )
            res = minimize(obj_unc, res.x, method="BFGS")
        nat = to_nat(res.x, n_beta)

        dist_params = nat[: self.k_dist]
        beta = nat[self.k_dist : self.k_dist + n_beta]
        theta = float(nat[-1])

        # Posterior (empirical-Bayes) frailty per group.
        H0 = self.dist.Hf(x, *dist_params)
        eta = np.exp(Zc @ beta) if n_beta else np.ones_like(x)
        it = 1.0 / theta
        D = np.bincount(inv, weights=w * (c == 0), minlength=n_groups)
        H = np.bincount(inv, weights=w * eta * H0, minlength=n_groups)
        post = (D + it) / (H + it)

        # Covariance of the natural parameters via a numerical Hessian.
        param_names = list(self.dist.param_names)
        param_names += [f"beta_{i}" for i in range(n_beta)]
        param_names += ["theta"]

        def nll_nat(v: npt.NDArray) -> float:
            return self._neg_ll_natural(v, x, c, w, Zc, inv, n_beta)

        covariance = None
        with np.errstate(all="ignore"):
            try:
                Hmat = _numerical_hessian(nll_nat, nat)
                cov = np.linalg.inv(Hmat)
                if np.all(np.isfinite(cov)):
                    covariance = cov
            except np.linalg.LinAlgError:
                covariance = None

        model = FrailtyModel()
        model.family = self.family
        model.dist = self.dist
        model.dist_params = np.asarray(dist_params, float)
        model.beta = np.asarray(beta, float)
        model.theta = theta
        model.k_dist = self.k_dist
        model.feature_names = feature_names
        model.group_labels = list(labels)
        model.frailties = {str(lab): float(u) for lab, u in zip(labels, post)}
        model.covariance = covariance
        model.param_names = param_names
        model.n_obs = n_obs
        model.n_events = int((c == 0).sum())
        model.n_groups = n_groups
        model._neg_ll = float(res.fun)
        return model

    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        group_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        formula: "str | None" = None,
        init: Any = None,
    ) -> FrailtyModel:
        """Fit from a :class:`pandas.DataFrame` naming the columns.

        Either ``Z_cols`` or ``formula`` describes the covariates (or neither,
        for a no-covariate frailty model); ``group_col`` names the cluster
        column. Covariate names / the formula transformer are retained so the
        fitted model predicts from raw-covariate DataFrames.
        """
        x = df[x_col].values
        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values
        groups = df[group_col].values

        feature_names = None
        model_spec = None
        if Z_cols is None and formula is None:
            Z = None
        else:
            Z, feature_names, model_spec = design_matrix_from_df(
                df, Z_cols=Z_cols, formula=formula
            )

        model = self.fit(x, Z=Z, c=c, n=n, groups=groups, init=init)
        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec
        return model
