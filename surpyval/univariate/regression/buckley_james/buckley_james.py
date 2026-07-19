"""
Buckley-James semi-parametric accelerated-failure-time regression.

The accelerated-failure-time model writes the (log) lifetime as a linear
function of the covariates plus an error term with an *unspecified*
distribution,

.. math::
    \\log T = \\beta' Z + \\varepsilon ,

so it is the accelerated-time counterpart of Cox proportional hazards (which
leaves the baseline *hazard* unspecified). Buckley & James (1979) fit it under
right-censoring by iterating two steps until the coefficients stop moving:

1. **Impute.** Replace each censored log-time by its conditional expectation
   given that it exceeds the censoring time, estimated from the Kaplan-Meier
   distribution of the current residuals ``e_i = log T_i - beta'Z_i``.
2. **Re-fit.** Update ``beta`` by (weighted) least squares of the imputed
   responses on the covariates.

Only the slope is identified from the least-squares step; the location of the
errors is carried by the residual distribution, so predictions use the
residual Kaplan-Meier directly. Coefficients are reported in surpyval's
accelerated-failure sign convention -- a *positive* coefficient accelerates
failure (shortens life), matching ``WeibullAFT`` and the proportional-hazards
models -- which is the negative of the textbook ``log T = gamma'Z + eps``
slope. Prediction is therefore ``S(t | Z) = S_eps(log t + beta'Z)``.

The residual Kaplan-Meier is given an Efron tail-correction (its largest
residual is treated as an event) so it is a proper distribution and the
conditional means in step 1 are always finite. The iteration can settle into a
two-point cycle rather than a fixed point -- a known feature of the estimator
-- which is detected and resolved by averaging the cycle.
"""

import json
import warnings

import numpy as np

from surpyval.utils import (
    wrangle_and_check_form_and_Z_cols,
    wrangle_Z,
    xcnt_handler,
)


def _residual_km(e, delta, w):
    """
    Weighted Kaplan-Meier of the residuals with an Efron tail-correction.

    Returns the sorted unique residual values, the (right-continuous) survival
    at each, and the probability mass (jump) the estimator places at each.
    Forcing the largest residual to be an event makes the estimator proper, so
    the jumps sum to one and conditional expectations above any point are
    finite.
    """
    order = np.argsort(e, kind="mergesort")
    e = e[order]
    delta = delta[order].astype(float).copy()
    w = w[order].astype(float)

    # Efron tail-correction: the largest residual(s) act as an event.
    delta[e == e[-1]] = 1.0

    uniq, first_idx = np.unique(e, return_index=True)
    w_cum = np.concatenate([[0.0], np.cumsum(w)])
    total = w_cum[-1]
    # At-risk weight for each unique time (residuals >= that time).
    r = total - w_cum[first_idx]

    d = np.zeros(uniq.shape[0])
    grp = np.searchsorted(uniq, e)
    np.add.at(d, grp, np.where(delta == 1.0, w, 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        surv = np.cumprod(1.0 - d / r)
    surv = np.clip(surv, 0.0, 1.0)
    s_prev = np.concatenate([[1.0], surv[:-1]])
    jumps = s_prev - surv
    return uniq, surv, jumps


def _impute(Y, delta, Zbeta, w):
    """
    Buckley-James imputed responses. Observed (``delta == 1``) rows keep their
    value; each censored row is replaced by ``Zbeta_i + E[e | e > e_i]``, the
    conditional mean of the residual above its censored value under the
    residual Kaplan-Meier. Where nothing lies above (the largest residual) the
    censored value is kept.
    """
    e = Y - Zbeta
    uniq, surv, jumps = _residual_km(e, delta, w)

    # Reverse cumulative sum of the jump-weighted residual values: at index j,
    # sum_{k > j} jumps[k] * uniq[k].
    contrib = (jumps * uniq)[::-1]
    tail_above = np.concatenate([np.cumsum(contrib)[::-1][1:], [0.0]])

    idx = np.searchsorted(uniq, e)
    s_at = surv[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_mean = np.where(s_at > 0, tail_above[idx] / s_at, np.nan)

    imputed = Zbeta + cond_mean
    # Observed rows, and censored rows with no mass above, keep the raw value.
    keep = (delta == 1.0) | ~np.isfinite(imputed)
    return np.where(keep, Y, imputed)


def _wls_slope(Z, Y, w):
    """Weighted least-squares slope of ``Y`` on ``Z`` with the intercept
    profiled out by centring (the location stays in the residuals)."""
    wsum = w.sum()
    Zbar = (w[:, None] * Z).sum(axis=0) / wsum
    Ybar = (w * Y).sum() / wsum
    Zc = Z - Zbar
    A = (w[:, None] * Zc).T @ Zc
    b = (w * (Y - Ybar))[None, :] @ Zc
    return np.linalg.solve(A, b.ravel())


def _fit_beta(Y, delta, Z, w, tol, max_iter):
    """Run the Buckley-James iteration and return
    ``(beta, n_iter, converged)``. A two-point cycle is resolved by averaging
    the cycle."""
    beta = _wls_slope(Z, Y, w)  # least squares ignoring censoring, as a start
    history = [beta]
    converged = False
    for it in range(1, max_iter + 1):
        imputed = _impute(Y, delta, Z @ beta, w)
        beta_new = _wls_slope(Z, imputed, w)
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            converged = True
            break
        # Two-cycle detection: the new iterate matches the one before last.
        if len(history) >= 2 and np.linalg.norm(beta_new - history[-2]) < tol:
            beta = 0.5 * (beta_new + beta)
            converged = True
            break
        history.append(beta_new)
        beta = beta_new
    return beta, it, converged


class BuckleyJamesModel:
    """
    A fitted Buckley-James accelerated-failure-time model.

    Predictions use the residual Kaplan-Meier: ``sf(t | Z) = S_eps(log t +
    beta'Z)``. ``coef`` are the covariate coefficients in surpyval's
    accelerated-failure convention: a positive coefficient accelerates failure
    (shortens life), matching ``WeibullAFT`` and the PH models.
    """

    feature_names = None
    formula = None
    _model_spec = None

    def __init__(self, beta, resid, resid_surv, n_iter, converged, data):
        self.beta = np.asarray(beta, dtype=float)
        self.params = self.beta
        self.coef = self.beta
        self._resid = resid
        self._resid_surv = resid_surv
        self.n_iter = n_iter
        self.converged = converged
        self._data = data  # (Y, delta, Z, w) for the bootstrap

    def _prepare_Z(self, Z):
        from ..regression_data import prepare_Z

        return prepare_Z(Z, self.feature_names, self._model_spec)

    def _resid_sf(self, r):
        # Right-continuous residual survival at query points ``r``.
        idx = np.searchsorted(self._resid, r, side="right") - 1
        out = np.where(
            idx < 0,
            1.0,
            self._resid_surv[np.clip(idx, 0, len(self._resid_surv) - 1)],
        )
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self):
        """
        Serialise this fitted Buckley-James model to a plain, JSON-serialisable
        dict.

        Predictions use the residual Kaplan-Meier ``sf(t | Z) =
        S_eps(log t + beta'Z)``, so what is stored is the coefficients ``beta``
        and the residual survival step arrays (``_resid`` and ``_resid_surv``).
        The fit data ``(Y, delta, Z, w)`` is stored too, so the restored model
        can still run :meth:`bootstrap_ci`.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out = {
            "model": "BuckleyJamesModel",
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "resid": np.asarray(self._resid, dtype=float).tolist(),
            "resid_surv": np.asarray(self._resid_surv, dtype=float).tolist(),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
        }
        if self._data is not None:
            Y, delta, Z, w = self._data
            out["data"] = {
                "Y": np.asarray(Y, dtype=float).tolist(),
                "delta": np.asarray(delta, dtype=float).tolist(),
                "Z": np.asarray(Z, dtype=float).tolist(),
                "w": np.asarray(w, dtype=float).tolist(),
            }
        if self.feature_names is not None:
            out["feature_names"] = list(self.feature_names)
        if self.formula is not None:
            out["formula"] = self.formula
        return out

    def to_json(self, fp):
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict):
        """
        Rebuild a Buckley-James model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        if model_dict.get("model") != "BuckleyJamesModel":
            raise ValueError(
                "Must create a Buckley-James model from a "
                "BuckleyJamesModel dict"
            )
        data = None
        if "data" in model_dict:
            d = model_dict["data"]
            data = (
                np.array(d["Y"], dtype=float),
                np.array(d["delta"], dtype=float),
                np.array(d["Z"], dtype=float),
                np.array(d["w"], dtype=float),
            )
        out = cls(
            np.array(model_dict["beta"], dtype=float),
            np.array(model_dict["resid"], dtype=float),
            np.array(model_dict["resid_surv"], dtype=float),
            int(model_dict["n_iter"]),
            bool(model_dict["converged"]),
            data,
        )
        out.feature_names = model_dict.get("feature_names")
        out.formula = model_dict.get("formula")
        return out

    @classmethod
    def from_json(cls, fp):
        """Load a model from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def sf(self, x, Z):
        """Survival ``P(T > x | Z) = S_eps(log x - beta'Z)`` for a single
        covariate vector ``Z``."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = self._prepare_Z(Z)
        Z = np.asarray(Z, dtype=float).ravel()
        # beta is the accelerated-failure (negated) slope, so the residual
        # r = log t - gamma'Z = log t + beta'Z.
        r = np.log(x) + Z @ self.beta
        return self._resid_sf(r)

    def ff(self, x, Z):
        return 1.0 - self.sf(x, Z)

    def Hf(self, x, Z):
        with np.errstate(divide="ignore"):
            return -np.log(self.sf(x, Z))

    def bootstrap_ci(self, alpha_ci=0.05, n_boot=200, seed=None):
        """
        Percentile bootstrap confidence intervals for the coefficients.

        Buckley-James has no simple closed-form standard error, so uncertainty
        is obtained by resampling observations with replacement, refitting, and
        taking percentiles of the coefficient distribution. Returns an
        ``(n_coef, 2)`` array of ``[lower, upper]`` bounds.
        """
        Y, delta, Z, w = self._data
        rng = np.random.default_rng(seed)
        n = Y.shape[0]
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                g, _, _ = _fit_beta(
                    Y[idx], delta[idx], Z[idx], w[idx], 1e-5, 100
                )
                boot.append(-g)  # report in the accelerated-failure sign
            except np.linalg.LinAlgError:
                continue
        boot = np.asarray(boot)
        lo = np.quantile(boot, alpha_ci / 2.0, axis=0)
        hi = np.quantile(boot, 1.0 - alpha_ci / 2.0, axis=0)
        return np.stack([lo, hi], axis=-1)

    def __repr__(self):
        lines = [
            "Buckley-James AFT SurPyval Model",
            "================================",
            "Kind                : Semi-Parametric AFT",
            f"Converged           : {self.converged} ({self.n_iter} iters)",
            "Coefficients (positive => accelerates failure):",
        ]
        names = self.feature_names or [
            f"beta_{i}" for i in range(self.beta.size)
        ]
        for nm, b in zip(names, self.beta):
            lines.append(f"   {nm:>10}  :  {b: .6f}")
        return "\n".join(lines)


class BuckleyJames_:
    def fit(
        self,
        x,
        Z,
        c=None,
        n=None,
        tol=1e-5,
        max_iter=100,
    ):
        """
        Fit the Buckley-James AFT model.

        Parameters
        ----------
        x : array_like
            Observed (positive) times.
        Z : array_like
            Covariate matrix, one row per observation.
        c : array_like, optional
            Censoring flags: 0 observed, 1 right-censored. Left and interval
            censoring are not supported. Defaults to all observed.
        n : array_like, optional
            Counts per row (case weights). Defaults to 1.
        tol : float, optional
            Convergence tolerance on the coefficient step. Default 1e-5.
        max_iter : int, optional
            Maximum Buckley-James iterations. Default 100.

        Returns
        -------
        BuckleyJamesModel
            The fitted model.
        """
        x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)
        Z, mask = wrangle_Z(Z)
        x, c, n = x[mask], c[mask], n[mask]
        x, c, n, Z = (a.astype(float) for a in (x, c, n, Z))

        if np.any((c != 0) & (c != 1)):
            raise ValueError(
                "Buckley-James supports only observed (c=0) and "
                "right-censored (c=1) data."
            )
        if np.any(x <= 0):
            raise ValueError(
                "Buckley-James models log(time); all times must be positive."
            )

        Y = np.log(x)
        delta = (c == 0).astype(float)
        # gamma is the textbook ``log T = gamma'Z + eps`` slope; report its
        # negative so a positive coefficient accelerates failure.
        gamma, n_iter, converged = _fit_beta(Y, delta, Z, n, tol, max_iter)
        if not converged:
            warnings.warn(
                "Buckley-James did not converge in {} iterations; returning "
                "the last iterate.".format(max_iter)
            )

        # Final residual distribution used for prediction.
        resid, resid_surv, _ = _residual_km(Y - Z @ gamma, delta, n)

        return BuckleyJamesModel(
            -gamma, resid, resid_surv, n_iter, converged, (Y, delta, Z, n)
        )

    def fit_from_df(
        self,
        df,
        x_col,
        Z_cols=None,
        c_col=None,
        n_col=None,
        formula=None,
        tol=1e-5,
        max_iter=100,
    ):
        """
        Fit a Buckley-James model from a pandas DataFrame. See :meth:`fit` for
        the estimator; ``Z_cols`` or ``formula`` selects the covariates.
        """
        Z, mask, form, feature_names, model_spec = (
            wrangle_and_check_form_and_Z_cols(Z_cols, formula, df)
        )
        sub = df.loc[mask]
        x = sub[x_col].values
        c = sub[c_col].values if c_col is not None else None
        n = sub[n_col].values if n_col is not None else None

        model = self.fit(x, Z, c=c, n=n, tol=tol, max_iter=max_iter)
        model.formula = form
        model.feature_names = feature_names
        model._model_spec = model_spec
        return model


BuckleyJames = BuckleyJames_()
