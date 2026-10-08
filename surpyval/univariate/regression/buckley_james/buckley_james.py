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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils import finite_covariate_mask
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import percentile_bounds
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import covariate_rows, keeps_query_shape
from surpyval.utils.validation import check_alpha_ci

from .._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    expand,
    warn_aliased,
)
from .._concordance import ConcordanceMixin
from .._prediction import (
    ConditionalSurvivalMixin,
    paired_probabilities,
    step_quantiles,
)
from .._summary import coefficient_names, coefficient_table
from ..regression_data import (
    LinearPredictorMixin,
    NoLikelihoodMixin,
    design_matrix_from_df,
    restore_covariate_meta,
    semi_parametric_inputs,
    serialise_covariate_meta,
)

# How far below a residual Kaplan-Meier step a query's residual may round
# and still be at the step, relative to max(|r|, 1) (see ``_resid_sf``).
RESID_ROUNDING = 1e-12


def _residual_km(
    e: npt.NDArray, delta: npt.NDArray, w: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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


def _impute(
    Y: npt.NDArray, delta: npt.NDArray, Zbeta: npt.NDArray, w: npt.NDArray
) -> npt.NDArray:
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


def _wls_slope(Z: npt.NDArray, Y: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """Weighted least-squares slope of ``Y`` on ``Z`` with the intercept
    profiled out by centring (the location stays in the residuals)."""
    wsum = w.sum()
    Zbar = (w[:, None] * Z).sum(axis=0) / wsum
    Ybar = (w * Y).sum() / wsum
    Zc = Z - Zbar
    A = (w[:, None] * Zc).T @ Zc
    b = (w * (Y - Ybar))[None, :] @ Zc
    return np.linalg.solve(A, b.ravel())


def _aliased(Z: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """The columns whose coefficients the least-squares step cannot
    determine (#476), to be aliased (see
    :mod:`surpyval.univariate.regression._aliasing`).

    The slope is fitted with the intercept profiled out, so a constant
    column (or any column of a single observation) is that intercept, and
    a column that is a linear combination of the others adds nothing to
    them: the centred Gram matrix is singular in their direction. Such a
    design escaped as a bare ``LinAlgError: Singular matrix``, and was
    then refused with a ``ValueError``; it is aliased, as R's ``lm``
    aliases it.
    """
    Zc = Z - (w[:, None] * Z).sum(axis=0) / w.sum()
    gram = (w[:, None] * Zc).T @ Zc
    return aliased_columns(gram, Z.shape[0], constant_columns(Z))


def _fit_beta(
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray, int, bool]:
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


class BuckleyJamesModel(
    ConditionalSurvivalMixin,
    NoLikelihoodMixin,
    LinearPredictorMixin,
    ConcordanceMixin,
    SerialisableMixin,
):
    """
    A fitted Buckley-James accelerated-failure-time model.

    Predictions use the residual Kaplan-Meier: ``sf(t | Z) = S_eps(log t +
    beta'Z)``. ``coef`` are the covariate coefficients in surpyval's
    accelerated-failure convention: a positive coefficient accelerates failure
    (shortens life), matching ``WeibullAFT`` and the PH models.

    Examples
    --------
    On the Rossi recidivism data, where ``arrest`` is 1 for an arrest (so
    the censoring flag is ``1 - arrest``), prior convictions (``prio``)
    shorten the time to arrest and financial aid (``fin``) lengthens it:

    >>> from surpyval import BuckleyJames
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> Z = df[["fin", "age", "prio"]].values
    >>> model = BuckleyJames.fit(x, Z, c=c)
    >>> model.beta.round(4)
    array([-0.2663, -0.0253,  0.0588])
    >>> model.converged
    True
    >>> model.sf([20, 52], [1, 25, 3]).round(4)
    array([0.9444, 0.8113])
    """

    feature_names: "list[str] | None" = None
    formula: "str | None" = None
    _model_spec: Any = None
    #: Covariates of the wrong width are refused by name (#657).
    _CHECKS_WIDTH = True

    #: The covariate coefficients (``params`` and ``coef`` are the same
    #: array), in the accelerated-failure convention.
    beta: npt.NDArray
    params: npt.NDArray
    coef: npt.NDArray
    #: The residual Kaplan-Meier the predictions read: the sorted
    #: residuals and the survival at each.
    _resid: npt.NDArray
    _resid_surv: npt.NDArray
    #: The iterations the fit took, and whether it converged.
    n_iter: int
    converged: bool
    #: The Buckley-James estimator solves its estimating equations by
    #: iterated least squares rather than maximising a likelihood: ``"not
    #: applicable"`` (one of ``MAXIMUM_STATES``,
    #: ``surpyval.utils.no_maximum``); ``converged`` says how it ended.
    maximum: str = "not applicable"
    #: The fitted ``(Y, delta, Z, w)``, for ``bootstrap_ci`` and
    #: ``concordance``; ``None`` when not kept.
    _data: "tuple | None"

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: each covariate's
        column (a formula, ``fit_from_df`` or a DataFrame ``Z``), else
        ``coef_0``, ``coef_1``, ... (#614), as in the parametric
        regression models."""
        return coefficient_names(self, len(self.params))

    def __init__(
        self,
        beta: npt.ArrayLike,
        resid: npt.NDArray,
        resid_surv: npt.NDArray,
        n_iter: int,
        converged: bool,
        data: "tuple | None",
    ) -> None:
        self.beta = np.asarray(beta, dtype=float)
        self.params = self.beta
        self.coef = self.beta
        self._resid = resid
        self._resid_surv = resid_surv
        self.n_iter = n_iter
        self.converged = converged
        self._data = data  # (Y, delta, Z, w) for the bootstrap

    _ALIASED_WHY = (
        "a constant column, which is the intercept the fit profiles out, "
        "or a linear combination of the others"
    )
    _NO_LIKELIHOOD_WHY = (
        "the Buckley-James estimator iterates least squares on imputed "
        "log times, with the residual distribution left unspecified; "
        "there is no likelihood to maximise"
    )

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        return -(Z_arr.reshape(x.size, -1) @ self._coef())

    def _concordance_data(self) -> "tuple | None":
        if self._data is None:
            return None
        Y, delta, Z, w = self._data
        return np.exp(Y), (delta == 0).astype(int), w, Z

    def _resid_sf(self, r: npt.NDArray) -> npt.NDArray:
        # Right-continuous residual survival at query points ``r``. A
        # residual within RESID_ROUNDING of a step is at the step: a time
        # is mapped to its residual by ``log t + beta'Z``, and ``qf``'s time
        # ``exp(r_k - beta'Z)`` mapped back lands an ulp either side of
        # ``r_k``, depending on how numpy's exp and log round (its AVX2 and
        # AVX-512 kernels differ). Landing below it, ``ff(qf(p))`` was the
        # step before p for one random query in ten, and on CI's runners
        # in #662's test. 1e-12 on the log scale is 1e-12 relative in time.
        tol = RESID_ROUNDING * np.maximum(np.abs(r), 1.0)
        idx = np.searchsorted(self._resid, r + tol, side="right") - 1
        out = np.where(
            idx < 0,
            1.0,
            self._resid_surv[np.clip(idx, 0, len(self._resid_surv) - 1)],
        )
        return out

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
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
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "BuckleyJamesModel":
        """
        Rebuild a Buckley-James model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "BuckleyJamesModel", "a Buckley-James model"
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
        restore_covariate_meta(out, model_dict)
        return out

    def _linear_predictor(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        """``beta'Z`` for each time in ``x``: ``Z`` is one covariate vector
        (used at every time) or one row per time, paired in the order
        given, as for the other regression models (#426)."""
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        p = self.beta.size
        if Z_arr.ndim == 0:
            Z_arr = Z_arr.reshape(1)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(1, -1)
        if Z_arr.ndim != 2 or Z_arr.shape[1] != p:
            raise ValueError(
                "Z must be one covariate vector of length {} or one such "
                "row per time; got an array of shape {}.".format(
                    p, np.shape(Z)
                )
            )
        if Z_arr.shape[0] not in (1, x.size):
            raise ValueError(
                "Z has {} covariate rows but there are {} times; give one "
                "covariate vector, or one row per time.".format(
                    Z_arr.shape[0], x.size
                )
            )
        # An aliased coefficient (nan, #476) is predicted with as 0.
        return Z_arr @ np.where(np.isnan(self.beta), 0.0, self.beta)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """Survival ``P(T > x | Z) = S_eps(log x + beta'Z)``; ``Z`` is one
        covariate vector (used at every time) or one row per time in
        ``x``, paired in the order given."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        # beta is the accelerated-failure (negated) slope, so the residual
        # r = log t - gamma'Z = log t + beta'Z. At and below time 0 nothing
        # has failed: survival 1 (log(0) = -inf gives that already, but
        # warned, and a negative time gave nan).
        positive = x > 0
        with np.errstate(divide="ignore"):
            r = np.log(np.where(positive, x, 1.0)) + self._linear_predictor(
                x, Z
            )
        # A missing covariate (a DataFrame row with a nan) gives nan, as in
        # the other families; the residual lookup read it as the last step
        # (survival 0). So does a missing time, which ``positive`` read as
        # "not after time 0" (survival 1).
        out = np.where(positive, self._resid_sf(r), 1.0)
        return np.where(np.isnan(r) | np.isnan(x), np.nan, out)

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """Failure probability ``1 - sf(x, Z)``; ``Z`` as for
        :meth:`sf`."""
        return 1.0 - self.sf(x, Z)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """Cumulative hazard ``-log sf(x, Z)``; ``Z`` as for
        :meth:`sf`."""
        with np.errstate(divide="ignore"):
            return -np.log(self.sf(x, Z))

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: npt.ArrayLike) -> npt.NDArray:
        """
        The quantile function: the first time at which the predicted
        failure probability ``ff(x, Z)`` reaches ``p`` (#662), ``nan``
        where it never does -- the residual Kaplan-Meier stops at the
        last residual, above ``1 - p`` where the data end censored. It is
        :math:`e^{q_\\epsilon(p) - \\beta' Z}`, :math:`q_\\epsilon` the
        residual Kaplan-Meier's quantile, taken as the non-parametric
        ``qf`` takes it (a curve within ``1e-9`` of ``p`` reaches it).
        ``Z`` is paired with ``p`` as :meth:`sf` pairs it with ``x``. A
        probability outside [0, 1] gives ``nan``, with a warning.

        Examples
        --------
        >>> from surpyval import BuckleyJames
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = BuckleyJames.fit(x, df[["fin", "age", "prio"]].values, c=c)
        >>> rows = [[0, 25, 3], [1, 25, 3], [0, 25, 10]]
        >>> b10 = model.qf(0.1, rows)
        >>> b10.round(2)
        array([21.71, 28.33, 14.38])
        >>> model.ff(b10, rows).round(3)
        array([0.102, 0.102, 0.102])
        """
        rows = covariate_rows(
            np.asarray(self._prepare_Z(Z), dtype=float), self.beta.size
        )
        u, rows, _ = paired_probabilities(p, rows)
        lp = self._linear_predictor(np.empty(u.size), rows)
        resid = step_quantiles(
            (1.0 - np.asarray(self._resid_surv, dtype=float))[None, :],
            self._resid,
            u,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            return np.exp(resid - lp)

    def summary(
        self,
        alpha_ci: float = 0.05,
        n_boot: "int | None" = None,
        random_state: Any = None,
    ) -> "pd.DataFrame":
        """
        The coefficient table (#662), in the layout of ``CoxPH``'s
        :meth:`summary`: each coefficient (in the accelerated-failure
        convention, as ``WeibullAFT``'s) and ``exp(coef)``, the factor by
        which a unit of the covariate shortens the life. Buckley-James has
        no closed-form standard error, so ``se(coef)``, ``z`` and ``p``
        are ``nan``; with ``n_boot`` the intervals are the percentile
        bootstrap intervals of :meth:`bootstrap_ci` (``n_boot`` refits,
        seeded by ``random_state``), else ``nan``.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.
        n_boot : int, optional
            The number of bootstrap refits for the intervals; ``None``
            (the default) gives none.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the bootstrap.

        Returns
        -------
        pandas.DataFrame
            One row per covariate, with the columns of ``CoxPH``'s
            :meth:`summary`.

        Examples
        --------
        >>> from surpyval import BuckleyJames
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = BuckleyJames.fit(x, df[["fin", "age", "prio"]].values, c=c)
        >>> table = model.summary(n_boot=50, random_state=1)
        >>> list(table.index)
        ['coef_0', 'coef_1', 'coef_2']
        >>> bool((table["coef lower 95%"] < table["coef"]).all())
        True
        """
        beta = np.asarray(self.beta, dtype=float)
        nan = np.full(beta.shape, np.nan)
        table = coefficient_table(
            self.parameter_names, beta, nan, alpha_ci, p=nan
        )
        if n_boot is not None:
            bounds = self.bootstrap_ci(alpha_ci, n_boot, random_state)
            level = "{:g}%".format(100 * (1 - alpha_ci))
            lower, upper = bounds[:, 0], bounds[:, 1]
            table["coef lower " + level] = lower
            table["coef upper " + level] = upper
            table["exp(coef) lower " + level] = np.exp(lower)
            table["exp(coef) upper " + level] = np.exp(upper)
        return table

    def bootstrap_ci(
        self,
        alpha_ci: float = 0.05,
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap confidence intervals for the coefficients.

        Buckley-James has no simple closed-form standard error, so uncertainty
        is obtained by resampling observations with replacement, refitting, and
        taking percentiles of the coefficient distribution. Counts ``n`` are
        frequency weights, so the observations resampled are the rows
        expanded by their counts: the bounds are those of the data written
        out one row per observation.

        Parameters
        ----------
        alpha_ci : float, optional
            One minus the confidence level of the intervals. Default 0.05.
        n_boot : int, optional
            The number of bootstrap resamples. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the resampling. ``None`` (the default) draws from
            numpy's global generator, so ``np.random.seed`` reproduces it.

        Returns
        -------
        numpy.ndarray
            An ``(n_coef, 2)`` array of ``[lower, upper]`` bounds.
        """
        check_alpha_ci(alpha_ci)
        if self._data is None:
            raise ValueError(
                "bootstrap_ci needs the fit data, which this model does not "
                "carry"
            )
        Y, delta, Z, w = self._data
        p = self.beta.size
        # The aliased columns (#476) are left out of every refit.
        kept = np.flatnonzero(~np.isnan(self.beta))
        Z = Z[:, kept]
        rng = as_generator(random_state)
        # The counts ``w`` are frequency weights: a row with count 3 is
        # three observations, as the fit itself treats it. The bootstrap
        # therefore resamples the *observations* -- the rows expanded by
        # their counts -- rather than the rows, which treated each count as
        # one cluster and gave intervals too wide for the data. (With unit
        # counts the two are the same draw.)
        units = np.repeat(np.arange(Y.shape[0]), np.round(w).astype(int))
        whole = np.allclose(w, np.round(w)) and units.size > 0
        boot = []
        for _ in range(n_boot):
            if whole:
                idx = units[rng.integers(0, units.size, size=units.size)]
                w_b = np.ones(idx.size)
            else:
                # Fractional weights have no expansion; draw new counts in
                # proportion to them instead.
                counts = rng.multinomial(Y.shape[0], w / w.sum())
                idx = np.flatnonzero(counts)
                w_b = w[idx] * counts[idx]
            try:
                g, _, _ = _fit_beta(Y[idx], delta[idx], Z[idx], w_b, 1e-5, 100)
                # Report in the accelerated-failure sign.
                boot.append(expand(-g, kept, p))
            except np.linalg.LinAlgError:
                continue
        return percentile_bounds(boot, alpha_ci)

    def __repr__(self) -> str:
        lines = [
            "Buckley-James AFT SurPyval Model",
            "================================",
            "Kind                : Semi-Parametric AFT",
            f"Converged           : {self.converged} ({self.n_iter} iters)",
        ]
        if self._data is not None:
            # The data line (#508); the fit keeps the event flag, 1 for a
            # failure, and the counts; Y is the log time, whose distinct
            # values are the distinct times.
            Y, delta, _, w = self._data
            lines.append(
                "Data                : "
                + data_summary(1 - np.asarray(delta, dtype=int), w, x=Y)
            )
        lines += [
            "Coefficients (positive => accelerates failure):",
        ]
        for nm, b in zip(self.parameter_names, self.beta):
            lines.append(f"   {nm:>10}  :  {b: .6f}")
        return "\n".join(lines)


class BuckleyJames_(FitterRepr):
    """
    The Buckley-James semi-parametric accelerated failure time estimator:
    a least-squares regression of :math:`\\log x` on the covariates in
    which each right-censored time is replaced by its conditional
    expectation under the Kaplan-Meier estimate of the residual
    distribution, iterated to convergence. No baseline distribution is
    assumed.

    Coefficients are reported with the package's AFT sign: a *positive*
    coefficient shortens life (the textbook ``log T = gamma'Z + eps``
    slope is ``-beta``). ``BuckleyJames`` is an instance of this class;
    its ``fit`` returns a
    :class:`~surpyval.univariate.regression.buckley_james.buckley_james.BuckleyJamesModel`.
    """

    #: The ``repr`` (#614)
    fitter_kind = "semi-parametric accelerated failure time fitter"

    @dataframe_covariates
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: "npt.ArrayLike | None" = None,
        n: "npt.ArrayLike | None" = None,
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> BuckleyJamesModel:
        """
        Fit the Buckley-James AFT model.

        Rows with a missing or infinite covariate are dropped, with a
        warning. A column that is constant across the observations (or a
        single observation) cannot be separated from the intercept, nor
        one that is a linear combination of the others from them: such a
        column is aliased, as in :class:`~surpyval.CoxPH`. Its coefficient
        is ``nan`` (``model.aliased`` lists it), the others are those of
        the fit without it, predictions take it as 0, and one warning
        names it.

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
            Counts per row (frequency weights). Defaults to 1.
        tol : float, optional
            Convergence tolerance on the coefficient step. Default 1e-5.
        max_iter : int, optional
            Maximum Buckley-James iterations. Default 100.

        Returns
        -------
        BuckleyJamesModel
            The fitted model. A warning is raised if the iteration did not
            converge within ``max_iter``.

        Examples
        --------
        Log-life falls by 0.5 per unit of the covariate; follow-up ends at
        12:

        >>> import numpy as np
        >>> from surpyval import BuckleyJames
        >>> rng = np.random.default_rng(2)
        >>> Z = rng.normal(size=(100, 1))
        >>> t = np.exp(2.0 - 0.5 * Z[:, 0] + rng.normal(0, 0.5, 100))
        >>> c = (t > 12).astype(int)
        >>> x = np.minimum(t, 12)
        >>> model = BuckleyJames.fit(x, Z, c=c)
        >>> model.beta.round(3)
        array([0.435])
        >>> model.bootstrap_ci(random_state=1).round(3)
        array([[0.33 , 0.541]])
        >>> model.sf([5, 10], [0.0]).round(4)
        array([0.7366, 0.2693])
        """
        x_a, c_a, n_a, _, Z_a = semi_parametric_inputs(
            x,
            Z,
            c,
            n,
            censoring=(
                "Buckley-James supports only observed (c=0) and "
                "right-censored (c=1) data."
            ),
        )

        if np.any(x_a <= 0):
            raise ValueError(
                "Buckley-James models log(time); all times must be positive."
            )
        if not np.any((c_a == 0) & (n_a > 0)):
            # It reported converged=True with the least-squares slope of
            # the censoring times (#648).
            raise ValueError(
                "BuckleyJames needs at least one event (c=0); with every "
                "observation censored there is no residual distribution to "
                "estimate."
            )
        p = Z_a.shape[1]
        aliased = _aliased(Z_a, n_a)
        kept = np.setdiff1d(np.arange(p), aliased)
        if aliased.size:
            warn_aliased(
                aliased,
                "they are constant (the intercept, which the least-squares "
                "step profiles out) or a linear combination of the other "
                "columns",
            )
        Z_k = Z_a[:, kept]

        Y = np.log(x_a)
        delta = (c_a == 0).astype(float)
        # gamma is the textbook ``log T = gamma'Z + eps`` slope; report its
        # negative so a positive coefficient accelerates failure.
        if kept.size:
            gamma, n_iter, converged = _fit_beta(
                Y, delta, Z_k, n_a, tol, max_iter
            )
        else:
            # Every column aliased: nothing to iterate.
            gamma, n_iter, converged = np.zeros(0), 0, True
        if not converged:
            warnings.warn(
                "Buckley-James did not converge in {} iterations; returning "
                "the last iterate.".format(max_iter)
            )

        # Final residual distribution used for prediction.
        resid, resid_surv, _ = _residual_km(Y - Z_k @ gamma, delta, n_a)

        return BuckleyJamesModel(
            expand(-gamma, kept, p),
            resid,
            resid_surv,
            n_iter,
            converged,
            (Y, delta, Z_a, n_a),
        )

    @column_arguments("x", "c", "n")
    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        formula: "str | None" = None,
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> BuckleyJamesModel:
        """
        Fit a Buckley-James model from a pandas DataFrame. See :meth:`fit` for
        the estimator; ``Z_cols`` or ``formula`` selects the covariates.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col : str
            The column of times.
        Z_cols : str or list of str, optional
            The covariate columns. Give either this or ``formula``.
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        formula : str, optional
            A formula (formulaic syntax) for the covariates.
        tol, max_iter : optional
            As for :meth:`fit`.

        Returns
        -------
        BuckleyJamesModel
            The fitted model, which keeps the covariate names (or formula)
            so it predicts from DataFrame rows.
        """
        check_columns(df, x_col=x_col, c_col=c_col, n_col=n_col)
        Z, feature_names, model_spec = design_matrix_from_df(
            df, Z_cols, formula
        )
        mask = finite_covariate_mask(Z)
        Z = Z[mask]
        sub = df.loc[mask]
        x = sub[x_col].values
        c = sub[c_col].values if c_col is not None else None
        n = sub[n_col].values if n_col is not None else None

        # The aliasing warning (#476) names the columns.
        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(x, Z, c=c, n=n, tol=tol, max_iter=max_iter)
        model.formula = formula
        model.feature_names = feature_names
        model._model_spec = model_spec
        return model


BuckleyJames = BuckleyJames_()
