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

from surpyval.serialisation import require_model_tag, stamp_schema
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import percentile_bounds
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

from .._aliasing import aliased_columns, constant_columns, expand, warn_aliased
from .._residual_aft import RESID_ROUNDING  # noqa: F401 (kept importable)
from .._residual_aft import ResidualAFTModel
from .._residual_aft import residual_km as _residual_km
from .._summary import coefficient_table
from ..regression_data import (
    FrameDesign,
    canonical_rows,
    restore_covariate_meta,
    semi_parametric_inputs,
    serialise_covariate_meta,
)


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


def aliased_design_columns(Z: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
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


class BuckleyJamesModel(ResidualAFTModel):
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

    feature_names: list[str] | None = None
    formula: str | None = None
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
    #: The fitted ``(Y, delta, Z, w)``, for ``param_cb`` and
    #: ``concordance``; ``None`` when not kept.
    _data: tuple | None

    def __init__(
        self,
        beta: npt.ArrayLike,
        resid: npt.NDArray,
        resid_surv: npt.NDArray,
        n_iter: int,
        converged: bool,
        data: tuple | None,
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

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted Buckley-James model to a plain, JSON-serialisable
        dict.

        Predictions use the residual Kaplan-Meier ``sf(t | Z) =
        S_eps(log t + beta'Z)``, so what is stored is the coefficients ``beta``
        and the residual survival step arrays (``_resid`` and ``_resid_surv``).
        The fit data ``(Y, delta, Z, w)`` is stored too, so the restored model
        can still run :meth:`param_cb`.

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
    def from_dict(cls, model_dict: dict) -> BuckleyJamesModel:
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

    def summary(
        self,
        alpha_ci: float = 0.05,
        n_boot: int | None = None,
        random_state: Any = None,
    ) -> pd.DataFrame:
        """
        The coefficient table (#662), in the layout of ``CoxPH``'s
        :meth:`summary`: each coefficient (in the accelerated-failure
        convention, as ``WeibullAFT``'s) and ``exp(coef)``, the factor by
        which a unit of the covariate shortens the life. Buckley-James has
        no closed-form standard error, so ``se(coef)``, ``z`` and ``p``
        are ``nan``; with ``n_boot`` the intervals are the percentile
        bootstrap intervals of :meth:`param_cb` (``n_boot`` refits,
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
            bounds = self.param_cb(
                None, alpha_ci, n_boot=n_boot, random_state=random_state
            )
            level = f"{100 * (1 - alpha_ci):g}%"
            lower, upper = bounds[:, 0], bounds[:, 1]
            table["coef lower " + level] = lower
            table["coef upper " + level] = upper
            table["exp(coef) lower " + level] = np.exp(lower)
            table["exp(coef) upper " + level] = np.exp(upper)
        return table

    def param_cb(
        self,
        name: str | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap confidence bounds on the coefficients.

        Buckley-James has no simple closed-form standard error, so the
        uncertainty is found by resampling the observations with
        replacement, refitting, and taking percentiles of the refitted
        coefficients. Counts ``n`` are frequency weights, so the
        observations resampled are the rows expanded by their counts: the
        bounds are those of the data written out one row per observation.
        (This was ``bootstrap_ci`` until v0.30.)

        Parameters
        ----------
        name : str, optional
            A coefficient, one of :attr:`parameter_names`. ``None`` (the
            default) bounds every coefficient.
        alpha_ci : float, optional
            One minus the confidence level of the intervals. Default 0.05.
        bound : ('two-sided', 'lower', 'upper'), optional
            A one-sided bound puts all of ``alpha_ci`` on one side, so it
            is that end of the two-sided interval at ``2 * alpha_ci``.
        n_boot : int, optional
            The number of bootstrap resamples. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the resampling. ``None`` (the default) draws from
            numpy's global generator, so ``np.random.seed`` reproduces it.

        Returns
        -------
        numpy.ndarray
            For every coefficient, an ``(n_coef, 2)`` array of ``[lower,
            upper]`` bounds (``(n_coef,)`` one-sided); for one, its
            ``[lower, upper]`` (or the one bound).
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        names = list(self.parameter_names)
        if name is not None and name not in names:
            raise ValueError(
                f"Unknown coefficient {name!r}; this model's coefficients "
                f"are {names}."
            )
        boot = self._bootstrap_draws(n_boot, random_state)
        out = np.asarray(percentile_bounds(boot, alpha_ci, bound))
        return out if name is None else out[names.index(name)]

    def bootstrap_ci(
        self,
        alpha_ci: float = 0.05,
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """Deprecated in v0.30, removed in v0.31: call
        ``param_cb(alpha_ci=..., n_boot=..., random_state=...)``, which
        gives the same bounds."""
        from surpyval.utils.deprecation import REMOVED_IN
        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            "BuckleyJamesModel.bootstrap_ci is deprecated and will be "
            f"removed in v{REMOVED_IN}; call param_cb(alpha_ci=..., "
            "n_boot=..., random_state=...), which gives the same bounds.",
            FutureWarning,
            stacklevel=caller_stacklevel(),
        )
        return self.param_cb(None, alpha_ci, "two-sided", n_boot, random_state)

    def _bootstrap_draws(self, n_boot: int, random_state: Any) -> list:
        """The coefficients of ``n_boot`` refits to resampled data."""
        if self._data is None:
            raise ValueError(
                "param_cb needs the fit data, which this model does not "
                "carry: its bounds are the bootstrap's, from refits to it."
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
        return boot

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
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
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
        names it. The rows are fitted sorted by every column (time,
        censoring, covariates, count), so the fit, and its bootstrap, is
        the same to the last digit whatever order they are given in.

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
        >>> model.param_cb(random_state=1).round(3)
        array([[0.342, 0.532]])
        >>> model.sf([5, 10], [0.0]).round(4)
        array([0.7366, 0.2693])
        """
        x_a, c_a, n_a, tl_a, Z_a = semi_parametric_inputs(
            x,
            Z,
            c,
            n,
            censoring=(
                "Buckley-James supports only observed (c=0) and "
                "right-censored (c=1) data."
            ),
        )
        # The rows in one order, every column sorted, so the fit (and its
        # bootstrap, which resamples them) is the same whatever order they
        # are given in (#760)
        order = canonical_rows(x_a, c_a, n_a, tl_a, Z_a)
        x_a, c_a, n_a, Z_a = (a[order] for a in (x_a, c_a, n_a, Z_a))

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
        aliased = aliased_design_columns(Z_a, n_a)
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

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
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
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        check_columns(df, x_col=x_col, c_col=c_col, n_col=n_col)
        design = FrameDesign(df, Z_cols, formula, drop_missing=True)
        return design.fit(
            self.fit,
            design.column(x_col),
            design.Z,
            c=design.column(c_col),
            n=design.column(n_col),
            tol=tol,
            max_iter=max_iter,
        )


BuckleyJames = BuckleyJames_()

# Temporary aliases for the names these helpers had before they became
# public. Remove them once the open branches that import the old names have
# landed.
_aliased = aliased_design_columns
