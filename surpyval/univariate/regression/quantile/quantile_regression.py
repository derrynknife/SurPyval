r"""
Censored quantile regression: covariate effects on the quantiles of the
time to an event, with no proportional hazards or accelerated failure time
assumption.

The model is linear in the covariates for each quantile of the log time,

.. math::
    Q_{\log T}(\tau \mid Z) = \beta_0(\tau) + \beta(\tau)' Z,
    \qquad Q_T(\tau \mid Z) = e^{\beta_0(\tau) + \beta(\tau)' Z},

with a coefficient vector of its own at every level :math:`\tau`, so a
covariate can stretch the early failures (the B10 life, :math:`\tau =
0.1`) and leave the median alone, or the other way round. An accelerated
failure time model is the special case in which every slope is the same at
every level, and a proportional hazards model gives slopes that vary with
:math:`\tau` in one particular way; this model assumes neither. A
coefficient :math:`\beta_k(\tau)` is the change in the log of the
:math:`\tau` quantile per unit of covariate :math:`k`, so
:math:`e^{\beta_k(\tau)}` is the factor by which a unit of it multiplies
that quantile: above 1 it lengthens the life.

Right-censored lives are taken by one of two estimators (see
:mod:`surpyval.univariate.regression.quantile._process`), each fitted on a
grid of levels from the bottom up, as R's ``quantreg::crq`` fits them:
Peng and Huang's (2008) martingale estimating equations (the default), and
Portnoy's (2003) recursively reweighted quantile regression. Neither has a
closed-form standard error; the intervals are bootstrap intervals,
resampling the units.

The fitted coefficient process is a step function of :math:`\tau`:
:math:`\beta(\tau) = \beta(\tau_j)` for :math:`\tau_{j-1} < \tau \le
\tau_j`. A unit's predicted distribution is read from its quantiles at the
grid levels, rearranged into increasing order where they cross
(Chernozhukov, Fernandez-Val and Galichon, 2010), so ``ff``, ``sf``,
``Hf`` and ``qf`` are those of one proper step distribution, which stops at
the last level the data determine: past it ``ff`` holds that level and
``qf`` is ``nan``.
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
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import bootstrap_quantile
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option
from surpyval.utils.warnings import ignoring_warnings

from .._aliasing import (
    aliased_columns,
    constant_columns,
    dataframe_covariates,
    warn_aliased,
)
from .._bootstrap import check_n_boot
from .._prediction import STEP_QF_TOL, paired_probabilities
from .._summary import coefficient_names, coefficient_table
from ..regression_data import (
    FrameDesign,
    NoLikelihoodMixin,
    canonical_rows,
    check_covariate_width,
    prepare_Z,
    restore_covariate_meta,
    semi_parametric_inputs,
    serialise_covariate_meta,
)
from ._process import METHODS, fit_process

#: The functions ``cb`` bounds ('R' and 'F' are aliases of 'sf' and 'ff').
_CB_ON = ("sf", "R", "ff", "F", "Hf")
_ON_ALIASES = {"R": "sf", "F": "ff"}
#: How close a level must be to a grid level to be read as it.
_TAU_TOL = 1e-9
#: The names of the estimators in a printout.
_METHOD_NAMES = {"peng-huang": "Peng-Huang", "portnoy": "Portnoy"}


def _grid(step: float) -> npt.NDArray:
    """The levels ``step, 2 step, ...`` below 1."""
    count = int(np.floor((1.0 - _TAU_TOL) / step))
    return np.round(np.arange(1, count + 1) * step, 12)


def _check_step(step: Any) -> float:
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, float, np.integer, np.floating))
        or not 0 < step <= 0.5
    ):
        raise ValueError(
            "'step' must be a number in (0, 0.5], the spacing of the grid "
            "of levels the quantile process is fitted on (0.01 fits "
            "0.01, 0.02, ..., 0.99); got {!r}".format(step)
        )
    return float(step)


def _aliased(Z: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """The columns of ``Z`` the fit cannot determine: a constant column,
    which the intercept already is, or one that is a linear combination of
    the others (the centred Gram matrix is singular in its direction)."""
    Zc = Z - (n[:, None] * Z).sum(axis=0) / n.sum()
    gram = (n[:, None] * Zc).T @ Zc
    return aliased_columns(gram, Z.shape[0], constant_columns(Z))


def _merged_rows(
    Y: npt.NDArray, delta: npt.NDArray, Z: npt.NDArray, n: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """The rows in one order (every column sorted), identical rows merged
    and their counts summed: the same rows whatever order they are given
    in, and a count ``k`` the same as ``k`` repeated rows."""
    order = canonical_rows(Y, delta, n, np.zeros(Y.size), Z)
    Y, delta, Z, n = Y[order], delta[order], Z[order], n[order]
    keys = np.column_stack([Y, delta, Z])
    new = np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)]
    starts = np.flatnonzero(new)
    return Y[new], delta[new], Z[new], np.add.reduceat(n, starts)


def _percentile(draws: npt.NDArray, alpha_ci: float, bound: str) -> Any:
    """The percentile bootstrap bound of each column of ``draws`` (one
    resample a row), leaving out the ``nan`` resamples."""
    with ignoring_warnings(RuntimeWarning):
        # (a column with no resample left is nan, said by the caller)
        if bound == "lower":
            return bootstrap_quantile(draws, alpha_ci, nan_skip=True)
        if bound == "upper":
            return bootstrap_quantile(draws, 1.0 - alpha_ci, nan_skip=True)
        lo = bootstrap_quantile(draws, alpha_ci / 2.0, nan_skip=True)
        hi = bootstrap_quantile(draws, 1.0 - alpha_ci / 2.0, nan_skip=True)
    return np.stack([lo, hi], axis=-1)


class _Process:
    """One fitted quantile process: the levels, the intercepts and the
    slopes (one row per level), and the predictions they give."""

    def __init__(
        self, taus: npt.NDArray, intercepts: npt.NDArray, coefs: npt.NDArray
    ) -> None:
        self.taus = taus
        self.intercepts = intercepts
        # An aliased coefficient (nan) is predicted with as 0.
        self.coefs = np.where(np.isnan(coefs), 0.0, coefs)
        self.dtau = np.diff(np.concatenate([[0.0], taus]))

    def quantiles(self, rows: npt.NDArray) -> npt.NDArray:
        """The quantile of each row at each level, ``(len(rows), L)``;
        summed term by term, so a row gives the same values whatever rows
        are beside it."""
        lp = np.broadcast_to(self.intercepts, (rows.shape[0], self.taus.size))
        for k in range(rows.shape[1]):
            lp = lp + rows[:, k, None] * self.coefs[None, :, k]
        with np.errstate(over="ignore"):
            return np.exp(lp)

    def ff(self, x: npt.NDArray, rows: npt.NDArray) -> npt.NDArray:
        """The failure probability at each time, paired with its row."""
        q = self.quantiles(rows)
        with np.errstate(invalid="ignore"):
            F = (q <= x[:, None]) @ self.dtau
        F = np.where(x > 0, F, 0.0)
        missing = np.isnan(x) | np.isnan(q).any(axis=1)
        return np.where(missing, np.nan, F)

    def qf(self, u: npt.NDArray, rows: npt.NDArray) -> npt.NDArray:
        """The quantile at each probability, paired with its row: the
        first of the row's rearranged quantiles at which the probability
        is reached, ``nan`` past the last level."""
        q = self.quantiles(rows)
        order = np.argsort(q, axis=1, kind="stable")
        ordered = np.take_along_axis(q, order, axis=1)
        reached = np.cumsum(self.dtau[order], axis=1)
        target = np.maximum(u - STEP_QF_TOL, np.finfo(float).tiny)
        with np.errstate(invalid="ignore"):
            hit = reached >= target[:, None]
        first = np.argmax(hit, axis=1)
        out = ordered[np.arange(u.size), first]
        out = np.where(hit.any(axis=1), out, np.nan)
        missing = np.isnan(u) | np.isnan(q).any(axis=1)
        return np.where(missing, np.nan, out)


class QuantileRegressionModel(NoLikelihoodMixin, SerialisableMixin):
    r"""
    A fitted censored quantile regression: the coefficient process
    :math:`\beta_0(\tau), \beta(\tau)` at the levels of a grid, from
    :func:`QuantileRegression.fit
    <surpyval.univariate.regression.quantile.quantile_regression.QuantileRegression_.fit>`.

    ``taus`` are the levels fitted, ``intercepts`` the intercept at each and
    ``coefs`` the covariate coefficients, one row per level and one column
    per covariate. :meth:`coefficients` reads them at any level, and
    :meth:`summary` tabulates them with bootstrap intervals. A coefficient
    is the change in the log of that quantile per unit of the covariate:
    ``exp(coef)`` is the factor by which a unit multiplies the quantile.

    Examples
    --------
    On the Rossi recidivism data (``arrest`` is 1 for an arrest, so the
    censoring flag is ``1 - arrest``), the tenth percentile of the time to
    arrest -- the B10 life -- by financial aid (``fin``), age and prior
    convictions (``prio``):

    >>> from surpyval import QuantileRegression
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> Z = df[["fin", "age", "prio"]].values
    >>> model = QuantileRegression.fit(x, Z, c=c)
    >>> model.coefficients(0.1).round(3)
    array([ 2.834,  0.32 ,  0.018, -0.071])
    >>> model.qf(0.1, [[0, 25, 3], [1, 25, 3]]).round(1)
    array([21.4, 29.5])
    """

    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None

    #: The levels fitted, from the grid's first to the last the data
    #: determine.
    taus: npt.NDArray
    #: The intercept at each level.
    intercepts: npt.NDArray
    #: The covariate coefficients, one row per level and one column per
    #: covariate (``nan`` for an aliased column).
    coefs: npt.NDArray
    #: The estimator: ``"peng-huang"`` or ``"portnoy"``.
    method: str
    #: The spacing of the grid of levels.
    step: float
    #: Censored quantile regression solves estimating equations (a linear
    #: programme at each level) rather than maximising a likelihood:
    #: ``"not applicable"`` (one of ``MAXIMUM_STATES``,
    #: ``surpyval.utils.no_maximum``).
    maximum: str = "not applicable"
    #: The process the predictions read.
    _process: _Process
    #: The bootstrap refits, ``((n_boot, seed), upto, refits)``, kept for a
    #: repeat call with the same integer seed; not pickled.
    _boot: tuple | None = None
    #: The fitted ``(Y, delta, Z, n)`` -- log times, event flags (1 for an
    #: event), covariates and counts -- for the bootstrap.
    _data: tuple | None

    _NO_LIKELIHOOD_WHY = (
        "censored quantile regression solves estimating equations, a "
        "linear programme at each quantile, with the distribution of the "
        "times left unspecified; there is no likelihood to maximise"
    )

    def __init__(
        self,
        taus: npt.ArrayLike,
        intercepts: npt.ArrayLike,
        coefs: npt.ArrayLike,
        method: str,
        step: float,
        data: tuple | None,
    ) -> None:
        self.taus = np.asarray(taus, dtype=float)
        self.intercepts = np.asarray(intercepts, dtype=float)
        self.coefs = np.asarray(coefs, dtype=float).reshape(self.taus.size, -1)
        self.method = method
        self.step = float(step)
        self._data = data
        self._process = _Process(self.taus, self.intercepts, self.coefs)

    def __getstate__(self) -> dict:
        # The bootstrap refits are rebuilt where a bound is asked for again.
        state = dict(self.__dict__)
        state.pop("_boot", None)
        return state

    # -- coefficients -------------------------------------------------------

    @property
    def tau_max(self) -> float:
        """The last level fitted: the highest quantile the data
        determine on the grid."""
        return float(self.taus[-1])

    @property
    def aliased(self) -> npt.NDArray:
        """The covariate columns whose coefficients the data cannot
        determine -- a constant column, which the intercept already is, or
        one that is a linear combination of the others -- and which the
        fit left out (their ``coefs`` are ``nan``; predictions take them as
        0)."""
        return np.flatnonzero(np.isnan(self.coefs).all(axis=0))

    def _names(self) -> list[str]:
        return coefficient_names(self, self.coefs.shape[1])

    def _level(self, tau: Any) -> npt.NDArray:
        """The grid index of each level in ``tau`` (the step function's:
        the first fitted level at or above it), refusing a level outside
        (0, 1) or past the last fitted."""
        t = np.asarray(tau, dtype=float)
        if np.any(~np.isfinite(t)) or np.any((t <= 0) | (t >= 1)):
            raise ValueError(
                "'tau' must be a level strictly between 0 and 1 (0.1 for "
                "the B10 life, 0.5 for the median); got {}".format(tau)
            )
        if np.any(t > self.tau_max + _TAU_TOL):
            raise ValueError(
                "'tau' = {} is past the last quantile the data determine, "
                "{:g}: above it the censoring leaves too few events. Ask "
                "for a level at or below {:g}.".format(
                    tau, self.tau_max, self.tau_max
                )
            )
        return np.searchsorted(self.taus, t - _TAU_TOL, side="left")

    def coefficients(self, tau: npt.ArrayLike) -> npt.NDArray:
        """
        The intercept and the covariate coefficients at level ``tau``: the
        fitted process at the first grid level at or above it.

        Parameters
        ----------
        tau : float or array like
            The level(s), in (0, 1) and at most :attr:`tau_max`.

        Returns
        -------
        numpy.ndarray
            ``[intercept, coef_0, coef_1, ...]`` for one level, or one such
            row per level.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 500)
        >>> t = np.exp(1.0 + 0.5 * z + (0.2 + 0.2 * z) * rng.normal(size=500))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> model.coefficients([0.1, 0.5, 0.9]).round(2)
        array([[0.73, 0.28],
               [1.01, 0.46],
               [1.17, 0.81]])
        """
        rows = np.column_stack([self.intercepts, self.coefs])
        return rows[self._level(tau)]

    # -- predictions --------------------------------------------------------

    def _prepare_Z(self, Z: Any) -> npt.NDArray:
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        check_covariate_width(Zp, self._names())
        return covariate_rows(Zp, self.coefs.shape[1])

    def _paired(
        self, x: npt.ArrayLike, Z: Any
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The times (flat) and their covariate rows, paired: one row for
        every time, or one time for every row."""
        x = np.asarray(x, dtype=float).reshape(-1)
        rows = self._prepare_Z(Z)
        check_paired_rows(x.size, rows.shape[0], grid=False)
        size = x.size if rows.shape[0] == 1 else rows.shape[0]
        x = np.broadcast_to(x, (size,))
        rows = np.broadcast_to(rows, (size, rows.shape[1]))
        return x, rows

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The probability of the event by ``x`` for covariates ``Z`` (one
        row for every time, or one row per time): the share of the levels
        whose quantile for that row is at or below ``x``. It rises in
        steps to :attr:`tau_max` and holds it after the last.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 500)
        >>> t = np.exp(1.0 + 0.5 * z + (0.2 + 0.2 * z) * rng.normal(size=500))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> model.ff([2.0, 4.0, 8.0], [0.5]).round(2)
        array([0.03, 0.71, 0.98])
        """
        x, rows = self._paired(x, Z)
        return self._process.ff(x, rows)

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """The survival ``1 - ff(x, Z)``; ``Z`` as for :meth:`ff`."""
        return 1.0 - self.ff(x, Z)

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """The cumulative hazard ``-log sf(x, Z)``; ``Z`` as for
        :meth:`ff`."""
        # 0.0 - log: where sf is 1, -log(1) is -0.0.
        return 0.0 - np.log(self.sf(x, Z))

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The time by which a share ``p`` of units with covariates ``Z``
        have had the event: ``qf(0.1, Z)`` is their B10 life. It is
        :math:`e^{\\beta_0(p) + \\beta(p)' Z}` where the quantiles at
        the grid levels are in order, and the rearranged quantile where
        they cross; ``nan`` past :attr:`tau_max`. ``Z`` is paired with
        ``p`` as :meth:`ff` pairs it with ``x``. A probability outside
        [0, 1] gives ``nan``, with a warning.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 500)
        >>> t = np.exp(1.0 + 0.5 * z + (0.2 + 0.2 * z) * rng.normal(size=500))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> model.qf([0.1, 0.5], [0.5]).round(2)
        array([2.38, 3.44])
        """
        rows = self._prepare_Z(Z)
        check_paired_rows(np.size(p), rows.shape[0], grid=False)
        u, rows, _ = paired_probabilities(p, rows)
        return self._process.qf(u, rows)

    # -- bootstrap ----------------------------------------------------------

    def _refits(
        self, n_boot: int, random_state: Any, upto: float
    ) -> list[_Process | None]:
        """The process refitted to ``n_boot`` resamples of the units, on
        the grid up to the level ``upto`` (1 for the whole process);
        ``None`` for a resample that cannot determine every coefficient.
        Kept for a repeat call with the same ``n_boot`` and integer seed."""
        if self._data is None:
            raise ValueError(
                "The bootstrap needs the fit data, which this model does "
                "not carry; refit it with QuantileRegression.fit."
            )
        n_boot = check_n_boot(n_boot)
        key = None
        if isinstance(random_state, (int, np.integer)) and not isinstance(
            random_state, bool
        ):
            key = (n_boot, int(random_state))
            if (
                self._boot is not None
                and self._boot[0] == key
                and self._boot[1] >= upto
            ):
                return self._boot[2]
        Y, delta, Z, n = self._data
        p = Z.shape[1]
        kept = np.flatnonzero(~np.isin(np.arange(p), self.aliased))
        grid = _grid(self.step)
        grid = grid[grid <= upto + _TAU_TOL]
        units = np.repeat(np.arange(Y.size), np.round(n).astype(int))
        rng = as_generator(random_state)
        out: list[_Process | None] = []
        for _ in range(n_boot):
            idx = units[rng.integers(0, units.size, size=units.size)]
            Yb, db, Zb, nb = _merged_rows(
                Y[idx], delta[idx], Z[idx][:, kept], np.ones(idx.size)
            )
            if not np.any(db == 1) or _aliased(Zb, nb).size:
                out.append(None)
                continue
            taus, a, b = fit_process(Yb, db, Zb, nb, grid, self.method)
            if taus.size == 0:
                out.append(None)
                continue
            coefs = np.zeros((taus.size, p))
            coefs[:, kept] = b
            out.append(_Process(taus, a, coefs))
        if key is not None:
            self._boot = (key, upto, out)
        return out

    @staticmethod
    def _warn_left_out(left_out: int, n_boot: int, what: str) -> None:
        if left_out:
            warnings.warn(
                "{} of {} bootstrap resamples could not estimate {} (a "
                "covariate constant in the resample, or too few events to "
                "reach the level) and were left out; the bounds are from "
                "the other {}. Use more data or a lower level for bounds "
                "from every resample.".format(
                    left_out, n_boot, what, n_boot - left_out
                ),
                stacklevel=3,
            )

    def bootstrap_ci(
        self,
        tau: float,
        alpha_ci: float = 0.05,
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap confidence intervals for the intercept and
        the coefficients at level ``tau``.

        The units are resampled with replacement (a row with count ``n``
        is ``n`` units), the process refitted on the grid up to ``tau``,
        and the intervals are the percentiles of the refitted
        coefficients. A resample whose process stops below ``tau``, or
        in which a covariate is constant, is left out, with a warning
        counting them.

        Parameters
        ----------
        tau : float
            The level, at most :attr:`tau_max`.
        alpha_ci : float, optional
            The intervals' total tail probability: 0.05 (the default) for
            95% two-sided intervals, 0.1 for 90%.
        n_boot : int, optional
            The number of resamples. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the resampling. ``None`` (the default) draws from
            numpy's global generator, so ``np.random.seed`` reproduces it.

        Returns
        -------
        numpy.ndarray
            An ``(1 + n_covariates, 2)`` array of ``[lower, upper]``, the
            intercept first.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 300)
        >>> t = np.exp(1.0 + 0.5 * z + 0.3 * rng.normal(size=300))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> ci = model.bootstrap_ci(0.5, 0.1, n_boot=50, random_state=1)
        >>> ci.round(2)
        array([[0.92, 1.05],
               [0.45, 0.57]])
        """
        check_alpha_ci(alpha_ci)
        self._level(tau)
        refits = self._refits(n_boot, random_state, float(tau))
        draws = np.full((n_boot, 1 + self.coefs.shape[1]), np.nan)
        for k, fit in enumerate(refits):
            if fit is not None and fit.taus[-1] >= tau - _TAU_TOL:
                j = np.searchsorted(fit.taus, tau - _TAU_TOL)
                draws[k] = np.r_[fit.intercepts[j], fit.coefs[j]]
        draws[:, 1:][:, self.aliased] = np.nan
        ok = ~np.isnan(draws[:, 0])
        self._warn_left_out(
            int(n_boot - ok.sum()), n_boot, f"the coefficients at {tau:g}"
        )
        bounds = _percentile(draws[ok], alpha_ci, "two-sided")
        return np.asarray(bounds, dtype=float).reshape(-1, 2)

    @keeps_query_shape
    def quantile_cb(
        self,
        p: npt.ArrayLike,
        Z: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap bounds on the quantile ``qf(p, Z)``: on the
        B10 life of units with covariates ``Z``, say, with
        ``p = 0.1``.

        The units are resampled as for :meth:`bootstrap_ci`, the whole
        process refitted, and each refit's quantile read as :meth:`qf`
        reads the model's; a resample whose process stops below a ``p``
        gives no quantile there and is left out of that bound, with a
        warning counting them. Calls with the same ``n_boot`` and integer
        ``random_state`` share one set of refits.

        Parameters
        ----------
        p : float or array like
            The probabilities.
        Z : array like or DataFrame
            The covariates, paired with ``p`` as :meth:`qf` pairs them.
        alpha_ci : float, optional
            The total tail probability of a two-sided interval, or the one
            tail of a one-sided bound. Default 0.05; a one-sided 90% lower
            bound on a B-life is ``alpha_ci=0.1, bound="lower"``.
        bound : ('two-sided', 'lower', 'upper'), optional
            Two-sided bounds put ``[lower, upper]`` on a last axis.
        n_boot : int, optional
            The number of resamples. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the resampling.

        Returns
        -------
        numpy.ndarray
            The bounds, with the shape of ``p`` (and a last axis of 2 for
            two-sided bounds).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 300)
        >>> t = np.exp(1.0 + 0.5 * z + 0.3 * rng.normal(size=300))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> model.qf(0.1, [1.0]).round(2)
        np.float64(3.17)
        >>> model.quantile_cb(0.1, [1.0], alpha_ci=0.1, bound="lower",
        ...                   n_boot=50, random_state=1).round(2)
        np.float64(3.03)
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        rows = self._prepare_Z(Z)
        check_paired_rows(np.size(p), rows.shape[0], grid=False)
        u, rows, _ = paired_probabilities(p, rows)
        draws = np.full((check_n_boot(n_boot), u.size), np.nan)
        # (a missing probability or covariate is nan, with no refit)
        wanted = ~np.isnan(u) & ~np.isnan(rows).any(axis=1)
        if wanted.any():
            refits = self._refits(n_boot, random_state, 1.0)
            for k, fit in enumerate(refits):
                if fit is not None:
                    draws[k, wanted] = fit.qf(u[wanted], rows[wanted])
            missing = np.isnan(draws[:, wanted]).any(axis=1)
            self._warn_left_out(
                int(missing.sum()), n_boot, "every quantile asked for"
            )
        out = _percentile(draws, alpha_ci, bound)
        if bound == "two-sided":
            return np.asarray(out, dtype=float).reshape(u.size, 2)
        return np.asarray(out, dtype=float).reshape(u.size)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Percentile bootstrap bounds on ``sf``, ``ff`` or ``Hf`` at ``x``
        for covariates ``Z``.

        The units are resampled as for :meth:`bootstrap_ci` and the whole
        process refitted; the bounds on ``sf`` are the percentiles of the
        refits' survival, and those on ``ff`` and ``Hf`` the same bounds
        transformed. Calls with the same ``n_boot`` and integer
        ``random_state`` share one set of refits.

        Parameters
        ----------
        x : float or array like
            The times.
        Z : array like or DataFrame
            The covariates, paired with ``x`` as :meth:`ff` pairs them.
        on : ('sf', 'ff', 'Hf'), optional
            The function to bound (``'R'`` and ``'F'`` are aliases of
            ``'sf'`` and ``'ff'``). Default ``'sf'``.
        alpha_ci : float, optional
            The total tail probability of a two-sided interval, or the one
            tail of a one-sided bound. Default 0.05.
        bound : ('two-sided', 'lower', 'upper'), optional
            Two-sided bounds put ``[lower, upper]`` on a last axis.
        n_boot : int, optional
            The number of resamples. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the resampling.

        Returns
        -------
        numpy.ndarray
            The bounds, with the shape of ``x`` (and a last axis of 2 for
            two-sided bounds).

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 300)
        >>> t = np.exp(1.0 + 0.5 * z + 0.3 * rng.normal(size=300))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> model.sf(4.0, [1.0]).round(3)
        np.float64(0.68)
        >>> model.cb(4.0, [1.0], n_boot=50, random_state=1).round(3)
        array([0.633, 0.752])
        """
        check_option("on", on, _CB_ON)
        on = _ON_ALIASES.get(on, on)
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        x, rows = self._paired(x, Z)
        refits = self._refits(n_boot, random_state, 1.0)
        draws = np.full((n_boot, x.size), np.nan)
        for k, fit in enumerate(refits):
            if fit is not None:
                draws[k] = 1.0 - fit.ff(x, rows)
        self._warn_left_out(
            sum(fit is None for fit in refits), n_boot, "the process"
        )
        if on == "sf":
            out = _percentile(draws, alpha_ci, bound)
        else:
            # ff and Hf fall as sf rises: the lower bound on either is
            # the upper bound on sf, transformed.
            side = {"lower": "upper", "upper": "lower"}.get(bound, bound)
            out = _percentile(draws, alpha_ci, side)
            if bound == "two-sided":
                out = out[..., ::-1]
            with np.errstate(divide="ignore"):
                out = 1.0 - out if on == "ff" else 0.0 - np.log(out)
        if bound == "two-sided":
            return np.asarray(out, dtype=float).reshape(x.size, 2)
        return np.asarray(out, dtype=float).reshape(x.size)

    def summary(
        self,
        tau: float = 0.5,
        *,
        alpha_ci: float = 0.05,
        n_boot: int | None = None,
        random_state: Any = None,
    ) -> pd.DataFrame:
        """
        The coefficient table at level ``tau``, in the layout of the other
        regression models' ``summary``: the intercept and each covariate's
        coefficient, and ``exp(coef)``, the factor by which a unit of the
        covariate multiplies the ``tau`` quantile (for the intercept, the
        quantile at ``Z = 0``). There is no closed-form standard error, so
        ``se(coef)``, ``z`` and ``p`` are ``nan``; with ``n_boot`` the
        intervals are those of :meth:`bootstrap_ci`, else ``nan``.

        Parameters
        ----------
        tau : float, optional
            The level. Default 0.5, the median.
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.
        n_boot : int, optional
            The number of resamples for the intervals; ``None`` (the
            default) gives none.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the bootstrap.

        Returns
        -------
        pandas.DataFrame
            One row for the intercept and one per covariate.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 300)
        >>> t = np.exp(1.0 + 0.5 * z + 0.3 * rng.normal(size=300))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> table = model.summary(0.1, alpha_ci=0.1, n_boot=50, random_state=1)
        >>> list(table.index)
        ['Intercept', 'coef_0']
        >>> table["coef"].round(2).tolist()
        [0.62, 0.53]
        """
        check_alpha_ci(alpha_ci)
        coef = self.coefficients(tau)
        nan = np.full(coef.shape, np.nan)
        table = coefficient_table(
            ["Intercept"] + self._names(), coef, nan, alpha_ci, p=nan
        )
        if n_boot is not None:
            bounds = self.bootstrap_ci(tau, alpha_ci, n_boot, random_state)
            level = f"{100 * (1 - alpha_ci):g}%"
            lower, upper = bounds[:, 0], bounds[:, 1]
            table["coef lower " + level] = lower
            table["coef upper " + level] = upper
            with np.errstate(over="ignore"):
                table["exp(coef) lower " + level] = np.exp(lower)
                table["exp(coef) upper " + level] = np.exp(upper)
        return table

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this model to a plain, JSON-serialisable dict: the
        coefficient process, the estimator and its grid, and the fit data
        (so the restored model still bootstraps).

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict[str, Any] = {
            "model": "QuantileRegressionModel",
            "method": self.method,
            "step": self.step,
            "taus": self.taus.tolist(),
            "intercepts": self.intercepts.tolist(),
            "coefs": self.coefs.tolist(),
        }
        if self._data is not None:
            Y, delta, Z, n = self._data
            out["data"] = {
                "Y": np.asarray(Y, dtype=float).tolist(),
                "delta": np.asarray(delta, dtype=float).tolist(),
                "Z": np.asarray(Z, dtype=float).tolist(),
                "n": np.asarray(n, dtype=float).tolist(),
            }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> QuantileRegressionModel:
        """
        Rebuild a censored quantile regression from a :meth:`to_dict`
        dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict,
            "QuantileRegressionModel",
            "a censored quantile regression",
        )
        check_option("method", model_dict["method"], METHODS)
        taus = np.array(model_dict["taus"], dtype=float)
        coefs = np.array(model_dict["coefs"], dtype=float)
        data = None
        if "data" in model_dict:
            d = model_dict["data"]
            Z = np.array(d["Z"], dtype=float).reshape(len(d["Y"]), -1)
            data = (
                np.array(d["Y"], dtype=float),
                np.array(d["delta"], dtype=float),
                Z,
                np.array(d["n"], dtype=float),
            )
        out = cls(
            taus,
            np.array(model_dict["intercepts"], dtype=float),
            coefs.reshape(taus.size, -1),
            str(model_dict["method"]),
            float(model_dict["step"]),
            data,
        )
        restore_covariate_meta(out, model_dict)
        return out

    def __repr__(self) -> str:
        lines = [
            "Censored Quantile Regression SurPyval Model",
            "===========================================",
            "Kind                : Semi-Parametric Quantile Regression",
            "Method              : {} (grid step {:g})".format(
                _METHOD_NAMES[self.method], self.step
            ),
            "Quantiles fitted    : {:g} to {:g}".format(
                self.taus[0], self.tau_max
            ),
        ]
        if self._data is not None:
            Y, delta, _, n = self._data
            lines.append(
                "Data                : "
                + data_summary(1 - np.asarray(delta, dtype=int), n, x=Y)
            )
        shown = [t for t in (0.1, 0.25, 0.5, 0.75, 0.9) if t <= self.tau_max]
        if shown:
            lines.append("Coefficients (log time) at each quantile:")
            names = ["Intercept"] + self._names()
            values = self.coefficients(shown).T
            lines.append(
                "    {:>12}".format("") + "".join(f"{t:>10g}" for t in shown)
            )
            for name, row in zip(names, values):
                lines.append(
                    f"    {name:>12}" + "".join(f"{v:>10.4f}" for v in row)
                )
        return "\n".join(lines)


class QuantileRegression_(FitterRepr):
    r"""
    Censored quantile regression: the quantiles of the time to an event,
    :math:`Q_T(\tau \mid Z) = e^{\beta_0(\tau) + \beta(\tau)' Z}`, with a
    coefficient vector for every level :math:`\tau` and no proportional
    hazards or accelerated failure time assumption, from right-censored
    data, by Peng and Huang's (2008) or Portnoy's (2003) estimator (R's
    ``quantreg::crq``). ``QuantileRegression`` is an instance of this
    class; its ``fit`` returns a
    :class:`~surpyval.univariate.regression.quantile.quantile_regression.QuantileRegressionModel`.
    """

    #: The ``repr``.
    fitter_kind = "semi-parametric censored quantile regression fitter"

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        method: str = "peng-huang",
        step: float = 0.01,
    ) -> QuantileRegressionModel:
        r"""
        Fit a censored quantile regression.

        The quantile process is fitted on the grid of levels ``step,
        2 step, ...``, from the bottom up, and stops at the last level the
        data determine (:attr:`QuantileRegressionModel.tau_max`): past the
        last event the censoring leaves nothing to fit. Rows with a missing
        or infinite covariate are dropped, with a warning. A covariate
        column that is constant (the intercept already is one), or a linear
        combination of the others, is aliased: its coefficients are
        ``nan``, the others those of the fit without it, and one warning
        names it. The rows are fitted sorted, with identical rows merged,
        so the fit is the same whatever order they are given in, and a
        count ``n = k`` is the same as ``k`` repeated rows.

        Parameters
        ----------
        x : array_like
            The (positive, finite) times.
        Z : array_like
            The covariate matrix, one row per observation.
        c : array_like, optional
            Censoring flags: 0 observed, 1 right censored. Left and
            interval censoring are not supported. Defaults to all
            observed.
        n : array_like, optional
            Counts per row. Defaults to 1.
        method : ('peng-huang', 'portnoy'), optional
            The estimator: Peng and Huang's (2008) martingale estimating
            equations (the default), or Portnoy's (2003) recursively
            reweighted quantile regression. With no censoring Portnoy's is
            ordinary quantile regression at each level; with no covariates
            it is the Kaplan-Meier quantile, and Peng and Huang's the
            quantile of the Nelson-Aalen survival, each up to the grid's
            resolution.
        step : float, optional
            The spacing of the grid of levels, in (0, 0.5]. Default 0.01,
            which fits 0.01, 0.02, ..., 0.99. A finer grid follows the
            estimator more closely (both are defined as the grid shrinks)
            and costs a linear programme per level.

        Returns
        -------
        QuantileRegressionModel
            The fitted model.

        Examples
        --------
        The times have a log-normal spread that grows with the covariate,
        so its effect on the early failures (the B10 life) is weaker than
        on the median; follow-up ends at 20:

        >>> import numpy as np
        >>> from surpyval import QuantileRegression
        >>> rng = np.random.default_rng(0)
        >>> z = rng.uniform(0, 2, 500)
        >>> t = np.exp(1.0 + 0.5 * z + (0.2 + 0.2 * z) * rng.normal(size=500))
        >>> c = (t > 20).astype(int)
        >>> model = QuantileRegression.fit(np.minimum(t, 20), z, c=c)
        >>> model.coefficients([0.1, 0.5]).round(2)
        array([[0.73, 0.28],
               [1.01, 0.46]])
        >>> model.qf(0.1, [[0.0], [2.0]]).round(2)
        array([2.07, 3.64])
        """
        check_option("method", method, METHODS)
        step = _check_step(step)
        x_a, c_a, n_a, _, Z_a = semi_parametric_inputs(
            x,
            Z,
            c,
            n,
            censoring=(
                "QuantileRegression supports only observed (c=0) and "
                "right-censored (c=1) data."
            ),
        )
        if np.any(~np.isfinite(x_a)) or np.any(x_a <= 0):
            raise ValueError(
                "QuantileRegression models log(time): every time in 'x' "
                "must be positive and finite (a unit still running at the "
                "end is right censored at its last time, c=1)."
            )
        delta = (c_a == 0).astype(float)
        Y, delta, Z_a, n_a = _merged_rows(np.log(x_a), delta, Z_a, n_a)
        if not np.any(delta == 1):
            raise ValueError(
                "QuantileRegression needs at least one event (c=0); with "
                "every observation censored no quantile can be estimated."
            )
        p = Z_a.shape[1]
        aliased = _aliased(Z_a, n_a)
        if aliased.size:
            warn_aliased(
                aliased,
                "they are constant (the intercept, which the model has "
                "already) or a linear combination of the other columns",
            )
        kept = np.setdiff1d(np.arange(p), aliased)
        taus, intercepts, slopes = fit_process(
            Y, delta, Z_a[:, kept], n_a, _grid(step), method
        )
        if taus.size == 0:
            raise ValueError(
                "QuantileRegression could not estimate even the first "
                "level of the grid ({:g}): too few events for the number "
                "of covariates. Use fewer covariates or a larger "
                "'step'.".format(step)
            )
        coefs = np.full((taus.size, p), np.nan)
        coefs[:, kept] = slopes
        return QuantileRegressionModel(
            taus, intercepts, coefs, method, step, (Y, delta, Z_a, n_a)
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
        method: str = "peng-huang",
        step: float = 0.01,
    ) -> QuantileRegressionModel:
        """
        Fit a censored quantile regression from a pandas DataFrame. See
        :meth:`fit` for the estimator; ``Z_cols`` or ``formula`` selects
        the covariates.

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
        method, step : optional
            As for :meth:`fit`.

        Returns
        -------
        QuantileRegressionModel
            The fitted model, which keeps the covariate names (or formula)
            so it predicts from DataFrame rows.

        Examples
        --------
        >>> from surpyval import QuantileRegression
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static().assign(c=lambda d: 1 - d["arrest"])
        >>> model = QuantileRegression.fit_from_df(
        ...     df, x_col="week", formula="fin + age + prio", c_col="c"
        ... )
        >>> model.summary(0.1)["coef"].round(3)
        covariate
        Intercept    2.834
        fin          0.320
        age          0.018
        prio        -0.071
        Name: coef, dtype: float64
        """
        check_columns(df, x_col=x_col, c_col=c_col, n_col=n_col)
        design = FrameDesign(df, Z_cols, formula, drop_missing=True)
        return design.fit(
            self.fit,
            design.column(x_col),
            design.Z,
            c=design.column(c_col),
            n=design.column(n_col),
            method=method,
            step=step,
        )


QuantileRegression = QuantileRegression_()
