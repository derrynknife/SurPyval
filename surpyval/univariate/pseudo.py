"""Pseudo-observations and the regressions fitted to them.

A pseudo-observation turns a censored time into a number that can be
used as if it were a complete response. For a quantity that the whole
sample estimates, ``theta`` (the Kaplan-Meier survival at a time, the
restricted mean survival time, a cause's cumulative incidence), unit
``i``'s pseudo-value is the jackknife

    theta_i = N theta - (N - 1) theta_(-i),

with ``theta_(-i)`` the same estimate with unit ``i`` left out and ``N``
the number of units (Andersen, Klein and Rosthoj 2003; Andersen and Perme
2010). With independent censoring the pseudo-values have, given the
covariates, the expected value of the quantity, so a generalised linear
model fitted to them by estimating equations (GEE) is a regression on the
quantity itself: on the chance of the event by a time, on the restricted
mean, or on a cause's incidence. The standard errors are the robust
(sandwich) ones, as R's ``geepack::geeglm(..., corstr = "independence")``
gives.

Cost: each leave-one-out estimate is computed once per distinct row (the
same time, censoring and cause), as a product over the distinct event
times, so the work is ``O(G K)`` for ``G`` distinct rows and ``K`` distinct
event times, done in blocks of rows: about 0.5 s for 5,000 units with
distinct times (survival and RMST).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from surpyval.univariate.competing_risks.labels import (
    label_mask,
    ordered_labels,
)
from surpyval.utils.covariates import check_covariate_rows

#: The links each quantity takes, the first its default.
LINKS = {
    "survival": ("cloglog", "logit"),
    "rmst": ("identity", "log"),
    "cif": ("cloglog", "logit"),
}

#: Cells of the leave-one-out product computed at once (rows x times).
_BLOCK = 2_000_000


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------
def _refuse_truncation(tl: Any) -> None:
    if tl is None:
        return
    tl = np.asarray(tl, dtype=float)
    if np.any(np.isfinite(tl) & (tl > 0)):
        raise ValueError(
            "Pseudo-observations are not available with delayed entry "
            "('tl'): the leave-one-out estimate of a left-truncated sample "
            "does not give pseudo-values with the right expectation. Leave "
            "out 'tl', or use a regression that takes it (CoxPH, or a "
            "parametric PH/AFT model with 'tl')."
        )


def _times_censoring(
    x: npt.ArrayLike, c: npt.ArrayLike | None, n: npt.ArrayLike | None
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``x``, ``c`` and ``n`` as float arrays in the given row order, with
    only exact failures (0) and right censoring (1)."""
    from surpyval.utils.data_formats import xcnt_handler

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError(
            "Pseudo-observations take one time per row in 'x' (a 1-D "
            "array); interval-censored rows (two columns) are not "
            "supported."
        )
    xa, ca, na, _ = xcnt_handler(x_arr, c, n, group_and_sort=False)
    xa, ca, na = (np.asarray(a, dtype=float) for a in (xa, ca, na))
    if np.any((ca != 0) & (ca != 1)):
        raise ValueError(
            "Pseudo-observations need exact failures (c = 0) and right "
            "censoring (c = 1) only; 'c' has left (-1) or interval (2) "
            "censored rows. Drop them, or use Turnbull for such data."
        )
    if not np.all(np.isfinite(xa)) or np.any(xa < 0):
        raise ValueError("'x' must be finite and non-negative.")
    if na.sum() < 2:
        raise ValueError(
            "Pseudo-observations need at least two units (a leave-one-out "
            "estimate of one unit has no data)."
        )
    return xa, ca, na


def _as_times(t: Any, name: str) -> tuple[npt.NDArray, bool]:
    if t is None:
        raise ValueError(f"Give the time(s) '{name}' to evaluate at.")
    scalar = np.ndim(t) == 0
    arr = np.atleast_1d(np.asarray(t, dtype=float))
    if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"'{name}' must be finite, non-negative times.")
    return arr, scalar


def _check_within(times: npt.NDArray, x: npt.NDArray, name: str) -> None:
    if np.any(times > x.max()):
        raise ValueError(
            f"'{name}' = {times.max():g} is past the largest time in the "
            f"data ({x.max():g}); the estimate is not defined there. Use "
            f"times at or before {x.max():g}."
        )


# ---------------------------------------------------------------------------
# The leave-one-out product-limit estimates.
# ---------------------------------------------------------------------------
def _loo(
    x: npt.NDArray,
    d: npt.NDArray,
    n: npt.NDArray,
    k: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """The distinct event times ``u``, each row's group, and per group (and
    for the full sample, last row) the all-cause Kaplan-Meier survival at
    each ``u`` and, with causes ``k`` (the cause's index for a failure, -1
    otherwise), the cumulative incidence of each cause at each ``u``.

    Returns ``(u, group_of_row, S, CIF)``: ``S`` has shape
    ``(groups + 1, K)`` and ``CIF`` ``(causes, groups + 1, K)`` (empty
    without ``k``)."""
    u = np.unique(x[d == 1])
    K = u.size
    kk = np.zeros(x.size, dtype=int) if k is None else k
    keys = np.stack([x, d, kk.astype(float)], axis=1)
    groups, group_of_row = np.unique(keys, axis=0, return_inverse=True)
    group_of_row = np.asarray(group_of_row).reshape(-1)
    gx, gd, gk = groups[:, 0], groups[:, 1], groups[:, 2].astype(int)

    # The full sample's numbers at risk and events at each u.
    order = np.searchsorted(u, x)  # position of x among u
    # n at risk at u_j: the units with x >= u_j (a reverse cumsum).
    idx = np.searchsorted(u, x, side="right")  # u_j <= x for j < idx
    cnt = np.bincount(idx, weights=n, minlength=K + 1)
    r = np.cumsum(cnt[::-1])[::-1][1:]
    ev = d == 1
    dj = np.bincount(order[ev], weights=n[ev], minlength=K)[:K]
    n_causes = 0 if k is None else int(kk.max()) + 1
    dk = np.zeros((n_causes, K))
    for j in range(n_causes):
        sel = ev & (kk == j)
        dk[j] = np.bincount(order[sel], weights=n[sel], minlength=K)[:K]

    G = groups.shape[0]
    S = np.empty((G + 1, K))
    CIF = np.empty((n_causes, G + 1, K))
    block = max(1, _BLOCK // max(K, 1))
    for start in range(0, G + 1, block):
        stop = min(G + 1, start + block)
        rows = np.arange(start, stop)
        full = rows == G
        rows_g = np.minimum(rows, G - 1)
        # A group's unit is at risk at u_j <= x and fails at u_j == x.
        at_risk = (u[None, :] <= gx[rows_g, None]) & ~full[:, None]
        fails = (
            (u[None, :] == gx[rows_g, None])
            & (gd[rows_g, None] == 1)
            & ~full[:, None]
        )
        r_ = r[None, :] - at_risk
        d_ = dj[None, :] - fails
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(r_ > 0, d_ / r_, 0.0)
        surv = np.cumprod(1.0 - h, axis=1)
        S[start:stop] = surv
        if n_causes:
            before = np.concatenate(
                [np.ones((stop - start, 1)), surv[:, :-1]], axis=1
            )
            for j in range(n_causes):
                dkj = dk[j][None, :] - (fails & (gk[rows_g, None] == j))
                with np.errstate(divide="ignore", invalid="ignore"):
                    hk = np.where(r_ > 0, dkj / r_, 0.0)
                CIF[j, start:stop] = np.cumsum(before * hk, axis=1)
    return u, group_of_row, S, CIF


def _at(
    u: npt.NDArray, values: npt.NDArray, t: npt.NDArray, start: float
) -> npt.NDArray:
    """A right-continuous step function with ``values[..., j]`` from
    ``u[j]`` (``start`` before ``u[0]``), at the times ``t``."""
    idx = np.searchsorted(u, t, side="right") - 1
    padded = np.concatenate(
        [np.full(values.shape[:-1] + (1,), start), values], axis=-1
    )
    return padded[..., idx + 1]


def _rmst(u: npt.NDArray, S: npt.NDArray, tau: float) -> npt.NDArray:
    """The area under each row's step survival ``S`` (``1`` before
    ``u[0]``) from 0 to ``tau``."""
    keep = u < tau
    b = np.concatenate([[0.0], u[keep], [tau]])
    width = np.diff(b)
    level = np.concatenate([np.ones((S.shape[0], 1)), S[:, keep]], axis=1)
    return level @ width


def _jackknife(
    full: npt.NDArray, loo: npt.NDArray, groups: npt.NDArray, N: float
) -> npt.NDArray:
    return N * full[None, ...] - (N - 1) * loo[groups]


# ---------------------------------------------------------------------------
# The pseudo-values.
# ---------------------------------------------------------------------------
def pseudo_survival(
    x: npt.ArrayLike,
    t: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Jackknife pseudo-values of the Kaplan-Meier survival at times ``t``.

    Each unit's pseudo-value is ``N S(t) - (N - 1) S_(-i)(t)``, with
    ``S`` the Kaplan-Meier estimate of all ``N`` units and ``S_(-i)`` the
    estimate with unit ``i`` left out. A unit that failed before ``t``
    gets a value near 0, one known to survive past ``t`` a value near 1,
    and one censored before ``t`` a value in between (it can fall a little
    outside [0, 1]). Their mean is ``S(t)``. They are R's
    ``pseudo::pseudosurv``.

    Parameters
    ----------
    x : array_like
        The failure or censoring time of each row.
    t : float or array_like
        The time(s) at which to take the survival; none past the largest
        time in ``x``.
    c : array_like, optional
        Censoring flags: 0 an event (a failure, a death), 1
        right-censored (the default is every row an event). Left and
        interval censoring are refused.
    n : array_like, optional
        Counts: a row with ``n = 3`` is three identical units, each with
        the row's pseudo-value.
    tl : array_like, optional
        Delayed entry is refused (named only so that the refusal can say
        what to use instead).

    Returns
    -------
    numpy.ndarray
        One value per row of ``x``: shape ``(rows,)`` for a scalar ``t``,
        ``(rows, len(t))`` otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import pseudo_survival
    >>> x = [1, 2, 3, 4, 5]
    >>> c = [0, 1, 0, 0, 1]
    >>> pseudo_survival(x, 3, c).round(4)
    array([ 0.    ,  0.6667, -0.3333,  1.1667,  1.1667])
    >>> round(float(np.mean(pseudo_survival(x, 3, c))), 4)  # S(3)
    0.5333
    """
    _refuse_truncation(tl)
    xa, ca, na = _times_censoring(x, c, n)
    times, scalar = _as_times(t, "t")
    _check_within(times, xa, "t")
    d = 1.0 - ca
    u, groups, S, _ = _loo(xa, d, na)
    at = _at(u, S, times, 1.0)
    out = _jackknife(at[-1], at[:-1], groups, na.sum())
    return out[:, 0] if scalar else out


def pseudo_rmst(
    x: npt.ArrayLike,
    tau: float,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Jackknife pseudo-values of the restricted mean survival time to
    ``tau``.

    The restricted mean survival time (RMST) is the average time lived
    up to ``tau``: the area under the Kaplan-Meier curve from 0 to
    ``tau``. Each unit's pseudo-value is ``N m - (N - 1) m_(-i)``, ``m``
    the RMST of all ``N`` units and ``m_(-i)`` that with unit ``i`` left
    out; for a unit followed to ``tau`` or to its failure it is close to
    ``min(x, tau)``. Their mean is ``m``. They are R's
    ``pseudo::pseudomean``.

    Parameters
    ----------
    x : array_like
        The failure or censoring time of each row.
    tau : float
        The end of the window; not past the largest time in ``x``.
    c : array_like, optional
        Censoring flags: 0 an event, 1 right-censored.
    n : array_like, optional
        Counts of identical units.
    tl : array_like, optional
        Delayed entry is refused.

    Returns
    -------
    numpy.ndarray
        One value per row of ``x``.

    Examples
    --------
    >>> from surpyval import pseudo_rmst
    >>> pseudo_rmst([1, 2, 3, 4, 5], 4, [0, 1, 0, 0, 1]).round(4)
    array([1.    , 3.6667, 2.6667, 4.1667, 4.1667])
    """
    _refuse_truncation(tl)
    xa, ca, na = _times_censoring(x, c, n)
    if np.ndim(tau) != 0:
        raise ValueError("'tau' is one time, the end of the window.")
    times, _ = _as_times(tau, "tau")
    _check_within(times, xa, "tau")
    if not np.any((ca == 0) & (xa <= times[0])):
        raise ValueError(
            f"No failures at or before tau = {times[0]:g}; the restricted "
            "mean is tau for everyone. Choose a later 'tau'."
        )
    u, groups, S, _ = _loo(xa, 1.0 - ca, na)
    m = _rmst(u, S, float(times[0]))
    return _jackknife(m[-1], m[:-1], groups, na.sum())


def pseudo_cif(
    x: npt.ArrayLike,
    e: npt.ArrayLike,
    t: npt.ArrayLike,
    cause: Any,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """
    Jackknife pseudo-values of a cause's cumulative incidence at times
    ``t``.

    With competing causes of failure, the cumulative incidence of a cause
    is the chance of failing from it by ``t``: the Aalen-Johansen
    estimate, as :class:`~surpyval.univariate.competing_risks.\
CompetingRisks` gives it. Each unit's pseudo-value is
    ``N F_k(t) - (N - 1) F_k,(-i)(t)``. A unit that failed from the cause
    by ``t`` gets a value near 1, one that failed from another cause or
    survived past ``t`` a value near 0. They are R's
    ``pseudo::pseudoci``.

    Parameters
    ----------
    x : array_like
        The failure or censoring time of each row.
    e : array_like
        The cause of each failure (any labels); a missing value
        (``None``, ``NaN``) marks a right-censored row.
    t : float or array_like
        The time(s) at which to take the incidence; none past the largest
        time in ``x``.
    cause : label
        The cause whose incidence is wanted; one of the labels in ``e``.
    c : array_like, optional
        Censoring flags (0 an event with a cause, 1 right-censored);
        derived from ``e`` if not given.
    n : array_like, optional
        Counts of identical units.
    tl : array_like, optional
        Delayed entry is refused.

    Returns
    -------
    numpy.ndarray
        One value per row: shape ``(rows,)`` for a scalar ``t``,
        ``(rows, len(t))`` otherwise.

    Examples
    --------
    >>> from surpyval import pseudo_cif
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> e = ["a", "b", None, "a", "b", None]
    >>> pseudo_cif(x, e, 4, "a").round(4)
    array([ 1.    ,  0.    ,  0.3333,  1.3333, -0.1667, -0.1667])
    """
    from surpyval.utils.validation import validate_cr_inputs, validate_event

    _refuse_truncation(tl)
    if np.ndim(x) != 1:
        raise ValueError("'x' must be a 1-D array of times.")
    xa, ca, na, ea = validate_cr_inputs(x, c, n, e, "Kaplan-Meier")
    if na.sum() < 2:
        raise ValueError("Pseudo-observations need at least two units.")
    causes = ordered_labels(ea)
    validate_event({k: i for i, k in enumerate(causes)}, cause)
    times, scalar = _as_times(t, "t")
    _check_within(times, xa, "t")
    # 0 a failure from the cause, 1 anything else.
    kk = np.where(label_mask(ea, cause) & (ca == 0), 0, 1)
    u, groups, _, CIF = _loo(xa, 1.0 - ca, na, kk)
    at = _at(u, CIF[0], times, 0.0)
    out = _jackknife(at[-1], at[:-1], groups, na.sum())
    return out[:, 0] if scalar else out


# ---------------------------------------------------------------------------
# The regression.
# ---------------------------------------------------------------------------
def _link(name: str) -> tuple[Any, Any, Any]:
    """(g, g inverse, d mu / d eta) of a link."""
    if name == "identity":
        return (lambda m: m), (lambda e: e), (lambda e: np.ones_like(e))
    if name == "log":
        return np.log, np.exp, np.exp
    if name == "logit":

        def inv(e: npt.NDArray) -> npt.NDArray:
            return 1.0 / (1.0 + np.exp(-e))

        return (
            (lambda m: np.log(m / (1 - m))),
            inv,
            (lambda e: inv(e) * (1 - inv(e))),
        )
    # cloglog: mu = 1 - exp(-exp(eta))
    return (
        (lambda m: np.log(-np.log(1 - m))),
        (lambda e: -np.expm1(-np.exp(e))),
        (lambda e: np.exp(e - np.exp(e))),
    )


def _gee(
    y: npt.NDArray,
    X: npt.NDArray,
    w: npt.NDArray,
    cluster: npt.NDArray,
    link: str,
    beta0: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, int]:
    """Solve the independence GEE ``sum w D' (y - mu) = 0`` (gaussian
    variance) by Gauss-Newton with step halving; return the coefficients,
    the robust covariance ``A^-1 B A^-1`` and the iterations."""
    _, ginv, dmu = _link(link)
    beta = beta0.copy()

    def ss(b: npt.NDArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            r = y - ginv(X @ b)
        v = float(np.sum(w * r * r))
        return v if np.isfinite(v) else np.inf

    current = ss(beta)
    it = 0
    for it in range(1, 201):
        eta = X @ beta
        D = dmu(eta)[:, None] * X
        A = D.T @ (w[:, None] * D)
        g = D.T @ (w * (y - ginv(eta)))
        step = np.linalg.solve(A, g)
        scale = 1.0
        while True:
            trial = beta + scale * step
            value = ss(trial)
            if value <= current or scale < 1e-10:
                break
            scale /= 2
        beta, current = trial, value
        if np.max(np.abs(scale * step) / (1.0 + np.abs(beta))) < 1e-12:
            break
    else:
        raise RuntimeError(
            "The estimating equations did not converge in 200 steps; try "
            "another link (identity or logit) or fewer covariates."
        )
    eta = X @ beta
    D = dmu(eta)[:, None] * X
    A = D.T @ (w[:, None] * D)
    contrib = D * (y - ginv(eta))[:, None]
    # Sum each unit's rows (its times), then the units' outer products,
    # each unit counted n times.
    _, inv = np.unique(cluster, return_inverse=True)
    U = np.zeros((inv.max() + 1, X.shape[1]))
    np.add.at(U, inv, contrib)
    wu = np.zeros(inv.max() + 1)
    wu[inv] = w
    B = U.T @ (wu[:, None] * U)
    A_inv = np.linalg.inv(A)
    return beta, A_inv @ B @ A_inv, it


class PseudoRegressionResult:
    """
    A regression fitted to pseudo-observations, from
    :func:`pseudo_regression`.

    The coefficients follow SurPyval's sign: **a positive coefficient
    means a shorter life**. For the survival regressions the response is
    the chance of the event by ``t``, ``F(t) = 1 - S(t)`` (with the
    complementary log-log link, ``log(-log S(t)) = a(t) + b'Z``, so
    ``exp(b)`` is a ratio of cumulative hazards at ``t``, close to a Cox
    hazard ratio); for the cumulative incidence the response is the
    incidence of the cause (a positive coefficient, more failures from
    it). For the restricted mean the response is the RMST and the
    covariates enter with a minus sign, ``g(RMST) = a - b'Z``: with the
    identity link ``-b`` is the difference in RMST per unit of the
    covariate (``b = 20`` means 20 time units *less* of the first
    ``tau``), with the log link ``exp(-b)`` the ratio of RMSTs.

    Attributes
    ----------
    quantity : str
        "survival", "rmst" or "cif".
    link : str
        The link of the model.
    times : numpy.ndarray
        The times of the survival or incidence (``[tau]`` for "rmst").
    cause : label or None
        The cause of a "cif" model.
    names : list of str
        The coefficients' names: one intercept per time
        (``"Intercept (t=...)"``, or ``"Intercept"`` with one time), then
        the covariates.
    coef : numpy.ndarray
        The coefficients.
    se : numpy.ndarray
        Their robust (sandwich) standard errors.
    ci : numpy.ndarray
        Their Wald intervals at ``1 - alpha_ci``, one row ``(lower,
        upper)`` per coefficient.
    p : numpy.ndarray
        The two-sided Wald p-values.
    alpha_ci : float
        The level of ``ci``.
    n_units : float
        The units analysed (the sum of the counts).
    iterations : int
        The steps the estimating equations took.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import pseudo_regression
    >>> rng = np.random.default_rng(1)
    >>> z = rng.integers(0, 2, 300)
    >>> x = rng.exponential(10 * np.exp(-0.7 * z))
    >>> res = pseudo_regression(x, z, quantity="rmst", tau=10)
    >>> bool(res.coef[1] > 0)  # z = 1 shortens life: positive
    True
    """

    def __init__(
        self,
        quantity: str,
        link: str,
        times: npt.NDArray,
        cause: Any,
        names: list[str],
        coef: npt.NDArray,
        cov: npt.NDArray,
        alpha_ci: float,
        n_units: float,
        iterations: int,
        n_covariates: int,
        sign: float,
    ) -> None:
        self.quantity = quantity
        self.link = link
        self.times = times
        self.cause = cause
        self.names = names
        self.coef = coef
        self._cov = cov
        self.se = np.sqrt(np.diag(cov))
        q = norm.ppf(1 - alpha_ci / 2)
        self.ci = np.stack([coef - q * self.se, coef + q * self.se], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.p = 2 * norm.sf(np.abs(coef / self.se))
        self.alpha_ci = alpha_ci
        self.n_units = n_units
        self.iterations = iterations
        self._p = n_covariates
        self._sign = sign

    def covariance(self) -> npt.NDArray:
        """The robust (sandwich) covariance of :attr:`coef`."""
        return self._cov.copy()

    def summary(self, alpha_ci: float | None = None) -> pd.DataFrame:
        """
        The coefficient table.

        Parameters
        ----------
        alpha_ci : float, optional
            One minus the level of the intervals (the fit's by default).

        Returns
        -------
        pandas.DataFrame
            One row per coefficient: ``coef``, ``exp(coef)`` (not for the
            identity link), ``se(coef)``, the interval of each, ``z`` and
            ``p``.
        """
        from surpyval.univariate.regression._summary import (
            coefficient_table,
        )

        return coefficient_table(
            self.names,
            self.coef,
            self.se,
            self.alpha_ci if alpha_ci is None else alpha_ci,
            exp=self.link != "identity",
        )

    def _eta(
        self, Z: npt.ArrayLike, t: Any
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim < 2:
            Z = Z.reshape(-1, self._p) if self._p else Z.reshape(-1, 0)
        if Z.shape[1] != self._p:
            raise ValueError(
                f"Z has {Z.shape[1]} column(s); the model has {self._p}."
            )
        if t is None:
            j = 0
            if self.times.size > 1:
                raise ValueError(
                    "Give 't', one of the model's times "
                    f"{list(self.times)}, to predict at."
                )
        else:
            hits = np.flatnonzero(np.isclose(self.times, float(t)))
            if hits.size == 0:
                raise ValueError(
                    f"t = {t} is not one of the model's times "
                    f"{list(self.times)}; the model has an intercept only "
                    "at those."
                )
            j = int(hits[0])
        T = self.times.size if self.quantity != "rmst" else 1
        X = np.zeros((Z.shape[0], T + self._p))
        X[:, j] = 1.0
        X[:, T:] = self._sign * Z
        eta = X @ self.coef
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, self._cov, X))
        return X, eta, se

    def _natural(self, eta: npt.NDArray) -> npt.NDArray:
        _, ginv, _ = _link(self.link)
        mu = ginv(eta)
        return 1.0 - mu if self.quantity == "survival" else mu

    def predict(self, Z: npt.ArrayLike, t: float | None = None) -> Any:
        """
        The fitted quantity for covariate rows ``Z``.

        Parameters
        ----------
        Z : array_like
            Covariate rows, one per prediction.
        t : float, optional
            Which of the model's times (needed with more than one).

        Returns
        -------
        numpy.ndarray
            The survival ``S(t)`` ("survival"), the RMST ("rmst") or the
            cause's incidence ("cif") of each row.
        """
        _, eta, _ = self._eta(Z, t)
        return self._natural(eta)

    def predict_cb(
        self,
        Z: npt.ArrayLike,
        t: float | None = None,
        alpha_ci: float = 0.05,
    ) -> npt.NDArray:
        """
        Wald bounds of :meth:`predict`, from the robust covariance on the
        link's scale, mapped back through the link.

        Parameters
        ----------
        Z : array_like
            Covariate rows, one per prediction.
        t : float, optional
            Which of the model's times (needed with more than one).
        alpha_ci : float, optional
            One minus the level of the bounds.

        Returns
        -------
        numpy.ndarray
            Shape ``(rows, 2)``: the lower and upper bound of each row.
        """
        _, eta, se = self._eta(Z, t)
        q = norm.ppf(1 - alpha_ci / 2)
        a = self._natural(eta - q * se)
        b = self._natural(eta + q * se)
        return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)

    def to_dict(self) -> dict:
        """The result as strict JSON (a non-finite number is ``None``)."""

        def finite(v: Any) -> Any:
            v = float(v)
            return v if np.isfinite(v) else None

        cause = self.cause
        if isinstance(cause, np.generic):
            cause = cause.item()
        return {
            "model": "pseudo-observation regression",
            "quantity": self.quantity,
            "link": self.link,
            "times": [float(t) for t in self.times],
            "cause": cause,
            "sign": (
                "a positive coefficient means a shorter life"
                + (
                    " (the covariates enter g(RMST) = a - b'Z)"
                    if self.quantity == "rmst"
                    else ""
                )
            ),
            "variance": "robust (sandwich), independence working model",
            "alpha_ci": self.alpha_ci,
            "n_units": finite(self.n_units),
            "names": list(self.names),
            "coef": [finite(v) for v in self.coef],
            "se": [finite(v) for v in self.se],
            "ci": [[finite(a), finite(b)] for a, b in self.ci],
            "p": [finite(v) for v in self.p],
            "covariance": [[finite(v) for v in row] for row in self._cov],
        }

    def __repr__(self) -> str:
        from surpyval.univariate.regression._summary import format_table

        what = {
            "survival": "survival probability",
            "rmst": "restricted mean survival time",
            "cif": f"cumulative incidence of cause {self.cause!r}",
        }[self.quantity]
        when = (
            f"tau = {self.times[0]:g}"
            if self.quantity == "rmst"
            else "t = " + ", ".join(f"{t:g}" for t in self.times)
        )
        level = f"{100 * (1 - self.alpha_ci):g}%"
        header = "Pseudo-observation regression"
        return (
            f"{header}\n{'=' * len(header)}\n"
            f"Quantity         : {what} ({when})\n"
            f"Link             : {self.link}\n"
            f"Units            : {self.n_units:g}\n"
            "Sign             : a positive coefficient means a shorter life\n"
            "Standard errors  : robust (sandwich)\n"
            + format_table(
                self.summary(),
                [
                    "coef",
                    "exp(coef)",
                    "se(coef)",
                    f"coef lower {level}",
                    f"coef upper {level}",
                    "p",
                ],
            )
        )


def pseudo_regression(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    quantity: str = "survival",
    t: npt.ArrayLike | None = None,
    tau: float | None = None,
    e: npt.ArrayLike | None = None,
    cause: Any = None,
    link: str | None = None,
    names: list[str] | None = None,
    alpha_ci: float = 0.05,
    tl: npt.ArrayLike | None = None,
) -> PseudoRegressionResult:
    """
    Regression on the survival probability, the restricted mean survival
    time or a cause's cumulative incidence, through pseudo-observations.

    Each unit's censored time is replaced by its jackknife pseudo-value
    of the quantity (:func:`pseudo_survival`, :func:`pseudo_rmst`,
    :func:`pseudo_cif`), and a generalised linear model is fitted to the
    pseudo-values by estimating equations, ``g(E[y]) = a(t) + s b'Z``,
    with an independence working model and robust (sandwich) standard
    errors. With several times ``t`` each time has its own intercept and
    the covariates' effects are shared. This is the analysis of
    Andersen, Klein and Rosthoj (2003) and Klein et al. (2008), what R's
    ``geepack::geeglm(..., id = id, corstr = "independence")`` fitted to
    ``pseudo::pseudosurv``, ``pseudomean`` or ``pseudoci`` gives.

    The sign: a positive coefficient means a shorter life. The survival
    regressions model the event probability ``F(t) = 1 - S(t)`` (with the
    default complementary log-log link, ``log(-log S(t)) = a(t) + b'Z``,
    a proportional-hazards-like model at the chosen times); the
    incidence regressions model the cause's incidence; the RMST
    regression is ``g(RMST) = a - b'Z``, so with the identity link a
    covariate's RMST difference is ``-b``.

    Parameters
    ----------
    x : array_like
        The failure or censoring time of each row.
    Z : array_like
        The covariates, one row per row of ``x`` (a 1-D array is one
        covariate). Rows with a missing or infinite covariate are dropped
        before the pseudo-values are computed, with a warning.
    c : array_like, optional
        Censoring flags: 0 an event, 1 right-censored.
    n : array_like, optional
        Counts of identical units (each its own unit in the sandwich).
    quantity : str, optional
        "survival" (the default), "rmst" or "cif".
    t : float or array_like, optional
        The time(s) of a "survival" or "cif" model (needed for them).
    tau : float, optional
        The end of the window of an "rmst" model (needed for it).
    e : array_like, optional
        The cause of each failure, for "cif" (``None`` censored).
    cause : label, optional
        The cause of a "cif" model.
    link : str, optional
        "cloglog" (the default) or "logit" for "survival" and "cif";
        "identity" (the default) or "log" for "rmst".
    names : list of str, optional
        The covariates' names (by default ``Z``'s DataFrame columns, or
        ``x0``, ``x1``, ...).
    alpha_ci : float, optional
        One minus the level of the intervals ``ci``.
    tl : array_like, optional
        Delayed entry is refused.

    Returns
    -------
    PseudoRegressionResult
        The coefficients, their robust standard errors, intervals and
        p-values, ``summary()``, ``predict``, ``predict_cb`` and
        ``to_dict()``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import pseudo_regression
    >>> rng = np.random.default_rng(0)
    >>> z = rng.normal(size=400)
    >>> x = rng.weibull(1.5, 400) * 10 * np.exp(-0.5 * z)
    >>> c = (x > 15).astype(int)
    >>> x = np.minimum(x, 15)
    >>> res = pseudo_regression(x, z, c, quantity="survival", t=[5, 10])
    >>> res.names
    ['Intercept (t=5)', 'Intercept (t=10)', 'x0']
    >>> bool(res.coef[2] > 0)  # higher z, shorter life
    True
    """
    from surpyval.utils.covariates import finite_covariate_mask

    if quantity not in LINKS:
        raise ValueError(f"quantity = {quantity!r}: use one of {list(LINKS)}.")
    if link is None:
        link = LINKS[quantity][0]
    if link not in LINKS[quantity]:
        raise ValueError(
            f"link = {link!r} is not available for quantity = {quantity!r}; "
            f"use one of {list(LINKS[quantity])}."
        )
    if quantity == "cif" and (e is None or cause is None):
        raise ValueError(
            "quantity = 'cif' needs the causes 'e' and the 'cause' to model."
        )
    if quantity == "rmst" and tau is None:
        raise ValueError("quantity = 'rmst' needs 'tau', the window's end.")
    if quantity != "rmst" and t is None:
        raise ValueError(f"quantity = {quantity!r} needs the time(s) 't'.")
    _refuse_truncation(tl)

    if names is None and isinstance(Z, pd.DataFrame):
        names = [str(col) for col in Z.columns]
    if isinstance(Z, (pd.DataFrame, pd.Series)):
        Z = Z.to_numpy()
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr[:, None]
    x_arr = np.asarray(x, dtype=float)
    check_covariate_rows(Z_arr, x_arr.shape[0])
    p = Z_arr.shape[1]
    if names is None:
        names = [f"x{i}" for i in range(p)]
    if len(names) != p:
        raise ValueError(f"'names' has {len(names)} names; Z has {p} columns.")

    keep = finite_covariate_mask(Z_arr)

    def sub(a: Any) -> Any:
        if a is None or np.ndim(a) == 0:
            return a
        return np.asarray(a, dtype=object if a is e else None)[keep]

    xk, Zk = x_arr[keep], Z_arr[keep]
    ck, nk = sub(c), sub(n)
    _, _, n_units = _times_censoring(xk, ck if quantity != "cif" else None, nk)

    if quantity == "survival":
        times, _ = _as_times(t, "t")
        y = 1.0 - np.atleast_2d(pseudo_survival(xk, times, ck, nk).T).T
        ok_c = np.zeros(xk.size) if ck is None else np.asarray(ck, float)
        for tj in times:
            if not np.any((ok_c == 0) & (xk <= tj)):
                raise ValueError(
                    f"No failures by t = {tj:g}: the event probability is "
                    "0 for everyone there. Leave that time out."
                )
        sign = 1.0
    elif quantity == "cif":
        times, _ = _as_times(t, "t")
        ek = sub(e)
        y = np.atleast_2d(pseudo_cif(xk, ek, times, cause, ck, nk).T).T
        has = np.array(
            [
                v is not None
                and not (isinstance(v, float) and np.isnan(v))
                and v == cause
                for v in ek
            ]
        )
        for tj in times:
            if not np.any(has & (xk <= tj)):
                raise ValueError(
                    f"No failures from cause {cause!r} by t = {tj:g}; leave "
                    "that time out."
                )
        sign = 1.0
    else:
        times = np.array([float(tau)])  # type: ignore[arg-type]
        y = pseudo_rmst(xk, float(tau), ck, nk)[:, None]  # type: ignore
        sign = -1.0

    m, T = y.shape
    counts = np.ones(m) if nk is None else np.asarray(nk, float) * np.ones(m)
    X = np.zeros((m * T, T + p))
    X[np.arange(m * T), np.tile(np.arange(T), m)] = 1.0
    X[:, T:] = sign * np.repeat(Zk, T, axis=0)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(
            "The covariates are collinear (with each other or with the "
            "intercept: a constant column): leave out the redundant ones."
        )
    yy = y.reshape(-1)
    w = np.repeat(counts, T)
    cluster = np.repeat(np.arange(m), T)

    # Start: each time's intercept at its mean pseudo-value (moved into
    # the link's range), the covariates' coefficients at 0.
    g, _, _ = _link(link)
    means = np.average(y, axis=0, weights=counts)
    lo, hi = (1e-3, 1 - 1e-3) if link in ("cloglog", "logit") else (1e-3, None)
    if link == "identity":
        start_mu = means
    else:
        start_mu = np.clip(means, lo, hi)
    beta0 = np.concatenate([g(start_mu), np.zeros(p)])
    beta, cov, its = _gee(yy, X, w, cluster, link, beta0)

    if T == 1:
        intercepts = ["Intercept"]
    else:
        intercepts = [f"Intercept (t={tj:g})" for tj in times]
    return PseudoRegressionResult(
        quantity=quantity,
        link=link,
        times=times,
        cause=cause if quantity == "cif" else None,
        names=intercepts + list(names),
        coef=beta,
        cov=cov,
        alpha_ci=alpha_ci,
        n_units=float(n_units.sum()),
        iterations=its,
        n_covariates=p,
        sign=sign,
    )


def pseudo_regression_from_df(
    df: pd.DataFrame,
    x_col: str,
    Z_cols: str | list[str] | None = None,
    formula: str | None = None,
    c_col: str | None = None,
    n_col: str | None = None,
    e_col: str | None = None,
    **kwargs: Any,
) -> PseudoRegressionResult:
    """
    :func:`pseudo_regression` from the columns of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The data.
    x_col : str
        The column of times.
    Z_cols : str or list of str, optional
        The covariate columns (or give ``formula``).
    formula : str, optional
        A formulaic formula for the covariates (``"age + C(sex)"``).
    c_col : str, optional
        The column of censoring flags.
    n_col : str, optional
        The column of counts.
    e_col : str, optional
        The column of causes, for ``quantity="cif"``.
    **kwargs
        The other arguments of :func:`pseudo_regression` (``quantity``,
        ``t``, ``tau``, ``cause``, ``link``, ``alpha_ci``).

    Returns
    -------
    PseudoRegressionResult
        As :func:`pseudo_regression`, with the covariates named.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from surpyval import pseudo_regression_from_df
    >>> rng = np.random.default_rng(2)
    >>> df = pd.DataFrame({"z": rng.integers(0, 2, 200)})
    >>> df["x"] = rng.exponential(5 * np.exp(-0.5 * df["z"]))
    >>> res = pseudo_regression_from_df(df, "x", "z", quantity="rmst", tau=5)
    >>> res.names
    ['Intercept', 'z']
    """
    from surpyval.univariate.regression.regression_data import (
        design_matrix_from_df,
    )
    from surpyval.utils.covariates import optional_column
    from surpyval.utils.dataframe import check_columns

    check_columns(df, x_col=x_col)
    Z, names, _ = design_matrix_from_df(df, Z_cols, formula)
    return pseudo_regression(
        df[x_col].to_numpy(),
        Z,
        c=optional_column(df, c_col),
        n=optional_column(df, n_col),
        e=optional_column(df, e_col),
        names=list(names),
        **kwargs,
    )
