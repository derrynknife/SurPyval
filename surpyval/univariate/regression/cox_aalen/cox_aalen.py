r"""
The Cox-Aalen model: multiplicative Cox terms on an additive Aalen
baseline.

**In plain words.** Some covariates scale the hazard by a constant
factor, as in Cox's model (a hazard ratio); others add to it an amount
that may change with time, as in Aalen's additive model. The Cox-Aalen
model has both: the hazard of a unit is its additive part (a baseline
plus each additive covariate's time-varying excess) multiplied by
``exp(beta'Z)`` for the multiplicative covariates. With only the
intercept in the additive part it is Cox's model; with no multiplicative
covariates it is Aalen's.

**The model.** With additive covariates :math:`X_i` (a leading 1 for the
baseline) and multiplicative ones :math:`Z_i`,

.. math::
    h_i(t) = \{X_i' \alpha(t)\}\, e^{\beta' Z_i},

and what the data determine are :math:`\beta` and the cumulative
regression functions :math:`A(t) = \int_0^t \alpha(s)\, ds`.

**The estimating equations** (Scheike and Zhang 2002; Martinussen and
Scheike 2006, chapter 7). Write, at each event time :math:`t`, :math:`Y`
for the at-risk indicators, :math:`W = \mathrm{diag}(n_i Y_i
e^{\beta' Z_i})` and :math:`dN` for the events. For a given
:math:`\beta`, :math:`dA` is the weighted least-squares fit of the events
on the additive covariates,

.. math::
    d\hat A(t; \beta) = (X' W X)^{-1} X' dN(t),

and :math:`\beta` solves the score of the full likelihood with
:math:`d\hat A(\beta)` put in,

.. math::
    U(\beta) = \sum_t Z' \{I - W X (X'WX)^{-1} X'\}\, dN(t)
             = \sum_t \sum_i \{Z_i - \tilde Z_i(t)\}\, dN_i(t) = 0,
    \quad \tilde Z_i(t) = Z'WX (X'WX)^{-1} X_i .

With :math:`X = 1` this is Cox's partial-likelihood score with Breslow's
handling of ties, and :math:`d\hat A` is Breslow's baseline. It is solved
by Newton's method with the derivative

.. math::
    I(\beta) = -\frac{\partial U}{\partial \beta}
      = \sum_t \Big[ Z' D_a Z - Z'WX (X'WX)^{-1} X' D_a Z \Big],
    \quad D_a = \mathrm{diag}(w_i\, X_i' d\hat A(t)) ,

with :math:`w_i = n_i Y_i e^{\beta' Z_i}`. As in Aalen's model, the fit
stops before the first event time at which :math:`X'WX` is singular (too
few units at risk to separate the additive covariates).

**The variance.** :math:`U(\beta_0) = \sum_i \int \{Z_i - \tilde
Z_i\}\, dM_i` exactly (the :math:`dA` part of :math:`dN` is projected
out), a martingale, so :math:`\hat\beta - \beta \approx I^{-1} U` and

.. math::
    \widehat{\mathrm{Var}}(\hat\beta) = I^{-1} \Sigma I^{-\top}, \quad
    \Sigma = \sum_t \sum_i \{Z_i - \tilde Z_i\}^{\otimes 2}\,
             w_i X_i' d\hat A(t),

the predictable variation of the score (for :math:`X = 1`,
:math:`\Sigma = I` and this is the inverse of Cox's information).
``robust_covariance()`` replaces :math:`\Sigma` by
:math:`\sum_i n_i \varepsilon_i \varepsilon_i'`, :math:`\varepsilon_i =
\int \{Z_i - \tilde Z_i\} d\hat M_i` the score's martingale residuals (for
:math:`X = 1`, the Lin-Wei robust variance). For the cumulative
coefficients, :math:`\hat A(t) - A(t) \approx \sum_{s \le t} (X'WX)^{-1}
X' dM - P(t) (\hat\beta - \beta)` with :math:`P(t) = \sum_{s \le t}
(X'WX)^{-1} X' D_a Z`, so

.. math::
    \widehat{\mathrm{Var}}\, \hat A(t) = C(t) - K(t) I^{-\top} P(t)'
      - P(t) I^{-1} K(t)' + P(t)\, \widehat{\mathrm{Var}}(\hat\beta)\,
      P(t)',

with :math:`C(t) = \sum_{s \le t} (X'WX)^{-1} X' \mathrm{diag}(dN) X
(X'WX)^{-1}`, the optional variation that ``AalenAdditive`` reports, and
:math:`K(t) = \sum_{s \le t} (X'WX)^{-1} \sum_i X_i \{Z_i - \tilde
Z_i\}' w_i X_i' d\hat A`, the covariance of the two martingale terms
(0 for :math:`X = 1`). With :math:`X = 1` this is the Tsiatis variance
that ``CoxPH``'s curve bounds use; with no :math:`Z` it is Aalen's.

**Predictions.** The cumulative hazard of a unit is :math:`H(t) =
e^{\beta' Z} X' \hat A(t)`. Nothing keeps the additive part positive, so
:math:`X'\hat A(t)` can fall; as ``AalenAdditive`` does, the prediction
is its running maximum from time 0, floored at 0, so survival stays in
:math:`[0, 1]` and never rises. Bounds on a prediction are Wald bounds,
by the delta method from the variances above.

References: Scheike, T. H. and Zhang, M.-J. (2002) An additive-
multiplicative Cox-Aalen regression model. *Scandinavian Journal of
Statistics* 29, 75-88. Martinussen, T. and Scheike, T. H. (2006)
*Dynamic Regression Models for Survival Data*, Springer, chapter 7.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.diagnostics.mixin import DiagnoseMixin
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

from .._aliasing import aliased_columns, constant_columns, warn_aliased
from .._concordance import ConcordanceMixin
from .._early_stop import beyond, warn_if_stopped
from .._fit_skeleton import draws_by_row
from .._prediction import (
    ConditionalSurvivalMixin,
    paired_probabilities,
    step_draws,
    step_quantiles,
    unique_rows,
)
from .._summary import coefficient_names, coefficient_table
from ..additive_hazards.aalen_additive import risk_set_sums
from ..proportional_hazards._curve_variance import survival_bounds
from ..regression_data import (
    FrameDesign,
    NoLikelihoodMixin,
    check_covariate_width,
    prepare_Z,
    restore_covariate_meta,
    semi_parametric_inputs,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd

#: The name of the baseline's column of ``A``.
BASELINE = "baseline"
_ON = ("sf", "R", "ff", "F", "Hf")
_ON_ALIASES = {"R": "sf", "F": "ff"}
_BOUND_TYPES = ("log-log", "log", "plain")


def _validate(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
) -> tuple[npt.NDArray, ...]:
    x_a, c_a, n_a, tl_a, Z_a = semi_parametric_inputs(
        x,
        Z,
        c,
        n,
        tl,
        censoring=(
            "The Cox-Aalen model supports only observed (c=0) and "
            "right-censored (c=1) data; for left or interval censored data "
            "fit CoxPH (whose full-likelihood route takes them) or a "
            "parametric regression."
        ),
        truncation=(
            "The Cox-Aalen model takes left truncation (delayed entry) "
            "only: 'tl' must be one entry time per row, not a [tl, tr] "
            "pair."
        ),
    )
    if np.any(x_a < 0) or np.any(tl_a[np.isfinite(tl_a)] < 0):
        raise ValueError(
            "The Cox-Aalen model integrates the hazard from time 0; every "
            "time 'x' and entry time 'tl' must be non-negative."
        )
    if not np.any((c_a == 0) & (n_a > 0)):
        raise ValueError(
            "CoxAalen needs at least one event (c=0); with every "
            "observation censored there is nothing to estimate."
        )
    tl_a = np.where(np.isfinite(tl_a), tl_a, 0.0)
    return x_a, c_a, n_a, tl_a, Z_a


def _additive_positions(additive: Any, p: int, names: list[str]) -> list:
    """The positions of the additive columns of ``Z``: ``additive`` is
    ``None`` (none), or column positions, or (with ``names``) column
    names."""
    if additive is None:
        return []
    items = [additive] if isinstance(additive, (str, int)) else additive
    out = []
    for a in items:
        if isinstance(a, str):
            if a not in names:
                raise ValueError(
                    f"additive names {a!r}, which is not a covariate; the "
                    f"covariates are {names}. Give the additive columns "
                    "by name with fit_from_df or a DataFrame Z, else by "
                    "position."
                )
            out.append(names.index(a))
        elif isinstance(a, (int, np.integer)) and 0 <= int(a) < p:
            out.append(int(a))
        else:
            raise ValueError(
                f"additive must name columns of Z, by position (0 to "
                f"{p - 1}) or by name; got {a!r}."
            )
    if len(set(out)) != len(out):
        raise ValueError(f"additive names a column twice: {additive!r}.")
    return sorted(out)


def _event_times(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    X: npt.NDArray,
) -> npt.NDArray:
    """The event times, up to the first at which ``X'YX`` is singular
    (``W`` has positive weights, so its rank does not depend on
    ``beta``)."""
    times = np.unique(x[(c == 0) & (n > 0)])
    q = X.shape[1]
    XX = n[:, None, None] * X[:, :, None] * X[:, None, :]
    S = risk_set_sums(x, tl, XX, times)
    for k in range(times.size):
        if np.linalg.matrix_rank(S[k]) < q:
            return times[:k]
    return times


def _pieces(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    X: npt.NDArray,
    Z: npt.NDArray,
    beta: npt.NDArray,
    times: npt.NDArray,
) -> dict:
    """Every per-event-time quantity of the estimating equations at
    ``beta`` (see the module docstring), with the weights scaled by
    ``exp(-shift)`` for range: ``dA`` (the scaled increments), ``L =
    Z'WX (X'WX)^{-1}``, the score ``U``, its derivative ``I``, its
    predictable variation ``S``, and the increments ``dC``, ``dK`` and
    ``dP`` of the variance of ``A`` (all as for the scaled ``A``)."""
    q, p = X.shape[1], Z.shape[1]
    r = q + p
    eta = Z @ beta if p else np.zeros(x.size)
    shift = float(eta.max()) if p else 0.0
    w = n * np.exp(eta - shift)
    V = np.column_stack([X, Z])
    VV = V[:, :, None] * V[:, None, :]
    S_VV = risk_set_sums(x, tl, w[:, None, None] * VV, times)
    XVV = X[:, :, None, None] * VV[:, None, :, :]
    S_XVV = risk_set_sums(x, tl, w[:, None, None, None] * XVV, times)
    ev = (c == 0) & (n > 0) & (x <= (times[-1] if times.size else -np.inf))
    bucket = np.searchsorted(times, x[ev])
    E_V = np.zeros((times.size, r))
    np.add.at(E_V, bucket, n[ev][:, None] * V[ev])
    E_XX = np.zeros((times.size, q, q))
    np.add.at(E_XX, bucket, n[ev][:, None, None] * VV[ev][:, :q, :q])

    Ainv = np.linalg.inv(S_VV[:, :q, :q])
    dA = np.einsum("kab,kb->ka", Ainv, E_V[:, :q])
    L = np.einsum("kpa,kab->kpb", S_VV[:, q:, :q], Ainv)
    dU = E_V[:, q:] - np.einsum("kpa,ka->kp", L, E_V[:, :q])
    Sa = np.einsum("km,kmab->kab", dA, S_XVV)
    Sa_XX, Sa_XZ, Sa_ZZ = Sa[:, :q, :q], Sa[:, :q, q:], Sa[:, q:, q:]
    LS = np.einsum("kpa,kab->kpb", L, Sa_XZ)
    dI = Sa_ZZ - LS
    dS = (
        Sa_ZZ
        - LS
        - np.swapaxes(LS, 1, 2)
        + np.einsum("kpa,kab,kqb->kpq", L, Sa_XX, L)
    )
    dC = np.einsum("kab,kbc,kdc->kad", Ainv, E_XX, Ainv)
    dK = np.einsum(
        "kab,kbp->kap",
        Ainv,
        Sa_XZ - np.einsum("kab,kpb->kap", Sa_XX, L),
    )
    dP = np.einsum("kab,kbp->kap", Ainv, Sa_XZ)
    return {
        "shift": shift,
        "dA": dA,
        "L": L,
        "U": dU.sum(axis=0),
        "I": dI.sum(axis=0),
        "S": dS.sum(axis=0),
        "dC": dC,
        "dK": dK,
        "dP": dP,
    }


def score(
    beta: npt.ArrayLike,
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    X: npt.NDArray,
    Z: npt.NDArray,
    times: npt.NDArray | None = None,
) -> npt.NDArray:
    """The score :math:`U(\\beta)` of the module docstring, for the
    additive design ``X`` (with its column of ones) and multiplicative
    covariates ``Z``, over the event times ``times`` (all of them up to
    the first singular one when ``None``)."""
    times = _event_times(x, c, n, tl, X) if times is None else times
    beta = np.asarray(beta, dtype=float)
    return _pieces(x, c, n, tl, X, Z, beta, times)["U"]


def _solve(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    X: npt.NDArray,
    Z: npt.NDArray,
    times: npt.NDArray,
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray, dict, int, bool]:
    """Newton's method on ``U(beta) = 0`` from 0, halving a step that
    does not reduce ``|U|``. Returns ``(beta, pieces, iterations,
    converged)``."""
    beta = np.zeros(Z.shape[1])
    pieces = _pieces(x, c, n, tl, X, Z, beta, times)
    if Z.shape[1] == 0:
        return beta, pieces, 0, True
    for it in range(1, max_iter + 1):
        U, I_ = pieces["U"], pieces["I"]
        try:
            step = np.linalg.solve(I_, U)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(I_, U, rcond=None)[0]
        size = float(np.linalg.norm(U))
        t = 1.0
        while True:
            trial = beta + t * step
            new = _pieces(x, c, n, tl, X, Z, trial, times)
            if (
                np.all(np.isfinite(new["U"]))
                and np.linalg.norm(new["U"]) <= size
            ) or t < 1e-8:
                break
            t *= 0.5
        moved = np.max(np.abs(t * step))
        beta, pieces = trial, new
        if moved < tol * (1.0 + np.max(np.abs(beta))):
            return beta, pieces, it, True
    return beta, pieces, max_iter, False


def _robust_middle(
    x: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    X: npt.NDArray,
    Z: npt.NDArray,
    beta: npt.NDArray,
    times: npt.NDArray,
    pieces: dict,
) -> npt.NDArray:
    """``sum_i n_i eps_i eps_i'``, ``eps_i = int (Z_i - L X_i) dM_i`` with
    ``dM_i = dN_i - e^{beta'Z_i} X_i' dA``: the score's martingale
    residuals."""
    q, p = X.shape[1], Z.shape[1]
    e = np.exp(Z @ beta - pieces["shift"])
    dA, L = pieces["dA"], pieces["L"]
    zero_q = np.zeros((1, q))
    A_cum = np.vstack([zero_q, np.cumsum(dA, axis=0)])
    M = L[:, :, :, None] * dA[:, None, None, :]
    M_cum = np.concatenate([np.zeros((1, p, q, q)), np.cumsum(M, axis=0)])
    hi = np.searchsorted(times, x, side="right")
    lo = np.searchsorted(times, tl, side="right")
    dAi = A_cum[hi] - A_cum[np.minimum(lo, hi)]
    dMi = M_cum[hi] - M_cum[np.minimum(lo, hi)]
    comp = e[:, None] * (
        Z * np.einsum("ia,ia->i", X, dAi)[:, None]
        - np.einsum("ipab,ia,ib->ip", dMi, X, X)
    )
    k = np.searchsorted(times, x)
    is_event = (c == 0) & (k < times.size)
    is_event &= times[np.minimum(k, times.size - 1)] == x
    Lk = L[np.minimum(k, max(times.size - 1, 0))]
    jump = Z - np.einsum("ipa,ia->ip", Lk, X)
    eps = np.where(is_event[:, None], jump, 0.0) - comp
    return (n[:, None] * eps).T @ eps


def _aliased(Z: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
    """The columns of ``Z`` the fit cannot separate: constant (the
    baseline absorbs it, in either part) or a linear combination of the
    other columns and the intercept."""
    if Z.shape[1] == 0:
        return np.array([], dtype=int)
    Zc = Z - (n @ Z) / float(n.sum())
    gram = (n[:, None] * Zc).T @ Zc
    return aliased_columns(gram, Z.shape[0], constant_columns(Z))


class CoxAalenModel(
    DiagnoseMixin,
    ConditionalSurvivalMixin,
    NoLikelihoodMixin,
    ConcordanceMixin,
    SerialisableMixin,
):
    r"""
    A fitted Cox-Aalen model, returned by :meth:`CoxAalen.fit
    <surpyval.CoxAalen>`: ``h(t | Z) = {X' alpha(t)} exp(beta' Z_m)``, the
    additive covariates ``X`` (a leading 1, then the columns named by
    ``additive``) with time-varying effects and the multiplicative ones
    ``Z_m`` (the other columns) with hazard ratios ``exp(beta)``.

    ``beta`` (and ``params``) are the multiplicative coefficients, one per
    entry of :attr:`parameter_names`; ``A`` holds the cumulative
    regression functions of the additive part at the event times ``x``, a
    column per entry of :attr:`term_names` (the baseline first), for a
    unit with ``Z_m = 0``. See the module docstring for the estimating
    equations and the variances.

    Examples
    --------
    On the Rossi recidivism data, with prior convictions acting on the
    hazard multiplicatively and financial aid additively:

    >>> from surpyval import CoxAalen
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> df["censored"] = 1 - df["arrest"]
    >>> model = CoxAalen.fit_from_df(
    ...     df, x_col="week", c_col="censored", Z_cols=["fin", "prio"],
    ...     additive=["fin"],
    ... )
    >>> model.parameter_names, model.term_names
    (['prio'], ['baseline', 'fin'])
    >>> model.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(4)
                coef  exp(coef)  se(coef)       p
    covariate
    prio       0.104     1.1096    0.0268  0.0001
    >>> model.cumulative_coefficients([26, 52]).round(4)
          baseline     fin
    time
    26.0    0.1129 -0.0383
    52.0    0.2627 -0.0872

    Each prior conviction multiplies the hazard of arrest by about 1.11;
    by week 52 financial aid has taken about 0.087 from the cumulative
    hazard.
    """

    #: The event times at which ``A`` is estimated.
    x: npt.NDArray
    #: The cumulative regression functions at ``x``, ``(len(x), q)``.
    A: npt.NDArray
    #: The multiplicative coefficients (``params`` and ``coef`` are the
    #: same array).
    beta: npt.NDArray
    params: npt.NDArray
    coef: npt.NDArray
    #: The positions in ``Z`` of the additive and multiplicative columns.
    additive: npt.NDArray
    multiplicative: npt.NDArray
    #: The covariance of each row of ``A`` (``(len(x), q, q)``) and of
    #: ``A`` with ``beta`` (``(len(x), q, p)``).
    _A_cov: npt.NDArray
    _A_beta_cov: npt.NDArray
    _cov: npt.NDArray
    _robust_cov: npt.NDArray
    _aliased: npt.NDArray
    n_iter: int
    converged: bool
    #: Fitted by estimating equations (a likelihood score with the
    #: additive increments put in), not by maximising a likelihood.
    maximum: str = "not applicable"
    kind: str = "Cox-Aalen"
    parameterization: str = "Semi-Parametric"
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: object = None
    _fit_data: tuple | None = None
    #: Where the fit stopped before the last event time (``X'WX``
    #: singular beyond it), else ``nan``; every prediction beyond it is
    #: ``nan``.
    stopped_at: float = float("nan")

    _NO_LIKELIHOOD_WHY = (
        "the Cox-Aalen model is fitted by estimating equations (the score "
        "with the additive increments, a least-squares fit, put in), not by "
        "maximising a likelihood"
    )

    # -- names and covariates ---------------------------------------------

    @property
    def _all_names(self) -> list[str]:
        width = int(self.additive.size + self.multiplicative.size)
        return coefficient_names(self, width)

    @property
    def parameter_names(self) -> list[str]:
        """The multiplicative covariates' names, one per entry of
        ``beta``."""
        names = self._all_names
        return [names[j] for j in self.multiplicative]

    @property
    def term_names(self) -> list[str]:
        """The names of the columns of ``A``: ``"baseline"``, then the
        additive covariates."""
        names = self._all_names
        return [BASELINE] + [names[j] for j in self.additive]

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` whose effects the data cannot determine:
        a constant column (which the baseline absorbs) or a linear
        combination of the others. Their entries of ``beta`` or columns of
        ``A`` are ``nan``, and predictions take them as 0."""
        return np.asarray(self._aliased, dtype=int)

    def _rows(self, Z: Any) -> tuple[npt.NDArray, npt.NDArray]:
        """The additive rows ``[1, Z_a]`` and multiplicative rows ``Z_m``
        of covariates as the predictions take them, aliased columns as
        0."""
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        width = int(self.additive.size + self.multiplicative.size)
        check_covariate_width(Zp, self._all_names)
        rows = np.asarray(covariate_rows(Zp, width), dtype=float)
        if self.aliased.size:
            rows = rows.copy()
            rows[:, self.aliased] = 0.0
        X = np.column_stack([np.ones(rows.shape[0]), rows[:, self.additive]])
        return X, rows[:, self.multiplicative]

    def _coef(self) -> npt.NDArray:
        return np.where(np.isnan(self.beta), 0.0, self.beta)

    def __repr__(self) -> str:
        lines = [
            "Semi-Parametric Regression SurPyval Model",
            "=========================================",
            "Type                : Cox-Aalen",
            "Kind                : additive (Aalen) baseline, "
            "multiplicative (Cox) covariates",
            f"Converged           : {self.converged} ({self.n_iter} iters)",
            "Multiplicative coefficients (exp = hazard ratio):",
        ]
        se = self.standard_errors()
        for nm, b, s in zip(self.parameter_names, self.beta, se):
            lines.append(f"   {nm:>10}  :  {b: .6f}  (se {s:.6f})")
        if self.x.size:
            lines.append(f"Cumulative additive effects at t = {self.x[-1]:g}:")
            for nm, a in zip(self.term_names, self.A[-1]):
                lines.append(f"   {nm:>10}  :  {a: .6f}")
        return "\n".join(lines)

    # -- the multiplicative coefficients ----------------------------------

    def covariance(self) -> npt.NDArray:
        """The covariance of ``beta``: ``I^{-1} Sigma I^{-T}`` with
        ``Sigma`` the score's predictable variation (see the module
        docstring); the inverse of Cox's information when the additive
        part is the baseline alone."""
        return self._cov.copy()

    def robust_covariance(self) -> npt.NDArray:
        """The robust covariance of ``beta``: ``I^{-1} B I^{-T}``, ``B`` the
        sum of the outer products of the score's martingale residuals
        (for an intercept-only additive part, Lin and Wei's)."""
        return self._robust_cov.copy()

    def standard_errors(self, robust: bool = False) -> npt.NDArray:
        """The standard errors of ``beta``, from :meth:`covariance` (or
        :meth:`robust_covariance`); ``nan`` where a variance is not
        positive."""
        cov = self._robust_cov if robust else self._cov
        var = np.diag(cov)
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.where(var >= 0, var, np.nan))

    def summary(
        self, alpha_ci: float = 0.05, robust: bool = False
    ) -> pd.DataFrame:
        """
        The multiplicative coefficients' table, in the layout of
        ``CoxPH``'s :meth:`summary`: each coefficient, ``exp(coef)`` (its
        hazard ratio), the standard error (``robust`` for the robust
        one), a two-sided ``1 - alpha_ci`` Wald interval, ``z`` and its
        p-value. The additive part is in :meth:`cumulative_coefficients`.
        """
        check_alpha_ci(alpha_ci)
        return coefficient_table(
            self.parameter_names,
            self.beta,
            self.standard_errors(robust),
            alpha_ci,
        )

    def param_cb(
        self,
        name: str | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        robust: bool = False,
    ) -> npt.NDArray:
        """
        Wald confidence bound(s) on a multiplicative coefficient, from
        :meth:`covariance` (``robust`` for :meth:`robust_covariance`):
        for ``name``, its ``[lower, upper]`` (or the one bound); for
        ``None`` (the default), every coefficient's, one row each.

        Only Wald bounds are available: the model has no likelihood for a
        likelihood-ratio bound (with the baseline alone additive, fit
        ``CoxPH`` for one), and no bootstrap is built (refit the model to
        resampled rows for one).
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        if method != "wald":
            raise ValueError(
                "CoxAalenModel has Wald bounds only (method='wald'): it has "
                "no likelihood for a likelihood-ratio bound (fit CoxPH for "
                "one when the additive part is the baseline alone), and no "
                "bootstrap is built (refit to resampled rows for one); got "
                f"method={method!r}."
            )
        names = self.parameter_names
        if name is not None and name not in names:
            raise ValueError(
                f"Unknown coefficient {name!r}; this model's multiplicative "
                f"coefficients are {names}."
            )
        b = np.asarray(self.beta, dtype=float)
        se = self.standard_errors(robust)
        if bound == "two-sided":
            z = norm.isf(alpha_ci / 2)
            out = np.column_stack([b - z * se, b + z * se])
        else:
            z = norm.isf(alpha_ci)
            out = b - z * se if bound == "lower" else b + z * se
        return out if name is None else out[names.index(name)]

    # -- the cumulative coefficients --------------------------------------

    def _beyond(self, t: npt.NDArray) -> npt.NDArray:
        """Whether each time is after the fit stopped short of the data
        (see :attr:`stopped_at`)."""
        return beyond(t, self.stopped_at)

    def _index(self, t: npt.NDArray) -> npt.NDArray:
        return np.searchsorted(self.x, t, side="right") - 1

    def _at(self, arr: npt.NDArray, t: npt.NDArray) -> npt.NDArray:
        """``arr`` (one entry per event time) in force at each of ``t``,
        0 before the first event time."""
        idx = self._index(t)
        shape = (t.size,) + arr.shape[1:]
        if arr.shape[0] == 0:
            return np.zeros(shape)
        out = arr[np.clip(idx, 0, None)]
        mask = (idx < 0).reshape((-1,) + (1,) * (arr.ndim - 1))
        return np.where(mask, 0.0, out)

    def _check_terms(self, terms: Any) -> npt.NDArray:
        names = self.term_names
        if terms is None:
            return np.arange(len(names))
        out = []
        for t in [terms] if isinstance(terms, (str, int)) else terms:
            if isinstance(t, str) and t in names:
                out.append(names.index(t))
            elif isinstance(t, (int, np.integer)) and 0 <= t < len(names):
                out.append(int(t))
            else:
                raise ValueError(
                    "Unknown term {!r}; the terms are {} (or their "
                    "positions 0 to {}).".format(t, names, len(names) - 1)
                )
        return np.asarray(out, dtype=int)

    def cumulative_coefficients(
        self, x: npt.ArrayLike | None = None, terms: Any = None
    ) -> pd.DataFrame:
        """
        The cumulative regression functions of the additive part, ``A_j(t)``,
        as a table: one row per time (the event times, or ``x``), one
        column per term (:attr:`term_names`), for a unit with every
        multiplicative covariate at 0. A step function, 0 before the first
        event time and held after the last.
        """
        import pandas as pd

        t = self.x if x is None else np.atleast_1d(np.asarray(x, float))
        cols = self._check_terms(terms)
        values = self._at(self.A, t)[:, cols]
        values[np.isnan(t) | self._beyond(t)] = np.nan
        return pd.DataFrame(
            values,
            index=pd.Index(t, name="time"),
            columns=[self.term_names[j] for j in cols],
        )

    def cumulative_coefficients_cb(
        self,
        x: npt.ArrayLike | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        terms: Any = None,
    ) -> npt.NDArray:
        """
        Pointwise Wald bounds on the cumulative regression functions,
        ``A_j(t) +/- z se``, the standard error allowing for the
        estimated ``beta`` (see the module docstring).
        """
        check_option("bound", bound, BOUNDS)
        check_alpha_ci(alpha_ci)
        t = self.x if x is None else np.atleast_1d(np.asarray(x, float))
        cols = self._check_terms(terms)
        A = self._at(self.A, t)[:, cols]
        A[np.isnan(t) | self._beyond(t)] = np.nan
        var = np.diagonal(self._at(self._A_cov, t), axis1=1, axis2=2)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.maximum(var[:, cols], 0.0))
        if bound == "two-sided":
            z = norm.isf(alpha_ci / 2)
            return np.stack([A - z * se, A + z * se], axis=-1)
        z = norm.isf(alpha_ci)
        return A - z * se if bound == "lower" else A + z * se

    # -- predictions ------------------------------------------------------

    def _cumulative_hazard(self, x: npt.NDArray, Z: Any) -> tuple:
        """``(H*, H, X, Zm, x, risk)``: the estimate ``H = e^{beta'Zm}
        X'A(x)``, its running maximum from 0 floored at 0 (what the model
        predicts with), the rows paired with ``x``, the times, and
        ``e^{beta'Zm}``."""
        X, Zm = self._rows(Z)
        check_paired_rows(x.size, X.shape[0], grid=False)
        if x.size == 1 and X.shape[0] > 1:
            x = np.repeat(x, X.shape[0])
        X = np.broadcast_to(X, (x.size, X.shape[1]))
        Zm = np.broadcast_to(Zm, (x.size, Zm.shape[1]))
        risk = np.exp(Zm @ self._coef())
        A = np.where(np.isnan(self.A), 0.0, self.A)
        if A.shape[0] == 0 or x.size == 0:
            zero = np.where(np.isnan(x), np.nan, 0.0)
            return zero, zero, X, Zm, x, risk
        uniq, which = unique_rows(X)
        grid = A @ uniq.T
        peaks = np.maximum.accumulate(np.maximum(grid, 0.0), axis=0)
        idx = self._index(x)
        at = np.clip(idx, 0, None)
        H = np.where(idx >= 0, grid[at, which], 0.0) * risk
        H_star = np.where(idx >= 0, peaks[at, which], 0.0) * risk
        bad = np.isnan(x) | np.isnan(X).any(axis=1) | np.isnan(risk)
        bad = bad | self._beyond(x)
        H = np.where(bad, np.nan, H)
        H_star = np.where(bad, np.nan, np.where(x <= 0, 0.0, H_star))
        return H_star, H, X, Zm, x, risk

    @keeps_query_shape
    def Hf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The cumulative hazard at ``x`` of a unit with covariates ``Z``
        (one row for every ``x``, or one row per ``x``): ``e^{beta'Z_m}``
        times the running maximum from time 0, floored at 0, of the
        additive part ``X'A(x)``. Nothing keeps the additive hazard
        positive, so its estimate can fall; as ``AalenAdditive`` does,
        the prediction is then held, so survival stays in [0, 1] and never
        rises. It is the estimate wherever the estimate is at its running
        maximum.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return self._cumulative_hazard(x, Z)[0]

    @keeps_query_shape
    def sf(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """Survival ``exp(-Hf(x, Z))``."""
        return np.exp(-self.Hf(x, Z))

    @keeps_query_shape
    def ff(self, x: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """Failure probability ``1 - sf(x, Z)``."""
        return -np.expm1(-self.Hf(x, Z))

    @keeps_query_shape
    def qf(self, p: npt.ArrayLike, Z: Any) -> npt.NDArray:
        """
        The quantile function: the first event time at which ``ff(x, Z)``
        reaches ``p``, ``nan`` where it never does. ``Z`` is paired with
        ``p`` as :meth:`sf` pairs it with ``x``.
        """
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        width = int(self.additive.size + self.multiplicative.size)
        rows = np.asarray(covariate_rows(Zp, width), dtype=float)
        u, rows, _ = paired_probabilities(p, rows)
        if not self.x.size or not u.size:
            return np.full(u.shape, np.nan)
        F = np.vstack([self.ff(self.x, rows[k]) for k in range(rows.shape[0])])
        return step_quantiles(F, self.x, u)

    def random(
        self, size: int, Z: Any, random_state: Any = None
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Draw ``size`` lifetimes for each covariate row of ``Z``, by inverse
        transform on the step survival curve (see ``AalenAdditiveModel``'s
        ``random``); a draw beyond the last event time is ``inf``.
        Returns the draws and the row each was drawn at.
        """
        Zp = prepare_Z(Z, self.feature_names, self._model_spec)
        width = int(self.additive.size + self.multiplicative.size)
        rows = np.asarray(covariate_rows(Zp, width), dtype=float)

        def invert(U: npt.NDArray, row: npt.NDArray) -> npt.NDArray:
            H = self.Hf(self.x, row) if self.x.size else np.zeros(0)
            out = step_draws(H, self.x, U)
            # A draw past where the fit stopped is unknown, not "outlived".
            if np.isfinite(self.stopped_at):
                out = np.where(np.isinf(out), np.nan, out)
            return out

        return draws_by_row(size, rows, random_state, invert)

    def _H_se(
        self,
        t: npt.NDArray,
        X: npt.NDArray,
        Zm: npt.NDArray,
        H: npt.NDArray,
        risk: npt.NDArray,
    ) -> npt.NDArray:
        """The delta-method standard error of ``H = e^{beta'Zm} X'A(t)``."""
        C = np.where(np.isnan(self._A_cov), 0.0, self._A_cov)
        D = np.where(np.isnan(self._A_beta_cov), 0.0, self._A_beta_cov)
        V = np.where(np.isnan(self._cov), 0.0, self._cov)
        g_A = risk[:, None] * X
        g_b = H[:, None] * Zm
        var = (
            np.einsum("ia,iab,ib->i", g_A, self._at(C, t), g_A)
            + 2 * np.einsum("ia,iap,ip->i", g_A, self._at(D, t), g_b)
            + np.einsum("ip,pq,iq->i", g_b, V, g_b)
        )
        var = np.where(self._index(t) >= 0, var, 0.0)
        return np.sqrt(np.maximum(var, 0.0))

    def cb(
        self,
        x: npt.ArrayLike,
        Z: Any,
        on: str = "sf",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "log-log",
        method: str = "wald",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on a unit's predicted ``sf``, ``ff`` or
        ``Hf``: the delta-method standard error of the cumulative hazard
        (see the module docstring), on the scale ``bound_type`` as for
        ``CoxPH.cb`` (``"log-log"``, the default, bounds ``log H``;
        ``"log"`` bounds ``log S``; ``"plain"`` bounds ``S``). With the
        additive part the baseline alone these are ``CoxPH``'s bounds
        (Breslow ties).

        Only Wald bounds are available: there is no likelihood for a
        likelihood-ratio bound and no bootstrap is built; refit the model
        to resampled rows for one.
        """
        check_option("on", on, _ON)
        on = _ON_ALIASES.get(on, on)
        check_option("bound", bound, BOUNDS)
        check_option("bound_type", bound_type, _BOUND_TYPES)
        if method != "wald":
            raise ValueError(
                "The Cox-Aalen model's curves have Wald bounds only "
                "(method='wald'): there is no likelihood for a "
                "likelihood-ratio bound and no bootstrap is built; refit "
                f"the model to resampled rows for one. Got method={method!r}."
            )
        check_alpha_ci(alpha_ci)
        x_arr = np.asarray(x, dtype=float)
        shape = x_arr.shape
        t = np.atleast_1d(x_arr).ravel()
        H_star, _, X, Zm, t, risk = self._cumulative_hazard(t, Z)
        if t.size != x_arr.size:
            shape = t.shape
        se = self._H_se(t, X, Zm, H_star, risk)
        z = norm.isf(alpha_ci / 2 if bound == "two-sided" else alpha_ci)
        lo, hi = survival_bounds(H_star, se, z, bound_type)
        if bound != "two-sided":
            upper_H = (bound == "upper") != (on == "sf")
            lo = hi = hi if upper_H else lo
        if on == "sf":
            lo, hi = np.exp(-hi), np.exp(-lo)
        elif on == "ff":
            lo, hi = -np.expm1(-lo), -np.expm1(-hi)
        if bound == "two-sided":
            return np.stack([lo, hi], axis=-1).reshape(shape + (2,))
        return lo.reshape(shape)

    # -- concordance ------------------------------------------------------

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        X, Zm = self._rows(Z)
        A = np.where(np.isnan(self.A), 0.0, self.A)
        last = A[-1] if A.shape[0] else np.zeros(X.shape[1])
        risk = (X @ last) * np.exp(Zm @ self._coef())
        return np.broadcast_to(risk, x.shape)

    def _concordance_data(self) -> tuple | None:
        if self._fit_data is None:
            return None
        x, c, n, _, Z = self._fit_data
        return (x, c, n, Z)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-serialisable dict: the
        event times, ``A`` and its covariances, ``beta`` and its
        covariances, the additive and multiplicative columns, the aliased
        columns and the covariates' names. Every prediction and bound
        round-trips; the fitted rows (for :meth:`concordance` without
        data) are not stored.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict[str, Any] = {
            "model": "CoxAalenModel",
            "x": np.asarray(self.x, dtype=float).tolist(),
            "A": np.asarray(self.A, dtype=float).tolist(),
            "A_cov": np.asarray(self._A_cov, dtype=float).tolist(),
            "A_beta_cov": np.asarray(self._A_beta_cov, dtype=float).tolist(),
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "covariance": np.asarray(self._cov, dtype=float).tolist(),
            "robust_covariance": np.asarray(
                self._robust_cov, dtype=float
            ).tolist(),
            "additive": self.additive.tolist(),
            "multiplicative": self.multiplicative.tolist(),
            "aliased": self.aliased.tolist(),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "stopped_at": (
                None if np.isnan(self.stopped_at) else float(self.stopped_at)
            ),
        }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> CoxAalenModel:
        """
        Rebuild a Cox-Aalen model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(model_dict, "CoxAalenModel", "a Cox-Aalen model")
        out = cls()
        out.additive = np.array(model_dict["additive"], dtype=int)
        out.multiplicative = np.array(model_dict["multiplicative"], dtype=int)
        q, p = out.additive.size + 1, out.multiplicative.size
        out.x = np.array(model_dict["x"], dtype=float)
        k = out.x.size
        out.A = np.array(model_dict["A"], dtype=float).reshape(k, q)
        out._A_cov = np.array(model_dict["A_cov"], dtype=float).reshape(
            k, q, q
        )
        out._A_beta_cov = np.array(
            model_dict["A_beta_cov"], dtype=float
        ).reshape(k, q, p)
        out.beta = np.array(model_dict["beta"], dtype=float).reshape(p)
        out.params = out.coef = out.beta
        out._cov = np.array(model_dict["covariance"], dtype=float).reshape(
            p, p
        )
        out._robust_cov = np.array(
            model_dict["robust_covariance"], dtype=float
        ).reshape(p, p)
        out._aliased = np.array(model_dict["aliased"], dtype=int)
        out.n_iter = int(model_dict["n_iter"])
        out.converged = bool(model_dict["converged"])
        stop = model_dict.get("stopped_at")
        out.stopped_at = float("nan") if stop is None else float(stop)
        restore_covariate_meta(out, model_dict)
        return out


class CoxAalen_(FitterRepr):
    r"""
    The Cox-Aalen model (Scheike and Zhang 2002): multiplicative Cox terms
    acting on an additive Aalen baseline with time-varying effects,

    .. math::
        h(x \mid Z) = \{X' \alpha(x)\}\, e^{\beta' Z_m}.

    ``additive`` names the columns of ``Z`` in the additive part (with the
    baseline); the others are multiplicative. With none additive it is
    ``CoxPH`` (Breslow ties); with every column additive, ``AalenAdditive``.
    ``CoxAalen`` is an instance of this class; its ``fit`` returns a
    :class:`CoxAalenModel`.
    """

    fitter_kind = "semi-parametric Cox-Aalen fitter"

    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        additive: Any = None,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> CoxAalenModel:
        """
        Fit the Cox-Aalen model.

        Parameters
        ----------
        x : array like
            The times of the events and censorings, at or above 0.
        Z : array like
            The covariates, one row per observation (both parts). Rows with
            a missing or infinite covariate are dropped, with a warning. A
            constant column, or one that is a linear combination of the
            others, is aliased: its coefficient (or column of ``A``) is
            ``nan``, predictions take it as 0, and one warning names it.
        c : array like, optional
            Censoring flags: 0 observed (an event), 1 right censored.
            Defaults to all observed.
        n : array like, optional
            The count of each row. Defaults to 1 each.
        tl : array like, optional
            Each row's entry time (left truncation): it is at risk on
            ``(tl, x]``. Defaults to 0.
        additive : int, str or list, optional
            The columns of ``Z`` whose effects are additive and change with
            time, by position (or by name for a DataFrame ``Z``). The
            others are multiplicative (hazard ratios). Defaults to none:
            the additive part is the baseline alone, Cox's model.
        tol : float, optional
            Newton's method stops when a step moves every coefficient by
            less than ``tol`` (relative to their size, plus 1). Default
            1e-10.
        max_iter : int, optional
            The most Newton steps. Default 100.

        Returns
        -------
        CoxAalenModel
            The fitted model; a warning is raised if Newton's method did
            not converge.

        Examples
        --------
        A binary covariate doubling the hazard, and a second adding 0.1 to
        it for the first two time units only:

        >>> import numpy as np
        >>> from surpyval import CoxAalen
        >>> rng = np.random.default_rng(5)
        >>> N = 2000
        >>> z, a = rng.binomial(1, 0.5, (2, N)).astype(float)
        >>> e = rng.exponential(size=N) / np.exp(np.log(2) * z)
        >>> early = 0.1 + 0.1 * a
        >>> t = np.where(e < 2 * early, e / early, 2 + (e - 2 * early) / 0.1)
        >>> c = (t > 8).astype(int)
        >>> model = CoxAalen.fit(np.minimum(t, 8), np.c_[z, a], c=c,
        ...                      additive=[1])
        >>> model.beta.round(2)
        array([0.71])
        >>> model.cumulative_coefficients([2, 6]).round(2)
              baseline  coef_1
        time
        2.0       0.22    0.18
        6.0       0.59    0.19

        The truth is ``beta = log 2 = 0.69``, ``A_0(t) = 0.1 t`` and
        ``A_1(t) = 0.1 min(t, 2)``.
        """
        names: list[str] = []
        if type(Z).__name__ == "DataFrame":
            names = [str(col) for col in getattr(Z, "columns")]
        width = int(np.shape(Z)[1]) if np.ndim(Z) == 2 else 1
        positions = _additive_positions(additive, width, names)
        return self._fit(x, Z, c, n, tl, positions, tol, max_iter)

    @dataframe_covariates
    def _fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None,
        n: npt.ArrayLike | None,
        tl: npt.ArrayLike | None,
        additive: list,
        tol: float,
        max_iter: int,
    ) -> CoxAalenModel:
        x_a, c_a, n_a, tl_a, Z_a = _validate(x, Z, c, n, tl)
        width = Z_a.shape[1]
        add = np.asarray(additive, dtype=int)
        mult = np.setdiff1d(np.arange(width), add)
        aliased = _aliased(Z_a, n_a)
        if aliased.size:
            warn_aliased(
                aliased,
                "they do not vary beyond a combination of the other "
                "columns (a constant column, which the baseline absorbs, "
                "or a linear combination of the others)",
            )
        add_k = np.setdiff1d(add, aliased)
        mult_k = np.setdiff1d(mult, aliased)
        X = np.column_stack([np.ones(x_a.size), Z_a[:, add_k]])
        Zm = Z_a[:, mult_k]
        times = _event_times(x_a, c_a, n_a, tl_a, X)
        if times.size == 0:
            raise ValueError(
                "CoxAalen could not estimate the additive part at any event "
                "time: too few units are at risk to separate the additive "
                "covariates. Make fewer of them additive."
            )
        stopped_at = warn_if_stopped(
            "CoxAalen",
            x_a,
            c_a,
            n_a,
            tl_a,
            times,
            "Make fewer covariates additive, or end the analysis at that "
            "time.",
        )
        beta_k, pieces, n_iter, converged = _solve(
            x_a, c_a, n_a, tl_a, X, Zm, times, tol, max_iter
        )
        if not converged:
            warnings.warn(
                f"CoxAalen's Newton's method did not converge in {max_iter} "
                "steps; returning the last estimate. Raise max_iter, or "
                "make fewer covariates additive."
            )
        q, p = X.shape[1], Zm.shape[1]
        scale = np.exp(-pieces["shift"])
        A = np.cumsum(pieces["dA"], axis=0) * scale
        C = np.cumsum(pieces["dC"], axis=0) * scale**2
        K = np.cumsum(pieces["dK"], axis=0) * scale
        P = np.cumsum(pieces["dP"], axis=0) * scale
        if p:
            I_inv = np.linalg.inv(pieces["I"])
            V = I_inv @ pieces["S"] @ I_inv.T
            B = _robust_middle(
                x_a, c_a, n_a, tl_a, X, Zm, beta_k, times, pieces
            )
            V_rob = I_inv @ B @ I_inv.T
            KI = np.einsum("kap,pq->kaq", K, I_inv.T)
            A_cov = (
                C
                - np.einsum("kap,kbp->kab", KI, P)
                - np.einsum("kap,kbp->kab", P, KI)
                + np.einsum("kap,pq,kbq->kab", P, V, P)
            )
            AB = KI - np.einsum("kap,pq->kaq", P, V)
        else:
            V = V_rob = np.zeros((0, 0))
            A_cov, AB = C, np.zeros((times.size, q, 0))
        A_cov = 0.5 * (A_cov + np.swapaxes(A_cov, 1, 2))

        # Back to every column, nan at the aliased ones.
        add_pos = np.r_[0, 1 + np.searchsorted(add, add_k)].astype(int)
        mult_pos = np.searchsorted(mult, mult_k).astype(int)
        Q, Pm, k = add.size + 1, mult.size, times.size
        model = CoxAalenModel()
        model.x = times
        model.A = np.full((k, Q), np.nan)
        model.A[:, add_pos] = A
        model._A_cov = np.full((k, Q, Q), np.nan)
        model._A_cov[np.ix_(np.arange(k), add_pos, add_pos)] = A_cov
        model._A_beta_cov = np.full((k, Q, Pm), np.nan)
        model._A_beta_cov[np.ix_(np.arange(k), add_pos, mult_pos)] = AB
        model.beta = np.full(Pm, np.nan)
        model.beta[mult_pos] = beta_k
        model.params = model.coef = model.beta
        model._cov = np.full((Pm, Pm), np.nan)
        model._cov[np.ix_(mult_pos, mult_pos)] = V
        model._robust_cov = np.full((Pm, Pm), np.nan)
        model._robust_cov[np.ix_(mult_pos, mult_pos)] = V_rob
        model.additive = add
        model.multiplicative = mult
        model._aliased = aliased
        model.n_iter = n_iter
        model.converged = converged
        model._fit_data = (x_a, c_a, n_a, tl_a, Z_a)
        model.stopped_at = stopped_at
        return model

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        additive: Any = None,
        formula: str | None = None,
        tol: float = 1e-10,
        max_iter: int = 100,
    ) -> CoxAalenModel:
        """
        Fit the Cox-Aalen model from a pandas DataFrame. See :meth:`fit`;
        ``Z_cols`` or ``formula`` selects the covariates and ``additive``
        names those (among them) whose effects are additive.
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        check_columns(df, x_col=x_col, c_col=c_col, n_col=n_col, tl_col=tl_col)
        design = FrameDesign(df, Z_cols, formula, drop_missing=True)
        names = list(design.feature_names or [])
        positions = _additive_positions(additive, len(names), names)
        return design.fit(
            self.fit,
            design.column(x_col),
            design.Z,
            c=design.column(c_col),
            n=design.column(n_col),
            tl=design.column(tl_col),
            additive=positions,
            tol=tol,
            max_iter=max_iter,
        )


CoxAalen = CoxAalen_()
