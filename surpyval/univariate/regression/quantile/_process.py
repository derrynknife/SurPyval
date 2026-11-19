r"""
The quantile process of a censored quantile regression, fitted on a grid.

Both estimators fit the conditional quantiles of the log time,
:math:`Q_{\log T}(\tau \mid Z) = \beta_0(\tau) + \beta(\tau)' Z`, at the
levels :math:`\tau_1 < \tau_2 < \dots` of a grid, one after the other, each
step a linear programme:

- **Peng and Huang (2008)** solve the martingale estimating equation of the
  counting process at :math:`\tau_j`,

  .. math::
      \sum_i Z_i \Big[ \delta_i I(Y_i \le Z_i'\beta_j)
      - \sum_{k < j} I(Y_i \ge Z_i'\beta_k)
      \{H(\tau_{k+1}) - H(\tau_k)\} \Big] = 0,
      \qquad H(\tau) = -\log(1 - \tau),

  with every unit at risk at :math:`\tau_0 = 0`, as the minimum of an L1
  objective over the events plus a linear term.
- **Portnoy (2003)** redistributes each censored unit's weight to the right
  once the quantile line has passed it: from the level :math:`\hat\tau_i` at
  which it was crossed, the unit keeps the weight
  :math:`(\tau - \hat\tau_i) / (1 - \hat\tau_i)` at its censoring time and
  puts the rest at :math:`+\infty`, and the weighted quantile regression is
  solved at each level. On the grid, a censored unit on or below the line
  at :math:`\tau_j` is crossed there, and the level is refitted until no
  more are (the grid form of Neocleous, Vanden Branden and Portnoy, 2006).

Each linear programme is solved through its dual (as many constraints as
coefficients) by the HiGHS dual simplex. A point at :math:`+\infty` is not a
row of it: its check loss is linear in the coefficients, so it enters the
objective as a linear term. The process stops at the first level whose
programme has no finite solution, or whose fitted quantile lies above every
event: past it, the data say nothing more about the quantiles.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

#: How close (on the standardised log-time scale) a unit's log time must be
#: to the fitted quantile to count as on it.
ON_LINE = 1e-9

#: The estimators, by the name ``fit`` takes.
METHODS = ("peng-huang", "portnoy")


def _weighted_l1(
    X: npt.NDArray,
    y: npt.NDArray,
    above: npt.NDArray,
    below: npt.NDArray,
    linear: npt.NDArray,
) -> npt.NDArray | None:
    r"""The coefficients :math:`b` that minimise

    .. math::
        \sum_i a_i (y_i - x_i'b)^+ + c_i (y_i - x_i'b)^- + g'b,

    with ``a = above``, ``c = below`` and ``g = linear``, or ``None`` where
    the minimum is not finite. Solved as the dual programme, :math:`\max
    y'l` subject to :math:`X'l = g` and :math:`-c \le l \le a`, whose
    equality multipliers are :math:`-b`."""
    keep = (above > 0) | (below > 0)
    result = linprog(
        -y[keep],
        A_eq=X[keep].T,
        b_eq=linear,
        bounds=np.column_stack([-below[keep], above[keep]]),
        method="highs-ds",
        options={"presolve": False},
    )
    if result.status != 0:
        return None
    return -np.asarray(result.eqlin.marginals, dtype=float)


def _informative(
    resid: npt.NDArray, delta: npt.NDArray, n: npt.NDArray
) -> bool:
    """Whether an event lies on or above the fitted quantile."""
    return bool(np.any((delta == 1) & (n > 0) & (resid >= -ON_LINE)))


def peng_huang(
    Y: npt.NDArray,
    delta: npt.NDArray,
    X: npt.NDArray,
    n: npt.NDArray,
    taus: npt.NDArray,
) -> npt.NDArray:
    """Peng and Huang's estimates at the levels ``taus``, one row each, as
    far as the process goes. ``X`` holds the intercept column."""
    dH = np.diff(np.concatenate([[0.0], -np.log1p(-taus)]))
    events = np.where(delta == 1, n, 0.0)
    observed = events @ X
    exposure = np.zeros(Y.size)
    at_risk = np.ones(Y.size)
    out: list[npt.NDArray] = []
    for j in range(taus.size):
        exposure = exposure + at_risk * dH[j]
        linear = observed - 2.0 * (n * exposure) @ X
        b = _weighted_l1(X, Y, events, events, linear)
        if b is None:
            break
        resid = Y - X @ b
        if not _informative(resid, delta, n):
            break
        out.append(b)
        at_risk = (resid >= -ON_LINE).astype(float)
    return np.asarray(out, dtype=float).reshape(-1, X.shape[1])


def portnoy(
    Y: npt.NDArray,
    delta: npt.NDArray,
    X: npt.NDArray,
    n: npt.NDArray,
    taus: npt.NDArray,
) -> npt.NDArray:
    """Portnoy's estimates at the levels ``taus``, one row each, as far as
    the process goes. ``X`` holds the intercept column."""
    censored = delta == 0
    crossed_at = np.full(Y.size, np.nan)
    out: list[npt.NDArray] = []
    for tau in taus:
        while True:
            crossed = ~np.isnan(crossed_at)
            start = np.where(crossed, crossed_at, 0.0)
            w = np.where(crossed, (tau - start) / (1.0 - start), 1.0)
            # The weight at +infinity: its check loss is -tau x'b.
            linear = -tau * ((n * (1.0 - w) * crossed) @ X)
            b = _weighted_l1(X, Y, tau * w * n, (1.0 - tau) * w * n, linear)
            if b is None:
                return np.asarray(out, dtype=float).reshape(-1, X.shape[1])
            resid = Y - X @ b
            newly = censored & ~crossed & (resid <= ON_LINE)
            if not newly.any():
                break
            crossed_at[newly] = tau
        if not _informative(resid, delta, n):
            break
        out.append(b)
    return np.asarray(out, dtype=float).reshape(-1, X.shape[1])


def fit_process(
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    n: npt.NDArray,
    taus: npt.NDArray,
    method: str,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The quantile process of the log times ``Y`` on the covariates ``Z``
    (no aliased column) at the grid ``taus``, as far as it goes:
    ``(taus, intercepts, slopes)``.

    The log times and each covariate are standardised by their (count
    weighted) mean and standard deviation before the programmes are solved,
    and the coefficients mapped back, so that a change of time unit, or of a
    covariate's scale or origin, poses the same programmes and moves the
    coefficients only as the model says it should."""
    w = n / n.sum()
    loc = float(w @ Y)
    scale = float(np.sqrt(w @ (Y - loc) ** 2)) or 1.0
    centre = w @ Z
    spread = np.sqrt(w @ (Z - centre) ** 2)
    spread = np.where(spread > 0, spread, 1.0)
    X = np.column_stack([np.ones(Y.size), (Z - centre) / spread])
    fit = peng_huang if method == "peng-huang" else portnoy
    theta = fit((Y - loc) / scale, delta, X, n, taus)
    slopes = scale * theta[:, 1:] / spread
    intercepts = loc + scale * theta[:, 0] - slopes @ centre
    return taus[: theta.shape[0]], intercepts, slopes
