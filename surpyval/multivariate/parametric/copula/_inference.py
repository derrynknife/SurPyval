"""The parameter covariance of a fitted copula model (#540).

A fitted :class:`CopulaModel` has the copula's parameters and its two
margins' parameters. Their covariance is the inverse of the information,
in one of two forms, by how the model was fitted:

- ``how="MLE"`` (one joint search): the inverse of the Hessian of the joint
  negative log-likelihood in every parameter the search estimated, the
  copula's and the margins' together.
- ``how="IFM"`` (inference functions for margins: each margin first, then
  the copula with the margins held): the Godambe (sandwich) information of
  the stacked estimating equations (Joe 2005, "Asymptotic efficiency of
  the two-stage estimation method for copula-based models", J. Multivariate
  Analysis 94; Joe 1997, section 10.1). With ``psi`` the stacked score of
  each row (each margin's own score in its parameters, then the copula
  stage's score in the copula's), ``H = -sum_i n_i d psi_i / d theta`` --
  block triangular, the margins' scores being free of the copula's
  parameters and of each other's -- and ``J = sum_i n_i psi_i psi_i'``,
  the covariance is ``H^-1 J H^-T``. The copula stage's Hessian alone
  treats the margins as known and understates the copula's variance; the
  ``H`` blocks of the copula's score in the margins' parameters carry the
  margins' uncertainty into it.

The copula likelihood is written for numpy (its rows are split by their
censoring codes), not for autograd, so the derivatives are central finite
differences (``numerical_hessian``), with the step of the regression
models' covariance: ``eps**(1/3) * max(|p|, 1e-2)``, shortened for a
parameter within a few steps of a bound of its space. The covariance is
inverted in step-scaled coordinates, as theirs is.

The parameters are laid out as the model reports them: the copula's (its
``parameter_names``), then each margin's as its own ``covariance()`` lays
them out -- the distribution's parameters, then the limited-failure
proportion ``p`` and the zero-inflation fraction ``f0`` where it has them.
An offset ``gamma`` is held at its estimate (a threshold is not regular,
and a univariate fit holds it too), as are a parameter fixed at fit time
and every parameter of a margin the fit did not estimate (one passed to an
IFM fit already fitted): each has a zero row and column. A parameter on a
bound of its space (an AMH ``theta`` of 1) has no Wald variance: its row
and column are ``nan``, and the rest are conditional on it.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as onp
import numpy.typing as npt

from surpyval.utils.linalg import numerical_hessian

from .copula import _ff_where_finite, _is_nonparametric

#: The finite-difference scale of every derivative here.
_H = onp.finfo(float).eps ** (1.0 / 3.0)


class ParameterLayout:
    """The parameter vector of a fitted copula model, in the order its
    covariance is reported (see the module docstring), and how to rebuild
    the model's parts from it.

    Attributes
    ----------
    values : numpy.ndarray
        Every entry's fitted value.
    bounds : list of tuple
        Each entry's ``(lower, upper)``; ``None`` for no bound.
    names : list of str
        Each entry's label: the copula's parameter names, then
        ``"margin d: name"``.
    estimated : numpy.ndarray of bool
        Whether the fit estimated the entry.
    n_cop : int
        The number of the copula's entries (the first ones).
    slices : list
        Each margin's entries (``None`` for a non-parametric margin).
    """

    def __init__(self, model: Any, margins_estimated: Any) -> None:
        copula = model.copula
        self.copula = copula
        self.margins = list(model.margins)
        self.n_cop = len(copula.parameter_names)
        values = list(onp.asarray(model.params, dtype=float))
        bounds = list(copula.bounds)
        names = list(copula.parameter_names)
        estimated = [True] * self.n_cop
        self.slices: list = []
        for d, (m, fitted) in enumerate(zip(self.margins, margins_estimated)):
            if _is_nonparametric(m) or not hasattr(m, "dist"):
                self.slices.append(None)
                continue
            start = len(values)
            entries = _margin_entries(m)
            fixed = _fixed_names(m)
            for name, value, bound in entries:
                values.append(value)
                bounds.append(bound)
                names.append(f"margin {d}: {name}")
                estimated.append(bool(fitted) and name not in fixed)
            self.slices.append(slice(start, len(values)))
        self.values = onp.asarray(values, dtype=float)
        self.bounds = bounds
        self.names = names
        self.estimated = onp.asarray(estimated, dtype=bool)

    @property
    def size(self) -> int:
        return self.values.size

    def on_bound(self) -> npt.NDArray:
        """Whether each entry is on (or beyond) a bound of its space."""
        out = onp.zeros(self.size, dtype=bool)
        for i, (low, high) in enumerate(self.bounds):
            v = self.values[i]
            out[i] = (low is not None and v <= low) or (
                high is not None and v >= high
            )
        return out

    def steps(self) -> npt.NDArray:
        """The finite-difference step of each entry: ``eps**(1/3) *
        max(|p|, 1e-2)``, or ``eps**(1/3)`` times its distance from a bound
        closer than ten such steps (the regression models' rule), so no
        difference leaves the parameter's space."""
        step = _H * onp.maximum(onp.abs(self.values), 1e-2)
        for i, (low, high) in enumerate(self.bounds):
            gap = min(
                self.values[i] - low if low is not None else onp.inf,
                high - self.values[i] if high is not None else onp.inf,
            )
            if 0 < gap < 10 * step[i]:
                step[i] = _H * gap
        return step

    def build(self, vector: npt.NDArray) -> "tuple[npt.NDArray, list] | None":
        """``(copula parameters, margins)`` at the full parameter
        ``vector``, or ``None`` where a margin's parameters are outside
        its space."""
        theta = onp.asarray(vector[: self.n_cop], dtype=float)
        margins = []
        for m, sl in zip(self.margins, self.slices):
            if sl is None:
                margins.append(m)
                continue
            built = _rebuild_margin(m, onp.asarray(vector[sl], dtype=float))
            if built is None:
                return None
            margins.append(built)
        return theta, margins


def _margin_entries(margin: Any) -> list:
    """``(name, value, bounds)`` of each entry of a parametric margin's
    covariance: the distribution's parameters, then ``p`` and ``f0``."""
    dist = margin.dist
    out = [
        (name, float(value), tuple(bound))
        for name, value, bound in zip(
            dist.parameter_names, margin.params, dist.bounds
        )
    ]
    if margin.lfp:
        out.append((margin.lfp_name, float(margin.p), (0, 1)))
    if margin.zi:
        out.append(("f0", float(margin.f0), (0, 1)))
    return out


def _fixed_names(margin: Any) -> set:
    """The names of the parameters the margin was fitted with fixed."""
    fixed = margin._user_fixed_idx() if hasattr(margin, "param_map") else ()
    names = {i: name for name, i in getattr(margin, "param_map", {}).items()}
    return {names[i] for i in fixed if i in names}


def _rebuild_margin(margin: Any, entries: npt.NDArray) -> Any:
    """The margin with its covariance entries set to ``entries`` (its
    offset held), or ``None`` outside the parameters' space."""
    k = len(margin.params)
    params = entries[:k]
    i = k
    p = f0 = None
    if margin.lfp:
        p, i = float(entries[i]), i + 1
    if margin.zi:
        f0 = float(entries[i])
    gamma = float(margin.gamma) if margin.offset else None
    try:
        return margin.dist.from_params(params, gamma=gamma, p=p, f0=f0)
    except ValueError:
        return None


# -- the per-row log-likelihoods -----------------------------------------
def margin_rows(margin: Any, dimension: tuple) -> npt.NDArray:
    """Each row's log-likelihood under one margin, as its univariate fit
    has it: the density for an observed entry, the survival for a
    right-censored one, the CDF for a left-censored one, the mass of the
    interval for an interval-censored one, each divided by the mass of
    the row's truncation window."""
    x, c, xl, xr, tl, tr = dimension
    x = onp.asarray(x, dtype=float)
    c = onp.asarray(c, dtype=int)
    out = onp.zeros(x.shape)
    with onp.errstate(divide="ignore", invalid="ignore"):
        for code, f in (
            (0, margin.df),
            (1, margin.sf),
            (-1, margin.ff),
        ):
            rows = c == code
            if rows.any():
                out[rows] = onp.log(onp.asarray(f(x[rows]), dtype=float))
        rows = c == 2
        if rows.any():
            lo = _ff_where_finite(margin, onp.asarray(xl)[rows], 0.0)
            hi = _ff_where_finite(margin, onp.asarray(xr)[rows], 1.0)
            out[rows] = onp.log(hi - lo)
        tl = onp.asarray(tl, dtype=float)
        tr = onp.asarray(tr, dtype=float)
        if onp.isfinite(tl).any() or onp.isfinite(tr).any():
            mass = _ff_where_finite(margin, tr, 1.0) - _ff_where_finite(
                margin, tl, 0.0
            )
            out = out - onp.log(mass)
    return out


def _copula_rows(
    layout: ParameterLayout, data: Any, vector: npt.NDArray
) -> npt.NDArray:
    """Each row's log-likelihood under the copula stage (the joint
    likelihood, margins' densities included), or ``nan`` where the
    margins are outside their space."""
    parts = layout.build(vector)
    if parts is None:
        return onp.full(len(data.n), onp.nan)
    theta, margins = parts
    dims = [
        layout.copula._prepare_dim(margins[d], *data.dimension(d))
        for d in range(data.D)
    ]
    with onp.errstate(all="ignore"):
        return onp.asarray(
            layout.copula._pair_loglik(theta, dims[0], dims[1]), dtype=float
        )


# -- the covariance ------------------------------------------------------
def joint_covariance(
    layout: ParameterLayout, data: Any, how: str
) -> tuple[npt.NDArray, bool]:
    """``(covariance, ok)``: the covariance of every entry of ``layout``
    (see the module docstring), and whether the information could be
    evaluated and inverted (where not, the estimated block is ``nan``)."""
    n_all = layout.size
    cov = onp.zeros((n_all, n_all))
    boundary = layout.on_bound() & layout.estimated
    free = onp.flatnonzero(layout.estimated & ~boundary)
    cov[boundary, :] = onp.nan
    cov[:, boundary] = onp.nan
    if free.size == 0:
        return cov, True
    step = layout.steps()
    weights = onp.asarray(data.n, dtype=float)
    margins_estimated = layout.estimated[layout.n_cop :].any()
    if how == "IFM" and margins_estimated:
        block = _godambe(layout, data, free, step, weights)
    else:
        block = _inverse_hessian(layout, data, free, step, weights)
    ok = block is not None
    cov[onp.ix_(free, free)] = onp.nan if block is None else block
    return cov, ok


def _neg_ll_in(
    rows: Callable[[npt.NDArray], npt.NDArray],
    base: npt.NDArray,
    free: npt.NDArray,
    weights: npt.NDArray,
) -> Callable[[npt.NDArray], float]:
    """The weighted negative log-likelihood of ``rows`` as a function of
    the ``free`` entries of ``base``."""

    def f(values: npt.NDArray) -> float:
        vector = base.copy()
        vector[free] = values
        return -float(onp.sum(weights * rows(vector)))

    return f


def _inverse_hessian(
    layout: ParameterLayout,
    data: Any,
    free: npt.NDArray,
    step: npt.NDArray,
    weights: npt.NDArray,
) -> "npt.NDArray | None":
    """The inverse of the joint likelihood's Hessian in the ``free``
    entries, or ``None`` where it is not finite or not invertible."""
    f = _neg_ll_in(
        lambda v: _copula_rows(layout, data, v), layout.values, free, weights
    )
    H = numerical_hessian(f, layout.values[free], step[free])
    return _scaled_inverse(H, step[free])


def _scaled_inverse(
    H: npt.NDArray, step: npt.NDArray
) -> "npt.NDArray | None":
    """``H^-1``, inverted in step-scaled coordinates (a parameter orders of
    magnitude from another leaves ``H`` too ill-conditioned to invert
    directly), or ``None``."""
    if not onp.all(onp.isfinite(H)):
        return None
    scale = onp.outer(step, step)
    try:
        inv = onp.linalg.inv(H * scale) * scale
    except onp.linalg.LinAlgError:
        return None
    return inv if onp.all(onp.isfinite(inv)) else None


def _row_scores(
    rows: Callable[[npt.NDArray], npt.NDArray],
    base: npt.NDArray,
    entries: npt.NDArray,
    step: npt.NDArray,
) -> npt.NDArray:
    """Each row's score (the derivative of its log-likelihood) in each of
    ``entries``, by central differences: one column per entry."""
    cols = []
    for k in entries:
        up, down = base.copy(), base.copy()
        up[k] += step[k]
        down[k] -= step[k]
        cols.append((rows(up) - rows(down)) / (2.0 * step[k]))
    return onp.column_stack(cols)


def _godambe(
    layout: ParameterLayout,
    data: Any,
    free: npt.NDArray,
    step: npt.NDArray,
    weights: npt.NDArray,
) -> "npt.NDArray | None":
    """The Godambe covariance ``H^-1 J H^-T`` of the two-stage (IFM)
    estimate in the ``free`` entries (see the module docstring), or
    ``None`` where it cannot be formed."""
    base = layout.values
    cop = free[free < layout.n_cop]
    pos = {int(k): j for j, k in enumerate(free)}
    H = onp.zeros((free.size, free.size))
    scores = onp.zeros((len(weights), free.size))

    def copula_rows(v: npt.NDArray) -> npt.NDArray:
        return _copula_rows(layout, data, v)

    # The copula stage: its score in the copula's parameters, and the
    # derivatives of that score in every free parameter (the copula's own
    # and the margins', which carry the margins' uncertainty).
    if cop.size:
        f = _neg_ll_in(copula_rows, base, free, weights)
        full = numerical_hessian(f, base[free], step[free])
        rows_c = [pos[int(k)] for k in cop]
        H[rows_c, :] = full[rows_c, :]
        scores[:, rows_c] = _row_scores(copula_rows, base, cop, step)
    # Each margin: its own score and Hessian in its own parameters.
    for d, sl in enumerate(layout.slices):
        if sl is None:
            continue
        mine = free[(free >= sl.start) & (free < sl.stop)]
        if mine.size == 0:
            continue
        dimension = data.dimension(d)

        def rows_d(
            v: npt.NDArray, sl: slice = sl, d: int = d, dim: tuple = dimension
        ) -> npt.NDArray:
            parts = layout.build(v)
            if parts is None:
                return onp.full(len(weights), onp.nan)
            return margin_rows(parts[1][d], dim)

        f = _neg_ll_in(rows_d, base, mine, weights)
        idx = [pos[int(k)] for k in mine]
        H[onp.ix_(idx, idx)] = numerical_hessian(f, base[mine], step[mine])
        scores[:, idx] = _row_scores(rows_d, base, mine, step)
    H_inv = _scaled_inverse(H, step[free])
    if H_inv is None or not onp.all(onp.isfinite(scores)):
        return None
    J = scores.T @ (weights[:, None] * scores)
    return H_inv @ J @ H_inv.T


def godambe_parts(
    layout: ParameterLayout, data: Any
) -> "tuple[npt.NDArray, npt.NDArray]":
    """``(naive, godambe)`` covariances of the copula's free parameters
    under IFM: the copula stage's inverse Hessian with the margins held
    (what treating them as known gives) and the Godambe covariance. For
    the documentation and its tests, which compare the two."""
    step = layout.steps()
    weights = onp.asarray(data.n, dtype=float)
    free = onp.flatnonzero(layout.estimated & ~layout.on_bound())
    cop = free[free < layout.n_cop]
    naive = _inverse_hessian(layout, data, cop, step, weights)
    full = _godambe(layout, data, free, step, weights)
    if naive is None or full is None:
        nan = onp.full((cop.size, cop.size), onp.nan)
        return nan, nan
    return naive, full[: cop.size, : cop.size]


def jacobian(
    func: Callable[[npt.NDArray], npt.NDArray],
    layout: ParameterLayout,
    entries: npt.NDArray,
) -> npt.NDArray:
    """The derivative of ``func`` (a vector function of the full
    parameter vector) in each of ``entries``, by central differences with
    :meth:`ParameterLayout.steps`: one column per entry."""
    step = layout.steps()
    return _row_scores(func, layout.values, entries, step)
