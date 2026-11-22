"""The parameter covariance of a fitted copula model (#540).

A fitted :class:`CopulaModel` has the copula's parameters and its two
margins' parameters. Their covariance is the inverse of the information,
in one of two forms, by how the model was fitted (a fit whose margins
are both empirical CDFs has the rank-based form at the end of this
docstring instead):

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
differences. Their steps come from the likelihood itself, so they do not
depend on the unit of time a margin's parameters are written in: each
entry's scale is the change in it that lowers the joint log-likelihood
by one half with the others held (about a standard error), the Hessians
are ``scale_free_hessian``'s, and the scores and the bounds' gradients
step ``eps**(1/3)`` of that scale, shortened for a parameter within a few
steps of a bound of its space. A step of ``eps**(1/3) * max(|p|, 1e-2)``
(the rule before) is far too long for a margin's scale that is small in
its unit: with times in millionths of the fixture's unit the standard
errors moved by up to 6e-3. The covariance is inverted in step-scaled
coordinates, as the regression models' is.

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

**Rank-based margins.** With both margins the empirical CDF of complete
data (``KaplanMeier`` margins, no censoring, truncation, counts or ties)
the copula's parameters maximise the pseudo-likelihood of the scaled
ranks ``U_i = R_i / (n + 1)``, and their covariance is Genest, Ghoudi and
Rivest's (1995, "A semiparametric estimation procedure of dependence
parameters in multivariate families of distributions", Biometrika 82,
proposition 2.1): with ``s = d log c / d theta`` the score of a row,
``B = -mean(d s / d theta)`` and, for each margin ``j``,

    ``W_ji = mean_k[ 1(U_ji <= U_jk) d s / d u_j (U_k) ]``

(the change in the score from the ranks of margin ``j`` being
estimated), ``Sigma = cov(s_i + W_1i + W_2i)`` and the covariance is
``B^-1 Sigma B^-T / n`` (:func:`pseudo_likelihood_covariance`). The
derivatives are central differences of the copula's log density, in the
parameters as above and in each ``u`` by ``eps**(1/3)`` of its distance
from the nearer end of ``(0, 1)``, the scale the densities vary on there.
The margins are step functions with no parameters, so the covariance is
of the copula's parameters alone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as onp
import numpy.typing as npt

from surpyval.utils.linalg import curvature_scales, scale_free_hessian

from .copula import _LOG_FLOOR, _ff_where_finite, _is_nonparametric

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

    def steps(self, scale: npt.NDArray | None = None) -> npt.NDArray:
        """The finite-difference step of each entry: ``eps**(1/3)`` of its
        ``scale`` (a standard error, or the likelihood's own scale from
        :func:`likelihood_scales`) where that is positive and finite, else
        ``eps**(1/3) * max(|p|, 1e-2)``; or ``eps**(1/3)`` times its
        distance from a bound closer than ten such steps (the regression
        models' rule), so no difference leaves the parameter's space."""
        step = _H * onp.maximum(onp.abs(self.values), 1e-2)
        if scale is not None:
            scale = onp.asarray(scale, dtype=float)
            usable = onp.isfinite(scale) & (scale > 0)
            step = onp.where(usable, _H * onp.where(usable, scale, 1.0), step)
        for i, (low, high) in enumerate(self.bounds):
            gap = min(
                self.values[i] - low if low is not None else onp.inf,
                high - self.values[i] if high is not None else onp.inf,
            )
            if 0 < gap < 10 * step[i]:
                step[i] = _H * gap
        return step

    def build(self, vector: npt.NDArray) -> tuple[npt.NDArray, list] | None:
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
    lfp_p = f0 = None
    if margin.lfp:
        lfp_p, i = float(entries[i]), i + 1
    if margin.zi:
        f0 = float(entries[i])
    gamma = float(margin.gamma) if margin.offset else None
    try:
        return margin.dist.from_params(params, gamma=gamma, lfp_p=lfp_p, f0=f0)
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
            layout.copula._rows_loglik(theta, dims), dtype=float
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
    weights = onp.asarray(data.n, dtype=float)
    step = layout.steps(likelihood_scales(layout, data, free, weights))
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
) -> npt.NDArray | None:
    """The inverse of the joint likelihood's Hessian in the ``free``
    entries, or ``None`` where it is not finite or not invertible."""
    f = _neg_ll_in(
        lambda v: _copula_rows(layout, data, v), layout.values, free, weights
    )
    H = scale_free_hessian(f, layout.values[free], fallback_step=step[free])
    return _scaled_inverse(H, step[free])


def likelihood_scales(
    layout: ParameterLayout,
    data: Any,
    free: npt.NDArray,
    weights: npt.NDArray,
) -> npt.NDArray:
    """Each entry's scale in the joint likelihood: the change in it, the
    others held, that lowers the log-likelihood by one half (a standard
    error, were the likelihood quadratic and the entry alone); ``nan``
    for an entry not in ``free`` or where there is no such change (a
    flat or one-sided direction). It moves with the unit an entry is
    written in, so steps in it do not depend on that unit."""
    out = onp.full(layout.size, onp.nan)
    if free.size == 0:
        return out
    f = _neg_ll_in(
        lambda v: _copula_rows(layout, data, v), layout.values, free, weights
    )
    out[free] = curvature_scales(f, layout.values[free])
    return out


def _scaled_inverse(H: npt.NDArray, step: npt.NDArray) -> npt.NDArray | None:
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
) -> npt.NDArray | None:
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
        full = scale_free_hessian(f, base[free], fallback_step=step[free])
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
        H[onp.ix_(idx, idx)] = scale_free_hessian(
            f, base[mine], fallback_step=step[mine]
        )
        scores[:, idx] = _row_scores(rows_d, base, mine, step)
    H_inv = _scaled_inverse(H, step[free])
    if H_inv is None or not onp.all(onp.isfinite(scores)):
        return None
    J = scores.T @ (weights[:, None] * scores)
    return H_inv @ J @ H_inv.T


def jacobian(
    func: Callable[[npt.NDArray], npt.NDArray],
    layout: ParameterLayout,
    entries: npt.NDArray,
    scale: npt.NDArray | None = None,
) -> npt.NDArray:
    """The derivative of ``func`` (a vector function of the full
    parameter vector) in each of ``entries``, by central differences with
    :meth:`ParameterLayout.steps` (``eps**(1/3)`` of each entry's
    ``scale``, its standard error, where given): one column per entry."""
    step = layout.steps(scale)
    return _row_scores(func, layout.values, entries, step)


# -- rank-based margins ---------------------------------------------------
def rank_pseudo_observations(
    model: Any,
) -> tuple[npt.NDArray, npt.NDArray] | str:
    """The pseudo-observations ``(u, v)`` of a fit whose covariance is the
    rank-based one (see the module docstring), or the reason, as text,
    that it is not: a margin that is parametric or was passed already
    fitted, incomplete data, tied values, or margins that are not the
    empirical CDF."""
    data = model.data
    if not all(_is_nonparametric(m) for m in model.margins):
        return (
            "one margin is parametric and the other is not, which the "
            "rank-based variance does not cover"
        )
    if not all(model._margins_estimated):
        return (
            "a non-parametric margin was passed already fitted, so its "
            "ranks are not those of the data the copula was fitted to"
        )
    counts = onp.asarray(data.n)
    if (onp.asarray(data.c) != 0).any():
        return "the data are censored"
    if onp.isfinite(onp.asarray(data.t, dtype=float)).any():
        return "the data are truncated"
    if (counts != 1).any():
        return "the data have counts `n` other than 1 (tied rows)"
    x = onp.asarray(data.x, dtype=float)
    n_rows = x.shape[0]
    out = []
    for d in range(2):
        column = x[:, d]
        if onp.unique(column).size != n_rows:
            tied = n_rows - onp.unique(column).size
            return (
                f"series {d} has tied values ({tied} of its {n_rows} rows "
                "repeat an earlier value), and ranks with ties are not the "
                "ranks of a continuous margin"
            )
        ranks = onp.empty(n_rows)
        ranks[onp.argsort(column)] = onp.arange(1, n_rows + 1)
        u = model.copula._prepare_dim(model.margins[d], *data.dimension(d))[
            "u"
        ]
        if not onp.allclose(u, ranks / (n_rows + 1.0), rtol=0, atol=1e-9):
            return (
                f"margin {d} is not the empirical CDF of its series (fit "
                "`KaplanMeier` margins for the ranks)"
            )
        out.append(ranks / (n_rows + 1.0))
    return out[0], out[1]


def pseudo_likelihood_covariance(
    layout: ParameterLayout, u: npt.NDArray, v: npt.NDArray
) -> tuple[npt.NDArray, bool]:
    """``(covariance, ok)``: Genest, Ghoudi and Rivest's covariance of the
    copula parameters that maximise the pseudo-likelihood of the scaled
    ranks ``u`` and ``v`` (see the module docstring), and whether it could
    be formed (where not, the estimated block is ``nan``). ``layout`` is
    of a model with no parametric margin: the copula's parameters alone."""
    copula = layout.copula
    size = layout.size
    cov = onp.zeros((size, size))
    boundary = layout.on_bound()
    free = onp.flatnonzero(~boundary)
    cov[boundary, :] = onp.nan
    cov[:, boundary] = onp.nan
    if free.size == 0:
        return cov, True
    n_rows = u.size

    def log_density(
        vector: npt.NDArray, uu: npt.NDArray = u, vv: npt.NDArray = v
    ) -> npt.NDArray:
        with onp.errstate(all="ignore"):
            density = onp.asarray(copula.pdf(uu, vv, *vector), dtype=float)
            return onp.log(onp.clip(density, _LOG_FLOOR, None))

    weights = onp.ones(n_rows)
    f = _neg_ll_in(log_density, layout.values, free, weights)
    scale = onp.full(size, onp.nan)
    scale[free] = curvature_scales(f, layout.values[free])
    step = layout.steps(scale)
    hessian = scale_free_hessian(
        f, layout.values[free], fallback_step=step[free]
    )
    inverse = _scaled_inverse(hessian, step[free])

    def scores(uu: npt.NDArray, vv: npt.NDArray) -> npt.NDArray:
        return _row_scores(
            lambda vector: log_density(vector, uu, vv),
            layout.values,
            free,
            step,
        )

    influence = scores(u, v)
    for axis, w in enumerate((u, v)):
        h = _H * onp.minimum(w, 1.0 - w)
        up, down = w + h, w - h
        if axis == 0:
            change = scores(up, v) - scores(down, v)
        else:
            change = scores(u, up) - scores(u, down)
        slope = change / (2.0 * h)[:, None]
        # W_i: the mean of the slopes of the rows ranked at or above i
        order = onp.argsort(w)
        above = onp.cumsum(slope[order][::-1], axis=0)[::-1] / n_rows
        influence[order] += above
    if inverse is None or not onp.all(onp.isfinite(influence)):
        cov[onp.ix_(free, free)] = onp.nan
        return cov, False
    sigma = onp.atleast_2d(onp.cov(influence, rowvar=False, ddof=1))
    # B = hessian / n, so B^-1 Sigma B^-T / n = n H^-1 Sigma H^-T
    cov[onp.ix_(free, free)] = n_rows * inverse @ sigma @ inverse.T
    return cov, True
