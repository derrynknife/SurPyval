"""Covariate columns a regression cannot estimate: aliased coefficients
(#476).

A constant column (where the model already has an intercept, or within
each stratum of a stratified Cox fit) or a column that is a linear
combination of others leaves the likelihood flat along some combination
of the coefficients, so their separate values are not determined. The
fits used to accept such data: a Cox coefficient ran off to 5.5e14 and
every prediction was nan, and the parametric fits split the effect
between the collinear columns wherever their optimiser stopped, silently.

As R's ``coxph`` and ``lm`` do, such a coefficient is *aliased*: the fit
runs on the other columns, reports ``nan`` for it (its standard error and
p-value too), predicts as though it were 0, and says so with one warning
naming the columns. The columns are taken in order, so it is the later of
two collinear columns that is aliased, as in R.
"""

import contextlib
import contextvars
import warnings
from typing import Any, Iterator

import numpy as np
import numpy.typing as npt

from surpyval.utils import _caller_stacklevel

#: ``(names, quiet)`` for the fit in progress: the names of the columns of
#: ``Z`` (from ``fit_from_df``) for the warning, and the columns it need
#: not name because the caller has warned of them already (a formula's
#: declared level with no rows, #377).
_COLUMNS: contextvars.ContextVar = contextvars.ContextVar(
    "surpyval_covariate_columns", default=(None, ())
)

_EPS = float(np.finfo(float).eps)


@contextlib.contextmanager
def covariate_columns(
    names: "list[str] | None", Z: "npt.ArrayLike | None" = None
) -> Iterator[None]:
    """Name the columns of ``Z`` for an aliasing warning raised inside the
    block (``fit_from_df`` wraps its call to ``fit`` in this). With ``Z``,
    its all-zero columns are left out of the warning: a formula's declared
    level with no rows arrives as one, and has been warned of already."""
    quiet: tuple = ()
    if Z is not None:
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 2 and Z_arr.shape[0] > 0:
            quiet = tuple(np.flatnonzero(np.all(Z_arr == 0, axis=0)))
    token = _COLUMNS.set((None if names is None else list(names), quiet))
    try:
        yield
    finally:
        _COLUMNS.reset(token)


def constant_columns(
    Z: npt.ArrayLike, groups: "npt.ArrayLike | None" = None
) -> npt.NDArray:
    """Mask of the columns of ``Z`` that are constant within every group
    of ``groups`` (the whole of ``Z`` when ``None``), to rounding: their
    range in each group is within ``N * eps`` of their largest magnitude,
    ``N`` the number of rows, the error of forming a mean of them."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    size = np.max(np.abs(Z), axis=0) if Z.size else np.zeros(Z.shape[1])
    tol = max(Z.shape[0], 1) * _EPS * size
    if groups is None:
        return np.ptp(Z, axis=0) <= tol if Z.size else np.ones(Z.shape[1], bool)
    groups = np.asarray(groups)
    out = np.ones(Z.shape[1], dtype=bool)
    for g in np.unique(groups):
        out &= np.ptp(Z[groups == g], axis=0) <= tol
    return out


def aliased_columns(
    gram: npt.ArrayLike,
    n_rows: int,
    constant: npt.NDArray,
    spread: "npt.ArrayLike | None" = None,
) -> npt.NDArray:
    """The indices of the aliased columns.

    ``gram`` is the ``p x p`` information the fit has about the
    coefficients, a sum of ``n_rows`` rows' outer products: the Gram
    matrix of the centred covariates for a parametric model, the
    risk-set covariance summed over the event times (the information at
    ``beta = 0``) for Cox. ``constant`` marks the columns aliased outright
    (:func:`constant_columns`). The others are scaled by their ``spread``
    (default the diagonal of ``gram``) so that the units of a column do
    not matter, and then taken in order, each kept only if it adds to
    the rank of the columns kept before it.

    The rank is ``numpy.linalg.matrix_rank``'s: an eigenvalue at or below
    ``S.max() * max(M, N) * eps`` is rounding, with ``N`` the number of
    rows summed into the matrix (the error of forming it), not a
    threshold chosen for the data.
    """
    gram = np.atleast_2d(np.asarray(gram, dtype=float))
    constant = np.asarray(constant, dtype=bool)
    p = gram.shape[0]
    rest = np.flatnonzero(~constant)
    if rest.size == 0:
        return np.arange(p)
    scale = np.diag(gram) if spread is None else np.asarray(spread, float)
    scale = scale[rest]
    good = np.isfinite(scale) & (scale > 0)
    aliased = list(np.flatnonzero(constant))
    aliased += list(rest[~good])
    rest = rest[good]
    if rest.size == 0:
        return np.array(sorted(aliased), dtype=int)
    root = np.sqrt(scale[good])
    G = gram[np.ix_(rest, rest)] / np.outer(root, root)
    if not np.all(np.isfinite(G)):
        return np.array(sorted(aliased), dtype=int)
    G = 0.5 * (G + G.T)
    top = float(np.max(np.linalg.eigvalsh(G)))
    tol = top * max(G.shape[0], int(n_rows)) * _EPS
    if top <= 0:
        return np.array(sorted(aliased + list(rest)), dtype=int)
    if np.min(np.linalg.eigvalsh(G)) > tol:
        return np.array(sorted(aliased), dtype=int)
    kept: list[int] = []
    for k in range(rest.size):
        trial = kept + [k]
        if np.min(np.linalg.eigvalsh(G[np.ix_(trial, trial)])) > tol:
            kept = trial
        else:
            aliased.append(int(rest[k]))
    return np.array(sorted(aliased), dtype=int)


def describe_columns(columns: "npt.ArrayLike") -> str:
    """``"7"``, or ``"7 ('prio')"`` when the columns are named."""
    names, _ = _COLUMNS.get()
    out = []
    for j in np.asarray(columns, dtype=int).tolist():
        if names is not None and j < len(names):
            out.append("{} ({!r})".format(j, names[j]))
        else:
            out.append(str(j))
    return ", ".join(out)


def warn_aliased(columns: "npt.ArrayLike", how: str) -> None:
    """One warning naming the aliased ``columns`` of ``Z``; ``how`` says
    in what sense they are redundant for this model."""
    _, quiet = _COLUMNS.get()
    shown = [j for j in np.asarray(columns, int).tolist() if j not in quiet]
    if not shown:
        return
    warnings.warn(
        "Covariate column(s) {} of Z cannot be estimated: {}. Their "
        "coefficients are aliased -- reported as nan, with nan standard "
        "errors and p-values, as R reports NA -- and the model is fitted "
        "on the other columns, whose estimates are unaffected; "
        "predictions take an aliased coefficient as 0. Remove the "
        "column(s) from Z to fit without this warning.".format(
            describe_columns(shown), how
        ),
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )


def expand(values: npt.ArrayLike, kept: npt.NDArray, p: int) -> Any:
    """``values`` for the ``kept`` columns placed in a length-``p`` vector
    with ``nan`` at the aliased ones."""
    out = np.full(p, np.nan)
    out[kept] = np.asarray(values, dtype=float)
    return out
