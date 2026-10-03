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
import functools
import inspect
import warnings
from typing import Any, Callable, Iterator, TypeVar

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

#: Where set (a list), :func:`warn_aliased` records ``(columns, how)``
#: there instead of warning: a model fitted as several fits of the same
#: covariates (one per cause) warns once, for all of them.
_COLLECT: contextvars.ContextVar = contextvars.ContextVar(
    "surpyval_aliased_collect", default=None
)

_EPS = float(np.finfo(float).eps)

F = TypeVar("F", bound=Callable[..., Any])


@contextlib.contextmanager
def collect_aliased() -> Iterator[list]:
    """Collect the aliasing warnings of the fits inside the block, to be
    given as one with :func:`warn_collected`."""
    found: list = []
    token = _COLLECT.set(found)
    try:
        yield found
    finally:
        _COLLECT.reset(token)


def warn_collected(found: list, where: str) -> None:
    """One warning for the aliased columns ``found`` by
    :func:`collect_aliased` (their union), saying ``where`` (e.g. "in the
    fits of causes 'a' and 'b'")."""
    if not found:
        return
    columns = sorted({j for cols, _ in found for j in cols})
    warn_aliased(columns, "{}, {}".format(found[0][1], where))


@contextlib.contextmanager
def covariate_columns(
    names: "list[str] | None", Z: Any = None, model_spec: Any = None
) -> Iterator[None]:
    """Name the columns of ``Z`` for an aliasing warning raised inside the
    block (``fit_from_df`` wraps its call to ``fit`` in this). Where the
    formula's ``model_spec`` recorded a declared level with no rows (#377)
    -- warned of already, and arriving as a column of zeros -- the
    all-zero columns of ``Z`` are left out of the warning."""
    quiet: tuple = ()
    states = getattr(model_spec, "encoder_state", None) or {}
    if Z is not None and any(
        "empty_levels" in state for _, state in states.values()
    ):
        Z_arr = np.asarray(Z, dtype=float)
        if Z_arr.ndim == 2 and Z_arr.shape[0] > 0:
            quiet = tuple(np.flatnonzero(np.all(Z_arr == 0, axis=0)))
    token = _COLUMNS.set((None if names is None else list(names), quiet))
    try:
        yield
    finally:
        _COLUMNS.reset(token)


def fit_columns() -> "list[str] | None":
    """The names of the columns of ``Z`` for the fit in progress (set by
    :func:`covariate_columns`: ``fit_from_df``, a formula, or a DataFrame
    ``Z``, :func:`dataframe_covariates`), or ``None``; they name the
    coefficients (#614)."""
    names, _ = _COLUMNS.get()
    return None if names is None else [str(name) for name in names]


def dataframe_covariates(fit: F) -> F:
    """Decorate a regression ``fit`` so that a :class:`pandas.DataFrame`
    ``Z`` is fitted as ``fit_from_df`` fits its columns (#614): as its
    values, with its column names naming the coefficients
    (:func:`fit_columns`) and an aliasing warning's columns, and kept as
    the model's ``feature_names``, by which it then reads a DataFrame of
    new covariates. Any other ``Z`` is passed as it is."""
    signature = inspect.signature(fit)

    @functools.wraps(fit)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            # The fit's own refusal of a wrong argument
            return fit(*args, **kwargs)
        Z = bound.arguments.get("Z")
        if Z is None or type(Z).__name__ != "DataFrame":
            return fit(*args, **kwargs)
        from surpyval.utils.covariates import numeric_columns

        names = [str(column) for column in Z.columns]
        try:
            # C order, as an array Z is: the fit is then the array's,
            # to the last digit
            values = np.ascontiguousarray(Z.to_numpy(dtype=float))
        except (ValueError, TypeError):
            # Names the columns that are not numeric
            numeric_columns(Z, list(dict.fromkeys(Z.columns)))
            raise
        bound.arguments["Z"] = values
        with covariate_columns(names):
            model = fit(*bound.args, **bound.kwargs)
        if getattr(model, "feature_names", None) is None:
            try:
                model.feature_names = names
            except AttributeError:
                pass
        return model

    return wrapper  # type: ignore[return-value]


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
        return (
            np.ptp(Z, axis=0) <= tol if Z.size else np.ones(Z.shape[1], bool)
        )
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
    aliased: list[int] = np.flatnonzero(constant).tolist()
    aliased += rest[~good].tolist()
    rest = rest[good]
    if rest.size == 0:
        return np.array(sorted(aliased), dtype=int)
    root = np.sqrt(scale[good])
    G = gram[np.ix_(rest, rest)] / np.outer(root, root)
    if not np.all(np.isfinite(G)):
        return np.array(sorted(aliased), dtype=int)
    G = 0.5 * (G + G.T)
    eig = np.linalg.eigvalsh(G)
    top = float(eig[-1])
    if top <= 0:
        return np.array(sorted(aliased + rest.tolist()), dtype=int)
    tol = top * max(G.shape[0], int(n_rows)) * _EPS
    if eig[0] > tol:
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
    collector = _COLLECT.get()
    if collector is not None:
        collector.append((shown, how))
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
