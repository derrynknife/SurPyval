"""SurPyval's censored and truncated likelihood as a PyMC model term."""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.pymc._logp import FAMILIES, Family, log1mexp
from surpyval.pymc._optional import pymc, pytensor_assert, pytensor_tensor
from surpyval.utils.surpyval_data import SurpyvalData

SUPPORTED = tuple(FAMILIES)

_DATA_ARGS = ("x", "c", "n", "t", "xl", "xr", "tl", "tr")


def _family(dist: Any, what: str) -> Family:
    """The pytensor functions of ``dist``, or a refusal naming the
    families that have them."""
    name = getattr(dist, "name", None)
    if hasattr(dist, "dist") and hasattr(dist, "params"):
        raise TypeError(
            f"{what} takes the distribution, such as surpyval.Weibull, not "
            f"a fitted model ({type(dist).__name__}). To start from a "
            "fitted model's estimates, use prior_from_model(model)."
        )
    if not isinstance(name, str) or name not in FAMILIES:
        raise ValueError(
            f"{what} has no PyMC likelihood for {name or dist!r}. It "
            f"supports {', '.join(SUPPORTED)}: pass one of those (as "
            "surpyval.Weibull, ...)."
        )
    return FAMILIES[name]


def as_surpyval_data(
    what: str, data: Any = None, **arrays: Any
) -> SurpyvalData:
    """``data`` as a :class:`SurpyvalData`, or one built from the arrays
    in SurPyval's usual forms, with the checks every ``fit`` runs. The
    arrays keep their order (``group_and_sort=False``), so a parameter
    given per row lines up with the rows as passed."""
    given = [k for k in _DATA_ARGS if arrays.get(k) is not None]
    if isinstance(data, SurpyvalData):
        if given:
            raise TypeError(
                f"{what}: data is a SurpyvalData, which holds the data: "
                f"leave out {', '.join(repr(g) for g in given)} (or pass "
                "the arrays and no SurpyvalData)."
            )
        from surpyval.utils.weights import BAYES_WHY, refuse_weighted_data

        refuse_weighted_data(
            data,
            what,
            BAYES_WHY,
            "Fit by maximum likelihood with the weights (Weibull.fit(data)), "
            "whose robust covariance gives valid Wald intervals, or pass "
            "the data without its weights (data.x, data.c, data.n, data.t).",
        )
        return data
    if data is not None:
        if hasattr(data, "to_xcnt") or hasattr(data, "x_i"):
            raise TypeError(
                f"{what} takes lifetime data (a SurpyvalData or the x, c, "
                f"n, t arrays), not a {type(data).__name__}."
            )
        if arrays.get("x") is not None:
            raise TypeError(
                f"{what}: give the times once, as data or as x, not both."
            )
        arrays["x"] = data
    if arrays.get("x") is None and arrays.get("xl") is None:
        raise ValueError(
            f"{what} needs data: a SurpyvalData, or the times as x (with "
            "c, n, t, tl, tr as needed) or as xl and xr."
        )
    return SurpyvalData(**arrays, group_and_sort=False)


def _parameters(
    pt: Any, dist: Any, params: dict, rows: int, what: str
) -> list:
    """Each of ``dist``'s parameters as a tensor, in its order: a scalar,
    or one value per data row (checked when the graph runs where the
    length is not known before)."""
    names = list(dist.parameter_names)
    unknown = sorted(set(params) - set(names))
    if unknown:
        raise TypeError(
            f"{what}: {dist.name} has no parameter "
            f"{', '.join(repr(u) for u in unknown)}; its parameters are "
            f"{', '.join(names)}."
        )
    missing = [p for p in names if p not in params]
    if missing:
        raise TypeError(
            f"{what}: give every parameter of {dist.name} "
            f"({', '.join(names)}); missing {', '.join(missing)}. Each is "
            "a PyMC variable, a pytensor expression or a number."
        )
    out = []
    for p in names:
        value = params[p]
        if not hasattr(value, "type"):
            value = np.asarray(value, dtype=float)
        tensor = pt.as_tensor_variable(value)
        if tensor.ndim > 1:
            raise ValueError(
                f"{what}: {p} has {tensor.ndim} dimensions. A parameter "
                f"is a scalar or one value per data row (length {rows})."
            )
        if tensor.ndim == 1:
            length = tensor.type.shape[0]
            if length is not None and length != rows:
                raise ValueError(
                    f"{what}: {p} has {length} values and the data "
                    f"{rows} rows. A parameter is a scalar or one value "
                    "per row, in the order of the data's rows."
                )
            if length is None:
                tensor = pytensor_assert(
                    f"{what}: {p} must have one value per data row "
                    f"({rows}), in the order of the data's rows"
                )(tensor, pt.eq(tensor.shape[0], rows))
        out.append(tensor)
    return out


def _take(params: list, index: np.ndarray) -> list:
    return [p if p.ndim == 0 else p[index] for p in params]


def _log_windows(
    pt: Any,
    family: Family,
    support: tuple,
    lo: np.ndarray,
    hi: np.ndarray,
    index: np.ndarray,
    params: list,
) -> Any:
    """``log[F(hi) - F(lo)]`` for each window, with a bound outside the
    support (or infinite) taken as its limit, the rows ``index``."""
    lo_in = np.isfinite(lo) & (lo > support[0])
    hi_in = np.isfinite(hi) & (hi < support[1])
    parts = []
    only_hi = ~lo_in & hi_in
    if only_hi.any():
        parts.append(
            family.log_ff(
                pt, hi[only_hi], *_take(params, index[only_hi])
            ).ravel()
        )
    only_lo = lo_in & ~hi_in
    if only_lo.any():
        parts.append(
            family.log_sf(
                pt, lo[only_lo], *_take(params, index[only_lo])
            ).ravel()
        )
    both = lo_in & hi_in
    if both.any():
        p = _take(params, index[both])
        a, b = lo[both], hi[both]
        log_fa = family.log_ff(pt, a, *p)
        log_fb = family.log_ff(pt, b, *p)
        log_sa = family.log_sf(pt, a, *p)
        log_sb = family.log_sf(pt, b, *p)
        # F(b) - F(a) from the CDF in the lower half and from the
        # survival function in the upper, where F rounds to 1.
        upper = log_fa > -np.log(2.0)
        by_ff = log_fb + log1mexp(pt, pt.switch(upper, -1.0, log_fa - log_fb))
        by_sf = log_sa + log1mexp(pt, pt.switch(upper, log_sb - log_sa, -1.0))
        parts.append(pt.switch(upper, by_sf, by_ff).ravel())
    if not parts:
        return pt.zeros(len(lo))
    order = np.concatenate(
        [np.flatnonzero(m) for m in (only_hi, only_lo, both) if m.any()]
        + [np.flatnonzero(~lo_in & ~hi_in)]
    )
    values = pt.concatenate(parts + [pt.zeros(int((~lo_in & ~hi_in).sum()))])
    return values[np.argsort(order)]


def log_likelihood(
    dist: Any,
    data: Any = None,
    *,
    x: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    xl: Any = None,
    xr: Any = None,
    tl: Any = None,
    tr: Any = None,
    **params: Any,
) -> Any:
    """
    SurPyval's log-likelihood of the data as a pytensor expression of
    the parameters, the term :func:`censored_likelihood` adds to a model.

    Each row contributes its own term, times its count ``n``: ``log f(x)``
    for a failure (``c = 0``), ``log S(x)`` right censored (``c = 1``),
    ``log F(x)`` left censored (``c = -1``) and ``log[F(xr) - F(xl)]``
    interval censored (``c = 2``); a row observed only inside the window
    ``(tl, tr]`` subtracts ``log[F(tr) - F(tl)]``. A censored row with a
    finite truncation bound on its censored side is an interval ending at
    that bound, as in ``fit``. This is the likelihood ``fit`` maximises,
    equal to the distribution's to rounding.

    Parameters
    ----------
    dist : distribution
        One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
    data : SurpyvalData or array-like, optional
        The data; an array is taken as ``x``.
    x, c, n, t, xl, xr, tl, tr : array-like, optional
        The data as arrays in SurPyval's usual forms (see
        :doc:`/Conventions`), checked as ``fit`` checks them, and kept in
        the order given.
    **params
        Every parameter of ``dist`` by name (``alpha=..., beta=...``): a
        number, a PyMC variable or a pytensor expression. A scalar applies
        to every row; a vector of one value per row (``len(data)``, in the
        data's row order) gives each row its own, for covariates.

    Returns
    -------
    TensorVariable
        A scalar: the sum of the rows' terms.

    Examples
    --------
    >>> import pytest; pt = pytest.importorskip("pytensor.tensor")
    >>> import surpyval as sp
    >>> from surpyval.pymc import log_likelihood
    >>> ll = log_likelihood(sp.Weibull, x=[1, 3, [2, 4]], c=[0, 1, 2],
    ...                     alpha=3.0, beta=1.5)
    >>> round(float(ll.eval()), 6)
    -3.440673
    >>> data = sp.SurpyvalData(x=[1, 3, [2, 4]], c=[0, 1, 2])
    >>> round(float(sp.Weibull._log_likelihood(data, 3.0, 1.5, 0, 0, 1)), 6)
    -3.440673
    """
    what = "log_likelihood"
    pt = pytensor_tensor(what)
    family = _family(dist, what)
    data = as_surpyval_data(
        what, data, x=x, c=c, n=n, t=t, xl=xl, xr=xr, tl=tl, tr=tr
    )
    return _build(pt, dist, family, data, params, what)


def _build(
    pt: Any,
    dist: Any,
    family: Family,
    data: SurpyvalData,
    params: dict,
    what: str,
) -> Any:
    dist._check_inside_support(data, every_row=True)
    rows = len(data.c)
    theta = _parameters(pt, dist, params, rows, what)
    support = tuple(float(s) for s in dist.support)
    total: Any = pt.as_tensor_variable(np.float64(0.0))
    for mask, xs, ns, fn in (
        (data.mask_o, data.x_o, data.n_o, family.log_df),
        (data.mask_r, data.x_r, data.n_r, family.log_sf),
        (data.mask_l, data.x_l, data.n_l, family.log_ff),
    ):
        if mask.any():
            index = np.flatnonzero(mask)
            terms = fn(pt, xs.astype(float), *_take(theta, index))
            total = total + pt.sum(ns.astype(float) * terms)
    if data.mask_i.any():
        index = np.flatnonzero(data.mask_i)
        terms = _log_windows(
            pt,
            family,
            support,
            data.x_il.astype(float),
            data.x_ir.astype(float),
            index,
            theta,
        )
        total = total + pt.sum(data.n_i.astype(float) * terms)
    if data.truncated_mask.any():
        index = np.flatnonzero(data.truncated_mask)
        terms = _log_windows(
            pt,
            family,
            support,
            data.x_tl.astype(float),
            data.x_tr.astype(float),
            index,
            theta,
        )
        total = total - pt.sum(data.n_t.astype(float) * terms)
    return total


def censored_likelihood(
    dist: Any,
    data: Any = None,
    *,
    name: str = "likelihood",
    x: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    xl: Any = None,
    xr: Any = None,
    tl: Any = None,
    tr: Any = None,
    **params: Any,
) -> Any:
    """
    Add SurPyval's censored and truncated likelihood of the data to the
    open PyMC model, as a ``pm.Potential`` called ``name``.

    The term is :func:`log_likelihood`: every row type SurPyval fits
    (failures, right, left and interval censoring, left and right
    truncation, counts) with the distribution's own stable formulas. The
    parameters are anything PyMC can build on: random variables,
    expressions of them (``pm.math.exp(log_alpha[0])``), or numbers.

    Parameters
    ----------
    dist : distribution
        One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
    data : SurpyvalData or array-like, optional
        The data; an array is taken as ``x``.
    name : str, default "likelihood"
        The name of the Potential in the model; give each data set its
        own.
    x, c, n, t, xl, xr, tl, tr : array-like, optional
        The data as arrays in SurPyval's usual forms, checked as ``fit``
        checks them, and kept in the order given.
    **params
        Every parameter of ``dist`` by name: a scalar applies to every
        row, a vector of one value per row (``len(data)``, in the data's
        row order) gives each row its own, as with covariates
        (``alpha=pm.math.exp(Z @ b)``).

    Returns
    -------
    TensorVariable
        The Potential. It counts in sampling and in ``pm.find_MAP``, as
        an observed variable would; like any Potential it is not drawn by
        ``pm.sample_prior_predictive`` or ``sample_posterior_predictive``
        and is not in the ``log_likelihood`` group LOO and WAIC read.

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import surpyval as sp
    >>> from surpyval.pymc import censored_likelihood
    >>> with pm.Model() as m:
    ...     beta = pm.LogNormal("beta", mu=0.5, sigma=0.5)
    ...     alpha = pm.LogNormal("alpha", mu=3.0, sigma=1.0)
    ...     term = censored_likelihood(
    ...         sp.Weibull, x=[12, 18, 25, 30], c=[0, 0, 1, 1],
    ...         alpha=alpha, beta=beta, name="tests")
    >>> [p.name for p in m.potentials]
    ['tests']
    """
    what = "censored_likelihood"
    pm = pymc(what)
    pt = pytensor_tensor(what)
    try:
        pm.modelcontext(None)
    except TypeError as err:
        raise TypeError(
            f"{what} adds a term to a PyMC model: call it inside "
            "`with pm.Model():`."
        ) from err
    family = _family(dist, what)
    data = as_surpyval_data(
        what, data, x=x, c=c, n=n, t=t, xl=xl, xr=xr, tl=tl, tr=tr
    )
    return pm.Potential(name, _build(pt, dist, family, data, params, what))
