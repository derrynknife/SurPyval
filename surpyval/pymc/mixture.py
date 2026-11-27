"""A mixture of one family's lives as a PyMC model term, with its
components in a fixed order.

A mixture's posterior has a peak for every way of numbering its
components: swap the labels of two components and the likelihood is the
same. A sampler then wanders between the copies (or stays in one by
luck), and every per-component summary is a blend of components. The
usual cure, and the default here, is to number the components by one
parameter: :func:`mixture_likelihood` adds the constraint that the
``order`` parameter (by default the family's first, a Weibull's scale
``alpha``) increases from component 0 to component ``m - 1``, so that
component 0 is always the one with the smallest value of it (for a
Weibull, the shortest characteristic life). Give that parameter PyMC's
ordered transform (``transform=pm.distributions.transforms.ordered``) so
that every draw satisfies it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.pymc._optional import pymc, pytensor_tensor
from surpyval.pymc.likelihood import (
    _family,
    _log_windows,
    as_surpyval_data,
)

#: Not given: the family's first parameter.
_FIRST = object()


def _components(pt: Any, dist: Any, params: dict, m: Any, what: str) -> tuple:
    """Each parameter as a vector of the ``m`` components' values, and
    ``m``."""
    names = list(dist.parameter_names)
    unknown = sorted(set(params) - set(names))
    missing = [p for p in names if p not in params]
    if unknown or missing:
        raise TypeError(
            f"{what}: give every parameter of {dist.name} "
            f"({', '.join(names)}) as one value per component"
            + (f"; {dist.name} has no {', '.join(unknown)}" if unknown else "")
            + (f"; missing {', '.join(missing)}" if missing else "")
            + "."
        )
    out = []
    for p in names:
        value = params[p]
        if not hasattr(value, "type"):
            value = np.asarray(value, dtype=float)
        tensor = pt.as_tensor_variable(value)
        if tensor.ndim != 1:
            raise ValueError(
                f"{what}: {p} must be a vector of one value per component; "
                f"it has {tensor.ndim} dimensions."
            )
        length = tensor.type.shape[0]
        if length is not None:
            if m is None:
                m = int(length)
            elif int(length) != m:
                raise ValueError(
                    f"{what}: {p} has {length} values for {m} components."
                )
        out.append(tensor)
    return out, m


def mixture_log_likelihood(
    dist: Any,
    data: Any = None,
    *,
    weights: Any,
    m: int | None = None,
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
    The log-likelihood of a mixture of ``m`` lives of the family ``dist``
    as a pytensor expression: what ``surpyval.MixtureModel(dist, m)``
    maximises.

    In words: each unit's life comes from component ``k`` with
    probability ``weights[k]``. A failure at ``x`` contributes the log of
    ``sum_k w_k f_k(x)``; a unit right censored at ``x``, the log of
    ``sum_k w_k S_k(x)``; left censored, ``sum_k w_k F_k(x)``; interval
    censored, ``sum_k w_k [F_k(xr) - F_k(xl)]``; each times its count
    ``n``. A unit seen only inside the window ``(tl, tr]`` subtracts the
    log of the mixture's probability of the window, ``sum_k w_k
    [F_k(tr) - F_k(tl)]``. The sums are taken as log-sum-exp of the
    components' log terms, so a unit no single component explains well
    is not lost to underflow.

    Parameters
    ----------
    dist : distribution
        One of :data:`surpyval.pymc.SUPPORTED`, such as ``surpyval.Weibull``.
    data, x, c, n, t, xl, xr, tl, tr
        The data, as for :func:`surpyval.pymc.log_likelihood`.
    weights : vector
        The components' weights, summing to 1 (a ``pm.Dirichlet``).
    m : int, optional
        The number of components, where the vectors' length is not known
        before the graph runs.
    **params
        Every parameter of ``dist`` by name, each a vector of one value
        per component.

    Returns
    -------
    TensorVariable
        A scalar.

    Examples
    --------
    >>> import pytest; pt = pytest.importorskip("pytensor.tensor")
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.pymc import mixture_log_likelihood
    >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    >>> fitted = sp.MixtureModel.fit(x, dist=sp.Weibull, m=2)
    >>> ll = mixture_log_likelihood(
    ...     sp.Weibull, x=x, weights=fitted.w,
    ...     alpha=fitted.params[:, 0], beta=fitted.params[:, 1])
    >>> round(float(ll.eval()), 4), round(fitted.log_likelihood, 4)
    (-48.7105, -48.7105)
    """
    what = "mixture_log_likelihood"
    pt = pytensor_tensor(what)
    return _mixture_term(
        pt, dist, data, weights, m, params, what, x, c, n, t, xl, xr, tl, tr
    )


def _mixture_term(
    pt: Any,
    dist: Any,
    data: Any,
    weights: Any,
    m: Any,
    params: dict,
    what: str,
    *arrays: Any,
) -> Any:
    x, c, n, t, xl, xr, tl, tr = arrays
    family = _family(dist, what)
    data = as_surpyval_data(
        what, data, x=x, c=c, n=n, t=t, xl=xl, xr=xr, tl=tl, tr=tr
    )
    dist._check_inside_support(data, every_row=True)
    theta, m = _components(pt, dist, params, m, what)
    if not hasattr(weights, "type"):
        weights = np.asarray(weights, dtype=float)
    w = pt.as_tensor_variable(weights)
    length = w.type.shape[0] if w.ndim == 1 else None
    if w.ndim != 1:
        raise ValueError(f"{what}: weights must be a vector, one a component.")
    if length is not None:
        m = int(length) if m is None else m
        if int(length) != m:
            raise ValueError(f"{what}: {length} weights for {m} components.")
    if m is None:
        raise ValueError(
            f"{what}: the number of components is not known before the "
            "graph runs: give m=."
        )
    support = tuple(float(s) for s in dist.support)
    rows = len(data.c)
    log_w = pt.log(w)

    def rows_of(k: int) -> Any:
        """Component ``k``'s log term of every row, in row order."""
        p = [v[k] for v in theta]
        out = pt.zeros(rows)
        for mask, xs, fn in (
            (data.mask_o, data.x_o, family.log_df),
            (data.mask_r, data.x_r, family.log_sf),
            (data.mask_l, data.x_l, family.log_ff),
        ):
            if mask.any():
                index = np.flatnonzero(mask)
                out = pt.set_subtensor(
                    out[index], fn(pt, xs.astype(float), *p)
                )
        if data.mask_i.any():
            index = np.flatnonzero(data.mask_i)
            terms = _log_windows(
                pt,
                family,
                support,
                data.x_il.astype(float),
                data.x_ir.astype(float),
                np.zeros(index.size, dtype=int),
                p,
            )
            out = pt.set_subtensor(out[index], terms)
        return out

    stacked = pt.stack([log_w[k] + rows_of(k) for k in range(m)])
    total = pt.sum(
        np.asarray(data.n, dtype=float) * pt.logsumexp(stacked, axis=0)
    )
    if data.truncated_mask.any():
        index = np.flatnonzero(data.truncated_mask)
        windows = []
        for k in range(m):
            p = [v[k] for v in theta]
            windows.append(
                log_w[k]
                + _log_windows(
                    pt,
                    family,
                    support,
                    data.x_tl.astype(float),
                    data.x_tr.astype(float),
                    np.zeros(index.size, dtype=int),
                    p,
                )
            )
        total = total - pt.sum(
            data.n_t.astype(float) * pt.logsumexp(pt.stack(windows), axis=0)
        )
    return total


def mixture_likelihood(
    dist: Any,
    data: Any = None,
    *,
    weights: Any,
    name: str = "likelihood",
    order: Any = _FIRST,
    m: int | None = None,
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
    Add a mixture's censored and truncated likelihood
    (:func:`mixture_log_likelihood`) to the open PyMC model, as a
    ``pm.Potential`` called ``name``, with its components numbered in
    increasing order of one parameter.

    Why the order: the likelihood is the same whichever component is
    called 0, so with ``m`` components the posterior has ``m!`` copies
    of every peak, and a sampler's draws of "component 0" are a blend of
    the components (label switching). Requiring the ``order`` parameter
    to increase from component 0 to component ``m - 1`` keeps one copy:
    component 0 is then, in every draw, the component with the smallest
    value of that parameter. By default it is the family's first
    parameter (a Weibull's scale ``alpha``, so component 0 has the
    shortest characteristic life; a LogNormal's ``mu``). The constraint
    is a second Potential, ``name + "_order"``, which is 0 where the
    parameter increases and ``-inf`` elsewhere; give that parameter
    PyMC's ordered transform (``transform=pm.distributions.transforms
    .ordered``, on its log for a positive parameter) so that every draw
    satisfies it and the sampler never meets the wall. With the order
    fixed, a component's summaries mean "the component with the
    shortest (second shortest, ...) characteristic life"; where two
    components have nearly the same value of the parameter, the order
    can still swap them between draws, and another ``order`` (one in
    which they differ) is the better choice. ``order=None`` leaves the
    labels free (for a sampler of the user's own that handles them).

    Parameters
    ----------
    dist, data, x, c, n, t, xl, xr, tl, tr, weights, m, **params
        As for :func:`mixture_log_likelihood`.
    name : str, default "likelihood"
        The name of the Potential.
    order : str or None, optional
        The parameter whose values increase with the component's number;
        by default the family's first. ``None`` for no order.

    Returns
    -------
    TensorVariable
        The likelihood's Potential.

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import surpyval as sp
    >>> from surpyval.pymc import mixture_likelihood
    >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    >>> with pm.Model() as model:
    ...     w = pm.Dirichlet("w", a=[1.0, 1.0])
    ...     log_alpha = pm.Normal(
    ...         "log_alpha", mu=2.0, sigma=1.0, shape=2,
    ...         transform=pm.distributions.transforms.ordered,
    ...         initval=[1.0, 2.5])
    ...     beta = pm.LogNormal("beta", mu=0.5, sigma=0.5, shape=2)
    ...     term = mixture_likelihood(
    ...         sp.Weibull, x=x, weights=w,
    ...         alpha=pm.math.exp(log_alpha), beta=beta)
    >>> [p.name for p in model.potentials]
    ['likelihood', 'likelihood_order']
    """
    what = "mixture_likelihood"
    pm = pymc(what)
    pt = pytensor_tensor(what)
    try:
        pm.modelcontext(None)
    except TypeError as err:
        raise TypeError(
            f"{what} adds a term to a PyMC model: call it inside "
            "`with pm.Model():`."
        ) from err
    names = (
        list(dist.parameter_names) if hasattr(dist, "parameter_names") else []
    )
    if order is _FIRST:
        order = names[0] if names else None
    if order is not None and order not in names:
        raise ValueError(
            f"{what}: order={order!r} is not a parameter of "
            f"{getattr(dist, 'name', dist)} ({', '.join(names)}); name "
            "one, or pass order=None for no order."
        )
    term = _mixture_term(
        pt, dist, data, weights, m, params, what, x, c, n, t, xl, xr, tl, tr
    )
    out = pm.Potential(name, term)
    if order is not None:
        value = params[order]
        if not hasattr(value, "type"):
            value = np.asarray(value, dtype=float)
        v = pt.as_tensor_variable(value)
        increasing = pt.all(pt.gt(v[1:], v[:-1]))
        pm.Potential(name + "_order", pt.switch(increasing, 0.0, -np.inf))
    return out


__all__ = ["mixture_likelihood", "mixture_log_likelihood"]
