"""Lifetime data as the tensors the torch loss takes."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from surpyval.pymc.likelihood import as_surpyval_data
from surpyval.torch._optional import torch as _torch


class Targets(NamedTuple):
    """Lifetime data as six 1-D tensors, one value per row, in the rows'
    order: what :func:`nll` takes as its target.

    A named tuple, so it indexes, unpacks and batches as a tuple does: a
    ``DataLoader`` over ``TensorDataset(X, *y)`` yields ``X`` and the six
    fields, and ``Targets(*fields)`` is the batch's target.

    Attributes
    ----------
    xl, xr : Tensor
        The ends of the interval the time is known to lie in: equal for a
        failure, ``xr = inf`` for a right-censored row and ``xl = -inf``
        for a left-censored one (or that side's truncation bound where it
        is finite, as ``fit`` takes such a row).
    c : Tensor
        The censoring codes (0 failed, 1 right, -1 left, 2 interval), as
        integers.
    n : Tensor
        The count of each row.
    tl, tr : Tensor
        The truncation window ``(tl, tr]`` each row was observed in;
        ``-inf`` and ``inf`` where there is none.
    """

    xl: Any
    xr: Any
    c: Any
    n: Any
    tl: Any
    tr: Any


def targets(
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
    dtype: Any = None,
    device: Any = None,
) -> Targets:
    """
    Lifetime data as tensors for :func:`nll`.

    The data are checked as every ``fit`` checks them and kept in the
    order given, so row ``i`` of the result is row ``i`` of your features.

    Parameters
    ----------
    data : SurpyvalData or array-like, optional
        The data; an array is taken as ``x``. A ``SurpyvalData`` gives its
        own rows, in its own order (it groups and sorts what it is given
        unless built with ``group_and_sort=False``).
    x, c, n, t, xl, xr, tl, tr : array-like, optional
        The data as arrays in SurPyval's usual forms (see
        :doc:`/Conventions`).
    dtype : torch.dtype, optional
        The floating type of the tensors; ``torch.float64`` if not given.
        :func:`nll` converts the target to its parameters' type, so this
        only matters for what you store.
    device : torch.device or str, optional
        Where to put the tensors.

    Returns
    -------
    Targets
        ``(xl, xr, c, n, tl, tr)``, each a 1-D tensor of one value per
        row.

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval.torch as spt
    >>> y = spt.targets(x=[1, 3, [2, 4], 5], c=[0, 1, 2, -1],
    ...                 n=[1, 2, 1, 1])
    >>> y.xl.tolist(), y.xr.tolist()
    ([1.0, 3.0, 2.0, -inf], [1.0, inf, 4.0, 5.0])
    >>> y.c.tolist(), y.n.tolist()
    ([0, 1, 2, -1], [1.0, 2.0, 1.0, 1.0])
    """
    what = "targets"
    torch = _torch(what)
    data = as_surpyval_data(
        what, data, x=x, c=c, n=n, t=t, xl=xl, xr=xr, tl=tl, tr=tr
    )
    if dtype is None:
        dtype = torch.float64
    elif not torch.empty(0, dtype=dtype).is_floating_point():
        raise TypeError(
            f"{what}: dtype must be a floating type (torch.float64 or "
            f"torch.float32), not {dtype}."
        )
    code = np.asarray(data.c)
    times = np.asarray(data.x, dtype=float)
    lead = times if times.ndim == 1 else times[:, 0]
    trail = times if times.ndim == 1 else times[:, -1]
    window = np.asarray(data.t, dtype=float)
    # A censored row's open side ends at its truncation bound there (the
    # infinite one where it has none), as SurpyvalData hands such a row
    # to fit: as the interval up to the bound.
    lower = np.where(code == -1, window[:, 0], lead)
    upper = np.where(code == 1, window[:, 1], trail)

    def tensor(values: Any) -> Any:
        return torch.as_tensor(
            np.array(values, dtype=float),
            dtype=dtype,
            device=device,
        )

    return Targets(
        xl=tensor(lower),
        xr=tensor(upper),
        c=torch.as_tensor(np.array(code, dtype=np.int64), device=device),
        n=tensor(data.n),
        tl=tensor(window[:, 0]),
        tr=tensor(window[:, 1]),
    )
