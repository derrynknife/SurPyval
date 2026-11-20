"""Predictions out of a network's parameters: survival curves as a
tensor, and any row as an ordinary SurPyval model."""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.torch._optional import torch as _torch
from surpyval.torch.loss import (
    _family,
    _finite_only,
    _mixture_parameters,
    _parameters,
)


def _components(
    torch: Any, dist: Any, params: Any, mixture: Any, what: str
) -> tuple:
    """The parameters as ``(N, K)`` tensors and the log weights
    ``(N, K)`` (``K = 1`` without a mixture), for ``N`` the rows of the
    parameters (1 if every one is shared)."""
    if mixture is not None:
        return _mixture_parameters(torch, dist, params, mixture, None, what)
    theta = _parameters(torch, dist, params, None, what)
    rows = max([len(t) for t in theta if t.ndim == 1], default=1)
    theta = [t.reshape(-1, 1).expand(rows, 1) for t in theta]
    return theta, theta[0].new_zeros(rows, 1)


def _times(torch: Any, times: Any, like: Any, what: str) -> Any:
    """``times`` as a 1-D tensor of ``like``'s type and device."""
    if isinstance(times, torch.Tensor):
        values = times.to(like.device, like.dtype)
    else:
        try:
            values = torch.as_tensor(
                np.asarray(times, dtype=float),
                dtype=like.dtype,
                device=like.device,
            )
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"{what}: times must be numbers (a 1-D array or tensor of "
                f"the times to evaluate at), not {times!r}."
            ) from err
    if values.ndim > 1:
        raise ValueError(
            f"{what}: times has shape {tuple(values.shape)}. It is a 1-D "
            "array of the T times every row is evaluated at, (T,), or "
            "one number; the result is (N, T)."
        )
    values = values.reshape(-1)
    if bool(torch.isnan(values).any()):
        raise ValueError(
            f"{what}: times has NaN entries; give each time as a number "
            "(an infinite time is allowed)."
        )
    return values


def sf(dist: Any, times: Any, params: Any, *, mixture: Any = None) -> Any:
    """
    The survival function of each row at each of ``times``, as a tensor:
    the predictions ``score_predictions`` takes.

    Parameters
    ----------
    dist : distribution
        One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
    times : array-like or Tensor
        The ``T`` times to evaluate at, shape ``(T,)`` (or one number).
        Every row is evaluated at all of them.
    params : dict
        Every parameter of ``dist`` by name, as :func:`constrain` returns
        and :func:`nll` takes: each a scalar or one value per row (with
        ``mixture``, ``(N, K)``, ``(K,)`` or a scalar).
    mixture : Tensor, optional
        The logits of a mixture's weights, ``(N, K)`` or ``(K,)``, as
        :func:`nll` takes them: each row's survival function is then
        ``sum_k w_k S_k(t)``.

    Returns
    -------
    Tensor
        Shape ``(N, T)``, one row per row of the parameters (``N = 1``
        where every parameter is shared), of the parameters' floating
        type and on their device. It is differentiable in the parameters
        (and the logits); for ``score_predictions`` and other NumPy code
        use ``.detach().cpu().numpy()``.

    Notes
    -----
    A time at or below the start of the distribution's support has
    survival 1, and one at or past its end 0, as for SurPyval's models.
    Row ``i`` equals ``to_surpyval(dist, params, row=i).sf(times)``.

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval as sp
    >>> import surpyval.torch as spt
    >>> params = {"alpha": torch.tensor([10.0, 20.0], dtype=torch.float64),
    ...           "beta": torch.tensor(2.0, dtype=torch.float64)}
    >>> spt.sf(sp.Weibull, [0.0, 10.0, 20.0], params).numpy().round(4)
    array([[1.    , 0.3679, 0.0183],
           [1.    , 0.7788, 0.3679]])

    A mixture's, row by row:

    >>> params = spt.constrain(sp.Weibull, torch.zeros(2, 3, 2))
    >>> spt.sf(sp.Weibull, [1.0, 2.0], params,
    ...        mixture=torch.zeros(2, 3)).shape
    torch.Size([2, 2])
    """
    what = "sf"
    torch = _torch(what)
    family = _family(dist, what)
    theta, log_w = _components(torch, dist, params, mixture, what)
    t = _times(torch, times, log_w, what)
    rows, components = log_w.shape
    # Every (row, component, time), row-major.
    shape = (rows, components, len(t))
    x = t.expand(shape).reshape(-1)
    p = [v.unsqueeze(-1).expand(shape).reshape(-1) for v in theta]
    lower, upper = (float(s) for s in dist.support)
    # 1 at or below the support, 0 at or past its end; worked out only
    # inside it.
    log_s = torch.where(x <= lower, 0.0, -float("inf")).to(x.dtype)
    index = torch.nonzero((x > lower) & (x < upper)).ravel()
    if len(index):

        def inside(at: Any) -> Any:
            rows_at = index[at]
            return family.log_sf(torch, x[rows_at], *(v[rows_at] for v in p))

        values = _finite_only(torch, inside(slice(None)), inside)
        log_s = log_s.index_put((index,), values)
    survival = torch.exp(log_s).reshape(shape)
    if mixture is None:
        return survival[:, 0, :]
    return (torch.exp(log_w).unsqueeze(-1) * survival).sum(dim=1)


def _row(row: Any, rows: int, what: str) -> int:
    """``row`` as an index into ``rows`` rows, or a refusal naming
    them."""
    if row is None:
        if rows == 1:
            return 0
        raise ValueError(
            f"{what}: the parameters have {rows} rows; say which one to "
            f"turn into a model with row= (0 to {rows - 1})."
        )
    if (
        isinstance(row, (bool, np.bool_))
        or not isinstance(row, (int, np.integer))
        or not -rows <= row < rows
    ):
        raise ValueError(
            f"{what}: row must be a whole number from 0 to {rows - 1} (or "
            f"from -{rows}, counting from the end): the parameters have "
            f"{rows} row{'s' if rows != 1 else ''}; got {row!r}."
        )
    return int(row) % rows


def to_surpyval(
    dist: Any, params: Any, *, row: Any = None, mixture: Any = None
) -> Any:
    """
    One row's prediction as an ordinary SurPyval model: its ``sf``,
    ``ff``, ``df``, ``hf``, ``qf``, ``mean``, ``random`` and plots.

    Parameters
    ----------
    dist : distribution
        One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
    params : dict
        Every parameter of ``dist`` by name, as :func:`nll` takes them
        (with ``mixture``, the components').
    row : int, optional
        The row to turn into a model, from 0 to ``N - 1`` (negative counts
        from the end). It may be left out only where the parameters have
        one row (each is shared).
    mixture : Tensor, optional
        The logits of a mixture's weights, as :func:`nll` takes them.

    Returns
    -------
    Parametric or MixtureModel
        ``dist.from_params(...)`` of the row's parameters, or, with
        ``mixture``, ``MixtureModel.from_params(dist, ...)`` of its
        components' parameters and weights. The values are taken out of
        the graph as double-precision numbers; the model knows nothing of
        torch.

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval as sp
    >>> import surpyval.torch as spt
    >>> params = {"alpha": torch.tensor([10.0, 20.0], dtype=torch.float64),
    ...           "beta": torch.tensor(2.0, dtype=torch.float64)}
    >>> model = spt.to_surpyval(sp.Weibull, params, row=1)
    >>> model.params, round(float(model.sf(20.0)), 4)
    (array([20.,  2.]), 0.3679)

    A mixture's row is a ``MixtureModel``:

    >>> params = {"alpha": torch.tensor([10.0, 50.0], dtype=torch.float64),
    ...           "beta": torch.tensor([2.0, 4.0], dtype=torch.float64)}
    >>> logits = torch.log(torch.tensor([0.3, 0.7], dtype=torch.float64))
    >>> mm = spt.to_surpyval(sp.Weibull, params, mixture=logits)
    >>> mm.w.round(6), mm.sf([5, 40]).round(4)
    (array([0.3, 0.7]), array([0.9336, 0.4647]))
    """
    what = "to_surpyval"
    torch = _torch(what)
    _family(dist, what)
    theta, log_w = _components(torch, dist, params, mixture, what)
    i = _row(row, log_w.shape[0], what)
    with torch.no_grad():
        values = (
            torch.stack([v[i] for v in theta], dim=-1)
            .to("cpu", torch.float64)
            .numpy()
        )
        log_weights = log_w[i].to("cpu", torch.float64).numpy()
    if mixture is None:
        return dist.from_params(values[0])
    from surpyval import MixtureModel

    return MixtureModel.from_params(dist, values, w_logits=log_weights)
