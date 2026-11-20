"""SurPyval's censored and truncated likelihood as a torch loss."""

from __future__ import annotations

from functools import cache
from typing import Any

from surpyval.torch._logp import FAMILIES, Family, log1mexp, softplus
from surpyval.torch._optional import torch as _torch
from surpyval.torch.data import Targets
from surpyval.utils.aliases import canonical_name

SUPPORTED = tuple(FAMILIES)

_REDUCTIONS = ("mean", "sum", "none")
_LOG2 = 0.6931471805599453


def _family(dist: Any, what: str) -> Family:
    """The torch functions of ``dist``, or a refusal naming the families
    that have them."""
    name = getattr(dist, "name", None)
    # By the canonical name: an alias (Gauss, Galton) is its distribution
    family = canonical_name(dist) if isinstance(name, str) else name
    if hasattr(dist, "dist") and hasattr(dist, "params"):
        raise TypeError(
            f"{what} takes the distribution, such as surpyval.Weibull, not "
            f"a fitted model ({type(dist).__name__}): pass model.dist."
        )
    if not isinstance(family, str) or family not in FAMILIES:
        shown = name if isinstance(name, str) else repr(dist)
        raise ValueError(
            f"{what} has no torch likelihood for {shown}. It "
            f"supports {', '.join(SUPPORTED)}: pass one of those (as "
            "surpyval.Weibull, ...)."
        )
    return FAMILIES[family]


def _reduction(reduction: Any, what: str) -> None:
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"{what}: reduction must be 'mean' (the loss per unit: the "
            "sum divided by the total count), 'sum' or 'none' (one value "
            f"per row), not {reduction!r}."
        )


def _step(torch: Any, like: Any, bound: float, towards: float) -> Any:
    """The number of ``like``'s type next to ``bound`` on the side of
    ``towards`` (1 above, -1 below)."""
    return torch.nextafter(
        like.new_tensor(bound), like.new_tensor(towards * float("inf"))
    )


def constrain(dist: Any, raw: Any) -> dict:
    """
    Unconstrained values, such as a network's outputs, as the parameters
    of ``dist``, each inside its range.

    Column ``j`` of ``raw`` becomes the ``j``-th parameter of ``dist``
    (``dist.parameter_names``), through the map its range
    (``dist.bounds``) calls for: ``softplus`` for a positive parameter,
    the identity for an unbounded one, and a scaled sigmoid for one
    between two finite bounds. Every map is smooth and increasing, so a
    gradient reaches the network through it.

    Parameters
    ----------
    dist : distribution
        One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
    raw : Tensor
        Shape ``(N, k)`` for ``N`` rows, or ``(k,)`` for one set of
        parameters shared by every row, where ``k`` is the number of
        parameters of ``dist``.

    Returns
    -------
    dict
        Each parameter's tensor by name, of shape ``(N,)`` (or a scalar
        tensor from a ``(k,)`` input): what :func:`nll` takes.

    Notes
    -----
    ``softplus`` of a very negative value rounds to 0, which is not
    inside a positive range: a value that rounds onto a bound is kept
    one step of its type inside it. For a network that should start at
    a sensible model, set its last layer's bias to the inverse map of a
    fit's estimates
    (``log(expm1(value))`` for a positive parameter).

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval as sp
    >>> import surpyval.torch as spt
    >>> raw = torch.tensor([[0.0, 1.0], [-2.0, 3.0]], dtype=torch.float64)
    >>> params = spt.constrain(sp.Weibull, raw)
    >>> list(params)
    ['alpha', 'beta']
    >>> [round(v, 6) for v in params["alpha"].tolist()]
    [0.693147, 0.126928]
    >>> list(spt.constrain(sp.LogNormal, raw)["mu"].tolist())
    [0.0, -2.0]
    """
    what = "constrain"
    torch = _torch(what)
    _family(dist, what)
    names = list(dist.parameter_names)
    if not isinstance(raw, torch.Tensor):
        raise TypeError(
            f"{what}: raw must be a torch tensor (a network's outputs), "
            f"not a {type(raw).__name__}."
        )
    if not raw.is_floating_point():
        raise TypeError(
            f"{what}: raw must be a floating tensor, not {raw.dtype}: "
            "use raw.double() (or .float())."
        )
    if raw.ndim not in (1, 2) or raw.shape[-1] != len(names):
        raise ValueError(
            f"{what}: raw has shape {tuple(raw.shape)}. {dist.name} has "
            f"{len(names)} parameter{'s' if len(names) > 1 else ''} "
            f"({', '.join(names)}, in that order), so raw must have shape "
            f"(N, {len(names)}), one column for each, or ({len(names)},) "
            "for values shared by every row."
        )
    out = {}
    for j, (name, (lo, hi)) in enumerate(zip(names, dist.bounds)):
        value = raw[..., j]
        # (rounding can land a value on a bound: it is kept one step of
        # its type inside)
        first = None if lo is None else _step(torch, value, lo, 1.0)
        last = None if hi is None else _step(torch, value, hi, -1.0)
        if lo is not None and hi is not None:
            value = lo + (hi - lo) * torch.sigmoid(value)
        elif lo is not None:
            value = lo + softplus(torch, value)
        elif hi is not None:
            value = hi - softplus(torch, -value)
        if first is not None or last is not None:
            value = value.clamp(first, last)
        out[name] = value
    return out


def _parameters(
    torch: Any, dist: Any, params: Any, rows: int, what: str
) -> list:
    """Each of ``dist``'s parameters as a tensor of one type on one
    device, in its order: a scalar, or one value per row."""
    names = list(dist.parameter_names)
    if not isinstance(params, dict):
        raise TypeError(
            f"{what}: params must be a dict of {dist.name}'s parameters "
            f"by name ({', '.join(names)}), as constrain returns, not a "
            f"{type(params).__name__}."
        )
    unknown = sorted(set(params) - set(names), key=str)
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
            "a tensor or a number."
        )
    given = [params[p] for p in names]
    tensors = [v for v in given if isinstance(v, torch.Tensor)]
    floats = [v.dtype for v in tensors if v.is_floating_point()]
    dtype = torch.float64
    if floats:
        dtype = floats[0]
        for other in floats[1:]:
            dtype = torch.promote_types(dtype, other)
    device = tensors[0].device if tensors else None
    out = []
    for name, value in zip(names, given):
        tensor = torch.as_tensor(value, dtype=dtype, device=device)
        if tensor.ndim > 1 or (tensor.ndim == 1 and len(tensor) != rows):
            raise ValueError(
                f"{what}: {name} has shape {tuple(tensor.shape)} and the "
                f"target {rows} row{'s' if rows != 1 else ''}. A "
                f"parameter is a scalar or one value per row (shape "
                f"({rows},), in the order of the target's rows): for an "
                "(N, 1) tensor use .squeeze(-1), and for a network's "
                "(N, k) outputs use constrain."
            )
        out.append(tensor)
    return out


def _target(torch: Any, target: Any, what: str) -> Targets:
    """``target`` as a :class:`Targets` of 1-D tensors of one length."""
    if not isinstance(target, Targets):
        fields = ", ".join(Targets._fields)
        try:
            target = Targets(*target)
        except TypeError as err:
            raise TypeError(
                f"{what}: target must be a Targets, as targets(...) "
                f"returns, or its six tensors in order ({fields}), not "
                f"a {type(target).__name__}."
            ) from err
    shapes = {
        name: tuple(value.shape) if isinstance(value, torch.Tensor) else None
        for name, value in zip(Targets._fields, target)
    }
    if any(
        s is None or len(s) != 1 or s != shapes["xl"] for s in shapes.values()
    ):
        raise ValueError(
            f"{what}: every field of the target must be a 1-D tensor of "
            f"one value per row, all of one length; got {shapes}. Build "
            "the target with targets(...), and index every field with "
            "the same rows for a batch."
        )
    return target


def _take(params: list, index: Any) -> list:
    return [p if p.ndim == 0 else p[index] for p in params]


def _check_inside_support(
    torch: Any, dist: Any, support: tuple, y: Targets, what: str
) -> None:
    """Refuse rows no parameters give any probability, as ``fit``
    refuses them (``_check_inside_support(data, every_row=True)``)."""
    lower, upper = support
    failed = y.c == 0
    # Where the time itself was recorded: xr for a left-censored row.
    lead = torch.where(y.c == -1, y.xr, y.xl)
    bad = (
        (failed & ((y.xl <= lower) | (y.xl >= upper)))
        | (~failed & ((y.xr <= lower) | (y.xl >= upper)))
        | (lead < lower)
    )
    if bool(bad.any()):
        rows = torch.nonzero(bad).ravel()[:5].tolist()
        raise ValueError(
            f"{what}: some rows of the target are outside the support of "
            f"the {dist.name} distribution (first rows: {rows}): a "
            f"failure must lie strictly between {lower} and {upper}, a "
            "censored row must leave the event some probability, and no "
            f"time may be below {lower}. Remove or correct those rows, or "
            "use a distribution whose support holds them."
        )


def _log_windows(
    torch: Any,
    family: Family,
    support: tuple,
    lo: Any,
    hi: Any,
    params: list,
) -> Any:
    """``log[F(hi) - F(lo)]`` for each row's window, with a bound outside
    the support (or infinite) taken as its limit: 0 where neither bound
    is inside. Each form is worked out on its own rows only, so no row
    meets a formula it does not use. ``params`` are the rows' own."""
    lo_in = torch.isfinite(lo) & (lo > support[0])
    hi_in = torch.isfinite(hi) & (hi < support[1])
    with torch.no_grad():
        # A bound so far in its tail that the probability beyond it
        # rounds to 0 is at its limit too: its log, -inf, would put a
        # NaN into the gradient of a window that does not depend on it.
        index = torch.nonzero(lo_in).ravel()
        if len(index):
            log_fa = family.log_ff(torch, lo[index], *_take(params, index))
            lo_in = lo_in.index_put((index,), torch.isfinite(log_fa))
        index = torch.nonzero(hi_in).ravel()
        if len(index):
            log_sb = family.log_sf(torch, hi[index], *_take(params, index))
            hi_in = hi_in.index_put((index,), torch.isfinite(log_sb))
    out = torch.zeros_like(lo)
    index = torch.nonzero(~lo_in & hi_in).ravel()
    if len(index):
        out = out.index_add(
            0, index, family.log_ff(torch, hi[index], *_take(params, index))
        )
    index = torch.nonzero(lo_in & ~hi_in).ravel()
    if len(index):
        out = out.index_add(
            0, index, family.log_sf(torch, lo[index], *_take(params, index))
        )
    index = torch.nonzero(lo_in & hi_in).ravel()
    if len(index):
        p = _take(params, index)
        a, b = lo[index], hi[index]
        log_fa = family.log_ff(torch, a, *p)
        log_fb = family.log_ff(torch, b, *p)
        log_sa = family.log_sf(torch, a, *p)
        log_sb = family.log_sf(torch, b, *p)
        # F(b) - F(a) from the CDF in the lower half and from the
        # survival function in the upper, where F rounds to 1.
        upper = log_fa > -_LOG2
        by_ff = log_fb + log1mexp(
            torch, torch.where(upper, -1.0, log_fa - log_fb)
        )
        by_sf = log_sa + log1mexp(
            torch, torch.where(upper, log_sb - log_sa, -1.0)
        )
        out = out.index_add(0, index, torch.where(upper, by_sf, by_ff))
    return out


def nll(
    dist: Any,
    target: Any,
    params: Any,
    *,
    reduction: str = "mean",
) -> Any:
    """
    The negative log-likelihood of censored and truncated data as a torch
    loss: the likelihood ``fit`` maximises, differentiable in the
    parameters.

    Each row contributes its own term, times its count ``n``: ``log f(x)``
    for a failure (``c = 0``), ``log S(x)`` right censored (``c = 1``),
    ``log F(x)`` left censored (``c = -1``) and ``log[F(xr) - F(xl)]``
    interval censored (``c = 2``); a row observed only inside the window
    ``(tl, tr]`` subtracts ``log[F(tr) - F(tl)]``. A censored row with a
    finite truncation bound on its censored side is an interval ending at
    that bound, as in ``fit``. In double precision the sum equals the
    distribution's own log-likelihood to 1e-10.

    Parameters
    ----------
    dist : distribution
        One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
    target : Targets
        The data, from :func:`targets` (or a batch of its rows: the six
        tensors indexed alike).
    params : dict
        Every parameter of ``dist`` by name, as :func:`constrain` returns:
        each a scalar (a tensor or a number), shared by every row, or a
        tensor of one value per row, in the target's row order. Each must
        be inside its range.
    reduction : {"mean", "sum", "none"}, default "mean"
        ``"sum"`` is the negative log-likelihood; ``"mean"`` divides it by
        the total count ``sum(n)``, the loss per unit, which does not
        change with the batch size; ``"none"`` gives each row's term
        (times its count), of shape ``(N,)``.

    Returns
    -------
    Tensor
        The loss, of the parameters' floating type and on their device
        (the target is converted to both). With no tensor among the
        parameters it is ``torch.float64``.

    Notes
    -----
    The value is finite, with a finite gradient, wherever the likelihood
    is: each row is given only the formula of its own kind, in a form
    that is stable in both tails. Double precision is what the 1e-10
    agreement with ``fit`` holds for; single precision works and is not
    held to it (a window far in a tail can round to probability 0 there,
    an infinite loss). The gradient is of the first order: the Gamma
    distribution's derivative in its shape cannot be differentiated
    again.

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval as sp
    >>> import surpyval.torch as spt
    >>> y = spt.targets(x=[1, 3, [2, 4]], c=[0, 1, 2])
    >>> params = {"alpha": torch.tensor(3.0, dtype=torch.float64),
    ...           "beta": torch.tensor(1.5, dtype=torch.float64)}
    >>> round(float(spt.nll(sp.Weibull, y, params, reduction="sum")), 6)
    3.440673
    >>> data = sp.SurpyvalData(x=[1, 3, [2, 4]], c=[0, 1, 2])
    >>> ll = sp.Weibull._log_likelihood(data, 3.0, 1.5, 0, 0, 1)
    >>> round(float(-ll), 6)
    3.440673

    A network's raw outputs become parameters through :func:`constrain`:

    >>> raw = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
    >>> loss = spt.nll(sp.Weibull, y, spt.constrain(sp.Weibull, raw))
    >>> loss.backward()
    >>> raw.grad.shape
    torch.Size([3, 2])
    """
    return _nll(dist, target, params, reduction, "nll")


def _nll(
    dist: Any, target: Any, params: Any, reduction: Any, what: str
) -> Any:
    torch = _torch(what)
    family = _family(dist, what)
    _reduction(reduction, what)
    target = _target(torch, target, what)
    theta = _parameters(torch, dist, params, len(target.xl), what)
    dtype, device = theta[0].dtype, theta[0].device
    y = Targets(
        *(
            v.to(device=device) if name == "c" else v.to(device, dtype)
            for name, v in zip(Targets._fields, target)
        )
    )
    support = tuple(float(s) for s in dist.support)
    _check_inside_support(torch, dist, support, y, what)
    failed = y.c == 0
    terms = torch.zeros_like(y.xl)
    index = torch.nonzero(failed).ravel()
    if len(index):
        terms = terms.index_add(
            0, index, family.log_df(torch, y.xl[index], *_take(theta, index))
        )
    index = torch.nonzero(~failed).ravel()
    if len(index):
        terms = terms.index_add(
            0,
            index,
            _log_windows(
                torch,
                family,
                support,
                y.xl[index],
                y.xr[index],
                _take(theta, index),
            ),
        )
    index = torch.nonzero(torch.isfinite(y.tl) | torch.isfinite(y.tr)).ravel()
    if len(index):
        terms = terms.index_add(
            0,
            index,
            -_log_windows(
                torch,
                family,
                support,
                y.tl[index],
                y.tr[index],
                _take(theta, index),
            ),
        )
    loss = -(y.n * terms)
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / y.n.sum()
    return loss


@cache
def nll_loss_class() -> Any:
    """The ``NLLLoss`` class, built when first asked for: it subclasses
    ``torch.nn.Module``, and ``import surpyval.torch`` must not import
    torch."""
    torch = _torch("NLLLoss")

    class NLLLoss(torch.nn.Module):  # type: ignore[name-defined]
        """
        :func:`nll` as a ``torch.nn.Module``: ``NLLLoss(dist)(target,
        params)`` is ``nll(dist, target, params)``.

        Parameters
        ----------
        dist : distribution
            One of :data:`SUPPORTED`, such as ``surpyval.Weibull``.
        reduction : {"mean", "sum", "none"}, default "mean"
            As for :func:`nll`.

        Examples
        --------
        >>> import pytest; torch = pytest.importorskip("torch")
        >>> import surpyval as sp
        >>> import surpyval.torch as spt
        >>> loss_fn = spt.NLLLoss(sp.Weibull, reduction="sum")
        >>> loss_fn
        NLLLoss(Weibull, reduction='sum')
        >>> y = spt.targets(x=[1, 3, [2, 4]], c=[0, 1, 2])
        >>> round(float(loss_fn(y, {"alpha": 3.0, "beta": 1.5})), 6)
        3.440673
        """

        def __init__(self, dist: Any, reduction: str = "mean") -> None:
            super().__init__()
            _family(dist, "NLLLoss")
            _reduction(reduction, "NLLLoss")
            self.dist = dist
            self.reduction = reduction

        def forward(self, target: Any, params: Any) -> Any:
            return _nll(self.dist, target, params, self.reduction, "NLLLoss")

        def extra_repr(self) -> str:
            return f"{self.dist.name}, reduction={self.reduction!r}"

    # Reachable (and pickled) as surpyval.torch.NLLLoss
    NLLLoss.__module__ = "surpyval.torch"
    NLLLoss.__qualname__ = "NLLLoss"
    return NLLLoss
