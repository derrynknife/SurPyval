"""Two losses that are not a distribution's likelihood: the Cox partial
likelihood and the discrete-time hazard likelihood, for a network's
outputs."""

from __future__ import annotations

from math import log as _log
from typing import Any

from surpyval.torch._optional import torch as _torch
from surpyval.torch.data import Targets

_LOG2 = _log(2.0)

COX_TIES = ("efron", "breslow")
_REDUCTIONS = ("mean", "sum")


def _reduction(reduction: Any, what: str, per: str) -> None:
    if reduction not in _REDUCTIONS:
        raise ValueError(
            f"{what}: reduction must be 'mean' (the sum divided by "
            f"{per}) or 'sum' (the negative log-likelihood itself), not "
            f"{reduction!r}."
        )


def _vector(torch: Any, value: Any, name: str, what: str) -> Any:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{what}: {name} must be a torch tensor (a network's outputs), "
            f"not a {type(value).__name__}."
        )
    if not value.is_floating_point():
        raise TypeError(
            f"{what}: {name} must be a floating tensor, not {value.dtype}: "
            "use .double() (or .float())."
        )
    return value


def cox_nll(
    risk: Any,
    target: Any,
    ties: str = "efron",
    reduction: str = "mean",
) -> Any:
    r"""
    The Cox negative log partial likelihood, as a loss for a network
    that outputs each row's log relative hazard.

    In words: at each time a failure happens, the model says how likely
    it was that the failure was the one that happened among all the units
    still running (the risk set), each unit's chance being in proportion
    to :math:`e^{\eta_i}`. The loss is minus the log of the product of
    those chances. It uses only the order of the times, so the network
    gives a risk score, not a lifetime, and a constant added to every
    output changes nothing. With ``risk`` the linear predictor
    :math:`\eta = Z\beta`, the value is exactly
    :class:`surpyval.CoxPH`'s negative log partial likelihood at those
    coefficients (DeepSurv's loss is this with a network in place of
    :math:`Z\beta`).

    With :math:`d_j` failures at time :math:`t_j` (counted by ``n``),
    :math:`R_j = \sum_{i \in \text{risk set}} n_i e^{\eta_i}` the risk
    set's weight, and :math:`T_j` the weight of the :math:`d_j` units that
    failed:

    - ``ties="breslow"``: each tied failure is scored against the whole
      risk set,
      :math:`-\sum_j \big[\sum_{i \in D_j} n_i \eta_i - d_j \log R_j\big]`;
    - ``ties="efron"`` (the default, as in ``CoxPH``): the tied failures
      are taken one at a time, each leaving the risk set a share of the
      tied weight,
      :math:`-\sum_j \big[\sum_{i \in D_j} n_i \eta_i - \sum_{l=0}^{d_j-1}
      \log\big(R_j - \tfrac{l}{d_j} T_j\big)\big]`.

    Without ties the two agree. The exact and Kalbfleisch-Prentice
    methods of ``CoxPH`` are not offered: they sum over orderings of the
    ties, which has no cheap form for a loss.

    A unit is in the risk set at :math:`t_j` if it entered before
    :math:`t_j` (its left truncation ``tl``, if it has one) and has not
    left before it (``x >= t_j``), as in ``CoxPH``.

    Parameters
    ----------
    risk : Tensor
        Shape ``(N,)``: each row's log relative hazard (a network's raw
        output; for ``(N, 1)`` use ``.squeeze(-1)``). Rows of a batch
        should be rows of one risk set structure: the risk sets are
        formed from the rows given, so a batch is the sample the loss is
        taken over.
    target : Targets
        From ``targets(x=..., c=..., n=..., tl=...)``. Rows are exact
        (``c = 0``) or right censored (``c = 1``); left and interval
        censoring and right truncation have no partial likelihood and are
        refused. ``n`` is a count of identical rows (a frequency weight).
    ties : {'efron', 'breslow'}, optional
        How tied failure times are handled; see above.
    reduction : {'mean', 'sum'}, optional
        ``'mean'`` (the default) divides by the number of failures (the
        sum of ``n`` over the failed rows), as DeepSurv's loss does;
        ``'sum'`` is the negative log partial likelihood itself, what
        ``CoxPH`` reports.

    Returns
    -------
    Tensor
        A scalar. A batch with no failure has no partial likelihood and
        is refused.

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval.torch as spt
    >>> y = spt.targets(x=[2.0, 3.0, 3.0, 5.0, 8.0], c=[0, 0, 0, 1, 0])
    >>> eta = torch.tensor([0.5, -0.2, 0.1, 0.0, 0.3], dtype=torch.float64)
    >>> loss = spt.cox_nll(eta, y, ties="breslow", reduction="sum")
    >>> round(float(loss), 6)
    4.283744
    """
    what = "cox_nll"
    torch = _torch(what)
    if ties not in COX_TIES:
        extra = ""
        if str(ties).lower() in ("exact", "kalbfleisch-prentice", "kp"):
            extra = (
                " The exact and Kalbfleisch-Prentice methods sum over "
                "orderings of the ties and are not offered as losses"
            )
        raise ValueError(
            f"{what}: ties must be 'efron' (the default) or 'breslow', "
            f"not {ties!r}.{extra}"
        )
    _reduction(reduction, what, "the number of failures")
    eta = _vector(torch, risk, "risk", what)
    if eta.ndim != 1:
        raise ValueError(
            f"{what}: risk has shape {tuple(eta.shape)}; it must be one "
            "value per row, shape (N,): for a network's (N, 1) output use "
            ".squeeze(-1)."
        )
    if not isinstance(target, Targets):
        raise TypeError(
            f"{what}: target must be a Targets, as targets(...) returns, "
            f"not a {type(target).__name__}."
        )
    if target.xl.shape != eta.shape:
        raise ValueError(
            f"{what}: risk has {len(eta)} rows and the target "
            f"{len(target.xl)}; they must be the same rows."
        )
    code = target.c
    if bool(((code != 0) & (code != 1)).any()):
        raise ValueError(
            f"{what}: the Cox partial likelihood takes exact (c = 0) and "
            "right-censored (c = 1) rows only; left and interval "
            "censoring have no risk set. Use nll with a distribution "
            "for them."
        )
    if bool(torch.isfinite(target.tr).any()):
        raise ValueError(
            f"{what}: right truncation has no partial likelihood. Use "
            "nll with a distribution."
        )
    dtype = eta.dtype
    x = target.xl.to(dtype=dtype, device=eta.device)
    count = target.n.to(dtype=dtype, device=eta.device)
    entry = target.tl.to(dtype=dtype, device=eta.device)
    failed = code.to(eta.device) == 0
    if not bool(failed.any()):
        raise ValueError(
            f"{what}: no row failed, so there is no partial likelihood. "
            "A batch needs at least one failure (with every unit still "
            "running at the time of each, a batch is the sample the risk "
            "sets are formed from)."
        )

    times = torch.unique(x[failed])  # ascending
    d = torch.zeros_like(times).index_add(
        0, torch.searchsorted(times, x[failed]), count[failed]
    )
    log_w = torch.log(count) + eta

    # The weight of the rows at risk at each failure time t (entry < t <= x).
    # The rows with x >= t are a suffix sum over sorted exit times. With
    # delayed entry, the rows not yet entered (entry >= t, all of which have
    # x >= t) are a suffix sum over sorted entry times, and the risk set is
    # the first less the second: O((N + T) log N). The subtraction is exact
    # to within a bit wherever the not-yet-entered weight is at most half
    # the total; at the failure times where it is more (a late entrant with a
    # much larger risk score), it would cancel, so there the risk set is
    # summed directly instead, in chunks of times.
    def suffix_logsum(keys: Any) -> Any:
        order = torch.argsort(keys)
        tail = torch.flip(
            torch.logcumsumexp(torch.flip(log_w[order], [0]), 0), [0]
        )
        tail = torch.cat([tail, tail.new_full((1,), -float("inf"))])
        return tail[torch.searchsorted(keys[order], times, right=False)]

    log_risk = suffix_logsum(x)
    if bool((entry >= times[0]).any()):
        gap = suffix_logsum(entry) - log_risk  # <= 0, -inf if none late
        close = gap > -_LOG2
        safe_gap = torch.where(close, -1.0, torch.clamp(gap, max=0.0))
        log_risk = log_risk + torch.log1p(-torch.exp(safe_gap))
        hard = torch.nonzero(close).flatten()
        if hard.numel():
            direct = []
            for start in range(0, int(hard.numel()), 512):
                t_chunk = times[hard[start : start + 512]]
                in_set = (entry[:, None] < t_chunk[None, :]) & (
                    x[:, None] >= t_chunk[None, :]
                )
                masked = torch.where(in_set, log_w[:, None], -float("inf"))
                direct.append(torch.logsumexp(masked, dim=0))
            log_risk = log_risk.index_put((hard,), torch.cat(direct))

    at = torch.searchsorted(times, x[failed])
    linear = (count[failed] * eta[failed]).sum()
    if ties == "breslow":
        log_terms = (d * log_risk).sum()
    else:
        whole = torch.round(d)
        if not bool((torch.abs(d - whole) < 1e-9).all()):
            raise ValueError(
                f"{what}: ties='efron' needs whole-number counts n at a "
                "failure time (it steps through the tied failures one at "
                "a time); use ties='breslow' for fractional counts."
            )
        steps = whole.to(torch.int64)
        which = torch.repeat_interleave(
            torch.arange(len(times), device=eta.device), steps
        )
        first = torch.cumsum(steps, 0) - steps
        position = torch.arange(int(steps.sum()), device=eta.device)
        l_of = (position - first[which]).to(dtype)
        log_w_failed = log_w[failed]
        shift = torch.full_like(times, -float("inf")).scatter_reduce(
            0, at, log_w_failed.detach(), reduce="amax", include_self=True
        )
        log_tied = (
            torch.log(
                torch.zeros_like(times).index_add(
                    0, at, torch.exp(log_w_failed - shift[at])
                )
            )
            + shift
        )
        share = torch.exp(log_tied - log_risk)[which] * l_of / whole[which]
        log_terms = (log_risk[which] + torch.log1p(-share)).sum()
    total = log_terms - linear
    if reduction == "mean":
        total = total / count[failed].sum()
    return total


def discrete_time_nll(
    logits: Any,
    index: Any,
    c: Any = None,
    n: Any = None,
    reduction: str = "mean",
) -> Any:
    r"""
    The discrete-time hazard negative log-likelihood, as a loss for a
    network that outputs one logit for each of ``K`` time intervals
    (the logistic hazard of Kvamme and Borgan; Tutz and Schmid).

    In words: a unit faces a chance of failing in the first interval, and
    if it survives it, a chance in the second, and so on. The network
    gives the log-odds of failing in each interval given the unit has
    reached it, :math:`h_k = 1 / (1 + e^{-\text{logit}_k})`. A unit that
    failed in interval :math:`k` contributes
    :math:`h_k \prod_{j<k}(1 - h_j)`; one still running after interval
    :math:`k` (right censored there) contributes :math:`\prod_{j \le k}
    (1 - h_j)`. The loss is minus the log of the product of the units'
    contributions. It is the binary cross-entropy of the hazards over the
    intervals each unit was at risk in, which is how it is usually
    trained.

    It equals the negative log-likelihood of a discrete lifetime model
    with these hazards (``surpyval.DiscreteWeibull`` and
    ``surpyval.Geometric`` for the shapes they allow), with the
    intervals numbered from 0: interval ``k`` is the discrete time
    ``k + 1``.

    Parameters
    ----------
    logits : Tensor
        Shape ``(N, K)``: each row's log-odds of failing in each of the
        ``K`` intervals given it reached it (a network's raw outputs), or
        ``(K,)`` for logits shared by every row.
    index : Tensor or array-like of int
        Shape ``(N,)``, each value from 0 to ``K - 1``: the interval in
        which the unit failed, or the last one it was seen to survive
        (when ``c`` says it is censored).
    c : array-like, optional
        The censoring flags in SurPyval's coding: 0 failed (the default
        for every row), 1 right censored at the end of its interval.
        Left and interval censoring are refused.
    n : array-like, optional
        A count of identical rows (a frequency weight); 1 by default.
    reduction : {'mean', 'sum'}, optional
        ``'mean'`` (the default) divides by the total count, the loss per
        unit; ``'sum'`` is the negative log-likelihood itself.

    Returns
    -------
    Tensor
        A scalar.

    Examples
    --------
    >>> import pytest; torch = pytest.importorskip("torch")
    >>> import surpyval.torch as spt
    >>> logits = torch.zeros(3, 4, dtype=torch.float64)  # h = 1/2 each
    >>> loss = spt.discrete_time_nll(logits, [0, 2, 3], c=[0, 0, 1],
    ...                              reduction="sum")
    >>> round(float(loss) / 0.6931471805599453, 6)  # in bits: 1 + 3 + 4
    8.0
    """
    what = "discrete_time_nll"
    torch = _torch(what)
    _reduction(reduction, what, "the total count")
    z = _vector(torch, logits, "logits", what)
    if z.ndim not in (1, 2):
        raise ValueError(
            f"{what}: logits has shape {tuple(z.shape)}; it must be "
            "(N, K), one logit for each of the K intervals for each row, "
            "or (K,) shared by every row."
        )
    k_idx = torch.as_tensor(index, device=z.device)
    if k_idx.is_floating_point() or k_idx.dtype == torch.bool:
        raise TypeError(
            f"{what}: index must be whole numbers (the interval each unit "
            f"is in), not {k_idx.dtype}."
        )
    k_idx = k_idx.to(torch.int64)
    if k_idx.ndim != 1:
        raise ValueError(
            f"{what}: index has shape {tuple(k_idx.shape)}; it must be "
            "one interval number for each row, shape (N,)."
        )
    rows = k_idx.shape[0]
    z = z.expand(rows, z.shape[-1]) if z.ndim == 1 else z
    if z.shape[0] != rows:
        raise ValueError(
            f"{what}: logits has {z.shape[0]} rows and index {rows}; they "
            "must be the same rows."
        )
    intervals = z.shape[1]
    if bool(((k_idx < 0) | (k_idx >= intervals)).any()):
        raise ValueError(
            f"{what}: index must be between 0 and {intervals - 1} (the "
            f"{intervals} intervals of logits), counting from 0; got "
            f"{int(k_idx.min())} to {int(k_idx.max())}."
        )
    if c is None:
        flag = torch.zeros(rows, dtype=torch.int64, device=z.device)
    else:
        flag = torch.as_tensor(c, device=z.device).to(torch.int64)
    if flag.shape != (rows,) or bool(((flag != 0) & (flag != 1)).any()):
        raise ValueError(
            f"{what}: c must be one flag for each of the {rows} rows, 0 "
            "(failed in its interval) or 1 (right censored at the end of "
            "it); left and interval censoring are not offered."
        )
    if n is None:
        count = torch.ones(rows, dtype=z.dtype, device=z.device)
    else:
        count = torch.as_tensor(n, dtype=z.dtype, device=z.device)
        if count.shape != (rows,) or bool((count < 0).any()):
            raise ValueError(
                f"{what}: n must be a non-negative count for each of the "
                f"{rows} rows."
            )
    # log(1 - h_j) = logsigmoid(-z_j); log h_k = logsigmoid(z_k)
    log_survive = torch.nn.functional.logsigmoid(-z)
    log_fail = torch.nn.functional.logsigmoid(z)
    positions = torch.arange(intervals, device=z.device)
    before = positions[None, :] < k_idx[:, None]
    survived = (log_survive * before).sum(1)
    last = log_fail.gather(1, k_idx[:, None]).squeeze(1)
    last_survived = log_survive.gather(1, k_idx[:, None]).squeeze(1)
    log_lik = survived + torch.where(flag == 0, last, last_survived)
    total = -(count * log_lik).sum()
    if reduction == "mean":
        total = total / count.sum()
    return total
