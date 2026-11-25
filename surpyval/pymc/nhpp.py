"""The power-law (Crow-AMSAA) NHPP likelihood in PyMC, for one system,
a fleet, or a fleet whose systems share a population of growth rates.

The power-law process has the expected number of events by ``t``
``(t / alpha) ** beta``: ``beta`` below 1 is reliability growth (the
events thin out), above 1 deterioration. :func:`nhpp_log_likelihood` is
SurPyval's exact likelihood of event histories under it (the same terms
as ``CrowAMSAA.fit``: exact event times, the end of each system's
observation, counts in windows, truncation) as a pytensor expression,
with ``alpha`` and ``beta`` one value for every system or one a group of
systems; :func:`nhpp_likelihood` adds it to the open model.

For a fleet whose systems are alike but not the same, give each group
(each system, by default) its own ``beta`` drawn from a population:
:func:`~surpyval.pymc.pooled_likelihood` with ``dist=CrowAMSAA`` builds
that, and :func:`~surpyval.pymc.group_models` turns the draws into a
:class:`~surpyval.recurrent.bayes.BayesianRecurrence` per group and one
for a new system.

PyMC is imported only when a helper is called.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from surpyval.pymc._optional import pymc, pytensor_tensor

#: The pieces of the likelihood, by kind: (the times' keys, the counts'
#: key), as ``EventHistoryData.split_for_nhpp_likelihood`` names them.
_PIECES = {
    "o": ("x_o", "x_o_prev"),
    "right": ("x_right_prev", "x_right"),
    "left": ("x_left_prev", "x_left"),
    "i": ("x_i_l", "x_i_r"),
    "sel": ("sel_from", "sel_to"),
    "gap": ("gap_from", "gap_to"),
}


def is_power_law(dist: Any) -> bool:
    """Whether ``dist`` is the Crow-AMSAA (power-law) process."""
    return type(dist).__name__ == "CrowAMSAA"


def event_data(
    what: str,
    x: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    tl: Any = None,
    tr: Any = None,
) -> tuple[Any, Any]:
    """The event histories as ``CrowAMSAA.fit`` reads them, and that fit
    (``None`` where the data have no events)."""
    from surpyval.recurrent import CrowAMSAA
    from surpyval.recurrent.bayes import _zero_event_data
    from surpyval.utils.event_history_data import EventHistoryData
    from surpyval.utils.no_maximum import quiet_maximum_warnings
    from surpyval.utils.warnings import ignoring_warnings

    if isinstance(x, EventHistoryData):
        given = [k for k, v in (("i", i), ("c", c), ("n", n)) if v is not None]
        if given:
            raise TypeError(
                f"{what}: x is an EventHistoryData, which holds the data: "
                f"leave out {', '.join(given)}."
            )
    try:
        with ignoring_warnings(), quiet_maximum_warnings():
            with np.errstate(all="ignore"):
                mle = CrowAMSAA.fit(x, i=i, c=c, n=n, t=t, tl=tl, tr=tr)
        return mle.data, mle
    except ValueError as err:
        data = _zero_event_data(x, i, c, n, t, tl, tr, None)
        if data is None:
            raise ValueError(f"{what}: {err}") from None
        return data, None


def _group_pieces(data: Any, item_group: np.ndarray, groups: int) -> dict:
    """Every group's pieces of the likelihood, joined, with each time's
    group: ``{key: array}`` and ``{kind: group index}``."""
    first, inverse = data.item_rows()
    joined: dict = {}
    index: dict = {k: [] for k in _PIECES}
    for g in range(groups):
        rows = np.flatnonzero(item_group[inverse] == g)
        if rows.size == 0:
            continue
        s = data[rows].split_for_nhpp_likelihood()
        for key, value in s.items():
            joined.setdefault(key, []).append(np.atleast_1d(value))
        for kind, (a, _) in _PIECES.items():
            index[kind].append(np.full(np.size(s[a]), g, dtype=int))
    out = {k: np.concatenate(v).astype(float) for k, v in joined.items()}
    idx = {
        k: (np.concatenate(v) if v else np.zeros(0, int))
        for k, v in index.items()
    }
    return {"pieces": out, "index": idx}


def _cif(pt: Any, x: np.ndarray, log_alpha: Any, beta: Any) -> Any:
    """``(x / alpha) ** beta``, 0 at ``x = 0`` with finite gradients."""
    pos = x > 0
    log_x = np.log(np.where(pos, x, 1.0))
    return pt.switch(pos, pt.exp(beta * (log_x - log_alpha)), 0.0)


def _log_lik(
    pt: Any, built: dict, log_alpha: Any, beta: Any, vector: bool
) -> Any:
    """The terms of ``nhpp_log_likelihood`` from the pieces."""
    from surpyval.pymc._logp import log1mexp

    s, idx = built["pieces"], built["index"]

    def at(kind: str) -> tuple[Any, Any]:
        if not vector:
            return log_alpha, beta
        return log_alpha[idx[kind]], beta[idx[kind]]

    total = pt.as_tensor_variable(np.float64(0.0))
    x_o = s.get("x_o", np.zeros(0))
    if x_o.size:
        la, b = at("o")
        log_iif = pt.log(b) - b * la + (b - 1.0) * np.log(x_o)
        total = total + pt.sum(
            log_iif + _cif(pt, s["x_o_prev"], la, b) - _cif(pt, x_o, la, b)
        )
    if np.size(s.get("x_right", [])):
        la, b = at("right")
        total = total + pt.sum(
            _cif(pt, s["x_right_prev"], la, b) - _cif(pt, s["x_right"], la, b)
        )
    for kind, n_key in (("left", "n_left"), ("i", "n_i")):
        lo_key, hi_key = _PIECES[kind]
        if np.size(s.get(hi_key, [])):
            la, b = at(kind)
            n = s[n_key]
            delta = _cif(pt, s[hi_key], la, b) - _cif(pt, s[lo_key], la, b)
            total = total + pt.sum(
                n * pt.log(delta) - delta - gammaln(n + 1.0)
            )
    for kind, sign in (("sel", -1.0), ("gap", 1.0)):
        lo_key, hi_key = _PIECES[kind]
        if np.size(s.get(hi_key, [])):
            la, b = at(kind)
            delta = _cif(pt, s[hi_key], la, b) - _cif(pt, s[lo_key], la, b)
            total = total + sign * pt.sum(log1mexp(pt, -delta))
    return total


def _item_groups(
    what: str, data: Any, group: Any, i: Any
) -> tuple[list, np.ndarray]:
    """The groups (sorted labels) and each system's group (in the order
    of ``data.ids``), from ``group``: ``None`` for each system its own
    group, a dict ``{system: group}``, or a label a row of ``i``."""
    ids = list(data.ids)
    if group is None:
        return [v.item() if hasattr(v, "item") else v for v in ids], (
            np.arange(len(ids))
        )
    if isinstance(group, dict):
        mapping = dict(group)
    else:
        g = np.asarray(group)
        if i is None:
            raise ValueError(
                f"{what}: a group a row needs the rows' systems, i (or "
                "give group as a dict {system: group})."
            )
        rows = np.asarray(i)
        if g.shape != rows.shape:
            raise ValueError(
                f"{what}: group must have one label per data row "
                f"({rows.size}); it has shape {g.shape}."
            )
        mapping = {}
        for sys_id, label in zip(rows.tolist(), g.tolist()):
            if mapping.setdefault(sys_id, label) != label:
                raise ValueError(
                    f"{what}: system {sys_id!r} has rows in more than one "
                    "group; a group is a set of whole systems."
                )
    missing = [k for k in ids if k not in mapping]
    if missing:
        raise ValueError(f"{what}: no group for the system(s) {missing[:5]}.")
    labels = sorted({mapping[k] for k in ids}, key=lambda v: (str(type(v)), v))
    where = {v: j for j, v in enumerate(labels)}
    return labels, np.array([where[mapping[k]] for k in ids], dtype=int)


def nhpp_log_likelihood(
    x: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    tl: Any = None,
    tr: Any = None,
    *,
    alpha: Any,
    beta: Any,
    group: Any = None,
) -> Any:
    """
    The power-law (Crow-AMSAA) NHPP log-likelihood of event histories,
    as a pytensor expression: exactly the likelihood ``CrowAMSAA.fit``
    maximises.

    Parameters
    ----------
    x, i, c, n, t, tl, tr
        The data, as for ``CrowAMSAA.fit``: event times pooled over
        systems (``i`` names each row's system; ``c=1`` rows close each
        system's observation), counts in windows, truncation; or an
        ``EventHistoryData`` as ``x``.
    alpha, beta
        The process's parameters: numbers or PyMC variables, one for
        every system; or, with ``group``, vectors of one value a group,
        in the order of the groups' sorted labels.
    group : array-like, optional
        Each row's group (a set of whole systems: a site, a build
        standard), or a dict ``{system: group}``; ``"system"`` for each
        system its own group (in the order of the systems' sorted ids).

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import numpy as np
    >>> from surpyval.pymc import nhpp_log_likelihood
    >>> from surpyval.recurrent import CrowAMSAA
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> c = [0] * 11 + [1]
    >>> ll = nhpp_log_likelihood(x, c=c, alpha=50.0, beta=0.8)
    >>> fit = CrowAMSAA.fit(x, c=c)
    >>> from surpyval.recurrent.bayes import RecurrenceTarget
    >>> exact = -RecurrenceTarget(fit.dist, fit.data).neg_ll([50.0, 0.8])
    >>> bool(np.isclose(ll.eval(), exact))
    True
    """
    what = "nhpp_log_likelihood"
    pt = pytensor_tensor(what)
    data, _ = event_data(what, x, i, c, n, t, tl, tr)
    return _expression(pt, data, alpha, beta, group, i, what)


def _expression(
    pt: Any, data: Any, alpha: Any, beta: Any, group: Any, i: Any, what: str
) -> Any:
    labels: list = []
    if group is None:
        built = _group_pieces(data, np.zeros(len(data.ids), int), 1)
        vector = False
    else:
        if isinstance(group, str) and group == "system":
            group = None
        labels, item_group = _item_groups(what, data, group, i)
        built = _group_pieces(data, item_group, len(labels))
        vector = True
    a, b = (
        pt.as_tensor_variable(
            v if hasattr(v, "type") else np.asarray(v, dtype=np.float64)
        )
        for v in (alpha, beta)
    )
    if vector:
        G = len(labels)
        for name, v in (("alpha", a), ("beta", b)):
            if v.ndim == 0:
                continue
            shape = getattr(v.type, "shape", (None,))
            if v.ndim != 1 or (shape[0] is not None and shape[0] != G):
                raise ValueError(
                    f"{what}: {name} has one value a group, {G} here (the "
                    "groups' sorted labels); got shape "
                    f"{shape}."
                )
        a = a * pt.ones(G) if a.ndim == 0 else a
        b = b * pt.ones(G) if b.ndim == 0 else b
    elif a.ndim or b.ndim:
        raise ValueError(
            f"{what}: alpha and beta are one value for every system here; "
            "for one a group, pass group= (or group='system')."
        )
    return _log_lik(pt, built, pt.log(a), b, vector)


def nhpp_likelihood(
    x: Any,
    i: Any = None,
    c: Any = None,
    n: Any = None,
    t: Any = None,
    tl: Any = None,
    tr: Any = None,
    *,
    alpha: Any,
    beta: Any,
    group: Any = None,
    name: str = "nhpp",
) -> Any:
    """
    Add the power-law NHPP likelihood of event histories to the open PyMC
    model (a ``pm.Potential`` named ``name``); see
    :func:`nhpp_log_likelihood`.

    Examples
    --------
    >>> import pytest; pm = pytest.importorskip("pymc")
    >>> import numpy as np
    >>> from surpyval.pymc import nhpp_likelihood
    >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
    >>> with pm.Model() as model:
    ...     beta = pm.LogNormal("beta", mu=np.log(0.8), sigma=0.3)
    ...     alpha = pm.LogNormal("alpha", mu=np.log(50.0), sigma=1.0)
    ...     term = nhpp_likelihood(x, c=[0] * 11 + [1], alpha=alpha, beta=beta)
    >>> term.name
    'nhpp'
    """
    what = "nhpp_likelihood"
    pm = pymc(what)
    pt = pytensor_tensor(what)
    try:
        pm.modelcontext(None)
    except TypeError as err:
        raise TypeError(
            f"{what} adds a term to a PyMC model: call it inside "
            "`with pm.Model():`."
        ) from err
    data, _ = event_data(what, x, i, c, n, t, tl, tr)
    term = _expression(pt, data, alpha, beta, group, i, what)
    return pm.Potential(name, term)


__all__ = ["nhpp_likelihood", "nhpp_log_likelihood"]
