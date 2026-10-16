"""The reshaping operations of lifetime data (:class:`SurpyvalData`) on
its arrays (#932): ``without``, ``with_covariates``, ``for_event_type``,
``relabel_event_types``, ``select``, ``censor_at`` and ``enter_at`` give
the same data as rebuilding every item's record and reshaping it
(:mod:`surpyval.utils.data_reshape`, which the tests compare them with),
without building the records: at 50,000 items a reshape took seconds
through the records and takes milliseconds here.

Each function takes the dataset and returns a new one built from the
changed arrays by the constructor, which checks them as it checks any
arrays.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from surpyval.utils.warnings import MaskedCauseWarning


def _columns(data: Any) -> np.ndarray:
    """``x`` as ``[lower, upper]`` rows."""
    x = np.asarray(data.x, dtype=float)
    return x.copy() if x.ndim == 2 else np.column_stack([x, x])


def rebuilt(
    data: Any,
    keep: "np.ndarray | None" = None,
    *,
    x: "np.ndarray | None" = None,
    c: "np.ndarray | None" = None,
    tl: "np.ndarray | None" = None,
    tr: "np.ndarray | None" = None,
    Z: Any = "same",
    e: Any = "same",
    names: Any = "same",
    time_varying: Any = "same",
) -> Any:
    """A dataset like ``data`` with the rows ``keep`` (all, where None)
    and the arrays given replaced: the item ids and time-varying
    covariates follow the rows kept."""
    from surpyval.utils.surpyval_data import SurpyvalData

    x = _columns(data) if x is None else np.asarray(x, dtype=float)
    c = np.asarray(data.c if c is None else c)
    n = np.asarray(data.n)
    tl = np.asarray(data.tl if tl is None else tl, dtype=float)
    tr = np.asarray(data.tr if tr is None else tr, dtype=float)
    Z = data.Z if isinstance(Z, str) else Z
    e = data.e if isinstance(e, str) else e
    names = data.covariate_names if isinstance(names, str) else names
    time_varying = (
        data.time_varying_covariates
        if isinstance(time_varying, str)
        else time_varying
    )
    ids = data.ids
    attributes = getattr(data, "attributes", None)
    if keep is not None:
        keep = np.asarray(keep, dtype=bool)
        if not keep.any():
            raise ValueError("No items are left.")
        rows = np.flatnonzero(keep)
        x, c, n, tl, tr = x[rows], c[rows], n[rows], tl[rows], tr[rows]
        Z = None if Z is None else np.asarray(Z)[rows]
        e = None if e is None else e[rows]
        ids = None if ids is None else [ids[r] for r in rows]
        if attributes is not None:
            attributes = [attributes[r] for r in rows]
    if not (c == 2).any():
        x = x[:, 0]
    if e is not None and all(v is None for v in e):
        # No typed failure is left: no event types, as from_items gives
        e = None
    truncated = np.isfinite(tl).any() or np.isfinite(tr).any()
    with warnings.catch_warnings():
        # Warned of when the data was built
        warnings.simplefilter("ignore", MaskedCauseWarning)
        out = SurpyvalData(
            x,
            c,
            n,
            tl=tl if truncated else None,
            tr=tr if truncated else None,
            Z=Z,
            group_and_sort=False,
            e=e,
            covariate_names=names if Z is not None else None,
            unit=data.unit,
        )
    out.ids = ids
    out.attributes = attributes
    if time_varying is not None:
        if ids is not None:
            out.time_varying_covariates = {i: time_varying[i] for i in ids}
        else:
            # Keyed by row: the rows kept, numbered afresh
            old_rows = (
                np.flatnonzero(keep) if keep is not None else np.arange(len(x))
            )
            out.time_varying_covariates = {
                new: time_varying[int(old)] for new, old in enumerate(old_rows)
            }
    return out


def without(data: Any, parts: tuple) -> Any:
    """``data`` without the ``parts`` (:func:`data_reshape.without`)."""
    changes: dict = {}
    if "covariates" in parts:
        changes["Z"] = None
        changes["names"] = None
    if "event_types" in parts:
        changes["e"] = None
    if "time_varying_covariates" in parts:
        changes["time_varying"] = None
    if "truncation" in parts:
        changes["tl"] = np.full(len(data.c), -np.inf)
        changes["tr"] = np.full(len(data.c), np.inf)
    out = rebuilt(data, **changes)
    if "attributes" in parts:
        out.attributes = None
    return out


def with_covariates(data: Any, columns: list) -> Any:
    """``data`` with only the covariate ``columns``, named or numbered
    (:func:`data_reshape.with_covariates`)."""
    Z = np.asarray(data.Z, dtype=float)
    Z = Z[:, None] if Z.ndim == 1 else Z
    width = Z.shape[1]
    names = data.covariate_names
    picked = []
    for column in columns:
        if isinstance(column, (int, np.integer)) and not isinstance(
            column, bool
        ):
            if not -width <= column < width:
                raise ValueError(
                    f"Covariate column {column!r} is not among the {width} "
                    f"column(s) (0 to {width - 1})."
                )
            picked.append(int(column) % width)
        elif names is None:
            raise ValueError(
                f"Covariates are picked by column number (0 to {width - 1}), "
                "and these covariates have no names (name them with "
                f"covariates={{name: value}}); got {column!r}."
            )
        elif column not in names:
            raise ValueError(
                f"This data has no covariate {column!r}; its covariates "
                f"are {list(names)!r}."
            )
        else:
            picked.append(list(names).index(column))
    if len(set(picked)) < len(picked):
        twice = [c for c in columns if columns.count(c) > 1] or columns
        raise ValueError(
            f"Covariate {twice[0]!r} is picked more than once: each column "
            "once (the same column twice is not two covariates)."
        )
    return rebuilt(
        data,
        Z=Z[:, picked],
        names=None if names is None else [names[j] for j in picked],
    )


def relabel(data: Any, mapping: dict) -> Any:
    """``data`` with its failures' event types renamed by ``mapping``
    (:func:`data_reshape.relabel`): a suspension keeps none."""
    e = np.empty(len(data.e), dtype=object)
    e[:] = [
        (v if flag == 1 else mapping.get(v, v))
        for v, flag in zip(data.e, np.asarray(data.c))
    ]
    return rebuilt(data, e=e)


def for_event_type(data: Any, types: set, masked: str) -> Any:
    """``data`` as data of the failure modes ``types`` (one, or several
    taken as one), as :func:`data_reshape.for_event_type` makes each
    record: the other modes' failures (and the masked causes, unless
    ``masked == "count"``) are right censored at the age the item was
    last known to be without this mode's failure; its right truncation is
    dropped. The result has no event types."""
    x = _columns(data)
    c = np.asarray(data.c).copy()
    tl = np.asarray(data.tl, dtype=float)
    of_type = np.array(
        [
            flag == 1 or v in types or (v is None and masked == "count")
            for v, flag in zip(data.e, c)
        ],
        dtype=bool,
    )
    other = ~of_type
    keep = np.ones(len(c), dtype=bool)
    # A failure of another mode at a known age: last known without this
    # mode's failure at that age
    observed = other & (c == 0)
    c[observed] = 1
    # Between inspections: without it at the last good inspection, unless
    # that was before the item came under observation
    interval = other & (c == 2)
    lower = x[:, 0]
    keep[interval & np.isfinite(tl) & (lower <= tl)] = False
    x[interval, 1] = lower[interval]
    c[interval] = 1
    # Before a look: no age known at which it was without this mode
    keep[other & (c == -1)] = False
    return rebuilt(
        data,
        keep,
        x=x,
        c=c,
        tr=np.full(len(c), np.inf),
        e=None,
    )


def select(data: Any, rows: np.ndarray) -> Any:
    """The rows ``rows`` of ``data``, in its order."""
    keep = np.zeros(len(data.c), dtype=bool)
    keep[rows] = True
    return rebuilt(data, keep)


def censor_at(data: Any, age: float) -> Any:
    """``data`` as seen if observation stopped at ``age``
    (:func:`data_reshape.censor_at`)."""
    x = _columns(data)
    c = np.asarray(data.c).copy()
    tl = np.asarray(data.tl, dtype=float)
    lower, upper = x[:, 0].copy(), x[:, 1].copy()
    keep = ~(np.isfinite(tl) & (tl >= age))
    changed = np.zeros(len(c), dtype=bool)
    # Seen past the end: last seen working at the end
    late = np.isin(c, (0, 1)) & (upper > age)
    changed |= late
    # Found failed at a look after the end: nothing is known of it
    keep &= ~((c == -1) & (upper > age))
    # Failed between inspections across the end: worked at the last good
    # one (unless that was before it came under observation); a window
    # wholly after the end: last seen working at the end
    across = (c == 2) & (lower < age) & (upper > age)
    keep &= ~(across & np.isfinite(tl) & (lower <= tl))
    after = (c == 2) & (lower >= age)
    new_age = np.where(across, lower, age)
    changed |= across | after
    lower[changed] = new_age[changed]
    upper[changed] = new_age[changed]
    c[changed] = 1
    e = data.e
    if e is not None:
        e = e.copy()
        e[changed] = None
    return rebuilt(data, keep, x=np.column_stack([lower, upper]), c=c, e=e)


def enter_at(data: Any, age: float) -> Any:
    """``data`` as if observed only from ``age``
    (:func:`data_reshape.enter_at`)."""
    x = _columns(data)
    c = np.asarray(data.c)
    tl = np.asarray(data.tl, dtype=float).copy()
    tr = np.asarray(data.tr, dtype=float)
    already = np.isfinite(tl) & (tl >= age)
    keep = np.ones(len(c), dtype=bool)
    # It failed by tr, at or before age: not known to reach age
    keep &= already | ~(np.isfinite(tr) & (tr <= age))
    # Not known to reach age without failing
    keep &= already | ~(np.isin(c, (0, 1)) & (x[:, 1] <= age))
    keep &= already | (c != -1)
    keep &= already | ~((c == 2) & (x[:, 0] < age))
    tl[~already] = age
    return rebuilt(data, keep, tl=tl)
