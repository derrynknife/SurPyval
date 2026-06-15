import numpy as np
import numpy.typing as npt

from surpyval.utils import coerce_xcnt_x, format_truncation

from .recurrent_event_data import RecurrentEventData


def reject_left_truncation(data: RecurrentEventData, model_name: str) -> None:
    """
    Virtual-age and history-dependent models (Kijima/G1/ARA/ARI) cannot be
    fitted to left-truncated (delayed-entry) data: the virtual age or
    intensity reduction at entry depends on the unobserved pre-entry failure
    history. Only the calendar-time NHPP models support delayed entry.
    """
    if np.any(np.asarray(data.tl) > 0):
        raise ValueError(
            "{} does not support left truncation (tl > 0): the state at entry "
            "depends on the unobserved pre-entry history. Use an NHPP "
            "intensity model (HPP, CrowAMSAA, Duane, CoxLewis) for delayed "
            "entry.".format(model_name)
        )


def validate_renewal_censoring(c: npt.ArrayLike, model_name: str) -> None:
    """
    The renewal models only define likelihood contributions for exact events
    (``c=0``) and right-censored observations (``c=1``). Interval (``c=2``) and
    left (``c=-1``) censoring are not supported; reject them here rather than
    let them be silently dropped from the likelihood sum.
    """
    unsupported = sorted(set(np.unique(c).tolist()) - {0, 1})
    if unsupported:
        raise ValueError(
            "{} only supports exact (c=0) and right-censored (c=1) "
            "observations; received unsupported censoring code(s) {}. "
            "Interval (c=2) and left (c=-1) censoring are not "
            "supported.".format(model_name, unsupported)
        )


def handle_xicn(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
    tr: npt.ArrayLike | None = None,
    Z: npt.ArrayLike | dict | None = None,
    as_recurrent_data: bool = True,
) -> (
    RecurrentEventData
    | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
):
    x = coerce_xcnt_x(x)

    if i is None:
        i = np.ones(x.shape[0])
    else:
        i = np.array(i)

    if n is None:
        n = np.ones(x.shape[0])
    else:
        n = np.array(n)

    if c is None:
        c = np.zeros(x.shape[0])
    else:
        c = np.array(c)

    # Truncation follows surpyval's xcnt convention (shared with the univariate
    # handler): the default window is the whole real line, so no assumption is
    # made about the sign of ``x`` (e.g. a log-intensity model can take
    # negative values); the integration origin for untruncated NHPP data falls
    # back to 0 in ``get_previous_x``.
    truncation = format_truncation(t, tl, tr, x.shape[0])
    tl_arr = truncation[:, 0]
    tr_arr = truncation[:, 1]

    Z_arr: npt.NDArray | None = None
    if Z is not None:
        if isinstance(Z, dict):
            Z_arr = np.array([Z[ii] for ii in i])
        else:
            Z_arr = np.array(Z, ndmin=2)
    # TODO: Z as a dict where the keys are the item numbers and the arrays
    # are the covariates for each i at all times (x)

    if x.shape[0] != i.shape[0]:
        raise ValueError("x and i must have the same length")
    if x.shape[0] != c.shape[0]:
        raise ValueError("x and c must have the same length")
    if x.shape[0] != n.shape[0]:
        raise ValueError("x and n must have the same length")

    if Z_arr is not None:
        if x.shape[0] != Z_arr.shape[0]:
            raise ValueError("x and Z must have the same length")

    if np.any((n > 1) & ((c == 0) | (c == 1))):
        raise ValueError(
            "Counts greater than 1 must be intervally or left censored"
        )

    # sort by item and x
    if x.ndim == 2:
        # Order 2D by the midpoint
        sort_order = np.lexsort((x.mean(axis=1), i))
    else:
        sort_order = np.lexsort((x, i))

    x, i, c, n = x[sort_order], i[sort_order], c[sort_order], n[sort_order]
    tl_arr, tr_arr = tl_arr[sort_order], tr_arr[sort_order]

    if Z_arr is not None:
        Z_arr = Z_arr[sort_order]

    unique_i, idx = np.unique(i, return_index=True)
    censoring_by_i = np.split(c, idx)[1:]

    for ii, arr in zip(unique_i, censoring_by_i):
        if 1 in arr:
            if (arr == 1).sum() > 1:
                raise ValueError(
                    f"Item {ii} has more than one right censored time"
                )
            if arr[-1] != 1:
                raise ValueError(
                    f"Item {ii} has right censored event which is not the last"
                )
        if -1 in arr:
            if (arr == -1).sum() > 1:
                raise ValueError(
                    f"Item {ii} has more than one left censored event"
                )
            if arr[0] != -1:
                raise ValueError(
                    f"Item {ii} has left censored event that is not the first"
                )

    if x.ndim == 2:
        times_by_i = np.split(x, idx)[1:]
        for ii, arr in zip(unique_i, times_by_i):
            starts = arr[1:][:, 0]
            ends = arr[:-1][:, 1]
            if (ends > starts).any():
                raise ValueError(f"Item {ii} has overlapping intervals")

    # Truncation defines a single observation window [tl, tr] per item, so the
    # bounds must be constant within an item and contain all of its events.
    tl_by_i = np.split(tl_arr, idx)[1:]
    tr_by_i = np.split(tr_arr, idx)[1:]
    x_lower = x if x.ndim == 1 else x[:, 0]
    x_upper = x if x.ndim == 1 else x[:, 1]
    xl_by_i = np.split(x_lower, idx)[1:]
    xu_by_i = np.split(x_upper, idx)[1:]
    for ii, tl_i, tr_i, xl_i, xu_i in zip(
        unique_i, tl_by_i, tr_by_i, xl_by_i, xu_by_i
    ):
        if not (np.all(tl_i == tl_i[0]) and np.all(tr_i == tr_i[0])):
            raise ValueError(
                f"Item {ii} has inconsistent truncation bounds; each item "
                "must have a single observation window."
            )
        if tl_i[0] > tr_i[0]:
            raise ValueError(f"Item {ii} has left truncation beyond right")
        if (xl_i < tl_i[0]).any() or (xu_i > tr_i[0]).any():
            raise ValueError(
                f"Item {ii} has events outside its truncation window "
                f"[{tl_i[0]}, {tr_i[0]}]"
            )

    if as_recurrent_data:
        data = RecurrentEventData(x, i, c, n, tl=tl_arr, tr=tr_arr)
        data.Z = Z_arr
        return data
    else:
        return x, i, c, n
