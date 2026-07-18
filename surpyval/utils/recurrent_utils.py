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


def reject_unsupported_nonparametric(
    data: RecurrentEventData, model_name: str
) -> None:
    """
    The nonparametric MCF estimators (``NonParametricCounting`` and
    ``CauseSpecificMCF``) currently only support exact events (``c=0``) and
    right-censored end-of-observation rows (``c=1``), with at most a left
    truncation (delayed entry) on the observation window. Right truncation,
    left censoring and interval censoring are not yet handled correctly by the
    risk-set construction, so reject them up front rather than silently
    returning a wrong MCF.
    """
    if np.any(np.isfinite(np.asarray(data.tr))):
        raise ValueError(
            "{} does not support right truncation (finite tr) yet.".format(
                model_name
            )
        )

    c = np.asarray(data.c)
    if np.any(c == -1):
        raise ValueError(
            "{} does not support left-censored (c=-1) observations "
            "yet.".format(model_name)
        )
    if np.any(c == 2):
        raise ValueError(
            "{} does not support interval-censored (c=2) observations "
            "yet.".format(model_name)
        )


def validate_memory(m: object) -> None:
    """
    The Arithmetic Reduction of Age/Intensity models (``ARA``/``ARI``) are
    parameterised by an integer memory ``m`` (how many prior failures the
    repair acts on), with ``m = numpy.inf`` recovering the infinite-memory
    limit. Reject anything that is neither a positive integer nor ``inf``.
    """
    if m == np.inf:
        return
    if not (isinstance(m, (int, np.integer)) and m >= 1):
        raise ValueError(
            "m must be a positive integer or numpy.inf; got {!r}".format(m)
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


def reject_gapped_observation(
    data: RecurrentEventData, model_name: str
) -> None:
    """
    The virtual-age / imperfect-repair models (Kijima/G1/ARA/ARI) cannot be
    fitted to gapped (multi-window) observation. Their likelihood is built from
    the virtual age carried across the *entire* history, so a gap in which the
    item was unobserved -- events may have occurred but were not recorded --
    breaks the age bookkeeping: the state at the start of a later window is
    unknown. The intensity (NHPP) models factorise over disjoint windows and so
    do support gaps.
    """
    if getattr(data, "window_map", None) is not None:
        raise ValueError(
            "{} does not support gapped (multi-window) observation: the "
            "virtual age at the start of a later window depends on the "
            "unobserved events during the gap. Use an NHPP intensity model "
            "(HPP, CrowAMSAA, Duane, CoxLewis) for gapped data.".format(
                model_name
            )
        )


def _expand_windows(
    x: npt.NDArray,
    i: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    windows: dict,
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    dict,
]:
    """
    Expand multi-window (gapped) observation into synthetic single-window
    sub-items.

    Poisson event counts over disjoint windows are independent, so a gapped
    item's NHPP likelihood factorises over its windows. Representing each
    window as its own single-window item -- its events (``c=0``) plus a
    right-censored close (``c=1``) at the window end, entering (``tl``) at the
    window start -- reproduces exactly that factorised likelihood, and makes
    the existing NHPP likelihood *and* the nonparametric MCF at-risk set
    handle gaps unchanged: an item is simply absent from the risk set during
    a gap.

    ``windows`` is a mapping ``{item: [(start, end), ...]}`` giving each item's
    disjoint observation windows. Every provided row must be an observed event
    (``c=0``); the windows supply the censoring / close rows. Returns the
    expanded ``(x, i, c, n, tl, tr)`` arrays and a ``window_map`` from each
    synthetic item id to its ``(real_item, (start, end))``.
    """
    if x.ndim != 1:
        raise ValueError(
            "windows (gapped observation) is not supported for "
            "interval-valued (2D) event times"
        )
    if not isinstance(windows, dict):
        raise ValueError(
            "windows must be a dict mapping each item to a list of "
            "(start, end) observation windows"
        )
    if np.any(np.asarray(c) != 0):
        raise ValueError(
            "with windows, every provided row must be an observed event "
            "(c=0); the observation windows supply the censoring / close rows"
        )

    unique_i = np.unique(i)
    missing = [ii for ii in unique_i.tolist() if ii not in windows]
    if missing:
        raise ValueError(
            "windows must be given for every item; missing "
            "{}".format(missing)
        )

    new_x: list = []
    new_i: list = []
    new_c: list = []
    new_n: list = []
    new_tl: list = []
    new_tr: list = []
    window_map: dict = {}
    synth = 0
    for ii in unique_i:
        wins = [tuple(w) for w in windows[ii]]
        if len(wins) == 0:
            raise ValueError("item {} has no observation windows".format(ii))
        for w in wins:
            if len(w) != 2:
                raise ValueError(
                    "item {} has a malformed window {!r}; each window must be "
                    "a (start, end) pair".format(ii, w)
                )
            a, b = float(w[0]), float(w[1])
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(
                    "item {} has a non-finite observation window "
                    "({}, {})".format(ii, w[0], w[1])
                )
            if not (a < b):
                raise ValueError(
                    "item {} has an empty or reversed observation window "
                    "({}, {}); require start < end".format(ii, w[0], w[1])
                )
        # sort windows by start and require they are disjoint (touching, i.e.
        # end == next start, is allowed)
        wins = sorted(wins, key=lambda w: float(w[0]))
        for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
            if float(a2) < float(b1):
                raise ValueError(
                    "item {} has overlapping observation windows "
                    "{} and {}".format(ii, (a1, b1), (a2, b2))
                )

        mask = np.asarray(i) == ii
        item_x = np.asarray(x)[mask]
        item_n = np.asarray(n)[mask]
        assigned = np.zeros(item_x.shape[0], dtype=bool)
        for w in wins:
            a, b = float(w[0]), float(w[1])
            synth += 1
            in_win = (item_x >= a) & (item_x <= b) & (~assigned)
            assigned |= in_win
            for xv, nv in zip(item_x[in_win], item_n[in_win]):
                new_x.append(float(xv))
                new_i.append(synth)
                new_c.append(0)
                new_n.append(nv)
                new_tl.append(a)
                new_tr.append(np.inf)
            # window close: right-censored end-of-window row at b
            new_x.append(b)
            new_i.append(synth)
            new_c.append(1)
            new_n.append(1)
            new_tl.append(a)
            new_tr.append(np.inf)
            window_map[synth] = (ii, (a, b))
        if not assigned.all():
            outside = item_x[~assigned].tolist()
            raise ValueError(
                "item {} has events outside all its observation windows: "
                "{}".format(ii, outside)
            )

    return (
        np.array(new_x, dtype=float),
        np.array(new_i),
        np.array(new_c, dtype=float),
        np.array(new_n),
        np.array(new_tl, dtype=float),
        np.array(new_tr, dtype=float),
        window_map,
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
    windows: dict | None = None,
) -> (
    RecurrentEventData
    | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
):
    x = coerce_xcnt_x(x)

    if x.shape[0] == 0:
        raise ValueError("'x' cannot be empty")

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

    # Gapped (multi-window) observation: each item is observed over several
    # disjoint windows with unobserved gaps between them. Expand each window
    # into a synthetic single-window sub-item so the rest of the handler --
    # and the NHPP likelihood and MCF at-risk set downstream -- treat the gaps
    # correctly without any special-casing (see ``_expand_windows``).
    window_map: dict | None = None
    tl_gap: npt.NDArray | None = None
    tr_gap: npt.NDArray | None = None
    if windows is not None:
        if t is not None or tl is not None or tr is not None:
            raise ValueError(
                "windows defines each item's observation windows, so t, tl "
                "and tr must not also be supplied"
            )
        if Z is not None:
            raise ValueError(
                "windows (gapped observation) does not support covariates Z"
            )
        x, i, c, n, tl_gap, tr_gap, window_map = _expand_windows(
            x, i, c, n, windows
        )

    # Truncation follows surpyval's xcnt convention (shared with the univariate
    # handler): the default window is the whole real line. No global sign
    # assumption is made about ``x`` here -- each item is instead validated
    # against its own observation window below. An item with an explicit
    # (possibly negative) left-truncation bound legitimately admits negative
    # event times; an untruncated item is integrated from the fallback origin
    # 0 (see ``get_previous_x``) and so must have non-negative event times.
    if tl_gap is not None and tr_gap is not None:
        tl_arr = tl_gap
        tr_arr = tr_gap
    else:
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

    # --- Value validation ------------------------------------------------
    # Reject malformed input with informative errors rather than letting
    # NaN/inf or nonsensical counts and codes flow silently into the
    # optimiser. NaN in ``x`` is already rejected by ``coerce_xcnt_x``; here
    # the remaining degenerate values are caught.
    if not np.isfinite(x).all():
        raise ValueError("Event times 'x' must be finite (no inf values)")

    if np.issubdtype(i.dtype, np.number) and not np.isfinite(i).all():
        raise ValueError("Item identifiers 'i' must be finite (no NaN or inf)")

    # Censoring codes: -1 left, 0 observed, 1 right, 2 interval. ``np.isin``
    # also flags NaN, which is never a valid code.
    valid_c = np.isin(c, [-1, 0, 1, 2])
    if not valid_c.all():
        bad = np.unique(c[~valid_c]).tolist()
        raise ValueError(
            "Censoring 'c' must be one of -1 (left), 0 (observed), "
            f"1 (right), or 2 (interval); got {bad}"
        )

    if not np.isfinite(n).all():
        raise ValueError("Counts 'n' must be finite")
    if np.any(n <= 0):
        raise ValueError("Counts 'n' must be strictly positive")

    # Truncation bounds may be +/-inf (the default open window) but a NaN
    # bound is meaningless.
    if np.isnan(tl_arr).any() or np.isnan(tr_arr).any():
        raise ValueError("Truncation bounds must not contain NaN")

    if (
        Z_arr is not None
        and np.issubdtype(Z_arr.dtype, np.number)
        and not np.isfinite(Z_arr).all()
    ):
        raise ValueError("Covariates 'Z' must be finite (no NaN or inf)")

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
        # The item's first interval is integrated from its entry time: the
        # left-truncation bound when finite, otherwise the fallback origin 0
        # (see RecurrentEventData.get_previous_x). Events below that origin
        # would give negative interarrival times, so they are rejected. This
        # is why untruncated event times must be non-negative while an
        # explicit (possibly negative) left-truncation window admits negative
        # times.
        lower = tl_i[0] if np.isfinite(tl_i[0]) else 0.0
        if (xl_i < lower).any() or (xu_i > tr_i[0]).any():
            raise ValueError(
                f"Item {ii} has events outside its observation window "
                f"[{lower}, {tr_i[0]}]"
            )

    if as_recurrent_data:
        data = RecurrentEventData(x, i, c, n, tl=tl_arr, tr=tr_arr)
        data.Z = Z_arr
        # ``window_map`` (synthetic-item id -> (real item, (start, end))) is
        # set only for gapped observation; it stays ``None`` otherwise and is
        # what ``reject_gapped_observation`` keys off. ``observation_windows``
        # keeps the user's original per-item windows for reference.
        data.window_map = window_map
        data.observation_windows = windows if window_map is not None else None
        return data
    else:
        return x, i, c, n
