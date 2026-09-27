from typing import Literal, overload

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
    # These models measure time from the start of each item's life (the
    # first gap starts at 0 whatever ``tl`` says), so a negative time --
    # admitted by ``handle_xicn`` under a negative ``tl`` -- would give a
    # negative gap.
    if np.any(np.asarray(data.x, dtype=float) < 0):
        raise ValueError(
            "{} measures times from the start of each item's life, so they "
            "cannot be negative.".format(model_name)
        )


def reject_unsupported_nonparametric(
    data: RecurrentEventData, model_name: str
) -> None:
    """
    The nonparametric MCF estimators (``NonParametricCounting`` and
    ``CauseSpecificMCF``) currently only support exact events (``c=0``) and
    right-censored end-of-observation rows (``c=1``), on an observation window
    that may be left truncated (delayed entry, ``tl``) and right truncated
    (``tr``, which closes the window like an end-of-observation row). Left
    and interval censoring are not yet handled correctly by the risk-set
    construction, so reject them up front rather than silently returning a
    wrong MCF.
    """
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


def validate_restoration(
    value: object,
    name: str,
    bounds: "tuple[float | None, float | None]",
    open_lower: bool = False,
) -> None:
    """
    Check a repair / restoration parameter given to ``fit_from_parameters``
    against the range its model is defined on (the fitters already search
    only inside it). Outside it the virtual ages go negative (a Kijima
    ``q < 0``, an ARA ``rho > 1``) or the G1 scale ``(1 + q) ** j`` stops
    being a positive scale (``q <= -1``), and the simulation returns
    meaningless or failing sequences instead of an error.
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            "{} must be a number; got {!r}".format(name, value)
        ) from None
    lower, upper = bounds
    low_ok = lower is None or (v > lower if open_lower else v >= lower)
    high_ok = upper is None or v <= upper
    if not (np.isfinite(v) and low_ok and high_ok):
        low_txt = "-inf" if lower is None else str(lower)
        high_txt = "inf" if upper is None else str(upper)
        interval = "{}{}, {}{}".format(
            "(" if open_lower or lower is None else "[",
            low_txt,
            high_txt,
            ")" if upper is None else "]",
        )
        raise ValueError(
            "{} must be finite and in {}; got {!r}".format(
                name, interval, value
            )
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


def validate_nhpp_data(data: RecurrentEventData, dist: object) -> None:
    """
    Reject data an intensity (NHPP) model cannot be fitted to, rather than
    return an optimiser's meaningless stopping point as the MLE.

    - No events at all (every row an end-of-observation ``c=1`` row): the
      likelihood only ever rewards a lower intensity, so there is no
      maximum.
    - Times outside the intensity's support. The power-law models
      (``CrowAMSAA``, ``Duane``) are defined for ``t >= 0`` and their
      intensity is 0 or infinite at ``t = 0``, so an event there makes the
      likelihood unbounded and a window reaching below 0 (a negative
      ``tl``) is outside the model altogether.
    - A single failure-truncated event for a model with two or more
      parameters: one time point cannot identify two parameters (the
      Crow-AMSAA ``beta`` runs off to infinity).
    """
    c = np.asarray(data.c)
    name = getattr(dist, "name", type(dist).__name__)
    if not np.any(c != 1):
        raise ValueError(
            "The data has no events (every row is an end-of-observation "
            "c=1 row), so the {} intensity cannot be estimated.".format(name)
        )

    lower, upper = getattr(dist, "support", (-np.inf, np.inf))
    x = np.asarray(data.x, dtype=float)
    x_lo = x if x.ndim == 1 else x[:, 0]
    x_hi = x if x.ndim == 1 else x[:, 1]
    tl = np.asarray(data.tl, dtype=float)
    tr = np.asarray(data.tr, dtype=float)
    # Every time the likelihood integrates over -- event and censoring
    # rows, and finite window bounds -- must lie in the closed support.
    times = np.concatenate(
        [x_lo, x_hi, tl[np.isfinite(tl)], tr[np.isfinite(tr)]]
    )
    if np.any(times < lower) or np.any(times > upper):
        raise ValueError(
            "The {} intensity is defined on [{}, {}], but the data has "
            "times (event, censoring or truncation) outside it.".format(
                name, lower, upper
            )
        )
    # An exact event on the boundary of the support has a zero or infinite
    # intensity there (the power law's t**(beta - 1) at t = 0).
    exact = x_lo[c == 0]
    if np.isfinite(lower) and np.any(exact == lower):
        raise ValueError(
            "The data has an event at t = {}, the edge of the {} "
            "intensity's support, where the intensity is 0 or infinite; "
            "the likelihood has no maximum. Record events at positive "
            "times.".format(lower, name)
        )

    n_params = len(getattr(dist, "param_names", ()))
    events = float(np.asarray(data.n)[c == 0].sum())
    window_closed = np.any(c != 0) or np.any(np.isfinite(tr))
    if n_params >= 2 and events == 1 and not window_closed:
        raise ValueError(
            "The data has a single event and no observation beyond it "
            "(failure truncated), which cannot identify the {} parameters "
            "of the {} intensity. Add the end of the observation window "
            "(a c=1 row or tr).".format(n_params, name)
        )


def validate_renewal_times(
    data: RecurrentEventData,
    dist: object,
    model_name: str,
    every_gap_from_new: bool = False,
) -> None:
    """
    Check the event times a lifetime-distribution renewal model can use.

    (Negative times are rejected by ``reject_left_truncation``.) For a
    lifetime distribution on ``[0, inf)`` a gap measured from virtual age
    0 must be positive: its density at 0 is 0 or
    infinite for most shapes (a Weibull's, for one), so the likelihood has
    no maximum. The first gap of every item starts at age 0, so an event
    at time 0 is rejected; with ``every_gap_from_new`` (the G1 process,
    where each gap is a rescaled fresh lifetime) a zero gap anywhere --
    a tied event time within an item -- is rejected too. The virtual-age
    models allow tied events later on, where the virtual age is positive.
    """
    x = np.asarray(data.x, dtype=float)
    support = getattr(dist, "support", (0.0, np.inf))
    if support[0] < 0:
        return
    c = np.asarray(data.c)
    gaps = data.get_interarrival_times()
    _, first = np.unique(data.i, return_index=True)
    is_first = np.zeros(len(x), dtype=bool)
    is_first[first] = True
    exact_zero = (gaps == 0) & (c == 0)
    if np.any(exact_zero & is_first):
        raise ValueError(
            "{} has an event at time 0: the gap from a new item's age 0 is "
            "then zero, where the {} density is 0 or infinite, so the "
            "likelihood has no maximum. Record events at positive "
            "times.".format(model_name, getattr(dist, "name", "lifetime"))
        )
    if every_gap_from_new and np.any(exact_zero):
        raise ValueError(
            "{} has tied event times within an item. Each G1 interarrival "
            "time is a rescaled fresh lifetime, so a zero gap is impossible "
            "under a continuous {} lifetime; combine tied events or "
            "separate them in time.".format(
                model_name, getattr(dist, "name", "lifetime")
            )
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


# ``as_recurrent_data`` picks the return shape, so the two cases are
# declared separately. Without this every caller taking the default gets
# the union back and has to narrow it, which is nine call sites saying
# the same thing about a value the argument already determined.
@overload
def handle_xicn(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
    tr: npt.ArrayLike | None = None,
    Z: npt.ArrayLike | dict | None = None,
    as_recurrent_data: Literal[True] = True,
    windows: dict | None = None,
    e: npt.ArrayLike | None = None,
) -> RecurrentEventData: ...


@overload
def handle_xicn(
    x: npt.ArrayLike,
    i: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | None = None,
    tr: npt.ArrayLike | None = None,
    Z: npt.ArrayLike | dict | None = None,
    *,
    as_recurrent_data: Literal[False],
    windows: dict | None = None,
    e: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]: ...


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
    e: npt.ArrayLike | None = None,
) -> (
    RecurrentEventData
    | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
):
    """
    Validate recurrent-event data given as arrays and assemble it into a
    :class:`~surpyval.utils.recurrent_event_data.RecurrentEventData`, the
    object every recurrent fitter's ``fit_from_recurrent_data`` takes.

    Each row is one event (or the right-censored end of observation) of
    the item named in ``i``, at time ``x`` measured from the start of that
    item's life.

    Parameters
    ----------
    x : array like
        The event times (a 2-D ``[left, right]`` row for an
        interval-censored count).
    i : array like, optional
        The item each row belongs to. Defaults to one item.
    c : array like, optional
        Censoring flags: 0 an observed event, 1 the right-censored end of
        observation, -1 left-censored and 2 interval-censored counts.
        Defaults to all observed. Rows are sorted by item and time; an
        end-of-observation row at the same time as an event goes after
        it, whatever the input order.
    n : array like, optional
        The number of events in each row. Defaults to 1.
    t : array like, optional
        (N, 2) truncation bounds per row. Use ``tl`` / ``tr`` for per-item
        bounds instead.
    tl, tr : array like or scalar, optional
        Left-truncation (start of observation) and right-truncation (end
        of observation) times. An item with both a ``c=1`` row and a
        finite ``tr`` must have them at the same time.
    Z : array like or dict, optional
        Covariates: one row per row of ``x`` (the same on every row of an
        item: covariates are per item), or a ``{item: covariates}``
        mapping applied to every row of that item.
    as_recurrent_data : bool, optional
        If :code:`True` (the default) return a ``RecurrentEventData``;
        otherwise return the validated ``(x, i, c, n)`` arrays.
    windows : dict, optional
        Gapped observation: ``{item: [(start, end), ...]}``. Every row must
        then be an observed event; the windows supply the censoring rows.
        Not combinable with ``t``/``tl``/``tr``, ``Z`` or ``e``.
    e : array like, optional
        The event type (mark) of each row, for the cause-specific models;
        a missing value marks a row with no cause (such as the censoring
        row).

    Returns
    -------
    RecurrentEventData or tuple
        The assembled data, or ``(x, i, c, n)``.

    Examples
    --------
    >>> from surpyval import handle_xicn
    >>> data = handle_xicn([2, 5, 8, 3, 9], i=[1, 1, 1, 2, 2],
    ...                    c=[0, 0, 1, 0, 1])
    >>> data.items
    [1, 2]
    """
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

    # Optional event-type marks (competing-risks recurrent process). Marks are
    # per-row and aligned with ``x``; ``None``/NaN marks (typically the
    # end-of-observation censoring row) are permitted. Kept as an object array
    # so string, integer or ``None`` marks all round-trip unchanged.
    e_arr: npt.NDArray | None = None
    if e is not None:
        from surpyval.utils import is_missing_event

        e_arr = np.array(e, dtype=object)
        if e_arr.shape[0] != x.shape[0]:
            raise ValueError("x and e must have the same length")
        # Normalise every "no attributed cause" marker (None, NaN, pandas NA)
        # to Python ``None`` so downstream cause bookkeeping (``event_types``,
        # cause-specific counts) sees a single missing sentinel.
        e_arr = np.array(
            [None if is_missing_event(v) else v for v in e_arr], dtype=object
        )

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
        if e_arr is not None:
            raise ValueError(
                "windows (gapped observation) does not support event-type "
                "marks e yet"
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
            missing = [ii for ii in np.unique(i).tolist() if ii not in Z]
            if missing:
                raise ValueError(
                    "Z has no covariates for item(s) {}".format(missing)
                )
            # a scalar value is a single covariate (it used to give a 1-D
            # array and an IndexError further on)
            Z_arr = np.array(
                [np.atleast_1d(np.asarray(Z[ii], dtype=float)) for ii in i]
            )
        else:
            Z_arr = np.asarray(Z, dtype=float)
            if Z_arr.ndim == 1:
                # one covariate: a value per row, not one row of values
                Z_arr = Z_arr.reshape(-1, 1)

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
    if i.dtype == object:
        from surpyval.utils import is_missing_event

        # A missing id cannot say which item a row belongs to, and the sort
        # below would otherwise fail with a bare "'<' not supported".
        if any(is_missing_event(v) for v in i):
            raise ValueError(
                "Item identifiers 'i' must not be missing (None or NaN)"
            )

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

    # Sort by item, then time, then censoring code. An end-of-observation
    # (c=1) row tied with an event at the same time closes the window after
    # it, so ties put it last (and a left-censored count, which covers the
    # time from entry, first); otherwise whether the input was accepted
    # depended on the order the tied rows happened to be given in.
    tie_order = np.where(c == 1, 3, c)
    x_key = x.mean(axis=1) if x.ndim == 2 else x  # 2D by the midpoint
    try:
        sort_order = np.lexsort((tie_order, x_key, i))
    except TypeError:
        raise ValueError(
            "Item identifiers 'i' must be of one comparable kind (all "
            "numbers or all strings)"
        ) from None

    x, i, c, n = x[sort_order], i[sort_order], c[sort_order], n[sort_order]
    tl_arr, tr_arr = tl_arr[sort_order], tr_arr[sort_order]

    if Z_arr is not None:
        Z_arr = Z_arr[sort_order]

    if e_arr is not None:
        e_arr = e_arr[sort_order]

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
    for ii, tl_i, tr_i, xl_i, xu_i, c_i in zip(
        unique_i, tl_by_i, tr_by_i, xl_by_i, xu_by_i, censoring_by_i
    ):
        if not (np.all(tl_i == tl_i[0]) and np.all(tr_i == tr_i[0])):
            raise ValueError(
                f"Item {ii} has inconsistent truncation bounds; each item "
                "must have a single observation window."
            )
        if tl_i[0] > tr_i[0]:
            raise ValueError(f"Item {ii} has left truncation beyond right")
        # An end-of-observation (c=1) row and a finite right truncation both
        # say where the item's window closes, so they must agree: a tr past
        # the c=1 row claims the item was watched (with no events) after its
        # observation ended. Models used to resolve this differently (the
        # cause-specific NHPP closed at the row, the others at tr).
        if np.isfinite(tr_i[0]) and c_i[-1] == 1 and xu_i[-1] < tr_i[0]:
            raise ValueError(
                f"Item {ii} has an end-of-observation (c=1) row at "
                f"{xu_i[-1]} before its right truncation time tr="
                f"{tr_i[0]}; both close the observation window, so they "
                "must agree (drop the c=1 row or set tr to its time)."
            )
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

    # Covariates describe the item, not the row: the proportional-intensity
    # likelihood, its tr window close and its diagnostics would otherwise
    # disagree about which row's values apply (the close and diagnostics
    # use the first row), so values that change within an item are
    # rejected rather than half used.
    if Z_arr is not None:
        for ii, Z_i in zip(unique_i, np.split(Z_arr, idx)[1:]):
            if not np.all(Z_i == Z_i[0]):
                raise ValueError(
                    f"Item {ii} has covariates Z that change between its "
                    "rows; covariates are per item (static) and must be "
                    "the same on every row of an item."
                )

    if as_recurrent_data:
        data = RecurrentEventData(x, i, c, n, e=e_arr, tl=tl_arr, tr=tr_arr)
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
