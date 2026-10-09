from __future__ import annotations

from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from surpyval.utils import coerce_xcnt_x, format_truncation

from .recurrent_event_data import RecurrentEventData


def item_label(item: object) -> str:
    """
    An item (unit, system) id as plain text for a message: the user's own
    label, without numpy's repr (``np.str_('pumpB')``, ``np.int64(1)``),
    and a whole-number float id (the default single item is ``1.0``)
    written as the integer.

    Examples
    --------
    >>> import numpy as np
    >>> item_label(np.str_("pumpB")), item_label(np.float64(1.0))
    ('pumpB', '1')
    """
    if isinstance(item, np.generic):
        item = item.item()
    if isinstance(item, float) and item.is_integer():
        return str(int(item))
    return str(item)


def number_text(value: object) -> str:
    """A time as plain text (``80.0`` rather than ``np.float64(80.0)``)."""
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _warn_negative_entry(
    data: RecurrentEventData, entry: npt.NDArray, model_name: str
) -> None:
    """
    Warn of a negative entry age ``tl`` on a renewal fit (#664). These
    models count each item's age from new, so a negative entry is almost
    always a data error (a calendar time, or an offset scale); it is
    still accepted (and taken as given, the item as new at entry), as the
    changelog for #615 says.
    """
    import warnings

    from surpyval.utils.warnings import caller_stacklevel

    first, _ = data.item_rows()
    negative = [
        "{} (tl={})".format(item_label(data.i[k]), number_text(entry[k]))
        for k in first
        if entry[k] < 0
    ]
    warnings.warn(
        "{} takes tl as each item's age at entry, but {} item(s) enter at "
        "a negative age: {}. On an age scale a negative entry is almost "
        "always a data error (calendar times, or another origin); the fit "
        "uses it as given, each item as new at its entry.".format(
            model_name,
            len(negative),
            ", ".join(negative[:5]) + (", ..." if len(negative) > 5 else ""),
        ),
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


def close_at_right_truncation(data: RecurrentEventData) -> RecurrentEventData:
    """
    The data with each item's finite right-truncation time ``tr`` written
    as the end-of-observation (``c=1``) row it stands for (#624).

    A finite ``tr`` closes an item's observation window, as the NHPP
    likelihoods and the MCF take it: the item was watched, with no further
    events, up to ``tr``. The imperfect-repair likelihoods read the window
    close from a ``c=1`` row only, so an item whose last row is before its
    ``tr`` gets a ``c=1`` row at ``tr``. An item already closed at its
    ``tr`` (a ``c=1`` row, or an event, there) is left as it is.

    Raises ``ValueError`` for an item whose rows go past its ``tr``, or
    whose ``c=1`` row is before it (``handle_xicn`` refuses both; this is
    for data assembled by hand).
    """
    tr = np.asarray(data.tr, dtype=float)
    if not np.isfinite(tr).any():
        return data
    x = np.asarray(data.x, dtype=float)
    x_upper = x if x.ndim == 1 else x[:, 1]
    c = np.asarray(data.c)
    at: list[int] = []
    for item in data.items:
        rows = np.flatnonzero(data.i == item)
        tr_item = tr[rows[0]]
        if not np.isfinite(tr_item):
            continue
        last = rows[np.argmax(x_upper[rows])]
        if x_upper[last] > tr_item:
            raise ValueError(
                "Item {} has a row at {} after its right truncation time "
                "tr={}; tr is the end of the item's observation, so every "
                "row must be at or before it.".format(
                    item_label(item),
                    number_text(x_upper[last]),
                    number_text(tr_item),
                )
            )
        if x_upper[last] == tr_item:
            continue
        if c[last] == 1:
            raise ValueError(
                "Item {} has an end-of-observation (c=1) row at {} before "
                "its right truncation time tr={}; both close the "
                "observation window, so they must agree (drop the c=1 row "
                "or set tr to its time).".format(
                    item_label(item),
                    number_text(x_upper[last]),
                    number_text(tr_item),
                )
            )
        at.append(int(rows[-1]) + 1)
    if not at:
        return data
    values = tr[np.asarray(at) - 1]
    new_x = (
        np.insert(x, at, values)
        if x.ndim == 1
        else np.insert(x, at, np.column_stack([values, values]), axis=0)
    )
    closed = RecurrentEventData(
        new_x,
        np.insert(data.i, at, data.i[np.asarray(at) - 1]),
        np.insert(c, at, 1),
        np.insert(data.n, at, 1),
        e=(
            None
            if data.e is None
            else np.insert(
                data.e.astype(object), at, np.full(len(at), None, object)
            )
        ),
        tl=np.insert(data.tl, at, data.tl[np.asarray(at) - 1]),
        tr=np.insert(tr, at, values),
    )
    Z = getattr(data, "Z", None)
    closed.Z = (
        None if Z is None else np.insert(Z, at, Z[np.asarray(at) - 1], axis=0)
    )
    return closed


def measure_from_entry(
    data: RecurrentEventData, model_name: str
) -> RecurrentEventData:
    """
    The data on the clock of a virtual-age / imperfect-repair model
    (Kijima, G1, ARA, ARI): each item's times measured from its entry,
    and each item's finite right-truncation time ``tr`` as its ``c=1``
    close (see :func:`close_at_right_truncation`).

    These models need an item's state when its observation begins: its
    virtual age, or the intensity reductions of its earlier repairs. With
    delayed entry (a left-truncation time ``tl``) that history is unknown,
    and the models take the item to be **as new at entry**: virtual age 0
    at ``tl``, as after an overhaul, with its clock restarting there. So an
    item's times (and its right-truncation time) are moved back by its
    ``tl``; an item with no ``tl`` is measured from 0, as before. Under
    this assumption the fit is the one of the items' histories since entry
    as if each had started new then (#615).

    Raises ``ValueError`` for a negative time without a ``tl`` (these
    models measure time from the start of the item's life).
    """
    data = close_at_right_truncation(data)
    tl = np.asarray(data.tl, dtype=float)
    entry = np.where(np.isfinite(tl), tl, 0.0)
    x = np.asarray(data.x, dtype=float)
    if np.any(entry < 0):
        _warn_negative_entry(data, entry, model_name)
    if np.any(entry != 0):
        tr = np.asarray(data.tr, dtype=float)
        shifted = RecurrentEventData(
            x - entry,
            data.i,
            data.c,
            data.n,
            e=data.e,
            tl=np.where(np.isfinite(tl), 0.0, tl),
            tr=tr - entry,
        )
        shifted.Z = data.Z
        return shifted
    # These models measure time from the start of each item's life (the
    # first gap starts at 0), so a negative time would give a negative gap.
    if np.any(x < 0):
        raise ValueError(
            "{} measures times from the start of each item's life, so they "
            "cannot be negative.".format(model_name)
        )
    return data


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

    n_params = len(getattr(dist, "parameter_names", ()))
    events = float(np.asarray(data.n)[c == 0].sum())
    window_closed = np.any(c != 0) or np.any(np.isfinite(tr))
    if n_params >= 2 and events == 1 and not window_closed:
        raise ValueError(
            "The data has a single event and no observation beyond it "
            "(failure truncated), which cannot identify the {} parameters "
            "of the {} intensity. Add the end of the observation window "
            "(a c=1 row or tr).".format(n_params, name)
        )


_INTENSITY_MODELS = "CrowAMSAA, Duane or CoxLewis"


def _is_intensity_model(dist: object) -> bool:
    return all(
        hasattr(dist, a)
        for a in ("iif", "cif", "from_params", "fit_from_recurrent_data")
    )


def validate_intensity_model(baseline: object, fitter: str) -> None:
    """
    Refuse a ``baseline`` that is not a recurrence intensity model, for the
    fitters that reduce a baseline intensity (ARI, #495).

    ARA and the generalized renewal processes take a *lifetime
    distribution* as ``dist``, so ``sp.Weibull`` is the natural thing to
    pass here too, and without this check it would fail with an
    ``AttributeError`` from inside the fit. (ARI's argument is
    ``baseline``, #507.)
    """
    if _is_intensity_model(baseline):
        return
    name = getattr(baseline, "name", repr(baseline))
    if hasattr(baseline, "fit") and hasattr(baseline, "hf"):
        raise ValueError(
            "{f}'s `baseline` is the baseline intensity model ({m}), not a "
            "lifetime distribution; got {n}. For imperfect repair with a "
            "{n} lifetime use ARA (reduction of age) or GeneralizedRenewal, "
            "or, for ARI with a power-law intensity (a Weibull hazard), "
            "baseline=CrowAMSAA.".format(f=fitter, m=_INTENSITY_MODELS, n=name)
        )
    raise ValueError(
        "{}'s `baseline` must be a recurrence intensity model ({}); got "
        "{!r}.".format(fitter, _INTENSITY_MODELS, baseline)
    )


def validate_lifetime_dist(dist: object, fitter: str) -> None:
    """
    Refuse a ``dist`` that is not a lifetime distribution, for the fitters
    whose ``dist`` is the distribution of the times between repairs (ARA
    and the generalized renewal processes, #495). The mirror image of
    :func:`validate_intensity_model`: an intensity model passed here would
    be read as a distribution and fail with an unrelated message.
    """
    if hasattr(dist, "fit") and hasattr(dist, "hf"):
        return
    name = getattr(dist, "name", repr(dist))
    if _is_intensity_model(dist):
        raise ValueError(
            "{f}'s `dist` is a lifetime distribution (e.g. Weibull, "
            "Exponential, LogNormal, Gamma), not an intensity model; got "
            "{n}. For imperfect repair that reduces a baseline intensity "
            "use ARI (arithmetic reduction of intensity), whose `baseline` "
            "is the intensity model.".format(f=fitter, n=name)
        )
    raise ValueError(
        "{}'s `dist` must be a lifetime distribution (e.g. Weibull, "
        "Exponential, LogNormal, Gamma); got {!r}.".format(fitter, dist)
    )


def _check_renewal_identifiable(
    gaps: npt.NDArray, c: npt.NDArray, dist: object, model_name: str
) -> None:
    """
    Refuse data with fewer distinct times between failures than the
    lifetime distribution has parameters: its likelihood is flat there.
    The fit used to fail inside the distribution's own fit, with advice
    in single-sample terms to "fix a parameter with `fixed=`", which the
    renewal fits do not take (#663).
    """
    k = getattr(dist, "k", None)
    if not isinstance(k, (int, np.integer)):
        return
    distinct = np.unique(gaps[c == 0]).size
    if distinct < k:
        name = getattr(dist, "name", "lifetime")
        raise ValueError(
            "{m} fits the {k} parameters of the {d} times between "
            "failures, with its repair parameter, but the data has only {n} "
            "distinct time(s) between failures, so no unique fit exists. "
            "Give more failures (more items, or a longer observation of "
            "each), or a lifetime distribution with fewer parameters "
            "(dist=Exponential).".format(m=model_name, k=k, d=name, n=distinct)
        )


def validate_renewal_times(
    data: RecurrentEventData,
    dist: object,
    model_name: str,
    every_gap_from_new: bool = False,
) -> None:
    """
    Check the event times a lifetime-distribution renewal model can use.

    (Negative times are rejected by ``measure_from_entry``.) For a
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
    c = np.asarray(data.c)
    gaps = data.get_interarrival_times()
    _check_renewal_identifiable(gaps, c, dist, model_name)
    support = getattr(dist, "support", (0.0, np.inf))
    if support[0] < 0:
        return
    _, first = np.unique(data.i, return_index=True)
    is_first = np.zeros(len(x), dtype=bool)
    is_first[first] = True
    exact_zero = (gaps == 0) & (c == 0)
    if np.any(exact_zero & is_first):
        # (An event at a delayed entry tl, which would be at age 0 here
        # too, is refused before this by handle_xicn, naming the tl.)
        k = int(np.flatnonzero(exact_zero & is_first)[0])
        raise ValueError(
            "{} has an event at time 0 (item {}): the gap from a new item's "
            "age 0 is then zero, where the {} density is 0 or infinite, so "
            "the likelihood has no maximum. Record events at positive "
            "times.".format(
                model_name,
                item_label(data.i[k]),
                getattr(dist, "name", "lifetime"),
            )
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
            raise ValueError(
                "item {} has no observation windows".format(item_label(ii))
            )
        for w in wins:
            if len(w) != 2:
                raise ValueError(
                    "item {} has a malformed window {!r}; each window must be "
                    "a (start, end) pair".format(item_label(ii), w)
                )
            a, b = float(w[0]), float(w[1])
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(
                    "item {} has a non-finite observation window "
                    "({}, {})".format(item_label(ii), w[0], w[1])
                )
            if not (a < b):
                raise ValueError(
                    "item {} has an empty or reversed observation window "
                    "({}, {}); require start < end".format(
                        item_label(ii), w[0], w[1]
                    )
                )
        # sort windows by start and require they are disjoint (touching, i.e.
        # end == next start, is allowed)
        wins = sorted(wins, key=lambda w: float(w[0]))
        for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
            if float(a2) < float(b1):
                raise ValueError(
                    "item {} has overlapping observation windows "
                    "({}, {}) and ({}, {})".format(
                        item_label(ii),
                        *(number_text(v) for v in (a1, b1, a2, b2)),
                    )
                )

        mask = np.asarray(i) == ii
        item_x = np.asarray(x)[mask]
        item_n = np.asarray(n)[mask]
        assigned = np.zeros(item_x.shape[0], dtype=bool)
        for w in wins:
            a, b = float(w[0]), float(w[1])
            synth += 1
            # A window (start, end] holds the events after its start, as
            # a delayed entry tl does: an event at a window's start
            # belongs to the window that ends there, if any (#658).
            in_win = (item_x > a) & (item_x <= b) & (~assigned)
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
                "{} (a window (start, end] holds the events after its "
                "start)".format(item_label(ii), outside)
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


def _xicn_defaults(
    x: npt.NDArray,
    i: npt.ArrayLike | None,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``i``, ``c`` and ``n`` as arrays: one item, observed, one event.

    A scalar applies to every row, as a scalar ``tl`` / ``tr`` does
    (``i=1``: one item; ``c=0``: every row an event).
    """
    rows = x.shape[0]

    def per_row(value: npt.ArrayLike | None, default: float) -> npt.NDArray:
        if value is None:
            return np.full(rows, default)
        arr = np.array(value)
        return np.full(rows, arr) if arr.ndim == 0 else arr

    return per_row(i, 1.0), per_row(c, 0.0), per_row(n, 1.0)


def _xicn_marks(
    e: npt.ArrayLike | None, c: npt.NDArray, x: npt.NDArray
) -> npt.NDArray | None:
    """The event-type marks (competing-risks recurrent process), or None.

    Marks are per row and aligned with ``x``; ``None``/NaN marks
    (typically the end-of-observation censoring row) are permitted. They
    are kept as an object array so string, integer or ``None`` marks all
    round-trip unchanged.
    """
    if e is None:
        return None
    from surpyval.utils import resolve_cr_censoring

    # One mark per row (a tuple mark is not split into a column), and
    # every "no attributed cause" marker (None, NaN, pandas NA) turned
    # into Python ``None`` so downstream cause bookkeeping
    # (``event_types``, cause-specific counts) sees a single missing
    # sentinel. The derived censoring flag is not used: ``c`` is set.
    e_arr, _ = resolve_cr_censoring(e, c)
    if e_arr.shape[0] != x.shape[0]:
        raise ValueError("x and e must have the same length")
    return e_arr


def _reject_window_conflicts(
    t: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    tr: npt.ArrayLike | None,
    Z: npt.ArrayLike | dict | None,
    e_arr: npt.NDArray | None,
) -> None:
    """Gapped observation (``windows``) excludes truncation, Z and marks."""
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


def _per_item_bound(
    bound: npt.ArrayLike | None, name: str, i: npt.NDArray, n_rows: int
) -> npt.ArrayLike | None:
    """A ``tl`` / ``tr`` given one value per item, as one value per row.

    The values are in the order of the sorted item ids (``np.unique(i)``,
    the order of ``RecurrentEventData.items``, and of an array ``T`` in
    the trend tests), and each row takes its item's value, wherever the
    item's rows are in ``x``. A bound with one entry per row of ``x`` is
    per row, even when that is also the number of items; a scalar, a
    2-D array or anything else passes through for ``format_truncation``
    to read (or refuse) as it always has. Any other length is refused
    here, naming both lengths that are accepted.
    """
    if bound is None or np.ndim(bound) != 1:
        return bound
    size = np.shape(bound)[0]
    if size == n_rows or i.shape[0] != n_rows:
        # Per row, as before; a mismatched ``i`` is refused by
        # ``_check_xicn_lengths``, so nothing here can be mapped.
        return bound
    _check_item_ids(i)
    try:
        items, inverse = np.unique(i, return_inverse=True)
    except TypeError:
        raise ValueError(
            "Item identifiers 'i' must be of one comparable kind (all "
            "numbers or all strings)"
        ) from None
    if size != items.shape[0]:
        raise ValueError(
            f"'{name}' must have one entry per row of 'x' ({n_rows}) or one "
            f"per item in 'i' ({items.shape[0]}, in sorted order of the "
            f"item ids); it has {size}"
        )
    return np.asarray(bound)[inverse.reshape(-1)]


def _xicn_covariates(
    Z: npt.ArrayLike | dict | None, i: npt.NDArray
) -> npt.NDArray | None:
    """Z as an (N, p) float array, one row per row of ``x``, or None."""
    if Z is None:
        return None
    if isinstance(Z, dict):
        missing = [ii for ii in np.unique(i).tolist() if ii not in Z]
        if missing:
            raise ValueError(
                "Z has no covariates for item(s) {}".format(missing)
            )
        # a scalar value is a single covariate, held as a 1-element row
        return np.array(
            [np.atleast_1d(np.asarray(Z[ii], dtype=float)) for ii in i]
        )
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        # one covariate: a value per row, not one row of values
        Z_arr = Z_arr.reshape(-1, 1)
    return Z_arr


def _check_xicn_lengths(
    x: npt.NDArray,
    i: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    Z_arr: npt.NDArray | None,
) -> None:
    """Every per-row array has one entry per row of ``x``."""
    if x.shape[0] != i.shape[0]:
        raise ValueError("x and i must have the same length")
    if x.shape[0] != c.shape[0]:
        raise ValueError("x and c must have the same length")
    if x.shape[0] != n.shape[0]:
        raise ValueError("x and n must have the same length")

    if Z_arr is not None:
        if x.shape[0] != Z_arr.shape[0]:
            raise ValueError("x and Z must have the same length")


def _check_item_ids(i: npt.NDArray) -> None:
    """Item identifiers are finite numbers or present objects."""
    if np.issubdtype(i.dtype, np.number) and not np.isfinite(i).all():
        raise ValueError("Item identifiers 'i' must be finite (no NaN or inf)")
    if i.dtype == object:
        from surpyval.utils import is_missing_event

        # A missing id cannot say which item a row belongs to, and the sort
        # would otherwise fail with a bare "'<' not supported".
        if any(is_missing_event(v) for v in i):
            raise ValueError(
                "Item identifiers 'i' must not be missing (None or NaN)"
            )


def _check_xicn_values(
    x: npt.NDArray,
    i: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl_arr: npt.NDArray,
    tr_arr: npt.NDArray,
    Z_arr: npt.NDArray | None,
) -> None:
    """Reject non-finite values and nonsensical counts and codes.

    Malformed input fails here with an informative error rather than
    flowing silently into the optimiser. NaN in ``x`` is already rejected
    by ``coerce_xcnt_x``; here the remaining degenerate values are caught.
    """
    if not np.isfinite(x).all():
        raise ValueError("Event times 'x' must be finite (no inf values)")

    _check_item_ids(i)

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

    _check_row_kinds(x, i, c, n)


def _first_row_text(
    mask: npt.NDArray, x: npt.NDArray, i: npt.NDArray, n: npt.NDArray
) -> str:
    """'item <label> has <x>' for the first row in ``mask``."""
    k = int(np.flatnonzero(mask)[0])
    xk = (
        "[{}, {}]".format(number_text(x[k, 0]), number_text(x[k, 1]))
        if x.ndim == 2
        else number_text(x[k])
    )
    return "item {} has n={} at x={}".format(
        item_label(i[k]), number_text(n[k]), xk
    )


def _check_row_kinds(
    x: npt.NDArray, i: npt.NDArray, c: npt.NDArray, n: npt.NDArray
) -> None:
    """Each row's ``x`` and ``n`` fit what its ``c`` says the row is.

    An exact event written as an interval ``[l, r]`` used to be accepted,
    and read as its left end by the intensity fits and as its midpoint by
    the MCF (#658).
    """
    # (c=2 with 1-D x is refused by the intensity likelihoods, which are
    # the ones that take interval counts: see
    # ``RecurrentEventData.split_for_nhpp_likelihood``. The renewal models
    # and the MCF refuse c=2 outright, with their own message.)
    if x.ndim == 2:
        spread = (c == 0) & (x[:, 0] != x[:, 1])
        if np.any(spread):
            k = int(np.flatnonzero(spread)[0])
            raise ValueError(
                "An exact event (c=0) is at one time, written [t, t] in "
                "2-D x, but item {} has [{}, {}]. For events counted "
                "somewhere in an interval set c=2 on that row.".format(
                    item_label(i[k]),
                    number_text(x[k, 0]),
                    number_text(x[k, 1]),
                )
            )
    many = n > 1
    if np.any(many & (c == 0)):
        raise ValueError(
            "An exact event row (c=0) stands for one event, but {}. For "
            "simultaneous events at one time repeat the row, once per "
            "event; n > 1 is for counts in an interval (c=2) or before "
            "a time (c=-1).".format(_first_row_text(many & (c == 0), x, i, n))
        )
    if np.any(many & (c == 1)):
        raise ValueError(
            "An end-of-observation row (c=1) closes one item's window, so "
            "its n must be 1, but {}.".format(
                _first_row_text(many & (c == 1), x, i, n)
            )
        )


def _xicn_sort_order(
    x: npt.NDArray, i: npt.NDArray, c: npt.NDArray
) -> npt.NDArray:
    """The row order: by item, then time, then censoring code.

    An end-of-observation (c=1) row tied with an event at the same time
    closes the window after it, so ties put it last (and a left-censored
    count, which covers the time from entry, first); whether the input is
    accepted therefore does not depend on the order tied rows are given in.
    """
    tie_order = np.where(c == 1, 3, c)
    x_key = x.mean(axis=1) if x.ndim == 2 else x  # 2D by the midpoint
    try:
        return np.lexsort((tie_order, x_key, i))
    except TypeError:
        raise ValueError(
            "Item identifiers 'i' must be of one comparable kind (all "
            "numbers or all strings)"
        ) from None


def _rows_in_order(
    order: npt.NDArray, *columns: npt.NDArray
) -> list[npt.NDArray]:
    """Each per-row array reordered by ``order``."""
    return [column[order] for column in columns]


def _check_censoring_positions(
    unique_i: npt.NDArray, censoring_by_i: list, items_given: bool = True
) -> None:
    """At most one c=1 row, last, and one c=-1 row, first, per item.

    Without ``i`` every row is one item's, so a log of several units read
    without its unit column fails here: the message says so (#658).
    """
    hint = (
        ""
        if items_given
        else (
            " No item ids were given (`i`, or `i_col` in fit_from_df), so "
            "every row is read as one item's; pass the item of each row."
        )
    )
    for ii, arr in zip(unique_i, censoring_by_i):
        label = item_label(ii)
        if 1 in arr:
            if (arr == 1).sum() > 1:
                raise ValueError(
                    f"Item {label} has more than one end-of-observation "
                    f"(right censored, c=1) row.{hint}"
                )
            if arr[-1] != 1:
                raise ValueError(
                    f"Item {label} has an end-of-observation (right "
                    f"censored, c=1) row before its last event; it must be "
                    f"the item's last row.{hint}"
                )
        if -1 in arr:
            if (arr == -1).sum() > 1:
                raise ValueError(
                    f"Item {label} has more than one left censored "
                    f"(c=-1) row.{hint}"
                )
            if arr[0] != -1:
                raise ValueError(
                    f"Item {label} has a left censored (c=-1) row that is "
                    f"not its first.{hint}"
                )


def _check_interval_overlaps(
    x: npt.NDArray, idx: npt.NDArray, unique_i: npt.NDArray
) -> None:
    """An item's interval-censored rows do not overlap (2-D ``x`` only)."""
    if x.ndim != 2:
        return
    times_by_i = np.split(x, idx)[1:]
    for ii, arr in zip(unique_i, times_by_i):
        starts = arr[1:][:, 0]
        ends = arr[:-1][:, 1]
        if (ends > starts).any():
            raise ValueError(
                f"Item {item_label(ii)} has overlapping intervals"
            )


def event_at_entry_error(
    item: object, times: npt.ArrayLike, tl: float, noun: str = "Item"
) -> ValueError:
    """
    The refusal of an event at or before an item's observation start.

    An item entering at ``tl`` is observed over ``(tl, T]``: the
    likelihood integrates its intensity from ``tl`` and the MCF counts it
    at risk only after ``tl``. An event at ``tl`` itself is outside that
    window, so every recurrent fit, the MCF, the trend tests and the
    renewal models refuse it with this one message (#658).
    """
    shown = np.unique(np.asarray(times, dtype=float))[:5]
    return ValueError(
        "{} {} has an event at {}, at or before its observation start "
        "tl={}. An item entering at tl is observed over (tl, T], so an "
        "event at tl falls outside its window: record the event after tl, "
        "or set tl earlier.".format(
            noun,
            item_label(item),
            ", ".join(number_text(t) for t in shown),
            number_text(tl),
        )
    )


def _check_item_window(
    ii: object,
    tl_i: npt.NDArray,
    tr_i: npt.NDArray,
    xl_i: npt.NDArray,
    xu_i: npt.NDArray,
    c_i: npt.NDArray,
) -> None:
    """One item's truncation bounds form one window holding its events."""
    label = item_label(ii)
    if not (np.all(tl_i == tl_i[0]) and np.all(tr_i == tr_i[0])):
        raise ValueError(
            f"Item {label} has inconsistent truncation bounds; each item "
            "must have a single observation window."
        )
    if tl_i[0] > tr_i[0]:
        raise ValueError(f"Item {label} has left truncation beyond right")
    # An end-of-observation (c=1) row and a finite right truncation both
    # say where the item's window closes, so they must agree: a tr past
    # the c=1 row claims the item was watched (with no events) after its
    # observation ended, and every model closes the window at one place.
    if np.isfinite(tr_i[0]) and c_i[-1] == 1 and xu_i[-1] < tr_i[0]:
        raise ValueError(
            f"Item {label} has an end-of-observation (c=1) row at "
            f"{number_text(xu_i[-1])} before its right truncation time tr="
            f"{number_text(tr_i[0])}; both close the observation window, "
            "so they must agree (drop the c=1 row or set tr to its time)."
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
            f"Item {label} has events outside its observation window "
            f"[{number_text(lower)}, {number_text(tr_i[0])}]"
        )
    # The window opens just after the entry: an event (any row but the
    # c=1 close) exactly at a finite tl is outside it too (see
    # ``event_at_entry_error``).
    if np.isfinite(tl_i[0]):
        at_entry = (c_i != 1) & (xu_i <= tl_i[0])
        if at_entry.any():
            raise event_at_entry_error(ii, xu_i[at_entry], tl_i[0])


def _check_observation_windows(
    x: npt.NDArray,
    tl_arr: npt.NDArray,
    tr_arr: npt.NDArray,
    idx: npt.NDArray,
    unique_i: npt.NDArray,
    censoring_by_i: list,
) -> None:
    """Truncation defines a single observation window [tl, tr] per item.

    The bounds must be constant within an item and contain all of its
    events.
    """
    tl_by_i = np.split(tl_arr, idx)[1:]
    tr_by_i = np.split(tr_arr, idx)[1:]
    x_lower = x if x.ndim == 1 else x[:, 0]
    x_upper = x if x.ndim == 1 else x[:, 1]
    xl_by_i = np.split(x_lower, idx)[1:]
    xu_by_i = np.split(x_upper, idx)[1:]
    for ii, tl_i, tr_i, xl_i, xu_i, c_i in zip(
        unique_i, tl_by_i, tr_by_i, xl_by_i, xu_by_i, censoring_by_i
    ):
        _check_item_window(ii, tl_i, tr_i, xl_i, xu_i, c_i)


def _check_static_covariates(
    Z_arr: npt.NDArray | None, idx: npt.NDArray, unique_i: npt.NDArray
) -> None:
    """Covariates describe the item, not the row.

    The proportional-intensity likelihood, its tr window close and its
    diagnostics would otherwise disagree about which row's values apply
    (the close and diagnostics use the first row), so values that change
    within an item are rejected rather than half used.
    """
    if Z_arr is None:
        return
    for ii, Z_i in zip(unique_i, np.split(Z_arr, idx)[1:]):
        if not np.all(Z_i == Z_i[0]):
            raise ValueError(
                f"Item {item_label(ii)} has covariates Z that change "
                "between its "
                "rows; covariates are per item (static) and must be "
                "the same on every row of an item."
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
        interval-censored count; in 2-D an exact event is ``[t, t]``).
    i : array like or scalar, optional
        The item each row belongs to. Defaults to one item.
    c : array like or scalar, optional
        Censoring flags: 0 an observed event, 1 the right-censored end of
        observation, -1 left-censored and 2 interval-censored counts
        (which need 2-D ``x``). Defaults to all observed. Rows are sorted
        by item and time; an end-of-observation row at the same time as
        an event goes after it, whatever the input order.
    n : array like or scalar, optional
        The number of events in each row. Defaults to 1. Only a count
        (``c=2`` or ``c=-1``) can stand for several events: repeat an
        exact event's row for simultaneous events. A scalar ``i``, ``c``
        or ``n`` applies to every row.
    t : array like, optional
        (N, 2) truncation bounds per row. Use ``tl`` / ``tr`` for per-item
        bounds instead.
    tl, tr : array like or scalar, optional
        Left-truncation (start of observation) and right-truncation (end
        of observation) times: a scalar for every item, one value per row
        (the same on every row of an item) or, with ``i``, one value per
        item, in the sorted order of the item ids (``np.unique(i)``, the
        order of ``RecurrentEventData.items``), whichever rows the item
        has in ``x``. A per-item bound is expanded to one value per row
        here, so the data built are those of the per-row form. When there
        are as many rows as items the bound is read per row; any other
        length is refused. An item with a finite ``tl`` is observed
        over ``(tl, T]``, so an event at ``tl`` itself is refused. An item
        with both a ``c=1`` row and a finite ``tr`` must have them at the
        same time.
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

    items_given = i is not None
    i, c, n = _xicn_defaults(x, i, c, n)
    e_arr = _xicn_marks(e, c, x)

    # Gapped (multi-window) observation: each item is observed over several
    # disjoint windows with unobserved gaps between them. Expand each window
    # into a synthetic single-window sub-item so the rest of the handler --
    # and the NHPP likelihood and MCF at-risk set downstream -- treat the gaps
    # correctly without any special-casing (see ``_expand_windows``).
    window_map: dict | None = None
    if windows is not None:
        _reject_window_conflicts(t, tl, tr, Z, e_arr)
        x, i, c, n, tl_arr, tr_arr, window_map = _expand_windows(
            x, i, c, n, windows
        )
    else:
        # Truncation follows surpyval's xcnt convention (shared with the
        # univariate handler): the default window is the whole real line.
        # No global sign assumption is made about ``x`` here -- each item
        # is instead validated against its own observation window below.
        # An item with an explicit (possibly negative) left-truncation
        # bound legitimately admits negative event times; an untruncated
        # item is integrated from the fallback origin 0 (see
        # ``get_previous_x``) and so must have non-negative event times.
        #
        # ``tl`` / ``tr`` may be given one value per item (sorted ids)
        # instead of one per row: they are expanded to one per row here,
        # before anything else reads them, so every fitter, the stored
        # data and its serialisation see the per-row arrays. Not with
        # ``t``, which ``format_truncation`` refuses alongside them.
        if items_given and t is None:
            tl = _per_item_bound(tl, "tl", i, x.shape[0])
            tr = _per_item_bound(tr, "tr", i, x.shape[0])
        truncation = format_truncation(t, tl, tr, x.shape[0])
        tl_arr = truncation[:, 0]
        tr_arr = truncation[:, 1]

    Z_arr = _xicn_covariates(Z, i)
    _check_xicn_lengths(x, i, c, n, Z_arr)
    _check_xicn_values(x, i, c, n, tl_arr, tr_arr, Z_arr)

    sort_order = _xicn_sort_order(x, i, c)
    x, i, c, n, tl_arr, tr_arr = _rows_in_order(
        sort_order, x, i, c, n, tl_arr, tr_arr
    )

    if Z_arr is not None:
        Z_arr = Z_arr[sort_order]

    if e_arr is not None:
        e_arr = e_arr[sort_order]

    unique_i, idx = np.unique(i, return_index=True)
    censoring_by_i = np.split(c, idx)[1:]

    _check_censoring_positions(unique_i, censoring_by_i, items_given)
    _check_interval_overlaps(x, idx, unique_i)
    _check_observation_windows(
        x, tl_arr, tr_arr, idx, unique_i, censoring_by_i
    )
    _check_static_covariates(Z_arr, idx, unique_i)

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
