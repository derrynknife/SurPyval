"""
Time-varying-covariate (start-stop) data handling for Cox proportional
hazards.

Time-varying covariates are represented in the *counting-process* (start-stop)
long format: each subject contributes one row per interval ``(start, stop]`` on
which its covariate vector is constant, and only the interval that ends at the
subject's event carries ``event = 1``. Because each interval is exactly a
left-truncated (delayed-entry) observation entering at ``start`` and leaving
at ``stop``, the Cox partial likelihood fits this format directly once it is
mapped to ``x = stop``, ``tl = start`` and ``c = 1 - event`` -- the
episode-splitting identity guarantees a subject split into contiguous
intervals with the same covariates gives the same fit as the unsplit subject.

``handle_tvc`` validates the start-stop structure and performs that mapping.
"""

import numpy as np


def handle_tvc(i, xl, xr, c, Z, n=None):
    """
    Validate start-stop time-varying-covariate data and map it to the arrays
    the Cox partial likelihood fits.

    Parameters
    ----------
    i : array_like
        Subject identifier for each interval row. Rows sharing an identifier
        are the successive observation intervals of one subject.
    xl, xr : array_like
        The open-closed observation interval ``(xl, xr]`` each row covers.
        ``xl`` is the delayed-entry (left-truncation) time; ``xr`` is the exit
        time. (These follow surpyval's ``xl``/``xr`` interval naming.)
    c : array_like
        Censoring flag at ``xr``, in surpyval's convention: ``0`` if the
        subject's terminal event occurs at ``xr`` on this interval, ``1`` if
        the interval end is right-censoring (administrative end / covariate
        change). A subject has at most one ``c = 0`` (event) row, and it must
        be its last interval. Note this is the *inverse* of the previous
        ``event`` flag.
    Z : array_like
        Covariate matrix, one row per interval, constant on that interval.
    n : array_like, optional
        Count weight per row. Defaults to 1.

    Returns
    -------
    x, c, n, tl, Z, ident : ndarray
        ``x = xr``, ``c`` (0 event, 1 right-censored), ``tl = xl``, and the
        (sorted-within-subject) covariate rows and identifiers, ready for
        ``CoxPH.fit(x=x, Z=Z, c=c, n=n, tl=tl)``.
    """
    ident = np.asarray(i)
    xl = np.asarray(xl, dtype=float)
    xr = np.asarray(xr, dtype=float)
    c = np.asarray(c)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = np.ones(ident.shape[0]) if n is None else np.asarray(n, dtype=float)

    nrow = ident.shape[0]
    for name, arr in (
        ("xl", xl),
        ("xr", xr),
        ("c", c),
        ("n", n),
    ):
        if arr.shape[0] != nrow:
            raise ValueError(
                "i, xl, xr, c, Z and n must have the same length; {} has {} "
                "rows, expected {}".format(name, arr.shape[0], nrow)
            )
    if Z.shape[0] != nrow:
        raise ValueError(
            "Z must have one row per interval (same length as i); got "
            "{} rows, expected {}".format(Z.shape[0], nrow)
        )
    if nrow == 0:
        raise ValueError("no interval rows were supplied")
    if not (np.isfinite(xl).all() and np.isfinite(xr).all()):
        raise ValueError("xl and xr must be finite")
    if np.any(xl >= xr):
        raise ValueError(
            "every interval must have xl < xr (an interval covers the "
            "open-closed window (xl, xr])"
        )
    if not np.isin(c, [0, 1]).all():
        raise ValueError("c must be 0 (event) or 1 (right-censored)")
    if not np.isfinite(Z).all():
        raise ValueError("Z must contain only finite values")

    # Reassemble in subject-contiguous, entry-sorted order so the returned
    # arrays are tidy and the per-subject checks are simple.
    order = np.lexsort((xl, ident))
    ident, xl, xr, c, n = (
        ident[order],
        xl[order],
        xr[order],
        c[order],
        n[order],
    )
    Z = Z[order]

    uniq, first, counts = np.unique(
        ident, return_index=True, return_counts=True
    )
    for u, f, cnt in zip(uniq, first, counts):
        s_xl = xl[f : f + cnt]
        s_xr = xr[f : f + cnt]
        s_c = c[f : f + cnt]
        # Non-overlapping: each interval must start at or after the previous
        # interval's exit (touching intervals are contiguous, which is the
        # usual case; a gap is allowed -- the subject is simply not at risk in
        # the gap).
        if cnt > 1 and np.any(s_xl[1:] < s_xr[:-1] - 1e-12):
            raise ValueError(
                "subject {!r} has overlapping intervals; (xl, xr] rows for a "
                "subject must not overlap".format(u)
            )
        n_events = int((s_c == 0).sum())
        if n_events > 1:
            raise ValueError(
                "subject {!r} has more than one event (c=0) interval; a "
                "subject may experience at most one terminal event".format(u)
            )
        if n_events == 1 and s_c[-1] != 0:
            raise ValueError(
                "subject {!r} has an event (c=0) on a non-final interval; the "
                "event must terminate observation (it can only fall on the "
                "interval with the largest xr)".format(u)
            )

    x = xr
    c = c.astype(int)
    tl = xl
    return x, c, n, tl, Z, ident


def handle_tvc_timeline(i, x, Z, c, n=None):
    """
    Convert a per-subject covariate *timeline* into the start-stop intervals
    that :func:`handle_tvc` (and ``CoxPH.fit_tvc``) consume.

    This is the timeline / ``xicnt``-style alternative to writing the
    ``(start, stop]`` intervals by hand: instead of one row per interval, give
    each subject's covariate history as a sequence of change-points in time,
    with the subject's terminal event / censoring marked on its last row.

    Parameters
    ----------
    i : array_like
        Subject identifier for each timeline row (the ``xicnt`` item id). Rows
        sharing an identifier are one subject's covariate history.
    x : array_like
        The time each row's covariate value *takes effect*. Within a subject
        the times must be strictly increasing; the subject's **first** time is
        its entry (delayed-entry / left-truncation) time and its **last** time
        is the event / censoring time.
    Z : array_like
        The covariate vector effective from this row's ``x`` until the
        subject's next row. The value on a subject's **last** (terminal) row is
        ignored -- that row only marks the exit time and status -- so it may be
        a repeat or a placeholder.
    c : array_like
        Censoring status. Only the value on each subject's **last** row is
        read: ``0`` for an event at that time, ``1`` for right-censored. The
        ``c`` on earlier (covariate-change) rows is ignored.
    n : array_like, optional
        Count weight per subject, read from the subject's terminal row.
        Defaults to 1.

    Returns
    -------
    i, xl, xr, c, Z, n : ndarray
        Start-stop interval arrays, ready for ``CoxPH.fit_tvc`` /
        :func:`handle_tvc`. A subject with ``m`` timeline rows yields ``m - 1``
        contiguous intervals ``(xl_k, xr_k]`` carrying covariate ``Z_k``, and
        the terminal status (``c = 0`` event) falls on the last interval.

    Notes
    -----
    A subject needs at least two rows (an entry row and a terminal row) to
    define one interval. Because times are strictly increasing within a
    subject, a covariate change and the terminal event cannot share an instant;
    put the change on the interval that ends at the event.
    """
    ident = np.asarray(i)
    x = np.asarray(x, dtype=float)
    c = np.asarray(c)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    nrow = ident.shape[0]
    n = np.ones(nrow) if n is None else np.asarray(n, dtype=float)

    for name, arr in (("x", x), ("c", c), ("n", n)):
        if arr.shape[0] != nrow:
            raise ValueError(
                "i, x, Z, c and n must have the same length; {} has {} "
                "rows, expected {}".format(name, arr.shape[0], nrow)
            )
    if Z.shape[0] != nrow:
        raise ValueError(
            "Z must have one row per timeline entry (same length as i); "
            "got {} rows, expected {}".format(Z.shape[0], nrow)
        )
    if nrow == 0:
        raise ValueError("no timeline rows were supplied")
    if not np.isfinite(x).all():
        raise ValueError("x (timeline times) must be finite")

    # Sort into subject-contiguous, time-increasing order.
    order = np.lexsort((x, ident))
    ident, x, c, n = ident[order], x[order], c[order], n[order]
    Z = Z[order]

    ids_out, xl_out, xr_out, c_out, n_out = [], [], [], [], []
    Z_out = []
    uniq, first, counts = np.unique(
        ident, return_index=True, return_counts=True
    )
    for u, f, cnt in zip(uniq, first, counts):
        if cnt < 2:
            raise ValueError(
                "subject {!r} has a single timeline row; a subject needs at "
                "least an entry row and a terminal (event/censoring) row to "
                "define one interval".format(u)
            )
        sx = x[f : f + cnt]
        sc = c[f : f + cnt]
        sZ = Z[f : f + cnt]
        sn = n[f : f + cnt]
        if np.any(np.diff(sx) <= 0):
            raise ValueError(
                "subject {!r} has non-increasing timeline times; each "
                "subject's rows must have strictly increasing x".format(u)
            )
        terminal = sc[-1]
        if terminal not in (0, 1):
            raise ValueError(
                "subject {!r} has a terminal status c={!r}; the last row's c "
                "must be 0 (event) or 1 (right-censored)".format(u, terminal)
            )
        subject_weight = sn[-1]
        # m rows -> m-1 intervals (xl_k, xr_k] carrying Z_k; the terminal
        # status attaches to the last interval, which ends at the exit time.
        # c follows surpyval's convention (0 event, 1 right-censored).
        for k in range(cnt - 1):
            ids_out.append(u)
            xl_out.append(sx[k])
            xr_out.append(sx[k + 1])
            is_last = k == cnt - 2
            c_out.append(0 if (is_last and terminal == 0) else 1)
            Z_out.append(sZ[k])
            n_out.append(subject_weight)

    Z_arr = np.array(Z_out)
    if not np.isfinite(Z_arr).all():
        raise ValueError(
            "Z must contain only finite values on every non-terminal timeline "
            "row (each covariate segment must have a defined value)"
        )
    return (
        np.array(ids_out),
        np.array(xl_out, dtype=float),
        np.array(xr_out, dtype=float),
        np.array(c_out, dtype=int),
        Z_arr,
        np.array(n_out, dtype=float),
    )
