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


def handle_tvc(ident, start, stop, event, Z, n=None):
    """
    Validate start-stop time-varying-covariate data and map it to the arrays
    the Cox partial likelihood fits.

    Parameters
    ----------
    ident : array_like
        Subject identifier for each interval row. Rows sharing an identifier
        are the successive observation intervals of one subject.
    start, stop : array_like
        The open-closed interval ``(start, stop]`` each row is observed over.
        ``start`` is the delayed-entry (left-truncation) time.
    event : array_like
        1 if the subject's terminal event occurs at ``stop`` on this interval,
        0 otherwise (interval end is an administrative censoring / covariate
        change). A subject has at most one ``event = 1`` row, and it must be
        its last interval.
    Z : array_like
        Covariate matrix, one row per interval, constant on that interval.
    n : array_like, optional
        Count weight per row. Defaults to 1.

    Returns
    -------
    x, c, n, tl, Z, ident : ndarray
        ``x = stop``, ``c = 1 - event`` (0 event, 1 right-censored),
        ``tl = start``, and the (sorted-within-subject) covariate rows and
        identifiers, ready for ``CoxPH.fit(x=x, Z=Z, c=c, n=n, tl=tl)``.
    """
    ident = np.asarray(ident)
    start = np.asarray(start, dtype=float)
    stop = np.asarray(stop, dtype=float)
    event = np.asarray(event)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = np.ones(ident.shape[0]) if n is None else np.asarray(n, dtype=float)

    nrow = ident.shape[0]
    for name, arr in (
        ("start", start),
        ("stop", stop),
        ("event", event),
        ("n", n),
    ):
        if arr.shape[0] != nrow:
            raise ValueError(
                "ident, start, stop, event, Z and n must have the same "
                "length; {} has {} rows, expected {}".format(
                    name, arr.shape[0], nrow
                )
            )
    if Z.shape[0] != nrow:
        raise ValueError(
            "Z must have one row per interval (same length as ident); got "
            "{} rows, expected {}".format(Z.shape[0], nrow)
        )
    if nrow == 0:
        raise ValueError("no interval rows were supplied")
    if not (np.isfinite(start).all() and np.isfinite(stop).all()):
        raise ValueError("start and stop must be finite")
    if np.any(start >= stop):
        raise ValueError(
            "every interval must have start < stop (an interval covers the "
            "open-closed window (start, stop])"
        )
    if not np.isin(event, [0, 1]).all():
        raise ValueError("event must be 0 or 1")
    if not np.isfinite(Z).all():
        raise ValueError("Z must contain only finite values")

    # Reassemble in subject-contiguous, start-sorted order so the returned
    # arrays are tidy and the per-subject checks are simple.
    order = np.lexsort((start, ident))
    ident, start, stop, event, n = (
        ident[order],
        start[order],
        stop[order],
        event[order],
        n[order],
    )
    Z = Z[order]

    uniq, first, counts = np.unique(
        ident, return_index=True, return_counts=True
    )
    for u, f, cnt in zip(uniq, first, counts):
        s_start = start[f : f + cnt]
        s_stop = stop[f : f + cnt]
        s_event = event[f : f + cnt]
        # Non-overlapping: each interval must start at or after the previous
        # interval's stop (touching intervals are contiguous, which is the
        # usual case; a gap is allowed -- the subject is simply not at risk in
        # the gap).
        if cnt > 1 and np.any(s_start[1:] < s_stop[:-1] - 1e-12):
            raise ValueError(
                "subject {!r} has overlapping intervals; start-stop rows for "
                "a subject must not overlap".format(u)
            )
        n_events = int((s_event == 1).sum())
        if n_events > 1:
            raise ValueError(
                "subject {!r} has more than one event=1 interval; a subject "
                "may experience at most one terminal event".format(u)
            )
        if n_events == 1 and s_event[-1] != 1:
            raise ValueError(
                "subject {!r} has event=1 on a non-final interval; the event "
                "must terminate observation (it can only fall on the interval "
                "with the largest stop time)".format(u)
            )

    x = stop
    c = np.where(event == 1, 0, 1).astype(int)
    tl = start
    return x, c, n, tl, Z, ident
