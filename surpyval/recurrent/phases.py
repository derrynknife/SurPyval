"""Counts by phase as the data of an intensity model."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

#: The fewest phases a two-parameter curve can be fitted to: one more
#: than its parameters.
MIN_PHASES = 3


def phases_to_data(
    counts: ArrayLike, lengths: ArrayLike | None = None
) -> dict:
    """
    The events counted in each of a run of phases (the defects found in
    each phase of a development, the failures in each stage of a test),
    as the data an intensity model's ``fit`` takes: one row a phase,
    ``[start, end]`` with ``c = 2`` and the phase's count as ``n``.

    Without ``lengths`` the phases are taken as equally long and time is
    the phase number: phase ``j`` is the interval ``[j - 1, j]``. With
    them, each phase is as long as its length, in order, from 0: time is
    then in the lengths' unit (weeks, say).

    Parameters
    ----------
    counts : array_like
        The count of each phase, in order: whole numbers of at least 0,
        for at least three phases (a curve of two parameters needs one
        more phase than it has parameters). A phase that found nothing
        is a 0, not a gap.
    lengths : array_like, optional
        The length of each phase, each above 0.

    Returns
    -------
    dict
        ``{"x": ..., "c": ..., "n": ...}``, to pass as
        ``RayleighDefects.fit(**phases_to_data(counts))`` (or to any
        other intensity model's ``fit``).

    Examples
    --------
    >>> from surpyval.recurrent import phases_to_data
    >>> data = phases_to_data([18, 44, 52])
    >>> data["x"].tolist()
    [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]
    >>> data["c"].tolist(), data["n"].tolist()
    ([2, 2, 2], [18, 44, 52])

    Phases of 6, 10 and 4 weeks:

    >>> phases_to_data([18, 44, 52], lengths=[6, 10, 4])["x"].tolist()
    [[0.0, 6.0], [6.0, 16.0], [16.0, 20.0]]
    """
    n = _whole_counts(counts)
    if n.size < MIN_PHASES:
        raise ValueError(
            "counts has {} phase(s); a curve needs at least {} to be "
            "fitted (one more than its two parameters). Give the count "
            "of every phase done so far, or wait for more "
            "phases.".format(n.size, MIN_PHASES)
        )
    if lengths is None:
        ends = np.arange(1.0, n.size + 1.0)
    else:
        spans = np.asarray(lengths, dtype=float)
        if spans.shape != n.shape:
            raise ValueError(
                "lengths must have one length for each of the {} phases "
                "in counts; got {}".format(n.size, spans.size)
            )
        if not np.all(np.isfinite(spans) & (spans > 0)):
            raise ValueError(
                "lengths must each be above 0 (a phase takes some time); "
                "got {}".format(spans[~(np.isfinite(spans) & (spans > 0))][:5])
            )
        ends = np.cumsum(spans)
    starts = np.concatenate([[0.0], ends[:-1]])
    return {
        "x": np.column_stack([starts, ends]),
        "c": np.full(n.size, 2),
        "n": n,
    }


def _whole_counts(counts: Any) -> np.ndarray:
    try:
        values = np.asarray(counts, dtype=float)
    except (TypeError, ValueError):
        values = np.array([np.nan])
    if values.ndim != 1:
        raise ValueError(
            "counts must be a flat list of the count of each phase; got "
            "shape {}".format(values.shape)
        )
    ok = np.isfinite(values) & (values >= 0) & (values == np.floor(values))
    if not np.all(ok):
        raise ValueError(
            "counts must be whole numbers of at least 0, one a phase; "
            "got {}".format(values[~ok][:5])
        )
    return values.astype(int)
