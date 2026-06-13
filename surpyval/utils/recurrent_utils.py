import numpy as np

from .recurrent_event_data import RecurrentEventData

# Number of trailing interarrival times used to estimate the geometric decay
# ratio in ``interarrivals_converge_below``.
DECAY_WINDOW = 8

# Safety factor applied to the extrapolated geometric tail. The sequence is
# only declared non-convergent when the gap to the target is at least this many
# times the extrapolated reachable distance, so a slowly-decaying sequence that
# is in fact about to cross the target is not terminated prematurely.
DECAY_MARGIN = 3.0


def interarrivals_converge_below(
    increments, running, target, window=DECAY_WINDOW, margin=DECAY_MARGIN
):
    """
    Geometric-tail reachability test for time-terminated simulation.

    A time-terminated sequence must either cross ``target`` or have its running
    total converge to a limit below ``target``. This detects the latter: it
    returns ``True`` only when the recent interarrival ``increments`` are
    decaying geometrically and ``margin`` times the extrapolated geometric tail
    still cannot bridge the remaining gap to ``target``.

    The decay ratio is estimated over a smoothing ``window`` and the test bails
    out whenever the increments are not clearly shrinking, so a burst of
    small-but-stable interarrival times (a high-intensity region well below
    ``target``) is never mistaken for convergence. The ``margin`` factor guards
    against the opposite error: terminating a slowly-decaying sequence that is
    actually a step or two away from crossing ``target``.

    Parameters
    ----------
    increments : sequence of float
        Interarrival times generated so far for the current sequence.
    running : float
        Current cumulative time (the sum of ``increments``).
    target : float
        The time-termination value ``T``.
    window : int, optional
        Number of trailing increments used to estimate the decay ratio.
    margin : float, optional
        Safety factor applied to the extrapolated geometric tail before
        comparing against the gap to ``target``.

    Returns
    -------
    bool
        ``True`` if the sequence cannot reach ``target`` under continued
        geometric decay (with the safety margin applied), otherwise ``False``.
    """
    # Need two full blocks to estimate a block-averaged decay ratio. Requiring
    # 2 * window events also means short sequences that genuinely reach the
    # target do so (via running > target) before this test can ever fire.
    if len(increments) < 2 * window:
        return False

    older = increments[-2 * window : -window]
    recent = increments[-window:]
    older_mean = sum(older) / window
    recent_mean = sum(recent) / window
    if older_mean <= 0.0 or recent_mean <= 0.0:
        return False

    # Block-averaged per-step decay ratio. Averaging over blocks smooths out
    # the heavy noise in individual interarrival times, so a noisy-but-stable
    # sequence is not mistaken for a decaying one.
    if recent_mean >= older_mean:
        # Not decaying; the remaining gap to target may still be closed.
        return False
    rho = (recent_mean / older_mean) ** (1.0 / window)

    # Supremum of the additional distance reachable if decay continues at rho,
    # extrapolated from the recent average increment level.
    max_additional = recent_mean * rho / (1.0 - rho)
    return (running + margin * max_additional) < target


def validate_renewal_censoring(c, model_name):
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


def handle_xicn(x, i=None, c=None, n=None, Z=None, as_recurrent_data=True):
    if isinstance(x, list):
        if any(isinstance(v, list) for v in x):
            x_ndarray = np.empty(shape=(len(x), 2))
            for idx, val in enumerate(x):
                x_ndarray[idx, :] = np.array(val)
            x = x_ndarray
            if (x[0, :] > x[1, :]).any():
                raise ValueError("x values must be monotonically increasing")
        else:
            x = np.array(x)

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

    if Z is not None:
        if isinstance(Z, dict):
            Z = np.array([Z[ii] for ii in i])
        else:
            Z = np.array(Z, ndmin=2)
    # TODO: Z as a dict where the keys are the item numbers and the arrays
    # are the covariates for each i at all times (x)

    if x.shape[0] != i.shape[0]:
        raise ValueError("x and i must have the same length")
    if x.shape[0] != c.shape[0]:
        raise ValueError("x and c must have the same length")
    if x.shape[0] != n.shape[0]:
        raise ValueError("x and n must have the same length")

    if Z is not None:
        if x.shape[0] != Z.shape[0]:
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

    if Z is not None:
        Z = Z[sort_order]

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

    if as_recurrent_data:
        data = RecurrentEventData(x, i, c, n)
        data.Z = Z
        return data
    else:
        return x, i, c, n
