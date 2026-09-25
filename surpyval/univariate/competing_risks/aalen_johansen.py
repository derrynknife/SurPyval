"""Shared Aalen-Johansen incidence increment (#299).

The S(t-) weighting that turns cause-specific hazard increments into
cumulative-incidence increments was implemented three times (the
nonparametric CIF, the CR-PH ``cif`` and Gray's pooled CIF), and one of
those copies went wrong (#278). It now lives here once.
"""

import numpy as np
import numpy.typing as npt


def aalen_johansen_iif(
    S: npt.NDArray, hazard_increments: npt.NDArray
) -> npt.NDArray:
    """Instantaneous incidence increments.

    The cause-specific hazard increment at ``t_i`` acts on the
    population still alive just *before* ``t_i``, so each increment is
    weighted by ``S(t_i-)`` — the survival after the previous event
    time — not ``S(t_i)`` (#253). ``S`` and the last axis of
    ``hazard_increments`` must be aligned on the same event-time grid;
    the cumulative incidence is the cumulative sum of the result.

    Parameters
    ----------
    S : array_like
        The all-cause (Kaplan-Meier) survival at each event time.
    hazard_increments : array_like
        The cause-specific hazard increments ``d_j / r`` at the same
        times, one row per cause (or a single 1-D row).

    Returns
    -------
    numpy array
        ``hazard_increments * S(t-)``, the same shape as
        ``hazard_increments``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks.aalen_johansen import (
    ...     aalen_johansen_iif,
    ... )
    >>> S = np.array([0.9, 0.8, 0.6])
    >>> aalen_johansen_iif(S, np.array([0.1, 0.0, 0.25]))
    array([0.1, 0. , 0.2])
    """
    S = np.asarray(S, dtype=float)
    S_prev = np.concatenate([[1.0], S[:-1]])
    return np.asarray(hazard_increments) * S_prev
