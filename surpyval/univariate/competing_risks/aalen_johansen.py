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
    """
    S = np.asarray(S, dtype=float)
    S_prev = np.concatenate([[1.0], S[:-1]])
    return np.asarray(hazard_increments) * S_prev
