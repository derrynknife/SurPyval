"""The warning that a cause has few events to fit it with (principle 22).

Ten events per estimated quantity is the usual rule for a regression's
coefficients (Peduzzi, Concato, Kemper, Holford and Feinstein, 1996, J.
Clin. Epidemiol. 49), and the threshold here: below it the estimates,
their p-values and their bounds rest on so few events that they are
unstable, and Wald intervals undercover.
"""

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.warnings import caller_stacklevel

#: Fewer events than this, per estimated coefficient for a regression and
#: per cause for a distribution fitted to each cause, are warned of.
FEW_EVENTS = 10


def event_counts(
    e: npt.NDArray, observed: npt.NDArray, n: npt.NDArray, causes: list
) -> dict[Any, int]:
    """The events of each cause: the counts ``n`` of the observed rows
    whose label is the cause."""
    from surpyval.univariate.competing_risks.labels import label_mask

    return {
        k: int(np.sum(np.asarray(n)[label_mask(e, k) & observed]))
        for k in causes
    }


def warn_few_events(counts: dict[Any, int], advice: str) -> None:
    """One warning naming each cause with fewer than :data:`FEW_EVENTS`
    events, with its count, and ``advice``: what to do about it."""
    few = {k: v for k, v in counts.items() if v < FEW_EVENTS}
    if not few:
        return
    from surpyval.serialisation import to_native

    listed = ", ".join(f"{to_native(k)!r}: {v}" for k, v in few.items())
    which = "cause has" if len(few) == 1 else "causes have"
    warnings.warn(
        f"{len(few)} {which} fewer than {FEW_EVENTS} events ({listed}): "
        f"{advice}",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )
