"""Refits from a known estimate (#522).

A caller that refits many resamples of the same data -- a bootstrap --
knows each resample's maximum is within sampling error of the full-data
estimate. :func:`warm_starts` lets a maximum-likelihood fit take an
``init`` as such a warm start, and :data:`DEGRADATION_REFIT` carries what
``DegradationAnalysis``'s bootstrap bounds reuse across their refits.
"""

import contextlib
import contextvars
from collections.abc import Iterator

# Set while a caller refits from starts it knows to be near the maximum
# (see ``warm_starts``).
_WARM_STARTS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "surpyval_warm_starts", default=False
)

#: Set by the degradation bootstrap bounds (``degradation._bounds.
#: bootstrap_cb``) while they refit resamples of a model's units:
#: ``{"units": {}, "life_init": params}``. A resampled unit is one of the
#: model's own, so ``DegradationAnalysis.fit`` keeps its path fit (and all
#: that is computed from it) by its measurements and reuses it, which is
#: the same computation; and it warm starts the life distribution from
#: the full-data estimate (``warm_starts``).
DEGRADATION_REFIT: contextvars.ContextVar["dict | None"] = (
    contextvars.ContextVar("surpyval_degradation_refit", default=None)
)


@contextlib.contextmanager
def warm_starts() -> Iterator[None]:
    """Take an ``init`` given to a maximum-likelihood fit as a warm start.

    A fit given ``init`` is also started from the default start, so that
    a poor ``init`` cannot return a worse model than none would
    (principle 13). Inside this block that second search is made only
    when the one from ``init`` has not reached a verified maximum (a zero
    gradient and a positive definite Hessian; see
    ``univariate.parametric.fitters.mle``). It is for a caller that
    refits many resamples from the full-data estimate
    (``DegradationAnalysis``'s bootstrap bounds, #522), whose starts are
    within sampling error of each resample's maximum.

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as sp
    >>> from surpyval.utils.refits import warm_starts
    >>> x = np.array([3.1, 4.7, 5.2, 6.9, 8.4, 9.0, 11.3, 12.8])
    >>> full = sp.Weibull.fit(x)
    >>> with warm_starts():
    ...     refit = sp.Weibull.fit(x[1:], init=full.params)
    >>> bool(np.allclose(refit.params, sp.Weibull.fit(x[1:]).params))
    True
    """
    token = _WARM_STARTS.set(True)
    try:
        yield
    finally:
        _WARM_STARTS.reset(token)


def warm_starts_on() -> bool:
    """Whether :func:`warm_starts` is in force.

    Examples
    --------
    >>> from surpyval.utils.refits import warm_starts, warm_starts_on
    >>> with warm_starts():
    ...     warm_starts_on()
    True
    >>> warm_starts_on()
    False
    """
    return _WARM_STARTS.get()
