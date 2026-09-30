"""The warning a fit gives when its likelihood has no finite maximum (#392).

Some data leave a model's likelihood without a finite maximum: it keeps
increasing as a parameter runs off to infinity or to the edge of its
space, and the optimiser stops wherever it gives up. The fit then warns,
with one message of one form, and still returns the model it reached, as
``CoxPH`` does for a monotone partial likelihood. Each model detects the
condition in its own way (the criterion is documented where it is
applied); this module only says so, once per fit, at the caller's line.

A caller that fits on the user's behalf and reads what the fit reached
from the model instead (``Parametric.maximum``; ``fit_best`` does) runs
the fits inside :func:`quiet_maximum_warnings`, which holds back this
warning and a maximum-likelihood fit's "did not reach a verified maximum"
warning so that the caller can say it once, in its own words.
"""

import contextlib
import contextvars
import warnings
from collections.abc import Iterator

from surpyval.utils import _caller_stacklevel

# Set while a caller that reads ``maximum`` off its models is fitting
_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "surpyval_quiet_maximum_warnings", default=False
)


@contextlib.contextmanager
def quiet_maximum_warnings() -> Iterator[None]:
    """Hold back the fits' warnings that their answer is not a verified
    maximum of the likelihood.

    Inside the block, :func:`warn_no_maximum` and a univariate
    maximum-likelihood fit's "did not reach a verified maximum" warnings
    are not given; every other warning is. The fitted models still record
    what they reached (``Parametric.maximum``), and the caller must say
    so itself.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.no_maximum import (
    ...     quiet_maximum_warnings,
    ...     warn_no_maximum,
    ... )
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     with quiet_maximum_warnings():
    ...         warn_no_maximum("theta runs away", "it means nothing", "stop")
    >>> caught
    []
    """
    token = _QUIET.set(True)
    try:
        yield
    finally:
        _QUIET.reset(token)


def maximum_warnings_quiet() -> bool:
    """Whether :func:`quiet_maximum_warnings` is holding the warnings back.

    Examples
    --------
    >>> from surpyval.utils.no_maximum import (
    ...     maximum_warnings_quiet,
    ...     quiet_maximum_warnings,
    ... )
    >>> with quiet_maximum_warnings():
    ...     maximum_warnings_quiet()
    True
    >>> maximum_warnings_quiet()
    False
    """
    return _QUIET.get()


def warn_no_maximum(what: str, consequence: str, advice: str) -> None:
    """Warn that a fit's likelihood has no finite maximum.

    The message reads ``"No finite maximum: <what>. <consequence>;
    <advice>."``, and the warning (a ``UserWarning``) is attributed to the
    first frame outside SurPyval, whichever entry point the fit was
    reached through. Nothing is given inside
    :func:`quiet_maximum_warnings`.

    Parameters
    ----------
    what : str
        Which parameter runs away, to where, and why (what in the data
        leaves the likelihood without a maximum).
    consequence : str
        Which reported quantities mean nothing as a result, e.g. ``"The
        reported sigma, its standard error and its bounds are
        meaningless"``.
    advice : str
        What to do instead.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.no_maximum import warn_no_maximum
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_no_maximum("theta runs away", "theta means nothing", "stop")
    >>> print(caught[0].message)
    No finite maximum: theta runs away. theta means nothing; stop.
    """
    if _QUIET.get():
        return
    warnings.warn(
        "No finite maximum: {}. {}; {}.".format(what, consequence, advice),
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )
