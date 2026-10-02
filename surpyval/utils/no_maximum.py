"""The warning a fit gives when its likelihood has no finite maximum (#392).

Some data leave a model's likelihood without a finite maximum: it keeps
increasing as a parameter runs off to infinity or to the edge of its
space, and the optimiser stops wherever it gives up. The fit then warns,
with one message of one form, and still returns the model it reached, as
``CoxPH`` does for a monotone partial likelihood. Each model detects the
condition in its own way (the criterion is documented where it is
applied); this module only says so, once per fit, at the caller's line.

A fit whose search stops at a point it cannot verify as a maximum (the
optimiser gave up, or the gradient is not zero there) says so with
:func:`warn_unverified`, again one message of one form for every model.

A caller that fits on the user's behalf and reads what the fit reached
from the model instead (``Parametric.maximum``; ``fit_best`` does) runs
the fits inside :func:`quiet_maximum_warnings`, which holds back both
warnings so that the caller can say it once, in its own words.
"""

import contextlib
import contextvars
import warnings
from collections.abc import Iterator

from surpyval.utils.warnings import caller_stacklevel

# Set while a caller that reads ``maximum`` off its models is fitting
_QUIET: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "surpyval_quiet_maximum_warnings", default=False
)


@contextlib.contextmanager
def quiet_maximum_warnings() -> Iterator[None]:
    """Hold back the fits' warnings that their answer is not a verified
    maximum of the likelihood.

    Inside the block, :func:`warn_no_maximum` and :func:`warn_unverified`
    give nothing; every other warning is given. A univariate parametric
    model still records what it reached (``Parametric.maximum``), and the
    caller must say so itself.

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
        stacklevel=caller_stacklevel(),
    )


def warn_unverified(
    what: str,
    reason: "str | None" = None,
    advice: str = "check the fit, or try another `init`",
) -> None:
    """Warn that a fit's search did not reach a verified maximum.

    A verified maximum is a point where the gradient of the
    log-likelihood is zero and it curves down in every direction
    (principle 13). The message reads ``"<what> did not reach a verified
    maximum of the likelihood (...); the parameters returned are the best
    point it found. The likelihood may have no maximum -- a parameter
    running off to a limit of its range -- or the search may have stalled
    (<reason>): <advice>."``, as a ``UserWarning`` attributed to the
    first frame outside SurPyval. Nothing is given inside
    :func:`quiet_maximum_warnings`.

    Parameters
    ----------
    what : str
        The search, as the subject of the sentence, e.g. ``"The additive
        hazards fit"``.
    reason : str, optional
        What the optimiser reported, if anything useful.
    advice : str, optional
        What to do instead. Default ``"check the fit, or try another
        `init`"``.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.no_maximum import warn_unverified
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_unverified("The fit")
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    The fit did not reach a verified maximum of the likelihood (a point
    where the gradient is zero and the log-likelihood curves down in every
    direction); the parameters returned are the best point it found. The
    likelihood may have no maximum -- a parameter running off to a limit
    of its range -- or the search may have stalled: check the fit, or try
    another `init`.
    """
    if _QUIET.get():
        return
    stalled = "stalled ({})".format(reason) if reason else "stalled"
    warnings.warn(
        "{} did not reach a verified maximum of the likelihood (a point "
        "where the gradient is zero and the log-likelihood curves down in "
        "every direction); the parameters returned are the best point it "
        "found. The likelihood may have no maximum -- a parameter running "
        "off to a limit of its range -- or the search may have {}: "
        "{}.".format(what, stalled, advice),
        UserWarning,
        stacklevel=caller_stacklevel(),
    )
