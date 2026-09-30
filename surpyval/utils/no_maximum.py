"""The warning a fit gives when its likelihood has no finite maximum (#392).

Some data leave a model's likelihood without a finite maximum: it keeps
increasing as a parameter runs off to infinity or to the edge of its
space, and the optimiser stops wherever it gives up. The fit then warns,
with one message of one form, and still returns the model it reached, as
``CoxPH`` does for a monotone partial likelihood. Each model detects the
condition in its own way (the criterion is documented where it is
applied); this module only says so, once per fit, at the caller's line.
"""

import warnings

from surpyval.utils import _caller_stacklevel


def warn_no_maximum(what: str, consequence: str, advice: str) -> None:
    """Warn that a fit's likelihood has no finite maximum.

    The message reads ``"No finite maximum: <what>. <consequence>;
    <advice>."``, and the warning (a ``UserWarning``) is attributed to the
    first frame outside SurPyval, whichever entry point the fit was
    reached through.

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
    warnings.warn(
        "No finite maximum: {}. {}; {}.".format(what, consequence, advice),
        UserWarning,
        stacklevel=_caller_stacklevel(),
    )
