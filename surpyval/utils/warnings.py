"""Where a warning points (the first frame outside surpyval), and the
warnings several models share."""

from __future__ import annotations


def caller_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame
    outside surpyval: the fit and ``fit_from_df`` paths reach the warning
    through different depths of library code. The package's own tests
    are callers too, so a test can check where a warning points."""
    import os
    import sys

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(package_dir, "tests") + os.sep
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        name = os.path.abspath(frame.f_code.co_filename)
        if not name.startswith(package_dir + os.sep) or name.startswith(
            tests_dir
        ):
            break
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    return level


def warn_no_covariance() -> None:
    """Warn, once at the caller, that a model's information matrix could
    not be evaluated or inverted, so its covariance is unavailable (and
    NaN).

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.warnings import warn_no_covariance
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warn_no_covariance()
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    The information matrix could not be evaluated or inverted (the optimum
    may be at a parameter boundary); covariance is unavailable.
    """
    import warnings

    warnings.warn(
        "The information matrix could not be evaluated or inverted (the "
        "optimum may be at a parameter boundary); covariance is "
        "unavailable.",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


class MaskedCauseWarning(UserWarning):
    """Failures whose event type was not recorded (masked causes): fits
    of every failure count them; a fit of one type treats them as
    ``for_event_type``'s ``masked`` says; competing-risks fits refuse
    them."""


def warn_masked(count: int, which: str) -> None:
    """Warn, once at the caller, of ``count`` failures with no recorded
    event type, ``which`` naming them (items or rows)."""
    import warnings

    have = "has" if count == 1 else "have"
    warnings.warn(
        f"{count} failure{'' if count == 1 else 's'} {have} no recorded "
        f"event type (masked causes): {which}. Fits of every failure count "
        "them; a fit of one type treats them as for_event_type's masked= "
        "says (not of that type, by default); competing-risks fits refuse "
        "them. relabel_event_types({None: 'unknown'}) makes them a type of "
        "their own.",
        MaskedCauseWarning,
        stacklevel=caller_stacklevel(),
    )


def warn_suspension_types(count: int, which: str) -> None:
    """Warn, once at the caller, that ``count`` suspensions had an event
    type, which was dropped."""
    import warnings

    warnings.warn(
        f"{count} suspension{'' if count == 1 else 's'} (right censored: "
        f"{which}) had an event type, which was dropped: a suspension has "
        "not failed, so it has none.",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


def named(values: list, limit: int = 5) -> str:
    """Up to ``limit`` of ``values``, for a message: "'A', 'B' and 3
    more"."""
    shown = ", ".join(repr(v) for v in values[:limit])
    extra = len(values) - limit
    return shown + (f" and {extra} more" if extra > 0 else "")
