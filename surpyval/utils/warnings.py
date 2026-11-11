"""Where a warning points (the first frame outside surpyval), the
warnings several models share, and holding warnings back in one thread
only (:class:`ignoring_warnings`)."""

from __future__ import annotations

import functools
import operator
import re
import sys
import threading
import warnings as _warnings
from collections.abc import Callable
from types import TracebackType
from typing import Any

# -- an "ignore" filter only its own thread matches --------------------------
#
# Python keeps one list of warning filters for the whole process, and asks
# of each filter, in turn, whether the warning's message matches the
# filter's (``filter_message.match(text)``), and its category and module.
# The filter below has, in the message's place, a ``threading.local``: its
# ``match`` is looked up in the calling thread, and is "never" in every
# thread except one inside an :class:`ignoring_warnings` block. So the
# filter stays in the shared list, and each thread turns it on and off for
# itself without touching the list.
#
# Everything the interpreter calls while it reads the list is C (a
# ``threading.local`` attribute, a ``functools.partial`` of an ``operator``
# function, a compiled pattern's ``match``): no Python code runs there, so
# no other thread can run, and change or free the list, half way through.
# That is why the filter is not written with ``__subclasscheck__`` or a
# Python ``match``, and why a filter, once added, is never taken out of
# the list again (only its threads' switches are turned off).

_NOTHING = object()
#: ``match`` of a thread that is ignoring nothing: false for any message.
_never: Callable[[Any], Any] = functools.partial(operator.is_, _NOTHING)
#: ``match`` of a thread ignoring every message.
_always: Callable[[Any], Any] = functools.partial(operator.is_not, _NOTHING)


class _ThreadSwitch(threading.local):
    """The message slot of a thread-local filter: ``match`` is each
    thread's own, and "never" until the thread sets it."""

    # (a staticmethod: a ``functools.partial`` in a class is bound to the
    # instance, as a method is, from Python 3.14, and warns of it in 3.13)
    match: Callable[[Any], Any] = staticmethod(_never)


@functools.lru_cache(maxsize=256)
def _any_of(messages: tuple[str, ...]) -> Callable[[Any], Any]:
    """``match`` for a thread ignoring the warnings whose message starts
    with any of ``messages`` (regular expressions, case ignored)."""
    return re.compile("|".join(f"(?:{m})" for m in messages), re.I).match


def _filters_changed() -> None:
    """Tell the interpreter the filters changed, as
    ``warnings.simplefilter`` does (it forgets which "show once" warnings
    it has shown, so they are judged against the filters as they are)."""
    changed = getattr(_warnings, "_filters_mutated", None)
    if changed is not None:
        changed()
        return
    held = getattr(_warnings, "_filters_mutated_lock_held", None)
    lock = getattr(_warnings, "_lock", None)
    if held is not None and lock is not None:  # pragma: no cover (3.14+)
        with lock:
            held()


def _switch_for(category: type[Warning]) -> _ThreadSwitch:
    """The switch of the thread-local "ignore" filter for ``category`` at
    the front of the filters, adding the filter if it is not there.

    "At the front" is among the leading filters of this kind: the newest
    filter wins in Python, so a block must come before whatever was added
    since the last one (a ``catch_warnings`` block around this call, a
    test's own filters). Adding at the front moves the other filters
    along without hiding any from a thread reading the list."""
    filters: list[Any] = _warnings.filters  # type: ignore[assignment]
    for item in filters:
        found = item[1] if isinstance(item, tuple) and len(item) == 5 else 0
        if not isinstance(found, _ThreadSwitch):
            break
        if item[2] is category:
            return found
    switch = _ThreadSwitch()
    filters.insert(0, ("ignore", switch, category, None, 0))
    return switch


class ignoring_warnings:
    """Ignore, in this thread only, the warnings raised inside the block.

    ``with warnings.catch_warnings(): warnings.simplefilter("ignore")``
    changes the one filter list every thread shares: while the block runs,
    every other thread's warnings are dropped too, and two threads
    leaving such blocks out of order can leave the "ignore" in place for
    good. This block instead switches on, for its own thread, a filter
    that no other thread matches, and switches it off when it ends; it
    never saves and restores the list. In its own thread it behaves as
    the standard form does: the newest filter wins, so a
    ``catch_warnings`` block opened inside it sees what it asks to see.

    Parameters
    ----------
    category : type, optional
        The warnings to ignore; every ``Warning`` by default.
    message : str, optional
        A regular expression the start of the message must match (case
        ignored), as in ``warnings.filterwarnings``.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.warnings import ignoring_warnings
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     with ignoring_warnings():
    ...         warnings.warn("held back")
    ...     warnings.warn("given")
    >>> [str(w.message) for w in caught]
    ['given']
    """

    __slots__ = ("_category", "_message", "_switch", "_standard")

    def __init__(
        self, category: type[Warning] = Warning, message: str = ""
    ) -> None:
        self._category = category
        self._message = message

    def __enter__(self) -> None:
        if getattr(sys.flags, "context_aware_warnings", False):
            # Python 3.14 run with context-aware warnings: the standard
            # form is already local to the thread (and its filters come
            # before the shared list this class adds to).
            self._standard = _warnings.catch_warnings()
            self._standard.__enter__()
            _warnings.filterwarnings(
                "ignore", self._message, category=self._category
            )
            return
        self._standard = None  # type: ignore[assignment]
        self._switch = switch = _switch_for(self._category)
        # (the thread's own attributes: the messages of its open blocks
        # on this filter, None for a block ignoring every message)
        open_blocks = switch.__dict__.setdefault("open_blocks", [])
        open_blocks.append(self._message or None)
        self._set(switch, open_blocks)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._standard is not None:
            self._standard.__exit__(exc_type, exc, tb)
            return
        switch = self._switch
        open_blocks = switch.__dict__["open_blocks"]
        # (blocks end in the reverse of the order they began in)
        mine = self._message or None
        for i in range(len(open_blocks) - 1, -1, -1):
            if open_blocks[i] == mine:
                del open_blocks[i]
                break
        self._set(switch, open_blocks)

    @staticmethod
    def _set(switch: _ThreadSwitch, open_blocks: list[str | None]) -> None:
        if not open_blocks:
            switch.match = _never
        elif None in open_blocks:
            switch.match = _always
        else:
            switch.match = _any_of(tuple(m for m in open_blocks if m))
        _filters_changed()


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
