"""Warnings that leak out of the package (#379, principle 22).

A warning a user sees should be one SurPyval meant to give: a
``UserWarning`` (or its own warning class, or a deliberate
``RuntimeWarning``) raised on purpose, once, saying what went wrong.
A raw numerical warning -- numpy's "divide by zero encountered in
log", an overflow inside autograd, scipy's "invalid value" -- that
escapes from the package is a *leak*: it says nothing the user can act
on, and it is often the only sign of a wrong or fragile result.

The conformance suite runs every test under :func:`watch` (see
``conftest.py``) and fails a test that leaks. A warning is a leak when,
at the moment it is raised,

- the first frame of the call stack (innermost outwards) that belongs
  to the ``surpyval`` package is package code, not ``surpyval/tests``
  -- so the tests' own arithmetic and third-party code the tests call
  directly are not the package's business; and
- either the warning was raised by third-party code (numpy's Python
  functions, scipy, autograd, pandas) that the package called -- it
  reached the user unhandled -- or it is one of numpy's floating-point
  warnings (:data:`NUMPY_FP`) raised by an expression in package code.

A warning the package raises itself with ``warnings.warn`` is
deliberate, whatever its class. numpy's floating-point warnings come
from C, so ``filterwarnings`` cannot tell them apart by module (they
are attributed to the Python line that called the ufunc, often in
autograd or numpy); the call stack, read while the warning is being
shown, can. The limits: a warning raised with an ``"error"`` filter,
or inside a ``catch_warnings(record=True)`` block (``pytest.warns``),
never reaches the hook; a warning filtered as a duplicate of one
already shown in the same test is seen once; and code run in another
thread or process is not watched.

Known leaks, each with the reason it is not yet fixed, are listed in
:data:`KNOWN_LEAKS`; ``test_warnings.py`` reproduces each one as a
strict xfail, so the day it is fixed the suite says so.
"""

import contextlib
import os
import re
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from types import FrameType

import surpyval

PACKAGE = os.path.dirname(os.path.abspath(surpyval.__file__)) + os.sep
TESTS = os.path.join(PACKAGE, "tests") + os.sep

# numpy's floating-point error warnings (np.errstate "warn").
NUMPY_FP = re.compile(
    r"(divide by zero|overflow|underflow|invalid value) encountered in "
)

# The frames of the warnings machinery and of this module, skipped when
# reading the stack.
_SKIP = {
    os.path.abspath(warnings.__file__),
    os.path.abspath(contextlib.__file__),
    os.path.abspath(__file__),
}


@dataclass(frozen=True)
class Leak:
    """A warning that escaped from package code."""

    category: str
    message: str
    # Where it was raised (file:line, possibly third-party) and the
    # package frame it escaped from ("<path in the package>:<function>").
    raised_at: str
    package_frame: str
    package_line: int

    def __str__(self) -> str:
        return (
            f"{self.category}: {self.message} (raised at "
            f"{self.raised_at}, from {self.package_frame}, line "
            f"{self.package_line})"
        )


# (package frame, message) -> why the leak is not fixed yet. The package
# frame is "<path relative to surpyval/>:<function>", so an entry does not
# go stale when lines move. Each has a strict-xfail reproduction in
# test_warnings.py.
KNOWN_LEAKS: dict[tuple[str, str], str] = {
    (
        "univariate/nonparametric/nonparametric.py:df",
        "invalid value encountered in multiply",
    ): (
        "#408: df = hf * exp(-Hf) is inf * 0 = NaN at and after the time "
        "a Kaplan-Meier estimate reaches zero: KaplanMeier.fit([1, 2, 3])"
        ".df([2.5, 3.5, 4.5]) is not finite, where the step "
        "probabilities are 1/3 and 0"
    ),
    (
        "univariate/regression/semi_parametric_regression_model.py:phi",
        "overflow encountered in exp",
    ): (
        "#409: CoxPH fitted to separated data with a constant covariate "
        "column gives that column a coefficient of 3.1e14, so exp(beta'Z) "
        "overflows and sf is NaN (properties/test_regression.py, "
        "test_rows_are_independent[CoxPH])"
    ),
    (
        "univariate/parametric/distributions/logistic.py:sf",
        "invalid value encountered in divide",
    ): (
        "#410: Logistic.sf is e / (1 + e) with e = exp(-(x - mu) / sigma), "
        "which is inf / inf = NaN once (mu - x) / sigma > 709: Logistic.sf(0, "
        "1000, 1) is NaN, not 1 (properties/test_parametric.py, "
        "test_fit_succeeds_or_refuses[Logistic], nightly profile)"
    ),
    (
        "univariate/parametric/distributions/logistic.py:sf",
        "overflow encountered in exp",
    ): "#410: the overflow behind the NaN of Logistic.sf (entry above)",
    (
        "utils/linalg.py:wald_bound_on_support",
        "invalid value encountered in sqrt",
    ): (
        "#411: a Wald bound from a negative variance is NaN with a raw sqrt "
        "warning: GeneralizedRenewal on the registry fixture puts q at "
        "2.7e-16, on its bound, where the inverse Hessian has variances "
        "-0.031 and -10.6, so param_cb('alpha') and param_cb('q') are "
        "[nan, nan] with nothing saying why (test_options.py)"
    ),
}


def _short(path: str) -> str:
    path = os.path.abspath(path)
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def _inspect(message, category) -> tuple[str | None, Leak | None]:
    """Where a warning being shown now comes from.

    ``("leak", leak)``, ``("deliberate", None)`` (raised on purpose by
    package code) or ``(None, None)`` (the tests' own code, or nothing of
    the package's on the stack). Must be called from within the
    ``showwarning`` hook, while the stack that raised the warning is
    still live.
    """
    frame: FrameType | None = sys._getframe(1)
    inner: FrameType | None = None
    while frame is not None:
        name = os.path.abspath(frame.f_code.co_filename)
        if name in _SKIP:
            frame = frame.f_back
            continue
        if inner is None:
            inner = frame
        if name.startswith(PACKAGE):
            break
        frame = frame.f_back
    if frame is None or inner is None:
        return None, None
    name = os.path.abspath(frame.f_code.co_filename)
    if name.startswith(TESTS):
        return None, None
    text = str(message)
    if frame is inner and not (
        issubclass(category, RuntimeWarning) and NUMPY_FP.match(text)
    ):
        return "deliberate", None
    rel = name[len(PACKAGE) :].replace(os.sep, "/")
    leak = Leak(
        category=category.__name__,
        message=text,
        raised_at="{}:{}".format(
            _short(inner.f_code.co_filename), inner.f_lineno
        ),
        package_frame=f"{rel}:{frame.f_code.co_name}",
        package_line=frame.f_lineno,
    )
    return "leak", leak


def classify(message, category) -> Leak | None:
    """The :class:`Leak` a warning being shown now is, or ``None``
    (called from a ``showwarning`` hook; see :func:`_inspect`)."""
    return _inspect(message, category)[1]


def is_known(leak: Leak) -> bool:
    return (leak.package_frame, leak.message) in KNOWN_LEAKS


@contextlib.contextmanager
def watch() -> Iterator[list[Leak]]:
    """Collect the leaks raised in the block into the yielded list.

    Every warning, leak or not, still goes on to the ``showwarning``
    that was in place (pytest's record), so the warnings summary is
    unchanged.
    """
    found: list[Leak] = []
    with warnings.catch_warnings():
        shown = warnings.showwarning

        def hook(message, category, filename, lineno, file=None, line=None):
            leak = classify(message, category)
            if leak is not None:
                found.append(leak)
            shown(message, category, filename, lineno, file, line)

        warnings.showwarning = hook
        yield found


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    """Silence the warnings raised in the block, except leaks.

    For fits, whose optimisers warn as they go: the deliberate warnings
    are dropped, and a leak goes on, as a warning, to the enclosing
    :func:`watch` (or, outside one, to pytest's summary).
    """
    with warnings.catch_warnings():
        # "default": each distinct warning once, so a warning repeated
        # in an optimiser's loop costs one call of the hook.
        warnings.simplefilter("default")
        shown = warnings.showwarning

        def hook(message, category, filename, lineno, file=None, line=None):
            if classify(message, category) is not None:
                shown(message, category, filename, lineno, file, line)

        warnings.showwarning = hook
        yield


@contextlib.contextmanager
def deliberate() -> Iterator[list[tuple[type, str]]]:
    """Record every warning the package raises on purpose in the block,
    repeats included, as ``(category, message)``; drop the rest.

    For the check that a problem is reported once per call. Leaks are
    passed on as warnings (to the enclosing :func:`watch`).
    """
    found: list[tuple[type, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        shown = warnings.showwarning

        def hook(message, category, filename, lineno, file=None, line=None):
            kind, _ = _inspect(message, category)
            if kind == "deliberate":
                found.append((category, str(message)))
            elif kind == "leak":
                shown(message, category, filename, lineno, file, line)

        warnings.showwarning = hook
        yield found


def report(leaks: list[Leak]) -> str:
    """One line per distinct leak."""
    lines = sorted({str(leak) for leak in leaks})
    return "\n".join(lines)


def fail_on_new(found: list[Leak]) -> None:
    """Fail the running test if ``found`` holds a leak not yet known."""
    import pytest

    new = [leak for leak in found if not is_known(leak)]
    if new:
        pytest.fail(
            "a raw warning leaked out of the package (see "
            "surpyval/tests/conformance/leaks.py):\n" + report(new),
            pytrace=False,
        )
