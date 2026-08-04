"""Suite-wide fixtures: doctest number comparison, and opt-in gating.

The first half of this file makes the ``--doctest-modules`` run compare
the numbers in an example's output as numbers rather than as text; see
the comment above ``RTOL``. The rest is the opt-in gating below.

Two groups are skipped unless asked for, because both are expensive and
neither guards a regression that the default run would miss quickly:

``ml``
    The beta-stage survival tree and forest tests. They fit hundreds of
    small Weibulls per test and take 97 of the suite's ~180 seconds --
    over half the runtime for 85 of its 2000-odd tests.

``invariants``
    The combinatorial fit-invariant sweep. It is a wide net rather than
    a targeted regression test, so it belongs in a deliberate run rather
    than in every edit-test cycle.

Continuous integration passes ``--run-ml`` only, so its coverage is
unchanged. The invariant sweep is deliberately *not* run there: it is a
net for exploring, cast deliberately when the fitting paths are being
worked on, and three and a half minutes on every pull request across
three Python versions buys little when its assertions hold. Run it
locally after touching a likelihood, an initialiser or an optimiser.

Marks are applied by path so the test modules themselves stay free of
suite-management boilerplate.

This lives at the repository root rather than beside the tests because
``pytest_addoption`` is only honoured in *initial* conftest files -- the
rootdir's, and those in directories named as arguments. The CI
invocation selects with ``--ignore`` and passes no path, so a conftest
under ``surpyval/tests`` is loaded too late to register the flags and
the run dies on "unrecognized arguments".
"""

import doctest
import math
import re

import pytest

# ---------------------------------------------------------------------------
# Numeric comparison for the ``--doctest-modules`` run
# ---------------------------------------------------------------------------
# doctest compares printed output as text. That is the wrong test for a
# library whose examples end in an optimiser: the same fit lands on
# ``b = 4.1995e-05`` under one Python and ``4.2032e-05`` under the next,
# and numpy prints eight significant digits either way, so a byte-exact
# comparison fails on a difference no reader would call a difference.
#
# The alternative -- trimming every documented number to the digits that
# happen to agree everywhere -- makes the docstring show something the
# user's own session will not produce, which is the thing these examples
# exist to avoid. So the examples record the real output, in full, and
# the numbers in it are compared as numbers.
#
# The fallback only runs after the ordinary text comparison has failed,
# and only fires when the two outputs are identical apart from their
# numeric literals -- same words, same brackets, same integer-vs-float
# shape ("1" never matches "1.", which is a dtype change worth
# failing on). What it forgives is the value drifting inside a
# tolerance. What it still catches is everything that actually went
# wrong when this was first switched on: a stale value from another
# parameterisation, a different function being called, the wrong array
# shape, an exception, a missing import.
#
# RTOL is set by the loosest genuine disagreement between supported
# Pythons -- the Duane example above, at 9e-4 -- with no margin beyond
# that. ATOL exists for the one other case, a restoration factor whose
# true value is zero and which surfaces as 1e-16 with whatever sign and
# mantissa the optimiser stopped on; relative tolerance is meaningless
# there.
RTOL = 1e-3
ATOL = 1e-12

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_BLANKLINE = re.compile(r"(?m)^%s\s*?$" % re.escape(doctest.BLANKLINE_MARKER))


def _skeleton(text: str) -> str:
    """The text with each number replaced by its *kind*.

    Integers and floats get different placeholders so that a change in
    dtype -- ``array([1, 2])`` becoming ``array([1., 2.])`` -- is still
    a failure rather than two numbers that happen to be equal.

    Whitespace is dropped entirely. numpy pads an array's columns to its
    widest element, so shortening one number moves the spaces around
    every other: ``[ 6.32508961 17.37701969]`` against
    ``[ 6.3250866 17.377018 ]``. Those spaces carry no meaning the
    numeric comparison below has not already made.
    """

    def mark(match: re.Match) -> str:
        token = match.group(0)
        return "~f" if ("." in token or "e" in token or "E" in token) else "~i"

    return "".join(_NUMBER.sub(mark, text).split())


def _numerically_equal(want: str, got: str) -> bool:
    # ``<BLANKLINE>`` stands for an empty line in the expected output.
    # The text comparison substitutes it before matching, so this one has
    # to as well, or a model repr with a blank line in it can never reach
    # the numeric comparison at all.
    want = _BLANKLINE.sub("", want)

    if _skeleton(want) != _skeleton(got):
        return False
    wants = _NUMBER.findall(want)
    gots = _NUMBER.findall(got)
    if not wants or len(wants) != len(gots):
        return False
    return all(
        math.isclose(float(w), float(g), rel_tol=RTOL, abs_tol=ATOL)
        for w, g in zip(wants, gots)
    )


_text_check_output = doctest.OutputChecker.check_output


def _check_output(self, want, got, optionflags):
    if _text_check_output(self, want, got, optionflags):
        return True
    return _numerically_equal(want, got)


# Patched on the base class rather than installed as a checker: pytest
# builds its own ``LiteralsOutputChecker`` subclass and calls up to this
# method, so overriding here survives both plain ``doctest`` and pytest,
# and does not depend on pytest's internals.
_patched = _check_output  # type: ignore[assignment]
doctest.OutputChecker.check_output = _patched  # type: ignore[method-assign]


def _forced_check_output(self, want, got, optionflags):
    """As above, but the numeric path is the *only* path.

    The fallback normally runs only when an example's output has
    actually drifted, which on any one machine is a handful of them. A
    gap in it -- the ``<BLANKLINE>`` markers it did not strip, say --
    therefore stays invisible locally and surfaces in CI, on whichever
    Python happens to compute a different last digit.

    Under ``--doctest-force-numeric`` every example whose output
    contains a number is compared numerically instead, so the fallback
    is exercised against all 229 of them rather than against today's
    accidental few. Outputs with no numbers keep the text comparison;
    there is nothing in them for this to compare.
    """
    if not _NUMBER.search(want):
        return _text_check_output(self, want, got, optionflags)
    return _numerically_equal(want, got)


OPT_IN = {
    "ml": (
        "--run-ml",
        "beta ML tree/forest tests",
        "surpyval/tests/beta/ml",
    ),
    "invariants": (
        "--run-invariants",
        "combinatorial fit-invariant sweep",
        "surpyval/tests/univariate/parametric/test_fit_invariants.py",
    ),
}


def pytest_addoption(parser):
    for _, (flag, description, _path) in OPT_IN.items():
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run the {description} (skipped by default)",
        )
    parser.addoption(
        "--doctest-force-numeric",
        action="store_true",
        default=False,
        help=(
            "compare every doctest example's numbers numerically, not "
            "only those whose text has drifted; exercises the fallback "
            "against all of them"
        ),
    )


def pytest_configure(config):
    for mark, (flag, description, _path) in OPT_IN.items():
        config.addinivalue_line(
            "markers", f"{mark}: {description}; opt in with {flag}"
        )
    if config.getoption("--doctest-force-numeric"):
        doctest.OutputChecker.check_output = (  # type: ignore[method-assign]
            _forced_check_output  # type: ignore[assignment]
        )


def pytest_collection_modifyitems(config, items):
    for mark, (flag, description, path) in OPT_IN.items():
        wanted = config.getoption(flag)
        for item in items:
            location = str(item.fspath).replace("\\", "/")
            if path not in location:
                continue
            item.add_marker(getattr(pytest.mark, mark))
            if not wanted:
                item.add_marker(
                    pytest.mark.skip(reason=f"needs {flag} ({description})")
                )
