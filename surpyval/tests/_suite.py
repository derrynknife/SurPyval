"""Pieces of the test-suite set-up that ship with the package.

The repository-root ``conftest.py`` registers the opt-in flags and
patches doctest's output comparison; ``surpyval/tests/conftest.py``
applies the opt-in gating. Both read what is here, which lives in the
package (rather than in the root conftest, which is not installed) so
that the suite as installed from a wheel collects and gates the same
way (#661): ``test_doctest_checker`` imported ``_numerically_equal`` from
the root conftest and could not be collected from an installed copy.
"""

import doctest
import math
import re

# Numeric comparison for the ``--doctest-modules`` run
#
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


#: The groups skipped unless asked for: mark -> (flag, description, the
#: path a test is in to belong to it). See the root ``conftest.py``.
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
    "calibration": (
        "--run-calibration",
        "statistical calibration studies (coverage, size, bias)",
        "surpyval/tests/calibration",
    ),
    "scenarios": (
        "--run-scenarios",
        "practitioner scenario cards",
        "surpyval/tests/scenarios",
    ),
}
