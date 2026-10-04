"""Suite-wide fixtures: doctest number comparison, and opt-in gating.

The first half of this file makes the ``--doctest-modules`` run compare
the numbers in an example's output as numbers rather than as text; see
the comment above ``RTOL``. The rest is the opt-in gating below.

Three groups are skipped unless asked for, because all are expensive and
none guards a regression that the default run would miss quickly:

``ml``
    The beta-stage survival tree and forest tests. They fit hundreds of
    small Weibulls per test and take 97 of the suite's ~180 seconds --
    over half the runtime for 85 of its 2000-odd tests.

``invariants``
    The combinatorial fit-invariant sweep. It is a wide net rather than
    a targeted regression test, so it belongs in a deliberate run rather
    than in every edit-test cycle.

``calibration``
    The simulation studies under ``surpyval/tests/calibration``: coverage
    of confidence intervals, test size and power, estimator bias. They
    check that the answers are statistically right rather than that the
    code runs, take ten to twenty minutes on four cores, and run nightly
    (.github/workflows/nightly.yml), not on pull requests. A test
    elsewhere joins them by carrying the ``calibration`` mark: the
    likelihood-ratio option sweeps of the slow families
    (conformance/registry.py, ``Bound.nightly``).

Continuous integration passes ``--run-ml`` only, so its coverage is
unchanged. The invariant sweep is deliberately *not* run there: it is a
net for exploring, cast deliberately when the fitting paths are being
worked on, and three and a half minutes on every pull request across
three Python versions buys little when its assertions hold. Run it
locally after touching a likelihood, an initialiser or an optimiser.

Marks are applied by path so the test modules themselves stay free of
suite-management boilerplate. The flags are registered here; the marks
and the skipping are in ``surpyval/tests/conftest.py``, which ships with
the package, so that the suite installed from a wheel (where this file is
not) skips the same groups by default (#661).

This lives at the repository root rather than beside the tests because
``pytest_addoption`` is only honoured in *initial* conftest files -- the
rootdir's, and those in directories named as arguments. The CI
invocation selects with ``--ignore`` and passes no path, so a conftest
under ``surpyval/tests`` is loaded too late to register the flags and
the run dies on "unrecognized arguments".
"""

import doctest

# ---------------------------------------------------------------------------
# Numeric comparison for the ``--doctest-modules`` run
# ---------------------------------------------------------------------------
# The comparison itself, and why, is in ``surpyval/tests/_suite.py``,
# which ships with the package; it is installed on doctest here.
from surpyval.tests._suite import (  # noqa: E402
    _NUMBER,
    OPT_IN,
    _numerically_equal,
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
_patched = _check_output
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
    there is nothing in them for this to compare. Nor does an output
    that elides part of itself with ``...`` under ``ELLIPSIS`` (an
    error's message cut short): the numbers in the elided part cannot be
    paired with the ones expected.
    """
    elided = optionflags & doctest.ELLIPSIS and "..." in want
    if not _NUMBER.search(want) or elided:
        return _text_check_output(self, want, got, optionflags)
    return _numerically_equal(want, got)


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
    if config.getoption("--doctest-force-numeric"):
        doctest.OutputChecker.check_output = _forced_check_output
