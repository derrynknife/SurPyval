"""Suite-wide fixtures: doctest number comparison, and opt-in gating.

The first half of this file makes the ``--doctest-modules`` run compare
the numbers in an example's output as numbers rather than as text; see
the comment above ``RTOL``. The rest is the opt-in gating below.

Four groups are skipped unless asked for, because all are expensive and
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
    code runs, take ten to twenty minutes on four cores, and run in the
    release pull request's full run (on 3.12) and when asked for with
    ``--run-calibration``. A test
    elsewhere joins them by carrying the ``calibration`` mark: the
    likelihood-ratio option sweeps of the slow families
    (conformance/registry.py, ``Bound.nightly``).

``scenarios``
    The practitioner scenario cards under ``surpyval/tests/scenarios``:
    end-to-end studies a reliability engineer would run (field data from
    a maintenance system, an accelerated life test, a warranty forecast,
    a growth test ...), each with data simulated from a known truth. A
    card checks that the package recovers the truth and answers the
    study's questions; what it cannot yet answer is a strict xfail led
    by its issue. Seconds in all, but end to end rather than unit by
    unit; they run in the release pull request's full run (on 3.12) and
    when asked for with ``--run-scenarios``.

Continuous integration passes ``--run-ml`` everywhere, and
``--run-calibration`` and ``--run-scenarios`` on 3.12. The invariant
sweep is deliberately *not* run there: it is a net for exploring, cast
deliberately when the fitting paths are being worked on, and three and a
half minutes across three Python versions buys little when its
assertions hold. Run it
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
import json
import statistics
import sys
import zlib
from pathlib import Path

import pytest

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
        "--shard",
        default=None,
        metavar="K/N",
        help=(
            "run only the K-th of N parts of the collected tests (1-based), "
            "chosen by a hash of each test's id; CI splits each "
            "interpreter's run into parts that run at once; balanced by "
            "the recorded test times when --shard-durations names a file"
        ),
    )
    parser.addoption(
        "--shard-durations",
        default=None,
        metavar="PATH",
        help=(
            "a JSON file of recorded test times (seconds by test id, as "
            "--store-durations writes); with --shard, the parts are "
            "balanced by these times instead of split by hash. A missing "
            "file falls back to the hash split"
        ),
    )
    parser.addoption(
        "--store-durations",
        default=None,
        metavar="PATH",
        help=(
            "write each test's time (setup, call and teardown, in "
            "seconds) to this JSON file at the end of the run"
        ),
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


def _read_durations(path):
    """The recorded test times in ``path``, or ``None`` with no file."""
    if not path or not Path(path).is_file():
        return None
    with open(path) as f:
        durations = json.load(f)
    if not isinstance(durations, dict):
        raise pytest.UsageError(
            f"--shard-durations {path}: give a JSON object of seconds by "
            "test id, as --store-durations writes"
        )
    return {k: float(v) for k, v in durations.items()}


def _balanced_parts(nodeids, durations, n):
    """Each test's part (0-based) out of ``n``, balanced by time.

    The longest tests are placed first, each in the part with the least
    time so far (ties to the lower part). A test with no recorded time (a
    new one) counts as the median recorded time. The order depends only on
    the ids and the times, so every xdist worker and the controller agree
    on the parts."""
    default = statistics.median(durations.values()) if durations else 1.0
    timed = sorted(
        ((durations.get(i, default), i) for i in set(nodeids)),
        key=lambda t: (-t[0], t[1]),
    )
    load = [0.0] * n
    part = {}
    for seconds, nodeid in timed:
        k = min(range(n), key=lambda j: (load[j], j))
        part[nodeid] = k
        load[k] += seconds
    return part


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """``--shard K/N``: keep the tests in part K of N.

    With ``--shard-durations`` naming a file of recorded times, the parts
    are balanced by those times (``_balanced_parts``); otherwise a test
    belongs to the part its id hashes to (CRC-32 of the test id, the same
    in every process). Either way the xdist workers and the controller
    agree on the part, and every test falls in exactly one part.
    Parametrised cases of one test land in different parts, which spreads
    the heavy families of cases."""
    shard = config.getoption("--shard")
    if not shard:
        return
    try:
        k, n = (int(v) for v in shard.split("/"))
    except ValueError:
        raise pytest.UsageError(f"--shard takes K/N, not {shard!r}")
    if not 1 <= k <= n:
        raise pytest.UsageError(f"--shard {shard}: K must be in 1..N")
    durations = _read_durations(config.getoption("--shard-durations"))
    if durations is None:
        parts = {
            item.nodeid: zlib.crc32(item.nodeid.encode()) % n for item in items
        }
    else:
        parts = _balanced_parts([item.nodeid for item in items], durations, n)
    keep, drop = [], []
    for item in items:
        (keep if parts[item.nodeid] == k - 1 else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


# ``--store-durations``: each test's time, summed over its setup, call and
# teardown. Under xdist the reports reach the controller, which writes the
# file; the workers write nothing.
_durations: dict = {}


def pytest_runtest_logreport(report):
    _durations[report.nodeid] = (
        _durations.get(report.nodeid, 0.0) + report.duration
    )


def pytest_sessionfinish(session):
    path = session.config.getoption("--store-durations")
    if not path or hasattr(session.config, "workerinput"):
        return
    with open(path, "w") as f:
        json.dump(
            {k: round(v, 3) for k, v in sorted(_durations.items())},
            f,
            indent=0,
        )


def pytest_runtest_setup(item):
    """Each doctest starts with no figure open: an example that draws on
    ``plt.gca()`` would otherwise draw on an earlier example's axes (its
    labels and lines), and the doctests run one after another in one
    process."""
    if isinstance(item, pytest.DoctestItem) and "matplotlib.pyplot" in (
        sys.modules
    ):
        sys.modules["matplotlib.pyplot"].close("all")


# ``surpyval/utils/score.py`` raises on import, to say where its function
# went (#653): the ``--doctest-modules`` run must not import it.
collect_ignore = [
    "surpyval/utils/score.py",
    # Kept only to raise an ImportError naming its replacement (removed in
    # v0.26), so importing it for the doctest run fails by design.
    "surpyval/utils/recurrent_event_data.py",
]
