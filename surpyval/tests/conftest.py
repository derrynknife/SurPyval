"""The opt-in gating (see the root ``conftest.py``), and the modules left
out when an optional test dependency is missing.

This ships with the package, so the suite installed from a wheel collects
and gates as it does in the repository (#661). There the flags are not
registered (the root conftest, which registers them, is not installed),
and every opt-in group is skipped.
"""

import importlib.util
import sys

import pytest

from surpyval.tests._suite import OPT_IN

# Modules that need a dependency of the ``tests`` extra rather than of the
# package: left out when it is not installed, rather than failing the
# whole collection with an ImportError (paths relative to this file).
_OPTIONAL = {
    "hypothesis": ["properties"],
    "sksurv": ["ml/forest/test_tree.py"],
    "bson": ["test_mongodb_serialisation.py"],
}

collect_ignore = [
    path
    for module, paths in _OPTIONAL.items()
    if importlib.util.find_spec(module) is None
    for path in paths
]


def pytest_configure(config):
    for mark, (flag, description, _path) in OPT_IN.items():
        config.addinivalue_line(
            "markers", f"{mark}: {description}; opt in with {flag}"
        )


def pytest_collection_modifyitems(config, items):
    for mark, (flag, description, path) in OPT_IN.items():
        wanted = config.getoption(flag, default=False)
        for item in items:
            location = str(item.fspath).replace("\\", "/")
            # A test outside the path opts in by carrying the mark (the
            # conformance sweeps that run with the calibration studies).
            if path not in location and item.get_closest_marker(mark) is None:
                continue
            item.add_marker(getattr(pytest.mark, mark))
            if not wanted:
                item.add_marker(
                    pytest.mark.skip(reason=f"needs {flag} ({description})")
                )


# The pooled coverage of the calibration studies. ``check_coverage``
# (calibration/_montecarlo.py) keeps the coverages it checks; they go onto
# the test's report here (which pytest-xdist carries back from its
# workers), and the run ends with the mean of each study's cells at each
# level. A study a point or two short in every cell passes every cell's
# check at a few hundred replicates; the mean shows it. A report only:
# nothing here passes or fails a test.
_MONTECARLO = "surpyval.tests.calibration._montecarlo"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    montecarlo = sys.modules.get(_MONTECARLO)
    if montecarlo is not None and montecarlo.POOLED:
        if call.when == "call":
            item.user_properties.append(
                ("pooled_coverage", list(montecarlo.POOLED))
            )
        montecarlo.POOLED.clear()
    yield


def pooled_lines(reports):
    """One line per study and level: the mean coverage of its cells, the
    shortfall from nominal in points, and how many cells are below."""
    lines = []
    for report in reports:
        cells: dict = {}
        for key, value in getattr(report, "user_properties", ()):
            if key != "pooled_coverage":
                continue
            for nominal, coverages in value:
                cells.setdefault(round(nominal, 6), []).extend(coverages)
        for nominal in sorted(cells):
            cov = cells[nominal]
            mean = sum(cov) / len(cov)
            below = sum(c < nominal for c in cov)
            lines.append(
                "{} at {:.4g}: mean {:.4f} over {} cells ({:+.1f} points; "
                "{} below nominal)".format(
                    report.nodeid,
                    nominal,
                    mean,
                    len(cov),
                    100 * (mean - nominal),
                    below,
                )
            )
    return lines


def pytest_terminal_summary(terminalreporter):
    reports = [
        report
        for outcome in ("passed", "failed", "xfailed", "xpassed")
        for report in terminalreporter.stats.get(outcome, ())
        if getattr(report, "when", None) == "call"
    ]
    lines = pooled_lines(sorted(reports, key=lambda r: r.nodeid))
    if lines:
        terminalreporter.section("pooled calibration coverage")
        for line in lines:
            terminalreporter.write_line(line)
