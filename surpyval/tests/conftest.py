"""Opt-in gating for the slow parts of the test suite.

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
"""

import pytest

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


def pytest_configure(config):
    for mark, (flag, description, _path) in OPT_IN.items():
        config.addinivalue_line(
            "markers", f"{mark}: {description}; opt in with {flag}"
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
