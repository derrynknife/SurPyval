"""The opt-in gating (see the root ``conftest.py``), and the modules left
out when an optional test dependency is missing.

This ships with the package, so the suite installed from a wheel collects
and gates as it does in the repository (#661). There the flags are not
registered (the root conftest, which registers them, is not installed),
and every opt-in group is skipped.
"""

import importlib.util

import pytest

from surpyval.tests._suite import OPT_IN

# Modules that need a dependency of the ``tests`` extra rather than of the
# package: left out when it is not installed, rather than failing the
# whole collection with an ImportError (paths relative to this file).
_OPTIONAL = {
    "hypothesis": ["properties"],
    "sksurv": ["beta/ml/forest/test_tree.py"],
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
