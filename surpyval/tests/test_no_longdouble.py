"""The library does not compute in extended precision.

``np.longdouble`` (``np.float128``) is double precision on arm64 (Apple
silicon) and Windows, and 80-bit only on x86-64 Linux, so a number
computed in it differs by platform. No library source mentions it; this
pins that (the tests are left out: they may name it to say why not).
"""

from pathlib import Path

import surpyval

PACKAGE = Path(surpyval.__file__).resolve().parent
TESTS = PACKAGE / "tests"
WORDS = ("longdouble", "float128")


def _sources():
    return sorted(p for p in PACKAGE.rglob("*.py") if TESTS not in p.parents)


def test_sources_are_found():
    names = {p.relative_to(PACKAGE).as_posix() for p in _sources()}
    assert "__init__.py" in names
    assert len(names) > 50
    assert not any(n.startswith("tests/") for n in names)


def test_no_extended_precision_in_the_library():
    found = [
        f"{p.relative_to(PACKAGE)}:{k}: {line.strip()}"
        for p in _sources()
        for k, line in enumerate(p.read_text("utf-8").splitlines(), 1)
        if any(w in line for w in WORDS)
    ]
    assert found == [], "\n".join(found)
