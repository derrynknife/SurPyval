"""
The names v0.21 deprecated are gone in v0.22 (#422, principle 21).

An old argument name is now an unknown argument, so the call raises
Python's own ``TypeError``, and ``surpyval.experimental`` no longer
exists (``surpyval.beta.ml`` holds the survival tree and forest). A few
representative old names from each area are checked here.

The names deprecated since are removed in
``surpyval.utils.deprecation.REMOVED_IN``: the last test fails once the
package's version reaches it while any deprecation shim is left, so the
removal is not forgotten.
"""

import ast
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import surpyval
import surpyval as sp
from surpyval.recurrent import CrowAMSAA, NonParametricCounting
from surpyval.tests.conformance.registry import CASE_BY_NAME, fitted
from surpyval.univariate.competing_risks import CompetingRisks, FineGray
from surpyval.utils.deprecation import REMOVED_IN

X = np.array([5.0, 10.0, 20.0])

# Recurrent events: three items.
XR = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60, 5, 18, 30, 50, 60.0]
IR = [1] * 6 + [2] * 5 + [3] * 5
CR = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

# Competing risks: causes "a" and "b", with censoring.
XC = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ZC = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], float)[:, None]
EC = np.array(["a", "b", "a", None, "b", "a", "a", "b", None, "a"])


def _parametric_cb():
    fitted(CASE_BY_NAME["Weibull"]).cb(t=X)


def _cox_method():
    x = np.array([1, 1, 2, 2, 3, 3, 4, 5], float)
    Z = np.array([0, 1, 0, 1, 1, 0, 1, 0], float)[:, None]
    sp.CoxPH.fit(x, Z, method="breslow")


def _recurrent_confidence():
    NonParametricCounting.fit(XR, i=IR, c=CR).mcf_cb(X, confidence=0.9)


def _recurrent_seed():
    CrowAMSAA.fit(XR, i=IR, c=CR).count_terminated_simulation(3, 2, seed=1)


def _degradation_t():
    fitted(CASE_BY_NAME["WienerProcess"]).sf(t=X)


def _competing_risks_method():
    CompetingRisks.fit(XC, EC, method="Kaplan-Meier")


def _fine_gray_cause():
    FineGray.fit(XC, ZC, EC, cause="b")


OLD_NAMES = {
    "Parametric.cb(t=)": (_parametric_cb, "t"),
    "CoxPH.fit(method=)": (_cox_method, "method"),
    "NonParametricCounting.mcf_cb(confidence=)": (
        _recurrent_confidence,
        "confidence",
    ),
    "count_terminated_simulation(seed=)": (_recurrent_seed, "seed"),
    "WienerProcessModel.sf(t=)": (_degradation_t, "t"),
    "CompetingRisks.fit(method=)": (_competing_risks_method, "method"),
    "FineGray.fit(cause=)": (_fine_gray_cause, "cause"),
}


@pytest.mark.parametrize("call, old", OLD_NAMES.values(), ids=list(OLD_NAMES))
def test_old_name_is_an_unknown_argument(call, old):
    with pytest.raises(
        TypeError, match=f"unexpected keyword argument '{old}'"
    ):
        call()


def test_experimental_alias_is_gone():
    sys.modules.pop("surpyval.experimental", None)
    with pytest.raises(ImportError):
        importlib.import_module("surpyval.experimental")
    assert not hasattr(sp, "experimental")
    # Its contents live in surpyval.beta.ml.
    from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree

    assert SurvivalTree is not None and RandomSurvivalForest is not None


# ---------------------------------------------------------------------------
# The next removal: nothing deprecated outlives REMOVED_IN
# ---------------------------------------------------------------------------
# What keeps an old name alive: the renaming helpers of
# surpyval.utils.deprecation, REMOVED_IN itself (every hand-written
# deprecation message quotes it), and a DeprecationWarning.
_SHIM_NAMES = frozenset(
    {
        "REMOVED_IN",
        "renamed_arguments",
        "RenamedAttribute",
        "renamed_class_attribute",
        "CallableList",
        "DeprecationWarning",
    }
)


def _deprecation_shims(source):
    """The lines of ``source`` that refer to a deprecation shim."""
    lines = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.alias):
            name = node.name
        else:
            continue
        if name in _SHIM_NAMES:
            lines.add(node.lineno)
    return sorted(lines)


def _package_shims():
    """``path:line`` of every deprecation shim in the package (the
    machinery in ``utils/deprecation.py`` and the tests aside)."""
    package = Path(sp.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix()
        if relative.startswith("tests/") or relative == (
            "utils/deprecation.py"
        ):
            continue
        for line in _deprecation_shims(path.read_text(encoding="utf-8")):
            found.append(f"surpyval/{relative}:{line}")
    return found


def _release(version):
    # MAJOR.MINOR (two parts since 0.22); padded, so "0.23" and "0.23.0"
    # compare equal.
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    return tuple(parts + [0] * (3 - len(parts)))


def test_deprecation_shims_are_found():
    source = """
from surpyval.utils.deprecation import REMOVED_IN, renamed_arguments
import warnings

@renamed_arguments(old="new")
def f(new=1):
    return new

def g():
    warnings.warn("g is going", DeprecationWarning)
    return 2
"""
    assert _deprecation_shims(source) == [2, 5, 10]
    assert _deprecation_shims("import warnings\nx = 1\n") == []


def test_deprecated_names_are_removed_by_removed_in():
    if _release(sp.__version__) < _release(REMOVED_IN):
        return
    shims = _package_shims()
    assert not shims, (
        f"surpyval {sp.__version__} has reached REMOVED_IN ({REMOVED_IN}), "
        "but deprecated names are still accepted. Remove each shim (and its "
        "tests and documentation), or move REMOVED_IN on for a deprecation "
        "meant to last longer:\n" + "\n".join(shims)
    )


# ---------------------------------------------------------------------------
# The alpha series/parallel composition is removed (#284).
# ---------------------------------------------------------------------------


class TestAlphaCompositionRemoved:
    def test_models_no_longer_importable(self):
        # The alpha tier that held them was deleted too, once empty. (The
        # package directory is checked rather than an import refused: an
        # editable install of another checkout would still serve one.)
        package = Path(surpyval.__file__).parent
        assert not (package / "alpha").exists()
