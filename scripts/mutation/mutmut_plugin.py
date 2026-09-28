"""A pytest plugin for running the test suite under mutmut (#396).

mutmut 3 renames every function it mutates: ``NonParametric.df``
becomes ``xǁNonParametricǁdf__mutmut_orig``, and each mutant is another
attribute beside it (``xǁNonParametricǁdf__mutmut_12``), behind a
trampoline of the original name. Two kinds of test notice the renaming
rather than the mutation, and fail the clean run mutmut starts with:

- the leak check (``surpyval/tests/conformance/leaks.py``) names the
  package frame a warning escaped from by its function's code name, so a
  known leak (``KNOWN_LEAKS``) no longer matches its entry. This plugin
  maps the mangled name back to the original before the lookup;
- the tests that list a model's or a module's attributes with ``dir()``
  (every uncertainty method is swept, every option has one meaning,
  every public item is documented, ...) find hundreds of new "public"
  methods. They check the API, not the numbers a mutant changes, so this
  plugin leaves them out of a mutmut run (:data:`INTROSPECTION`).

Two more things make mutmut's bookkeeping match the suite:

- the conformance suite caches each case's fitted model
  (``registry._fitted``) and each uncertainty method's result
  (``test_options._CACHE``) for the session. mutmut records which tests
  reach a function in one clean run, and runs a mutant against those
  tests only; with the caches, only the first test to fit a case or
  compute a bound reaches the code, so a mutant is tried against a
  fraction of the tests that would catch it (``cb`` ignoring
  ``alpha_ci`` "survived", yet fails 12 of the Kaplan-Meier option
  checks). A mutant tried in a worker forked after the caches were
  filled would not reach the code at all. This plugin empties both
  caches before every test;
- ``MUTATION_CASES`` (a comma-separated list of conformance case names,
  set by ``run.sh``) keeps only the conformance tests of those cases:
  the others exercise other modules and cost time.

It changes nothing outside a mutmut run: ``run.sh`` copies it into the
mutation copy of the repository and loads it with ``-p mutmut_plugin``.
"""

import os
import re
import sys

_MANGLED = re.compile(r"^x(?:ǁ[^ǁ]*ǁ|_)(?P<name>.+?)__mutmut_(?:orig|\d+)$")

# Tests (node id prefixes, relative to the repository root) that
# enumerate attributes with ``dir()`` and so see mutmut's copies.
INTROSPECTION = (
    "surpyval/tests/conformance/test_options.py::"
    "test_every_uncertainty_method_is_swept",
    "surpyval/tests/conformance/test_options.py::test_option_convention",
    "surpyval/tests/conformance/test_defaults.py::"
    "test_entry_points_share_defaults",
    "surpyval/tests/conformance/test_completeness.py",
    "surpyval/tests/conformance/test_documentation.py",
    "surpyval/tests/univariate/parametric/test_shared_signatures.py",
)


def _unmangle(package_frame: str) -> str:
    path, _, name = package_frame.rpartition(":")
    match = _MANGLED.match(name)
    return f"{path}:{match['name']}" if match else package_frame


def pytest_configure(config):
    from surpyval.tests.conformance import leaks

    if getattr(leaks.is_known, "_unmangled", False):
        return

    def is_known(leak):
        return (_unmangle(leak.package_frame), leak.message) in (
            leaks.KNOWN_LEAKS
        )

    is_known._unmangled = True  # type: ignore[attr-defined]
    leaks.is_known = is_known


def _other_case(item, cases):
    # True for a conformance test of a case not in ``cases``.
    callspec = getattr(item, "callspec", None)
    if not cases or callspec is None:
        return False
    names = [
        getattr(v, "name", None)
        for v in callspec.params.values()
        if type(v).__name__ == "Case"
    ]
    return bool(names) and not set(names) & cases


def pytest_collection_modifyitems(config, items):
    cases = {c for c in os.environ.get("MUTATION_CASES", "").split(",") if c}
    kept, dropped = [], []
    for item in items:
        drop = item.nodeid.startswith(INTROSPECTION) or _other_case(
            item, cases
        )
        (dropped if drop else kept).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def pytest_runtest_setup(item):
    # Every test fits its models and computes its bounds afresh (see the
    # module docstring).
    for name, module in list(sys.modules.items()):
        if name.endswith("tests.conformance.registry"):
            module._fitted.cache_clear()
        elif name.endswith("tests.conformance.test_options"):
            module._CACHE.clear()
