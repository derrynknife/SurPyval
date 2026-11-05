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
  plugin leaves them out of a mutmut run (:data:`INTROSPECTION`), with
  the check of the package's imports, which reads the source mutmut
  rewrote;
- the tests that check a warning points at the caller's line (its
  ``filename`` is the test file: the warning's ``stacklevel`` is right)
  find mutmut's trampoline instead, one frame further down for each
  mutated function on the way. This plugin wraps ``warnings.warn`` to
  step over the trampoline frames, so ``stacklevel`` counts the frames of
  the code as written (:func:`_warn_over_trampolines`), and
  ``surpyval.utils.warnings.caller_stacklevel``, which looks for the
  first frame outside the package, to count over them too rather than
  stop at the first trampoline (:func:`_caller_stacklevel`).

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
import warnings

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
    # Reads the package's source, where mutmut adds imports of its own
    "surpyval/tests/test_private_imports.py",
)


def _unmangle(package_frame: str) -> str:
    path, _, name = package_frame.rpartition(":")
    match = _MANGLED.match(name)
    return f"{path}:{match['name']}" if match else package_frame


_TRAMPOLINE = os.sep + os.path.join("mutmut", "mutation", "trampoline.py")


def _warn_over_trampolines(warn):
    """``warn`` with its ``stacklevel`` counted over the frames of the code
    as written: each mutmut trampoline frame between the caller and the
    frame the level names is stepped over, and the wrapper's own frame."""

    def wrapped(message, category=None, stacklevel=1, *args, **kwargs):
        frame, level, extra = sys._getframe(1), 1, 0
        while frame is not None and level < stacklevel:
            frame = frame.f_back
            if frame is None:
                break
            if frame.f_code.co_filename.endswith(_TRAMPOLINE):
                extra += 1
            else:
                level += 1
        return warn(message, category, stacklevel + extra + 1, *args, **kwargs)

    wrapped._over_trampolines = True
    return wrapped


def _trampoline_file() -> str:
    import importlib

    trampoline = importlib.import_module("mutmut.mutation.trampoline")

    return os.path.abspath(str(trampoline.__file__))


def _caller_stacklevel() -> int:
    """``surpyval.utils.warnings.caller_stacklevel`` with mutmut's
    trampoline frames neither ending the walk nor counted (its code
    replaces that function's, so it runs with that module's globals)."""
    import os
    import sys

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(package_dir, "tests") + os.sep
    trampoline = os.sep + os.path.join("mutmut", "mutation", "trampoline.py")
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        name = os.path.abspath(frame.f_code.co_filename)
        if name.endswith(trampoline):
            frame = frame.f_back  # type: ignore[assignment]
            continue
        if not name.startswith(package_dir + os.sep) or name.startswith(
            tests_dir
        ):
            break
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    return level


def pytest_configure(config):
    # Before the package is imported, so a ``from warnings import warn``
    # in it takes the wrapper too
    if not getattr(warnings.warn, "_over_trampolines", False):
        warnings.warn = _warn_over_trampolines(warnings.warn)

    from surpyval.tests.conformance import leaks
    from surpyval.utils import warnings as package_warnings

    package_warnings.caller_stacklevel.__code__ = _caller_stacklevel.__code__
    # The leak check names the frame that called warnings.warn: look past
    # the wrapper, and past the trampolines to the package frames.
    leaks._SKIP.add(os.path.abspath(__file__))
    leaks._SKIP.add(_trampoline_file())

    if getattr(leaks.is_known, "_unmangled", False):
        return

    def is_known(leak):
        return (_unmangle(leak.package_frame), leak.message) in (
            leaks.KNOWN_LEAKS
        )

    is_known._unmangled = True
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
