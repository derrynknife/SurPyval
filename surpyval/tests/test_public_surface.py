"""The public names, frozen: none disappears without deprecation.

``public_surface.txt``, beside this module, lists every public name of
the namespaces users import from (:data:`NAMESPACES`), one dotted name a
line. A namespace with an ``__all__`` promises what it lists; one without
promises the names it defines that have no leading underscore (not the
modules it imports, nor what it imports from elsewhere). ``surpyval.beta``
is left out: its interface may change without notice (the deprecation
policy in ``docs/Contributing.rst``).

The first test fails when a listed name has gone without going through
deprecation: an old name must still be reachable with a
``DeprecationWarning`` or ``FutureWarning``
(:mod:`surpyval.utils.deprecation`), or raise an error that names its
replacement (:mod:`surpyval.utils.removed_names`, or a "moved" hint such as
``surpyval.TrendTestResult``'s). The second fails when a new public name
is not yet in the list.

To update the list deliberately, after adding a name (with a changelog
line under "Added")::

    python -m surpyval.tests.test_public_surface --update

It prints the names it adds and drops. A name still deprecated, or gone
with a removal hint, stays listed until it is dropped here. The helper
refuses to drop a name that has neither unless given ``--force``: for an
old name whose removal hint has itself been retired, or a removal the
maintainer has decided to make without a deprecation (said in the
changelog under "Removed").
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import warnings
from pathlib import Path
from types import ModuleType

#: The namespaces whose names are public and frozen: the package, the
#: subpackages the documentation imports from, and the univariate
#: families' packages (each reachable as ``surpyval.<name>`` too).
NAMESPACES: tuple[str, ...] = (
    "surpyval",
    "surpyval.univariate.parametric",
    "surpyval.univariate.nonparametric",
    "surpyval.univariate.regression",
    "surpyval.univariate.regression.kinds",
    "surpyval.univariate.competing_risks",
    "surpyval.recurrent",
    "surpyval.degradation",
    "surpyval.multivariate",
    "surpyval.ml",
    "surpyval.life_models",
    "surpyval.metrics",
    "surpyval.diagnostics",
    "surpyval.bayes",
    "surpyval.bayes.prior",
    "surpyval.test_planning",
    "surpyval.datasets",
    "surpyval.pymc",
    "surpyval.utils",
)

SNAPSHOT = Path(__file__).with_name("public_surface.txt")

#: What an error for an old name says when it names the replacement:
#: ``removed_names.removed_message`` and the package's "moved" hints.
_HINTS = ("removed in v", ": it is in ", "; it is now ")

_UPDATE = "python -m surpyval.tests.test_public_surface --update"


def _defined_in(obj: object, package: str) -> bool:
    """Whether ``obj`` (a class, function or fitter instance) is defined in
    ``package`` or below it; objects with no module (constants) count."""
    owner: object
    if inspect.isclass(obj) or inspect.isfunction(obj):
        owner = obj.__module__
    elif hasattr(obj, "fit") and not inspect.isroutine(obj):
        owner = type(obj).__module__
    else:
        owner = getattr(obj, "__module__", None)
    if not isinstance(owner, str):
        return True
    return owner == package or owner.startswith(package + ".")


def public_names(namespace: str) -> set[str]:
    """The public names of ``namespace``, dotted."""
    module = importlib.import_module(namespace)
    names = getattr(module, "__all__", None)
    if names is None:
        # (``dir``, not ``vars``: a name the namespace loads on first use,
        # through its ``__getattr__`` and ``__dir__``, is public too)
        names = [
            name
            for name in dir(module)
            if not name.startswith("_")
            and not isinstance(getattr(module, name), ModuleType)
            and _defined_in(getattr(module, name), namespace)
        ]
    return {f"{namespace}.{name}" for name in names}


def current_surface() -> set[str]:
    """Every public name of :data:`NAMESPACES`, dotted."""
    surface: set[str] = set()
    for namespace in NAMESPACES:
        surface |= public_names(namespace)
    return surface


def read_snapshot() -> set[str]:
    lines = SNAPSHOT.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip()}


def how_retired(dotted: str) -> str | None:
    """How a name gone from the public surface is handled: ``"deprecated"``
    (still reachable, with a warning), ``"hint"`` (an error naming the
    replacement), or ``None`` (gone silently, or still there with no
    warning)."""
    namespace, _, name = dotted.rpartition(".")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            # A whole namespace may be the deprecated or removed name.
            getattr(importlib.import_module(namespace), name)
        except (AttributeError, ImportError) as error:
            return "hint" if any(h in str(error) for h in _HINTS) else None
    if any(
        issubclass(w.category, (DeprecationWarning, FutureWarning))
        for w in caught
    ):
        return "deprecated"
    return None


def test_how_retired_tells_the_kinds_apart() -> None:
    # Removed with a hint (removed_names), moved with a hint, deprecated
    # by date, a removed module, and two that would be silent.
    assert how_retired("surpyval.RecurrentEventData") == "hint"
    assert how_retired("surpyval.TrendTestResult") == "hint"
    assert how_retired("surpyval.utils.score.score") == "hint"
    assert how_retired("surpyval.fs_to_xcnt") == "deprecated"
    assert how_retired("surpyval.recurrent.NoSuchModel") is None
    assert how_retired("surpyval.Weibull") is None  # there, no warning


def test_no_public_name_disappears_without_deprecation() -> None:
    gone = sorted(read_snapshot() - current_surface())
    silent = [name for name in gone if how_retired(name) is None]
    assert not silent, (
        "These public names are gone (or no longer public) without a "
        "deprecation: {}. Restore each and deprecate it first "
        "(surpyval.utils.deprecation: it warns for at least one release, "
        "then raises an error naming the replacement, "
        "surpyval.utils.removed_names), as the deprecation policy in "
        "docs/Contributing.rst says. If the removal is decided without "
        "one, run `{} --force` and say so in docs/changelog.rst under "
        "Removed.".format(silent, _UPDATE)
    )


def test_every_public_name_is_in_the_snapshot() -> None:
    new = sorted(current_surface() - read_snapshot())
    assert not new, (
        "New public names not in surpyval/tests/public_surface.txt: {}. "
        "If each is meant to be public (it then follows the deprecation "
        "policy), run `{}` and add a line to docs/changelog.rst under "
        "Added; otherwise give it a leading underscore or leave it out of "
        "its namespace's __all__.".format(new, _UPDATE)
    )


def update(force: bool = False) -> int:
    """Rewrite :data:`SNAPSHOT` from the package as it is, printing what
    changed; 1 (and no change) when a dropped name has no deprecation or
    removal hint and ``force`` is not set."""
    old, new = read_snapshot(), current_surface()
    dropped = sorted(old - new)
    silent = [name for name in dropped if how_retired(name) is None]
    if silent and not force:
        print(
            "Not updated: these names are gone without a deprecation or a "
            f"removal hint: {silent}. Deprecate them first, or pass --force "
            "for a removal decided without one."
        )
        return 1
    for name in sorted(new - old):
        print(f"+ {name}")
    for name in dropped:
        print(f"- {name}")
    SNAPSHOT.write_text("".join(f"{n}\n" for n in sorted(new)), "utf-8")
    print(f"{len(new)} public names in {SNAPSHOT.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check or update the frozen list of public names."
    )
    parser.add_argument(
        "--update", action="store_true", help="rewrite the list"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --update, drop names that have no deprecation",
    )
    args = parser.parse_args(argv)
    if args.update:
        return update(force=args.force)
    added = sorted(current_surface() - read_snapshot())
    gone = sorted(read_snapshot() - current_surface())
    print(f"new: {added}\ngone: {gone}")
    return 1 if added or gone else 0


if __name__ == "__main__":
    sys.exit(main())
