"""Every public item is documented with an example (principle 23, #379).

Walks the same public namespaces as ``test_completeness`` and requires a
docstring with a runnable ``>>>`` example for each public class, function
and fitter. A fitter's ``fit`` docstring counts as its example, since that
is where a user looks first. The examples themselves run as doctests in
continuous integration, so this checks only that they exist.

The items that still lack one are listed in ``MISSING`` against #402. The
test fails for a new public item without a docstring and example, and for
a listed item that has gained one, so the list only shrinks.
"""

import inspect

import pytest

from surpyval.tests.conformance.test_completeness import NAMESPACES, _public

# Public items with a docstring but no example yet (#402).
MISSING_EXAMPLE: frozenset[str] = frozenset()
# Public items with no docstring at all yet (#402).
MISSING_DOCSTRING: frozenset[str] = frozenset()


def _items():
    """Each public object once, under the first (shortest) name it is
    exported as; ``surpyval.alpha`` is experimental and exempt."""
    seen = {}
    for namespace in NAMESPACES:
        if namespace == "surpyval.alpha":
            continue
        for name, obj in _public(namespace):
            seen.setdefault(id(obj), (name, obj))
    return sorted(seen.values(), key=lambda item: item[0])


def _documentation(obj):
    """The docstrings a user reads for ``obj``: its own, its class's (for
    a fitter instance) and its ``fit`` method's."""
    parts = [inspect.getdoc(obj)]
    if not (inspect.isclass(obj) or inspect.isfunction(obj)):
        parts.append(inspect.getdoc(type(obj)))
    fit = getattr(obj, "fit", None)
    if fit is not None:
        parts.append(inspect.getdoc(fit))
    return "\n".join(p for p in parts if p)


ITEMS = _items()


@pytest.mark.parametrize("name, obj", ITEMS, ids=[n for n, _ in ITEMS])
def test_public_item_is_documented_with_an_example(name, obj):
    doc = _documentation(obj)
    if name in MISSING_DOCSTRING:
        assert not doc.strip(), (
            f"{name} now has a docstring: remove it from MISSING_DOCSTRING "
            "(and add it to MISSING_EXAMPLE if it has no example yet)"
        )
        return
    assert doc.strip(), f"{name} has no docstring"
    if name in MISSING_EXAMPLE:
        assert (
            ">>>" not in doc
        ), f"{name} now has an example: remove it from MISSING_EXAMPLE"
        return
    assert ">>>" in doc, f"{name} has no '>>>' example in its docstring"


def test_known_gaps_are_public_items():
    names = {name for name, _ in ITEMS}
    stale = (MISSING_EXAMPLE | MISSING_DOCSTRING) - names
    assert not stale, f"no longer public (remove from the lists): {stale}"
