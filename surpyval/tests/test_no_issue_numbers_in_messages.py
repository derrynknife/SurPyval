"""Error and warning messages say what to do, not where an issue is.

A message that ends "tracked in #NNN" sends a user to an issue page,
which may be closed or moved, instead of saying what to do (principles 2
and 22). No string in the library's code names an issue number, an
issue URL or "tracked in"; docstrings and comments may (they are for
readers of the code and the API reference). The tests are left out.
"""

import ast
import re
from pathlib import Path

import surpyval

PACKAGE = Path(surpyval.__file__).resolve().parent
TESTS = PACKAGE / "tests"
# ``#`` and two to four digits standing alone: an issue reference, not a
# colour (``#008300``).
ISSUE = re.compile(r"#\d{2,4}\b|/issues/\d|tracked in", re.IGNORECASE)


def _sources():
    return sorted(p for p in PACKAGE.rglob("*.py") if TESTS not in p.parents)


def _docstrings(tree):
    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, kinds) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(
                first.value, ast.Constant
            ):
                out.add(id(first.value))
    return out


def _message_strings(path):
    tree = ast.parse(path.read_text("utf-8"))
    docs = _docstrings(tree)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docs
        ):
            yield node.lineno, node.value


def test_the_check_finds_an_issue_reference():
    assert ISSUE.search("not built yet; tracked in #867.")
    assert ISSUE.search("see (#476).")
    assert not ISSUE.search("color='#008300'")


def test_no_message_names_an_issue():
    found = [
        f"{p.relative_to(PACKAGE)}:{line}: {text[:80]!r}"
        for p in _sources()
        for line, text in _message_strings(p)
        if ISSUE.search(text)
    ]
    assert found == [], "\n".join(found)
