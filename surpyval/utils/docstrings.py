"""
Keeping repeated docstring sections in step with one source.

A distribution's methods (``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``,
``mean``, ``moment``, ...) all describe the same parameters. Each method's
docstring keeps its own literal copy of those entries, so the text reads
the same in the source, in an editor's hover, in ``help()`` and in the
documentation. The class states them once more, in ``_param_docs``, and
that is the copy to edit: ``scripts/sync_docstrings.py --write`` rewrites
the method copies from it, and ``surpyval/tests/utils/
test_shared_docstrings.py`` fails, naming the class, method and
difference, when a copy has drifted.

This works on source text, not on imported objects, and nothing in the
package imports it at run time.
"""

from __future__ import annotations

import ast
import difflib
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path

#: The sections that end a Parameters section.
_SECTIONS = (
    "Returns",
    "Yields",
    "Raises",
    "Warns",
    "Examples",
    "Notes",
    "See Also",
    "References",
)


@dataclass
class Copy:
    """One method's copy of its class's parameter entries.

    ``start`` and ``end`` are 0-based source line indices (``end``
    exclusive) of the copy in its file; ``found`` the lines there and
    ``expected`` the lines ``_param_docs`` gives."""

    path: Path
    cls: str
    method: str
    start: int
    end: int
    found: list[str]
    expected: list[str]

    @property
    def in_step(self) -> bool:
        return self.found == self.expected

    def diff(self) -> str:
        """The difference, as a unified diff from the class's entries to
        the method's copy."""
        return "\n".join(
            difflib.unified_diff(
                self.expected,
                self.found,
                f"{self.cls}._param_docs",
                f"{self.cls}.{self.method}",
                lineterm="",
            )
        )


def _entry_names(entries: list[str]) -> list[str]:
    names = []
    for line in entries:
        match = re.match(r"^([A-Za-z_]\w*) :", line)
        if match:
            names.append(match.group(1))
    return names


def _entry_end(lines: list[str], i: int, end: int, indent: str) -> int:
    """The line after the entry that starts at ``lines[i]``: its
    description lines are indented deeper and not blank."""
    j = i + 1
    while (
        j < end
        and lines[j].strip()
        and lines[j].startswith(indent)
        and lines[j][len(indent)] in " \t"
    ):
        j += 1
    return j


def copies(path: str | Path) -> list[Copy]:
    """Every method copy of a ``_param_docs`` in the file ``path``.

    A copy starts at the method's Parameters entry for the first of the
    class's parameters and runs over as many entries as the class has
    parameters. A method that describes none of them has no copy."""
    path = Path(path)
    lines = path.read_text().split("\n")
    out = []
    for cls in ast.walk(ast.parse("\n".join(lines))):
        if not isinstance(cls, ast.ClassDef):
            continue
        spec = None
        for node in cls.body:
            if (
                isinstance(node, ast.Assign)
                and [getattr(t, "id", None) for t in node.targets]
                == ["_param_docs"]
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                spec = textwrap.dedent(node.value.value).strip("\n").rstrip()
        if spec is None:
            continue
        spec_lines = spec.split("\n")
        names = _entry_names(spec_lines)
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not fn.body or not isinstance(fn.body[0], ast.Expr):
                continue
            doc = fn.body[0].value
            if not (
                isinstance(doc, ast.Constant) and isinstance(doc.value, str)
            ):
                continue
            first, last = doc.lineno - 1, doc.end_lineno or doc.lineno
            heads = [
                k
                for k in range(first, last)
                if lines[k].strip() == "Parameters"
            ]
            if not heads:
                continue
            k = heads[0] + 1
            stop = next(
                (m for m in range(k, last) if lines[m].strip() in _SECTIONS),
                last,
            )
            pattern = re.compile(rf"^(\s*){re.escape(names[0])} :")
            start = next(
                (m for m in range(k, stop) if pattern.match(lines[m])), None
            )
            if start is None:
                continue
            indent = pattern.match(lines[start]).group(1)  # type: ignore
            end = start
            for _ in names:
                if end >= stop or not lines[end].startswith(indent):
                    break
                end = _entry_end(lines, end, stop, indent)
            if (
                _entry_names(
                    textwrap.dedent("\n".join(lines[start:end])).split("\n")
                )
                != names
            ):
                # Not a copy: the name means something else here (a
                # Binomial fit's ``n`` is the count of each observation).
                continue
            expected = [indent + line if line else line for line in spec_lines]
            out.append(
                Copy(
                    path,
                    cls.name,
                    fn.name,
                    start,
                    end,
                    lines[start:end],
                    expected,
                )
            )
    return out


def sync(path: str | Path) -> list[Copy]:
    """Rewrite every drifted copy in the file ``path`` from its class's
    ``_param_docs``; the copies that were rewritten."""
    path = Path(path)
    drifted = [c for c in copies(path) if not c.in_step]
    if drifted:
        lines = path.read_text().split("\n")
        for c in sorted(drifted, key=lambda c: c.start, reverse=True):
            lines[c.start : c.end] = c.expected
        path.write_text("\n".join(lines))
    return drifted
