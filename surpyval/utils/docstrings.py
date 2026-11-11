"""
Shared sections of docstrings, written once and filled in at import.

A distribution's methods (``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``,
``mean``, ``moment``, ...) all describe the same parameters. Rather than
repeat those entries in every method, a class states them once, in
``_param_docs``, and each method docstring has a ``{params}`` line where
they belong. ``fill_param_docs`` replaces that line with the entries,
indented to match, when the class is created.

The filled text is the docstring Python, ``help()``, Sphinx and doctest
read (``__doc__``), so what a user sees is unchanged. Only an editor that
reads the source without importing it shows the ``{params}`` line.
"""

from __future__ import annotations

import re
import textwrap
import types

#: The line, on its own in a docstring, that ``fill_param_docs`` replaces.
PARAMS_MARKER = "{params}"

_MARKER_LINE = re.compile(r"^([ \t]*)\{params\}[ \t]*$", re.MULTILINE)


def fill_params(doc: str, entries: str) -> str:
    """``doc`` with each ``{params}`` line replaced by ``entries``
    (dedented text, one entry per line block), indented to the line's own
    indentation.

    >>> doc = "Parameters\\n    ----------\\n    x : float\\n    {params}\\n"
    >>> print(fill_params(doc, "a : float\\n    The scale."))
    Parameters
        ----------
        x : float
        a : float
            The scale.
    <BLANKLINE>
    """

    def indented(match: re.Match[str]) -> str:
        indent = match.group(1)
        return "\n".join(
            indent + line if line else line for line in entries.split("\n")
        )

    return _MARKER_LINE.sub(indented, doc)


def fill_param_docs(cls: type) -> None:
    """Fill the ``{params}`` line of every method docstring that ``cls``
    defines with the class's own ``_param_docs`` entries.

    Only the class's own attributes are read and changed: a subclass that
    states no ``_param_docs`` leaves its methods as they are. Under
    ``python -OO`` there are no docstrings and nothing is done.
    """
    text = cls.__dict__.get("_param_docs")
    if not isinstance(text, str):
        return
    entries = textwrap.dedent(text).strip("\n").rstrip()
    for attr in cls.__dict__.values():
        fn = attr
        if isinstance(fn, (staticmethod, classmethod)):
            fn = fn.__func__
        if not isinstance(fn, types.FunctionType):
            continue
        doc = fn.__doc__
        if doc and PARAMS_MARKER in doc:
            fn.__doc__ = fill_params(doc, entries)
