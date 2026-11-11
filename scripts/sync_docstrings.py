#!/usr/bin/env python3
"""Keep the repeated parameter entries of the distribution docstrings in
step with each class's ``_param_docs``.

Every method of a distribution repeats the class's parameter entries in
its docstring, literally, so the source reads the same as ``help()``. The
class's ``_param_docs`` is the one copy to edit; this script rewrites the
method copies from it.

    python scripts/sync_docstrings.py --check   # list drifted copies
    python scripts/sync_docstrings.py --write   # rewrite them

``--check`` exits non-zero when a copy has drifted, with the difference;
``surpyval/tests/utils/test_shared_docstrings.py`` makes the same check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from surpyval.utils.docstrings import copies, sync  # noqa: E402


def files() -> list[Path]:
    """The library's source files that state a ``_param_docs``."""
    return sorted(
        p
        for p in (ROOT / "surpyval").rglob("*.py")
        if "tests" not in p.parts and "_param_docs =" in p.read_text()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    drifted = 0
    for path in files():
        found = sync(path) if args.write else copies(path)
        for c in found:
            if args.write or not c.in_step:
                drifted += 1
                print(f"{path.relative_to(ROOT)}: {c.cls}.{c.method}")
                if args.check:
                    print(c.diff())
    if args.write:
        print(f"rewrote {drifted} copies")
        return 0
    print(f"{drifted} drifted copies")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
