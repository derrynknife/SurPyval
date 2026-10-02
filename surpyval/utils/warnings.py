"""Where a warning points: the first frame outside surpyval."""

from __future__ import annotations


def caller_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame
    outside surpyval: the fit and ``fit_from_df`` paths reach the warning
    through different depths of library code. The package's own tests
    are callers too, so a test can check where a warning points."""
    import os
    import sys

    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_dir = os.path.join(package_dir, "tests") + os.sep
    frame = sys._getframe(1)
    level = 1
    while frame is not None:
        name = os.path.abspath(frame.f_code.co_filename)
        if not name.startswith(package_dir + os.sep) or name.startswith(
            tests_dir
        ):
            break
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    return level
