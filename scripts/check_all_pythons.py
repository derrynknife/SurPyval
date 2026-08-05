#!/usr/bin/env python3
"""Run the deployment checks locally, across every supported Python.

Continuous integration runs the test suite only on the release pull
request into ``master`` (see docs/Contributing.rst). That keeps the
edit-review loop fast, at the cost of finding an interpreter-specific
failure at release time rather than when it lands. This script is the
other half of that trade: it runs what CI would have run, on all three
interpreters, before you push.

It exists because the failure it guards against is not hypothetical.
The doctest numeric comparison added in 0.19.1 passed on 3.11 -- the
interpreter it was written on -- and failed on 3.12 and 3.13, because an
optimiser landed on a different last digit. Nothing short of actually
running the other interpreters would have found it.

    python scripts/check_all_pythons.py                # 3.11, 3.12, 3.13
    python scripts/check_all_pythons.py 3.12           # just one
    python scripts/check_all_pythons.py --skip-install # reuse as-is

Environments live in ``.venvs/py3.X`` (git-ignored) and are reused
between runs, so only the first is slow. ``uv`` is used when it is
installed, because it makes the install step take seconds; otherwise
this falls back to ``venv`` and ``pip``, which works the same and takes
longer.

Exits non-zero if any check on any interpreter fails.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENVS = ROOT / ".venvs"
DEFAULT_VERSIONS = ["3.11", "3.12", "3.13"]

# What CI runs, minus lint (which runs on every pull request anyway, so
# it is not what this script is for). Kept in the same order and with
# the same arguments as .github/workflows/actions.yml -- if they drift,
# this stops being a preview of CI and becomes its own thing.
CHECKS: list[tuple[str, list[str]]] = [
    (
        "test suite",
        [
            "-m",
            "pytest",
            "-n",
            "auto",
            "-q",
            "--ignore=surpyval/tests/alpha",
            "--run-ml",
        ],
    ),
    (
        "doctests",
        [
            "-m",
            "pytest",
            "--doctest-modules",
            "surpyval",
            "-q",
            "--ignore=surpyval/tests",
            "--ignore=surpyval/alpha",
        ],
    ),
    (
        "doctests (numeric forced)",
        [
            "-m",
            "pytest",
            "--doctest-modules",
            "surpyval",
            "-q",
            "--ignore=surpyval/tests",
            "--ignore=surpyval/alpha",
            "--doctest-force-numeric",
        ],
    ),
]


def run(cmd: list[str], **kwargs) -> int:
    """Run a command, streaming its output, and return its exit status."""
    return subprocess.call(cmd, cwd=ROOT, **kwargs)


def interpreter_for(venv: Path) -> Path:
    bin_dir = "Scripts" if sys.platform == "win32" else "bin"
    exe = "python.exe" if sys.platform == "win32" else "python"
    return venv / bin_dir / exe


def ensure_env(version: str, skip_install: bool) -> Path | None:
    """Create (or reuse) the environment for ``version``.

    Returns the interpreter path, or None if the interpreter is not
    available on this machine -- which is a thing to report, not a
    thing to crash on.
    """
    venv = VENVS / f"py{version}"
    python = interpreter_for(venv)
    uv = shutil.which("uv")

    if not python.exists():
        print(f"\n[{version}] creating {venv.relative_to(ROOT)}")
        if uv:
            created = run([uv, "venv", "--python", version, str(venv)])
        else:
            base = shutil.which(f"python{version}")
            if base is None:
                print(f"[{version}] no python{version} on PATH -- skipping")
                return None
            created = run([base, "-m", "venv", str(venv)])
        if created != 0 or not python.exists():
            print(f"[{version}] could not create an environment -- skipping")
            return None

    if not skip_install:
        print(f"[{version}] installing")
        # pytest-xdist is not in the tests extra; it is a development
        # tool, and the suite's ``-n auto`` needs it.
        target = ["-e", ".[tests]", "pytest-xdist"]
        if uv:
            installed = run(
                [uv, "pip", "install", "-q", "--python", str(python), *target]
            )
        else:
            installed = run(
                [str(python), "-m", "pip", "install", "-q", *target]
            )
        if installed != 0:
            print(f"[{version}] install failed -- skipping")
            return None

    return python


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the CI checks locally on every supported Python."
    )
    parser.add_argument(
        "versions",
        nargs="*",
        default=DEFAULT_VERSIONS,
        help="Python versions to check (default: %s)"
        % " ".join(DEFAULT_VERSIONS),
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="reuse the environments as they are, without reinstalling",
    )
    args = parser.parse_args()

    VENVS.mkdir(exist_ok=True)
    results: list[tuple[str, str, str, float]] = []

    for version in args.versions:
        python = ensure_env(version, args.skip_install)
        if python is None:
            results.append((version, "environment", "UNAVAILABLE", 0.0))
            continue
        for name, cmd in CHECKS:
            print(f"\n[{version}] {name}")
            started = time.monotonic()
            status = run([str(python), *cmd])
            elapsed = time.monotonic() - started
            results.append(
                (version, name, "ok" if status == 0 else "FAILED", elapsed)
            )

    print("\n" + "=" * 60)
    failed = False
    for version, name, status, elapsed in results:
        if status != "ok":
            failed = True
        print(f"{version:6s} {name:28s} {status:12s} {elapsed:6.1f}s")
    print("=" * 60)

    if failed:
        print("\nSomething failed. Do not push.")
        return 1
    print("\nAll checks passed on every interpreter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
