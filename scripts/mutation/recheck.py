"""Rerun the surviving mutants of a finished mutmut run against new tests
(#396).

    python recheck.py <copy> <tests>... [-k EXPR] [--jobs N] [--out FILE]
        [--previous FILE]

``<copy>`` is the module's mutation copy (``$MUTATION_WORKDIR/<module>/
repo``, holding mutmut's ``mutants/`` directory); ``<tests>`` are test
paths relative to the repository root, e.g. the new kill tests. The
tests are copied into ``mutants/``, and each surviving mutant is run
against them alone, so the new tests can be judged in minutes instead of
rerunning the whole module (mutmut reruns only mutants whose code
changed, and would not pick up new tests). A mutant is killed when the
tests fail, and a timeout counts as killed, as in mutmut.

``-k`` is passed to pytest: leave out a strict xfail that pins a bug
the mutation copy does not have (it would pass there, fail as XPASS and
count every mutant as killed).

``--previous`` takes the ``--out`` file of an earlier recheck and runs
only the mutants it lists as survived (after adding more tests).

Prints one line per mutant (``killed``, ``survived`` or ``timeout``)
and the totals; ``--out`` also writes the lines to a file.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def survivors(mutants: Path) -> list[str]:
    names = []
    for meta in sorted(mutants.rglob("*.py.meta")):
        codes = json.loads(meta.read_text())["exit_code_by_key"]
        names += [name for name, code in codes.items() if code == 0]
    return names


def run_one(mutants: Path, tests: list[str], name: str, timeout: int) -> str:
    env = dict(os.environ, MUTANT_UNDER_TEST=name, PYTHONPATH=str(mutants))
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q"]
            + ["-p", "no:cacheprovider"]
            + tests,
            cwd=mutants,
            env=env,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timeout"
    if done.returncode == 5:  # no tests collected: a broken selection
        raise RuntimeError(done.stdout.decode()[-2000:])
    return "survived" if done.returncode == 0 else "killed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("copy", type=Path)
    parser.add_argument("tests", nargs="+")
    parser.add_argument("-k", dest="keyword")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()

    mutants = args.copy.resolve() / "mutants"
    for test in args.tests:
        src, dst = args.copy / test, mutants / test
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    tests = args.tests + (["-k", args.keyword] if args.keyword else [])
    names = survivors(mutants)
    if args.previous:
        before = dict(
            line.strip().rsplit(": ", 1)
            for line in args.previous.read_text().splitlines()
            if line.strip()
        )
        names = [n for n in names if before.get(n) == "survived"]
    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(run_one, mutants, tests, n, args.timeout): n
            for n in names
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
            print(futures[future], results[futures[future]], flush=True)

    lines = [f"{name}: {results[name]}" for name in names]
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = {
        s: list(results.values()).count(s) for s in set(results.values())
    }
    print(f"{len(names)} survivors rechecked: {counts}")


if __name__ == "__main__":
    main()
