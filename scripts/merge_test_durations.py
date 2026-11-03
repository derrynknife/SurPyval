#!/usr/bin/env python3
"""Merge the test times recorded by CI into one file for the shard split.

Each part of the release run's test suite records its tests' times
(``--store-durations``) and uploads them; the ``coverage`` job merges the
parts with this script and uploads the result as the ``test-durations``
artifact. Commit that file as ``.github/test_durations.json`` and the next
run balances its four parts by these times (``--shard-durations``, in the
root conftest.py) instead of splitting them by a hash of the test id.

A test timed on more than one interpreter gets the mean of its times.

    python scripts/merge_test_durations.py OUT IN [IN ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def merge(paths: list[str]) -> dict[str, float]:
    times: defaultdict[str, list[float]] = defaultdict(list)
    for path in paths:
        with open(path) as f:
            for nodeid, seconds in json.load(f).items():
                times[nodeid].append(float(seconds))
    return {k: round(sum(v) / len(v), 3) for k, v in sorted(times.items())}


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        sys.exit("usage: merge_test_durations.py OUT IN [IN ...]")
    out, *paths = argv
    merged = merge(paths)
    with open(out, "w") as f:
        json.dump(merged, f, indent=0)
        f.write("\n")
    print(f"{len(merged)} tests from {len(paths)} files -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
