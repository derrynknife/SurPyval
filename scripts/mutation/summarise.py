"""Summarise a mutmut run (#396): counts and score per file.

    python summarise.py <results.txt> [<mutmut> <survivors.txt>]

``results.txt`` is the output of ``mutmut results --all true``. With the
path of the ``mutmut`` executable and an output file, the diff of every
surviving mutant is also written there (``mutmut show``), for triage.

The score is ``(killed + timeout) / (all - skipped - not checked)``: a
mutant no test reaches ("no tests") counts against it, as a survivor
would, and one not run (a sampled run) is left out. Mutants triaged as
equivalent are not known to mutmut; the README keeps those counts by
hand.
"""

import collections
import re
import subprocess
import sys

_LINE = re.compile(r"^\s*(?P<name>\S+): (?P<status>.+?)\s*$")


def module_of(name: str) -> str:
    # surpyval.a.b.x_func__mutmut_3 / surpyval.a.b.xǁClassǁm__mutmut_3
    return name.rsplit(".", 1)[0]


def main(argv: list[str]) -> None:
    counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    survivors = []
    with open(argv[1], encoding="utf-8") as fh:
        for line in fh:
            match = _LINE.match(line)
            if not match:
                continue
            name, status = match["name"], match["status"]
            counts[module_of(name)][status] += 1
            if status == "survived":
                survivors.append(name)

    total: collections.Counter = collections.Counter()
    for module in sorted(counts):
        c = counts[module]
        total.update(c)
        print(_row(module, c))
    if len(counts) > 1:
        print(_row("total", total))

    if len(argv) > 3:
        mutmut, out = argv[2], argv[3]
        with open(out, "w", encoding="utf-8") as fh:
            for name in survivors:
                shown = subprocess.run(
                    [mutmut, "show", name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                fh.write(f"# {name}\n{shown.stdout}\n")


def _row(module: str, c: collections.Counter) -> str:
    counted = sum(c.values()) - c["skipped"] - c["not checked"]
    caught = c["killed"] + c["timeout"]
    score = caught / counted if counted else float("nan")
    parts = ", ".join(f"{k} {v}" for k, v in sorted(c.items()))
    return f"{module}: {sum(c.values())} mutants ({parts}); score {score:.1%}"


if __name__ == "__main__":
    main(sys.argv)
