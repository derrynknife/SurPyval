# Mutation testing (#396)

Coverage says a line ran; it does not say a test would notice if the line
were wrong. Mutation testing checks that directly: a tool makes one small
change to the package at a time (`<` to `<=`, `+ 1` to `- 1`, a dropped
argument, `side="right"` removed) and reruns the tests. A change no test
notices (a *surviving mutant*) is a blind spot of the suite. It is slow,
so it is run occasionally (monthly, or after a large change to a module),
not on pull requests.

## Running it

    scripts/mutation/run.sh nonparametric

Modules: `nonparametric` (the four files of the Kaplan-Meier, Nelson-Aalen
and Fleming-Harrington estimators and the `NonParametric` model),
`turnbull`, `cox_ph`, `parametric_fitter`, `competing_risks`; each is a
list of files to mutate, the tests to run against them and the
conformance cases to keep (see `modules()` in `run.sh`).

The run happens in a copy (mutmut rewrites the code it mutates; never run
it in a checkout you are editing): by default the working tree, tracked
and untracked files, or `MUTATION_REV=HEAD` for a commit. The results are
left in `$MUTATION_WORKDIR/<module>` (default `/tmp/surpyval-mutation`):

- `results.txt`: every mutant and its outcome (`mutmut results --all true`);
- `survivors.txt`: the diff of every surviving mutant (`mutmut show`);
- `summary.txt`: counts and score per file (`summarise.py`).

`MUTATION_JOBS` (default 2) is the number of mutants tested at once. The
first run creates a venv with uv (`MUTATION_VENV` to reuse one).

To see which survivors new tests kill without rerunning everything:

    $MUTATION_WORKDIR/venv/bin/python scripts/mutation/recheck.py \
        $MUTATION_WORKDIR/nonparametric/repo \
        surpyval/tests/mutation/test_nonparametric_kills.py --out recheck.txt

It copies the tests into the mutation copy and runs each surviving mutant
against them alone, in a fresh process (3 to 5 s a mutant; 26 min for 593
survivors with 2 jobs); a mutant killed there is killed by the suite with
those tests. `--previous recheck.txt` reruns only what an earlier recheck
left alive, and `-k` leaves out a strict xfail that pins a bug the
mutated commit does not have yet (it would XPASS and "kill" everything).

## Tool

mutmut 3.8.0 (Python 3.11). It runs each mutant against only the tests
that reach the mutated function (recorded in one clean run), and stops at
the first failure, which makes a module of 3000 mutants take hours rather
than days. cosmic-ray 8.7.0 was tried first: it generated 3776 mutations
for `nonparametric.py` alone but runs the whole test command for every
one, with no per-function test selection, so the same test set would take
days.

`mutmut_plugin.py` (loaded with `-p mutmut_plugin`) adapts the suite to
mutmut:

- the leak check names package frames by their code name, which mutmut
  mangles; the plugin unmangles it before looking up a known leak;
- the tests that list attributes with `dir()` (API completeness,
  documentation, option conventions) see mutmut's copies of every method
  and are left out: they check the API, not numbers;
- **the conformance suite's session caches are emptied before every
  test.** The suite fits each case once (`registry._fitted`) and computes
  each bound once (`test_options._CACHE`). mutmut associates a test with
  a function only if the test reaches it in the clean run, so with the
  caches only the first test to fit a case or compute a bound was
  associated, and a mutant was tried against a fraction of the tests that
  catch it. Without this, `cb` ignoring `alpha_ci` "survived", yet fails
  12 Kaplan-Meier option checks. (A suite that caches results across
  tests hides mutants the same way from any coverage-based selection.)
- `MUTATION_CASES` keeps only the conformance tests of the module's own
  cases.

## Test selection

For `nonparametric`: `surpyval/tests/univariate/nonparametric`,
`surpyval/tests/reference/test_nonparametric.py`,
`surpyval/tests/properties/test_nonparametric.py` (the default "fast"
hypothesis profile), `surpyval/tests/mutation/test_nonparametric_kills.py`
and the conformance suite with `-m "not slow"`, restricted to the
Kaplan-Meier, Nelson-Aalen, Fleming-Harrington and Turnbull cases: 1061
tests, 98 s in one process. Doctests are not included (they are part of
the documentation build).

## Results

Score = (killed + timeout) / mutants run. "Before" is the suite as it
was; "after" adds `surpyval/tests/mutation/test_nonparametric_kills.py`
(written from the survivors, checked with `recheck.py`).

### nonparametric, 2026-09-28, commit 85ed4c5

mutmut 3.8.0, 2 workers. 3059 mutants; 357 of the 457 mutants of
`band` and `_band_critical_value` were not run (each costs about 47 s of
tests; 97 + a seeded sample of 50 of `band`'s 229 and a sample of 50 of
`_band_critical_value`'s 325 were), so 2702 were. No timeouts.

| file | mutants run | killed before | score before | killed after | score after | left: message / cosmetic / equivalent / dead code / pinned bug |
|---|---|---|---|---|---|---|
| nonparametric.py | 2365 | 1824 | 77.1% | 2087 | 88.2% | 153 / 16 / 94 / 14 / 1 |
| kaplan_meier.py | 75 | 60 | 80.0% | 66 | 88.0% | 0 / 0 / 4 / 0 / 5 |
| nelson_aalen.py | 64 | 50 | 78.1% | 57 | 89.1% | 0 / 0 / 7 / 0 / 0 |
| fleming_harrington.py | 198 | 175 | 88.4% | 184 | 92.9% | 0 / 0 / 14 / 0 / 0 |
| total | 2702 | 2109 | 78.1% | 2394 | 88.6% | 153 / 16 / 119 / 14 / 6 |

Without the equivalent and dead-code mutants in the denominator the
score after is 93.2% (2394 / 2569). "Message" survivors change only the
text of an error message; "cosmetic" ones a plot's title, labels or
marker style. The 12 "killed after" mutants of the estimators'
`__init__` (the model name) are not a gap in the suite: the singletons
are built at import, before mutmut's forked worker switches the mutant
on, so mutmut cannot test them; in a fresh process (`recheck.py`) every
test kills them.

Time: about 1 h 40 min of mutmut in three runs (the first, with the
caches, stopped at 2160 mutants; the second reran every mutant it had
not killed; the third ran the rest and the samples), then 26 min for
`recheck.py` on the 593 survivors. A full run of the module is estimated
at 2.5 to 3.5 hours with 2 workers, a third of it in the band mutants.
