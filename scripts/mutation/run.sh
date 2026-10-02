#!/usr/bin/env bash
# Mutation testing of one SurPyval module with mutmut (#396).
#
#     scripts/mutation/run.sh <module>
#
# <module> is one of: nonparametric, turnbull, cox_ph, parametric_fitter,
# competing_risks (see modules() below). The run happens in a copy of the
# working tree, never in the checkout itself -- mutmut rewrites the source
# it mutates -- and leaves its results in $MUTATION_WORKDIR/<module>:
#
#     results.txt   every mutant and its outcome (mutmut results --all)
#     survivors.txt the surviving mutants, one diff each
#     summary.txt   the counts and the score, per file
#
# Environment:
#     MUTATION_WORKDIR  where the copies go (default /tmp/surpyval-mutation)
#     MUTATION_JOBS     parallel mutants (default 2)
#     MUTATION_REV      mutate this commit (git archive) instead of the
#                       working tree, e.g. HEAD while the checkout is
#                       being edited
#     MUTATION_VENV     the venv to use (default $MUTATION_WORKDIR/venv,
#                       created with uv when missing)
#     PYTHON            the interpreter for the venv (default python3.11)
#
# A second run of the same module reuses the copy and mutmut's cache, so
# only mutants whose function changed are tested again (a survivor is not
# retried against new tests: use recheck.py); delete the module's
# directory for a clean run. See README.md for the timings. Do not edit
# this file while it runs: bash reads it as it goes.
set -euo pipefail

MUTMUT_VERSION=3.8.0
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
module=${1:?"usage: run.sh <module> (nonparametric, turnbull, cox_ph, parametric_fitter, competing_risks)"}
workdir=${MUTATION_WORKDIR:-/tmp/surpyval-mutation}
jobs=${MUTATION_JOBS:-2}
python=${PYTHON:-python3.11}

np=surpyval/univariate/nonparametric
cr=surpyval/univariate/competing_risks
conformance=surpyval/tests/conformance

# The files to mutate, the tests to run against them, and the conformance
# cases to keep (empty: all; the plugin drops the other cases' tests).
# mutmut runs only the tests that reach the mutated function, so a broad
# selection costs time mostly in the clean run it starts with.
modules() {
    cases=""
    case "$1" in
    nonparametric)
        mutate="$np/nonparametric.py $np/_support.py $np/_bands.py
                $np/kaplan_meier.py $np/nelson_aalen.py
                $np/fleming_harrington.py"
        tests="surpyval/tests/univariate/nonparametric
               surpyval/tests/reference/test_nonparametric.py
               surpyval/tests/properties/test_nonparametric.py
               surpyval/tests/mutation/test_nonparametric_kills.py
               $conformance"
        cases="KaplanMeier,NelsonAalen,FlemingHarrington,Turnbull" ;;
    turnbull)
        mutate="$np/turnbull.py"
        tests="surpyval/tests/univariate/nonparametric
               surpyval/tests/reference/test_nonparametric.py
               surpyval/tests/properties/test_nonparametric.py
               $conformance"
        cases="Turnbull" ;;
    cox_ph)
        mutate="surpyval/univariate/regression/proportional_hazards/cox_ph.py"
        tests="surpyval/tests/univariate/regression
               surpyval/tests/reference/test_cox.py
               surpyval/tests/properties/test_regression.py
               $conformance"
        cases="CoxPH,CoxPH[strata]" ;;
    parametric_fitter)
        mutate="surpyval/univariate/parametric/parametric_fitter.py
                surpyval/univariate/parametric/optimised_fit.py
                surpyval/univariate/parametric/_fit_inputs.py"
        tests="surpyval/tests/univariate/parametric
               surpyval/tests/reference/test_parametric.py
               surpyval/tests/properties/test_parametric.py
               $conformance" ;;
    competing_risks)
        mutate="$cr/nonparametric/competing_risks.py $cr/aalen_johansen.py"
        tests="surpyval/tests/univariate/competing_risks
               surpyval/tests/reference/test_competing_risks.py
               $conformance"
        cases="CompetingRisks[Nelson-Aalen],CompetingRisks[Kaplan-Meier]" ;;
    *)
        echo "unknown module: $1" >&2
        exit 2 ;;
    esac
}
modules "$module"

dest="$workdir/$module"
copy="$dest/repo"
venv=${MUTATION_VENV:-$workdir/venv}
mkdir -p "$dest"

# The working tree as it is (tracked and untracked, not ignored), so
# uncommitted tests take part, or the commit MUTATION_REV. Only files that
# changed are recopied, and only mutants of changed functions rerun.
mkdir -p "$copy"
if [ -n "${MUTATION_REV:-}" ]; then
    git -C "$root" archive "$MUTATION_REV" | tar -x -C "$copy"
else
    git -C "$root" ls-files -z --cached --others --exclude-standard |
        (cd "$root" && tar --null -T - -cf -) |
        tar -x -C "$copy"
fi
cp "$here/mutmut_plugin.py" "$copy/"

if [ ! -x "$venv/bin/mutmut" ]; then
    uv venv -q -p "$python" "$venv"
    VIRTUAL_ENV="$venv" uv pip install -q -e "$copy[tests]" \
        "mutmut==$MUTMUT_VERSION"
fi
# mutmut puts mutants/ first on sys.path, so the editable install may
# point anywhere; point it at this copy all the same.
VIRTUAL_ENV="$venv" uv pip install -q --no-deps -e "$copy"

{
    echo "[mutmut]"
    echo "source_paths=surpyval/"
    echo "only_mutate="
    for f in $mutate; do echo "    $f"; done
    echo "also_copy="
    echo "    conftest.py"
    echo "    mutmut_plugin.py"
    echo "    stubs/"
    # -m "not slow": the pull-request selection of the conformance suite.
    # The property tests run their default ("fast") profile.
    printf "pytest_add_cli_args=\n    -p\n    mutmut_plugin\n"
    printf "    -p\n    no:cacheprovider\n    -m\n    not slow\n"
    echo "pytest_add_cli_args_test_selection="
    # A path missing from the copy (a test added later) is left out.
    for t in $tests; do
        if [ -e "$copy/$t" ]; then echo "    $t"; fi
    done
} >"$copy/setup.cfg"

cd "$copy"
# A source file newer than its mutated copy is regenerated by mutmut.
start=$(date +%s)
export MUTATION_CASES="$cases"
"$venv/bin/mutmut" run --max-children "$jobs" >"$dest/run.log" 2>&1 || {
    echo "mutmut failed; see $dest/run.log" >&2
    tail -20 "$dest/run.log" >&2
    exit 1
}
echo "mutmut run took $(($(date +%s) - start)) s" | tee "$dest/time.txt"
"$venv/bin/mutmut" results --all true >"$dest/results.txt" 2>&1
"$venv/bin/python" "$here/summarise.py" "$dest/results.txt" \
    "$venv/bin/mutmut" "$dest/survivors.txt" | tee "$dest/summary.txt"
