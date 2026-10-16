# CLAUDE.md

Pointers for an agent working in this repository. The documents below are
the sources of truth; read the one for your task rather than relying on
this summary, and fix them, not this file, when they are wrong.

- `docs/Contributing.rst`: setup, where a test goes, the conformance suite,
  scenario cards, refactor snapshots, mutation testing, branching, releases
  and the docs build.
- `docs/Conventions.rst`: the data model (`x`, `c`, `n`, `t`), censoring
  codes, missing values, units, seeds, serialisation.
- `docs/Design Principles.rst`: the numbered rules every model keeps and
  the tests that check each one. Review a change against it.
- `DECISIONS.md`: design decisions and who made them. Check the principles
  before asking the maintainer; record a decision you make here.
- `ROADMAP.md`: the larger ideas, and what has been decided against.
- `MODEL_ATLAS.md`: a map for placing a new model; not a plan.
- `.claude/skills/surpyval/SKILL.md`: how to *use* the package.

## Branches and releases

- Branch from `dev`; open pull requests into `dev`. Never into
  `main`.
- A release is one pull request `dev` -> `main`, then the publish
  workflow run on `main`
  (`gh workflow run publish.yml --ref main -f version=X.Y`), which
  publishes, tags and creates the GitHub Release.
- CI runs only on the pull request into `main`. Pull requests into
  `dev` get no checks, so test locally before pushing.

## Running things

Python 3.11, 3.12 and 3.13 are supported (`requires-python = ">=3.11"`).
Do not use 3.12-only syntax. `pip install -r requirements_dev.txt`.

```bash
python -m pytest -n auto --run-ml                         # full suite
python -m pytest -n auto --run-calibration surpyval/tests/calibration
python -m pytest --run-scenarios surpyval/tests/scenarios
python -m pytest surpyval/tests/conformance -m "not slow" # ~40 s
python -m pytest surpyval/tests/conformance               # everything
python -m pytest --doctest-modules surpyval --ignore=surpyval/tests
python -m sphinx -b html -W --keep-going docs docs/_build/html  # needs .[docs]
black --check surpyval conftest.py scripts
isort --check-only surpyval conftest.py scripts
flake8 surpyval conftest.py scripts
mypy surpyval
```

The calibration studies take about 20 minutes on four cores. For a change,
run the tests for what it touches (`-k` on the conformance suite), its
doctests and lint; `scripts/check_all_pythons.py` covers numerical changes
on every interpreter.

Known failures on `dev` (checked 2026-10-17): four `test_541_*` tests
near rho = +/-1 in `surpyval/tests/multivariate/test_copula_verification.py`
(`elliptical_density_near_rho_one` and `gaussian_cdf_and_h_function_near_rho_one`,
each at 0.99999999 and -0.999999999999). They are not caused by your change.

## Rules that are easy to miss

- **Simulation and calibration tests are deterministic.** Seeds are fixed
  before the result is seen. Never change a seed, a tolerance or a sample
  size to make a test pass. A marginal or failing result is reported as it
  is, with the numbers, and investigated.
- **Scope.** SurPyval is a general statistics package for anyone with
  time-to-event data, not only engineers. Reliability test planning belongs
  here (`surpyval.test_planning`). Physics of failure does not.
- **Every bound.** Each model aims to offer every bound type (`cb` methods
  and `on=` targets, likelihood-ratio, bootstrap). A missing one is a
  documented gap: the call refuses and says why and what to use instead,
  and the conformance registry declares only the `Bound`s a model has.
- **Refusals say what to do.** An error names the argument and the fix; a
  warning says what happened, with counts, and what to do about it
  (principles 2 and 22).
- A new public model is registered in the conformance suite, or
  `test_completeness.py` fails.
- Describe a user-visible change in `docs/changelog.rst` under the
  unreleased version.
