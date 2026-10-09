Contributing
============

If you want to contribute to SurPyval, please do! Please review the current open `feature requests
<https://github.com/derrynknife/SurPyval/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement>`_ to see if your desired feature is in the requests. If not, please raise a new one to notify the community. We can assign the feature to you to branch and develop.

Setting up
----------

SurPyval supports Python 3.11, 3.12 and 3.13. From a clone of the repository,
install the package in editable mode with the test and development tools, and
the pre-commit hooks:

.. code-block:: bash

    pip install -r requirements_dev.txt   # installs -e .[tests] plus the tools
    pre-commit install

Code style is enforced rather than requested. The pre-commit hooks
(``.pre-commit-config.yaml``) run isort, pyupgrade (Python 3.11+ syntax),
black (line length 79), flake8 and mypy on every commit, and the lint job in
continuous integration runs ``flake8``, ``isort --check-only`` and ``black
--check`` on the ``surpyval`` package, ``conftest.py`` and ``scripts/``, and
``mypy`` on the package. flake8 also caps each function's McCabe complexity
at 25 (``max-complexity`` in ``pyproject.toml``): a function over it is split
into named steps. mypy reports a ``type:
ignore`` that silences nothing, a redundant cast and an ``==`` between types
that cannot be equal; an ignore needed under one Python's numpy stubs but not
another's carries the ``unused-ignore`` code as well. mypy is strict about
annotations: every function in the package must be type annotated
(``disallow_untyped_defs``); only the tests, ``conftest.py`` and ``scripts/``
are exempt.

To run the tests as continuous integration does:

.. code-block:: bash

    python -m pytest -n auto --run-ml
    python -m pytest --doctest-modules surpyval --ignore=surpyval/tests

The first line is the test suite (``-n auto`` spreads it over your cores, and
``--run-ml`` includes the slow survival tree and forest tests, which are
skipped by default). The second executes the ``>>>`` examples in the
docstrings and checks their printed output, so a docstring example is a
tested promise like any other. ``--run-invariants`` opts in to a slower
combinatorial sweep of the parametric fitting paths, worth running after
changing a likelihood, an initial guess or an optimiser.
``--run-scenarios`` runs the practitioner scenario cards (see below).
``--run-calibration`` runs the statistical calibration studies (confidence
interval coverage, test size and power, estimator bias; about 20 minutes on
four cores), which the release pull request's full run also runs, on
3.12. They include ``test_refit_registry.py``, which draws data from every model in the conformance registry that can
simulate from itself and checks that the refits recover it (#397); a newly
registered model must be added to its ``PLANS`` or ``EXCLUDED``.
The property-based tests in ``surpyval/tests/properties`` run a short
derandomized search by default (under a minute);
``SURPYVAL_HYPOTHESIS_PROFILE=nightly`` makes it thorough (400 examples a
test). When one finds a failure, it prints a minimal example: pin it in
``surpyval/tests/properties/test_known_failures.py`` with the issue number.

Describe any change a user would notice in ``docs/changelog.rst``, under the
unreleased version at the top.

Where a test goes
-----------------

The tests are organised by feature. ``surpyval/tests`` follows the package
(``univariate/parametric``, ``univariate/regression``, ``recurrent``,
``degradation`` and so on), and each module in it covers one model,
estimator or behaviour: ``test_turnbull.py``, ``test_mcf.py``,
``test_information_criteria.py``. **A fix's regression test goes in the
test module of the feature it fixes, with the issue number in the test's
name** (``test_issue_310_lognormal_no_longer_runs_away``), so that the next
person to change the feature finds it beside the others. Do not start a
module for a round of fixes or a piece of work (``test_parametric_fixes3.py``,
``test_serialisation_round4.py``, ``test_tvc_phase2.py``): record the round
in the commit message and the changelog instead. Start a new module only for
a feature that has none yet. A data maker or helper that more than one
module needs goes in ``surpyval/tests/_helpers.py`` rather than being
copied. ``surpyval/tests/review`` and ``surpyval/tests/mutation`` keep the
per-module layout described below.

The conformance suite
---------------------

``surpyval/tests/conformance`` checks every public model against the same
battery of general properties. The models are listed once, in
``registry.py``; the property modules beside it run each registered model
through each property that applies to it:

- identities between its functions: ``sf + ff = 1``, ``Hf = -log sf``,
  ``df = hf sf`` (and the discrete analogue), ``qf(ff(x)) = x``, and the
  cumulative incidences of the causes summing to the all-cause ``ff``;
- vectorisation: a scalar, a 2-D and an empty query, a permuted query, and
  covariate rows evaluated together or one at a time;
- invariances of the fit: a change of time unit, a permutation of the data
  rows, counts ``n`` against the same rows repeated, and a covariate
  column multiplied by a constant (by ``fit`` and ``fit_tvc``, at scales
  from 1e-6 to 731: the likelihood reached must not change);
- valid values: probabilities in [0, 1] and monotone in time, no NaN at a
  valid time;
- that the derivatives a fit or its inference takes -- the gradient and
  Hessian of its likelihood in its search space, by autograd or the
  model's own, and the delta-method gradients behind a parametric ``cb``
  -- agree with finite differences at the fit (``test_derivatives.py``;
  the model classes that differentiate are listed in ``DIFFERENTIATED``,
  and the cases of those that do not in ``NOT_DIFFERENTIATED``);
- the missing-value rule (see :doc:`Conventions`), seed reproducibility of
  the random draws, a strict-JSON ``to_dict`` / ``from_dict`` round trip
  that keeps every prediction, and a ``pickle`` round trip of the fitted
  model and of each alternate fit path's (``test_pickle.py``);
- that the alternate ways of fitting a model (``fit_from_df``, a formula,
  ``from_params``, ``fit_tvc`` ...) agree with ``fit``, and, for a model
  class that declares its attributes (``DECLARED_ATTRIBUTES``), that every
  way of building it gives it the same attributes, each declared on the
  class (``test_attributes.py``);
- every option of every confidence bound, ``interp=`` value and estimation
  option (``test_options.py``), behaviour outside the data
  (``test_outside_data.py``), that a fit which cannot converge says
  so (``test_convergence.py``), and that a covariate column the data
  cannot determine is aliased, not given an arbitrary coefficient
  (``test_aliasing.py``; a model with covariates declares its
  ``coefficients``);
- that a fit which maximises a likelihood says what it reached
  (``test_maximum.py``): its model's ``maximum`` is ``"verified"``,
  ``"unverified"`` or ``"no finite maximum"``, it warns exactly when that
  is not a verified maximum, and a verified maximum has a zero gradient and
  a positive-definite Hessian of the likelihood at the reported
  parameters. **A new likelihood fitter must set** ``maximum``
  (``surpyval.utils.no_maximum.MAXIMUM_STATES``), from a check of its
  answer -- ``is_local_minimum``, or ``verify_or_polish`` for a search
  whose answer may need polishing -- with ``warn_unverified`` or
  ``warn_no_maximum`` where it is not a verified maximum, and pass this
  property (a family's likelihood goes in ``SEARCHES`` there; a fit that
  is not a likelihood maximisation is excluded with the reason);
- that no raw numpy, scipy or autograd warning escapes the package, and
  each deliberate warning appears once (``test_warnings.py``);
- that every model spells the model-comparison values alike
  (``neg_ll()``, ``aic()``, ``aic_c()``, ``bic()`` and ``covariance()``
  methods, ``log_likelihood`` a value; ``test_comparison.py``), that each
  name a report reads is the same kind -- a method or a value -- on every
  model, and that every full-likelihood fit can be ranked by ``aic`` and
  ``bic`` (``test_surface.py``).

``test_completeness.py`` walks the public namespaces and fails for any public
class or fitter that is neither registered nor listed in ``OUT_OF_SCOPE``
with a reason. So **a new model is registered**: add a ``Case`` in
``registry_cases.py`` (the family helpers in ``registry_families.py`` --
``continuous``, ``regression`` and the rest -- do most of it), giving a small
deterministic fixture (``registry_fixtures.py``), the fit, how its functions
are called, and its alternate fit paths. ``registry.py`` gathers them,
applies the known failures and holds ``OUT_OF_SCOPE``; import from it. If a
property cannot hold for it, exclude it in ``exclude`` with the reason (a
step function has no density, a point mass no quantile inverse). If it should
hold and does not, that is a bug: list it in ``KNOWN_FAILURES``
(``registry_known_failures.py``) with a one-line
description, which makes it a strict xfail -- the suite stays green, and
turns red the day the bug is fixed, as the reminder to remove the entry.
Only a failure whose outcome depends on the numpy / scipy build (an
optimiser started far from the maximum) is listed in ``NON_STRICT`` as
well, so that either outcome passes.

The fast form leaves out the refits marked ``slow`` (the less common
variants of families whose main member runs); the full suite includes those:

.. code-block:: bash

    python -m pytest surpyval/tests/conformance -m "not slow"   # ~40 s
    python -m pytest surpyval/tests/conformance                 # everything

**When a bug is found, add the property, not only the test.** Each fix gets a
regression test for its own case; ask as well which general property the bug
broke, and if the battery does not check it yet, add it to the property
modules, so every registered model is checked for it from then on.

The properties enforce the package's :doc:`Design Principles`: the rules every
model keeps, each listed with the tests that check it. Review a change against
that list, and when a bug breaks a principle its check missed, extend the
check.

Practitioner scenario cards
---------------------------

The conformance suite checks that each model keeps its contract; it does
not check that the models add up to an answer. ``surpyval/tests/scenarios``
does: each module is a *card*, an end-to-end study a reliability engineer
runs, on data simulated from a known truth in the shape it arrives in the
field -- a maintenance system's date table with inspections and a records
start date, a Nevada chart of warranty returns, the readouts of an
accelerated test, a growth test of several prototypes. A card names a
persona and the questions the standard references ask of that kind of
study (Meeker and Escobar, Nelson, Abernethy, MIL-HDBK-189C, IEC 61508),
then answers them through the public API only.

Each answer is checked against an *oracle*, never against the package
itself:

- the truth the data were simulated from, inside the fit's own
  confidence interval (each card has a fixed seed, so this is
  deterministic);
- an independent likelihood written in the card with numpy and scipy
  (``scenarios/_oracles.py``): the maximum the package reports must be the
  maximum;
- an equivalent parameterisation: the same model written two ways (an
  accelerated life model and an AFT model on transformed stresses; a
  covariate in two units) must reach the same likelihood.

A question the package cannot yet answer without the analyst writing the
method by hand is a strict xfail whose reason is led by its issue. When
the issue is fixed the test passes and the strict xfail turns the run red:
update the card to the API the fix chose, with an oracle for the answer.
A gap whose API cannot be guessed is written in the card's docstring with
its issue instead. (The first cards found fourteen such gaps -- a bound
on the demonstrated MTBF of a growth test, a B10 by operating condition,
a forecast from each unit's own repair history, among others; #570-#583.
Each is now answered, and checked, through the API its fix chose.)

The cards take seconds with ``--run-scenarios``, and the release pull
request's full run runs them on 3.12. Add a card
for a kind of study the package claims to support but no card covers;
its first run is a practitioner review of that part of the package, and
the issues it files become its xfails.

Proving a refactor changed nothing
----------------------------------

A refactor (moving, merging or splitting code) must not change what the
package computes or says. ``scripts/refactor/snapshot.py`` records a
snapshot of it before the change and another after, and compares them:

.. code-block:: bash

    python scripts/refactor/snapshot.py record /tmp/before.json
    # ... make the change ...
    python scripts/refactor/snapshot.py record /tmp/after.json
    python scripts/refactor/snapshot.py compare /tmp/before.json /tmp/after.json

A snapshot holds, for every case in the conformance registry: the fitted
parameters, ``neg_ll``, ``aic`` and ``bic``, every function at the
registry's query, every confidence bound the case declares (each side and
``on=``), ``to_dict()``, ``repr`` and ``summary()``, the warnings each call
raises, each alternate fit path, and the error each of a corpus of invalid
inputs raises. It adds the fits the registry does not reach (time-varying
covariates, seeded bootstraps, recurrent data mixing every kind of
censoring), the public API (names, signatures and defaults), the modules
``import surpyval`` loads, and the IDs of the collected tests and
doctests. It takes two to three minutes on two cores (``-j`` sets the
workers; ``--full`` adds the likelihood-ratio bounds the conformance suite
runs only in the full suite, about two minutes more).

Floats are stored exactly, so ``compare`` is bit-exact unless given
``--rtol``. A pure move or merge must compare clean. Splitting a function
into steps may change results by at most ``--rtol 1e-12``, with the reason
for each difference given in the pull request. A change that is meant to
alter the API (a removed name, a new method) shows only in the ``api``
section; say so. The snapshot depends on the numpy and scipy build, so
compare snapshots recorded in the same environment, and with the same
version of the script (copy it aside if the change edits it). The snapshots
themselves are not committed.

Mutation testing
----------------

Coverage shows that a line ran, not that a test would notice if it were
wrong. ``scripts/mutation/run.sh <module>`` (mutmut; see
``scripts/mutation/README.md``) changes a module one small edit at a time --
``<`` to ``<=``, a dropped argument, ``side="right"`` removed -- in a copy of
the repository, and reruns the tests that reach the changed function; an
edit no test notices is a *surviving mutant*. It takes hours (2.5 to 3.5 for
the non-parametric module on two workers), so run it after a large change to
a module or before a release, not on a pull request, and record the score in
the README.

Triage every survivor: a plausible bug no test would notice gets a test --
preferably a conformance property that covers every model, else one in
``surpyval/tests/mutation/test_<module>_kills.py``; a change with no
observable effect is *equivalent*; code whose mutants can never be observed
is a simplification candidate; a survivor that shows a bug is pinned as a
strict xfail with its issue number. ``scripts/mutation/recheck.py`` checks
new tests against the survivors in minutes. A fixture that caches results
across tests hides mutants from mutmut: the plugin clears the conformance
caches, and a new cache needs the same.

Reviewing a module by bug class
-------------------------------

Tests find what someone thought to check; a review looks for what nobody
did. Review a module against the kinds of bug this package has actually
had, choosing modules by lowest branch coverage and by how often they have
been fixed (``git log --follow -p <file>``). Run

.. code-block:: bash

    python -m pytest --cov=<package path> --cov-branch \
        --cov-report=term-missing <its tests>

and read the uncovered branches first. Then, for each public function and
each entry point that reaches it, try:

1. **Ties**: an event and a censoring at the same time; events tied among
   themselves; an interval endpoint equal to an exact time.
2. **Order**: unsorted rows; unsorted or duplicated queries.
3. **Endpoints**: the first and last piece, a query exactly on a boundary,
   a start before or at 0, x = 0.
4. **Degenerate values**: NaN, inf and empty input, in the data and in the
   queries; probabilities outside [0, 1].
5. **Tails and scale**: 1e-6 and 1e6 times the natural scale; ``log_*``
   functions against the logs of the plain ones; a special case (e.g. an
   exponentiated Weibull with mu = 1) against its parent distribution.
6. **Shapes**: scalar, 1-D, 2-D and empty queries; array parameters where a
   docstring allows them.
7. **Counts**: ``n = k`` against k repeated rows.
8. **Truncation**: ``tl`` / ``tr`` at, just inside and outside an
   observation.
9. **Reference software**: R ``survival``, ``cmprsk``, lifelines and
   scikit-survival, minding each one's reporting convention.
10. **Entry points**: ``fit``, ``fit_from_df``, ``from_dict`` / JSON and the
    ``fit_from_*`` helpers.
11. **User-supplied names and labels**: collisions with attribute names;
    tuple and mixed labels.
12. **Messages**: every error names the argument (principle 2), and no raw
    numpy warning escapes (principle 22).

A suspected bug counts only with a numerical reproduction. Pin it as a
strict xfail whose reason starts with its issue number in
``surpyval/tests/review/test_<module>_review.py``, and where it breaks a
general property, add that property to the conformance suite so every
model is checked for it.

Branching and releases
----------------------

SurPyval uses a two-tier branch model to keep continuous integration and the
documentation build from running on every change:

* **master** is the release branch. It is only updated at a version release,
  and pushing a ``v*`` tag to it publishes the package and rebuilds the hosted
  documentation.
* **develop** is the long-lived integration branch. Feature work is done on a
  short-lived branch and opened as a pull request into ``develop``.
* At release time ``develop`` is merged into ``master`` in a single pull
  request and the new version is tagged.

Versions have two parts, ``MAJOR.MINOR`` (``0.22``, tagged ``v0.22``), since
0.22; earlier releases had three. Every release, fixes only or not, takes the
next minor number. A name deprecated in one release is removed in the next
(``REMOVED_IN`` in ``surpyval/utils/deprecation.py``), and a test fails once
the version reaches it while the old names are still accepted.

Continuous integration (``.github/workflows/actions.yml``) therefore runs
once, on the **pull request into master** (the release), and on nothing
else: not on pull requests into ``develop`` and not on pushes to any branch.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event
     - Jobs
   * - Pull request into ``develop``
     - none
   * - Pull request into ``master`` (the release)
     - lint, the conformance suite, the test suite across three
       interpreters (each split into four parts that run at once, with
       the docstring examples; 3.12 adds the calibration studies, the
       scenario cards and coverage), the wheel check and the
       documentation build
   * - Merge into ``master``
     - none in this repository; Read the Docs rebuilds the hosted
       documentation
   * - Push of a ``v*`` tag
     - ``.github/workflows/publish.yml`` checks that the tagged commit is
       the merge of a pull request into ``master`` whose full run passed
       (every job), and that the tag matches the version in
       ``pyproject.toml``; it then builds the package and publishes it to
       PyPI. It runs no tests of its own. Read the Docs builds the tagged
       documentation

The full test suite and the documentation build run once per release, on the
release pull request. That run tests the pull request's merge result, so a
failure is found before ``master`` changes: fix it on ``develop`` (the
pull request picks the fix up) and merge the release when the run is green.
Waiting on the full suite at every step made a release take hours.

Every change is tested locally by the tests for what it changed, before it is
pushed:

* the test modules covering the code touched, and the conformance tests for
  the models it touches (``surpyval/tests/conformance``, filtered with
  ``-k``);
* the doctests of the modules touched
  (``python -m pytest --doctest-modules <module>``);
* lint, formatting and ``mypy``, as the lint job runs them;
* the documentation pages that call what changed, when a public function's
  behaviour changes (documentation cells call the real API);
* for a numerical change, its tests on more than one interpreter.
  ``scripts/check_all_pythons.py`` runs the whole suite on 3.11, 3.12 and
  3.13; the environments it keeps in ``.venvs/`` (git-ignored) run any
  selection:

  .. code-block:: bash

      python scripts/check_all_pythons.py 3.11   # create .venvs/py3.11 once
      .venvs/py3.11/bin/python -m pytest surpyval/tests/univariate/parametric

Documentation
-------------

The documentation executes its own code examples when it is built. Code in
``.. jupyter-execute::`` directives is run in a Jupyter kernel during the
Sphinx build, and the text output and matplotlib figures are embedded in the
rendered pages. This means the examples and images never go stale — they
always reflect the installed version of SurPyval — and an example that no
longer runs will fail the documentation build.

To build the documentation locally:

.. code-block:: bash

    pip install -e ".[docs]"
    sphinx-build -b html -W --keep-going docs docs/_build/html

``-W`` turns warnings into errors, which is how continuous integration and
Read the Docs both build. A Sphinx warning is rarely cosmetic -- a broken
cross-reference silently renders as plain text, a mistyped ``autoclass`` path
drops the class from the page entirely -- so the build is kept at zero
warnings rather than at "succeeded, N warnings". ``--keep-going`` reports all
of them instead of stopping at the first.

When writing documentation, prefer ``.. jupyter-execute::`` over static
``.. code-block:: python`` blocks with pasted outputs or screenshots. All
cells in a page share one kernel, so later cells can use variables defined
in earlier ones. Anything a cell writes to stderr fails the build, so cells
must run without warnings: fix the example (better data or arguments) rather
than hide a warning. Only when the warning is itself the point being taught,
add the ``:stderr:`` option so it is rendered in the page. Seed any
randomness, so the numbers quoted in the text are the numbers the build
prints.

Checking the numbers quoted in the text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cell's output is regenerated on every build, but a number quoted in the
prose around it ("a shape of about 2.1", "83 wear-out failures", "the lower
AIC") is not, and it goes stale silently when the output changes. So every
such claim is checked by a hidden cell after the paragraph that makes it:

.. code-block:: rst

    The shape :math:`\beta \approx 2.1` is greater than one, ...

    .. jupyter-execute::
        :hide-code:
        :hide-output:

        assert round(model.params[1], 1) == 2.1, model.params

The cell runs in the page's kernel like any other and renders nothing; a
failing ``assert`` stops the build with its traceback. Some conventions:

- Check the claim at the precision the text states it: ``round(x, 1) == 2.1``
  for "about 2.1", a comparison for "lower", "wider" or "inside the interval".
- Give the value as the assertion message, so a failure shows what the output
  now is.
- Prefix names used only by a check with ``_``, and never change a variable a
  later cell uses. If a claim needs a value that a visible cell computed but
  did not keep (inside a loop, say), keep it in that cell (``aic[df] = ...``)
  rather than repeating an expensive computation in the check.
- Inputs (the parameters a simulation was drawn from), definitions and
  literature values need no check.
- When a check fails because an output changed, correct the text to the new
  output rather than loosening the check.

To iterate on a page, rerun the build command above: Sphinx keeps its doctree
cache in ``docs/_build/html/.doctrees``, so after the first full build a
rebuild re-reads, and so re-executes, only the pages that changed.
