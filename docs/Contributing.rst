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
continuous integration runs ``flake8``, ``mypy`` and ``black --check`` on the
``surpyval`` package. mypy is strict about annotations: every function in the
package must be type annotated (``disallow_untyped_defs``); only the tests and
the ``surpyval.alpha`` tree are exempt.

To run the tests as continuous integration does:

.. code-block:: bash

    python -m pytest -n auto --ignore=surpyval/tests/alpha --run-ml
    python -m pytest --doctest-modules surpyval --ignore=surpyval/tests --ignore=surpyval/alpha

The first line is the test suite (``-n auto`` spreads it over your cores, and
``--run-ml`` includes the slow survival tree and forest tests, which are
skipped by default). The second executes the ``>>>`` examples in the
docstrings and checks their printed output, so a docstring example is a
tested promise like any other. ``--run-invariants`` opts in to a slower
combinatorial sweep of the parametric fitting paths, worth running after
changing a likelihood, an initial guess or an optimiser.
``--run-calibration`` runs the statistical calibration studies (confidence
interval coverage, test size and power, estimator bias; about 15 minutes on
four cores), which also run nightly against ``develop`` from
``.github/workflows/nightly.yml``.

Describe any change a user would notice in ``docs/changelog.rst``, under the
unreleased version at the top.

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
  rows, and counts ``n`` against the same rows repeated;
- valid values: probabilities in [0, 1] and monotone in time, no NaN at a
  valid time;
- the missing-value rule (see :doc:`Conventions`), seed reproducibility of
  the random draws, and a strict-JSON ``to_dict`` / ``from_dict`` round trip
  that keeps every prediction;
- that the alternate ways of fitting a model (``fit_from_df``, a formula,
  ``from_params``, ``fit_tvc`` ...) agree with ``fit``.

``test_completeness.py`` walks the public namespaces and fails for any public
class or fitter that is neither registered nor listed in ``OUT_OF_SCOPE``
with a reason. So **a new model is registered**: add a ``Case`` to
``registry.py`` (the family helpers there -- ``continuous``, ``regression``
and the rest -- do most of it), giving a small deterministic fixture, the fit,
how its functions are called, and its alternate fit paths. If a property
cannot hold for it, exclude it in ``exclude`` with the reason (a step
function has no density, a point mass no quantile inverse). If it should hold
and does not, that is a bug: list it in ``KNOWN_FAILURES`` with a one-line
description, which makes it a strict xfail -- the suite stays green, and
turns red the day the bug is fixed, as the reminder to remove the entry.

Continuous integration runs the suite on every pull request, without the
refits marked ``slow`` (the less common variants of families whose main
member runs); the full suite includes those:

.. code-block:: bash

    python -m pytest surpyval/tests/conformance -m "not slow"   # ~40 s
    python -m pytest surpyval/tests/conformance                 # everything

**When a bug is found, add the property, not only the test.** Each fix gets a
regression test for its own case; ask as well which general property the bug
broke, and if the battery does not check it yet, add it to the property
modules, so every registered model is checked for it from then on.

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

Continuous integration (``.github/workflows/actions.yml``) therefore runs on
**pull requests into develop or master** and on **pushes to master**, rather
than on every push to every branch. Not every job runs on every event:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event
     - Jobs
   * - Pull request into ``develop``
     - lint and the conformance suite (about a minute each)
   * - Pull request into ``master`` (the release)
     - lint, the conformance suite, the test suite across three
       interpreters, and the documentation build (about ten minutes)
   * - Push to ``master``
     - lint, the conformance suite and the test suite; Read the Docs
       rebuilds the hosted documentation
   * - Push of a ``v*`` tag
     - ``.github/workflows/publish.yml`` checks that the tag matches the
       version in ``pyproject.toml``, builds the package and publishes it
       to PyPI; Read the Docs builds the tagged documentation

The test job also runs the docstring examples (twice: once as text, once
forcing a numerical comparison of every number) and reports coverage.

The test suite and the documentation build are both gated at the release
rather than on every pull request because of what they cost: the suite is
roughly nine minutes across the three interpreters and the documentation build
around three from cold, against about one for lint. Paying that on every
feature pull request made the edit-review loop the slowest part of working on
the package, and with a single maintainer running the suite locally before
pushing, the pull-request run was mostly confirming what was already known.

The trade-off is real and worth understanding before you rely on it. A failure
that appears on only one interpreter, or a change that breaks a documentation
example, is now found when the release pull request is opened -- with a
release's worth of commits to search through rather than one. So:

* Run the suite locally before pushing, and across more than one interpreter
  when you have touched anything numerical. ``scripts/check_all_pythons.py``
  does exactly that -- it runs the test suite and the doctests as continuous
  integration would (not lint, which runs on every pull request anyway, and
  not the documentation build), on 3.11, 3.12 and 3.13:

  .. code-block:: bash

      python scripts/check_all_pythons.py                 # all three
      python scripts/check_all_pythons.py 3.12            # just one
      python scripts/check_all_pythons.py --skip-install  # reuse as-is

  It keeps its environments in ``.venvs/`` (git-ignored) and reuses them, so
  only the first run pays for the installs. It uses ``uv`` when that is
  available and falls back to ``venv`` and ``pip`` when it is not.

* Build the documentation locally when you change the behaviour of a public
  function, since documentation cells call the real API.
* On a long-running branch, open the release pull request early and let it sit,
  so the full run has somewhere to fail before the release itself.

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
