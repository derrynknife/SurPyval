Contributing
============

If you want to contribute to SurPyval, please do! Please review the current open `feature reqeusts 
<https://github.com/derrynknife/SurPyval/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement>`_ to see if your desired feature is in the requests. If not, please raise a new one to notify the community. We can assign you feature for you to branch and develop.

SurPyval is in the process of complying with the PEP8 standard so please make all contributions as per that standard.

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
than on every push to every branch. Read the Docs is configured to build
``master`` and tags only. The net effect is that the full test suite and the
documentation build run once per pull request and once per release, instead of
once per intermediate commit.

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

    pip install -e .
    pip install -r docs/requirements.txt
    sphinx-build -b html docs docs/_build/html

When writing documentation, prefer ``.. jupyter-execute::`` over static
``.. code-block:: python`` blocks with pasted outputs or screenshots. All
cells in a page share one kernel, so later cells can use variables defined
in earlier ones. If a cell intentionally emits a warning, add the
``:stderr:`` option so the warning is rendered in the page rather than
failing the build log.


