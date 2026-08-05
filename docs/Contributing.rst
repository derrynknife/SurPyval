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
than on every push to every branch. The lint and test jobs run on both; the
documentation build runs on the release pull request into ``master`` only,
where it reproduces the hosted build. Read the Docs itself is configured to
build ``master`` and tags only.

The documentation build is gated at the release rather than on every pull
request because it executes every code cell in the documentation, which takes
around three minutes from cold rather than the seconds a lint job costs. The
trade-off is that a change which breaks a
documentation example is caught when the release is prepared rather than when
it is merged into ``develop`` -- so if you change the behaviour of a public
function, it is worth building the docs locally before opening the pull
request.

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
    sphinx-build -b html docs docs/_build/html

When writing documentation, prefer ``.. jupyter-execute::`` over static
``.. code-block:: python`` blocks with pasted outputs or screenshots. All
cells in a page share one kernel, so later cells can use variables defined
in earlier ones. If a cell intentionally emits a warning, add the
``:stderr:`` option so the warning is rendered in the page rather than
failing the build log.


