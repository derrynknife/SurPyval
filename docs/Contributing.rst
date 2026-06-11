Contributing
============

If you want to contribute to SurPyval, please do! Please review the current open `feature reqeusts 
<https://github.com/derrynknife/SurPyval/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement>`_ to see if your desired feature is in the requests. If not, please raise a new one to notify the community. We can assign you feature for you to branch and develop.

SurPyval is in the process of complying with the PEP8 standard so please make all contributions as per that standard.

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


