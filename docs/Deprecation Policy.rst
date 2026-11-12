Deprecation Policy
==================

In short: when a public name in SurPyval is renamed or removed, your code
keeps working for at least one release (from 1.0, at least two minor
releases and six months; see `From version 1.0`_) and warns you,
naming what to use instead and the release that removes the old name. After that release,
the old name fails with an error that still names the replacement. Every
deprecation and removal is listed in the :doc:`changelog`.

What is public
--------------

- The names you can import from ``surpyval`` (``surpyval.__all__``, which
  is what ``from surpyval import *`` gives).
- The names of the subpackages the documentation imports from:
  ``surpyval.recurrent``, ``surpyval.degradation``,
  ``surpyval.multivariate``, ``surpyval.ml``, ``surpyval.life_models``,
  ``surpyval.metrics``, ``surpyval.diagnostics``, ``surpyval.bayes`` (and
  ``surpyval.prior``), ``surpyval.test_planning``, ``surpyval.datasets``,
  ``surpyval.pymc``, ``surpyval.utils`` and the four
  ``surpyval.univariate`` packages (``parametric``, ``nonparametric``,
  ``regression`` and ``competing_risks``), and
  ``surpyval.univariate.regression.kinds``, the names of the regression
  families. Where a namespace has an ``__all__``, that list is what it
  promises.
- The documented methods, attributes and arguments of those classes and
  functions.

Not public, and free to change in any release without a warning:

- anything whose name starts with an underscore (``_handled_xcnt_to_xrd``,
  ``surpyval._autograd_special``), and any name in ``surpyval.utils``
  that its ``__all__`` does not list;
- ``surpyval.beta``, the tier for models whose interface has not settled
  yet (it is empty at present);
- the test suite, ``surpyval.tests``.

How a name is retired
---------------------

1. **Deprecated.** The old name still works and warns: a
   ``DeprecationWarning`` that names the new name and the release that
   removes the old one, for example

   .. code-block:: text

       DeprecationWarning: AalenAdditive_.fit: 'coef_penalty' is
       deprecated and will be removed in v0.28; use 'penalizer'.

   The changelog lists it under **Deprecated**.
2. **Removed**, in the next release. A few deprecations are given longer,
   each with its removal release stated in the warning
   (``CoxPH.check_ph``, deprecated in 0.26, goes in 0.28). Some are
   deprecated by date rather than by release, with a ``FutureWarning``
   that gives the date: the top-level data-format functions
   (``surpyval.fs_to_xcnt`` and the rest) and ``surpyval.beta.ml`` (now
   ``surpyval.ml``) go in the first release on or after 12 April 2027.
   The changelog lists each removal under **Removed**.
3. **A hint after removal.** For a release or two after it is removed, the
   old name raises the error Python would raise anyway (an
   ``AttributeError``, ``TypeError`` or ``ImportError``) with the release
   that removed it and its replacement in the message, so the traceback
   says what to change:

   .. code-block:: text

       AttributeError: module 'surpyval' has no attribute
       'RecurrentEventData': it was removed in v0.26; it is
       surpyval.EventHistoryData (from surpyval import EventHistoryData)

   After that, the hint is retired and the old name gets Python's plain
   error.

An argument that a call by position misreads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where an argument given by position is silently read as something other
than what that place holds in SurPyval's other methods of the same name
(``Parametric.mean(5)`` takes the 5 as ``defective``, where a
non-parametric model's ``mean(5)`` takes it as ``tau``), the call warns
with a ``FutureWarning``, not a ``DeprecationWarning``: the risk is a
wrong number, not an error, so the warning is shown wherever the call is
made. It says what the value was taken as and the keyword to use.
Passing it by name silences it, and the result is unchanged. The
argument stays positional: making it keyword-only, or refusing the call,
would be a breaking change, and is not made before 2.0.

A test in the package keeps a list of every public name and fails when
one disappears without going through these steps.

Seeing the warnings
~~~~~~~~~~~~~~~~~~~

Python hides a ``DeprecationWarning`` unless it is raised by code run
directly as a script or in a notebook. pytest shows them; to make your
tests fail on them, and so find every old name before it is removed, run

.. code-block:: bash

    python -m pytest -W error::DeprecationWarning -W error::FutureWarning

Results that may change
-----------------------

Numbers can differ in their last digits between releases, platforms and
versions of numpy and scipy, within the tolerance the optimiser or the
numerical integration works to. That is not announced. A deliberate change
to a result (a fix, a corrected formula, a different default) is listed
in the changelog under **Changed** or **Fixed**, with how much the numbers
move where that is known.

Before version 1.0
------------------

SurPyval is not yet at version 1.0, and two things follow from that:

- **Releases come often**, sometimes days apart, and each takes the next
  minor number (0.26, 0.27, ...). One release of deprecation can therefore
  be short. Read the changelog's **Deprecated** and **Removed** entries
  before upgrading, and pin the version (``surpyval==0.27``) where an
  upgrade must not surprise you.
- **A change of behaviour that cannot carry a warning** can come in any
  release, listed under **Changed**: a corrected result, a changed
  default, or a convention made consistent across the package. For
  example, 0.27 gave the distributional regression's coefficients the
  sign the rest of the package uses (a positive coefficient means a
  shorter life), and converts models saved by 0.26 as they load.

These rules apply until 1.0 is released.

From version 1.0
----------------

From 1.0, SurPyval follows `semantic versioning <https://semver.org>`_:
a release numbered 1.x never breaks code written for an earlier 1.x
release. A breaking change waits for 2.0.

- **What the promise covers** is the public interface set out in
  `What is public`_ above, together with the layout of the dictionaries
  that ``to_dict`` writes (see :doc:`surpyval.serialisation`). It also
  covers how a custom distribution is written: its cumulative hazard
  uses ``autograd.numpy``, and a 1.x release will not ask for a
  different array library.
- **A breaking change** is one that makes code using that interface fail
  or mean something different: removing or renaming a public name, an
  argument or a keyword; changing what an argument given by position is
  read as; changing a default; changing the type or shape a public
  function returns; removing a key from, or renaming a key in, a saved
  model's dictionary, or no longer reading a dictionary an earlier
  release wrote.
- **Deprecation in 1.x.** A public name, argument or keyword can still be
  deprecated in a 1.x release, following `How a name is retired`_: it
  keeps working, with a warning that names its replacement, for at least
  two minor releases *and* at least six months, whichever is later. A
  deprecated name is removed only in 2.0 or later.
- **Not breaking, and so allowed in any 1.x release:**

  - a bug fix that corrects a number (a wrong formula or
    bound), listed in the changelog under **Changed** with how much
    the numbers move;
  - the small differences in the last digits described in
    `Results that may change`_;
  - raising the oldest supported version of Python or numpy (and of the
    other required packages), stated in the changelog;
  - new names, new optional arguments with defaults that keep the old
    behaviour, and new keys in a saved model's dictionary;
  - anything not public, as listed above.

``surpyval.beta`` is outside the promise in 1.x as it is now: a model
leaves it for the stable namespaces once its interface has settled.
