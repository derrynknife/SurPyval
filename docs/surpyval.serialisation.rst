Saving and Loading Models
=========================

A fitted model can be saved and restored later, or in another process,
without refitting. Almost every fitted SurPyval model can be serialised
to a plain dictionary with ``to_dict()`` and written to JSON with
``to_json(path)``. The dictionaries hold only plain Python types, so they
can also go straight into a document store (the round trip through
BSON/MongoDB is tested). What a restored model keeps, and what needs the
original data, is described in :doc:`Conventions`.

Restoring takes one call whichever class wrote the file: the
package-level readers dispatch on the serialised dictionary itself.

.. code:: python

    import surpyval

    model = surpyval.Weibull.fit(x)
    model.to_json("weibull.json")

    restored = surpyval.from_json("weibull.json")  # any model's file
    restored = surpyval.from_dict(model_dict)      # any model's dict

The class-level readers on the fitted-model classes
(``Parametric.from_dict``, ``SemiParametricRegressionModel.from_dict``,
... and the matching ``from_json``) remain available when the model class
is known up front; each rejects a dictionary written by a different class
with a ``ValueError``. The readers belong to the *model* classes, not to
the fitters: ``Weibull.from_dict`` and ``CoxPH.from_dict`` do not exist.

``CompetingRisksProportionalHazards`` serialises the same way, for both
``how="Cox"`` and ``how="Fine-Gray"``; its per-cause optimiser results
(``results``) are not stored.

Some details differ between families:

- Only the univariate ``Parametric`` and ``NonParametric`` models take
  ``to_dict(with_data=True)``, which stores the fitted data as well so
  that ``plot`` and likelihood-ratio bounds (parametric) or
  ``bootstrap_cb`` (non-parametric) work after restoring;
  ``to_json(path, with_data=True)`` writes the same to a file (any
  other model's ``to_json`` refuses ``with_data=True`` with a
  ``TypeError``). The information criteria need no data: a fitted
  univariate or regression model stores their sample size (``"ic_n"``),
  so ``bic`` and ``aic_c`` work on the restored model either way.
- The degenerate ``NeverOccurs`` and ``InstantlyOccurs`` distributions
  have no fitted state: the class itself is the model, so ``to_dict``,
  ``to_json``, ``from_dict`` and ``from_json`` are called on the class
  (``NeverOccurs.to_json(path)``), and ``surpyval.from_json`` returns the
  class.
- Every reader -- the package-level ones and each class's own
  ``from_dict`` / ``from_json`` alike -- checks the ``"schema"`` (an
  integer no newer than this SurPyval), names the entry a truncated or
  hand-edited dictionary is missing, and refuses the parameters of a
  univariate parametric model that fall outside the distribution's
  bounds, each with a ``ValueError``.
- These cannot be saved and raise an error from ``to_dict``: a
  stratified Cox model, an accelerated-life model with a user-defined
  life model, a regression fitted with a formula that uses a
  data-dependent transform (such as ``scale()``), and a copula of a
  custom family.
- A model of a ``Discretize(...)`` distribution is read back like any
  other. A model of a ``CustomDistribution`` stores the distribution's
  name only (its cumulative hazard is a Python function):
  ``from_dict`` reads it back in a session that has constructed the same
  ``CustomDistribution`` again, and otherwise raises a ``ValueError``
  saying so.

Infinite and NaN values
-----------------------

A fitted model holds values that are not finite numbers: an untruncated
bound (``-inf``, ``inf``), a cumulative hazard after the last death
(``inf``), an undefined variance (``nan``). JSON cannot write them --
Python's ``json`` writes the non-standard ``Infinity`` and ``NaN``, which
strict parsers (JavaScript's ``JSON.parse``, many databases) refuse -- so
the dictionaries use a convention of their own, and are strict JSON:

- each non-finite value is written as ``null``;
- the dictionary holding it records what every such ``null`` stood for
  under ``"non_finite"``: for each kind present (``"inf"``, ``"-inf"``,
  ``"nan"``) a list of `JSON Pointers
  <https://www.rfc-editor.org/rfc/rfc6901>`_ to the values, relative to
  that dictionary. A model dictionary nested in another (a copula's
  margins, a tree's leaves) carries its own record. A ``null`` that no
  record names is an ordinary missing value.

.. code:: python

    >>> surpyval.KaplanMeier.fit([3, 4, 5]).to_dict()
    {..., 'H': [0.405..., 1.098..., None],
     'greenwood': [0.166..., 0.666..., None],
     'non_finite': {'inf': ['/H/2'], 'nan': ['/greenwood/2']}, 'schema': 2}

Every reader restores the original values, so a round trip is exact. A
consumer in another language sees ``null`` where no number applies, and
can use the record to recover the values. Dictionaries and files written
before this convention (``"schema"`` 0 or 1, holding ``Infinity`` or
``NaN``, which Python's ``json`` reads) still load; an older SurPyval
refuses a schema-2 file with an error asking for an upgrade rather than
misreading its ``null`` values. ``SurpyvalData.to_json`` uses the same
convention. :func:`~surpyval.serialisation.encode_non_finite` and
:func:`~surpyval.serialisation.decode_non_finite` apply and undo it on
any dictionary.

.. autofunction:: surpyval.serialisation.from_dict

.. autofunction:: surpyval.serialisation.from_json

.. autofunction:: surpyval.serialisation.encode_non_finite

.. autofunction:: surpyval.serialisation.decode_non_finite
