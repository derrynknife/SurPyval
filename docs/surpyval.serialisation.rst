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
  that ``bic``, ``aic_c`` and ``plot`` (parametric) or ``bootstrap_cb``
  (non-parametric) work after restoring. ``to_json`` never stores the
  data.
- The degenerate ``NeverOccurs`` and ``InstantlyOccurs`` distributions
  have ``to_dict`` but no ``to_json``; write their dictionary with
  ``json.dump``.
- These cannot be saved and raise an error from ``to_dict``: a
  stratified Cox model, an accelerated-life model with a user-defined
  life model, a regression fitted with a formula that uses a
  data-dependent transform (such as ``scale()``), and a copula of a
  custom family.
- A model of a ``CustomDistribution`` or of a ``Discretize(...)``
  distribution is written, but ``from_dict`` cannot rebuild it (it only
  resolves SurPyval's own distributions by name) and raises a
  ``ValueError``. Save its ``params`` and rebuild it with the
  distribution's ``from_params``.
- The dictionaries can hold ``inf`` and ``-inf`` (an untruncated bound, a
  cumulative hazard after the last death). Python's ``json`` module and
  BSON store these, but they are not strict JSON, so a strict parser in
  another language may refuse the file.

.. autofunction:: surpyval.serialisation.from_dict

.. autofunction:: surpyval.serialisation.from_json
