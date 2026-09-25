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

.. autofunction:: surpyval.serialisation.from_dict

.. autofunction:: surpyval.serialisation.from_json
