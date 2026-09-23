Saving and Loading Models
=========================

Every fitted SurPyval model can be serialised to a plain dictionary
with ``to_dict()`` and written to JSON with ``to_json(path)``. The
dictionaries hold only plain Python types, so they can also go
straight into a document store (the round trip through BSON/MongoDB is
tested).

Restoring takes one call whichever class wrote the file: the
package-level readers dispatch on the serialised dictionary itself.

.. code:: python

    import surpyval

    model = surpyval.Weibull.fit(x)
    model.to_json("weibull.json")

    restored = surpyval.from_json("weibull.json")  # any model's file
    restored = surpyval.from_dict(model_dict)      # any model's dict

The class-level readers (``Weibull.from_dict``, ``CoxPH.from_dict``,
...) remain available when the model class is known up front; each
rejects a dictionary written by a different class with a
``ValueError`` naming the expected model.

.. autofunction:: surpyval.serialisation.from_dict

.. autofunction:: surpyval.serialisation.from_json
