Fixed Event Probability
=======================

A proportion ``p`` of units experience the event and the rest never do,
with nothing said about *when*: :math:`F(x) = p` at every ``x``. It is
the mixture of :doc:`InstantlyOccurs <degenerate>` (weight ``p``) and
:doc:`NeverOccurs <degenerate>` (weight ``1 - p``). ``fit`` estimates
``p`` as the proportion of ones in 0/1 outcomes, and ``from_params``
builds the model from a known ``p``. It has ``sf``, ``ff`` and ``Hf``
but no density, hazard rate or quantile, since it has no time axis; its
``moment`` and ``mean`` are those of the 0/1 event indicator, ``p``. For a
single pass/fail flip whose survival steps at the outcome, see
:doc:`bernoulli`.

The uncertainty of a fitted ``p`` is that of a proportion:
``param_cb("p")`` gives the exact (Clopper-Pearson, the default) or
``method="jeffreys"`` interval, and ``covariance()`` the variance
:math:`\hat p (1 - \hat p) / N` (``nan`` with no events or only events).
What the intervals mean, why they are preferred to plus or minus two
standard errors near 0 or 1, the formulas and a coverage study are on the
:ref:`Bernoulli page <probability-uncertainty>`.

.. code:: python

    import surpyval as sp

    model = sp.FixedEventProbability.fit([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    model.param_cb("p")                     # [0.0252, 0.5561]
    model.param_cb("p", method="jeffreys")  # [0.0441, 0.5028]
    model.covariance()                      # [[0.016]]

.. autoclass:: surpyval.univariate.parametric.distributions.fixed_event_probability.FixedEventProbability_
   :members:
   :inherited-members:
