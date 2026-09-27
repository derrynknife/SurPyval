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

.. autoclass:: surpyval.univariate.parametric.distributions.fixed_event_probability.FixedEventProbability_
   :members:
   :inherited-members:
