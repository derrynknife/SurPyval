Exact Event Time
================

A degenerate distribution placing all its mass at a single event time
``T``: the survival is 1 before ``T`` and 0 from ``T`` on. Build it from
a known ``T`` with ``from_params``, or estimate ``T`` with ``fit`` from
right-censored ("not yet") and left-censored ("already") checks, which
returns the midpoint between the latest "not yet" and the earliest
"already". A point mass has no density or hazard rate, so ``df`` and
``hf`` raise ``NotImplementedError``; ``sf``, ``ff``, ``Hf``, ``qf``,
``mean`` and ``moment`` are exact.

.. autoclass:: surpyval.univariate.parametric.distributions.exact_event_time.ExactEventTime_
   :members:
   :inherited-members:
