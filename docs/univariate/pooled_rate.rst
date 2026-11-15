Pooled Rates Across Sources
===========================

The two-stage (hierarchical) pooling of event counts from several
sources: gamma-Poisson for events in a time at risk, beta-binomial for
events out of a number of trials. It gives the population (between-source)
distribution, the average across sources, the range for a new source like
these, each source's own and shrunk estimate, and a heterogeneity test.
The guide, in plain words and then the maths, is
:doc:`../Pooling Rates Across Sources`.

.. autofunction:: surpyval.pooled_rate

.. autoclass:: surpyval.PooledRate
   :members: n_sources, population, sources, update, assumptions, method_description, to_dict, from_dict
