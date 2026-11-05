Binomial Distribution
=====================

The Binomial distribution: the number of events in a fixed number
of independent trials.

The uncertainty of a fitted ``p`` comes from the events and trials of all
the rows together: ``param_cb("p")`` gives the exact (Clopper-Pearson,
the default) or ``method="jeffreys"`` interval, and ``covariance()`` the
variance :math:`\hat p (1 - \hat p) / N` (``nan`` with no events or only
events), in the order of ``[n, p]`` with zeros for the known ``n``. What
the intervals mean, why they are preferred to plus or minus two standard
errors near 0 or 1, the formulas and a coverage study are on the
:ref:`Bernoulli page <probability-uncertainty>`.

.. code:: python

    import surpyval as sp

    model = sp.Binomial.fit([2, 3, 1, 4], n_trials=5)
    model.param_cb("p")                     # [0.272, 0.728]
    model.param_cb("p", method="jeffreys")  # [0.2934, 0.7066]
    model.covariance()                      # [[0, 0], [0, 0.0125]]

.. autoclass:: surpyval.univariate.parametric.distributions.binomial.Binomial_
   :members:
   :inherited-members:
