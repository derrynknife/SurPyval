Multivariate Modelling with SurPyval
====================================

Every distribution covered so far is *univariate*: each unit has a single
event time and units are treated as independent replicates. The
``surpyval.multivariate`` module opens the **multivariate** axis of the model
atlas -- several *correlated* event-time series modelled jointly. The
dependence between the series is specified with a **copula**, while the
marginal behaviour of each series is any existing SurPyval distribution.

A copula ``C(u_1, u_2)`` is a joint distribution on uniform margins. Coupling
it with real margins :math:`F_1, F_2` gives the joint CDF

.. math::

   H(x_1, x_2) = C\big(F_1(x_1), F_2(x_2)\big).

This cleanly separates *what each series looks like on its own* (the margins)
from *how the series move together* (the copula).

Available copulas
-----------------

.. list-table::
   :header-rows: 1

   * - Copula
     - Parameter
     - Dependence
   * - ``Independence``
     - none
     - none (:math:`\tau = 0`)
   * - ``Clayton``
     - :math:`\theta > 0`
     - lower-tail
   * - ``Gumbel``
     - :math:`\theta \geq 1`
     - upper-tail
   * - ``Frank``
     - :math:`\theta \neq 0`
     - symmetric, no tail
   * - ``Gaussian``
     - :math:`\rho \in (-1, 1)`
     - symmetric, no tail

Fitting a copula
----------------

The copulas live in their own package (like ``surpyval.recurrent``) and are
not imported into the top-level namespace:

.. code:: python

    import surpyval as surv
    from surpyval.multivariate import Clayton

    # x1, x2 are correlated lifetimes; margins are SurPyval distributions
    model = Clayton.fit(
        [x1, x2],
        margins=[surv.Weibull, surv.LogNormal],
        how="IFM",
    )

    model.params          # fitted copula parameter (theta)
    model.kendall_tau()   # implied Kendall's tau
    model.margins         # the two fitted univariate models

Two estimation strategies are available via ``how``:

* ``"IFM"`` (*Inference Functions for Margins*, the default) fits each margin
  independently and then fits the single copula parameter holding the margins
  fixed. Robust and fast.
* ``"MLE"`` jointly optimises the copula parameter together with all margin
  parameters.

Censoring and truncation
------------------------

The differentiator of the SurPyval copula implementation is that the joint
likelihood supports the **full** censoring and truncation matrix, per
dimension, using the same convention as the univariate models
(``c`` of ``0`` observed, ``1`` right, ``-1`` left, ``2`` interval; ``t`` for
a truncation window). Each series of a joint observation may be censored
independently:

.. code:: python

    # dimension-wise censoring codes, one column per series
    model = Clayton.fit(
        [x1, x2],
        c=[c1, c2],
        margins=[surv.Weibull, surv.Weibull],
    )

Internally every censoring type reduces to evaluating the copula CDF and its
partial derivatives at the margin-transformed bounds (interval censoring, for
example, is inclusion-exclusion on the rectangle corners of ``C``).

Working with a fitted model
---------------------------

.. code:: python

    model.cdf([[10, 18], [5, 25]])   # joint P(X1 <= x1, X2 <= x2)
    model.sf([[10, 18]])             # joint survival P(X1 > x1, X2 > x2)
    model.pdf([[10, 18]])            # joint density
    model.conditional_cdf(x, given_dim=0)   # the copula h-function

    samples = model.random(1000)     # correlated (N, 2) samples
    model.kendall_tau()
    model.spearman_rho()
    model.tail_dependence()          # (lambda_lower, lambda_upper)

Building a model from known parameters
--------------------------------------

As with the univariate distributions, a model can be created directly from
parameters and pre-built margins -- useful for Monte-Carlo simulation:

.. code:: python

    model = Clayton.from_params(
        2.0,
        margins=[surv.Weibull.from_params([10, 2]),
                 surv.LogNormal.from_params([3, 0.4])],
    )
    correlated_draws = model.random(10_000)
