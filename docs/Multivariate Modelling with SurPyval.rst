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
not imported into the top-level namespace. Here we simulate correlated
lifetimes from a known Clayton copula (see the last section) and check the fit
recovers it:

.. jupyter-execute::

    import warnings
    warnings.filterwarnings("ignore")   # copula optimisers explore log(0) regions

    import numpy as np
    import surpyval as surv
    from surpyval.multivariate import Clayton

    truth = Clayton.from_params(
        2.0,
        margins=[surv.Weibull.from_params([10.0, 2.0]),
                 surv.LogNormal.from_params([2.5, 0.5])],
    )
    data = truth.random(3000, random_state=1)
    x1, x2 = data[:, 0], data[:, 1]

    # margins are SurPyval distributions, fitted along with the copula
    model = Clayton.fit(
        [x1, x2],
        margins=[surv.Weibull, surv.LogNormal],
        how="IFM",
    )
    print("theta       :", model.params)
    print("Kendall's tau:", round(model.kendall_tau(), 3))
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
independently — pass one censoring column per series. Here each series is
right-censored at its own threshold, and the fit still recovers the copula
parameter:

.. jupyter-execute::

    thresholds = np.array([14.0, 16.0])
    c = (data > thresholds).astype(int)          # per-series right-censoring
    x_obs = np.minimum(data, thresholds)

    model_c = Clayton.fit(
        [x_obs[:, 0], x_obs[:, 1]],
        c=[c[:, 0], c[:, 1]],                    # one column per series
        margins=[surv.Weibull, surv.Weibull],
        how="IFM",
    )
    print("censored fraction:", round(c.mean(), 2))
    print("theta (censored) :", model_c.params)

Internally every censoring type reduces to evaluating the copula CDF and its
partial derivatives at the margin-transformed bounds (interval censoring, for
example, is inclusion-exclusion on the rectangle corners of ``C``).

Working with a fitted model
---------------------------

The fitted model exposes the joint distribution functions, the dependence
measures, and a correlated sampler:

.. jupyter-execute::

    print("joint cdf  :", model.cdf([[10, 18], [5, 25]]))  # P(X1<=x1, X2<=x2)
    print("joint sf   :", model.sf([[10, 18]]))            # P(X1>x1, X2>x2)
    print("joint pdf  :", model.pdf([[10, 18]]))
    print("cond. cdf  :", model.conditional_cdf(np.array([[10, 18]]),
                                                 given_dim=0))  # h-function
    print("Kendall tau:", round(model.kendall_tau(), 3))
    print("Spearman   :", round(model.spearman_rho(), 3))
    print("tail dep.  :", model.tail_dependence())          # (lower, upper)

    model.random(3, random_state=0)     # correlated (N, 2) samples

Building a model from known parameters
--------------------------------------

As with the univariate distributions, a model can be created directly from
parameters and pre-built margins -- useful for Monte-Carlo simulation (this is
exactly how the data above was generated):

.. jupyter-execute::

    sim = Clayton.from_params(
        2.0,
        margins=[surv.Weibull.from_params([10, 2]),
                 surv.LogNormal.from_params([3, 0.4])],
    )
    sim.random(5, random_state=0)
