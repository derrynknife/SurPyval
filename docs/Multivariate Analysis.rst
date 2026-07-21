Multivariate Analysis
=====================

Every distribution and model covered so far is *univariate*: each unit has a
single event time, and units are treated as independent replicates.
Multivariate survival analysis relaxes the independence: it models several
*correlated* event-time series jointly. A pair of failure times on the two
bearings of one shaft, the times to two related complications in one patient,
or the lifetimes of two components sharing an environment are all naturally
dependent, and pretending otherwise understates the joint risk.

The difficulty is that a joint distribution mixes two very different things:
what each series looks like *on its own*, and how the series *move together*.
The **copula** is the device that separates them.

Copulas and Sklar's theorem
---------------------------

A copula :math:`C(u_1, u_2)` is simply a joint distribution function whose
margins are uniform on :math:`[0, 1]`. Its whole job is to encode dependence,
stripped of any marginal shape. Sklar's theorem says that *any* joint
distribution can be written this way: given marginal CDFs :math:`F_1, F_2`, the
joint CDF is

.. math::

   H(x_1, x_2) = C\big(F_1(x_1),\, F_2(x_2)\big).

Because :math:`F_1` and :math:`F_2` map each series onto uniform margins, the
copula :math:`C` carries *only* the dependence structure. This is the key
modelling freedom: the margins can be any survival distribution — a Weibull for
one series, a LogNormal for the other — while the copula, chosen separately,
governs how they are coupled. The marginal question ("how long does this
component last?") and the dependence question ("do the two fail together?") are
answered by different parts of the model.

Families of dependence
----------------------

Different copula families describe qualitatively different dependence,
especially in the *tails* — whether two series tend to fail together at short
lives (lower-tail dependence) or survive together to long lives (upper-tail
dependence):

.. list-table::
   :header-rows: 1

   * - Copula
     - Parameter
     - Dependence
   * - Independence
     - none
     - none (:math:`\tau = 0`)
   * - Clayton
     - :math:`\theta > 0`
     - lower-tail (joint early failure)
   * - Gumbel
     - :math:`\theta \geq 1`
     - upper-tail (joint long survival)
   * - Frank
     - :math:`\theta \neq 0`
     - symmetric, no tail
   * - Gaussian
     - :math:`\rho \in (-1, 1)`
     - symmetric, no tail

The strength of dependence is summarised by rank measures that do not depend on
the margins — **Kendall's** :math:`\tau` and **Spearman's** :math:`\rho` — and
the tendency to fail (or survive) together in the extremes by the **tail
dependence** coefficients. Choosing a family is largely a question of which
tail behaviour matches the physics or the clinical reality.

Estimation
----------

Fitting a copula model means estimating both the marginal parameters and the
copula parameter. Two strategies trade off robustness against efficiency:

- **IFM** (*Inference Functions for Margins*) fits each margin independently and
  then estimates the copula parameter with the margins held fixed. It is fast
  and robust, and is the usual default.
- **MLE** optimises the copula parameter jointly with all marginal parameters.
  It is more efficient when the model is well specified, at a higher
  computational cost.

Because the margins are ordinary survival distributions, the joint likelihood
inherits survival analysis's treatment of incomplete data: each series of a
joint observation can be independently right, left or interval censored, or
truncated, using the same conventions as the univariate models. Every censoring
type reduces to evaluating the copula CDF and its partial derivatives at the
margin-transformed bounds — interval censoring, for instance, is
inclusion-exclusion on the rectangle corners of :math:`C`.

For worked examples — fitting a copula, handling per-series censoring, querying
the joint distribution and dependence measures, and simulating correlated
lifetimes — see the :doc:`Multivariate Modelling with SurPyval` page.
