Multivariate Analysis
=====================

Every distribution and model covered so far is *univariate*: each unit has a
single event time, and units are treated as independent replicates.
Multivariate survival analysis relaxes the independence: it models several
*correlated* event-time series jointly. A pair of failure times on the two
bearings of one shaft, the times to two related complications in one patient,
or the lifetimes of two components sharing an environment are all naturally
dependent, and pretending otherwise understates the joint risk.

How much can independence get wrong? Take two pumps in a redundant
(parallel) pair, each with a 10% chance of failing within a year. If they fail
independently, the chance that *both* fail — and the system goes down — is
:math:`0.1 \times 0.1 = 1\%`. If they share a cause of early failure (the same
batch of seals, the same contaminated fluid), the joint probability can be
several times larger, even though each pump on its own is exactly as reliable
as before. Nothing in the two marginal distributions reveals this; it lives
entirely in the dependence between them.

The difficulty is that a joint distribution mixes two very different things:
what each series looks like *on its own*, and how the series *move together*.
The **copula** is the device that separates them. SurPyval's
``surpyval.multivariate`` module implements bivariate copula models whose
margins are ordinary SurPyval distributions; worked examples are on the
:doc:`Multivariate Modelling with SurPyval` page and the API on
:doc:`surpyval.multivariate`.

Copulas and Sklar's theorem
---------------------------

The probability integral transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The starting point is a fact every simulation relies on: if :math:`X` has a
continuous CDF :math:`F`, then :math:`U = F(X)` is uniform on :math:`[0, 1]`.
Transforming each series by its own CDF therefore strips away its marginal
shape — Weibull, LogNormal, whatever it was — and leaves a uniform variable.
What remains *between* the transformed variables :math:`U_1 = F_1(X_1)` and
:math:`U_2 = F_2(X_2)` is pure dependence: if the bearings tend to fail early
together, :math:`U_1` and :math:`U_2` tend to be small together.

Definition and Sklar's theorem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A copula :math:`C(u_1, u_2)` is simply a joint distribution function whose
margins are uniform on :math:`[0, 1]`:
:math:`C(u_1, u_2) = P(U_1 \leq u_1, U_2 \leq u_2)`. Its whole job is to
encode dependence, stripped of any marginal shape. Sklar's theorem
[Sklar1959mv]_ says that *any* joint distribution can be written this way:
given marginal CDFs :math:`F_1, F_2`, the joint CDF is

.. math::

   H(x_1, x_2) = C\big(F_1(x_1),\, F_2(x_2)\big).

For continuous margins the copula is unique, and conversely *any* copula
combined with *any* margins through this formula gives a valid joint
distribution. Because :math:`F_1` and :math:`F_2` map each series onto uniform
margins, the copula :math:`C` carries *only* the dependence structure. This is
the key modelling freedom: the margins can be any survival distribution — a
Weibull for one series, a LogNormal for the other — while the copula, chosen
separately, governs how they are coupled. The marginal question ("how long
does this component last?") and the dependence question ("do the two fail
together?") are answered by different parts of the model.

Two copulas bracket all the others. Independence is :math:`C(u_1, u_2) = u_1
u_2`. Every copula lies between the Fréchet-Hoeffding bounds

.. math::

   \max(u_1 + u_2 - 1,\, 0) \;\leq\; C(u_1, u_2) \;\leq\; \min(u_1, u_2),

the lower bound describing perfect negative dependence (one series is a
decreasing function of the other) and the upper bound perfect positive
dependence (one is an increasing function of the other). A copula family with a
parameter moves between independence and one or both of these extremes.

Everything else from the copula
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writing :math:`u_1 = F_1(x_1)` and :math:`u_2 = F_2(x_2)`, every joint
quantity used in survival analysis follows from :math:`C` and the margins:

- the **joint CDF** :math:`H(x_1, x_2) = P(X_1 \leq x_1, X_2 \leq x_2) =
  C(u_1, u_2)` — both have failed by their respective times;
- the **joint survival function**, by inclusion-exclusion,

  .. math::

     S(x_1, x_2) = P(X_1 > x_1,\, X_2 > x_2) = 1 - u_1 - u_2 + C(u_1, u_2);

- the **joint density** :math:`f(x_1, x_2) = c(u_1, u_2)\, f_1(x_1)\,
  f_2(x_2)`, where :math:`c = \partial^2 C / \partial u_1 \partial u_2` is the
  copula density and :math:`f_1, f_2` the marginal densities;
- the **conditional distribution** of one series given the exact value of the
  other, through the partial derivative of :math:`C` (often called the
  *h-function*):

  .. math::

     P(X_2 \leq x_2 \mid X_1 = x_1) = \frac{\partial C(u_1, u_2)}{\partial u_1}.

The h-function is also the key to simulation: draw :math:`u_1` uniform, draw
a second uniform :math:`w`, and solve :math:`\partial C/\partial u_1(u_1,
u_2) = w` for :math:`u_2`; then map back through the margins' quantile
functions, :math:`x_j = F_j^{-1}(u_j)`.

A note on orientation. SurPyval applies the copula to the marginal *CDFs*, as
in Sklar's theorem above. Some survival texts instead apply a copula to the
marginal *survival* functions, :math:`S(x_1, x_2) = \tilde{C}(S_1(x_1),
S_2(x_2))` (a "survival copula"); for example, the shared gamma-frailty model
is a Clayton copula applied in this second way. The same family name then
describes a different model — the tail that is dependent is flipped — so when
comparing with published results check which convention was used.

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

Measuring dependence
~~~~~~~~~~~~~~~~~~~~

The usual (Pearson) correlation is a poor summary for lifetimes: it depends on
the margins, and it changes if you take logs of the times. The copula-based
measures depend only on :math:`C`, so they are the same whatever the margins
and whatever monotone transformation is applied to the times:

- **Kendall's tau** is the probability that two independent pairs are
  *concordant* (the unit that fails first on series 1 also fails first on
  series 2) minus the probability they are discordant:
  :math:`\tau = 4\,E[C(U_1, U_2)] - 1 \in [-1, 1]`.
- **Spearman's rho** is the ordinary correlation of the uniforms,
  :math:`\rho_S = \operatorname{corr}(U_1, U_2) = 12\int_0^1\!\int_0^1
  C(u_1, u_2)\, du_1\, du_2 - 3`.
- The **tail-dependence coefficients** measure clustering in the extremes:

  .. math::

     \lambda_L = \lim_{q \to 0^+} P\big(U_1 \leq q \mid U_2 \leq q\big)
               = \lim_{q \to 0^+} \frac{C(q, q)}{q},
     \qquad
     \lambda_U = \lim_{q \to 1^-} P\big(U_1 > q \mid U_2 > q\big).

  :math:`\lambda_L > 0` means that, given one unit is among the very earliest
  failures, there is a non-vanishing probability that its partner is too:
  early failures come in pairs. :math:`\lambda_U > 0` is the same statement
  for the longest survivors.

Two copulas can share the same :math:`\tau` and still differ greatly in the
tails, which is exactly where the risk of joint failure is decided. Fit the
family, not just the correlation.

The families in detail
~~~~~~~~~~~~~~~~~~~~~~

The formulas below are standard; [Nelsen2006mv]_ derives them and many more.

**Independence** — :math:`C(u_1, u_2) = u_1 u_2`. No parameter; the joint
distribution is the product of the margins. It is the baseline against which
the others are judged: if a dependent copula does not fit clearly better, the
simpler independent model may do.

**Clayton** —

.. math::

   C(u_1, u_2) = \big(u_1^{-\theta} + u_2^{-\theta} - 1\big)^{-1/\theta},
   \qquad \theta > 0,

   \tau = \frac{\theta}{\theta + 2}, \qquad
   \lambda_L = 2^{-1/\theta}, \qquad \lambda_U = 0 .

:math:`\theta \to 0` gives independence and :math:`\theta \to \infty` perfect
positive dependence. Dependence concentrates in the *lower* tail: the two
series are most tightly linked among early failures. Use it for **common-cause
early failure** — a shared manufacturing defect, a shared harsh start-up, a
shared contamination event — where knowing one unit failed young is strong
evidence its partner will too.

**Gumbel** (Gumbel-Hougaard) —

.. math::

   C(u_1, u_2) = \exp\!\Big(-\big[(-\ln u_1)^{\theta} +
                 (-\ln u_2)^{\theta}\big]^{1/\theta}\Big),
   \qquad \theta \geq 1,

   \tau = 1 - \frac{1}{\theta}, \qquad
   \lambda_L = 0, \qquad \lambda_U = 2 - 2^{1/\theta} .

:math:`\theta = 1` is independence. Dependence concentrates in the *upper*
tail: the series are most tightly linked among the longest lives. Use it when
a shared *benign* factor (a gentle duty cycle, a robust batch) lets pairs
survive to old age together. (The Gumbel copula is unrelated to the univariate
Gumbel distribution, ``surpyval.Gumbel``.)

**Frank** —

.. math::

   C(u_1, u_2) = -\frac{1}{\theta}\ln\!\left(1 +
       \frac{(e^{-\theta u_1} - 1)(e^{-\theta u_2} - 1)}{e^{-\theta} - 1}\right),
   \qquad \theta \neq 0,

   \tau = 1 - \frac{4}{\theta}\big[1 - D_1(\theta)\big],
   \qquad \lambda_L = \lambda_U = 0,

where :math:`D_1(\theta) = \frac{1}{\theta}\int_0^{\theta}
\frac{s}{e^s - 1}\,ds` is the first Debye function. :math:`\theta > 0` gives
positive and :math:`\theta < 0` negative dependence, symmetric in the two
tails and with no tail dependence. Use it for moderate, "everywhere-alike"
association, or whenever the dependence may be *negative* (one series tends to
be long when the other is short, as when two failure modes compete for the
same weakness).

**Gaussian** —

.. math::

   C(u_1, u_2) = \Phi_2\big(\Phi^{-1}(u_1), \Phi^{-1}(u_2); \rho\big),
   \qquad -1 < \rho < 1,

   \tau = \frac{2}{\pi}\arcsin\rho, \qquad
   \rho_S = \frac{6}{\pi}\arcsin\frac{\rho}{2}, \qquad
   \lambda_L = \lambda_U = 0,

where :math:`\Phi` is the standard normal CDF and :math:`\Phi_2(\cdot,
\cdot; \rho)` the bivariate normal CDF with correlation :math:`\rho`. It is
the dependence of a bivariate normal distribution transplanted onto arbitrary
margins, so :math:`\rho` has the familiar meaning of a correlation between the
normal scores :math:`\Phi^{-1}(u_j)`. Positive and negative dependence are both
allowed. Its lack of tail dependence means that, however large :math:`\rho`,
joint *extreme* events become asymptotically independent — a Gaussian copula
can understate the risk of joint early failure when the true dependence is
Clayton-like.

In SurPyval, Kendall's tau is computed in closed form for all five families
(Frank's through a numerically evaluated Debye integral), and so are
Spearman's rho for the Gaussian and Independence copulas and the
tail-dependence coefficients (which are zero except for Clayton and Gumbel).
Spearman's rho for Clayton, Gumbel and Frank is estimated from a fixed-seed
sample of 50,000 draws from the copula, so it is accurate to roughly two
decimal places.

Choosing a family
~~~~~~~~~~~~~~~~~

- Start from the physics: is there a mechanism that makes *early* failures
  cluster (Clayton), *long* lives cluster (Gumbel), or a diffuse association
  with no special tail (Frank, Gaussian)?
- Only Frank and Gaussian can express **negative** dependence. Clayton
  (:math:`\theta > 0`) and Gumbel (:math:`\theta \geq 1`) are positive-only in
  SurPyval; fitted to negatively dependent data they are pushed to their
  independence boundary, so check the sign of the empirical Kendall's tau
  first.
- With complete data, compare fitted families by their log-likelihood (or AIC —
  every family here has one parameter, so the comparison is the same) and by
  plotting simulated samples against the data. The how-to page shows both.

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

The likelihood with censored data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because the margins are ordinary survival distributions, the joint likelihood
inherits survival analysis's treatment of incomplete data: each series of a
joint observation can be independently right, left or interval censored, or
truncated, using the same conventions as the univariate models. Every censoring
type reduces to evaluating the copula CDF and its partial derivatives at the
margin-transformed bounds — interval censoring, for instance, is
inclusion-exclusion on the rectangle corners of :math:`C`.

The rule is mechanical. For each series, look at what is known about it and
apply one operation to :math:`C` in that series' argument:

.. list-table::
   :header-rows: 1

   * - What is known about :math:`X_j`
     - Censoring code ``c``
     - Operation on argument :math:`j` of :math:`C`
   * - observed exactly at :math:`x`
     - ``0``
     - differentiate at :math:`u = F_j(x)`, and multiply by :math:`f_j(x)`
   * - right censored, :math:`X_j > x`
     - ``1``
     - :math:`C(\cdot, 1) - C(\cdot, F_j(x))`
   * - left censored, :math:`X_j \leq x`
     - ``-1``
     - evaluate at :math:`F_j(x)`
   * - interval censored, :math:`x_l < X_j \leq x_r`
     - ``2``
     - :math:`C(\cdot, F_j(x_r)) - C(\cdot, F_j(x_l))`

Applying the operations for both series gives the row's likelihood. Some
examples, with :math:`u_j = F_j(x_j)`:

.. math::

   \text{both observed:}\quad & c(u_1, u_2)\, f_1(x_1)\, f_2(x_2) \\
   \text{1 observed, 2 right censored:}\quad &
       f_1(x_1)\Big[1 - \frac{\partial C}{\partial u_1}(u_1, u_2)\Big] \\
   \text{both right censored:}\quad & 1 - u_1 - u_2 + C(u_1, u_2) \\
   \text{both left censored:}\quad & C(u_1, u_2)

The second line reads naturally: the density of seeing series 1 fail at
:math:`x_1`, times the conditional probability that series 2 had *not* failed
by :math:`x_2` given that — the h-function at work.

**Truncation** conditions on the row having been observable at all. If
series :math:`j` could only be seen inside the window :math:`(t_{l,j},
t_{r,j})`, each row's likelihood is divided by the copula mass of the
rectangle :math:`[F_1(t_{l,1}), F_1(t_{r,1})] \times [F_2(t_{l,2}),
F_2(t_{r,2})]`, i.e. the probability of both series landing in their windows,
again by inclusion-exclusion on :math:`C`. When only one series is truncated
this mass reduces to a marginal probability and does not involve the copula
parameter at all — truncation then acts through the margins.

**Counts** ``n`` weight each row's log-likelihood, so a row standing for
five identical units counts five times.

How SurPyval fits
~~~~~~~~~~~~~~~~~

Every family has the same ``fit(x, c=None, n=None, t=None, margins=None,
how="IFM", xl=None, xr=None)``; ``margins`` (one per series) is required and
exactly two series are supported. With ``how="IFM"`` [JoeXu1996mv]_:

1. each margin is fitted by the usual univariate maximum likelihood to its own
   series, honouring that series' censoring codes (interval-censored entries
   through ``xl``/``xr``), the row counts ``n`` and that series' truncation
   window; margins passed as already-fitted models are used as they are;
2. with the margins fixed, the copula parameter is chosen to maximise the
   copula log-likelihood above. The search runs on an unconstrained
   transformation of the parameter (for the Gaussian copula,
   :math:`\rho = \tanh(\cdot)`), starting from the value that matches the
   empirical Kendall's tau of the rows where both series are observed.

With ``how="MLE"`` the IFM solution is the starting point for a joint search
over the copula parameter and every margin parameter, maximising the full
likelihood.

Some consequences worth knowing:

- The IFM first stage truncates each margin by its own series' window only.
  When the observation rule is joint -- a row is seen only if series 1
  passed a burn-in, say -- the rows are also a selected sample of the other
  series, which its margin cannot know about, and both that margin and the
  copula parameter come out biased. ``how="MLE"`` divides each row by the
  copula mass of the whole truncation rectangle and so accounts for the
  selection; use it whenever truncation of one series selects the rows of
  another.
- The copula parameter is estimated on the scale :math:`u_j = F_j(x_j)`, so a
  poorly chosen margin distorts it. Check the margins with the univariate
  tools first (see :doc:`Parametric SurPyval Modelling`).
- The fitted model reports the point estimates; no standard errors or
  likelihood values are attached to it.
- Margin probabilities are kept a tiny distance (:math:`10^{-10}`) inside
  :math:`(0, 1)` to keep the Archimedean formulas finite.
- Only bivariate models are supported; more than two series raise a
  ``NotImplementedError``.

For worked examples — fitting a copula, handling per-series censoring, querying
the joint distribution and dependence measures, and simulating correlated
lifetimes — see the :doc:`Multivariate Modelling with SurPyval` page.

.. rubric:: References

.. [Sklar1959mv] Sklar, A. (1959). Fonctions de répartition à n dimensions et
   leurs marges. *Publications de l'Institut de Statistique de l'Université
   de Paris*, 8, 229–231.

.. [JoeXu1996mv] Joe, H., & Xu, J. J. (1996). The estimation method of
   inference functions for margins for multivariate models. Technical Report
   166, Department of Statistics, University of British Columbia.

.. [Nelsen2006mv] Nelsen, R. B. (2006). *An Introduction to Copulas*
   (2nd ed.). Springer. The standard reference for the families, dependence
   measures and tail dependence used on this page.
