Joint Models
============

A measurement read again and again on each item -- a pump's vibration, a
gearbox's oil debris, a tyre's tread depth, a patient's biomarker -- often
says how close the item is to failing. The item's risk rises as the
measurement does. But the measurement is noisy, it is read only at
inspections, and it stops when the item fails. A **joint model** describes
the measurement (the *marker*) and the lifetime together, so that each is
read in the light of the other. In reliability terms it gives each asset's
remaining life given its readings so far, updated as new readings arrive.

SurPyval's joint model (``WeibullJointPH``, #891) is the standard one
(Rizopoulos, 2012):

- **the marker** follows a degradation path model -- a straight line in
  age for ``WeibullJointPH``, or any of the degradation module's paths
  (``QuadraticPath``, ``ExponentialPath``, ...) or one of your own -- with
  parameters of the item's own (drawn, across items, from a multivariate
  normal), read with normal noise;
- **the hazard** is Weibull proportional hazards in the path's *current,
  noise-free* value, and the item's covariates, which also shift its path.

The hazard need not follow the marker's level. It can follow **how fast the
marker is changing** (its slope), or **the exposure accumulated so far**
(the area under its path), or more than one of these: see `What the hazard
follows: the link`_. And the marker need not multiply the hazard at all:
in ``WeibullJointAFT`` it makes the item **age faster**, the life used up
at a high reading not given back when the reading falls (see `How the
reading acts: a higher failure rate, or faster ageing`_).

Why not the obvious approaches?
-------------------------------

Two shortcuts are common, and both are biased:

- **The last reading as a time-varying covariate** (a time-varying Cox or
  Weibull PH fit, the reading carried forward to the next). It takes a noisy
  reading as the truth, and between readings a value that is out of date.
  Both pull the marker's estimated effect towards zero -- regression
  dilution -- however many items there are.
- **The marker alone** (a degradation model). The readings stop when an item
  fails, and the items that fail are the ones whose marker was high: fitting
  the readings without the failures under-states how markers rise. And a
  threshold on the marker cannot describe failures that come at any level,
  with a risk that only *grows* with it.

The joint likelihood fixes both: each item's line is integrated over what
its readings leave uncertain, and its readings and its lifetime inform the
same line. (A middle way, fitting each item's line first and the lifetimes
on those lines second, is quick but still biased: see `A quick first look:
the two-stage fit`_.) (A third approach, which models no path at all, is
:doc:`landmarking <Landmark Analysis>`: a model of the units still at risk
at a time, by their last readings then.)

The model
---------

Item :math:`i` has readings :math:`y_{ij}` at ages :math:`t_{ij}`,
covariates :math:`z_i`, and fails or is censored at :math:`T_i`. Its marker
follows the path model :math:`m(t; \theta)` (``path(t, *theta)``; for
``WeibullJointPH`` the line :math:`a + b t`) with parameters
:math:`\theta_i` of its own, read with noise:

.. math::

    y_{ij} = m(t_{ij}; \theta_i) + \varepsilon_{ij}, \qquad
    \theta_{iV} \sim N(\mu_V + \Delta_V z_i, D), \qquad
    \theta_{iC} = \mu_C + \Delta_C z_i,

the parameters that vary between items (:math:`V`, all of them by default)
multivariate normal about means the covariates shift, the others
(:math:`C`) common to every item, and :math:`\varepsilon_{ij} \sim N(0,
\sigma^2)`. By default every covariate shifts every path parameter's mean
and enters the hazard; a fit can put each covariate in one part or both
(:math:`\Delta` and :math:`\gamma` are then 0 where a covariate is left
out: see `Covariates in each part`_). The hazard is

.. math::

    h_i(t) = \frac{\beta}{\alpha}\left(\frac{t}{\alpha}\right)^{\beta-1}
    \exp\left(\gamma' z_i + \zeta\,\bigl(m(t; \theta_i) - r\bigr)\right).

:math:`\zeta` (``marker_effect``) is the change in the log hazard per unit of
the marker: ``exp(marker_effect)`` is the hazard ratio of one unit. The
marker enters relative to :math:`r` (``marker_reference``: the mean reading
of the data, unless the fit is given another), so the Weibull :math:`\alpha`
and :math:`\beta` are the baseline of an item with its marker at
:math:`r` and its covariates at 0.

Given its path, an item's readings and lifetime are independent, so its
likelihood is the integral over its varying parameters

.. math::

    L_i = \int p(y_i \mid \theta)\, h_i(T_i \mid \theta)^{d_i}\,
    S_i(T_i \mid \theta)\, \phi(\theta_V; \mu_V + \Delta_V z_i, D)\,
    d\theta_V,

:math:`d_i` 1 for a failure and 0 for a censoring. The fit maximises the sum
of the :math:`\log L_i`, each integral by adaptive Gauss--Hermite
quadrature: a product grid of 7 points a dimension (49 for two varying
parameters, 343 for three; see :ref:`how the integral is done
<joint-integration>` for four and five, and for more, where there is no
grid), centred
at each item's
most likely parameters and scaled by how sure its readings make them (the
Gauss--Newton curvature, from the path's Jacobian: exact for a path linear
in its parameters). The nodes are held fixed while the parameters are
searched (with autograd's gradient, in the parameters' standard errors),
and a step is taken only as far as the likelihood with the nodes placed
again at the new point improves; this repeats until the log-likelihood
stops moving. A common path parameter is searched through the path model
itself (its Jacobian, and differences of it for the Hessian). The
cumulative hazard along a path is by 15-point Gauss--Legendre quadrature.
The fit starts from the path fitted to each item alone (the path model's
own ``fit``; items with too few readings for it are left out of this
step), the spread of those fits about the covariates' means, and a
survival fit with each item's predicted path plugged in -- a two-stage
estimate, which is biased (`A quick first look: the two-stage fit`_
returns it as a model of its own). The covariance is the inverse
of the Hessian of the negative log-likelihood, by autograd.

An item found failed at an inspection is known only to have failed between
two ages, :math:`l_i < T_i \le u_i` (``interval_censored``; ``left_censored``
is the interval from its entry, or 0): :math:`h_i(T_i)^{d_i} S_i(T_i)` is
then :math:`S_i(l_i \mid \theta) - S_i(u_i \mid \theta)`, the hazard
integrated over the interval directly (so a narrow interval late in life
loses no precision). An item that came under observation at age
:math:`L_i > 0` (``left_truncation``) is in the data because it reached
:math:`L_i`, and one right truncated at :math:`R_i` (``right_truncation``)
because it failed by then: its likelihood is divided by the probability
of that, :math:`\int (S(L_i \mid \theta) - S(R_i \mid \theta))
\phi(\theta_V)\, d\theta_V`, over the population's paths. The
likelihood of every kind of item agrees with an independent dense-grid
integral over the random effects to within 1e-5 an item
(``tests/univariate/joint/test_joint_extensions.py``).

.. _joint-integration:

How the integral is done
~~~~~~~~~~~~~~~~~~~~~~~~

Each item's likelihood is an average over the path parameters it could
have (its own level, its own rate, ...), and that average is a sum over a
grid of 7 values of each, placed where that item's readings and lifetime
say its parameters are. The grid grows quickly with the number of varying
parameters: 7 nodes for one, 49 for two, 343 for three, 2,401 for four and
16,807 for five, for every item at every step of the search. A sixth
would make it 117,649, so beyond five there is no grid: the integral is
Laplace's approximation.

``integration=`` on ``fit`` (and ``fit_from_df``) chooses how:

- ``"grid"``: the full grid. The default up to five varying parameters:
  the more accurate choice.
- ``"pruned"``: the grid without the nodes whose weight is negligible --
  its far corners, where several parameters are extreme at once. The rule
  is fixed: the lowest-weight nodes are left out, as many as hold less
  than 1e-9 of the grid's weight between them (nodes of equal weight
  together), whatever the data. That is none of the nodes for up to three
  parameters, 80 of 2,401 for four and 1,392 of 16,807 for five. An
  option, never the default: it trades a little accuracy for a little
  speed, and in practice saves little (below).
- ``"laplace"``: Laplace's approximation, with no grid. The default
  beyond five varying parameters, where a grid is refused, and an option
  below (:ref:`below <joint-laplace>`).

The model says which it used (``model.integration``, and the
``Integration`` line of its printout), keeps it when saved, and predicts
on the same kind of grid (a Laplace fit predicts on the full grid up to
five varying parameters, and by Laplace's approximation beyond). A model
saved before the choice existed reads as ``"grid"``.

.. code:: python

    from surpyval import JointPH, Weibull

    fitter = JointPH(Weibull, cubic)            # a, b, c and d all vary
    model = fitter.fit(data)                    # the full grid: 2,401 nodes an item
    model.integration                           # 'grid'
    pruned = fitter.fit(data, integration="pruned")

The pruned grid is a small saving that costs no accuracy: 3% fewer nodes
with four varying parameters and 8% fewer with five (one evaluation of the
likelihood and its gradient is 1% and 6% quicker). Where the integrand is
exactly normal (no marker effect, a path linear in its parameters) both
grids reproduce the closed form in four and five dimensions, the pruned one
short by the weight it leaves out, 3e-10 and 8e-10 of an item's likelihood.
With a marker effect the two log-likelihoods differ by that much again on
items read often, and by up to 7e-7 an item in the hardest case tried (five
varying parameters, a reading every 0.7 years with noise five times the
usual, a marker effect of 2); a fleet of 60 items fitted on each grid gave
estimates within 1e-6 of a standard error of each other
(``tests/univariate/joint/test_joint_integration.py``).

Two things to expect with four or five varying parameters. The fit takes
minutes where two or three take seconds -- the fleet of 60 items took two
to five minutes -- and most of that is the search over the ten or fifteen
variances and correlations of the items' spread, not the grid. And the
7 points themselves are less exact where items have few, noisy readings
and the marker effect is strong: against 11 points a dimension the
log-likelihood of such a fleet was off by up to 5e-5 an item (it was
within 4e-10 an item on a fleet read often). That is the grid's own
error, the same on the full grid and the pruned one. Fewer correlations
(next) make that search smaller.

.. _joint-laplace:

Laplace's approximation
^^^^^^^^^^^^^^^^^^^^^^^

A grid samples each item's integrand at many points. Laplace's
approximation uses only where it peaks: the integrand's height at the
item's most likely parameters :math:`\hat b_i`, and how sharply it falls
away from them,

.. math::

    \log L_i \approx g_i(\hat b_i) + \frac{q}{2} \log 2\pi
    - \frac{1}{2} \log \det\bigl(-g_i''(\hat b_i)\bigr),

with :math:`g_i` the log of the integrand and :math:`q` the number of
varying parameters. That is the integral of the normal curve that matches
the integrand at its peak, so it is exact where the integrand is normal (a
path linear in its parameters and no marker effect), and off by a term of
the order of one over the item's number of readings otherwise (Tierney and
Kadane 1986; for joint models, Rizopoulos, Verbeke and Lesaffre 2009). The
curvature is the integrand's own, by differences on :math:`q^2 + q + 1`
points around the peak: 21 evaluations an item for four varying parameters
where the grid takes 2,401, and 43 for six, where there is no grid.

.. code:: python

    model = JointPH(Weibull, cubic).fit(data, integration="laplace")
    model.integration                           # 'laplace'

    # six varying parameters: Laplace's approximation without asking
    JointPH(Weibull, six_bumps).fit(data).integration    # 'laplace'

The fit maximises the approximated likelihood exactly: each item's peak
moves as the parameters change, and the gradient the search follows and
the curvature the standard errors come from both account for that (the
gradient through the implicit-function theorem, the curvature by
differences of that gradient with every peak found again). A fit by
Laplace's approximation has ``aic()``, ``bic()`` and ``neg_ll()`` like any
other, of the approximated likelihood: compare models fitted the same way.

How close it is to the grid, measured
(``tests/univariate/joint/test_joint_laplace.py`` and
``scripts/studies/joint_recovery/laplace_few_readings.py``):

- With no marker effect it reproduces the closed form to 1e-8 an item in
  two, three, four and six dimensions.
- With a marker effect, at the simulation's truth, its log-likelihood was
  within about 1e-6 an item of an 11-point grid's on fleets read every 0.1
  years
  (up to 31 readings an item), and within 5e-4 an item read every 0.5
  years with three times the noise (up to 7 readings).
- Fitted to the recovery study's pumps (300 a fleet, two varying
  parameters, read monthly: up to 25 readings), its estimates were within
  0.02 standard errors of the full grid's on each of five fleets, and its
  standard errors within 0.3%. A fleet of 60 items with four varying
  parameters gave estimates within 0.001 standard errors and standard
  errors within 0.3%, in 9 seconds where the full grid took 555.
- A recovery study of 200 such fleets of 100 items (four varying
  parameters, all six correlations estimated, their truth 0;
  ``scripts/studies/joint_recovery/study.py cubic`` with
  ``INTEGRATION=laplace``, ``summary_cubic_laplace.json``) took about 45
  seconds a fleet. 61 of the 200 fits had no standard errors: the
  estimated correlation matrix ran to the edge of where a correlation
  matrix can be (on one, its smallest eigenvalue was 2e-4), as six
  correlations among 100 items can; three of them, fitted again with
  ``random_effects="diagonal"``, reached a verified maximum with standard
  errors. On the 139 that had them, the 95% intervals covered the truth
  89% to 98% of the time (the path's coefficients 91% to 96%, the
  marker's effect 94%), and the marker's effect was 2.7% high (Monte
  Carlo error 1.2%), the Weibull's scale 1.4% high (0.6%) and one
  correlation 0.07 low (0.03). These are of the fits that had standard
  errors, not of all 200.
- It degrades as the readings thin out and are noisy. The same pumps read
  every 90 days (at most 9 readings; 6% to 9% of the pumps with three or
  fewer): within 0.08 standard errors. Every 180 days (at most 5; 36% to
  42% with three or fewer): 0.10 to 0.18. Once a year (at most 3): 0.23 to
  0.41, and one fit of five did not reach a verified maximum. The
  difference is largest in the marker's effect. Where the readings are
  precise it holds up with few: fleets of a quadratic path with three
  varying parameters and less than a third of the pumps' noise stayed within 0.005
  standard errors with at most 4 readings an item.

A second run, its design and its rule for the warning fixed before it ran,
read 20 more fleets of the pumps every 100 to 160 days (9% to 36% of the
pumps with three readings or fewer): every fleet with up to 15% so read
was within 0.074 standard errors, and from 16% fleets were off by up to
0.164 (five of the fourteen beyond 15% were over 0.1). So the fit warns, with
the count, when more than 15% of the items have three readings or fewer,
and says to use a grid (with up to five varying
parameters) or to let fewer parameters vary. Two more things it does not
do. It does not take truncated items (late entry, or in the data only
because they failed by an age): the chance of an item being in the data is
an integral with no readings in it, which the approximation is not
accurate for, so the fit refuses and says to use a grid. And beyond five
varying parameters its predictions are also by Laplace's approximation
(each integral at its own peak), so the survival of an item with few
readings or none is approximate there; up to five they are on the full
grid.

Beyond five varying parameters the fit starts differently too. The start
from each item's own fitted path fails where a path has many parameters
and an item barely as many readings, so the items' fits are summarised by
their median and spread, and the marker's own mixed model is settled
first. A fleet of 120 items with six varying parameters (a marker that is
a sum of six smooth bumps along the age, the height of each an item's own,
uncorrelated) was recovered within 3 standard errors in about three
minutes.

Which path parameters are correlated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default every pair of varying parameters is correlated between items:
an item that starts high may rise faster, or slower, and the fit
estimates by how much (``corr_a_b``). That is one correlation for two
varying parameters, three for three, six for four and ten for five, and a
fleet of tens of items cannot pin down ten. ``random_effects=`` on the
fitter says which are correlated:

- ``"full"``: every pair (the default; the model as it always was).
- ``"diagonal"``: none. Each varying parameter varies between items
  independently of the others.
- blocks of names, such as ``[("a", "b"), ("c",)]``: those in a block are
  correlated with each other and with none outside it. Every varying
  parameter is in exactly one block; a name that is not a varying
  parameter, or one named twice or left out, is refused, saying which.

.. code:: python

    from surpyval import JointPH, Weibull
    from surpyval.degradation import QuadraticPath

    independent = JointPH(Weibull, random_effects="diagonal").fit(data)
    independent.parameter_names[:4]       # ['a', 'b', 'sd_a', 'sd_b']
    blocks = JointPH(Weibull, QuadraticPath, random_effects=[("a", "b"), ("c",)])

A correlation left out is 0, not estimated: it has no entry in
``params``, in ``summary()`` or in the saved model, and ``aic()`` counts
the parameters that are estimated. The structures nest -- the full one
holds every other with some correlations at 0 -- so they compare on the
same data by ``aic()`` or by the likelihood ratio. The model keeps the
structure as ``random_effects_structure`` (``"full"``, ``"diagonal"``,
or the blocks as a tuple of tuples of names), and its printout and the
summary's heading of the spread say it. A model with a structure other
than the full one saves as schema 5, which SurPyval 0.28 refuses rather
than read its parameters as a full one's; a model saved before the
choice existed reads as ``"full"``.

A spread at 0: a parameter that does not vary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes a parameter named in ``varying`` does not, in these data, vary
between items: every item's rate looks the same, say. Its spread is then
estimated at 0, the lower edge of what a spread can be, and the model is
the same as the one with that parameter common to all items. Two things
follow, and SurPyval handles both.

First, the search. It works on the log of each sd, so an sd of 0 sits at
minus infinity, where the likelihood is flat: the search can wander
towards it and stall before the other parameters reach their best
values, ending (unverified) below the fit with the parameter common. A
fit of the larger model should never be worse than the smaller model it
contains. So, where a fit puts a spread at 0 (it moves no item's marker
by as much as 1% of the readings' noise at any reading age), or ends
unverified with a spread whose removal costs the log-likelihood less
than 1.92 (the data cannot tell it from 0 at the 5% level), the fit is
made again with those parameters common to all items, and then the full
model is searched once more from there, with their spreads small (or,
where the first answer already beats the fit on the edge, from that
answer). If that search rises off the edge (by more than 1e-4 in
log-likelihood, an absolute amount: round-off of the adaptive nodes is
below it), its answer is kept. Otherwise the fit is held on the edge.
Either way the fit reported is never worse than any candidate the
searches reached by more than 1e-4 in log-likelihood. A verified fit with every spread clear of
0 is not searched again, so ordinary fits cost nothing more. Ask whether
the parameter varies by comparing it with the fit where it is common, by
``aic()`` or ``likelihood_ratio_test`` (below).

Second, what is reported. A fit held on the edge says so:

- ``model.spreads_at_zero`` names the parameters, and the printout's
  ``Fit`` line says ``with sd_rate at the lower bound, 0``.
- ``sd_<name>`` is exactly 0, and its correlations with the other varying
  parameters are 0. None of them has a standard error (``nan``): a Wald
  interval needs the likelihood to curve on both sides of the estimate,
  and at an edge there is only one side.
- Everything else (the other parameters, their standard errors and
  bounds, ``neg_ll()``, predictions) is that of the fit with the
  parameter common to all items. ``aic()`` still counts the spread's
  parameters, so it is that fit's AIC plus 2 for each.
- ``maximum`` is that fit's, usually ``"verified"``, with no warning.

To ask whether the parameter varies at all, compare the two fits:
``aic()``, or the likelihood-ratio test, ``varying_fit.likelihood_ratio_test(common_fit)``.
The test knows the null value is on an edge: with one parameter left
out, its p-value is half that of chi-squared on the correlations left
out and half on those and the sd (Self and Liang, 1987; Stram and Lee,
1994), smaller than the plain chi-squared p-value. For a spread held at 0
the statistic is 0 and the p-value 1.

A correlation of +-1: one parameter a multiple of another
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spread has a second edge. Two varying parameters can be perfectly
correlated: each item's rate, say, is its level's departure from the
fleet's mean times a fixed number, so an item that starts high rises
slowly by exactly as much as its start predicts. Nothing of the rate's
spread is then its own. This is common where a parameter barely varies
in truth: on straight lines whose rates are all the same, the fit often
estimates a small rate spread tied entirely to the level (a correlation
of -1) rather than none at all, because that fits the readings' small
wobbles a little better.

The search meets this edge as it meets a spread of 0: the part of the
rate's spread that the level's does not explain is, on the search's log
scale, at minus infinity, so the search stalls short of it and ends
unverified. So, where that part moves no item's marker by 1% of the
noise at any reading age, or the search ends unverified and removing
that part costs the log-likelihood less than 1.92, the fit is made on
the edge: the level is integrated over, item by item, and the rate is
the fitted multiple of its departure. Then the full model is searched
from there with the rate given a small part of its own, as for a spread
at 0, and kept if the likelihood rises off the edge by more than 1e-4.

A fit held on this edge says so:

- ``model.perfect_correlations`` names the correlation
  (``["corr_a_b"]``), and the printout's ``Fit`` line says ``with
  corr_a_b at its edge, -1 (b an exact multiple of a about their
  means)``.
- The correlation is exactly +1 or -1, with no standard error
  (``nan``), for the reason a spread at 0 has none. The held parameter's
  sd is the multiple times the other's, with its standard error; its
  correlations with any other varying parameter are the other one's,
  times that sign.
- Everything else (estimates, standard errors, ``neg_ll()``,
  predictions, ``maximum``) is the fit on the edge's, and ``aic()``
  counts the correlation as a parameter.
- ``likelihood_ratio_test`` refuses to test a correlation of +-1 against
  a free one. The two fits have the same parameters, and the null value
  is on the edge of the valid matrices, where the statistic's reference
  is a mixture of chi-squared distributions whose weights depend on the
  information matrix; that is not built. The held fit is the maximum of
  the free model, so its likelihood is the free model's.

Only a pair is held: one parameter a multiple of one other. A spread of
three or more parameters can also sit on the edge without any pair at
+-1 (its correlation matrix singular in a direction mixing them all);
such a fit is not held, and can end unverified with the warning.

Within about 1e-10 of a singular spread the grid's nodes, placed by the
inverse of the integrand's curvature, need a matrix too ill-conditioned
to invert as it is (its smallest eigenvalue below a millionth of a
millionth of its largest, in each parameter's own scale). There the
curvature is taken in the prior's own coordinates instead, where it is
well conditioned. Before, the nodes were placed too wide in the narrow
direction and the grid overstated the likelihood at such points; only a
fit that landed exactly on a singular spread was affected, and it ended
unverified.

The data: readings on the item records
--------------------------------------

The readings are the ``markers`` of each item's
:class:`~surpyval.ItemLifetime`, ``{age: value}``, taken while it was
observed (from its ``left_truncation``, else 0, to its failure or
censoring). Here is a fleet of 300 pumps followed for two years and read
monthly, simulated from the model: vibration starts near 3 mm/s and rises
about 1 mm/s a year, each pump at its own level and rate, read with noise
of 0.7 mm/s; each mm/s multiplies the hazard by :math:`e^{0.9} = 2.5`.

.. jupyter-execute::

    import numpy as np
    from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH

    rng = np.random.default_rng(42)


    def pump(k):
        start = rng.normal(3.0, 0.6)             # vibration when new, mm/s
        rate = rng.normal(1.0, 0.5) / 365        # its rise, mm/s a day
        # hazard: Weibull (800 days, shape 2) at 4 mm/s, x e^0.9 per mm/s
        t = np.linspace(0, 3000, 30001)
        h = (2 / 800) * (t / 800) * np.exp(0.9 * (start + rate * t - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(rng.exponential(), H, t)
        end = min(life, 730.0)                    # followed for two years
        ages = np.arange(0.0, end, 30.0)          # read monthly
        readings = start + rate * ages + rng.normal(0, 0.7, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(
            f"P{k}",
            **{outcome: round(end, 1)},
            markers=dict(zip(ages, readings.round(2))),
        )


    fleet = SurpyvalData.from_items([pump(k) for k in range(300)])
    print(fleet.summary())

Each record holds its readings, and checks them: a reading after the
item's failure or censoring, or before it came under observation, is
refused, naming the item. A new reading of an item still running is added
with ``update_markers`` (after ``extend``-ing it to that age), which returns a
new record:

.. jupyter-execute::

    p = fleet.item("P7")
    print(p.outcome, p.value, len(p.markers), "readings")

    pump_now = ItemLifetime("P900", right_censored=60, markers={0: 3.1, 30: 3.4})
    pump_now.extend(90).update_markers(90, 3.9)

.. jupyter-execute::
    :raises: ValueError

    pump_now.update_markers(120, 4.2)

The dataset carries the readings through every reshape:
``fleet.censor_at(365)`` keeps those up to 365 days, ``enter_at`` those from
the new start, ``select`` and slicing those of the items kept, and
``to_json`` writes them. Every other fit refuses data that holds markers --
it would otherwise ignore them -- and says how to leave them out:

.. jupyter-execute::
    :raises: ValueError

    from surpyval import Weibull

    Weibull.fit(fleet)

.. jupyter-execute::

    Weibull.fit(fleet.without("markers")).params

Fitting
-------

``WeibullJointPH.fit`` takes the dataset, as every fitter does:

.. jupyter-execute::

    model = WeibullJointPH.fit(fleet, unit="days")
    model

The path recovers the simulation (a line from ``a`` = 3 mm/s, rising ``b``
= 1/365 = 0.00274 mm/s a day, the pumps' levels spread by ``sd_a`` = 0.6 and
rates by ``sd_b`` = 0.5/365, noise ``sd_noise`` = 0.7), and the survival part
a marker effect of about 0.9. The names come from the parts: the path
model's own parameter names (``a``, ``b``), ``sd_<name>`` and
``corr_<name>_<name>`` for their spread between items, ``sd_noise``, the
Weibull ``alpha`` and ``beta`` (the heading says at which marker value
``alpha`` is), each covariate's coefficient in the hazard (``<cov>``) and
shifts of the path means (``a_<cov>``, ``b_<cov>``), and
``marker_effect``. The data can also be given as arrays, as for every
fitter -- ``WeibullJointPH.fit(x, c=c, Z=Z, markers=[{age: value, ...},
...])``, one ``{age: value}`` per item -- or as a DataFrame of one row per
item, ``fit_from_df(df, x_col=..., markers_col=..., c_col=...,
Z_cols=[...])``, its ``markers_col`` holding each item's ``{age: value}``.
The model has ``params`` (named by ``parameter_names``), ``covariance()``,
``standard_errors()``, ``neg_ll()``, ``aic()``, ``bic()`` and
``summary()``, as every model does:

.. jupyter-execute::

    model.summary()[["coef", "se(coef)", "exp(coef)"]].round(4)

``neg_ll()`` is the negative log-likelihood of the readings and the lifetimes
together, so ``aic()`` and ``bic()`` compare joint models of the same data;
they are not comparable with a model of the lifetimes alone.

The time-varying covariate fit, by contrast, with each pump's last reading
carried forward (each reading a step of its ``time_varying_covariates``):

.. jupyter-execute::

    from surpyval import WeibullPH

    centre = model.marker_reference
    last_reading = SurpyvalData.from_items(
        [
            item.clone(
                markers=None,
                time_varying_covariates={
                    age: [value - centre] for age, value in item.markers.items()
                },
            )
            for item in fleet.to_items()
        ]
    )
    naive = WeibullPH.fit(last_reading)
    print(f"last reading carried forward: {naive.params[-1]:.2f}")
    print(f"joint model:                  {model.params[-1]:.2f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(naive.params[-1], 2) == 0.55
    assert round(model.params[-1], 2) == 0.98
    assert round(model.standard_errors()[-1], 2) == 0.10
    assert abs(model.params[-1] - 0.9) < model.standard_errors()[-1]
    assert model.maximum == "verified"
    est = dict(zip(model.parameter_names, model.params))
    assert round(est["a"], 1) == 3.0
    assert round(est["b"], 5) == 0.00273
    assert round(est["sd_noise"], 1) == 0.7
    assert round(est["sd_a"], 1) == 0.6

The last-reading fit puts the marker's effect at 0.55, 39% below the 0.9
it was simulated with; the joint model's 0.98 is within its standard error
(0.10) of it. The issue's simulation study (#891) found the same over 300
such fleets: the last reading carried forward was biased by -39%, its 95%
intervals covering the truth 0.3% of the time; the joint model by +1%,
covering it 96% of the time.

From measurements
~~~~~~~~~~~~~~~~~

The readings and the lifetimes are two things, and they can be kept as
two: the life data, one row per pump, and the readings as a
:class:`~surpyval.Measurements` (see :doc:`Conventions`), built from a
long table of one row per reading. The fit matches the two by the pump's
id, never by position, so the tables need not be in the same order:

.. jupyter-execute::

    import pandas as pd
    from surpyval import Measurements

    readings = pd.DataFrame(
        [
            (pump_id, age, value)
            for pump_id, held in fleet.markers.items()
            for age, value in held.items()
        ],
        columns=["pump", "day", "vibration"],
    ).sample(frac=1.0, random_state=0)
    vibration = Measurements.from_dataframe(
        readings, i_col="pump", x_col="day", m_col="vibration"
    )
    life = fleet.without("markers")
    separate = WeibullJointPH.fit(life, m=vibration, unit="days")
    separate.params[-1], model.params[-1]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.array_equal(separate.params, model.params)

``m`` is the other name of ``markers``; a Measurements may also be given
beside arrays, with each row's unit as ``i``,
``WeibullJointPH.fit(life.x, c=life.c, i=life.i, m=vibration)``, or in a
DataFrame fit, ``fit_from_df(df, x_col=..., c_col=..., i_col="pump",
m=vibration)``. A pump with no readings is fitted on its lifetime alone,
as an item with ``{}`` is; readings of a pump with no life row are warned
of, naming it, and not used. The model takes one marker: readings of
several measurements are refused until one is chosen with
``vibration.with_measurement(name)``.

Covariates in each part
~~~~~~~~~~~~~~~~~~~~~~~

A record's ``covariates`` enter both parts by default: each path
parameter's mean (``a_<name>``, ``b_<name>``: a shift of the level, of the
rate) and the hazard (``<name>``, a log hazard ratio at a given marker).
Named covariates (``covariates={"site": 1}``) name the parameters; a
covariate the data cannot determine (constant, or a combination of others)
is aliased, its coefficients ``nan``, with one warning.

A condition often acts on one part only. A heavy load can make a pump's
vibration rise faster without making it any more likely to fail *at a
given vibration*: it acts through the marker, on the path. A different
site -- another maintenance crew, another duty -- can change the risk at a
given vibration without changing how vibration rises: it acts on the
hazard. ``hazard_covariates=[...]`` names the covariates in the hazard and
``path_covariates=`` those that shift the path, a list (every path
parameter) or ``{path parameter: [...]}`` (here only the rate ``b``); each
defaults to every covariate, ``[]`` is none, and a covariate must be in at
least one. Putting a covariate where it does not act wastes the data on
coefficients that are 0, and putting it only where it does not act biases
the others. A fleet with ``site`` acting on the hazard only and ``load``
on the rate only:

.. jupyter-execute::

    gen = np.random.default_rng(11)


    def monitored(k):
        site, load = gen.integers(0, 2, 2)
        start = gen.normal(3.0, 0.6)
        rate = (gen.normal(1.0, 0.5) + 0.6 * load) / 365  # load: faster rise
        t = np.linspace(0, 3000, 30001)
        h = (2 / 800) * (t / 800) * np.exp(
            0.5 * site + 0.9 * (start + rate * t - 4.0)  # site: more risk
        )
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(gen.exponential(), H, t)
        end = min(life, 730.0)
        ages = np.arange(0.0, end, 30.0)
        readings = start + rate * ages + gen.normal(0, 0.7, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(
            f"M{k}",
            **{outcome: round(end, 1)},
            covariates={"site": float(site), "load": float(load)},
            markers=dict(zip(ages, readings.round(2))),
        ), life


    monitored_fleet = [monitored(k) for k in range(300)]
    duty = SurpyvalData.from_items([item for item, _ in monitored_fleet])
    split = WeibullJointPH.fit(
        duty,
        hazard_covariates=["site"],
        path_covariates={"b": ["load"]},
        marker_reference=4.0,
    )
    split.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert split.maximum == "verified"
    assert split.parameter_names == [
        "a", "b", "b_load", "sd_a", "sd_b", "corr_a_b", "sd_noise",
        "alpha", "beta", "site", "marker_effect",
    ]
    est = dict(zip(split.parameter_names, split.params))
    se = dict(zip(split.parameter_names, split.standard_errors()))
    for name, truth in [("site", 0.5), ("b_load", 0.6 / 365),
                        ("marker_effect", 0.9)]:
        assert abs(est[name] - truth) < 3 * se[name], name

``b_load`` is the extra rise a day under load (0.6/365 = 0.0016 simulated)
and ``site`` the log hazard ratio of the second site at a given vibration
(0.5); there is no ``a_load``, ``b_site``, ``a_site`` or ``load``. Every
covariate is still needed to predict (``Z`` names them all), and the model
saves which part each is in. A model fitted, or saved, before the parts
existed has every covariate in both.

Late entry
~~~~~~~~~~

Items that came under observation late -- a fleet whose records start when
monitoring was installed -- are ``left_truncation``-ed, and the fit
accounts for having seen only the ones that reached that age; items in
the data only because they failed by an age (a recall of failed units, say)
are ``right_truncation``-ed, likewise.

Found failed at an inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where a failure is only found at the next inspection, the item failed
between its last two inspections: an ``interval_censored`` record
``[last seen running, found failed]``, or ``left_censored`` if it was found
failed at the first. The same pumps, inspected (and read) every 90 days:

.. jupyter-execute::

    def inspected(item, life, every=90.0):
        stops = np.arange(every, 730.0 + 1e-9, every)
        found = stops[stops >= life]
        if not found.size:
            return item.clone(right_censored=stops[-1], markers={
                a: v for a, v in item.markers.items() if a <= stops[-1]
            })
        last = found[0] - every
        readings = {a: v for a, v in item.markers.items() if a <= last}
        if last <= 0:
            return item.clone(left_censored=found[0], markers=readings)
        return item.clone(interval_censored=[last, found[0]], markers=readings)


    found = SurpyvalData.from_items([inspected(*pair) for pair in monitored_fleet])
    print((found.c == 2).sum(), "interval and", (found.c == -1).sum(), "left censored")
    from_inspections = WeibullJointPH.fit(
        found,
        hazard_covariates=["site"],
        path_covariates={"b": ["load"]},
        marker_reference=4.0,
    )
    from_inspections.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert from_inspections.maximum == "verified"
    est = dict(zip(from_inspections.parameter_names, from_inspections.params))
    se = dict(
        zip(from_inspections.parameter_names, from_inspections.standard_errors())
    )
    for name, truth in [("site", 0.5), ("b_load", 0.6 / 365),
                        ("marker_effect", 0.9)]:
        assert abs(est[name] - truth) < 3 * se[name], name

The estimates are close to the exact data's, the standard errors of the
survival part a little wider: the inspections lose when, within 90 days,
each pump failed. With arrays, an interval is a ``[lower, upper]`` row of
``x`` with ``c = 2`` (and ``fit_from_df`` takes ``xl_col`` and ``xr_col``).

.. _joint-failure-modes:

More than one way to fail: failure modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Units fail in more than one way, and a reading usually speaks to only
some of them: vibration warns of a bearing failure, not of a seal leak.
Fitting every failure alike waters the reading's effect down, and
treating the other failures as "still running" overstates the life.
Give each failure its mode (``event_type`` on the record, ``e`` with
arrays, ``e_col`` with a DataFrame) and the model has a hazard for each
mode, with a Weibull baseline, coefficients and a marker effect of its
own:

.. math::

    h_{ik}(t) = \frac{\beta_k}{\alpha_k}\left(\frac{t}{\alpha_k}
    \right)^{\beta_k - 1} \exp\left(\gamma_k' z_i + \zeta_k
    \left(m(t; \theta_i) - r\right)\right).

An item fails by whichever mode comes first, and is then no longer at
risk of the others; the marker part is one for all the modes. Here some
of the pumps' failures are seal leaks, which come at random and have
nothing to do with the vibration (made-up numbers):

.. jupyter-execute::

    rng = np.random.default_rng(7)


    def pump_with_modes(k):
        start = rng.normal(3.0, 0.6)
        rate = rng.normal(1.0, 0.5) / 365
        t = np.linspace(0, 6000, 60001)
        h = (2 / 1500) * (t / 1500) * np.exp(0.9 * (start + rate * t - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        bearing = np.interp(rng.exponential(), H, t)
        seal = 2500 * rng.weibull(1.2)
        life = min(bearing, seal)
        end = min(life, 730.0)
        ages = np.arange(0, end, 30.0)
        readings = start + rate * ages + rng.normal(0, 0.7, ages.size)
        if life > 730:
            return ItemLifetime(f"M{k}", right_censored=end,
                                markers=dict(zip(ages, readings)))
        return ItemLifetime(
            f"M{k}", observed=end, markers=dict(zip(ages, readings)),
            event_type="bearing" if bearing < seal else "seal",
        )


    with_modes = SurpyvalData.from_items([pump_with_modes(k) for k in range(300)])
    modes = WeibullJointPH.fit(with_modes, marker_reference=4.0, unit="days")
    modes

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert modes.maximum == "verified"
    assert modes.causes == ["bearing", "seal"]
    est = dict(zip(modes.parameter_names, modes.params))
    se = dict(zip(modes.parameter_names, modes.standard_errors()))
    for name, truth in [("bearing: marker_effect", 0.9),
                        ("seal: marker_effect", 0.0),
                        ("bearing: beta", 2.0), ("seal: beta", 1.2)]:
        assert abs(est[name] - truth) < 3 * se[name], name
    assert abs(est["bearing: marker_effect"]) > 3 * se["bearing: marker_effect"]
    assert abs(est["seal: marker_effect"]) < 2 * se["seal: marker_effect"]

The parameters of each mode are named ``"<mode>: <name>"``, as
``ParametricCompetingRisks`` names its causes', and the table has a part
for each mode. The bearing's marker effect is close to the simulation's
0.9; the seal's is within its standard error of 0: the vibration says
nothing about the seals. ``modes.causes`` lists the modes and
``modes.n_events_by_cause`` the failures of each. The model's ``sf`` (and
``predict``) is the survival of every mode together, the chance the
pump is still running.

Should the modes have a parameter in common, name it in ``shared``:
``"beta"``, a hazard covariate, or a link's effect. The fit with an
effect shared is nested in the fit without, and the likelihood-ratio
test of the two asks whether the reading speaks to the modes
differently:

.. jupyter-execute::

    one_effect = WeibullJointPH.fit(
        with_modes, marker_reference=4.0, shared=["marker_effect"]
    )
    test = modes.likelihood_ratio_test(one_effect)
    print(one_effect.parameter_names[-5:])
    print(round(test["statistic"], 2), test["df"], round(test["p_value"], 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert one_effect.maximum == "verified"
    assert test["df"] == 1 and test["p_value"] < 0.05

The test rejects one effect for both: the modes differ in what the
vibration says about them.

A failure found at an inspection (left or interval censored) needs its
mode too; the likelihood is then the mode's hazard times the survival of
every mode, integrated over the interval. What is refused, each saying
what to do: a failure with no mode (a failure of unknown mode is not
fitted yet), a mode on an item still running, a mode with fewer failures
than it has parameters of its own (the message gives the counts),
``"alpha"`` in ``shared``, the accelerated survival part (``JointAFT``)
and the two-stage estimate. Data with one mode only, or with no event
types, give the model of one kind of failure, exactly as before. Each
mode adds its cumulative hazard to every item's likelihood, so a fit
with two modes takes about twice as long.

The chance of each mode for a running item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a pump still running, its readings so far say not only how likely it
is to fail in the next year but by which mode: a planner wants, say,
"bearing 34%, seal 8%" to know which job to prepare. ``predict(...).cif(x, mode)``
gives the chance that the item fails by ``mode`` by age ``x``, given its
readings and that it was running at the age now (the mode's cumulative
incidence from now). The modes' chances add up to ``ff(x)``, the chance
of failing at all, and with ``sf(x)`` to 1; ``total_cif(x)`` is that sum.
``cb(x, on="cif", mode=...)`` bounds each mode's chance (the same delta
method as ``cb(on="ff")``, on ``log(-log(1 - cif))``, so the bounds stay
between 0 and 1), and ``to_dict()`` carries each mode's curve and bounds
under ``"cif"``, so a saved prediction can be drawn without the model.

.. jupyter-execute::

    pump = modes.predict(markers={0: 3.4, 120: 3.9, 240: 4.3, 360: 4.8},
                         given=400)
    year_on = 765
    for mode in modes.causes:
        lower, upper = pump.cb(year_on, on="cif", mode=mode)
        print(f"{mode}: {pump.cif(year_on, mode):.3f} ({lower:.3f} to {upper:.3f})")
    print(f"still running: {pump.sf(year_on):.3f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    every = sum(pump.cif(year_on, m) for m in modes.causes)
    assert abs(every + pump.sf(year_on) - 1) < 1e-10
    assert pump.cif(year_on, "bearing") > pump.cif(year_on, "seal")

A pump whose vibration is climbing is most at risk from its bearing.

How it is worked out: over the item's path, given its readings and its
survival to now, each mode's chance is the chance of failing by that
mode before any other, the mode's hazard times the survival of every
mode,

.. math::

    \mathrm{CIF}_k(x \mid T > u, y) = E\left[\int_u^x h_k(s \mid \theta)
    \frac{S(s \mid \theta)}{S(u \mid \theta)}\, ds
    \;\middle|\; T > u, y\right],

averaged on the nodes placed at the item's distribution given its
readings and its survival to ``u``. In practice each mode's chance is
the chance of failing at all, ``ff(x)`` (as ``sf`` computes it), times
the mode's share of it from that average, the integral over age by the
15-point rule of the cumulative hazard. The share is taken at each
node, inside the integral over age and the average over the item's path,
so the result is that expectation; the shares add to 1, so the chances
add to ``ff(x)`` exactly, and the time rule's error moves only the split
between the modes: against a brute-force integral, 1e-8 or less within
500 days of now and 1.3e-5 at 1100 days beyond it.
With the marker effects 0 the chances are ``ParametricCompetingRisks``'
cumulative incidences from the age now, ``(CIF_k(x) - CIF_k(u)) /
S(u)``, to 1.1e-11 (relative). Each mode's bounds are its own, one age at a time: the
upper bounds of the modes need not add to the upper bound on ``ff``. An
infinite age gives ``nan`` (use a late finite one for the chance of
ever failing by a mode). A model of one way of failing has
``cif(x) == ff(x)`` and refuses a ``mode``.

In RePyability a ``JointNode`` follows the item's overall ``sf``; a node
for each mode is not part of this.

The marker reference
~~~~~~~~~~~~~~~~~~~~

``alpha`` is the baseline of an item whose marker is at
``marker_reference``, by default the mean reading (3.66 mm/s here). A
round value reads more easily: ``fit(..., marker_reference=4.0)`` reports
``alpha`` for an item at 4 mm/s. Nothing else moves -- the other
parameters, the likelihood and every prediction are the same:

.. jupyter-execute::

    at_four = WeibullJointPH.fit(fleet, unit="days", marker_reference=4.0)
    print(round(model.marker_reference, 3), round(model.params[6], 1))
    print(at_four.marker_reference, round(at_four.params[6], 1))
    at_four.summary().index[6]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(np.delete(at_four.params, 6), np.delete(model.params, 6))
    assert at_four.parameter_names[6] == "alpha"

Other paths
~~~~~~~~~~~

``JointPH(Weibull, path_model)`` takes any of the degradation module's path
models, or a ``PathModel`` of your own; ``varying=`` names the path
parameters that vary between items (the others are common to all). A wear
signal growing exponentially, ``a exp(b t)``, from a level and at a rate of
each item's own:

.. jupyter-execute::

    from surpyval import JointPH, Weibull
    from surpyval.degradation import ExponentialPath, QuadraticPath


    def wearing(k):
        a, b = rng.normal(3.0, 0.4), rng.normal(0.0008, 0.0003)
        t = np.linspace(0, 3000, 30001)
        h = (2 / 800) * (t / 800) * np.exp(0.9 * (a * np.exp(b * t) - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(rng.exponential(), H, t)
        end = min(life, 730.0)
        ages = np.arange(0.0, end, 30.0)
        y = a * np.exp(b * ages) + rng.normal(0, 0.3, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, y)))


    wear = SurpyvalData.from_items([wearing(k) for k in range(200)])
    exponential = JointPH(Weibull, ExponentialPath).fit(wear, marker_reference=4.0)
    exponential.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert exponential.maximum == "verified"
    z = (exponential.params[-1] - 0.9) / exponential.standard_errors()[-1]
    assert abs(z) < 3

Each parameter is on its own scale by default: the rate ``b`` above is
normal between items, so a fleet whose rates are near 0 would give some
items a falling path. A rate that must stay positive can go on a log
scale with the degradation module's ``LinkedPathModel``: then ``log(b)``
is normal between items (``b`` log-normal), and the names say so --
``log(b)``, its spread ``sd_log(b)`` and correlation ``corr_a_log(b)``
are on the log scale, and a covariate's shift ``log(b)_<cov>`` is a log
ratio of rates:

.. jupyter-execute::

    from surpyval.degradation import LinkedPathModel

    log_rate = JointPH(Weibull, LinkedPathModel(ExponentialPath, {"b": "log"}))
    log_rate = log_rate.fit(wear, marker_reference=4.0)
    log_rate.summary()[["coef", "se(coef)"]].round(5)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert log_rate.maximum == "verified"
    assert log_rate.parameter_names[:5] == [
        "a", "log(b)", "sd_a", "sd_log(b)", "corr_a_log(b)"
    ]
    z = (log_rate.params[-1] - 0.9) / log_rate.standard_errors()[-1]
    assert abs(z) < 3

``varying=`` then names ``log(b)``. The two fits describe the same fleet
with a different spread of rates, and predict alike where the data are.

.. _joint-spline:

A reading that bends: a spline path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many readings follow no formula: flat for years and then a knee, or a
fall after an overhaul and then a rise. ``SplinePath`` is a smooth curve
of joined pieces (a natural cubic spline) that bends where the readings
do. The fleet's mean curve is the spline, and each item differs from it
by a level and a rate of its own:

.. math::

    m_i(t) = \text{level}_i + \text{rate}_i\, t + \sum_k s_k N_k(t).

The pieces join at *knots*. ``SplinePath.natural(df=3)`` places two
between the first and last reading ages, at the thirds of all the reading
ages (as R's ``ns(time, 3)`` does), and the fitted model's ``path_model``
records them; ``SplinePath(knots=[...], boundary=(lo, hi))`` takes knots
you choose, at an age where the reading is known to turn, say. Each
:math:`N_k` is 0 up to its knot, so ``level`` and ``rate`` are the level
at age 0 and the rate before the first knot, and each ``s_k`` (in the
reading's unit) bends the curve after its knot. With no knots at all
(``SplinePath()``, or ``df=1``) the path is the straight line, and the
fit is ``LinearPath``'s to the last digit.

A fleet of 150 pumps whose vibration stays near 3 mm/s for a year and
then climbs, fitted with one knot at the year, where the readings are
known to turn:

.. jupyter-execute::

    from surpyval.degradation import LinearPath, SplinePath


    knee_rng = np.random.default_rng(452)


    def knee(k):
        level, climb = knee_rng.normal(3.0, 0.5), knee_rng.normal(0.006, 0.001)
        t = np.linspace(0, 6000, 30001)
        m = level + climb * np.maximum(t - 365.0, 0.0)
        h = (2 / 800) * (t / 800) * np.exp(0.9 * (m - 4.0))
        H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
        life = np.interp(knee_rng.exponential(), H, t)
        end = min(life, 730.0)
        ages = np.arange(0.0, end, 30.0)
        y = level + climb * np.maximum(ages - 365.0, 0.0)
        y = y + knee_rng.normal(0, 0.3, ages.size)
        outcome = "observed" if life <= 730 else "right_censored"
        return ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, y)))


    bent = SurpyvalData.from_items([knee(k) for k in range(150)])
    turn = SplinePath(knots=[365.0], boundary=(0.0, 730.0))
    spline = JointPH(Weibull, turn, varying=("level", "rate"))
    spline = spline.fit(bent, marker_reference=4.0)
    line = JointPH(Weibull, LinearPath).fit(bent, marker_reference=4.0)
    print(spline.path_model)
    print(round(line.aic() - spline.aic(), 1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert spline.maximum == "verified"
    assert spline.parameter_names[:3] == ["level", "rate", "s1"]
    assert line.aic() - spline.aic() > 50

``aic()`` compares the two fits of the same data: the line is far behind.
What it costs is in the predictions. A pump read to 450 days, 85 days
past the turn, whose vibration has started to climb:

.. jupyter-execute::

    seen = np.arange(0.0, 451.0, 30.0)
    turning = dict(zip(seen, 3.0 + 0.006 * np.maximum(seen - 365.0, 0.0)))
    for fitted in (spline, line):
        ahead = fitted.predict(markers=turning, given=450.0)
        print(
            fitted.path_model.name,
            round(float(ahead.marker(450.0)), 2),
            round(float(ahead.marker(630.0)), 2),
            round(float(ahead.sf(630.0)), 3),
        )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    truth_450, truth_630 = 3.0 + 0.006 * 85, 3.0 + 0.006 * 265
    by_spline = spline.predict(markers=turning, given=450.0)
    by_line = line.predict(markers=turning, given=450.0)
    assert abs(by_spline.marker(630.0) - truth_630) < abs(
        by_line.marker(630.0) - truth_630
    )
    assert abs(by_spline.marker(450.0) - truth_450) < 0.25

The pump's vibration is truly 3.51 mm/s at 450 days and 4.59 at 630.
Both fits read the level at 450 days about right, but the line cannot
turn with the readings: it projects 3.84 at 630, where the spline
projects 4.5. The chances of reaching 630 days are close here (0.734
and 0.737): each fit's baseline hazard has absorbed its own picture of
the fleet's path, so the line's error shows first in the marker it
forecasts.

**How well it recovers the truth.** A simulation study (run on
2026-11-25 with ``scripts/studies/joint_recovery/knee_study.py``, whose
summary is beside it; this page does not re-run it) drew 200 fleets of
300 pumps on a knee at 365 days: level 3 (sd 0.5), rate 0 (sd 0.3 a
year), a common bend ``s1`` of 3, readings every 30 days to 730 with
noise sd 0.4, and the hazard above (``alpha`` 800, ``beta`` 2, marker
effect 0.9). Each fleet was fitted with the spline and with a line.

- All 200 spline fits reached a verified maximum, and ``aic()``
  preferred the spline in all 200.
- The spline recovered every parameter. The largest bias was the marker
  effect's, +1.7% (Monte Carlo se 0.8%), then ``alpha``'s, +0.5% (0.3%).
  The 95% intervals covered the truth 93.5% to 98% of the time, and the
  spread of the estimates was 0.91 to 1.08 times their standard errors.
- The line is wrong about the risk itself. It takes the marker effect
  as 0.63, not 0.9 (-30%), ``beta`` as 2.61, not 2 (+31%), and ``alpha``
  as 750, not 800 (-6%). Its intervals cover the true values only 31%
  to 44% of the time.
- Predicting from the readings to 450 days, for the first 40 pumps still
  running then (8,000 predictions), the spline's marker at 450 days was
  off by 0.16 (root mean square) with no bias (-0.0002, se 0.002). The
  line's was off by 0.17, and read 0.056 low on average (se 0.002).
  Survival to 540 days (on average 0.87) was off by 0.025 with the
  spline and 0.035 with the line, unbiased in both: the line's baseline
  hazard absorbs its picture of the path, as above.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _s, _l = by_spline, by_line
    assert round(float(_l.marker(630.0)), 2) == 3.84, _l.marker(630.0)
    assert round(float(_s.marker(630.0)), 1) == 4.5, _s.marker(630.0)
    assert round(float(_l.sf(630.0)), 3) == 0.734, _l.sf(630.0)
    assert round(float(_s.sf(630.0)), 3) == 0.737, _s.sf(630.0)

Choosing the knots: more knots follow more turns and need more readings.
A spline with more parameters than the items' median number of readings
is refused, saying how many knots the readings support; compare ``df``
by ``aic()``. On the fleet above, ``SplinePath.natural(df=3)`` places
its knots at 210 and 420 days (the readings stop at failure, so their
ages crowd early) and fits as well as the knot at the year (AIC 3672.9
against 3668.6), its mean curve within 0.2 mm/s of the truth at every
age read; where a turn is known, a knot there needs fewer parameters. Beyond the last knot the curve is extended as a straight
line, so a prediction far past the readings rests on the last piece's
slope: place the upper boundary where the readings end (the default).
Every link works on a spline: the slope and the area under it
(``link="slope"``, ``"cumulative"``) are in closed form, and the
integrals of the hazard over age are taken piece by piece between the
knots, where the curve's third derivative jumps (one unsplit rule was
off by up to :math:`10^{-4}` of a cumulative hazard on a bent path; the
split one is within :math:`10^{-7}`). With a ``weight`` on the
accumulated exposure the weighted area is by the quadrature above, not
split at the knots: within :math:`10^{-6}` of the area on a knee. The
bends ``s_k`` are common to all items: for a spline, ``varying``
defaults to ``("level", "rate")``, where every other path's default is
all of its parameters.

Items that bend differently: varying bends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Where the items turn at different paces (a sharper knee on some pumps
than on others), the bends can vary between items too: name them in
``varying``, ``JointPH(Weibull, SplinePath.natural(df=3),
varying=("level", "rate", "s1", "s2"))``. The default structure is then a
block of ``level`` and ``rate``, correlated with each other, and each
bend on its own: a bend's spread is told from the readings after the
knot where its piece starts, and an item's correlations with it would
need many more items to tell apart (``random_effects="full"`` asks for
them). Each varying parameter adds a dimension to every item's integral
(``integration="pruned"`` is the default from four), so such a fit takes
minutes. Compare it with the common-bend fit of the same data by
``aic()``.

A varying bend is refused where the items' median number of readings
after the knot its piece starts at is below the number of varying
parameters (the message gives the counts, and says to leave the bend out
of ``varying`` or to use knots the readings reach well past): with fewer,
the readings cannot pin down each item's own bend at all.

Whether the bends really differ between items is a question for
``aic()``. Fit the model twice, once with the bend common to all items
and once with it in ``varying``, and compare ``common.aic()`` with
``varying.aic()``; the lower is the better account of the data for its
size. The common-bend model is the varying one with the bend's spread
held at 0 (the likelihoods are then equal), so the varying fit has one
parameter more. No warning is given where a bend's spread is estimated
at 0: the study below shows the readings usually cannot tell a small
spread from none, so a warning on the estimate alone misled. On such
data the search for the larger fit could stop short of the common-bend
fit's likelihood, with a spread run to 0 (#1334). A spread at 0 is now
held on its edge, as for any varying parameter (`A spread at 0: a
parameter that does not vary`_): the fit is then the common-bend fit,
with ``sd_s1`` exactly 0 and named in ``spreads_at_zero``, so a varying
fit reaches at least the likelihood of the common-bend fit it contains.

**How well it recovers the truth.** A simulation study (run on
2026-11-25 with ``scripts/studies/joint_recovery/shape_study.py``, whose
summary is beside it; this page does not re-run it) drew 200 fleets of
150 pumps. Each pump turned up at its own pace after a year: level 3
(sd 0.5), rate 0 (sd 0.3 a year), bend ``s1`` 3 with sd 1.0, all
independent, read every 30 days to 730 with noise sd 0.4, and the hazard
above. Each fleet was fitted with the bend common and with it varying.

- 198 of the 200 fleets were fitted both ways; in the other two a fit
  ended with no covariance. Refitted with the current code, both fit
  both ways. In one of them (seed 2123) the varying fit had stopped
  short, with ``sd_rate`` run to 0 and an AIC 3.2 worse than the common
  fit's: the stall described above (#1334). With the second search it
  reaches a verified maximum 20.0 higher in log-likelihood than before,
  with an AIC 37 better than the common fit's.
- All 198 varying fits reached a verified maximum, and ``aic()``
  preferred the varying bend in all 198 (by 48.5 at the median, 9.8 at
  the least): here the spread is real.
- The varying fit recovered every parameter. ``sd_s1`` averaged 0.996
  against 1.0, with 96% coverage. The marker effect's bias was +1.5%
  (se 1.2%), and the intervals covered 91% to 97% of the time.
- The common-bend fit, missing the bend's spread, moves it elsewhere.
  ``sd_rate`` comes out 44% too high and its interval never covers the
  truth. The level-rate correlation reads -0.16, not 0 (60% coverage),
  and the bend 4% low (71% coverage). Its survival part holds up: the
  marker effect +2.7% (se 1.2%), 97% coverage.

So where the items' knees differ, let them vary and compare by
``aic()``.

**Where the bends are alike, or nearly.** A second study
(``scripts/studies/joint_recovery/bend_null_study.py``, its design fixed
before it ran, run on 2026-11-26) fitted the same fleets with the bend's
spread between pumps 0, 0.1 and 0.3 (against 1.0 above). This run is a
reduced one, 10 fleets for each spread, so each share below is uncertain
by about 15 points; the design has 200.

- With no spread (sd 0), the spread was estimated at 0 in 4 of 10
  fleets, and ``aic()`` preferred the varying fit in 2. A spread that is
  truly 0 is estimated above 0 in about half of fleets, as any variance
  at its boundary is, so an estimate of 0 is no reliable sign of it;
  ``aic()`` preferred the common bend in 8 of 10.
- With a small spread, the spread was estimated at 0 in 5 of 10 fleets at
  sd 0.1 and 2 of 10 at sd 0.3, and ``aic()`` preferred the varying fit in
  3 and 1. Spreads this small are not told from none by 150 pumps read
  monthly for two years.
- No varying fit fell short of the common-bend fit's likelihood (30 of
  30, with the second search from the common-bend fit that this run
  had, which the boundary fit above has since replaced).
- The readings rule refused none of the fleets, even cut at 450 or 540
  days. The bend ``s1`` of a one-knot spline acts from the start of the
  boundary, so every reading counts (a median of 15 or 18 per pump), and
  the rule is met.

So an estimate of 0 says little either way, and SurPyval gives no
warning on it (it did until this study: the warning fired at the rates
above, missing most fleets with no spread and firing on many with
some). Compare the common-bend and varying-bend fits by ``aic()``.

What the hazard follows: the link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far the risk has followed the marker's *level*: a pump at 5 mm/s is at
more risk than one at 3 mm/s, however each got there. That is one of three
things a marker can say about risk, and ``JointPH(..., link=...)`` chooses
between them:

- ``link="value"`` (the default): **risk follows the level.** What matters
  is where the marker is now. A vibration level that is a symptom of the
  present damage; a biomarker that measures how ill the patient is today.
- ``link="slope"``: **risk follows how fast it is changing.** What matters
  is the marker's rate of rise, not where it stands. Bearings that settle
  at different vibration levels by their mounting, where a level that is
  *climbing* is the sign of trouble; a tumour marker whose doubling time,
  not its level, says how aggressive the disease is.
- ``link="cumulative"``: **risk follows the accumulated exposure.** What
  matters is the total so far, the area under the marker's path from age
  0: every hour at a high reading adds to the damage, and it does not go
  away when the reading falls. A load, temperature or contamination
  reading where the wear done is the dose received; a patient's cumulative
  exposure to a raised blood pressure or blood sugar.

A list gives the hazard more than one, each with an effect of its own:
``link=["value", "slope"]`` asks whether, at a given level, a marker that
is rising is at more risk than one that is steady.

In formulas, with :math:`m_i(t) = m(t; \theta_i)` the item's noise-free
path, the hazard is

.. math::

    h_i(t) = h_0(t) \exp\left(\gamma' z_i
    + \zeta_v \bigl(m_i(t) - r\bigr)
    + \zeta_s\, m_i'(t)
    + \zeta_c \int_0^t m_i(s)\, ds\right),

with only the terms of the links chosen: ``marker_effect``
(:math:`\zeta_v`) per unit of the marker, ``slope_effect``
(:math:`\zeta_s`) per unit of the marker *a unit of time*, and
``cumulative_effect`` (:math:`\zeta_c`) per unit of the marker *times a
unit of time*. The slope is the path model's own derivative (``slope``;
exact for the built-in paths, central differences for a path of one's
own without one), and the area is by 15-point Gauss--Legendre quadrature
(exact for a line or a quadratic). Everything else -- the data, the
quadrature over each item's path, the predictions, the bounds, saving --
is as before.

Three things to know:

- **The baseline.** ``alpha`` and ``beta`` are the Weibull of an item whose
  slope is 0 and whose accumulated area is 0 (the printout says so:
  ``alpha at slope = 0``). ``marker_reference`` is the value's reference
  only, and is refused where the hazard does not follow the value. The
  area is of the marker itself, from age 0: for the exposure *above* a
  level -- the hours above a rated temperature, say -- subtract that level
  from the readings first.
- **Paths whose slope starts infinite.** The cumulative hazard runs from
  age 0, so the slope link needs a path whose slope is finite there. A
  logarithmic path (slope :math:`b / t`) and a Lloyd-Lipow path (slope
  :math:`b / t^2`) are refused with the slope link when the fitter is
  built, and a power path :math:`a t^b` is refused by the fit when its
  average item's power is below 1 (slope :math:`a b t^{b - 1}`): near age
  0 the failure rate would be infinite for every item whose slope has the
  sign of its effect. Each refusal names the links the path can take
  instead, and paths whose slope is finite at 0 (a line, a quadratic, an
  exponential).
- **The units.** A slope's and an area's effects depend on the unit of
  age as well as the marker's: a ``slope_effect`` per mm/s a *day* is 365
  times one per mm/s a *year*, and a ``cumulative_effect`` per
  (mm/s)·day is 365 times one per (mm/s)·year. Ages in years (or a rate
  per year) keep them readable. Give both units -- ``unit=`` for the ages
  and ``marker_unit=`` for the readings (a ``Measurements`` read with
  ``units=`` gives its own) -- and the model says what each effect is per,
  in ``effect_units`` and on its printout's ``Effect units`` line. They
  are labels only: nothing is converted.
- **What the data can tell apart.** For a straight line the slope is the
  item's rate ``b``, constant in age: its effect is found from how
  lifetimes differ *between* items with different rates, so it needs
  rates that vary between items (``b`` in ``varying``, or shifted by a
  covariate). The links are compared by ``aic()``, the models having the
  same data, or all at once by ``compare_links`` (see `Comparing the links`_).

A fleet of gearboxes, their oil debris read monthly for two years (ages
in years). Each starts near 3 ppm and rises about 1 ppm a year, at a level
and rate of its own; what wears a gearbox out here is how fast the debris
is rising, each ppm a year multiplying the hazard by :math:`e^{1.0} = 2.7`:

.. jupyter-execute::

    gen = np.random.default_rng(5)


    def gearboxes(n, alpha, beta, slope=0.0, cumulative=0.0):
        spread = np.array([[0.36, 0.06], [0.06, 0.25]])  # sds 0.6, 0.5
        lines = gen.multivariate_normal([3.0, 1.0], spread, size=n)
        target = gen.exponential(size=n)
        t = np.linspace(0.0, 40.0, 400001)
        items = []
        for k, (level, rate) in enumerate(lines):
            log_ratio = slope * rate + cumulative * (level * t + rate * t**2 / 2)
            h = beta / alpha * (t / alpha) ** (beta - 1) * np.exp(log_ratio)
            H = np.concatenate([[0], np.cumsum((h[1:] + h[:-1]) / 2 * np.diff(t))])
            life = np.interp(target[k], H, t)
            end = min(life, 2.0)                      # followed for two years
            ages = np.arange(0.0, end + 1e-9, 1 / 12)  # read monthly
            readings = level + rate * ages + gen.normal(0, 0.7, ages.size)
            outcome = "observed" if life <= 2.0 else "right_censored"
            items.append(
                ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, readings)))
            )
        return SurpyvalData.from_items(items)


    rising = gearboxes(300, alpha=3.0, beta=2.0, slope=1.0)
    by_slope = JointPH(Weibull, link="slope").fit(rising, unit="years")
    by_slope.summary()[["coef", "se(coef)", "exp(coef)"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_slope.maximum == "verified"
    assert by_slope.parameter_names[-1] == "slope_effect"
    z = (by_slope.params[-1] - 1.0) / by_slope.standard_errors()[-1]
    assert abs(z) < 3
    assert round(by_slope.params[-1], 2) == 0.96

The ``slope_effect`` recovers the 1.0 it was simulated with. The model
that follows the level instead fits the same fleet worse:

.. jupyter-execute::

    by_level = JointPH(Weibull).fit(rising, unit="years")
    print(f"AIC, slope link: {by_slope.aic():.1f}")
    print(f"AIC, value link: {by_level.aic():.1f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_slope.aic() < by_level.aic()

Two gearboxes at the same reading now, one steady and one climbing, are
then at different risk:

.. jupyter-execute::

    steady = {0.0: 3.5, 0.3: 3.5, 0.6: 3.6, 0.9: 3.5}
    climbing = {0.0: 2.6, 0.3: 3.0, 0.6: 3.3, 0.9: 3.6}
    for name, readings in [("steady", steady), ("climbing", climbing)]:
        print(name, by_slope.ff(1.5, markers=readings, given=0.9).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_slope.ff(1.5, markers=climbing, given=0.9) > by_slope.ff(
        1.5, markers=steady, given=0.9
    )

The same fleet with the damage accumulating instead -- every ppm-year of
debris so far multiplying the hazard by :math:`e^{0.3}`:

.. jupyter-execute::

    worn = gearboxes(300, alpha=4.0, beta=1.5, cumulative=0.3)
    by_area = JointPH(Weibull, link="cumulative").fit(
        worn, unit="years", marker_unit="ppm"
    )
    by_area.summary()[["coef", "se(coef)", "exp(coef)"]].round(3).tail(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert by_area.maximum == "verified"
    z = (by_area.params[-1] - 0.3) / by_area.standard_errors()[-1]
    assert abs(z) < 3
    assert by_area.effect_units == {"cumulative_effect": "ppm·years"}

With the units given, the model says what the effect is per: each
ppm·year of debris so far multiplies the hazard by ``exp(coef)``.

.. jupyter-execute::

    by_area.effect_units

A medical example: the Mayo Clinic trial in primary biliary cirrhosis
(``load_pbc2``), 312 patients whose serum bilirubin was measured at each
visit, followed to death (a transplant or the end of follow-up is a
censoring). Does a bilirubin that is *rising* say more than its level
does? The log of the bilirubin as a line in years, the hazard following
both its level and its slope:

.. jupyter-execute::

    from surpyval.datasets import load_pbc2

    pbc = load_pbc2()
    pbc["log_bilirubin"] = np.log(pbc["serBilir"])
    patients = pbc.groupby("id").agg(
        years=("years", "first"), died=("status2", "first")
    )
    patients["censored"] = 1 - patients["died"]
    patients["readings"] = [
        dict(zip(visits["year"], visits["log_bilirubin"]))
        for _, visits in pbc.groupby("id")
    ]
    level_and_slope = JointPH(Weibull, link=["value", "slope"]).fit_from_df(
        patients,
        x_col="years",
        c_col="censored",
        markers_col="readings",
        marker_reference=0.0,
        unit="years",
    )
    level_and_slope.summary()[["coef", "se(coef)", "exp(coef)", "p"]].round(3).tail(4)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert level_and_slope.maximum == "verified"
    est = dict(zip(level_and_slope.parameter_names, level_and_slope.params))
    se = dict(
        zip(level_and_slope.parameter_names, level_and_slope.standard_errors())
    )
    assert round(est["marker_effect"], 2) == 1.03
    assert round(est["slope_effect"], 1) == 3.0
    assert est["slope_effect"] > 2.5 * se["slope_effect"]

Both matter. A patient whose log bilirubin is one unit higher is at about
2.8 times the hazard, and at the same level one whose log bilirubin is
rising 0.1 a year faster is at :math:`e^{0.30} = 1.3` times the hazard.
R's ``JM`` (``jointModel`` with ``parameterization = "both"``) gives the
same estimates on these data, as it does for the value, the slope and the
cumulative links alone: the comparison, with the treatment in the hazard,
is ``tests/reference/test_joint_links.py``.

Comparing the links
~~~~~~~~~~~~~~~~~~~

Which link a marker's risk follows is a question about each dataset, and
the data answer it by comparing fits. A fitter's ``compare_links`` fits
the same model -- its baseline, path model and varying parameters -- with
each link to the same data, in one call. By default it fits five: the
value, the slope and the area alone, and the slope and the area each beside
the value (``links=[...]`` chooses others, each a link or a list of them).
On the gearboxes whose risk follows how fast the debris rises:

.. jupyter-execute::

    links = JointPH(Weibull).compare_links(rising, unit="years")
    links.table[["k", "log_likelihood", "aic", "delta", "weight"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert links.best_link == ("slope",)
    assert list(links.table.index[:2]) == ["slope", "value + slope"]
    assert links.table["weight"].iloc[0] > 0.6

The table is ranked by AIC (``metric="bic"`` or ``"aic_c"`` for another
criterion), best first. AIC is :math:`2k - 2 \log L`, with :math:`L` the
likelihood of the readings and the lifetimes together: the links' AICs
compare with each other, not with that of a model of the lifetimes alone.
``delta`` is each link's AIC less the best's and ``weight`` its Akaike
weight, its share of :math:`e^{-\Delta/2}`: the slope link has about two
thirds and the value with the slope the rest, the links without the slope
almost none. ``links.best`` is the best fit, and
``links.models["value + slope"]`` any other.

Two links less than 2 apart are no real preference, and the simpler of
them is the best: the one with fewer parameters, and on a tie the level
before the slope before the area. So a link one effect richer than another
is the best only where it is more than 2 lower, which, by AIC, is where the
likelihood-ratio test of the extra effect rejects it at the 4.55% level; a
``delta`` can be below 0 for a richer link the data do not prefer. The
comparison says the same in words, row by row (``reads_as``: "as good: no
real preference" within 2, "worse" from 2 to 10, "clearly worse" beyond,
after Burnham and Anderson), and as a whole:

.. jupyter-execute::

    print(links.verdict)
    links.table[["delta", "lr_against", "lr_p_value", "reads_as"]].round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert links.verdict.startswith(
        "Risk follows how fast the marker is changing. Adding the level "
        "does not improve the fit."
    )

``lr_against`` and ``lr_p_value`` give, for a link with another nested in
it, the likelihood-ratio test of its extra effects against that one (the
best, where the best is nested in it). ``links.to_dict()`` holds the
verdict, the table and the tests as strict JSON; save a fitted model with
its own ``to_dict``, as ``links.best.to_dict()``.

A link whose terms are all in another's is that model with the extra
effects at 0, so the two are also compared by a likelihood-ratio test:
twice the gain in log-likelihood, on one degree of freedom an extra effect.

.. jupyter-execute::

    links.likelihood_ratio_tests().round(4)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    tests = links.likelihood_ratio_tests().set_index(["simpler", "richer"])
    assert tests.loc[("value", "value + slope"), "p_value"] < 0.01
    assert tests.loc[("slope", "value + slope"), "p_value"] > 0.05

The slope adds to the value (p = 0.0002) and the value adds nothing to the
slope (p = 0.37): risk here follows how fast the debris is rising, not
where it stands. Links that are not nested, such as the value and the
slope, are compared by the table only.

Each candidate is fitted as its own ``fit`` would fit it, to the same
maximum. The value link is fitted first, and the others start from its
estimates of the path (the path changes little from link to link), which
saves about a third of the time; a fit from that start that does not reach
a verified maximum is made again from its own start. A fit that is not a verified maximum is shown in the table as
``set aside`` and is neither ranked nor tested, and a link that cannot be
fitted is listed with the reason (``links.failed``); each is said in one
warning. A path that cannot take some of the default links is compared
without them, saying which: a logarithmic path without the slope, and a
Lloyd-Lipow path, which also has no finite area from age 0, can take only
the value, so there is nothing to compare and it is refused.

A reading that acts later: a lag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every link so far acts at once: the hazard at age :math:`t` follows the
path at :math:`t`. Some mechanisms have a lead time -- debris in the oil
now says what the bearing will do some hundreds of hours on. ``lag=``
shifts every link by a fixed time, in the lifetimes' unit:
``JointPH(Weibull, link="value", lag=200.0)`` makes the hazard at age
:math:`t` follow the path at :math:`\max(t - 200, 0)`. Until the lag
has passed the path is held at its age-0 value: the value link follows
the value at age 0, and the slope and the area are 0 there. The
cumulative hazard is integrated on each side of the lag (where the
lagged path has a kink, and the slope link a jump), so the likelihood
keeps its accuracy. ``JointAFT`` takes ``lag=`` the same way. With
``lag=0`` (the default) the model is the one without a lag, number for
number. The fitted model keeps its ``lag``, prints it, saves it and
predicts with it.

The lag is fixed, not estimated: the likelihood has a kink in it at every
item's age, so it is not smooth in the lag. Choose it from what you know
of the mechanism, or fit the lags you have in mind and compare them by
AIC (the same data and the same number of parameters):

.. code-block:: python

    fits = {lag: JointPH(Weibull, lag=lag).fit(data) for lag in (0.0, 0.25, 0.5)}
    {lag: model.aic() for lag, model in fits.items()}

Expect the data to say little about a lag when the marker is close to a
straight line: the line ``L`` earlier is the line now less ``b L``, which
the marker's effect and the baseline largely absorb. On 300 simulated
items whose hazard followed the line half a year earlier, the AICs of
lags 0, 0.25 and 0.5 were within 0.6 of each other. A path that is
infinite at age 0 (logarithmic, Lloyd-Lipow) cannot lag its value, and is
refused.

Exposure that fades: a weight on the accumulated exposure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``"cumulative"`` link follows the area under the path from age 0, so
an hour of exposure ten years ago counts like one last week. Where
damage heals, is repaired or washes out, that is wrong: recent exposure
matters and old exposure does not. ``weight=`` says how much of the
exposure still counts, by how long ago it was:

.. math::
    \int_0^t w(t - s)\, m(s)\, ds, \qquad
    w(u) = e^{-u / \mathrm{scale}} \quad \text{or} \quad
    w(u) = 1 \text{ for } u \le \mathrm{width},\ 0 \text{ beyond.}

.. code-block:: python

    # exposure fades: what was accumulated u ago counts exp(-u / 0.5)
    faded = JointPH(Weibull, link="cumulative", weight=("exponential", 0.5))
    # only the last year's exposure counts
    recent = JointPH(Weibull, link="cumulative", weight=("window", 1.0))
    plain = JointPH(Weibull, link="cumulative")
    fits = {name: f.fit(data) for name, f in
            {"faded": faded, "recent": recent, "all": plain}.items()}
    {name: model.aic() for name, model in fits.items()}

The scale and the width are in the lifetimes' unit of time. Like the lag
they are fixed, not estimated: fit the ones you have in mind and compare
them by AIC (the same data, the same number of parameters). A weight
needs the cumulative link (alone or with others; the weight is on the
area only) and is refused without it. ``cumulative_effect`` is still per
unit of the marker times a unit of time (``effect_units``): with an
exponential weight, a marker held at a level :math:`m` for much longer
than the scale has a weighted area of :math:`m \times \mathrm{scale}`,
so the weighted area is then the recent level in other units; with a
window, the area is the mean level over the window times its width. A
window wider than every age in the data is the plain cumulative link.
With a ``lag`` the weighted exposure acts ``lag`` later. The fitted
model keeps its ``weight``, prints it, saves it and predicts with it;
``compare_links`` keeps it on the candidates that have the cumulative
link.

The weighted area is by quadrature for every path: a window by the plain
area's 15-point rule over the window, an exponential weight over the
last 30 scales in three pieces of 15 points (what is older counts less
than :math:`10^{-13}`); both agree with the closed forms for line,
quadratic and exponential paths to :math:`10^{-9}`. With a window the
cumulative hazard is also integrated on each side of the age at which
the window stops reaching back to age 0, where the weighted area has a
kink.

A path's parameters can be held common: ``JointPH(Weibull, QuadraticPath,
varying=("a", "b"))`` gives each item its own level and rate, and every item
the same curvature ``c``. Up to five varying parameters are integrated on
a grid of 7 points in each, and more by Laplace's approximation (see
:ref:`how the integral is done <joint-integration>`). A path model fitted to a curve it cannot follow biases
the joint model, as it does a degradation model: choose it as you would
there (see :doc:`Degradation Analysis`).

How the reading acts: a higher failure rate, or faster ageing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model so far is a proportional hazards model (``JointPH``): a high
reading multiplies the failure rate for as long as it is high, and when the
reading comes back down the rate comes down with it, as though the high
stretch had never happened. ``JointAFT`` (``WeibullJointAFT`` for a
line) is the other way a reading can act: it makes the item **age
faster**. A month at a high reading uses up more than a month of life, and
the life used up is not given back when the reading falls. It is the
cumulative-exposure model of step-stress accelerated life tests, with the
measured reading as the stress, so its effects mean what they mean in an
accelerated test of the same product. It suits a reading of a stress that
consumes life -- a load, a temperature, a speed -- and a biomarker that
hastens the course of a disease; an effect reads as "a life so many times
shorter", which many find easier than a hazard ratio. (It is a statistical model of the readings and the
lifetimes, not a model of the physics of the damage.)

By age :math:`t` the item has used up the *effective age*

.. math::

    A_i(t) = \int_0^t \exp\left(\gamma' z_i
    + \zeta_v \bigl(m_i(s) - r\bigr) + \zeta_s\, m_i'(s)\right) ds,

and its survival and hazard are the baseline Weibull's at that age:

.. math::

    S_i(t) = \exp\left(-\left(\frac{A_i(t)}{\alpha}\right)^{\beta}
    \right), \qquad
    h_i(t) = \frac{\beta}{\alpha}\left(\frac{A_i(t)}{\alpha}
    \right)^{\beta - 1} A_i'(t).

A worked example, with made-up numbers. Say each ppm of oil debris above 3
ppm makes a gearbox age :math:`e^{0.3}` times as fast (``marker_effect``
0.3, ``marker_reference`` 3), and the baseline is a Weibull of scale 6
years and shape 2. At 3 ppm a gearbox ages a year a year; at 5 ppm it ages
:math:`e^{0.3 \times 2} = 1.82` years a year. One that ran two years at 3
ppm and then a year at 5 ppm has an effective age of :math:`2 + 1.82 =
3.82` years at the age of three, and a survival of :math:`\exp(-(3.82 /
6)^2) = 0.67`, against 0.78 for one that stayed at 3 ppm. If its debris
then falls back to 3 ppm, the proportional hazards model returns it to the
failure rate of a three-year-old (0.17 a year); the accelerated model keeps
it a 3.82-year-old (0.21 a year).

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _A = 2 + np.exp(0.3 * 2)
    assert round(np.exp(0.6), 2) == 1.82 and round(_A, 2) == 3.82
    assert round(float(Weibull.sf(_A, 6.0, 2.0)), 2) == 0.67
    assert round(float(Weibull.sf(3.0, 6.0, 2.0)), 2) == 0.78
    assert round(float(Weibull.hf(3.0, 6.0, 2.0)), 2) == 0.17
    assert round(float(Weibull.hf(_A, 6.0, 2.0)), 2) == 0.21

The fitter is built and used as ``JointPH``'s is: the same data, path
models, ``varying``, ``hazard_covariates`` and ``path_covariates``,
``marker_reference``, ``fit``, ``fit_from_df``, predictions and
saving, and the same parameter names. A fleet of 300 gearboxes whose debris
is a line in years, each ageing :math:`e^{0.6}` times as fast for each ppm
above 4:

.. jupyter-execute::

    from surpyval import WeibullJointAFT

    gen_aft = np.random.default_rng(29)


    def loaded(n, alpha=2.5, beta=2.0, effect=0.6, centre=4.0):
        spread = np.array([[0.36, 0.06], [0.06, 0.25]])  # sds 0.6, 0.5
        lines = gen_aft.multivariate_normal([3.0, 1.0], spread, size=n)
        items = []
        for k, (level, rate) in enumerate(lines):
            used = alpha * gen_aft.weibull(beta)  # the effective age it reaches
            # a line's effective age: e^(effect (level - centre)) (e^(k t) - 1) / k
            start, speed = np.exp(effect * (level - centre)), effect * rate
            grow = speed * used / start
            life = np.log1p(grow) / speed if grow > -1 else np.inf
            end = min(life, 2.0)                       # followed for two years
            ages = np.arange(0.0, end + 1e-9, 1 / 12)  # read monthly
            readings = level + rate * ages + gen_aft.normal(0, 0.7, ages.size)
            outcome = "observed" if life <= 2.0 else "right_censored"
            items.append(
                ItemLifetime(k, **{outcome: end}, markers=dict(zip(ages, readings)))
            )
        return SurpyvalData.from_items(items)


    gearboxes_aft = loaded(300)
    aged = WeibullJointAFT.fit(gearboxes_aft, marker_reference=4.0, unit="years")
    aged.summary()[["coef", "se(coef)", "exp(coef)"]].round(3).tail(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert aged.maximum == "verified" and aged.survival == "AFT"
    _p = dict(zip(aged.parameter_names, aged.params))
    _se = dict(zip(aged.parameter_names, aged.standard_errors()))
    for _name, _truth in [("marker_effect", 0.6), ("alpha", 2.5), ("beta", 2.0)]:
        assert abs(_p[_name] - _truth) < 3 * _se[_name], _name

``exp(coef)`` of ``marker_effect`` is how many times as fast a gearbox
ages for each ppm more (0.62 against the 0.6 simulated); ``alpha`` and
``beta`` are the Weibull of a gearbox held at the reference, 4 ppm. The
two models of the same fleet compare by ``aic()``, as the links do:

.. jupyter-execute::

    multiplied = WeibullJointPH.fit(gearboxes_aft, marker_reference=4.0, unit="years")
    print(f"AIC, faster ageing:        {aged.aic():.1f}")
    print(f"AIC, higher failure rate:  {multiplied.aic():.1f}")
    print(f"marker_effect: {aged.params[-1]:.2f} (ageing), "
          f"{multiplied.params[-1]:.2f} (hazard)")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(aged.params[-1], 2) == 0.62
    assert round(multiplied.params[-1], 2) == 1.13
    assert round(aged.aic() - multiplied.aic()) == 4

Here the data cannot tell the two apart: the proportional hazards model
even has the lower AIC, by 4, though the fleet was simulated from the
accelerated one. A line that only rises gives both models much the same
shape of risk, the hazard model's effect (1.13) taking the place of
roughly :math:`\beta` times the ageing one (:math:`2.05 \times 0.62 =
1.27`).
The two part company where a reading rises and falls, or where the
baseline is not a Weibull; with paths that only rise, choose the model by
what the reading is rather than by a few units of AIC.

Four things to know:

- **The links.** ``link="value"`` (the default) and ``"slope"`` (or both)
  set the ageing rate; ``"cumulative"`` is refused, because the
  effective age already accumulates the reading over the item's life.
- **When the two models agree.** With a reading that never changes and a
  Weibull baseline the accelerated model is the proportional hazards one,
  each effect multiplied by ``beta`` (an ageing rate :math:`e^{\zeta}`
  is a hazard ratio :math:`e^{\beta\zeta}`): the two fits of such a fleet
  have the same likelihood and parameters to 1e-4
  (``tests/univariate/joint/test_joint_aft.py``). Once the reading
  moves they differ.
- **The history counts.** The effective age at ``given`` depends on the
  whole path from age 0, so ages are ages since new, and the readings
  should start early in the item's life: before the first reading the
  item's path is the path model's, extended back from its readings.
- **Which model.** By what the reading is: a symptom of the present
  damage (proportional hazards), or a stress that uses up life
  (accelerated); and by ``aic()`` where the readings move enough for
  the data to tell.

The fitted model says which survival part it has (``aged.survival`` is
``"AFT"``; the printout's survival line and the summary's heading say
so), saves it with ``to_dict`` (a file that SurPyval 0.28 and older
refuse rather than misread as proportional hazards) and predicts as before:

.. jupyter-execute::

    now = aged.predict(markers={0.0: 3.4, 0.5: 3.9, 1.0: 4.5}, given=1.0)
    print("life left, 10/50/90%:", now.remaining_life([0.1, 0.5, 0.9]).round(2))
    print(aged.cb([1.5, 2.0], markers=now.markers, given=1.0).round(3))

The likelihood is a second one beside the proportional hazards one, with
the same quadrature over each item's path: the effective age by the
cumulative hazard's 15-point Gauss--Legendre rule, and an interval's
cumulative hazard from the effective age at its start and the age used up
over it, integrated directly. It agrees with an independent dense-grid
integral for every kind of item to within 1e-5 an item, for the value and
the slope links and with covariates in each part.

A recovery study of the accelerated model (run on 2026-11-18 with
``scripts/studies/joint_recovery/study.py``'s ``aft_`` paths; this page
does not re-run it) fitted 200 fleets for each of the four paths of the
study below -- the same pumps, readings and paths, each pump ageing at
the rate :math:`e^{0.45 (m - 4)}` against a Weibull of 800 days and shape
2 in effective age -- with ``JointAFT`` and ``marker_reference=4.0``.
All 800 fits reached a verified maximum. The marker effect's bias was
+2.6% (Monte Carlo se 1.3%) for the line, +2.5% (1.1%) for the
exponential, +2.3% (1.2%) with ``b`` on a log link and +1.1% (1.5%) for
the quadratic, its 95% interval covering the truth 94.5%, 94.0%, 93.0%
and 95.5% of the time. Of the 37 parameters, 36 intervals covered the
truth between 92% and 98% of the time; the exception was the quadratic's
``sd_a``, 0.905 (0.021), whose estimate was 0.8% (0.3%) low. The spread of
the estimates was 0.91 to 1.07 times their mean standard error, except the
quadratic's ``alpha`` at 1.21, from two fleets with long tails (1.01
without them). Biases of more than two Monte Carlo standard errors, all
small: the marker effect for the line and the exponential (above),
``alpha`` +0.7% (0.4%) for the line and +2.1% (0.9%) for the quadratic,
and the exponential's ``corr_a_b``, +0.011 (0.005). The per-parameter
table is ``scripts/studies/joint_recovery/table.txt``.

How well it recovers the truth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simulation study (run on 2026-10-16 with the script
``scripts/studies/joint_recovery/study.py``, whose summaries are beside it; this page does not re-run it) fitted 200
simulated fleets for each of four paths. Each fleet had 300 pumps read
monthly for up to two years, a Weibull hazard (``alpha`` 800 days, ``beta``
2 at a marker of 4) and a marker effect of 0.9, and was fitted with
``marker_reference=4.0``. The paths were:

- the line, ``a`` 3 (sd 0.6) and ``b`` 1/365 (sd 0.5/365), noise 0.7;
- the exponential ``a exp(b t)``, ``a`` 3 (sd 0.4) and ``b`` 0.0008
  (sd 0.0003), noise 0.3;
- the same with ``b`` on a log link, ``log(b)`` normal about log(0.0008)
  with sd 0.3;
- the quadratic, ``a`` 3 (sd 0.5), ``b`` 0.5/365 (sd 0.3/365) and ``c``
  1/365² common to all, noise 0.4.

All 800 fits reached a verified maximum except two of the log-link ones
(fitted, but unverified). For the marker effect:

- line: bias +0.4% (Monte Carlo se 0.8%), coverage of the 95% interval
  0.930 (0.018);
- exponential: +0.9% (0.7%), 0.980 (0.010);
- log link: +2.0% (0.7%), 0.955 (0.015);
- quadratic: +0.5% (0.7%), 0.960 (0.014).

Over all 37 parameters of the four models, 36 intervals covered the truth
between 92% and 98% of the time. The exception was the log-link model's
``sd_a`` at 0.900 (0.021). The spread of the estimates was 0.86 to 1.14
times their mean standard error.

Some biases are small but more than two Monte Carlo standard errors from
0:

- ``beta``: +1.4% (0.6%) for the line and +2.2% (0.9%) for the
  quadratic. The Weibull shape's maximum likelihood estimate is biased
  upward in samples this size; a Weibull fitted alone to 300
  lifetimes censored at 730 days (2000 samples, ``scripts/studies/joint_recovery/beta_bias.py``)
  is +0.7% (0.1%).
- the exponential's ``sd_b``: -0.9% (0.3%).
- the log-link model's ``alpha``, ``marker_effect`` and
  ``corr_a_log(b)``: +1.3% (0.4%), +2.0% (0.7%) and +0.011 (0.005).

These are small-sample biases of maximum likelihood, not the quadrature.
Refitting the same fleets with 11 points a dimension instead of 7 moved
no estimate by more than 0.0104 of its standard error. With four times the
pumps (100 fleets of 1200), the log-link model's biases fell to +0.4%
(0.25%), +0.2% (0.5%) and +0.005 (0.0035), and every interval covered the
truth between 93% and 98% of the time.

.. _joint-several-markers:

Several markers on one item
---------------------------

**In plain words.** An item is often read in more than one way: a pump's
vibration *and* its bearing temperature, a patient's bilirubin *and*
albumin. A joint model of several markers takes them all. Each marker
follows a path of its own (a line, a curve, a spline), each has its own
reading noise, and each has its own effect on the risk of failure. The
items' paths are also allowed to move together: an item whose vibration
starts high may tend to run hot too. Learning that is what lets one
marker sharpen what the model knows about another, and it stops a model
of one marker from crediting it with another's effect. With two
correlated readings, a model of vibration alone credits vibration with
part of temperature's effect, and ranks a hot but smooth pump wrongly.

**The model.** Marker :math:`k` of item :math:`i` is read at ages of its
own,

.. math::
    y_{ijk} = m_k(t_{ijk}; \theta_{ik}) + \varepsilon_{ijk}, \qquad
    \varepsilon_{ijk} \sim N(0, \sigma_k^2),

the varying parameters of every marker are one normal vector with one
covariance :math:`D` across the markers (estimated through its Cholesky
factor, so it is always a valid covariance), and the hazard follows each
marker through its own links,

.. math::
    h_i(t) = h_0(t) \exp\Big(\gamma' z_i + \sum_k \sum_l \zeta_{kl}\,
    f_l\big(m_k(\cdot; \theta_{ik})\big)(t)\Big),

:math:`f_l` the marker's current value (less its reference), slope or
area, as for one marker. Given an item's path parameters the readings of
every marker and its lifetime are independent, so its likelihood is the
one-marker integral over all of them together.

**Fitting.** ``path_model`` is a dict, one path model a marker; ``link``,
``varying`` and ``random_effects`` may be given by marker:

.. code-block:: python

    from surpyval import JointPH, Weibull
    from surpyval.degradation import LinearPath, QuadraticPath

    fitter = JointPH(
        Weibull,
        {"vibration": LinearPath, "temperature": QuadraticPath},
        varying={"temperature": ["a", "b"]},  # its curvature common
        link={"vibration": "value", "temperature": ["value", "slope"]},
    )
    model = fitter.fit(data)  # a SurpyvalData whose m holds both

- ``link``: one link or a list for every marker, or a dict by marker. A
  marker given ``[]`` (or left out of the dict) is read and modelled, and
  not in the hazard: its readings still inform the others through the
  correlations.
- ``varying``: each marker's default (every parameter; a spline's level
  and rate), a dict by marker, or a list of names ``"<marker>.<name>"``.
- ``random_effects``: ``"full"`` (the default: every pair of varying
  parameters correlated, across the markers too), ``"by_marker"`` (each
  marker's correlated with each other and with none of another
  marker's), ``"diagonal"``, or blocks of ``"<marker>.<name>"`` names.
- ``marker_reference`` and ``marker_unit`` are dicts by marker.

The parameters are named by marker: ``vibration.a``,
``sd_vibration.a``, ``corr_vibration.a_temperature.a``,
``vibration.sd_noise``, ``vibration.marker_effect``; ``alpha`` and
``beta`` are the baseline's at every followed marker's reference (each
marker's mean reading unless given). A model of one marker given as a
dict is the single-marker model, with these names.

**The data.** Markers are usually read at different ages, so the
readings are in long form, one per reading: the unit, the age, the
marker's name and the value. That is a :class:`~surpyval.Measurements` of
named measurements (``Measurements.from_dataframe(readings, i_col=...,
x_col=..., m_col=..., m_name_col=...)``, or ``{marker: {unit: {age:
value}}}``), held by the data as ``data.m`` or given as ``m=`` with each
row's unit as ``i=``; with arrays, ``markers=`` is one ``{marker: {age:
value}}`` per row. Each marker's readings are checked against the item's
observation as one marker's are. A long form needs no value for a marker
not read at a visit and no padding; a wide table (a column a marker)
would force the markers onto one timetable. Measurements the model does
not name are not used, and a single-marker fitter refuses several,
saying how to keep one or fit them together.

**How many parameters can vary.** The integral over an item's varying
parameters is over all the markers' together: two markers each with a
level and a rate is four. Up to five in all it is on the adaptive grid
(7 points a dimension: 2,401 nodes an item for four, 16,807 for five);
beyond five it is Laplace's approximation (``integration="laplace"``),
up to eight in all (``MAX_MARKER_VARYING``). More is refused, saying to
name fewer in ``varying``: an item's integral costs ``q^2 + q + 1``
evaluations by Laplace's approximation and the covariance has
``q (q + 1) / 2`` parameters (36 for eight), too many to estimate from
a usual fleet. ``random_effects="by_marker"`` or fewer varying
parameters is the remedy for a fleet that cannot tell the correlations
apart. A four-dimensional fit takes about a minute for 150 items read
about 30 times each, and twenty minutes for the PBC trial below (312
patients); ``integration="laplace"`` is faster at some cost in accuracy
where items have few readings (:ref:`Laplace's approximation
<joint-laplace>`).

**Bounds and tests.** Each association has a Wald interval and test in
``summary()``. A likelihood-ratio test of a marker's effect is the fit
without it (``link={"temperature": []}``) against the fit with it,
``model.likelihood_ratio_test(nested)``; the test of the correlations
across the markers is ``random_effects="by_marker"`` against
``"full"``. Likelihood-ratio and bootstrap intervals are not offered for
the joint models, of one marker or several.

**Prediction.** ``sf``, ``ff``, ``qf``, ``mean``, ``cb`` and ``predict``
take every marker's readings so far, ``markers={"vibration": {...},
"temperature": {...}}``; a marker with none so far may be left out. The
dynamic prediction's ``marker(x, name)`` and ``marker_interval(x,
name=...)`` give one marker's path given all of them. An
``ItemLifetime`` holds one marker's readings, so ``sf(x, item)`` is
refused for a model of several.

**Checked by.**

- A model of one marker given as a dict equals the single-marker fit
  to 2e-15 (parameters, standard errors and log-likelihood).
- Two markers, the second not in the hazard and uncorrelated with the
  first (``random_effects="by_marker"``), equal the first marker's joint
  fit (to 1e-9) plus a linear mixed model of the second: R's
  ``nlme::lme(..., method = "ML")`` gives the same fixed effects, spread
  and noise to 3e-10 and the same log-likelihood to 1e-9
  (``scripts/reference/reference_r_joint_markers.R``).
- A recovery study (``scripts/studies/joint_recovery/markers_study.py``,
  its design and seeds fixed before it ran; run 2026-11-25): 200 fleets
  of 200 items read for vibration every 30 days and temperature every
  60, the four path parameters correlated across the markers (the
  starts 0.5), both markers in the hazard (0.8 and 0.08 a unit), every
  parameter varying and every correlation estimated on the full grid.
  All 200 fits reached a verified maximum (55 s each, the median). The
  95% intervals of all 20 parameters covered the truth 92.0% to 97.0% of
  the time (Monte Carlo se 1.5%): the effects 92.5% (vibration) and
  95.5% (temperature), the cross-correlations 92.0% to 95.0%. Biases
  are within two Monte Carlo errors but vibration's effect, +3.3% (Monte
  Carlo se 1.0%), and ``alpha``, +1.0% (0.5%); temperature's effect
  -0.9% (1.5%), the noise sds and the paths' spreads within 0.3%. The
  marker effect's small-sample bias is of the size the one-marker
  studies found for the log-link path at 300 items (+2.0%), which
  shrank with more items there; not run at a larger size here.
- On the Mayo PBC trial (``pbc2``: 312 patients, 140 deaths; log
  bilirubin and albumin, each a line in years with a level and a rate of
  each patient's own, the four correlated; the hazard in both current
  values), against R's multivariate joint models
  (``scripts/studies/joint_recovery/markers_pbc.py`` and ``.R``). Neither
  R package has the Weibull baseline: ``JMbayes2`` is Bayesian (posterior
  means, a spline baseline) and ``joineRML`` fits a Cox baseline by Monte
  Carlo EM, so they agree within their standard errors, not to digits.
  Bilirubin's effect (per unit of log bilirubin) 0.950 (se 0.119),
  against 0.947 (posterior sd 0.149) and 0.909 (0.124); albumin's (per
  g/dl) -2.70 (0.35), against -2.85 (0.39) and -2.77 (0.34). The paths
  agree to the second or third figure (bilirubin's level 0.490, 0.491,
  0.484; its rate 0.195, 0.192, 0.193 a year; albumin's 3.548, 3.548,
  3.551 and -0.111, -0.111, -0.111), as do the noise sds (0.347, 0.347,
  0.347; 0.319, 0.320, 0.319), the spreads (within 0.004 of each other
  but bilirubin's rate, 0.188 against 0.185 and 0.187) and the
  correlations across the markers (the levels -0.545, against -0.526
  and -0.544; the rates -0.836, against -0.788 and -0.838). The
  SurPyval fit reached a verified maximum on the full grid in 20
  minutes.

Each marker can act on the hazard after a delay of its own, and weight
its accumulated exposure its own way: ``lag`` and ``weight`` take one
value for every marker, or a dict by marker, as ``varying`` and ``link``
do. ``lag={"vibration": 30}`` has the hazard follow the vibration 30
days late and the temperature at once (a marker left out has no lag),
and ``weight={"vibration": ("window", 90)}`` counts only the last 90
days of a vibration whose area the hazard follows (``link={"vibration":
["value", "cumulative"]}``). As with one marker, each is fixed, not
estimated: compare the choices you have in mind by ``aic()``.

``compare_links`` asks what the hazard follows of one marker:
``fitter.compare_links(data, marker="vibration")`` fits the vibration's
links in turn, each with every other marker's links as the fitter has
them, and ranks them as with one marker; ``models`` and the table are by
the vibration's link and the ``verdict`` names it. Whether a spline
marker's bends vary between items (``varying=("vibration.s1",)``) is,
as with one marker, a comparison by ``aic()`` of the fits with and
without them in ``varying``.

**Failure modes with several markers.** Each marker may warn of a
different way of failing: vibration of a bearing failure, temperature of
a seal leak. Give each failure its mode (``e=``, ``e_col=``, or the
records' ``event_type``), as with one marker (:ref:`failure modes
<joint-failure-modes>`), and each mode gets a Weibull baseline, hazard
coefficients and an effect of each marker's link of its own, named
``"bearing: vibration.marker_effect"``, ``"seal:
temperature.marker_effect"`` and so on; the markers' part is one for
every mode. For mode :math:`k`,

.. math::

    h_{ik}(t) = \frac{\beta_k}{\alpha_k}\left(\frac{t}{\alpha_k}
    \right)^{\beta_k - 1} \exp\Big(\gamma_k' z_i + \sum_{(j, l)}
    \zeta_{kjl}\, g_l\big(m_j(\cdot; \theta_i)\big)(t)\Big),

one effect :math:`\zeta_{kjl}` for each mode, marker :math:`j` and link
:math:`l` (each followed marker's value measured from its reference).
``shared=`` holds what the modes have in common: ``"beta"``, a hazard
covariate, or one marker's link effect by its full name
(``shared=["temperature.marker_effect"]``: the temperature warns of
both modes alike, the vibration of each its own way); the fit with an
effect shared is nested in the fit without, so
``model.likelihood_ratio_test(shared_model)`` tests it. ``alpha`` is
never shared (it depends on where the covariates are centred). Lags,
weights, ``compare_links(..., marker=)``, predictions, bounds and saving
work as without modes; the survival is that of every mode together.
``JointAFT`` and the two-stage estimate do not take modes yet, as with
one marker.

The modes are not fitted one at a time. Each item's path is uncertain,
and it is integrated over once with every mode's hazard in the
integrand, so a fit of one mode with the others' failures treated as
still running is not this fit's answer. It is usually close, because
the readings, not the failures, mostly settle an item's path: on a
seeded fleet of 150 items the censored fits' survival parameters were
within 0.014 standard errors of the joint fit's. Where a mode follows
no marker the likelihood does factor exactly: that mode is a plain
Weibull.

Checked by:

- one mode is the model without modes, bit for bit;
- one marker given as a dict, with modes, equals ``WeibullJointPH``
  with modes: parameters to 3e-15, standard errors to 6e-15, the same
  log-likelihood;
- two modes with the same hazard are the model without modes at twice
  the hazard, less :math:`\log 2` a failure, and a mode with every
  effect 0 factors out as a plain Weibull (both to 1e-10 in the
  log-likelihood);
- a seeded fleet of 300 (``tests/univariate/joint/_multi_fleet.py``:
  bearing failures that follow vibration with 0.8 a unit, seal failures
  that follow temperature with 0.08) recovers every parameter but the
  correlations within 3.5 standard errors, each marker's effect on its
  own mode clear of 0 (p below 0.001);
- a recovery study of 200 fleets of 250 items
  (``calibration/test_coverage_joint_marker_causes.py``, its design and
  seeds fixed beforehand) is committed and not yet run in full. A
  reduced run of 12 fleets of 200 (seeds 46000 to 46011; about 76
  bearing and 53 seal failures a fleet; two minutes a fit) reached a
  verified maximum every time. The four effects averaged 0.865 (Monte
  Carlo se 0.032; truth 0.8) and 0.002 (0.006; 0) for the bearing, 0.058
  (0.042; 0) and 0.069 (0.008; 0.08) for the seal, each within two Monte
  Carlo errors of the truth. The largest standardised departure was the
  vibration's rate, 2.2% low (2.3 Monte Carlo errors), reported as
  found; twelve fleets cannot say more, which is what the full study is
  for.

Not yet: a shared health index (one hidden condition behind every
reading).

Prediction: survival given the readings so far
----------------------------------------------

The model predicts for one item: ``model.sf(x, Z, markers=..., given=...)``
is its survival to each age ``x`` given its covariates ``Z``, its readings
so far ``markers`` (``{age: value}``) and that it was still running at age
``given`` (1 at or before it), over the paths its readings leave possible;
``ff`` is its failure probability, and ``cb`` bounds either from the
parameters' uncertainty (the delta method on :math:`\log(-\log S)`, with
the parameters' covariance). With no readings and no ``given`` it is a new
item's survival, averaged over the population's paths. An item record
gives all three at once: ``model.sf(x, item)`` reads its covariates, its
``markers`` and its ``right_censored`` age as ``given``, so the two forms
are the same prediction. Two pumps a year old, one whose vibration has
stayed low and one whose has risen:

.. jupyter-execute::

    quiet = ItemLifetime(
        "Q", right_censored=365, markers={0: 3.0, 120: 3.1, 240: 3.0, 360: 3.2}
    )
    rising = ItemLifetime(
        "R", right_censored=365, markers={0: 3.0, 120: 3.6, 240: 4.3, 360: 4.9}
    )
    for item in (quiet, rising):
        print(item.id, model.ff([425, 545], item).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert model.ff(545, rising) > 2 * model.ff(545, quiet)
    assert model.ff(545, rising) == model.ff(
        545, markers=rising.markers, given=365
    )

The same without the records, and for a new pump:

.. jupyter-execute::

    print(model.ff([425, 545], markers=rising.markers, given=365).round(3))
    print(model.ff([365, 730]).round(3))  # a new pump, no readings yet

The rising pump is more than twice as likely to fail in the next six months.
Bounds, and a new reading as it arrives (the prediction is updated by
asking again with the updated record):

.. jupyter-execute::

    print(model.cb([425, 545], rising, on="ff").round(3))
    rising_later = rising.extend(395).update_markers(395, 5.6)
    print(model.ff(545, rising_later).round(3))

.. jupyter-execute::

    import matplotlib.pyplot as plt

    x = np.linspace(365, 900, 100)
    fig, ax = plt.subplots()
    for item in (quiet, rising):
        band = model.cb(x, item)
        ax.plot(x, model.sf(x, item), label=f"pump {item.id}")
        ax.fill_between(x, band[:, 0], band[:, 1], alpha=0.2)
    ax.set_xlabel("age (days)")
    ax.set_ylabel("survival, given the readings to day 365")
    ax.legend();

``sf`` keeps the package's shapes (a scalar ``x`` gives a scalar, an array an
array of its shape, and a two-sided ``cb`` adds a last ``[lower, upper]``
axis). Far beyond the data the survival can level off above 0: the lines
are straight, and one whose slope is negative takes the hazard towards 0, as
the model says (a reason to read predictions far beyond the data with care;
past about ten times the data's ages the levelled-off value moves by the
quadrature's error, about 1e-5 of itself).

One item's dynamic prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``model.predict(item)`` (or ``predict(Z, markers=..., given=...)``) holds one
item's prediction as an object, for the questions asked of one asset: its
survival, failure probability and hazard from now on given its readings
(``sf``, ``ff``, ``hf``, ``df``, ``Hf``), its remaining life
(``remaining_life(p)``, the life left that it outlives with probability
``1 - p``, and ``qf`` the age), a Wald band from the parameters'
uncertainty (``cb``, the same as ``model.cb``), its marker smoothed through
the readings and projected beyond them (``marker`` and ``marker_interval``,
the noise-free path's mean and interval given the readings and that it
reached its age now), and draws of its failure age (``random``). Its
``sf``, ``ff`` and ``cb`` are the model's for the same item; the hazard is
the item's at that age given its readings and that it reached it.

.. jupyter-execute::

    now = model.predict(rising)
    print(now)
    print("life left, 10/50/90%:", now.remaining_life([0.1, 0.5, 0.9]).round(0))
    print("hazard a day at 425 and 545:", now.hf([425, 545]).round(5))
    print("vibration at 365 and 545:", now.marker([365, 545]).round(2))
    print(now.marker_interval(545).round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert now.sf(545) == model.sf(545, rising)
    assert np.allclose(now.ff(now.qf([0.1, 0.5, 0.9])), [0.1, 0.5, 0.9])
    assert now.remaining_life(0.5) < model.predict(quiet).remaining_life(0.5)

The 90% range of its life left is wide -- a prediction of one pump's life,
not a confidence interval -- and the band of :meth:`cb` adds the
uncertainty of the fitted parameters to the survival. A calibration study
(``tests/calibration/test_coverage_joint.py``, 200 simulated fleets of 200
pumps with the covariates of `Covariates in each part`_, with exact failure
times and with 90-day inspections) found the 95% band of a held-out pump's
conditional survival 60, 180 and 365 days ahead covering the true
conditional survival 93.0%, 89.5% and 93.0% of the time (exact times) and
93.5%, 94.5% and 94.0% (inspections); each parameter's 95% interval
covered its truth 93.0% to 96.5% of the time (exact) and 91.5% to 97.5%
(inspections), the standard errors 0.91 to 1.09 times the estimates'
spread. The Wald band can run a little narrow for this sample size, most
at 180 days ahead with exact times.

How good are the predictions
----------------------------

A joint model gives each running pump a chance of failing in the next six
months. Before acting on those chances, check how good they were on pumps
whose outcome is known. Three questions, at an age ``s`` (standing at one
year old, say) and over a window ``w`` (the next 180 days), among the
pumps still running at ``s``:

- **Do the pumps that failed get the higher chances?** The dynamic AUC
  (:func:`~surpyval.metrics.dynamic.dynamic_auc`): take one pump that
  failed in the window and one that did not; the AUC is how often the
  model gave the one that failed the higher chance. 0.5 is a coin toss, 1
  is perfect.
- **How far are the chances from what happened?** The dynamic Brier score
  (:func:`~surpyval.metrics.dynamic.dynamic_brier`): the mean squared
  difference between each pump's chance and 1 if it failed in the
  window, 0 if not. Lower is better.
- **Does "30%" mean about 30%?** Calibration
  (:func:`~surpyval.metrics.dynamic.dynamic_calibration`): group the pumps
  by their chance, and set each group's mean chance beside the share that
  failed.

Each pump's chance is made from what was known at ``s``: its readings up to
age ``s``, never those after. And each question is worth asking against a
simpler model, to see whether the readings earn their cost: here a Weibull
fitted to the ages alone, which gives every pump of the same age the same
chance. A dict of models scores each on the same pumps and compares them,
with the difference's interval and p-value:

.. jupyter-execute::

    from surpyval.metrics import dynamic_auc, dynamic_brier, dynamic_calibration

    models = {"age only": Weibull.fit(fleet.without("markers")), "joint": model}
    auc = dynamic_auc(models, fleet, at=[180, 365], horizon=180)
    auc

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert (auc.estimate("age only") == 0.5).all()
    assert (auc.estimate("joint") > 0.6).all()
    assert auc.estimate("joint")[1] > auc.estimate("joint")[0]

The age alone ranks the pumps no better than a coin toss: at one age it
gives them all the same chance. The readings rank them well, and better at
a year old than at six months, when each pump's readings say more about its
line. The Brier score tells the same story in the error of the chances:

.. jupyter-execute::

    brier = dynamic_brier(models, fleet, at=[180, 365], horizon=180)
    brier.comparisons

.. jupyter-execute::
    :hide-code:
    :hide-output:

    p = brier.comparisons["p_value"].to_numpy()
    assert p[0] > 0.05 and p[1] < 0.01
    assert (brier.estimate("joint") < brier.estimate("age only")).all()

At six months the readings do not yet cut the error by more than chance; at
a year they clearly do. Calibration, in groups of predicted chance:

.. jupyter-execute::

    cal = dynamic_calibration(
        model, fleet, at=365, horizon=180, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.6, 1]
    )
    columns = ["risk_from", "risk_to", "n", "predicted", "observed", "lower", "upper"]
    cal.table[columns].round(3)

A pump censored inside the window (still running when last seen) is not
known to have failed or not. It is accounted for, as in Rizopoulos (2011)
and Blanche et al. (2015): the AUC and the Brier score weight the pumps
whose outcome is known by the inverse of the chance of being seen to the end
of the window (the reverse Kaplan-Meier of the pumps at risk at ``s``), and
calibration's share that failed is each group's Kaplan-Meier estimate. With
no censoring inside the window they are the plain AUC, mean squared error
and share. A calibration study (``tests/calibration/test_metric_bias.py``:
200 simulated data sets of 300 units, 30% of them censored, scored with the
true model's predictions) found the mean estimates within Monte Carlo
error of the true values at two ages.

The pumps scored above are the ones the model was fitted to, which flatters
it. ``cross_validate=k`` gives honest numbers: the pumps are split at random
into ``k`` groups, the model is refitted ``k`` times without one group, and
each pump is predicted by the fit that never saw it. Give the fitters in
place of the fitted models (``LandmarkCox`` and functions of the data work
too, for a like-for-like comparison with landmarking):

.. jupyter-execute::

    honest = dynamic_brier(
        {"age only": Weibull, "joint": WeibullJointPH},
        fleet, at=[180, 365], horizon=180, cross_validate=5, random_state=1,
    )
    honest.scores[["model", "at", "estimate", "se"]]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(honest.estimate("joint"), brier.estimate("joint"), atol=0.01)

With 300 pumps the model barely flatters itself here; with fewer units or
a richer model the difference grows. The intervals are two-sided 95% by
default (``alpha_ci=0.1`` for 90%), and the result says so.

A quick first look: the two-stage fit
-------------------------------------

There is a quicker way to put the readings and the lifetimes together, in
two steps. First, fit each pump's line to its readings alone, pulling a
pump with few readings towards the fleet's average line (the less its own
readings say, the further it is pulled; statisticians call the result a
best linear unbiased predictor). Then fit the lifetimes by Weibull
proportional hazards with each pump's line as a time-varying covariate, as
if the lines were known. It takes a fraction of a second, against seconds
or minutes for the joint fit, and it is the method most published analyses
used before joint models. It is ``fit(..., how="two-stage")``, and it is
exactly where the joint fit starts its search:

.. jupyter-execute::

    quick = WeibullJointPH.fit(fleet, unit="days", how="two-stage")
    quick

In formulas, step one fits :math:`y_{ij} = m(t_{ij}; \theta_i) +
\varepsilon_{ij}`, :math:`\theta_i \sim N(\mu + \Delta z_i, D)`, to the
readings alone (the path fitted to each item, then a moment estimate of
the means and of :math:`D`), and takes each item's most likely path given
its readings, :math:`\hat\theta_i`. Step two maximises the likelihood of
the lifetimes with the hazard :math:`h_0(t) \exp(\gamma' z_i + \zeta\,
m(t; \hat\theta_i))` (or the slope or the area: the ``link``), the path
taken as a known covariate. ``JointAFT`` fitters take it too, with the
accelerated survival part in step two.

The result is a joint model like any other -- the same parameters,
predictions (``sf``, ``cb``, ``predict``), accuracy measures and saving
-- and it says what it is (``quick.how`` is ``"two-stage"``). It is
approximate in three ways, which the printout and ``summary()`` state:

- **Its estimates are biased.** Step two forgets that each line is an
  estimate, and that a pump's readings stop when it fails, so the pumps
  that failed early have the least certain lines.
- **Its standard errors are too small.** Those of the survival part take
  each line as known, so they and the bands of its predictions (``cb``)
  are too narrow; the path part (step one's moment estimates) has none
  (``nan``).
- **It has no likelihood.** The two steps fit different things, so there
  is no likelihood of the whole model: ``neg_ll()``, ``aic()``,
  ``aic_c()``, ``bic()`` and ``compare_links`` refuse it.

.. jupyter-execute::
    :raises: ValueError

    quick.aic()

With a reading every month, as here, the two fits are close:

.. jupyter-execute::

    def effect(m):
        return f"{m.params[-1]:.2f} (se {m.standard_errors()[-1]:.2f})"


    print("two-stage:  ", effect(quick))
    print("joint model:", effect(model))

With a reading every six months they part:

.. jupyter-execute::

    every_six_months = SurpyvalData.from_items(
        [
            item.clone(
                markers={a: v for a, v in item.markers.items() if a % 180 == 0}
            )
            for item in fleet.to_items()
        ]
    )
    sparse_quick = WeibullJointPH.fit(every_six_months, how="two-stage")
    sparse_joint = WeibullJointPH.fit(every_six_months)
    print("two-stage:  ", effect(sparse_quick))
    print("joint model:", effect(sparse_joint))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert quick.how == "two-stage" and quick.maximum == "not applicable"
    assert round(quick.params[-1], 2) == 0.99
    assert round(sparse_quick.params[-1], 2) == 0.86
    assert round(sparse_joint.params[-1], 2) == 1.04
    assert sparse_quick.standard_errors()[-1] < sparse_joint.standard_errors()[-1]
    assert np.isnan(quick.standard_errors()[0])

The two-stage effect is the lower, and claims to be the more certain (a
standard error of 0.10 against 0.15).

Its predictions are compared with the joint model's as any two models'
are, by their accuracy (see `How good are the predictions`_); give the
two-stage fit as a function of the data to cross-validate it:

.. jupyter-execute::

    two_ways = dynamic_brier(
        {
            "two-stage": lambda data: WeibullJointPH.fit(data, how="two-stage"),
            "joint": WeibullJointPH,
        },
        fleet, at=[180, 365], horizon=180, cross_validate=5, random_state=1,
    )
    two_ways.comparisons

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert (two_ways.estimate("joint") <= two_ways.estimate("two-stage") + 0.002).all()
    p = two_ways.comparisons["p_value"].to_numpy()
    assert p[0] > 0.5 and p[1] < 0.05

Predicting each pump from a fit that never saw it, the two are alike at
six months; at a year the joint model's chances are a little better (a
Brier score lower by 0.006, p = 0.03). Here the quick fit would have done
for a first look; the study below says where it does not.

How much the two-stage fit loses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simulation study (run on 2026-11-20 with
``scripts/studies/joint_recovery/study.py`` and ``HOW=two-stage``; this
page does not re-run it) fitted the two-stage estimate to the same 200
fleets of each path as the joint model's studies (`How well it recovers the
truth`_, and the accelerated model's), and to 200 fleets of the line read
every 180 days instead of every 30, fitted both ways. For the marker
effect, the bias (Monte Carlo standard error) and how often the 95%
interval covered the truth:

.. list-table::
   :header-rows: 1

   * - fleets
     - joint: bias
     - joint: coverage
     - two-stage: bias
     - two-stage: coverage
   * - line
     - +0.4% (0.8%)
     - 0.930
     - +0.3% (0.8%)
     - 0.900
   * - exponential
     - +0.9% (0.7%)
     - 0.980
     - +0.7% (0.7%)
     - 0.965
   * - exponential, log link
     - +2.0% (0.7%)
     - 0.955
     - +4.8% (0.7%)
     - 0.935
   * - quadratic
     - +0.5% (0.7%)
     - 0.960
     - -4.8% (0.9%)
     - 0.830
   * - line, accelerated
     - +2.6% (1.3%)
     - 0.945
     - -0.1% (1.3%)
     - 0.920
   * - exponential, accelerated
     - +2.5% (1.1%)
     - 0.940
     - +1.5% (1.1%)
     - 0.950
   * - log link, accelerated
     - +2.3% (1.2%)
     - 0.930
     - -1.9% (1.2%)
     - 0.930
   * - quadratic, accelerated
     - +1.1% (1.5%)
     - 0.955
     - -1.0% (2.4%)
     - 0.800
   * - line, read every 180 days
     - +2.4% (1.1%)
     - 0.920
     - -8.0% (0.9%)
     - 0.805

With a reading every month and a path that does not curve much, the
two-stage effect is about as good as the joint model's; what it loses is
in its intervals, which cover the truth less often than they say (90% of
the time for the line, the estimates' spread 1.11 times their standard
error). The loss grows where the path curves (the quadratic: intervals
covering 80% to 83% of the time, the spread 1.4 to 1.7 times the standard
error) and where readings are few (every 180 days: 8% low, covering 80.5%
of the time, where the joint fit is within 2.4% and covers 92%). Step one's
spread between items is poor: the line's ``sd_b`` came out 14% low, the
log link's ``sd_log(b)`` 68% low and the quadratic's ``sd_b`` five times
the truth (+403%), where the joint fits' were within 1%; so a two-stage
model's predictions for items with few readings, which lean on that
spread, deserve the least trust. Each two-stage fit took about a tenth of
a second. The per-parameter tables are in
``scripts/studies/joint_recovery/table.txt``.

Use it for a first look at a new dataset, and to see how far the joint
model moves the answer; report the joint fit.

Saving and loading
------------------

The model saves to strict JSON and loads back, predicting and bounding as
before:

.. jupyter-execute::

    import surpyval

    restored = surpyval.from_json(model.to_json())
    restored.sf(545, rising) == model.sf(545, rising)

What it does and does not do
----------------------------

The joint model (#891) is the standard one. It has:

- one marker, following any degradation path model (a straight line by
  default; a spline, ``SplinePath``, where the reading bends as no
  formula does), with any number of its parameters varying between items
  (on a grid up to five, by Laplace's approximation beyond)
  (multivariate normal on their natural scale, or on a log scale with a
  ``LinkedPathModel``; every pair correlated, none, or blocks of them,
  ``random_effects=``) and the rest common;
- or several markers (``path_model`` a dict), each with its own path,
  noise and effects in the hazard, their varying parameters correlated
  across the markers (up to eight in all; see :ref:`several markers
  <joint-several-markers>`);
- a Weibull proportional hazards survival part (``JointPH(Weibull,
  path_model)``; ``WeibullJointPH`` is the line; other baselines are
  refused) in the path's current value, its slope, the area under it, or
  more than one of them (``link=``);
- or an accelerated (AFT) survival part on the Weibull (``JointAFT(Weibull,
  path_model)``; ``WeibullJointAFT`` is the line), the path's value, its
  slope or both setting how fast the item ages;
- covariates in both parts (every path parameter's mean, and the hazard)
  by default, or each in the parts named by ``hazard_covariates`` and
  ``path_covariates``;
- failures, right, left and interval censoring, with left and right
  truncation;
- competing failure modes with the proportional hazards part (the
  failures' event types): a baseline, coefficients and effects for each
  mode (with several markers an effect of each marker's link), any of
  them shared (``shared=``), see :ref:`failure modes
  <joint-failure-modes>`;
- predictions for one item given its readings, with Wald (delta method)
  bounds, as arrays (``sf``, ``ff``, ``cb``) or as one item's dynamic
  prediction (``predict``: the hazard, remaining life, the smoothed and
  projected marker, and draws);
- the accuracy of those predictions at an age over a window (dynamic AUC,
  Brier score and calibration in ``surpyval.metrics``, with
  cross-validation by unit);
- the quick, approximate two-stage fit (``how="two-stage"``) for a first
  look and for comparison.

Not yet (see #891's roadmap): other baselines (a
log-normal or log-logistic one for the accelerated part, whose risk can
rise and then fall, and spline baselines); a shared health index of
several markers (see :ref:`several markers <joint-several-markers>`). The
accelerated part has not yet been compared with R's ``JM``
(``method = "weibull-AFT-aGH"``, the same model) on reference data, nor
had its prediction band put through a calibration study. The two-stage fit
has not yet been compared with R's two-step equivalent (``nlme::lme``,
then ``survival`` with the fitted paths) on reference data, and its path
part has no standard errors.

References
----------

- Rizopoulos, D. (2012). *Joint Models for Longitudinal and Time-to-Event
  Data, with Applications in R*. Chapman & Hall/CRC.
- Rizopoulos, D. (2011). Dynamic predictions and prospective accuracy in
  joint models for longitudinal and time-to-event data. *Biometrics*, 67,
  819--829.
- Blanche, P., Proust-Lima, C., Loubère, L., Berr, C., Dartigues, J.-F. and
  Jacqmin-Gadda, H. (2015). Quantifying and comparing dynamic predictive
  accuracy of joint models for longitudinal marker and time-to-event in
  presence of censoring and competing risks. *Biometrics*, 71, 102--113.
- Wulfsohn, M. S. and Tsiatis, A. A. (1997). A joint model for survival and
  longitudinal data measured with error. *Biometrics*, 53, 330--339.
- Tsiatis, A. A. and Davidian, M. (2004). Joint modeling of longitudinal and
  time-to-event data: an overview. *Statistica Sinica*, 14, 809--834.
- Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. *Journal of the American Statistical Association*, 82,
  605--610.
- Stram, D. O. and Lee, J. W. (1994). Variance components testing in the
  longitudinal mixed effects model. *Biometrics*, 50, 1171--1177.

API reference
-------------

``WeibullJointPH`` is ``JointPH(Weibull, LinearPath)`` and
``WeibullJointAFT`` is ``JointAFT(Weibull, LinearPath)``, instances of the
fitter classes below; import everything from ``surpyval`` (or
``surpyval.univariate.joint``).

.. autofunction:: surpyval.univariate.joint.joint_ph.JointPH

.. autoclass:: surpyval.univariate.joint.joint_ph.JointPHFitter
   :members: fit, fit_from_df, compare_links

.. autofunction:: surpyval.univariate.joint.joint_aft.JointAFT

.. autoclass:: surpyval.univariate.joint.joint_aft.JointAFTFitter

.. autoclass:: surpyval.univariate.joint.joint_model.JointModel
   :members: sf, ff, cb, predict, summary, covariance, standard_errors, to_dict, from_dict

.. autoclass:: surpyval.univariate.joint.compare.LinkComparison
   :members: table, likelihood_ratio_tests, best_link, best_label

.. autoclass:: surpyval.univariate.joint.dynamic.DynamicPrediction
   :members: sf, ff, hf, df, Hf, qf, remaining_life, cb, marker, marker_interval, random
