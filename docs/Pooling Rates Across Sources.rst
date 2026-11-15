Pooling Rates Across Sources
============================

Often the evidence about how often something happens comes from several
places, each with its own count: infections at several hospitals with the
patient-days at each, failures of one kind of pump at several plants with
their running hours, adverse events in several trials with the doses
given, failures to start out of the starts attempted at several sites.
``surv.pooled_rate`` combines them.

In plain words
--------------

**The sources really differ.** A hospital in one city and a hospital in
another do not share one true infection rate, and neither do two plants
running the same pump in different service. Adding every count and every
exposure into one pot assumes they do. When they don't, the one-pot answer
is dominated by the biggest source and reports a precision nobody has.

**So ask a different question.** Not "what is the one rate?", but "across
sources like these, how are the rates spread?" The sources are treated as
a sample from a population of sources, each with its own rate. The method
estimates that population from the counts, and every source still counts
for as much as its exposure says: a source with ten events in a million
hours speaks louder than one with one event in a hundred thousand, and a
source with no events in a long exposure is real evidence of a low rate.

**It gives two different ranges, and says which is which.**

- **The average across sources**: how well the middle of the spread is
  known. This narrows as sources are added.
- **A new source like these**: where the rate of one more source, about
  which nothing else is known, probably lies. This is the population's own
  spread (plus the uncertainty in it), so it stays wide however many
  sources there are if they really differ. It is the one to use as a
  starting belief for a new site, hospital or study.

**Each source gets two estimates.** Its own (its events over its exposure)
and a *shrunk* one, pulled towards the others: a source with little
exposure is pulled a long way, one with a lot hardly at all. The shrunk
estimates are usually better guesses of each source's true rate than the
raw ones, especially for small sources.

**Do the sources differ at all?** A heterogeneity test (Pearson's
chi-square that every source has one rate) and :math:`I^2`, the share of
the variation that is more than chance, say how much the sources disagree.

**At least four sources.** With two or three sources, how different a
fourth would be cannot be estimated: the range for a new source comes out
either absurdly wide or wrongly narrow. Below ``min_sources`` (default 4)
the function gives the sources side by side, each with its own interval,
the one-pot estimate and the heterogeneity test, and no population, with a
warning saying so. With one source it gives that source's own (Jeffreys)
interval.

Rates and probabilities
-----------------------

``kind="rate"`` (the default) is for events in a time at risk: patient-days,
operating hours, person-years, kilometres. ``kind="probability"`` is for
events out of a number of trials or demands: failures to start out of
starts, adverse events out of doses, defects out of items inspected.
The two are never converted into each other.

An example
----------

Six sources of failures to start, out of the starts (demands) attempted.
The first is a large modern data set; the others are older and smaller.
These inputs are illustrative.

.. jupyter-execute::

    import surpyval as surv

    failures = [227, 137, 25, 3, 42, 6]
    demands = [410593, 49000, 2100, 260, 3700, 2400]
    pooled = surv.pooled_rate(
        failures, demands, kind="probability", alpha_ci=0.1,
        labels=["A", "B", "C", "D", "E", "F"],
    )
    print(pooled)

Read the three answers in turn:

- **One pot**: about 0.00094 per demand, with a range of 1.2 times from
  end to end. It is dominated by source A's huge exposure, and no other
  source's own range comes near it.
- **The average across sources**: about 0.0067, known to within about 5
  times.
- **A new source like these**: median about 0.0046, but anywhere from
  about 0.00017 to 0.026, a range of about 150 times. That is the method
  saying, correctly, that these sources disagree by a lot. A range this
  wide is a cue to split the sources into groups that are more alike (here,
  the modern source and the older ones) and pool each group.

The sources side by side, alone and shrunk:

.. jupyter-execute::

    print(pooled.sources.round(5).to_string())

Source D, with only 260 demands, is pulled furthest towards the others;
source A, with 410,593, does not move.

The population (between-source) distribution, as a SurPyval ``Beta`` here
(a ``Gamma`` for rates), and its parameters:

.. jupyter-execute::

    print(pooled.population_params)
    print(pooled.population.qf([0.05, 0.5, 0.95]))

The heterogeneity test:

.. jupyter-execute::

    print(pooled.heterogeneity)

Updating with a new source's own data
-------------------------------------

The pooled population is a starting belief for a new source; its own
counts then update it (the "second stage"). ``update`` gives the new
source's estimate from its own events and exposure:

.. jupyter-execute::

    for own_failures in (0, 1, 4):
        print(own_failures, "in 400:", pooled.update(own_failures, 400))

With a wide population the new source's own data quickly take over; with a
narrow one they barely move it. Compare the update with the source's own
data alone before relying on it: a narrow population from the wrong kind
of source can hold the answer far from what the new source's data say.

A rate example
--------------

Infections and patient-days at eight hospitals (illustrative), one with no
infections:

.. jupyter-execute::

    infections = [12, 4, 0, 9, 21, 3, 7, 15]
    patient_days = [8000, 5200, 1900, 4100, 9800, 3500, 6100, 7300]
    hosp = surv.pooled_rate(infections, patient_days, alpha_ci=0.1)
    print(hosp)
    print(hosp.sources[["own", "shrunk", "shrunk_lower", "shrunk_upper"]].round(5))

The hospital with no infections gets a shrunk rate above zero, with a range,
because 1,900 patient-days is not enough to show that its rate is zero.

Bayes or maximum likelihood
---------------------------

``method="Bayes"`` (the default) puts a weak prior on the population's two
parameters and averages every interval over their posterior, computed on a
grid (exact up to the grid, and deterministic). ``method="MLE"`` fits them
by marginal maximum likelihood (the "empirical Bayes" estimate), with a
profile-likelihood interval for the average and a parametric bootstrap for
the others.

With few sources the difference matters. The MLE treats the population's
spread as known once estimated, and with four to ten sources that spread
is itself very uncertain. In a simulation with six sources, the nominal
90% range for a new source covered the new source's rate 88% of the time
under ``"Bayes"`` and 72% under ``"MLE"``; the average's interval covered
the true mean 91% and 85% of the time. Use ``"MLE"`` for many sources, or
to compare with an empirical-Bayes analysis.

The maths
---------

**Rates.** Source :math:`i` has :math:`x_i` events in exposure :math:`T_i`
with its own rate :math:`\lambda_i`:

.. math::

    x_i \sim \text{Poisson}(\lambda_i T_i), \qquad
    \lambda_i \sim \text{Gamma}(a, b),

shape :math:`a`, rate :math:`b`, mean :math:`a / b`. With
:math:`\lambda_i` integrated out each :math:`x_i` is negative binomial,

.. math::

    P(x_i \mid a, b) = \frac{\Gamma(x_i + a)}{\Gamma(a)\, x_i!}
    \left(\frac{b}{b + T_i}\right)^{a}
    \left(\frac{T_i}{b + T_i}\right)^{x_i},

and the likelihood of :math:`(a, b)` is the product over the sources.

**Probabilities.** :math:`x_i \sim \text{Binomial}(n_i, p_i)` and
:math:`p_i \sim \text{Beta}(a, b)`, so :math:`x_i` is beta-binomial,
:math:`P(x_i \mid a, b) = \binom{n_i}{x_i} B(x_i + a, n_i - x_i + b) /
B(a, b)`.

**Estimating the population.** Under ``"Bayes"`` the prior is flat on the
log of the mean rate :math:`a / b` (the logit of the mean probability) and
:math:`p(a) \propto (1 + a)^{-2}` on the shape, a proper, weak prior that
does not depend on the unit of time. The posterior of the mean and the
shape is computed on a grid of 161 by 161 points, refined once around
where it has weight. Under ``"MLE"`` the same likelihood is maximised; the
likelihood can be largest with no spread at all (:math:`a \to \infty`,
every source one rate), and the MLE then says so (an infinite shape).

**The results.**

- The population mean's interval: the quantiles of its posterior
  (``"Bayes"``); the profile-likelihood interval (``"MLE"``).
- A new source: its rate is a draw from the population,
  :math:`\text{Gamma}(a, b)`, averaged over the posterior of
  :math:`(a, b)` (``"Bayes"``) or over the bootstrap refits (``"MLE"``).
  This is the predictive distribution; the population "averaged over the
  uncertainty" is not exactly a gamma, so :attr:`population` is the gamma
  (or beta) with the same median and upper quantile, which keeps the
  upper side honest.
- A source's shrunk estimate: given the population, its rate has the
  conjugate posterior :math:`\text{Gamma}(a + x_i, b + T_i)`
  (:math:`\text{Beta}(a + x_i, b + n_i - x_i)`), again averaged over
  :math:`(a, b)`.
- A source's own interval: Jeffreys, :math:`\text{Gamma}(x_i + 1/2, T_i)`
  or :math:`\text{Beta}(x_i + 1/2, n_i - x_i + 1/2)`, which is defined with
  no events.
- Heterogeneity: Pearson's
  :math:`Q = \sum_i (x_i - T_i \hat r)^2 / (T_i \hat r)` with
  :math:`\hat r = \sum x_i / \sum T_i`, on :math:`k - 1` degrees of
  freedom (the 2 by :math:`k` table's chi-square for probabilities), and
  :math:`I^2 = \max(0, (Q - (k - 1)) / Q)`.

This is the "two-stage" or hierarchical Bayes treatment of generic data in
probabilistic risk assessment (NUREG/CR-6823, chapter 8), and the
empirical-Bayes analysis of event rates of Gaver and O'Muircheartaigh
(1987). Averaging over the posterior of :math:`(a, b)` does directly what
the Kass-Steffey correction to the empirical-Bayes intervals approximates.

**Checks.** The tests reproduce the BUGS "Pumps" example (ten pumps'
failures and operating times, with that example's own prior), whose
published posterior means of :math:`a` and :math:`b` are about 0.70 and
0.93, and its per-pump posterior means to 0.2% of an independent numerical
integration; the MLE agrees with SciPy's negative binomial and
beta-binomial likelihoods maximised directly; and a simulation study checks
the coverage of the intervals.

References
----------

Atwood, C. L. et al. (2003), *Handbook of Parameter Estimation for
Probabilistic Risk Assessment*, NUREG/CR-6823, US Nuclear Regulatory
Commission.

Gaver, D. P. and O'Muircheartaigh, I. G. (1987), "Robust empirical Bayes
analyses of event rates", *Technometrics* 29, 1-15.

Gelfand, A. E. and Smith, A. F. M. (1990), "Sampling-based approaches to
calculating marginal densities", *Journal of the American Statistical
Association* 85, 398-409.

Higgins, J. P. T. and Thompson, S. G. (2002), "Quantifying heterogeneity
in a meta-analysis", *Statistics in Medicine* 21, 1539-1558.

API: :doc:`univariate/pooled_rate`.
