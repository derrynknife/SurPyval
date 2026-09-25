.. surpyval documentation master file, created by
   sphinx-quickstart on Thu Mar 19 20:15:57 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



SurPyval - Survival Analysis in Python
======================================

*SurPyval* is a Python package for survival analysis: the statistics of how
long things last. The "thing" can be a bearing, a patient, a loan, a
recession or a marathon runner, and the "time" can be hours, cycles,
kilometres or dollars. What makes these problems different from ordinary
statistics is that the data is almost never complete. Some units have not
failed yet when the study ends (*censoring*), some were only inspected now
and then, and some never reached your data set at all because of how it was
collected (*truncation*). SurPyval is built around handling every one of
those situations correctly, in any combination, with one consistent
interface.

The package grew out of reliability engineering, so it has the tools an
engineer expects (probability plots, offsets, B-lives, accelerated life
tests, burn-in and warranty calculations, repairable systems, degradation),
alongside the tools of biostatistics and econometrics (Kaplan-Meier, Cox
regression, competing risks, frailty, predictive metrics).

Installation
------------

*SurPyval* needs Python 3.11 or later and installs with pip:

.. code-block:: bash

    $ pip install surpyval

To work on SurPyval itself, see :doc:`Contributing`.

Where to start
--------------

- New to SurPyval? Read the :doc:`Quickstart`. In ten minutes it fits a
  first model, shows why censoring matters, and gives a small example of
  every area of the package.
- New to survival data, or not sure whether your data is censored or
  truncated? Read :doc:`Types of Data`. Getting this right matters more than
  any choice of model.
- Have data in a spreadsheet, a list of failures and survivors, or a
  maintenance log? :doc:`Data Wrangler Examples` shows how to turn it into
  the ``x``, ``c``, ``n``, ``t`` arrays every fitter takes, and
  :doc:`Conventions` defines those arrays exactly.

How the documentation is organised
----------------------------------

The pages are in five groups, in the menu on the left:

- **Quickstart & Intro**: the pages above, plus
  :doc:`Handy References - Aide-mémoire`, a one-page summary of the
  functions of a distribution and how they relate.
- **Survival Analysis**: the *theory* pages. Each explains a family of
  methods from first principles: the problem it solves, the mathematics, how
  to read the results, what can go wrong and when to use something else.
  They are written so that you can learn the methods from them.
- **SurPyval Modelling**: the *how-to* pages. Each shows how to do that
  analysis in SurPyval, with worked examples whose output is computed when
  the documentation is built, so it always matches the current version. It
  also includes :doc:`applications`, complete examples from several fields.
  Each theory page links to its how-to page and the other way round.
- **SurPyval**: the API reference (:doc:`surpyval`), with every class,
  function and argument, and the :doc:`changelog`.
- **Community Guidelines**: where to get help, how to report a problem and
  how to contribute.

What SurPyval covers
--------------------

Every model is created the same way: a *fitter* (``surv.Weibull``,
``surv.KaplanMeier``, ``surv.CoxPH``, ...) has a ``fit()`` method that
takes your data and returns a fitted *model*, which you then ask for
survival probabilities, hazards, quantiles, confidence bounds, plots and
so on.

.. list-table::
   :header-rows: 1
   :widths: 22 58 20

   * - Area
     - What is in it
     - Theory / how-to
   * - Non-parametric estimation
     - Kaplan-Meier, Nelson-Aalen, Fleming-Harrington and Turnbull
       estimators with confidence bounds and bands; the log-rank test;
       restricted mean survival time; success-run (zero-failure) testing.
     - :doc:`Non-Parametric Estimation` /
       :doc:`Non-Parametric SurPyval Modelling`
   * - Parametric distributions
     - Continuous lifetime distributions (Weibull, Exponential, Gamma,
       LogNormal, LogLogistic, Gumbel, Normal, Logistic, Exponentiated
       Weibull, Beta, Rayleigh, Uniform, ...) and discrete ones (Geometric,
       Poisson, Binomial, Negative Binomial, Beta-Geometric, discrete
       Weibull), fitted by any of the five estimation methods in the table
       below. Offsets (the three-parameter Weibull), limited failure
       populations and zero-inflation; mixture models; your own
       distribution from a cumulative hazard; the Royston-Parmar flexible
       spline model; automatic choice of distribution with ``fit_best``.
     - :doc:`Parametric Estimation` /
       :doc:`Parametric SurPyval Modelling`
   * - Regression
     - Proportional hazards (Cox, with strata and time-varying covariates,
       and parametric), accelerated failure time, accelerated life with
       life-stress relationships (power law, Eyring, exponential/Arrhenius
       and others), proportional odds, additive hazards, Buckley-James,
       shared frailty; Brier score and time-dependent AUC for validating
       predictions.
     - :doc:`regression analysis` /
       :doc:`Regression Modelling with SurPyval`
   * - Competing risks
     - Items that can fail from one of several causes: cumulative
       incidence (Aalen-Johansen), parametric cause-specific models,
       cause-specific Cox regression, Fine-Gray subdistribution regression
       and Gray's test.
     - :doc:`Competing Risks Analysis` /
       :doc:`Competing Risks SurPyval Modelling`
   * - Recurrent events
     - Repairable items that fail again and again: the mean cumulative
       function, homogeneous and non-homogeneous Poisson processes
       (Crow-AMSAA, Duane, Cox-Lewis), imperfect repair (Kijima, G1, ARA,
       ARI), trend tests, and proportional-intensity regression.
     - :doc:`Recurrent Event Analysis` /
       :doc:`Recurrent Event Modelling with SurPyval`;
       :doc:`Recurrent Event Regression Analysis` /
       :doc:`Recurrent Event Regression Modelling with SurPyval`
   * - Degradation
     - Measurements that drift towards a failure threshold: general path
       models, Wiener and Gamma processes, destructive degradation,
       remaining useful life of a monitored unit, and accelerated and
       step-stress degradation tests.
     - :doc:`Degradation Analysis` /
       :doc:`Degradation Modelling with SurPyval`
   * - Multivariate
     - Dependent lifetimes joined by a copula (Clayton, Frank, Gumbel,
       Gaussian, independence), with censored and truncated margins.
     - :doc:`Multivariate Analysis` /
       :doc:`Multivariate Modelling with SurPyval`

Every fitted model can be saved to a dictionary or a JSON file and restored
with ``surv.from_dict`` or ``surv.from_json`` (see :doc:`Conventions`). A
survival tree and a random survival forest are available as beta-stage
models in ``surpyval.beta.ml`` (see :doc:`surpyval.beta`).

Estimating a single distribution
--------------------------------

SurPyval is unusual in how many ways it can estimate the parameters of a
distribution. Most packages offer maximum likelihood, and perhaps
probability plotting; SurPyval grew out of reproducing the probability
plotting of engineering practice and found along the way that there are
many ways to estimate parameters, each with its own strengths. The methods,
and the data each can use, are:

.. list-table:: SurPyval Modelling Methods
   :header-rows: 1
   :widths: 24 14 10 26 26

   * - Method
     - Para/Non-Para
     - Observed
     - Censored
     - Truncated
   * - Maximum Likelihood (MLE)
     - Parametric
     - Yes
     - Yes: left, right and interval
     - Yes: left and right, a different value for each observation if
       needed
   * - Maximum Product Spacing (MPS)
     - Parametric
     - Yes
     - Left and right; not interval
     - Yes, but one common ``tl`` and/or ``tr`` for all observations
   * - Probability Plotting (MPP)
     - Parametric
     - Yes
     - Right; left and interval only with ``heuristic="Turnbull"``
     - Left (with the Kaplan-Meier, Nelson-Aalen, Fleming-Harrington or
       Turnbull heuristic); right only with ``heuristic="Turnbull"``
   * - Mean Square Error (MSE)
     - Parametric
     - Yes
     - Yes: left, right and interval
     - No
   * - Method of Moments (MOM)
     - Parametric
     - Yes
     - No
     - No
   * - Kaplan-Meier
     - Non-Parametric
     - Yes
     - Right only
     - Left only
   * - Nelson-Aalen
     - Non-Parametric
     - Yes
     - Right only
     - Left only
   * - Fleming-Harrington
     - Non-Parametric
     - Yes
     - Right only
     - Left only
   * - Turnbull
     - Non-Parametric
     - Yes
     - Yes: left, right and interval
     - Yes: left and right

Maximum likelihood is the default (``how="MLE"``) and, with Turnbull for a
non-parametric view, handles every combination of data. The other methods
have their own uses: probability plotting is the traditional engineering
method and gives the familiar straight-line plot; maximum product spacing
is particularly good for offset distributions and distributions with a
finite bound, where the likelihood can be unbounded; and the method of
moments is quick for complete data. Some distributions do not support every
method (probability plotting needs a distribution that can be drawn as a
straight line, and discrete distributions cannot use MPS). Whenever a
method cannot use your data, SurPyval raises an error saying so rather than
silently ignoring part of it. The methods are explained in
:doc:`Parametric Estimation` and :doc:`Non-Parametric Estimation`, and
:doc:`Types of Data` has the same table split by type of censoring and
truncation.

A word of encouragement
-----------------------

Becoming a competent survival analyst depends on a strong understanding of
censoring, truncation and observations, together with a solid understanding
of the distributions used to describe lifetimes. Recognising that a real
situation is censored or truncated is what keeps an analysis from going
wrong, and it can be surprisingly difficult; :doc:`Types of Data` is the
place to build that skill. A good understanding of the distributions lets
you reason about the process that generated your data, and so choose an
appropriate model, if any, for your problem. Survival analysis is an
extremely powerful, and thoroughly interesting, tool, so don't give up; or
if you do give up, do the survival statistics on it.


Contents:
=========

.. toctree::
   :maxdepth: 2
   :caption: Quickstart & Intro

   Quickstart
   Types of Data
   Conventions
   Handy References - Aide-mémoire
   Data Wrangler Examples

.. toctree::
   :maxdepth: 2
   :caption: Survival Analysis

   Non-Parametric Estimation
   Parametric Estimation
   regression analysis
   Competing Risks Analysis
   Recurrent Event Analysis
   Recurrent Event Regression Analysis
   Degradation Analysis
   Multivariate Analysis

.. toctree::
   :maxdepth: 2
   :caption: SurPyval Modelling

   Non-Parametric SurPyval Modelling
   Parametric SurPyval Modelling
   Regression Modelling with SurPyval
   applications
   Competing Risks SurPyval Modelling
   Recurrent Event Modelling with SurPyval
   Recurrent Event Regression Modelling with SurPyval
   Degradation Modelling with SurPyval
   Multivariate Modelling with SurPyval

.. toctree::
   :maxdepth: 2
   :caption: SurPyval

   surpyval
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Community Guidelines

   Support
   Report an Issue <https://github.com/derrynknife/SurPyval/issues>
   Contributing
   acknowledgements



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
