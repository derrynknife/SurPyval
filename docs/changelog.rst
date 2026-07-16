Changelog
=========

v0.13.0 (unreleased)
--------------------

Distributions
~~~~~~~~~~~~~~

- Added three Tier-2 discrete distributions: ``Poisson`` (the count
  distribution on ``{0, 1, 2, ...}``, distinct from the recurrent Poisson
  *processes*), ``BetaGeometric`` (a discrete-time frailty model — Geometric
  with a Beta-mixed failure probability, whose marginal hazard decreases with
  time), and ``Discretize(distribution)``, a factory that turns any
  non-negative continuous distribution into its integer-binned counterpart
  (``K = ceil(T)``, so ``P(K=k) = F(k) - F(k-1)`` and the discrete survival
  equals the continuous survival), fit by MLE on the underlying parameters.
- ``Beta.fit(how="MPP")`` now raises a clear ``ValueError`` (the Beta has no
  linearising probability plot) instead of a raw ``NotImplementedError``, and
  points to ``MLE`` / ``MSE`` / ``MOM``.

Competing risks
~~~~~~~~~~~~~~~

- Added ``ParametricCompetingRisks``, a fully parametric competing-risks model:
  a parametric distribution is fitted to each cause's cause-specific hazard
  (the joint likelihood factorises, so each cause is fitted with the other
  causes' events treated as right-censored) and smooth, extrapolatable
  cumulative-incidence functions are assembled from them. Provides ``fit`` /
  ``fit_from_df`` (with a per-cause distribution mapping), all-cause and
  cause-specific ``hf`` / ``Hf`` / ``sf`` / ``ff``, the subdistribution density
  ``iif``, the cumulative incidence ``cif``, ``probability_of_cause``, sampling
  via ``random``, and ``aic`` / ``bic`` / ``neg_ll``. Complements the existing
  nonparametric ``CompetingRisks`` estimator and the semi-parametric
  cause-specific Cox / Fine-Gray regression models.

Regression — Cox proportional hazards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added time-varying-covariate support in counting-process (start-stop)
  format: ``CoxPH.fit_tvc`` / ``fit_tvc_from_df`` take one row per interval
  ``(ident, start, stop, event, Z)``, validated by ``handle_tvc``, and
  ``SemiParametricRegressionModel.predict_tvc`` gives a subject's survival
  along a supplied covariate path.
- **Fixed** the Breslow baseline hazard to respect left-truncation / delayed
  entry (``tl``) and case weights (``n``); ``H0`` was previously wrong for any
  delayed-entry fit even though the coefficients were correct.
- ``CoxPH.fit`` gained a minimisation fallback so staggered delayed-entry data
  (e.g. the start-stop representation) converges where the root-finder stalled.
- Right / interval truncation is now rejected with a clear, Cox-specific error
  (a 2-D ``tl``), since the forward partial likelihood cannot express it.

Truncation
~~~~~~~~~~

- Verified and tested that the parametric AFT / PO / PH truncation correction
  uses each row's own covariates: a covariate-recovery test confirms the
  coefficient and scale are recovered from left-, right-, interval- and
  partially-truncated data.

Documentation
~~~~~~~~~~~~~~

- Added worked, executed examples for regression confidence bounds,
  Buckley-James AFT, competing-risks regression (Fine-Gray + cause-specific
  Cox), degradation ADT covariates and two-stage bounds, the copula module,
  and the combined data-input flexibility; wrote the Maximum Product of
  Spacings (MPS) estimation theory section.

v0.12.0 (15 Jul 2026)
---------------------

A large release consolidating the regression, recurrent-event, competing-risks,
degradation, and multivariate work accumulated since ``v0.10.1``. Requires
Python 3.11+ and NumPy 2.

Regression
~~~~~~~~~~

- Standardised every univariate regression fitter (accelerated failure time,
  proportional hazards, proportional odds, additive hazards, accelerated life)
  on a common instance-based ``fit()`` / ``fit_from_df()`` API with pandas and
  `formulaic <https://matthewwardrop.github.io/formulaic/>`_ formula support.
- ``CoxPH`` gained the Efron tie handling in addition to Breslow, and its
  analytic (Efron) information matrix is now correct, so standard errors and
  p-values are produced for tied data.
- Added delta-method confidence bounds to the parametric regression models:
  ``cb()`` on a predicted function at a covariate vector, ``param_cb()`` on a
  single coefficient, and ``covariance()`` / ``standard_errors()`` /
  ``parameter_names()`` on the fitted parameters.
- Added ``BuckleyJames``, a semi-parametric accelerated-failure-time model with
  an unspecified error distribution (the accelerated-time counterpart of Cox),
  fitted by the Buckley-James imputation iteration with percentile-bootstrap
  coefficient intervals.
- Added a parametric ``AdditiveHazards`` regression fitter.

Competing risks
~~~~~~~~~~~~~~~~

- Added a competing-risks regression module with a cause-specific Cox model and
  a Fine-Gray subdistribution-hazard model (``CompetingRisksProportionalHazards``),
  each with ``fit()`` / ``fit_from_df()`` and cumulative-incidence prediction.

Recurrent events
~~~~~~~~~~~~~~~~~

- Standardised the recurrent-model API on the same instance-based fitters the
  univariate distributions use: ``HPP``, ``CrowAMSAA``, ``Duane``,
  ``CoxLewis``, ``NonParametricCounting``, the renewal fitters
  (``GeneralizedRenewal``/``GeneralizedOneRenewal``/``ARA``/``ARI``) and the
  proportional-intensity fitters are now configured singleton instances with an
  instance-method ``fit()``. Public ``Model.fit(...)`` calls are unchanged;
  internally provided by the ``surpyval.utils.fitter.singleton_fitter``
  decorator. Removed the unused ``ParametricRecurrenceRegressionModel`` stub.
- Added parameter-uncertainty and diagnostic support to the recurrent models,
  and removed the ``dist='t'`` heuristic from the recurrent ``mcf_cb``.

Degradation
~~~~~~~~~~~

- Added the ``surpyval.degradation`` pseudo-failure-time analysis module:
  per-unit path fits over a library of path models, extrapolation to a failure
  threshold, and a fitted life distribution, with population path-parameter
  estimation (Lu-Meeker two-stage and REML) and Bayesian remaining-useful-life
  prediction (``predict_rul``).
- Added two-stage (delta-method and bootstrap) confidence bounds on the fitted
  life model that fold in the first-stage path/extrapolation uncertainty
  (``DegradationModel.cb`` / ``life_parameter_covariance``).
- Added Stage-1 accelerated degradation testing (ADT) covariates: passing
  ``Z`` to ``DegradationAnalysis.fit`` fits a regression life model on the
  pseudo failure times so life can be predicted at any stress condition.

Multivariate
~~~~~~~~~~~~~

- Added a ``surpyval.multivariate`` module with copula models over the
  univariate distributions.

Distributions and core
~~~~~~~~~~~~~~~~~~~~~~~~

- Added discrete lifetime distributions.
- Hardened input validation in the ``handle_xicn`` / ``xcnt_handler`` data
  handlers, and fixed a reserved-attribute clash.
- Simulation and ``dist='t'`` cleanups.

v0.10.1.0 (25 Mar 2022)
-----------------------

- Changed plot methods to now take 'Axis' object. This allows a user to pass in an existing axis.
- plot functions now return an Axis object instead of the Lines2D object. Allows for easy user update after plotting.
- Added fs_to_xcn as it was dropped in 10.0.1.
- Changed all imports for numpy to be done from the surpyval module. This will allow for easy maintenance in future in the event of deprecated autograd.

v0.10.0.1 (22 Nov 2021)
-----------------------

- Removed fsl_to_xcn function and replaced with fsli_to_xcn function that is able to take any combination of fsli.

v0.10.0 (9 Aug 2021)
--------------------

- Version snapshot for JOSS review

v0.9.0 (5 Aug 2021)
-------------------

- Better initial estimates in the ``_parameter_initialiser`` for the lfp data (use max F from nonp estimate...)
- `issue #13 <https://github.com/derrynknife/SurPyval/issues/13>`_ - Better failures when insufficient data provided.
- `issue #12 <https://github.com/derrynknife/SurPyval/issues/12>`_ - Created ``fsli_to_xcn`` helper function.
- Fixed bug in confidence bounds implementation for offset distributions. CBs were not using the offset and were therefore way out. Now fixed.
- Created a  ``NonParametric.cb()`` method to match ``Parametric`` API for confidence bounds.
- Cleaned up NonParametric code (removed some technical debt and duplicated code).
- Changed the ``__repr__`` function in ``NonParametric`` to be aligned to ``Parametric``
- Updated the docstring for ``fit()`` for ``NonParametric``
- Fixed bug in ``NonParametric`` that required the ``x`` input to be in order for the functions (e.g. ``df`` etc.).
- ``CoxPH`` released.
- General AL fitter in beta
- General PH fitter in beta
- Created ``Linear``, ``Power``, ``InversePower``, ``Exponential``, ``InverseExponential``, ``Eyring``, ``InverseEyring``, ``DualPower``, ``PowerExponential``, ``DualExponential`` life models.
- Created ``GeneralLogLinear`` life model for variable stress count input.
- For each combination of a SurPyval distribution and life model, there is an instance to use ``fit()``. For example there are ``WeibullDualExponential``, ``LogNormalPower``, ``ExponentialExponential`` etc.
- Docs Updates:
	- Add application examples to docs:
		- Reliability Engineering
		- Actuary / Demography
		- `Social Science/Criminology <https://link.springer.com/article/10.1007/s10940-021-09499-5>`_
		- Boston Housing
		- Medical science
		- `Economics <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0232615>`_
		- Biology - Ware, J.H., Demets, D.L.: Reanalysis of some baboon descent data. Biometrics 459–463 (1976).

v0.8.0 (27 July 2021)
---------------------

- Made backwards incompatible changes to ``LFP`` models, these are now created with the ``lfp=True`` keyword in the ``fit()`` method
- Created ability to fit zero-inflated models. Simply pass the ``zi=True`` option to the ``fit()`` method.
- Chanages to ``utils.xcnt_handler`` to ensure ``x``, ``xl``, and ``xr`` are handled consistently.
- changed the way ``__repr__`` displays a Parametric object.
- Changed the default for plotting to be ``Fleming-Harrington``. This was a result of seeing how poorly the ``Nelson-Aalen`` method fits zero inflated models. FH therefore offers the best performance of a Non-Parametric estimate at the low values of the survival function (as KM reaches 0 for fully observed data) and at high values (KM is good but NA is poor).
- Added a Fleming-Harrington method to the Turnbull class.
- Improved stability with dedicated ``log_sf``, ``log_ff``, and ``log_df`` functions. Less chance of overflows and therefore better convergence.
- Changed interpolation method of ``NonParametric``. Allows for use of cubic interpolation
- Changed ``from_params`` to accept lfp and zi (or any combo)
- Changed ``random()`` in ``Parametric`` so that lfp or zi models can be simulated!
- Improved the way surpyval fails
- Substantial docs updates.


v0.7.0 (19 July 2021)
---------------------

- Major changes to the confidence bounds for ``Parametric`` models. Now use the ``cb()`` method for every bound.
- Removed the ``OffsetParametric`` class and made ``Parametric`` class now work with (or without) an offset.
- Minor doc updates.