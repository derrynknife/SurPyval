Saving and Loading Models
=========================

A fitted model can be saved and restored later, or in another process,
without refitting. Almost every fitted SurPyval model can be serialised
to a plain dictionary with ``to_dict()`` and written to JSON with
``to_json(path)``. The dictionaries hold only plain Python types, so they
can also go straight into a document store (the round trip through
BSON/MongoDB is tested). What a restored model keeps, and what needs the
original data, is described in :doc:`Conventions`.

Restoring takes one call whichever class wrote the file: the
package-level readers dispatch on the serialised dictionary itself.

.. code:: python

    import surpyval

    model = surpyval.Weibull.fit(x)
    model.to_json("weibull.json")

    restored = surpyval.from_json("weibull.json")  # any model's file
    restored = surpyval.from_dict(model_dict)      # any model's dict

The class-level readers on the fitted-model classes
(``Parametric.from_dict``, ``SemiParametricRegressionModel.from_dict``,
... and the matching ``from_json``) remain available when the model class
is known up front; each rejects a dictionary written by a different class
with a ``ValueError``. The readers belong to the *model* classes, not to
the fitters: ``Weibull.from_dict`` and ``CoxPH.from_dict`` do not exist.

``CompetingRisksProportionalHazards`` serialises the same way, for both
``model="Cox"`` and ``model="Fine-Gray"``; its per-cause optimiser results
(``results``) are not stored.

Some details differ between families:

- Only the univariate ``Parametric`` and ``NonParametric`` models take
  ``to_dict(with_data=True)``, which stores the fitted data as well so
  that ``plot`` and likelihood-ratio bounds (parametric) or
  ``bootstrap_cb`` (non-parametric) work after restoring;
  ``to_json(path, with_data=True)`` writes the same to a file (any
  other model's ``to_json`` refuses ``with_data=True`` with a
  ``TypeError``). The information criteria need no data: a fitted
  univariate or regression model stores their sample size (``"ic_n"``),
  so ``bic`` and ``aic_c`` work on the restored model either way.
- The degenerate ``NeverOccurs`` and ``InstantlyOccurs`` distributions
  have no fitted state: the class itself is the model, so ``to_dict``,
  ``to_json``, ``from_dict`` and ``from_json`` are called on the class
  (``NeverOccurs.to_json(path)``), and ``surpyval.from_json`` returns the
  class.
- Every reader -- the package-level ones and each class's own
  ``from_dict`` / ``from_json`` alike -- checks the ``"schema"`` (an
  integer no newer than this SurPyval), names the entry a truncated or
  hand-edited dictionary is missing, and refuses the parameters of a
  univariate parametric model that fall outside the distribution's
  bounds, each with a ``ValueError``.
- These cannot be saved and raise an error from ``to_dict``: an
  accelerated-life model with a user-defined
  life model, and a copula of a custom family. A regression fitted with
  a formula is saved with its levels and fitted transform statistics
  (``C(...)``, ``scale()``, ``poly()``, splines); a formula whose state
  cannot be stored as JSON raises at ``to_dict`` instead.
- A model of a ``Discretize(...)`` distribution is read back like any
  other. A model of a ``CustomDistribution`` stores the distribution's
  name only (its cumulative hazard is a Python function):
  ``from_dict`` reads it back in a session that has constructed the same
  ``CustomDistribution`` again, and otherwise raises a ``ValueError``
  saying so.

Infinite and NaN values
-----------------------

A fitted model holds values that are not finite numbers: an untruncated
bound (``-inf``, ``inf``), a cumulative hazard after the last death
(``inf``), an undefined variance (``nan``). JSON cannot write them --
Python's ``json`` writes the non-standard ``Infinity`` and ``NaN``, which
strict parsers (JavaScript's ``JSON.parse``, many databases) refuse -- so
the dictionaries use a convention of their own, and are strict JSON:

- each non-finite value is written as ``null``;
- the dictionary holding it records what every such ``null`` stood for
  under ``"non_finite"``: for each kind present (``"inf"``, ``"-inf"``,
  ``"nan"``) a list of `JSON Pointers
  <https://www.rfc-editor.org/rfc/rfc6901>`_ to the values, relative to
  that dictionary. A model dictionary nested in another (a copula's
  margins, a tree's leaves) carries its own record. A ``null`` that no
  record names is an ordinary missing value.

.. code:: python

    >>> surpyval.KaplanMeier.fit([3, 4, 5]).to_dict()
    {..., 'H': [0.405..., 1.098..., None],
     'greenwood': [0.166..., 0.666..., None],
     'non_finite': {'inf': ['/H/2'], 'nan': ['/greenwood/2']}, 'schema': 2}

Every reader restores the original values, so a round trip is exact. A
consumer in another language sees ``null`` where no number applies, and
can use the record to recover the values. Dictionaries and files written
before this convention (``"schema"`` 0 or 1, holding ``Infinity`` or
``NaN``, which Python's ``json`` reads) still load.

Each file is stamped with the *oldest* schema version that reads it
correctly (:func:`~surpyval.serialisation.required_schema`). A file with no
non-finite value -- most fitted parametric, regression, recurrent, copula
and degradation models -- has exactly the schema-1 layout and is stamped
``1``, so SurPyval 0.20, which reads schema 1, still loads it (the test suite
checks this against that release). A file that records non-finite values as
``null``, as most non-parametric models and anything saved with
``with_data=True`` do, is stamped ``2``; an older SurPyval refuses it with an
error asking for an upgrade rather than misreading its ``null`` values. So is
a regression model fitted with a formula that SurPyval 0.20 cannot rebuild
(a wrapped categorical such as ``C(g)``, integer category levels, or a
fitted transform such as ``scale(z)``); a formula of plain columns and
string categoricals is stamped ``1``. So is a regression model fitted with
``center=True`` (``CoxPH``, ``FineGray``,
``CompetingRisksProportionalHazards`` or a parametric regression) whose
covariate means ``center`` are not all zero: its baseline is that of a unit
at ``center``, which SurPyval 0.20 would ignore, reading the baseline as at
``Z = 0``. A model with its baseline at ``Z = 0``, the default, has no
``center`` and the layout of before. ``SurpyvalData.to_json`` uses the same
convention. :func:`~surpyval.serialisation.encode_non_finite` and
:func:`~surpyval.serialisation.decode_non_finite` apply and undo it on
any dictionary.

The later versions, each refused by the SurPyval releases before it:

.. list-table::
   :header-rows: 1

   * - Schema
     - Stamped on
     - First read by
   * - 3
     - ``SurpyvalData`` with sampling weights ``w``: a reader without
       weights would drop them. (Weighted models were schema 3 in 0.26.)
     - 0.26
   * - 4
     - A model fitted with weights ``w`` (``"weighted": true``), whose
       ``"hess_inv"`` is the inverse Hessian and ``"covariance"`` the
       robust covariance (0.26 wrote the robust covariance as
       ``"hess_inv"``); a stratified Cox model (``"strata"``); a model
       whose coefficients carry their signs (``"coefficient_signs"``).
     - 0.27

.. _saved-model-dictionary:

The saved-model dictionary
--------------------------

Code outside SurPyval (a web service, a database query, a program in
another language) can read the dictionary ``to_dict`` writes without
loading the model. Its layout is a contract, set out here key by key:

- Every dictionary written by an earlier release keeps loading, whatever
  its ``"schema"``.
- From 1.0, a 1.x release may **add** keys, but never renames or removes
  one, nor changes what an existing key holds. A reader should therefore
  ignore keys it does not know. (See :doc:`Deprecation Policy`.)
- A key listed as written "when it applies" is left out, not written as
  ``null``, when it does not apply.

The tables below were made by fitting one model of every class the
readers know and listing its keys; ``surpyval/tests/test_json_serialisation.py``
checks them against the code, so they cannot drift.

Keys in every dictionary
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - What it holds
   * - ``schema``
     - The oldest layout version that reads the dictionary correctly, an
       integer (see above).
   * - ``model`` or ``parameterization``
     - Which class wrote it, so that :func:`~surpyval.serialisation.from_dict`
       can choose the reader. Most classes write ``"model"``, set to the
       class name. Four write ``"parameterization"`` instead:
       ``"parametric"`` (``Parametric``), ``"non-parametric"``
       (``NonParametric``, whose ``"model"`` holds the estimator,
       ``"Kaplan-Meier"`` for example), ``"parametric-regression"``
       (``ParametricRegressionModel``) and ``"copula"``
       (``CopulaModel``). A few classes write both; ``"model"`` then
       decides.

.. _saved-model-optional-keys:

Keys any model writes when they apply
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - When, and what it holds
   * - ``non_finite``
     - The dictionary holds a value that is not a finite number: where
       each ``null`` came from (see `Infinite and NaN values`_).
   * - ``unit``
     - The model has a unit (``unit=`` on ``fit``): its text.
   * - ``weighted``
     - The model was fitted with sampling weights ``w``: ``true``.
   * - ``effective_n``
     - With ``weighted``: the effective sample size.
   * - ``data``
     - ``to_dict(with_data=True)`` (``Parametric`` and ``NonParametric``):
       the fitted data, ``x``, ``c``, ``n``, ``t`` (and ``w``). Some
       models that need their data to predict always store it under this
       key (listed below).
   * - ``feature_names``
     - A regression model whose covariates have names: the names, in
       the order of the coefficients.
   * - ``formula``
     - A regression model fitted with a formula: its text.
   * - ``formula_meta``
     - With ``formula``: the levels and fitted transform statistics
       needed to build the same design matrix again.
   * - ``center``
     - A regression model fitted with ``center=True``: the covariate
       values its baseline is at.

Names shared between families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where a key below appears, it means the same thing whichever model wrote
it:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - What it holds
   * - ``params``
     - The fitted parameters (or coefficients), a list of numbers.
   * - ``param_names``
     - The names of ``params``, in the same order.
   * - ``covariance``
     - The parameters' covariance matrix (with weights, the robust one).
   * - ``hess_inv``
     - The inverse Hessian of the negative log-likelihood. Without
       weights it equals ``covariance``.
   * - ``_neg_ll``
     - The fitted negative log-likelihood.
   * - ``ic_n``
     - The sample size used by ``bic`` and ``aic_c``.
   * - ``maximum``
     - Whether the fit reached a maximum of the likelihood.
   * - ``data_summary``
     - The "Data" line of the model's printout.
   * - ``distribution`` or ``dist``
     - The name of the distribution (``"Weibull"``).
   * - ``how``
     - The fitting method (``"MLE"``, ``"MPP"``, ...).
   * - ``kind``
     - The model's family within its class (``"Cox"``, ``"PH"``, ...).
   * - ``x``
     - The times at which a step-function estimate is stored.

Keys of each class
~~~~~~~~~~~~~~~~~~

"Always" lists the keys every fitted model of that class writes, besides
``schema`` and the identifying key; "When it applies" lists the keys
written only by some fits (a Turnbull estimate, a stratified or
delayed-entry Cox model, a parametric model with fixed parameters, and
so on) beyond those in `Keys any model writes when they apply`_. A model
nested inside another (a competing-risks model's per-cause models, a
forest's trees, a copula's margins) is a dictionary of its own class,
with its own ``schema``.

.. list-table::
   :header-rows: 1
   :widths: 22 22 36 20

   * - Class
     - Identified by
     - Always
     - When it applies
   * - ``Parametric``
     - ``"parameterization": "parametric"``
     - ``distribution``, ``how``, ``param_names``, ``params``, ``lfp``, ``p``, ``zi``, ``f0``, ``offset``, ``gamma``, ``hess_inv``, ``covariance``, ``_neg_ll``, ``maximum``, ``data_summary``, ``ic_n``
     - ``fixed``, ``custom``, ``event_counts``, ``n_trials``
   * - ``NonParametric``
     - ``"parameterization": "non-parametric"``
     - ``x``, ``r``, ``d``, ``R``, ``F``, ``H``, ``greenwood``, ``data_summary``
     - ``estimator``, ``tol``, ``max_iter``, ``algorithm``, ``support``, ``band_n``
   * - ``ParametricRegressionModel``
     - ``"parameterization": "parametric-regression"``
     - ``kind``, ``distribution``, ``phi_param_map``, ``params``, ``k``, ``k_dist``, ``fixed``, ``gamma``, ``p``, ``f0``, ``covariance``, ``_neg_ll``, ``maximum``, ``ic_n``, ``data_summary``
     - ``reg_model_name``, ``inference_centring``, ``life_model_name``
   * - ``DistributionalRegressionModel``
     - ``"model": "DistributionalRegressionModel"``
     - ``kind``, ``distribution``, ``links``, ``coefficient_signs``, ``covariate_positions``, ``width``, ``phi_param_map``, ``params``, ``aliased_coefficients``, ``k``, ``k_dist``, ``fixed``, ``covariance``, ``inference_centring``, ``_neg_ll``, ``maximum``, ``ic_n``, ``data_summary``
     - --
   * - ``MixtureCureModel``
     - ``"model": "MixtureCureModel"``
     - ``kind``, ``distribution``, ``links``, ``coefficient_signs``, ``covariate_positions``, ``width``, ``phi_param_map``, ``params``, ``aliased_coefficients``, ``k``, ``k_dist``, ``fixed``, ``covariance``, ``inference_centring``, ``_neg_ll``, ``maximum``, ``ic_n``, ``data_summary``, ``latency``
     - --
   * - ``ZeroInflatedModel``
     - ``"model": "ZeroInflatedModel"``
     - ``kind``, ``distribution``, ``links``, ``coefficient_signs``, ``covariate_positions``, ``width``, ``phi_param_map``, ``params``, ``aliased_coefficients``, ``k``, ``k_dist``, ``fixed``, ``covariance``, ``inference_centring``, ``_neg_ll``, ``maximum``, ``ic_n``, ``data_summary``, ``latency``
     - --
   * - ``SemiParametricRegressionModel``
     - ``"model": "SemiParametricRegressionModel"``
     - ``kind``, ``beta``, ``params``, ``x``, ``r``, ``d``, ``h0``, ``H0``, ``tie_method``, ``baseline_method``, ``is_tvc``, ``p_values``, ``se``, ``covariance``, ``_neg_ll``, ``ic_n``, ``maximum``, ``data_summary``
     - ``tl``, ``strata``, ``firth``
   * - ``FrailtyModel``
     - ``"model": "FrailtyModel"``
     - ``kind``, ``family``, ``distribution``, ``dist_params``, ``beta``, ``theta``, ``k_dist``, ``param_names``, ``group_labels``, ``frailties``, ``n_obs``, ``n_events``, ``n_groups``, ``n_events_weighted``, ``n_obs_weighted``, ``_neg_ll``, ``maximum``, ``covariance``
     - --
   * - ``CoxFrailtyModel``
     - ``"model": "CoxFrailtyModel"``
     - ``kind``, ``family``, ``tie_method``, ``beta``, ``theta``, ``center``, ``x``, ``h0``, ``H0``, ``parameter_names``, ``group_labels``, ``frailties``, ``n_obs``, ``n_events``, ``n_groups``, ``_neg_ll``, ``log_likelihood_no_frailty``, ``k``, ``n_events_weighted``, ``n_obs_weighted``, ``data_summary``, ``maximum``, ``covariance``
     - --
   * - ``AdditiveHazardsModel``
     - ``"model": "AdditiveHazardsModel"``
     - ``beta``, ``params``, ``x``, ``h0``, ``H0``, ``covariance``, ``se``, ``p_values``, ``drift``
     - --
   * - ``AalenAdditiveModel``
     - ``"model": "AalenAdditiveModel"``
     - ``x``, ``B``, ``B_var``, ``B_cov``, ``n_terms``, ``penalizer``, ``aliased``
     - --
   * - ``BuckleyJamesModel``
     - ``"model": "BuckleyJamesModel"``
     - ``beta``, ``resid``, ``resid_surv``, ``n_iter``, ``converged``, ``data``
     - --
   * - ``ProportionalOddsModel``
     - ``"model": "ProportionalOddsModel"``
     - ``beta``, ``params``, ``se``, ``covariance``, ``p_values``, ``x``, ``d``, ``g0``, ``G0``, ``_neg_ll``, ``ic_n``, ``n_iter``, ``maximum``, ``data_summary``
     - --
   * - ``MixtureModel``
     - ``"model": "MixtureModel"``
     - ``dist``, ``m``, ``params``, ``w``, ``maximum``, ``_neg_ll``, ``ic_n``, ``theta_covariance``
     - ``custom``
   * - ``RoystonParmarModel``
     - ``"model": "RoystonParmarModel"``
     - ``scale``, ``knots``, ``params``, ``n``, ``n_events``, ``_neg_ll``, ``maximum``, ``ic_n``, ``covariance``
     - --
   * - ``FineGrayModel``
     - ``"model": "FineGrayModel"``
     - ``cause``, ``beta``, ``p_values``, ``baseline_times``, ``baseline_cumhaz``, ``_neg_ll``, ``maximum``, ``covariance``, ``se``, ``model_covariance``, ``ic_n``
     - --
   * - ``CompetingRisksProportionalHazards``
     - ``"model": "CompetingRisksProportionalHazards"``
     - ``how``, ``event_idx_map``, ``n_event_types``, ``x``, ``betas``, ``h0_e``, ``maximum``, ``_neg_ll``, ``ic_n``, ``covariances``
     - ``fg_models``
   * - ``ParametricCompetingRisks``
     - ``"model": "ParametricCompetingRisks"``
     - ``causes``, ``models``
     - ``joint``
   * - ``ParametricCompetingRisksRegression``
     - ``"model": "ParametricCompetingRisksRegression"``
     - ``causes``, ``models``
     - --
   * - ``ParametricMultiState``
     - ``"model": "ParametricMultiState"``
     - ``transitions``, ``x``, ``tl``, ``state``, ``e``, ``n``, ``i``, ``clock``, ``models``
     - --
   * - ``ParametricMultiStateRegression``
     - ``"model": "ParametricMultiStateRegression"``
     - ``transitions``, ``x``, ``tl``, ``state``, ``e``, ``n``, ``i``, ``clock``, ``models``, ``Z``, ``covariate_names``
     - --
   * - ``CompetingRisks``
     - ``"model": "CompetingRisks"``
     - ``event_idx_map``, ``n_event_types``, ``method``, ``x``, ``d``, ``r``, ``h0``, ``H0``, ``S``, ``d_e``, ``h0_e``, ``H0_e``, ``IIF``, ``CIF``
     - ``support``
   * - ``MultiState``
     - ``"model": "MultiState"``
     - ``transitions``, ``x``, ``tl``, ``state``, ``e``, ``n``, ``i``
     - --
   * - ``MultiStateProportionalHazards``
     - ``"model": "MultiStateProportionalHazards"``
     - ``transitions``, ``x``, ``tl``, ``state``, ``e``, ``n``, ``i``, ``Z``, ``covariate_names``, ``shared``, ``tie_method``
     - --
   * - ``CauseSpecificMCF``
     - ``"model": "CauseSpecificMCF"``
     - ``event_types``, ``models``
     - ``support``
   * - ``CauseSpecificNHPP``
     - ``"model": "CauseSpecificNHPP"``
     - ``dist``, ``event_types``, ``models``, ``covariances``, ``end``
     - --
   * - ``CauseSpecificProportionalIntensity``
     - ``"model": "CauseSpecificProportionalIntensity"``
     - ``baseline``, ``event_types``, ``models``, ``covariances``
     - ``feature_names``
   * - ``NonParametricCounting``
     - ``"model": "NonParametricCounting"``
     - ``x``, ``mcf_hat``, ``var``, ``origin``
     - ``support``
   * - ``ParametricRecurrenceModel``
     - ``"model": "ParametricRecurrenceModel"``
     - ``dist``, ``params``, ``how``, ``covariance``, ``maximum``
     - --
   * - ``ProportionalIntensityModel``
     - ``"model": "ProportionalIntensityModel"``
     - ``kind``, ``dist``, ``param_names``, ``params``, ``coeffs``, ``covariance``, ``maximum``
     - ``feature_names``
   * - ``RecurrentCoxModel``
     - ``"model": "RecurrentCoxModel"``
     - ``kind``, ``cox``, ``robust_covariance``, ``n_items``, ``n_events``, ``max_events``
     - ``strata``
   * - ``RenewalModel``
     - ``"model": "RenewalModel"``
     - ``family``, ``dist``, ``params``, ``restoration``, ``how``, ``covariance``, ``maximum``, ``kijima_type``
     - ``m``
   * - ``DegradationModel``
     - ``"model": "DegradationModel"``
     - ``x``, ``y``, ``i``, ``units``, ``threshold``, ``path_model``, ``path_params``, ``pseudo_failure_times``, ``c``, ``life_model``, ``measurement_var``, ``path_param_mean``, ``path_param_cov``, ``path_param_sample_cov``, ``population_method``, ``path_selection``, ``Z``, ``how``, ``links``, ``path_param_fixed``, ``path_param_fixed_names``, ``path_param_link_cov``
     - --
   * - ``InducedFailureDistribution``
     - ``"model": "InducedFailureDistribution"``
     - ``samples``, ``threshold``, ``path_name``
     - ``stress``
   * - ``WienerProcessModel``
     - ``"model": "WienerProcessModel"``
     - ``mu``, ``sigma``, ``threshold``, ``y0``, ``covariance``, ``_neg_ll``, ``ic_n``, ``maximum``
     - ``gamma``, ``stress_ref``, ``Z_cols``
   * - ``GammaProcessModel``
     - ``"model": "GammaProcessModel"``
     - ``alpha``, ``beta``, ``threshold``, ``y0``, ``covariance``, ``_neg_ll``, ``ic_n``, ``maximum``
     - ``gamma``, ``stress_ref``, ``Z_cols``
   * - ``DestructiveDegradationModel``
     - ``"model": "DestructiveDegradationModel"``
     - ``distribution``, ``transform``, ``direction``, ``beta``, ``sigma``, ``threshold``, ``_neg_ll``, ``transform_scores``, ``data``, ``maximum``
     - --
   * - ``CopulaModel``
     - ``"parameterization": "copula"``
     - ``copula``, ``params``, ``how``, ``margins``, ``maximum``
     - ``rotation``, ``_neg_ll``, ``ic_n``, ``k``, ``covariance``
   * - ``SurvivalTree``
     - ``"model": "SurvivalTree"``
     - ``kind``, ``n_features_split``, ``selection``, ``alpha_split``, ``min_split_gain``, ``root``, ``n_covariates``
     - --
   * - ``RandomSurvivalForest``
     - ``"model": "RandomSurvivalForest"``
     - ``kind``, ``n_trees``, ``bootstrap``, ``selection``, ``alpha_split``, ``min_split_gain``, ``trees``
     - --
   * - ``NeverOccurs``
     - ``"model": "NeverOccurs"``
     - --
     - --
   * - ``InstantlyOccurs``
     - ``"model": "InstantlyOccurs"``
     - --
     - --
   * - ``JointModel``
     - ``"model": "JointModel"``
     - ``distribution``, ``path_model``, ``links``, ``varying``, ``link``, ``params``, ``parameter_names``, ``aliased``, ``path_covariates``, ``hazard_covariates``, ``marker_reference``, ``centre``, ``covariate_means``, ``n_items``, ``n_events``, ``n_readings``, ``_neg_ll``, ``maximum``, ``covariance``
     - --
   * - ``BayesianParametric``
     - ``"model": "BayesianParametric"``
     - ``distribution``, ``offset``, ``lfp``, ``zi``, ``fixed``, ``method``, ``priors``, ``data``, ``posterior``, ``log_evidence``, ``diagnostics``, ``exact``, ``surpyval_version``
     - --


Readers and helpers
-------------------

.. autofunction:: surpyval.serialisation.from_dict

.. autofunction:: surpyval.serialisation.from_json

.. autofunction:: surpyval.serialisation.encode_non_finite

.. autofunction:: surpyval.serialisation.decode_non_finite

.. autofunction:: surpyval.serialisation.required_schema
