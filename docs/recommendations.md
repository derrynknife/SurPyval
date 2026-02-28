# Recommendations for SurPyval Documentation and Package Structure

Based on an architectural and codebase review, here are several recommendations for improving SurPyval's package structure and documentation, specifically regarding Accelerated Failure Time (AFT) modeling, Proportional Hazards (PH) modeling, and Counting Processes.

## 1. Package Structure Recommendations

### Consolidate Regression and Recurrence Regression Models
Currently, there is a divergence in how regression models are structured.
*   Standard survival regression models are located in `surpyval/regression/` (e.g., `cox_ph.py`, `accelerated_failure_time.py`, `proportional_hazards_fitter.py`).
*   Recurrence/Counting regression models are located in `surpyval/recurrence/regression/` (e.g., `proportional_intensity.py`, `hpp_proportional_intensity.py`, `nhpp_proportional_intensity.py`).

**Recommendation:** It would improve discoverability and logical flow to consolidate all regression-based modeling under `surpyval/regression/`. You could introduce sub-packages such as `surpyval/regression/survival/` for classic survival models (PH, AFT) and `surpyval/regression/recurrence/` for proportional intensity and recurrence regressions.

### Standardize Naming Conventions
There are naming inconsistencies between class definitions and what is exposed to the user.
*   `AcceleratedFailureTimeFitter` is defined, but when used with life models, the dynamically generated classes are suffixed with `AL` (Accelerated Life) instead of `AFT` (e.g., `WeibullPowerAL`). However, the explicitly defined `WeibullInversePowerAFT` uses `AFT`.
*   **Recommendation:** Unify terminology. If standardizing on Accelerated Failure Time, ensure dynamically generated class names use the `AFT` suffix (e.g., `WeibullPowerAFT`) rather than `AL`.

### Extracting `lifemodels`
The `lifemodels` directory under `surpyval/regression/` contains multiple models (`power`, `exponential`, `linear`, etc.). These are primarily used for AFT modeling.
*   **Recommendation:** Move `lifemodels` into a more descriptive submodule like `surpyval/regression/aft_models/` or keep it as `surpyval/regression/lifemodels/` but ensure it is explicitly linked to AFT in the documentation.

## 2. Documentation Improvements

### Accelerated Failure Time (AFT) Modeling
The documentation for AFT modeling is currently very sparse or completely missing.
*   In `docs/regression/parametric.rst`, there is a detailed section for "Parametric Proportional Hazards," but there is no equivalent section for AFT models, despite `surpyval/regression/__init__.py` creating numerous AFT models dynamically (e.g., `WeibullInversePowerAFT`, `NormalLinearAL`).
*   The `docs/surpyval.regression.rst` file lists "Accelerated Time Models" and "Accelerated Life Models" as headers but contains no links to sub-pages or content under them.
*   **Recommendation:** Create a dedicated file `docs/regression/accelerated_failure_time.rst` to document the AFT modeling capabilities. Explain the difference between `AFT` and `AL` terminology if both are kept. Provide examples of how to fit data using these dynamic classes.

### Proportional Hazards (PH) Modeling
The Proportional Hazards documentation is present but can be expanded.
*   `docs/regression/parametric.rst` provides a good overview of `ExponentialPH` and `WeibullPH` but doesn't fully detail how the baseline hazard is parameterized or how covariates are handled under the hood.
*   The Semi-Parametric Cox PH model (`surpyval.regression.cox_ph.CoxPH_`) is documented but lacks clear examples in `docs/regression/cox_ph.rst` of how to handle left-truncation, right-censoring, and tie-breaking methods (Breslow vs. Efron).
*   **Recommendation:** Add comprehensive code examples demonstrating `CoxPH.fit()` and `CoxPH.fit_from_df()`, illustrating tie-handling options.

### Counting Processes (Recurrence/Renewal Models)
The Counting Processes models (HPP, NHPP, Proportional Intensity) have robust docstrings in the source code (e.g., `ProportionalIntensityNHPP.fit()`), but this isn't fully bubbled up to the Sphinx documentation in a user-friendly tutorial format.
*   **Recommendation:** Create a dedicated tutorial in the docs (e.g., `docs/Recurrent Event Regression Modelling with SurPyval.rst`) that leverages the `rossi_static` dataset examples currently hidden in the docstrings. Show how to interpret the `beta` coefficients and the baseline rate parameters.
*   Ensure the difference between `ProportionalIntensityHPP` and `ProportionalIntensityNHPP` is explicitly documented with mathematical formulations, just like the PH models.

## Conclusion
SurPyval has an incredibly rich set of features, particularly the dynamic generation of `AL`/`AFT` parameter substitution fitters and counting process regressions. Standardizing the terminology, consolidating regression modules, and expanding the documentation to showcase these advanced capabilities will significantly lower the barrier to entry for new users.
