#!/bin/sh
# Regenerate every stored reference result for surpyval/tests/reference
# (#379): the shared fixtures, the R results and the Python results.
#
#     scripts/reference/regenerate.sh [python-with-lifelines]
#
# The optional argument is a Python interpreter that has lifelines and
# scikit-survival (see reference_python.py for why lifelines lives in its
# own virtual environment); it defaults to "python". Each step rewrites its
# files in surpyval/tests/reference/data and reproduces them byte for byte
# when nothing has changed, so `git diff` shows exactly what moved.
set -e
cd "$(dirname "$0")/../.."
python scripts/reference/make_fixtures.py
Rscript scripts/reference/reference_r.R
Rscript scripts/reference/reference_r_frailty.R
Rscript scripts/reference/reference_r_competing_diagnostics.R
Rscript scripts/reference/reference_r_odds_additive.R
Rscript scripts/reference/reference_r_aalen_diagnostics.R
Rscript scripts/reference/reference_r_ph_aft.R
Rscript scripts/reference/reference_r_ph_aft_tvc.R
# flexsurv and gamlss (in R_LIBS_USER) for diagnose() of distributional fits
Rscript scripts/reference/reference_r_distributional_diagnostics.R
Rscript scripts/reference/reference_r_calibration.R
Rscript scripts/reference/reference_r_frailty_diag.R
Rscript scripts/reference/reference_r_stratified_cox.R
Rscript scripts/reference/reference_r_cox_tt.R
Rscript scripts/reference/reference_r_rmst.R
Rscript scripts/reference/reference_r_custom_recipes.R
Rscript scripts/reference/reference_r_concordance.R
Rscript scripts/reference/reference_r_metric_inference.R
Rscript scripts/reference/reference_r_competing_metrics.R
Rscript scripts/reference/reference_r_pseudo.R
Rscript scripts/reference/reference_r_cox_curve.R
Rscript scripts/reference/reference_r_competing_entry.R
# cmprsk (in R_LIBS_USER) for the Fine-Gray model's two covariances
Rscript scripts/reference/reference_r_fine_gray_variance.R
# survival, cmprsk and survRM2 (in R_LIBS_USER) for the restricted mean time lost
Rscript scripts/reference/reference_r_rmtl.R
# flexsurv (in R_LIBS_USER) for the parametric competing-risks regression
Rscript scripts/reference/reference_r_parametric_competing_risks_regression.R
Rscript scripts/reference/reference_r_reda.R
Rscript scripts/reference/reference_r_likelihood_contour.R
Rscript scripts/reference/reference_r_weighted_competing_regression.R
Rscript scripts/reference/reference_r_weighted_aalen.R  # prints the values typed into test_weighted_aalen.py
# Prints the values test_weighted_distributional_rp.py holds
Rscript scripts/reference/reference_r_weighted_distributional_rp.R
Rscript scripts/reference/reference_r_weighted_aalen_johansen.R
# Prints the delayed-entry references typed into
# surpyval/tests/reference/test_aalen_johansen_entry.py
Rscript scripts/reference/reference_r_aalen_johansen_entry.R
# Prints the values test_life_at.py holds
Rscript scripts/reference/reference_r_regression_at.R
# Prints its values, which are typed into test_turnbull_weighted.py
Rscript scripts/reference/reference_r_weighted_turnbull.R
# Prints the weighted penalised Cox references typed into
# surpyval/tests/univariate/regression/test_penalised_weighted.py
Rscript scripts/reference/reference_r_weighted_penalised.R surpyval/datasets
# Prints the bounds on a probability typed into
# surpyval/tests/univariate/parametric/test_probability_uncertainty.py
Rscript scripts/reference/reference_r_fixed_probability.R
# Prints the survreg references typed into
# surpyval/tests/univariate/regression/test_general_linear.py
Rscript scripts/reference/reference_r_general_linear.R
# Prints the flexsurv references typed into
# surpyval/tests/univariate/parametric/test_royston_parmar_with_params.py
Rscript scripts/reference/reference_r_royston_parmar_given.R
# Prints the flexsurv references typed into
# surpyval/tests/univariate/competing_risks/test_parametric_competing_risks_uncertainty.py
Rscript scripts/reference/reference_r_parametric_competing_risks.R
# Prints the values test_parametric_competing_risks_interval.py holds
Rscript scripts/reference/reference_r_parametric_competing_risks_interval.R
# The semi-parametric recurrent-event regressions (bladder2)
Rscript scripts/reference/reference_r_recurrent_cox.R
# JM (in R_LIBS_USER) for the joint model's links; prints the values typed
# into surpyval/tests/reference/test_joint_links.py (about ten minutes)
Rscript scripts/reference/reference_r_joint_links.R
# mstate (in R_LIBS_USER) for the multi-state models
Rscript scripts/reference/reference_r_multistate.R
# msm (in R_LIBS_USER) for the panel-data multi-state models
Rscript scripts/reference/reference_r_multistate_panel.R
# gsDesign and rpact (in R_LIBS_USER) for the group-sequential designs
Rscript scripts/reference/reference_r_group_sequential.R
"${1:-python}" scripts/reference/reference_python.py
