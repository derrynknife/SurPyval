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
Rscript scripts/reference/reference_r_ph_aft.R
Rscript scripts/reference/reference_r_calibration.R
Rscript scripts/reference/reference_r_frailty_diag.R
Rscript scripts/reference/reference_r_stratified_cox.R
Rscript scripts/reference/reference_r_cox_tt.R
Rscript scripts/reference/reference_r_rmst.R
Rscript scripts/reference/reference_r_concordance.R
Rscript scripts/reference/reference_r_metric_inference.R
Rscript scripts/reference/reference_r_competing_metrics.R
Rscript scripts/reference/reference_r_pseudo.R
Rscript scripts/reference/reference_r_cox_curve.R
Rscript scripts/reference/reference_r_reda.R
Rscript scripts/reference/reference_r_likelihood_contour.R
Rscript scripts/reference/reference_r_weighted_competing_regression.R
Rscript scripts/reference/reference_r_weighted_aalen.R  # prints the values typed into test_weighted_aalen.py
# Prints the values test_weighted_distributional_rp.py holds
Rscript scripts/reference/reference_r_weighted_distributional_rp.R
Rscript scripts/reference/reference_r_weighted_aalen_johansen.R
# Prints its values, which are typed into test_turnbull_weighted.py
Rscript scripts/reference/reference_r_weighted_turnbull.R
# Prints the weighted penalised Cox references typed into
# surpyval/tests/univariate/regression/test_penalised_weighted.py
Rscript scripts/reference/reference_r_weighted_penalised.R surpyval/datasets
# Prints the survreg references typed into
# surpyval/tests/univariate/regression/test_general_linear.py
Rscript scripts/reference/reference_r_general_linear.R
"${1:-python}" scripts/reference/reference_python.py
