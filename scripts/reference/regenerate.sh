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
"${1:-python}" scripts/reference/reference_python.py
