# Reference values from R for SurPyval's natural cubic spline basis
# (surpyval/utils/splines.py).
#
#     Rscript scripts/reference/reference_r_natural_spline.R
#
# Prints R's splines::ns basis (a B-spline form of the natural cubic
# spline, with intercept) at fixed points, for the knots below, as Python
# literals pasted into surpyval/tests/utils/test_splines.py. ns() spans the
# same curves as both of SurPyval's forms, so each of its columns is an
# exact linear combination of SurPyval's columns: the test checks that.
# Needs only base R (splines ships with it).
library(splines)
x <- c(-2, 0, 0.7, 1.5, 2.2, 3.1, 4.4, 5.9, 7.3, 8, 9.5)
knots <- c(1.5, 4.4, 6)
boundary <- c(0, 8)
B <- ns(x, knots = knots, Boundary.knots = boundary, intercept = TRUE)
cat("# R", R.version$major, ".", R.version$minor, ", splines::ns\n", sep = "")
cat("R_NS_X = [", paste(sprintf("%.17g", x), collapse = ", "), "]\n")
cat("R_NS = [\n")
for (i in seq_len(nrow(B))) {
  cat("    [", paste(sprintf("%.17g", B[i, ]), collapse = ", "), "],\n")
}
cat("]\n")
