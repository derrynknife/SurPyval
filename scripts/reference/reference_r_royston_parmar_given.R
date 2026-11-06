# Reference values for a Royston-Parmar model built from given spline
# coefficients and knots (RoystonParmarModel.with_params), printed for
# surpyval/tests/univariate/parametric/test_royston_parmar_with_params.py.
#
#     Rscript scripts/reference/reference_r_royston_parmar_given.R
#
# Written with R 4.6.2 and flexsurv 2.3.2 (the versions are printed).
#
# flexsurv's psurvspline, Hsurvspline, hsurvspline, dsurvspline and
# qsurvspline at fixed gamma and knots (log time, boundary knots
# included; timescale = "log") are the same restricted cubic spline
# (Royston and Parmar 2002) on the same three link scales: "hazard"
# log H(t), "odds" log((1 - S) / S) and "normal" -qnorm(S(t)). The
# times run from below the first knot to beyond the last, where the
# spline is linear.
suppressPackageStartupMessages(library(flexsurv))
cat("R ", R.version$major, ".", R.version$minor, " flexsurv ",
    as.character(packageVersion("flexsurv")), "\n", sep = "")

knots <- log(c(2, 8, 20, 60))
gamma <- c(-4, 1.3, 0.05, -0.02)
t <- c(0.5, 2, 5, 13, 30, 60, 150)
p <- c(0.01, 0.1, 0.5, 0.9, 0.99)
fmt <- function(v) paste(sprintf("%.17g", v), collapse = ", ")
for (scale in c("hazard", "odds", "normal")) {
  cat(scale, "\n")
  cat("  sf:", fmt(psurvspline(t, gamma, knots = knots, scale = scale,
                                lower.tail = FALSE)), "\n")
  cat("  Hf:", fmt(Hsurvspline(t, gamma, knots = knots, scale = scale)),
      "\n")
  cat("  hf:", fmt(hsurvspline(t, gamma, knots = knots, scale = scale)),
      "\n")
  cat("  df:", fmt(dsurvspline(t, gamma, knots = knots, scale = scale)),
      "\n")
  cat("  qf:", fmt(qsurvspline(p, gamma, knots = knots, scale = scale)),
      "\n")
  cat("  mean:", fmt(mean_survspline(gamma, knots = knots, scale = scale)),
      "\n")
}
