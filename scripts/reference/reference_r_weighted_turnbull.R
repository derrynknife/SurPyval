# Reference values for the Turnbull estimate of interval censored data with
# sampling weights (surpyval.Turnbull.fit(..., w=)), printed for
# surpyval/tests/univariate/nonparametric/test_turnbull_weighted.py.
#
#     Rscript scripts/reference/reference_r_weighted_turnbull.R
#
# Written with R 4.6.2 and icenReg 2.0.16 (the versions are printed).
#
# icenReg::ic_np is the EM-ICM non-parametric MLE and takes case weights:
# each row's log-likelihood term is multiplied by its weight, the
# weighted NPMLE. B = c(0, 1) makes every interval (l, u], SurPyval's
# convention; u = Inf is a right-censored row. The data are typed out
# below (and in the test); they were drawn once with numpy
# (default_rng(20261105)): Weibull(1.4) times of scale 10, each widened to
# an inspection interval, a fifth right censored, weights uniform on
# (0.3, 3) rounded to 0.01.
suppressPackageStartupMessages(library(icenReg))
cat("R", R.version$major, ".", R.version$minor, " icenReg ",
    as.character(packageVersion("icenReg")), "\n", sep = "")

l <- c(6, 3, 6, 2, 6, 0, 0, 1, 4, 17, 10, 10, 0, 11, 2, 1, 7, 2, 7, 7, 14,
       18, 0, 6, 22, 0, 12, 0, 1, 2)
u <- c(8, Inf, 11, 6, 10, 6, 3, 7, 8, Inf, 12, 14, 5, 18, 4, Inf, 13, 8, 12,
       12, 20, Inf, Inf, 10, Inf, Inf, 15, 8, 8, 6)
w <- c(2.25, 1.03, 2.19, 2.25, 0.71, 0.49, 0.83, 2.86, 0.98, 1.52, 1.88,
       0.65, 1.61, 2.66, 2.74, 2.81, 2.69, 2.75, 2.02, 2.18, 1.11, 2.4, 2.9,
       1.9, 1.69, 1.42, 2.65, 0.44, 1.65, 2.38)
d <- data.frame(l = l, u = u)

show <- function(label, fit) {
  ints <- fit$T_bull_Intervals
  cat(label, "\n")
  cat("  lower:", format(ints[1, ], digits = 17), "\n")
  cat("  upper:", format(ints[2, ], digits = 17), "\n")
  cat("  p:", format(fit$p_hat, digits = 17), "\n")
}

show("weighted", ic_np(cbind(l, u) ~ 0, data = d, B = c(0, 1), weights = w,
                       tol = 1e-12, maxIter = 100000))
show("unweighted", ic_np(cbind(l, u) ~ 0, data = d, B = c(0, 1),
                         tol = 1e-12, maxIter = 100000))
