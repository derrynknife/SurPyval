# Reference values for Aalen's additive hazards model with sampling
# weights (surpyval.AalenAdditive.fit(..., w=)), printed for
# surpyval/tests/reference/test_weighted_aalen.py.
#
#     Rscript scripts/reference/reference_r_weighted_aalen.R
#
# Written with R 4.6.2 and timereg 2.0.7 (the versions are printed).
#
# timereg::aalen(..., weights = w, robust = 1) is SurPyval's weighted
# estimator: at each event time the increments are the weighted least
# squares dB = (X'WYX)^-1 X'W dN, and robvar.cum is the robust variance,
# the sum over the units of their squared influence (martingale
# residuals mapped as the increments are). The data are
# surpyval.tests._helpers.weighted_reference_data(), typed out below.
# timereg breaks tied times at random, so the ties are broken here first,
# by adding 0.001 * (rank within the tie - 1) after sorting by time, then
# events first, then row; the test repeats it. aalen() does not look for
# the weights in `data` (a bare `weights = w` is the global w, in the
# unsorted order), so they are passed as df$w.
suppressPackageStartupMessages({
  library(survival)
  library(timereg)
})
cat("R", R.version$major, ".", R.version$minor, " timereg ",
    as.character(packageVersion("timereg")), "\n", sep = "")

x <- c(2.9, 5.8, 4.3, 3.7, 8.3, 2.7, 4.3, 2.2, 1.0, 3.8, 8.8, 8.7, 6.9,
       2.0, 3.2, 3.2, 1.7, 6.3, 6.6, 14.5, 2.1, 2.3, 4.3, 8.3, 0.8, 7.8,
       2.9, 6.1, 7.3, 9.6, 0.1, 3.6, 3.7, 6.1, 1.0, 1.6, 3.9, 3.2, 1.1,
       8.3)
d <- c(1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
       1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0)
w <- c(1.82, 1.39, 2.27, 1.96, 0.34, 2.47, 1.59, 0.58, 1.79, 0.45, 1.18,
       1.56, 2.73, 2.24, 2.29, 1.96, 1.34, 2.88, 0.77, 1.23, 1.89, 2.03,
       1.67, 2.51, 0.99, 1.73, 2.96, 0.36, 1.33, 0.87, 1.32, 1.61, 1.57,
       1.36, 0.81, 2.89, 2.25, 1.37, 1.01, 0.53)
z1 <- c(0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0)
z2 <- c(1.3, 0.36, -1.35, -1.95, -0.22, 0.76, -0.02, 0.05, 0.02, 0.74,
        -0.47, -0.32, 0.54, 1.19, 0.19, 0.07, 1.69, -0.24, -1.33, -1.1,
        0.2, 0.89, 0.76, -1.8, 0.38, -0.98, -0.31, -1.37, -1.09, -1.58,
        -0.21, 0.09, 1.69, 0.56, -0.05, 0.71, 0.38, -0.55, 1.21, -0.69)
tl <- c(0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.9, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.6, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1,
        0.4)
df <- data.frame(x, d, w, z1, z2, tl, row = seq_along(x))
df <- df[order(df$x, -df$d, df$row), ]
df$x <- df$x + 0.001 * (ave(df$x, df$x, FUN = seq_along) - 1)

times <- c(0.5, 1.5, 3, 5, 7)
show <- function(label, fit) {
  # B(t) and its robust variance at the times, a row per time: baseline,
  # z1, z2.
  at <- sapply(times, function(t) max(which(fit$cum[, 1] <= t)))
  cat(label, "\n")
  cat("  cum\n")
  write.table(format(fit$cum[at, -1], digits = 15), quote = FALSE,
              row.names = FALSE, col.names = FALSE)
  cat("  robvar.cum\n")
  write.table(format(fit$robvar.cum[at, -1], digits = 15), quote = FALSE,
              row.names = FALSE, col.names = FALSE)
}
show("weighted", aalen(Surv(x, d) ~ z1 + z2, data = df, weights = df$w,
                       robust = 1, n.sim = 0))
show("weighted_tl", aalen(Surv(tl, x, d) ~ z1 + z2, data = df, weights = df$w,
                          robust = 1, n.sim = 0))
