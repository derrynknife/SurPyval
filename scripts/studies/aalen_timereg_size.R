# The size of timereg's test for time-invariant effects in Aalen's
# additive model, on the design of
# surpyval/tests/calibration/test_diagnose_aalen.py's null (constant
# effects, every hazard positive): h(t | z) = 0.1 + 0.08 z1 + 0.1 z2, z1
# binary, z2 uniform on (0, 1), follow-up ending uniformly on (2, 14) and
# at 12. 200 data sets at n = 100 and at n = 300 (seeds 4500 and 4700,
# fixed before the run), aalen(..., robust = 1, n.sim = 500), rejection at
# the 5% level.
#
#     R_LIBS_USER=~/Library/R-lib Rscript scripts/studies/aalen_timereg_size.R
#
# About a minute. Result with timereg 2.0.7 (R 4.6.2): at n = 100 the KS
# test rejected z1 0.470 and z2 0.335 of the time (CvM 0.400, 0.300); at
# n = 300, 0.170 and 0.145 (CvM 0.165, 0.135). SurPyval's
# constant_effects_test, which resamples the events rather than the
# martingale residuals, rejected 0.035 and 0.030 at n = 300.

suppressPackageStartupMessages({library(survival); library(timereg)})
sim <- function(n) {
  z1 <- rbinom(n, 1, 0.5); z2 <- runif(n)
  early <- 0.1 + 0.08 * z1 + 0.1 * z2; late <- early
  e <- rexp(n)
  t <- ifelse(e < 4 * early, e / early, 4 + (e - 4 * early) / late)
  end <- pmin(runif(n, 2, 14), 12)
  data.frame(x = pmin(t, end), status = as.numeric(t <= end), z1 = z1, z2 = z2)
}
for (n in c(100, 300)) {
  set.seed(4400 + n)
  rej <- matrix(0, 2, 2); reps <- 200
  for (r in 1:reps) {
    d <- sim(n)
    fit <- aalen(Surv(x, status) ~ z1 + z2, data = d, robust = 1, n.sim = 500)
    rej[, 1] <- rej[, 1] + (fit$pval.testBeqC[2:3] < 0.05)
    rej[, 2] <- rej[, 2] + (fit$pval.testBeqC.is[2:3] < 0.05)
  }
  cat("n =", n, " rejection rates (rows z1, z2; cols KS, CvM):\n"); print(rej / reps)
}
