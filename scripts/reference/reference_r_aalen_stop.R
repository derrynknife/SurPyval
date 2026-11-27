# Reference values for Aalen's additive hazards model where the fit
# stops short of the data (surpyval.AalenAdditive.fit and its
# stopped_at), printed for
# surpyval/tests/univariate/regression/test_aalen_additive.py.
#
#     R_LIBS_USER=~/Library/R-lib Rscript scripts/reference/reference_r_aalen_stop.R
#
# Written with R 4.6.2 and timereg 2.0.7 (the versions are printed).
#
# Every unit with a = 1 has left by t = 2.2, while the a = 0 units fail
# on to t = 5.1. At the event time 2.4 the ten units at risk all have
# a = 0, so X'YX is singular and the increments cannot be estimated
# there or after. timereg::aalen() estimates B to the last event time
# before that (1.9) and then holds it at that value to the last event
# time (cum[, 1] runs to 5.1). SurPyval agrees with it up to 1.9 and,
# rather than holding B, stops there: stopped_at = 1.9, and every
# prediction after it is nan. The times have no ties.
suppressPackageStartupMessages({
  library(survival)
  library(timereg)
})
cat("R", R.version$major, ".", R.version$minor, " timereg ",
    as.character(packageVersion("timereg")), "\n", sep = "")

x <- c(0.4, 0.9, 1.3, 1.8, 1.1, 1.6, 2.0, 2.2,
       0.5, 0.7, 1.0, 1.4, 1.9, 2.4, 2.9, 3.3, 3.8, 4.6, 5.1,
       1.2, 2.7, 4.0, 5.5, 6.0)
d <- c(1, 1, 1, 1, 0, 0, 0, 0,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       0, 0, 0, 0, 0)
a <- c(rep(1, 8), rep(0, 16))
fit <- aalen(Surv(x, d) ~ a, data = data.frame(x, d, a), n.sim = 0,
             robust = 0)
cat("time, cum (baseline, a), var.cum (baseline, a)\n")
write.table(format(cbind(fit$cum, fit$var.cum[, -1]), digits = 15),
            quote = FALSE, row.names = FALSE, col.names = FALSE)
