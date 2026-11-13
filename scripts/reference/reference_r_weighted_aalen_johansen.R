# Reference values for the Aalen-Johansen cumulative incidence of data with
# sampling weights (surpyval.CompetingRisks.fit(..., w=)), printed for
# surpyval/tests/reference/test_weighted_aalen_johansen.py.
#
#     Rscript scripts/reference/reference_r_weighted_aalen_johansen.R
#
# Written with R 4.6.2 and survival 3.8.12 (the versions are printed).
#
# survfit(Surv(x, factor(e)) ~ 1, weights = w) fits the multi-state
# (competing risks) Aalen-Johansen estimate; with weights its std.err is
# the infinitesimal-jackknife (robust) one, the same with id = and
# robust = TRUE. The data are surpyval.tests._helpers.
# weighted_reference_data(), typed out below, with the cause of each
# failure 1 where z1 = 1 and 2 otherwise (e = 0 censored). "ends" adds a
# unit failing from cause 1 at 15 (weight 1.1), after every other time, so
# that no one is left at risk: the last step has S = 0.
suppressPackageStartupMessages(library(survival))
cat("R", R.version$major, ".", R.version$minor, " survival ",
    as.character(packageVersion("survival")), "\n", sep = "")

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
e <- ifelse(d == 1, ifelse(z1 == 1, 1, 2), 0)
times <- c(1, 2.2, 3.2, 4.3, 6.1, 8.3, 9.6, 15)

show <- function(label, x, e, w) {
  fit <- survfit(Surv(x, factor(e, 0:2)) ~ 1, weights = w)
  robust <- survfit(Surv(x, factor(e, 0:2)) ~ 1, weights = w,
                    id = seq_along(x), robust = TRUE)
  stopifnot(isTRUE(all.equal(fit$std.err, robust$std.err, tolerance = 0)))
  s <- summary(fit, times = times, extend = TRUE)
  for (k in 1:2) {
    cat(sprintf("%s cif%d = [%s]\n", label, k,
                paste(sprintf("%.15g", s$pstate[, k + 1]), collapse = ", ")))
    cat(sprintf("%s se%d = [%s]\n", label, k,
                paste(sprintf("%.15g", s$std.err[, k + 1]), collapse = ", ")))
  }
}
show("weighted", x, e, w)
show("unit", x, e, rep(1, length(x)))
show("ends", c(x, 15), c(e, 1), c(w, 1.1))
