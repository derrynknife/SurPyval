# Reference values for the log-rank tests of data with sampling weights
# (surpyval.logrank(..., w=)), printed for
# surpyval/tests/reference/test_weighted_logrank.py.
#
#     Rscript scripts/reference/reference_r_weighted_logrank.R
#
# Written with R 4.6.2 and survival 3.8.12 (the versions are printed).
#
# The weighted log-rank test with a design-based variance is the robust
# score test of a Cox model of the group indicators at beta = 0:
# coxph(..., weights = w, robust = TRUE)$rscore. SurPyval's log-rank
# numerator takes tied events as Breslow's partial likelihood does, so the
# reference fits use ties = "breslow". The data are
# surpyval.tests._helpers.weighted_reference_data(), typed out below.
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
z2 <- c(1.3, 0.36, -1.35, -1.95, -0.22, 0.76, -0.02, 0.05, 0.02, 0.74,
        -0.47, -0.32, 0.54, 1.19, 0.19, 0.07, 1.69, -0.24, -1.33, -1.1,
        0.2, 0.89, 0.76, -1.8, 0.38, -0.98, -0.31, -1.37, -1.09, -1.58,
        -0.21, 0.09, 1.69, 0.56, -0.05, 0.71, 0.38, -0.55, 1.21, -0.69)
tl <- c(0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.9, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.6, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1,
        0.4)
g2 <- factor(z1)
g3 <- factor(findInterval(z2, c(-0.5, 0.5)))  # 0, 1, 2
s <- as.integer(z2 > 0)
id <- seq_along(x)  # one row per unit (needed with start-stop data)

show <- function(label, fit) {
  cat(sprintf("%-28s %.15g\n", label, fit$rscore))
}
show("two_groups", coxph(Surv(x, d) ~ g2, weights = w, robust = TRUE,
                         ties = "breslow"))
show("unit_weights", coxph(Surv(x, d) ~ g2, weights = rep(1, 40),
                           robust = TRUE, ties = "breslow"))
show("three_groups_tl", coxph(Surv(tl, x, d) ~ g3, weights = w, id = id,
                              robust = TRUE, ties = "breslow"))
show("strata_tl", coxph(Surv(tl, x, d) ~ g2 + strata(s), weights = w, id = id,
                        robust = TRUE, ties = "breslow"))
# t0 = 5: events after 5 censored at 5
x5 <- pmin(x, 5); d5 <- ifelse(x > 5, 0, d)
show("t0_5", coxph(Surv(x5, d5) ~ g3, weights = w, robust = TRUE,
                   ties = "breslow"))
