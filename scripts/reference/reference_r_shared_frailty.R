# Reference results from R for the shared gamma-frailty Poisson process
# (surpyval.recurrent, fit(..., frailty = "gamma")), for
# surpyval/tests/reference/test_shared_frailty.py.
#
# Run from anywhere:
#
#     Rscript scripts/reference/reference_r_shared_frailty.R
#
# Written with R 4.6.2 and MASS 7.3.66. It prints the values the test holds.
#
# Units all watched from 0 to one common time T under a power-law
# intensity with a gamma frailty have a likelihood that splits in two:
# the counts are negative binomial with one common mean (the cumulative
# intensity at T) and size 1 / theta, and, given the counts, the event
# times carry the shape alone. So the frailty variance and the mean count
# of the SurPyval fit are those of a negative binomial fit to the counts,
# MASS::glm.nb(n ~ 1): theta_surpyval = 1 / theta_MASS and the cumulative
# intensity at T is exp(intercept). The counts are the test's fixture.
suppressMessages(library(MASS))
n <- c(0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9,
       10, 11, 12, 14, 16, 18, 21, 25)
fit <- glm.nb(n ~ 1, control = glm.control(epsilon = 1e-14, maxit = 200))
options(digits = 15)
cat("R", as.character(getRversion()), "MASS",
    as.character(packageVersion("MASS")), "\n")
cat("mean count          ", exp(coef(fit)[[1]]), "\n")
cat("1 / theta (MASS)    ", fit$theta, "\n")
cat("frailty variance    ", 1 / fit$theta, "\n")
cat("log-likelihood      ", as.numeric(logLik(fit)), "\n")
