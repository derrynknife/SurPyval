# Reference results from R flexsurv for ParametricCompetingRisks: the
# per-cause Weibull fits, their covariance, and the cumulative incidence
# and all-cause survival of the fitted model with simulated bounds.
#
# Prints the values typed into
# surpyval/tests/univariate/competing_risks/test_parametric_competing_risks_uncertainty.py
# (as reference_r_weighted_aalen.R does for its test):
#
#     Rscript scripts/reference/reference_r_parametric_competing_risks.R
#
# Needs survival, flexsurv and cmprsk. Run with R 4.6.2, survival 3.8.12,
# flexsurv 2.3.2 and cmprsk 2.2.12.
#
# What each value is:
#
# * Each cause's fit, flexsurvreg(Surv(x, e == k) ~ 1, dist = "weibull")
#   with the other cause censored (the likelihood factorises by cause),
#   with optim's relative tolerance tightened to 1e-14: the maximised
#   log-likelihood, the scale (SurPyval's alpha) and shape (beta), and
#   their covariance on the natural scale, J V J with V = vcov(fit) on
#   flexsurv's log scale and J = diag(scale, shape).
# * The cumulative incidence of each cause and the all-cause survival at
#   t = 4, 8 and 15: the transition probabilities of the competing-risks
#   multi-state model built from the two fits (fmsm, pmatrix.fs), solved
#   by deSolve's ODE solver with its tolerance tightened to 1e-12.
# * Their 95% intervals from pmatrix.fs(ci = TRUE, B = 20000): flexsurv
#   simulates parameters from the normal on the log scale and takes the
#   percentiles of the probabilities. That is not the delta method, so
#   the test compares them loosely.
# * cmprsk::cuminc's non-parametric (Aalen-Johansen) estimate, printed as
#   a sanity check only.

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(cmprsk)
})

x <- c(4.74, 1.3, 8.49, 3.92, 0.21, 3.05, 3.64, 4.46, 1.57, 16.35, 0.13,
       10.95, 2.94, 4.03, 4.13, 6.08, 10.66, 4.59, 0.53, 0.95, 6.69, 4.62,
       0.88, 7.28, 2.93, 0.07, 12.74, 0.14, 4.81, 3.84, 4.74, 3.39, 19.74,
       5.85, 4.18, 5.85, 2.86, 7.08, 14.88, 15.95)
e <- c(0, 2, 1, 0, 2, 0, 1, 1, 2, 0, 0, 1, 1, 2, 1, 1, 1, 2, 0, 2, 2, 2, 1,
       1, 0, 2, 1, 2, 1, 2, 0, 0, 1, 0, 0, 2, 2, 1, 1, 1)
d <- data.frame(x = x, e = e)
times <- c(4, 8, 15)

cat("versions: survival", as.character(packageVersion("survival")),
    "flexsurv", as.character(packageVersion("flexsurv")),
    "cmprsk", as.character(packageVersion("cmprsk")), "\n")

fits <- list()
for (k in 1:2) {
    fit <- flexsurvreg(Surv(x, e == k) ~ 1, data = d, dist = "weibull",
                       control = list(reltol = 1e-14, maxit = 10000))
    fits[[k]] <- fit
    est <- fit$res[, "est"]
    J <- diag(c(est[["shape"]], est[["scale"]]))
    V <- J %*% vcov(fit) %*% J
    # SurPyval's order: alpha (scale), beta (shape)
    V <- V[c(2, 1), c(2, 1)]
    cat(sprintf("cause %d loglik %.10f\n", k, fit$loglik))
    cat(sprintf("cause %d alpha %.10f beta %.10f\n", k,
                est[["scale"]], est[["shape"]]))
    cat(sprintf("cause %d cov %.10e %.10e %.10e\n", k,
                V[1, 1], V[1, 2], V[2, 2]))
}

tmat <- rbind(c(NA, 1, 2), c(NA, NA, NA), c(NA, NA, NA))
msm <- fmsm(fits[[1]], fits[[2]], trans = tmat)
for (t in times) {
    p <- pmatrix.fs(msm, trans = tmat, t = t, tvar = "trans",
                    rtol = 1e-12, atol = 1e-12)
    cat(sprintf("t %g cif1 %.10f cif2 %.10f sf %.10f\n", t,
                p[1, 2], p[1, 3], p[1, 1]))
}
set.seed(20261105)
for (t in times) {
    p <- pmatrix.fs(msm, trans = tmat, t = t, ci = TRUE, B = 20000)
    lo <- attr(p, "lower")
    hi <- attr(p, "upper")
    cat(sprintf("t %g ci cif1 %.4f %.4f cif2 %.4f %.4f sf %.4f %.4f\n", t,
                lo[1, 2], hi[1, 2], lo[1, 3], hi[1, 3], lo[1, 1], hi[1, 1]))
}

aj <- cuminc(d$x, d$e, cencode = 0)
print(timepoints(aj, times))
