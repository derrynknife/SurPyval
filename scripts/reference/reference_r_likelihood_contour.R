# Reference values for the joint likelihood-ratio contour of a two-parameter
# fit (Parametric.likelihood_contour), stored in
# surpyval/tests/reference/data/weibullr_contour.json for
# surpyval/tests/reference/test_likelihood_contour.py.
#
#     Rscript scripts/reference/reference_r_likelihood_contour.R
#
# Needs WeibullR (1.2.4 was used; it builds with Rcpp) and jsonlite. Run
# from the repository root. Re-running reproduces the file.
#
# The data: 20 Weibull lifetimes (shape 1.8, scale 100) right censored by
# uniform(40, 200) times, drawn with set.seed(237) and rounded to 0.1;
# they are typed out below so the file does not depend on R's generators.
#
# Two references, at 90% and 95% joint regions (dof = 2, the chi-squared
# with two degrees of freedom: the cut is logL_max - qchisq(1 - a, 2) / 2):
#
# * WeibullR::MLEcontour(..., dof = 2): the points of its contour, in
#   (Eta, Beta) = SurPyval's (alpha, beta). Its points are found to its own
#   radial tolerance (RadLimit = 1e-5 by default), so they lie on the
#   contour to about that precision, not to SurPyval's.
# * The exact extremes of each parameter over the region, in base R: the
#   profile log-likelihood of one parameter (the other maximised with
#   optimize) solved with uniroot for where it falls to the cut. These are
#   what SurPyval's ``bounds`` are, and the contour's tangent points.

suppressPackageStartupMessages({
    library(WeibullR)
    library(jsonlite)
})

x <- c(61.9, 74.2, 82.2, 106.3, 76.2, 40.7, 83.2, 42.3, 50.4, 82.4, 78.1,
       79.1, 129.4, 52.0, 108.4, 44.1, 120.4, 34.3, 80.1, 48.6)
# 1 = failed, 0 = right censored (SurPyval's c is 1 - d).
d <- c(0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1)

loglik <- function(eta, beta) {
    z <- (x / eta)^beta
    sum(d * (log(beta) - log(eta) + (beta - 1) * log(x / eta)) - z)
}

fit <- optim(c(log(90), log(2)), function(p) -loglik(exp(p[1]), exp(p[2])),
             control = list(reltol = 1e-14, maxit = 10000))
fit <- optim(fit$par, function(p) -loglik(exp(p[1]), exp(p[2])),
             method = "BFGS", control = list(reltol = 1e-16))
mle <- exp(fit$par)
ll_max <- -fit$value

fr <- mleframe(x[d == 1], s = x[d == 0])
wr_fit <- mlefit(fr)

profile_eta <- function(eta) {
    optimize(function(b) loglik(eta, b), c(0.05, 50), maximum = TRUE,
             tol = 1e-12)$objective
}
profile_beta <- function(beta) {
    optimize(function(e) loglik(e, beta), c(1, 1000), maximum = TRUE,
             tol = 1e-12)$objective
}
extremes <- function(prof, hat, cut) {
    f <- function(v) prof(v) - cut
    lo <- uniroot(f, c(hat / 10, hat), tol = 1e-13)$root
    hi <- uniroot(f, c(hat, hat * 10), tol = 1e-13)$root
    c(lo, hi)
}

levels <- list()
for (alpha in c(0.1, 0.05)) {
    crit <- qchisq(1 - alpha, 2)
    cut <- ll_max - crit / 2
    ct <- MLEcontour(fr, CL = 1 - alpha, dof = 2)
    levels[[sprintf("%g", alpha)]] <- list(
        alpha_ci = alpha,
        critical_value = crit,
        log_likelihood_cut = cut,
        call = sprintf("MLEcontour(mleframe(x[d == 1], s = x[d == 0]), CL = %g, dof = 2)",
                       1 - alpha),
        points = list(alpha = ct$Eta, beta = ct$Beta),
        bounds = list(
            alpha = extremes(profile_eta, mle[1], cut),
            beta = extremes(profile_beta, mle[2], cut)
        )
    )
}

out <- list(
    generator = "scripts/reference/reference_r_likelihood_contour.R",
    software = "R WeibullR",
    version = paste0("R ", R.version$major, ".", R.version$minor,
                     ", WeibullR ", as.character(packageVersion("WeibullR"))),
    data = list(x = x, c = 1 - d),
    mle = list(alpha = mle[1], beta = mle[2], log_likelihood = ll_max),
    weibullr_mle = list(alpha = unname(wr_fit[1]), beta = unname(wr_fit[2]),
                        log_likelihood = unname(wr_fit[3])),
    levels = levels
)
dir <- file.path("surpyval", "tests", "reference", "data")
writeLines(toJSON(out, digits = I(15), auto_unbox = TRUE, pretty = TRUE),
           file.path(dir, "weibullr_contour.json"))
cat(out$version, "\n")
