# Reference results for ParametricCompetingRisks with left- and
# interval-censored rows: the joint maximum-likelihood fit of two Weibull
# causes, written out directly in base R (no R package fits a parametric
# competing-risks model to interval-censored data: flexsurv, cmprsk and
# mstate take exact times; icenReg has no competing risks).
#
# Prints the values typed into
# surpyval/tests/univariate/competing_risks/test_parametric_competing_risks_interval.py:
#
#     Rscript scripts/reference/reference_r_parametric_competing_risks_interval.R
#
# Needs numDeriv. Run with R 4.6.2 and numDeriv 2016.8.1.1.
#
# The data are those of reference_r_parametric_competing_risks.R, with
# every other unit inspected every 2 time units instead of watched: a
# failure of such a unit is known only to lie in (2j, 2j + 2] (left
# censored at 2 when j = 0); the other units' failures and every censored
# unit are as they were.
#
# The likelihood: an exact failure from cause k at x contributes
# f_k(x) S_j(x), a censored unit S_1(x) S_2(x), and a window (a, b] of
# cause k the integral of f_k(u) S_j(u) over it, by integrate() with a
# relative tolerance of 1e-12. It is maximised by optim (BFGS on the log
# parameters, relative tolerance 1e-15), from the causes' own fits; the
# covariance is the inverse of numDeriv's Richardson Hessian of the
# negative log-likelihood in the natural parameters (alpha, beta per
# cause). The cumulative incidences at t = 4, 8 and 15 are integrate()'s.

suppressPackageStartupMessages(library(numDeriv))

x <- c(4.74, 1.3, 8.49, 3.92, 0.21, 3.05, 3.64, 4.46, 1.57, 16.35, 0.13,
       10.95, 2.94, 4.03, 4.13, 6.08, 10.66, 4.59, 0.53, 0.95, 6.69, 4.62,
       0.88, 7.28, 2.93, 0.07, 12.74, 0.14, 4.81, 3.84, 4.74, 3.39, 19.74,
       5.85, 4.18, 5.85, 2.86, 7.08, 14.88, 15.95)
e <- c(0, 2, 1, 0, 2, 0, 1, 1, 2, 0, 0, 1, 1, 2, 1, 1, 1, 2, 0, 2, 2, 2, 1,
       1, 0, 2, 1, 2, 1, 2, 0, 0, 1, 0, 0, 2, 2, 1, 1, 1)
inspected <- (seq_along(x) %% 2 == 1) & (e != 0)
hi <- 2 * ceiling(x / 2)
lo <- hi - 2

cat("versions: R", as.character(getRversion()),
    "numDeriv", as.character(packageVersion("numDeriv")), "\n")

S <- function(u, p) pweibull(u, shape = p[2], scale = p[1], lower.tail = FALSE)
f <- function(u, p) dweibull(u, shape = p[2], scale = p[1])

window <- function(a, b, pk, pj) {
    integrate(function(u) f(u, pk) * S(u, pj), a, b,
              rel.tol = 1e-12, abs.tol = 0, subdivisions = 2000L,
              stop.on.error = FALSE)$value
}

nll <- function(theta) {
    if (any(!is.finite(theta)) || any(theta <= 0) || any(theta > 1e4)) {
        return(1e10)
    }
    p <- list(theta[1:2], theta[3:4])
    ll <- 0
    for (i in seq_along(x)) {
        if (e[i] == 0) {
            ll <- ll + log(S(x[i], p[[1]])) + log(S(x[i], p[[2]]))
        } else {
            k <- e[i]
            j <- 3 - k
            if (inspected[i]) {
                ll <- ll + log(window(lo[i], hi[i], p[[k]], p[[j]]))
            } else {
                ll <- ll + log(f(x[i], p[[k]])) + log(S(x[i], p[[j]]))
            }
        }
    }
    if (!is.finite(ll)) 1e10 else -ll
}

start <- c(10, 1.5, 10, 1.5)
fit <- optim(log(start), function(v) nll(exp(v)), method = "BFGS",
             control = list(reltol = 1e-15, maxit = 1000))
fit <- optim(fit$par, function(v) nll(exp(v)), method = "Nelder-Mead",
             control = list(reltol = 1e-15, maxit = 5000))
fit <- optim(fit$par, function(v) nll(exp(v)), method = "BFGS",
             control = list(reltol = 1e-15, maxit = 1000))
theta <- exp(fit$par)
H <- hessian(nll, theta)
V <- solve(H)

cat("convergence:", fit$convergence, "\n")
cat("params (1: alpha, beta, 2: alpha, beta):\n")
cat(sprintf("%.10g", theta), sep = ", ")
cat("\nneg_ll:", sprintf("%.12g", fit$value), "\n")
cat("covariance (rows):\n")
for (r in 1:4) {
    cat(sprintf("%.8g", V[r, ]), sep = ", ")
    cat("\n")
}
cif <- function(t, k) {
    p <- list(theta[1:2], theta[3:4])
    window(0, t, p[[k]], p[[3 - k]])
}
for (k in 1:2) {
    cat("cif cause", k, "at 4, 8, 15:",
        sprintf("%.10g", sapply(c(4, 8, 15), cif, k = k)), "\n")
}
