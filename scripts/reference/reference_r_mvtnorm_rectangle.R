# Reference rectangle probabilities of the multivariate normal and t
# distributions from R's mvtnorm, for the lattice rule of four or more
# coordinates in copula/_rectangle.py.
#
# Writes surpyval/tests/reference/data/r_mvtnorm_rectangle.json: for each
# case the correlation matrix, the limits (null for an infinite one), the
# degrees of freedom (null for the normal) and mvtnorm's probability with
# its error estimate. Generated with R 4.6.2 and mvtnorm 1.5.0; run from the
# repository root:
#
#     Rscript scripts/reference/reference_r_mvtnorm_rectangle.R
#
# The normal cases use Miwa's algorithm (deterministic; 4096 grid points),
# checked against Genz-Bretz at an absolute tolerance of 1e-10: the two
# agree to 1e-7 or the script stops (they agree to 4e-8 or better), and
# the recorded error is the larger of the difference and Genz-Bretz's
# estimate. The t cases use Genz-Bretz at an absolute tolerance of 1e-7
# (Miwa's algorithm is the normal's only).
# A case is a random correlation matrix (a Wishart draw rescaled) and
# limits mixing right-, left- and interval-censored coordinates, as a
# censored row's are.

options(warn = -1)  # (Miwa: "Approximating +/-Inf by +/-1000")
suppressMessages({
  library(mvtnorm)
  library(jsonlite)
})

set.seed(20261125)
random_corr <- function(k) {
  a <- matrix(rnorm(k * k), k, k)
  s <- a %*% t(a)
  d <- sqrt(diag(s))
  s / outer(d, d)
}
limits <- function(k) {
  kind <- sample(c("right", "left", "interval"), k, replace = TRUE,
                 prob = c(0.5, 0.3, 0.2))
  at <- rnorm(k, 0, 1.2)
  lo <- ifelse(kind == "left", -Inf, at)
  hi <- ifelse(kind == "right", Inf,
               ifelse(kind == "left", at, at + runif(k, 0.3, 2)))
  list(lo = lo, hi = hi)
}
inf_null <- function(v) lapply(v, function(x) if (is.finite(x)) x else NULL)

cases <- list()
for (k in c(4, 5, 6, 8)) {
  for (i in 1:4) {
    R <- random_corr(k)
    lim <- limits(k)
    miwa <- pmvnorm(lim$lo, lim$hi, corr = R,
                    algorithm = Miwa(steps = 4096))
    genz <- pmvnorm(lim$lo, lim$hi, corr = R,
                    algorithm = GenzBretz(maxpts = 5e7, abseps = 1e-10,
                                          releps = 0))
    if (abs(miwa - genz) > 1e-7) {
      stop(sprintf("Miwa and Genz-Bretz disagree (k = %d): %g", k,
                   miwa - genz))
    }
    cases[[length(cases) + 1]] <- list(
      k = k, nu = NULL, corr = R, lo = inf_null(lim$lo),
      hi = inf_null(lim$hi), value = as.numeric(miwa),
      error = max(attr(genz, "error"), abs(miwa - genz)),
      algorithm = "Miwa(steps = 4096)"
    )
  }
}
for (nu in c(3, 8)) {
  for (k in c(4, 6)) {
    for (i in 1:3) {
      R <- random_corr(k)
      lim <- limits(k)
      value <- pmvt(lim$lo, lim$hi, corr = R, df = nu,
                    algorithm = GenzBretz(maxpts = 1e7, abseps = 1e-7,
                                          releps = 0))
      cases[[length(cases) + 1]] <- list(
        k = k, nu = nu, corr = R, lo = inf_null(lim$lo),
        hi = inf_null(lim$hi), value = as.numeric(value),
        error = attr(value, "error"),
        algorithm = "GenzBretz(maxpts = 1e7, abseps = 1e-7)"
      )
    }
  }
}

out <- list(
  software = "R mvtnorm",
  version = as.character(packageVersion("mvtnorm")),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  call = "pmvnorm(lower, upper, corr) / pmvt(lower, upper, corr, df)",
  cases = cases
)
writeLines(
  toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", pretty = TRUE),
  "surpyval/tests/reference/data/r_mvtnorm_rectangle.json"
)
