# Reference rectangle probabilities of three coordinates of the
# multivariate normal and t distributions from R's mvtnorm, for
# copula/_rectangle.py (a row with three censored coordinates).
#
# Writes surpyval/tests/reference/data/r_mvtnorm_rectangle3.json, in the
# layout of r_mvtnorm_rectangle.json. Generated with R 4.6.2 and mvtnorm
# 1.5.0; run from the repository root:
#
#     Rscript scripts/reference/reference_r_mvtnorm_rectangle3.R
#
# A case is a correlation matrix and limits mixing right-, left- and
# interval-censored coordinates, as a censored row's are. The matrices are
# random (a Wishart draw rescaled), and also nearly singular: a random
# matrix whose smallest eigenvalue is set to 7e-4, 1e-3, 1e-2 or 0.05
# before it is rescaled to unit diagonal. The normal uses Miwa's algorithm
# (deterministic; 4096 grid points), checked against Genz-Bretz at an
# absolute tolerance of 1e-12: the two agree to 1e-7 or the script stops,
# and the recorded error is the larger of the difference and Genz-Bretz's
# estimate. The t (3 and 8 degrees of freedom) uses Genz-Bretz at an
# absolute tolerance of 1e-9.

options(warn = -1)  # (Miwa: "Approximating +/-Inf by +/-1000")
suppressMessages({
  library(mvtnorm)
  library(jsonlite)
})

set.seed(20261126)
random_corr <- function(k, smallest = NA) {
  a <- matrix(rnorm(k * k), k, k)
  s <- a %*% t(a)
  d <- sqrt(diag(s))
  s <- s / outer(d, d)
  if (!is.na(smallest)) {
    # Set the smallest eigenvalue, rescale, and repeat until the rescaled
    # matrix's smallest eigenvalue is the one asked for
    for (it in 1:50) {
      e <- eigen(s, symmetric = TRUE)
      lam <- e$values
      lam[length(lam)] <- smallest * lam[length(lam)] /
        min(eigen(s, symmetric = TRUE, only.values = TRUE)$values)
      s <- e$vectors %*% diag(lam) %*% t(e$vectors)
      d <- sqrt(diag(s))
      s <- s / outer(d, d)
    }
  }
  (s + t(s)) / 2
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
add <- function(R, lim, nu, smallest) {
  if (is.null(nu)) {
    genz <- pmvnorm(lim$lo, lim$hi, corr = R,
                    algorithm = GenzBretz(maxpts = 5e7, abseps = 1e-12,
                                          releps = 0))
    miwa <- pmvnorm(lim$lo, lim$hi, corr = R,
                    algorithm = Miwa(steps = 4096))
    if (abs(miwa - genz) > 1e-7) {
      stop(sprintf("Miwa and Genz-Bretz disagree: %g", miwa - genz))
    }
    value <- miwa
    error <- max(attr(genz, "error"), abs(miwa - genz))
    algorithm <- "Miwa(steps = 4096)"
  } else {
    v <- pmvt(lim$lo, lim$hi, corr = R, df = nu,
              algorithm = GenzBretz(maxpts = 5e7, abseps = 1e-9,
                                    releps = 0))
    value <- v
    error <- attr(v, "error")
    algorithm <- "GenzBretz(maxpts = 5e7, abseps = 1e-9)"
  }
  cases[[length(cases) + 1]] <<- list(
    k = 3, nu = nu, smallest_eigenvalue = min(eigen(R)$values),
    corr = R, lo = inf_null(lim$lo), hi = inf_null(lim$hi),
    value = as.numeric(value), error = error, algorithm = algorithm
  )
}
for (nu in list(NULL, 3, 8)) {
  for (smallest in c(NA, NA, NA, NA, 7e-4, 1e-3, 1e-2, 0.05)) {
    for (i in 1:2) {
      add(random_corr(3, smallest), limits(3), nu, smallest)
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
  "surpyval/tests/reference/data/r_mvtnorm_rectangle3.json"
)
