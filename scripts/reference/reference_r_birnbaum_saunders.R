# Reference results from R for surpyval.BirnbaumSaunders (shape alpha,
# scale beta, the median).
#
# Writes surpyval/tests/reference/data/r_birnbaum_saunders.json, which
# surpyval/tests/reference/test_birnbaum_saunders.py::test_against_r_vgam
# reads when it is present (the test is skipped until then):
#
# * functions: VGAM's dbisa, pbisa and qbisa (scale = beta, shape = alpha)
#   at fixed parameters and times, with bssn's pbssn at skewness 0 (the
#   Birnbaum-Saunders itself) as a second opinion where bssn is installed;
# * fits: the maximum likelihood fit to survival's lung data (right
#   censored), by optim on a likelihood written with VGAM's dbisa and
#   pbisa, standard errors from the inverse of optim's Hessian.
#
#     Rscript scripts/reference/reference_r_birnbaum_saunders.R
#
# Needs VGAM, survival and jsonlite (bssn optional). NOT RUN YET: written
# against VGAM's documentation of dbisa / pbisa / qbisa; check on the first
# run that pbisa takes lower.tail and log.p (if it does not, use
# log(1 - pbisa(...)) away from the far tail, where the lung data are).

suppressPackageStartupMessages({
    library(VGAM)
    library(survival)
    library(jsonlite)
})

params <- list(c(0.05, 1), c(0.5, 10), c(1, 3), c(2.5, 200), c(8, 0.01))
t <- c(0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000)
p <- c(1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6)

out_f <- list(t = t, p = p, params = params, pdf = list(), cdf = list(),
              sf = list(), quantile = list())
have_bssn <- requireNamespace("bssn", quietly = TRUE)
if (have_bssn) out_f$cdf_bssn <- list()
for (k in seq_along(params)) {
    a <- params[[k]][1]
    b <- params[[k]][2]
    out_f$pdf[[k]] <- dbisa(t, scale = b, shape = a)
    out_f$cdf[[k]] <- pbisa(t, scale = b, shape = a)
    out_f$sf[[k]] <- pbisa(t, scale = b, shape = a, lower.tail = FALSE)
    out_f$quantile[[k]] <- qbisa(p, scale = b, shape = a)
    if (have_bssn) {
        out_f$cdf_bssn[[k]] <- bssn::pbssn(t, alpha = a, beta = b,
                                           lambda = 0)
    }
}

# lung: status 2 = died (surpyval c = 0), 1 = censored (c = 1)
x <- lung$time
cc <- ifelse(lung$status == 2, 0L, 1L)

negll <- function(u) {
    a <- exp(u[1])
    b <- exp(u[2])
    -(sum(dbisa(x[cc == 0], scale = b, shape = a, log = TRUE)) +
      sum(pbisa(x[cc == 1], scale = b, shape = a, lower.tail = FALSE,
                log.p = TRUE)))
}
start <- c(log(sd(log(x))), log(median(x)))
fit <- optim(start, negll, method = "BFGS", hessian = FALSE,
             control = list(reltol = 1e-14, maxit = 10000))
fit <- optim(fit$par, negll, method = "Nelder-Mead", hessian = FALSE,
             control = list(reltol = 1e-15, maxit = 20000))
est <- exp(fit$par)
# the Hessian in the natural parameters (not the log ones searched)
negll_nat <- function(q) negll(log(q))
h <- optimHess(est, negll_nat)
se <- sqrt(diag(solve(h)))

out <- list(
    software = list(
        R = R.version.string,
        VGAM = as.character(packageVersion("VGAM")),
        survival = as.character(packageVersion("survival")),
        bssn = if (have_bssn) as.character(packageVersion("bssn")) else NULL
    ),
    functions = out_f,
    fits = list(
        lung = list(x = x, c = cc, loglik = -fit$value, est = est, se = se)
    )
)
path <- file.path("surpyval", "tests", "reference", "data",
                  "r_birnbaum_saunders.json")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, null = "null"), path)
cat("wrote", path, "\n")
