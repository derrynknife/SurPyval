# Reference results from R for Dabrowska's bivariate survival estimator,
# surpyval.multivariate.Dabrowska (Dabrowska 1988).
#
# Writes surpyval/tests/reference/data/r_mhazard_dabrowska.json: for each
# data set, its rows, the software and its version, the call, and the
# estimate on the grid of the two series' failure times. Generated with
# R 4.6.2 and mhazard 0.2.3 (npSurv2, estimator = "dabrowska").
#
#     Rscript scripts/reference/reference_r_mhazard_dabrowska.R
#
# Needs mhazard and jsonlite. The data are made here with a fixed seed
# and rounded to three decimals (or to whole numbers, for ties), so the
# stored rows are the rows R used, digit for digit.
#
# The data sets:
#
# * continuous: 40 pairs, Gaussian-copula dependent exponential lives,
#   each series censored by its own exponential time;
# * ties: 80 pairs, the same rounded up to whole numbers (ties within
#   each series and between failures and censorings);
# * small: 12 hand-written pairs with ties across the two series.

suppressPackageStartupMessages({
    library(mhazard)
    library(jsonlite)
})

make <- function(N, digits) {
    z1 <- rnorm(N)
    z2 <- 0.6 * z1 + 0.8 * rnorm(N)
    t1 <- -log(1 - pnorm(z1)) * 10
    t2 <- -log(1 - pnorm(z2)) * 5
    c1 <- rexp(N, 1 / 15)
    c2 <- rexp(N, 1 / 8)
    x1 <- pmin(t1, c1)
    x2 <- pmin(t2, c2)
    if (digits == 0) {
        x1 <- ceiling(x1)
        x2 <- ceiling(x2)
    } else {
        x1 <- round(x1, digits) + 0.001
        x2 <- round(x2, digits) + 0.001
    }
    list(x1 = x1, x2 = x2, d1 = as.integer(t1 <= c1),
         d2 = as.integer(t2 <= c2))
}

set.seed(20261125)
sets <- list(
    continuous = make(40, 3),
    ties = make(80, 0),
    small = list(
        x1 = c(1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9),
        x2 = c(2, 1, 3, 3, 2, 5, 4, 6, 1, 7, 3, 8),
        d1 = c(1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1),
        d2 = c(1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1)
    )
)

out <- list()
for (name in names(sets)) {
    s <- sets[[name]]
    r <- npSurv2(s$x1, s$x2, s$d1, s$d2, estimator = "dabrowska")
    out[[name]] <- list(
        software = "R mhazard",
        version = as.character(packageVersion("mhazard")),
        call = "npSurv2(x1, x2, d1, d2, estimator = 'dabrowska')$Fhat_est",
        x1 = s$x1, x2 = s$x2, d1 = s$d1, d2 = s$d2,
        t1 = r$T1, t2 = r$T2,
        sf = unname(split(r$Fhat_est, row(r$Fhat_est)))
    )
}
path <- file.path("surpyval", "tests", "reference", "data",
                  "r_mhazard_dabrowska.json")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), path)
