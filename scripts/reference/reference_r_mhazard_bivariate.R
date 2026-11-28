# Reference results from R for the Prentice-Cai (Volterra) and Lin-Ying
# bivariate survival estimators, surpyval.multivariate.Dabrowska.fit(...,
# estimator="prentice-cai" / "lin-ying").
#
# Writes surpyval/tests/reference/data/mhazard_bivariate_r.json, which
# surpyval/tests/multivariate/test_bivariate_estimators.py reads when it is
# present (the test is skipped until then). For each data set: its rows,
# the software and its version, and the estimate of each estimator on the
# grid R returns.
#
#     Rscript scripts/reference/reference_r_mhazard_bivariate.R
#
# Needs mhazard and jsonlite. NOT RUN YET: written against the npSurv2
# documentation of mhazard 0.2.3 (estimator = "volterra" and "lin-ying");
# check the call on the first run. The data are made here with a fixed
# seed and rounded, so the stored rows are the rows R used.
#
# Every data set has common censoring (one follow-up time per pair), so
# that Lin-Ying's estimator applies to it:
#
# * continuous: 40 pairs, Gaussian-copula dependent exponential lives;
# * ties: 80 pairs, the same rounded up to whole numbers;
# * small: 12 hand-written pairs.

suppressPackageStartupMessages({
    library(mhazard)
    library(jsonlite)
})

make <- function(N, digits) {
    z1 <- rnorm(N)
    z2 <- 0.6 * z1 + 0.8 * rnorm(N)
    t1 <- -log(1 - pnorm(z1)) * 10
    t2 <- -log(1 - pnorm(z2)) * 5
    cc <- rexp(N, 1 / 12)
    if (digits == 0) {
        t1 <- ceiling(t1)
        t2 <- ceiling(t2)
        cc <- ceiling(cc)
    } else {
        t1 <- round(t1, digits) + 0.001
        t2 <- round(t2, digits) + 0.001
        cc <- round(cc, digits) + 0.0005
    }
    list(x1 = pmin(t1, cc), x2 = pmin(t2, cc),
         d1 = as.integer(t1 <= cc), d2 = as.integer(t2 <= cc))
}

set.seed(20261126)
sets <- list(
    continuous = make(40, 3),
    ties = make(80, 0),
    small = list(
        x1 = c(1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 9),
        x2 = c(2, 1, 3, 2, 4, 5, 3, 6, 7, 5, 8, 9),
        d1 = c(1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0),
        d2 = c(1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0)
    )
)

out <- list()
for (name in names(sets)) {
    s <- sets[[name]]
    base <- npSurv2(s$x1, s$x2, s$d1, s$d2, estimator = "dabrowska")
    sf <- list()
    for (est in c("volterra", "lin-ying")) {
        r <- npSurv2(s$x1, s$x2, s$d1, s$d2, newT1 = base$T1,
                     newT2 = base$T2, estimator = est)
        key <- if (est == "volterra") "prentice-cai" else est
        sf[[key]] <- unname(split(r$Fhat_est, row(r$Fhat_est)))
    }
    out[[name]] <- list(
        software = "R mhazard",
        version = as.character(packageVersion("mhazard")),
        call = "npSurv2(x1, x2, d1, d2, newT1, newT2, estimator)$Fhat_est",
        x1 = s$x1, x2 = s$x2, d1 = s$d1, d2 = s$d2,
        t1 = base$T1, t2 = base$T2,
        sf = sf
    )
}
path <- file.path("surpyval", "tests", "reference", "data",
                  "mhazard_bivariate_r.json")
writeLines(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), path)
