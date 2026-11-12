# Reference results from R's WeibullR for rank-regression (probability
# plotting) fits, their pivotal bounds, and the reduced bias adjustment
# (RBA) of maximum-likelihood fits.
#
# Writes surpyval/tests/reference/data/r_weibullr.json in the layout of
# the other reference files: for each entry the fixture, the software,
# its version, the call, the settings and the values. Generated with
# R 4.6.2 and WeibullR 1.2.4.
#
#     python scripts/reference/make_fixtures.py
#     Rscript scripts/reference/reference_r_weibullr.R
#
# Run from the repository root, after make_fixtures.py. Needs WeibullR
# and jsonlite. Re-running reproduces the file byte for byte (WeibullR's
# pivotal simulation sets its own seed, 1234).
#
# The data are the shared fixture "rank_regression": ten lifetimes,
# complete, censored after the sixth failure (type II), or with two
# suspensions among them. Every fit uses Benard's plotting positions
# (pp = "benard", Johnson's adjusted ranks for suspensions), which
# SurPyval calls heuristic = "Benard".
#
# * mrr_*: the rank-regression fit (eta, beta or meanlog, sdlog), x on y
#   (WeibullR's default), and its "pivotal-rr" bounds, 90%
#   two-sided, S = 1e4, at the probabilities `unrel`. These bounds are
#   constructed differently from the exact pivotal intervals of Lawless
#   (2003) that SurPyval uses (percentiles of the simulated log quantiles
#   u* + b* z_p, rotated about the fitted line, rather than those of
#   (u* - z_p) / b*), so they are recorded, not compared. The y on x
#   comparison is left out: on this fixture WeibullR 1.2.4's
#   reg_method = "YonX" (lslr(getPPP(x, ppos = "Benard"),
#   reg_method = "YonX")) returns an eta of 1.8e-4 and the reciprocal of
#   the x on y shape, so there is nothing comparable.
# * mle_rba_*: the "mle" and "mle-rba" fits (the MLE with the shape, or
#   the lognormal sdlog, multiplied by rba()). The two maximum-likelihood
#   fits differ by about 1e-4 (relative): WeibullR's complete-sample
#   lognormal meanlog is 4.651327, the mean of the logs 4.651412. So the
#   adjustment is compared through the ratio of the two fits.
# * rba_factors: rba(r) for r = 3..30, 50, 100, 343 and 344, for the
#   Weibull (median and mean basis) and the lognormal (WeibullR returns
#   1 above 343 failures).

suppressPackageStartupMessages({
    library(WeibullR)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))$rank_regression$columns
x <- fx$x

version <- as.character(packageVersion("WeibullR"))
unrel <- c(0.01, 0.1, 0.5, 0.9)

entry <- function(call, settings, values) {
    list(
        fixture = "rank_regression",
        software = "R WeibullR",
        version = version,
        call = call,
        settings = settings,
        values = values
    )
}

mrr <- function(fail, susp, dist, method) {
    obj <- wblr.fit(
        wblr(fail, susp, pp = "benard"),
        dist = dist, method.fit = method
    )
    obj <- wblr.conf(
        obj,
        method.conf = "pivotal-rr", ci = 0.9, S = 1e4, seed = 1234L,
        dp = "user", user_dp = unrel
    )
    fit <- obj$fit[[1]]$fit_vec
    b <- obj$fit[[1]]$conf[[1]]$bounds
    b <- b[b$unrel %in% unrel, ]
    list(
        p1 = unname(fit[1]), p2 = unname(fit[2]),
        unrel = b$unrel, lower = b$Lower, datum = b$Datum, upper = b$Upper
    )
}

refs <- list()
designs <- list(
    complete = list(fail = x, susp = NULL),
    type2 = list(fail = x[1:6], susp = fx$x_type2[7:10]),
    mixed = list(fail = x[fx$c_mixed == 0], susp = x[fx$c_mixed == 1])
)
for (d in names(designs)) {
    for (dist in c("weibull", "lognormal2p")) {
        for (method in c("rr-xony")) {
            id <- paste("mrr", dist, sub("rr-", "", method), d, sep = "_")
            des <- designs[[d]]
            refs[[id]] <- entry(
                sprintf(
                    paste0(
                        "wblr.conf(wblr.fit(wblr(fail, susp, pp = 'benard'),",
                        " dist = '%s', method.fit = '%s'), method.conf = ",
                        "'pivotal-rr', ci = 0.9, S = 1e4, seed = 1234)"
                    ),
                    dist, method
                ),
                list(design = d, dist = dist, method = method),
                mrr(des$fail, des$susp, dist, method)
            )
        }
    }
}

for (d in c("complete", "mixed")) {
    for (dist in c("weibull", "lognormal2p")) {
        des <- designs[[d]]
        fits <- lapply(c("mle", "mle-rba"), function(m) {
            wblr.fit(wblr(des$fail, des$susp), dist = dist,
                     method.fit = m)$fit[[1]]$fit_vec
        })
        refs[[paste("mle_rba", dist, d, sep = "_")]] <- entry(
            sprintf(
                paste0("wblr.fit(wblr(fail, susp), dist = '%s', ",
                       "method.fit = 'mle' and 'mle-rba')"),
                dist
            ),
            list(design = d, dist = dist),
            list(
                mle_p1 = unname(fits[[1]][1]), mle_p2 = unname(fits[[1]][2]),
                p1 = unname(fits[[2]][1]), p2 = unname(fits[[2]][2])
            )
        )
    }
}

r <- c(3:30, 50, 100, 343, 344)
refs[["rba_factors"]] <- entry(
    "rba(r, dist, basis)",
    list(r = "3..30, 50, 100, 343, 344"),
    list(
        r = r,
        weibull_median = sapply(r, rba, dist = "weibull", basis = "median"),
        weibull_mean = sapply(r, rba, dist = "weibull", basis = "mean"),
        lognormal = sapply(r, rba, dist = "lognormal")
    )
)

out <- list(
    generator = "scripts/reference/reference_r_weibullr.R",
    R = R.version.string,
    references = refs
)
writeLines(
    toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
    file.path(data_dir, "r_weibullr.json")
)
