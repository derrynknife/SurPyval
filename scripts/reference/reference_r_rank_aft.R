# Reference results from R for the rank-based semi-parametric AFT model,
# surpyval.RankAFT.fit(x, Z, c=c, rank_weights="gehan" / "logrank").
#
# Writes surpyval/tests/reference/data/rank_aft_r.json, which
# surpyval/tests/univariate/regression/test_rank_aft.py reads when it is
# present (test_against_r_rank_aft is skipped until then). For each case:
# its rows, the rank weights, the software and its version, the
# coefficients (R's sign: the slope of log time, the negative of
# surpyval's) and their standard errors.
#
#     Rscript scripts/reference/reference_r_rank_aft.R
#
# Needs aftgee and jsonlite. NOT RUN YET: written against the aftsrr
# documentation (aftgee 1.2) -- eqType = "is" (induced smoothing) and
# se = "ISMB" (the induced-smoothing sandwich) are the choices closest to
# surpyval's; check the call and the names of the returned pieces on the
# first run. aftgee's smoothing covariance and its stopping rule are its
# own, so agreement is expected to a few parts in a thousand on the
# coefficients and a few per cent on the standard errors, not to the last
# digit.

suppressPackageStartupMessages({
    library(survival)
    library(aftgee)
    library(jsonlite)
})

cases <- list()

add_case <- function(name, time, status, X, weights) {
    df <- data.frame(time = time, status = status, X)
    form <- as.formula(
        paste("Surv(time, status) ~", paste(colnames(X), collapse = " + "))
    )
    fit <- aftsrr(form, data = df, rankWeights = weights, eqType = "is",
                  se = "ISMB", B = 0)
    cases[[paste0(name, "-", weights)]] <<- list(
        x = time,
        delta = status,
        Z = unname(as.matrix(X)),
        rank_weights = weights,
        coef = unname(fit$beta),
        se = unname(sqrt(diag(fit$covmat$ISMB))),
        software = paste("aftgee", packageVersion("aftgee"))
    )
}

# survival::lung, complete cases of age and sex; status 2 is a death.
lung_cc <- na.omit(survival::lung[, c("time", "status", "age", "sex")])
for (w in c("gehan", "logrank")) {
    add_case("lung", lung_cc$time, as.integer(lung_cc$status == 2),
             lung_cc[, c("age", "sex")], w)
}

# Simulated: Weibull lives with a binary and a uniform covariate, uniform
# censoring, times rounded to one decimal (ties).
set.seed(20261127)
N <- 150
z1 <- rbinom(N, 1, 0.5)
z2 <- runif(N, -1, 1)
t_ <- exp(1 + 0.5 * z1 - z2 + 0.5 * log(rexp(N)))
cc <- runif(N, 1, 15)
x <- round(pmin(t_, cc), 1)
st <- as.integer(t_ <= cc)
for (w in c("gehan", "logrank")) {
    add_case("simulated", x, st, data.frame(z1 = z1, z2 = z2), w)
}

out <- list(cases = cases, R = R.version.string)
path <- file.path("surpyval", "tests", "reference", "data", "rank_aft_r.json")
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), path)
cat("wrote", path, "\n")
