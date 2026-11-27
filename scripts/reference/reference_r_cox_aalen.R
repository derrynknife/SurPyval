# Reference results from R for the Cox-Aalen model,
# surpyval.CoxAalen.fit(x, Z, c=c, additive=[...]).
#
# Writes surpyval/tests/reference/data/cox_aalen_r.json, which
# surpyval/tests/univariate/regression/test_cox_aalen.py reads when it is
# present (test_against_r_timereg is skipped until then). For each case:
# its rows, which columns are additive, the software and its version, the
# multiplicative coefficients and their standard errors, and the
# cumulative coefficients of the additive part at a grid of times.
#
#     Rscript scripts/reference/reference_r_cox_aalen.R
#
# Needs timereg and jsonlite. NOT RUN YET: written against the cox.aalen
# documentation (timereg 2.0) -- additive terms bare, multiplicative ones
# in prop(); n.sim = 0 (no resampling tests) and robust = 0 for the
# model-based variance. cox.aalen's default weights and its variance
# estimate are its own (check var.gamma against surpyval's covariance()
# and robvar.gamma against robust_covariance() on the first run), so
# agreement is expected on the coefficients and cumulative coefficients to
# optimiser tolerance, and on the standard errors to a few per cent, not
# to the last digit.

suppressPackageStartupMessages({
    library(survival)
    library(timereg)
    library(jsonlite)
})

cases <- list()

add_case <- function(name, time, status, Z, additive, times) {
    df <- data.frame(time = time, status = status, Z)
    cols <- colnames(Z)
    add <- cols[additive + 1]
    mult <- setdiff(cols, add)
    rhs <- c(if (length(add)) add else "1", paste0("prop(", mult, ")"))
    form <- as.formula(
        paste("Surv(time, status) ~", paste(rhs, collapse = " + "))
    )
    fit <- cox.aalen(form, data = df, n.sim = 0, robust = 0)
    cum <- fit$cum
    at <- sapply(times, function(t) max(which(cum[, 1] <= t)))
    cases[[name]] <<- list(
        x = time,
        delta = status,
        Z = unname(as.matrix(Z)),
        additive = additive,
        gamma = unname(as.numeric(fit$gamma)),
        se_gamma = unname(sqrt(diag(fit$var.gamma))),
        times = times,
        cum = unname(cum[at, -1, drop = FALSE]),
        software = paste("timereg", packageVersion("timereg"))
    )
}

# survival::lung, complete cases; status 2 is a death. sex additive.
lung_cc <- na.omit(survival::lung[, c("time", "status", "age", "sex")])
add_case("lung-sex-additive", lung_cc$time, as.integer(lung_cc$status == 2),
         lung_cc[, c("age", "sex")], additive = 1L,
         times = c(100, 300, 500))

# Simulated: h = (0.2 + 0.3 a 1(t < 2)) exp(0.7 z), censored on (2, 10).
set.seed(20261127)
N <- 300
z <- rbinom(N, 1, 0.5)
a <- runif(N)
e <- rexp(N) * exp(-0.7 * z)
early <- 0.2 + 0.3 * a
t_ <- ifelse(e < 2 * early, e / early, 2 + (e - 2 * early) / 0.2)
end <- runif(N, 2, 10)
add_case("simulated", pmin(t_, end), as.integer(t_ <= end),
         data.frame(z = z, a = a), additive = 1L, times = c(1, 3, 6))

out <- list(cases = cases, R = R.version.string)
path <- file.path("surpyval", "tests", "reference", "data", "cox_aalen_r.json")
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA), path)
cat("wrote", path, "\n")
