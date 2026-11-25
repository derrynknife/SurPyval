# Reference results from R for SurPyval's built-in GeneralizedGamma
# (flexsurv's "gengamma", Prentice's (mu, sigma, Q)) and Gompertz
# (flexsurv's "gompertz", (shape, rate)).
#
# Writes surpyval/tests/reference/data/r_distributions.json, with the
# software, its version, the call and the values of each entry, as the
# other generators do. The fits are to survival's lung data, the lung
# fixture already in fixtures.json, so make_fixtures.py needs no change.
#
#     Rscript scripts/reference/reference_r_distributions.R
#
# Needs survival, flexsurv and jsonlite. Run with R 4.6.2, survival 3.8.12
# and flexsurv 2.3.2. Re-running reproduces the file byte for byte.
#
# What each value is:
#
# * The density, distribution function, log survival function and log
#   density (dgengamma, pgengamma) at fixed times, and the quantiles
#   (qgengamma) at fixed probabilities, for parameter sets with both
#   signs of Q, Q near 0 and Q = 0 (the lognormal).
# * The fits: flexsurvreg(Surv(time, status) ~ 1) on lung, right
#   censored, and on lung left truncated at entry = floor(time / 4)
#   (Surv(entry, time, status)), with optim's relative tolerance
#   tightened to 1e-14: the maximised log-likelihood, the estimates and
#   their standard errors. For the Gompertz also ~ sex, which in
#   flexsurv acts on the rate: the proportional hazards model.

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
lung_d <- as.data.frame(fx[["lung"]]$columns)
lung_d$status <- 1 - lung_d$c
lung_d$entry <- floor(lung_d$time / 4)
stopifnot(isTRUE(all.equal(lung_d$time, survival::lung$time)))

versions <- list(
    survival = as.character(packageVersion("survival")),
    flexsurv = as.character(packageVersion("flexsurv"))
)

refs <- list()
add <- function(id, entry) refs[[id]] <<- entry
tight <- list(reltol = 1e-14, maxit = 10000)

functions_entry <- function(dist, call, pars, times, probs, d, p, q) {
    list(
        fixture = "lung", software = "R flexsurv",
        version = versions$flexsurv, call = call,
        settings = list(t = times, p = probs, params = pars,
                        note = "no data: the fixture field is nominal"),
        values = list(
            pdf = lapply(pars, function(a) do.call(d, c(list(times), a))),
            log_pdf = lapply(pars, function(a)
                do.call(d, c(list(times), a, list(log = TRUE)))),
            cdf = lapply(pars, function(a) do.call(p, c(list(times), a))),
            log_sf = lapply(pars, function(a)
                do.call(p, c(list(times), a,
                             list(lower.tail = FALSE, log.p = TRUE)))),
            quantile = lapply(pars, function(a)
                do.call(q, c(list(probs), a)))
        )
    )
}

fit_entry <- function(dist, formula, label) {
    fit <- flexsurvreg(formula, data = lung_d, dist = dist, control = tight)
    list(
        fixture = "lung", software = "R flexsurv",
        version = versions$flexsurv,
        call = sprintf(
            "flexsurvreg(%s, data = lung, dist = '%s', control = list(reltol = 1e-14))",
            label, dist),
        settings = list(dist = dist, formula = label,
                        entry = "floor(time / 4) where truncated"),
        values = list(
            loglik = fit$loglik,
            names = rownames(fit$res),
            est = unname(fit$res[, "est"]),
            se = unname(fit$res[, "se"])
        )
    )
}

times <- c(0.05, 0.3, 1, 2, 5, 10, 30, 100)
probs <- c(1e-6, 0.001, 0.1, 0.5, 0.9, 0.999)

# ---- generalised gamma -------------------------------------------------
gg_pars <- list(c(1.5, 0.6, 0.5), c(1.5, 0.6, -0.5), c(0.2, 1.2, 2),
                c(0.2, 1.2, -2), c(2, 0.4, 0.05), c(1, 0.8, 1),
                c(1, 0.8, 0), c(3, 1.5, 0.8))
add("functions_gengamma", functions_entry(
    "gengamma", "dgengamma / pgengamma / qgengamma(., mu, sigma, Q)",
    gg_pars, times, probs, dgengamma, pgengamma, qgengamma))
add("lung_gengamma", fit_entry(
    "gengamma", Surv(time, status) ~ 1, "Surv(time, status) ~ 1"))
add("lung_gengamma_truncated", fit_entry(
    "gengamma", Surv(entry, time, status) ~ 1,
    "Surv(entry, time, status) ~ 1"))

# ---- Gompertz -----------------------------------------------------------
gz_pars <- list(c(0.1, 0.01), c(-0.1, 0.05), c(0, 0.2), c(1e-6, 0.2),
                c(2, 1e-3), c(-2, 0.5))
gz_probs <- c(1e-6, 0.001, 0.1, 0.5, 0.9)
add("functions_gompertz", functions_entry(
    "gompertz", "dgompertz / pgompertz / qgompertz(., shape, rate)",
    gz_pars, times, gz_probs, dgompertz, pgompertz, qgompertz))
# flexsurv's time is in days on lung; in years the shape is not tiny
lung_d$time_y <- lung_d$time / 365.25
lung_d$entry_y <- lung_d$entry / 365.25
add("lung_gompertz", fit_entry(
    "gompertz", Surv(time_y, status) ~ 1, "Surv(time / 365.25, status) ~ 1"))
add("lung_gompertz_truncated", fit_entry(
    "gompertz", Surv(entry_y, time_y, status) ~ 1,
    "Surv(entry / 365.25, time / 365.25, status) ~ 1"))
add("lung_gompertz_sex", fit_entry(
    "gompertz", Surv(time_y, status) ~ sex,
    "Surv(time / 365.25, status) ~ sex"))

out <- list(
    generator = "scripts/reference/reference_r_distributions.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_distributions.json"))
cat("wrote", file.path(data_dir, "r_distributions.json"), "\n")
