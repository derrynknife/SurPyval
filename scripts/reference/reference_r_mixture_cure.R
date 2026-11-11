# Reference results from R for mixture-cure regression: covariates on the
# cured fraction and on the latency distribution, fitted jointly by
# maximum likelihood.
#
# Writes surpyval/tests/reference/data/r_mixture_cure.json in the layout
# of the other reference files: for each fit the fixture, the software,
# its version, the call, the settings and the values. Generated with
# R 4.6.2, flexsurv 2.3.2 and flexsurvcure 1.3.3.
#
#     python scripts/reference/make_fixtures.py
#     Rscript scripts/reference/reference_r_mixture_cure.R
#
# Run from the repository root, after make_fixtures.py. Needs
# flexsurvcure and jsonlite. Re-running reproduces the file byte for
# byte. The data are simulated here (seed 20261111) and kept as the
# shared fixture "cure_regression"; the script stops if its draws
# differ from the fixture.
#
# The data: 400 units with a binary covariate z1 and a continuous z2. A
# unit is susceptible with probability plogis(-0.3 + 0.9 z1 + 0.6 z2); a
# susceptible unit's life is Weibull, shape 1.8, scale 10 exp(-0.5 z1 +
# 0.3 z2); follow-up ends at a uniform(15, 40) time. `entry` is a late
# entry age (uniform(0, 4)) used by one fit, which keeps the units still
# running at it.
#
# flexsurvcure's theta is the *cured* fraction, with the covariates of
# the main formula; the scale (or meanlog) takes those of `anc`. Each
# entry records the coefficients as flexsurvcure reports them
# (res.t: the estimates on the link scale, with their standard errors),
# the log-likelihood, and the survival at fixed times for two covariate
# rows.
#
# * weibull_logit: dist = "weibull" (shape, scale), link = "logistic".
# * lnorm_probit: dist = "lnorm", link = "probit".
# * llogis_loglog: dist = "llogis", link = "loglog" (theta = exp(-exp(
#   eta)), so 1 - theta is the complementary log-log of eta).
# * weibull_logit_entry: weibull_logit with left truncation at `entry`
#   (flexsurvreg's default starting values are out of range for
#   start-stop data, so `inits` are given).
# * weibull_logit_scale_only: theta ~ 1 (no covariates on the cure
#   fraction), covariates on the scale.

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(flexsurvcure)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}

set.seed(20261111)
n <- 400
z1 <- rbinom(n, 1, 0.5)
z2 <- round(rnorm(n), 4)
susceptible <- runif(n) < plogis(-0.3 + 0.9 * z1 + 0.6 * z2)
life <- rweibull(n, shape = 1.8, scale = 10 * exp(-0.5 * z1 + 0.3 * z2))
follow <- runif(n, 15, 40)
event <- as.numeric(susceptible & life < follow)
time <- round(ifelse(event == 1, life, follow), 4)
entry <- round(runif(n, 0, 4), 4)
d <- data.frame(time = time, event = event, z1 = z1, z2 = z2, entry = entry)
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
cure <- as.data.frame(fx$cure_regression$columns)
for (col in names(d)) {
    stopifnot(isTRUE(all.equal(cure[[col]], as.numeric(d[[col]]),
                               tolerance = 0)))
}

times <- c(2, 5, 10, 20, 35)
nd <- data.frame(z1 = c(0, 1), z2 = c(0, 0.5))
refs <- list()

add <- function(id, call, data = d) {
    fit <- eval(parse(text = call))
    s <- summary(fit, newdata = nd, t = times, type = "survival", ci = FALSE,
                 tidy = FALSE)
    refs[[id]] <<- list(
        fixture = "cure_regression",
        software = "R flexsurvcure",
        version = as.character(packageVersion("flexsurvcure")),
        call = call,
        settings = list(
            flexsurv = as.character(packageVersion("flexsurv")),
            rows = if (identical(data, d)) "all" else
                "the rows with time > entry (late entry)",
            theta = "the cured fraction, with the main formula's covariates"
        ),
        values = list(
            names = rownames(fit$res.t),
            est = unname(fit$res.t[, "est"]),
            se = unname(fit$res.t[, "se"]),
            loglik = as.numeric(logLik(fit)),
            n = nrow(model.frame(fit)),
            times = times,
            newdata = as.list(nd),
            sf_row1 = s[[1]]$est,
            sf_row2 = s[[2]]$est
        )
    )
}

add("weibull_logit", paste(
    "flexsurvcure(Surv(time, event) ~ z1 + z2, data = data,",
    "anc = list(scale = ~ z1 + z2), dist = 'weibull', link = 'logistic',",
    "mixture = TRUE)"))
add("lnorm_probit", paste(
    "flexsurvcure(Surv(time, event) ~ z1 + z2, data = data,",
    "anc = list(meanlog = ~ z1 + z2), dist = 'lnorm', link = 'probit',",
    "mixture = TRUE)"))
add("llogis_loglog", paste(
    "flexsurvcure(Surv(time, event) ~ z1 + z2, data = data,",
    "anc = list(scale = ~ z1 + z2), dist = 'llogis', link = 'loglog',",
    "mixture = TRUE)"))
late <- d[d$time > d$entry, ]
add("weibull_logit_entry", paste(
    "flexsurvcure(Surv(entry, time, event) ~ z1 + z2, data = data,",
    "anc = list(scale = ~ z1 + z2), dist = 'weibull', link = 'logistic',",
    "mixture = TRUE, inits = c(0.5, 1.5, 10, 0, 0, 0, 0))"), data = late)
add("weibull_logit_scale_only", paste(
    "flexsurvcure(Surv(time, event) ~ 1, data = data,",
    "anc = list(scale = ~ z1 + z2), dist = 'weibull', link = 'logistic',",
    "mixture = TRUE)"))

out <- list(
    generator = "scripts/reference/reference_r_mixture_cure.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_mixture_cure.json"))
cat("wrote", file.path(data_dir, "r_mixture_cure.json"), "\n")
