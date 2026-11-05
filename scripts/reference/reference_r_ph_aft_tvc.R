# Reference results from R for diagnose() of a parametric proportional
# hazards model fitted to time-varying covariates (WeibullPH.fit_tvc,
# ExponentialPH.fit_tvc): the per-unit residuals and the likelihood-ratio
# tests, on start-stop data.
#
# Writes surpyval/tests/reference/data/r_ph_aft_tvc.json, with the
# software, its version, the call and the values of each entry, as the
# other generators do. The data are the heart fixture (survival::heart, the
# Stanford heart transplant data in start-stop form, already in
# fixtures.json), so make_fixtures.py needs no change.
#
#     Rscript scripts/reference/reference_r_ph_aft_tvc.R
#
# Needs survival, flexsurv and jsonlite. Run with R 4.6.2, survival
# 3.8.12 and flexsurv 2.3.2. Re-running reproduces the file byte for byte.
#
# What each value is:
#
# * The fit: flexsurvreg(Surv(start, stop, event) ~ age + transplant,
#   dist = "weibullPH" or "exp"), whose likelihood of counting-process
#   rows is that of left-truncated rows, the likelihood fit_tvc maximises.
#   flexsurv's Weibull PH has H(t | z) = m t^a exp(beta'z): SurPyval's
#   WeibullPH with beta = a and alpha = m^(-1 / a).
# * The residuals: each row's cumulative-hazard increment H(stop | z_row)
#   - H(start | z_row) at the R estimate; each unit's Cox-Snell residual,
#   the sum of its rows' increments, and its martingale residual, its
#   event indicator less that (units in the order of sort(unique(id))).
# * The test of a step in a coefficient at tau (proportional_hazards):
#   tau is the lower median of the event times (the smallest event time
#   at which half the events have happened), the rows are split there
#   (survSplit), and `late` marks the pieces after it. The null model has
#   a step in the baseline (`late`), the alternative also `late:<the
#   covariate>` (or every covariate's, for GLOBAL); the statistic is twice
#   the difference of the log-likelihoods.
# * The test of a quadratic term in age (functional_form): the fit with
#   I(age^2) against the fit without it, as twice the difference of the
#   log-likelihoods (centring and scaling age do not change it).

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
heart_d <- as.data.frame(fx[["heart"]]$columns)
heart_d$event <- 1 - heart_d$c
stopifnot(
    isTRUE(all.equal(heart_d$start, survival::heart$start)),
    isTRUE(all.equal(heart_d$stop, survival::heart$stop)),
    isTRUE(all.equal(heart_d$event, survival::heart$event)),
    isTRUE(all.equal(heart_d$age, survival::heart$age))
)

tight <- list(reltol = 1e-14, maxit = 10000)
fit_of <- function(formula, data, dist) {
    flexsurvreg(formula, data = data, dist = dist, control = tight)
}

events <- sort(heart_d$stop[heart_d$event == 1])
share <- seq_along(events) / length(events)
tau <- events[which(share >= 0.5 - 1e-12)[1]]
split <- survSplit(Surv(start, stop, event) ~ ., data = heart_d,
                   cut = tau, episode = "piece")
split$late <- as.numeric(split$piece == 2)

refs <- list()
for (dist in c("weibullPH", "exp")) {
    base <- Surv(start, stop, event) ~ age + transplant
    fit <- fit_of(base, heart_d, dist)
    est <- fit$res[, "est"]
    if (dist == "weibullPH") {
        a <- est[["shape"]]
        m <- est[["scale"]]
    } else {
        a <- 1
        m <- est[["rate"]]
    }
    b <- est[c("age", "transplant")]
    lp <- heart_d$age * b[["age"]] + heart_d$transplant * b[["transplant"]]
    H <- function(t) m * t^a * exp(lp)
    increment <- H(heart_d$stop) - H(heart_d$start)
    ids <- sort(unique(heart_d$id))
    unit <- match(heart_d$id, ids)
    cox_snell <- as.vector(tapply(increment, unit, sum))
    failed <- as.vector(tapply(heart_d$event, unit, max))

    step <- function(extra) {
        null <- fit_of(Surv(start, stop, event) ~ age + transplant + late,
                       split, dist)
        alt <- fit_of(
            as.formula(paste("Surv(start, stop, event) ~ age + transplant",
                             "+ late +", paste0("late:", extra,
                                                collapse = " + "))),
            split, dist)
        2 * (alt$loglik - null$loglik)
    }
    square <- fit_of(Surv(start, stop, event) ~ age + I(age^2) + transplant,
                     heart_d, dist)

    refs[[paste0("heart_", dist)]] <- list(
        fixture = "heart",
        software = "flexsurv",
        version = as.character(packageVersion("flexsurv")),
        survival_version = as.character(packageVersion("survival")),
        call = sprintf(
            "flexsurvreg(Surv(start, stop, event) ~ age + transplant, dist = '%s')",
            dist),
        settings = list(
            dist = dist, tau = tau, reltol = 1e-14,
            parameterisation = "H(t | z) = m t^a exp(b'z); alpha = m^(-1/a)",
            units = "sort(unique(id))"
        ),
        values = list(
            shape = a, scale_m = m,
            alpha = m^(-1 / a),
            coef = unname(b),
            loglik = fit$loglik,
            increment = increment,
            unit_ids = ids,
            cox_snell = cox_snell,
            martingale = failed - cox_snell,
            step_age = step("age"),
            step_transplant = step("transplant"),
            step_global = step(c("age", "transplant")),
            square_age = 2 * (square$loglik - fit$loglik)
        )
    )
}

out <- list(
    generator = "scripts/reference/reference_r_ph_aft_tvc.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_ph_aft_tvc.json"))
cat("wrote", file.path(data_dir, "r_ph_aft_tvc.json"), "\n")
