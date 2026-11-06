# Reference results from R for the cause-specific Cox model with delayed
# entry (left truncation): CompetingRisksProportionalHazards.fit(..., tl=).
#
# Writes surpyval/tests/reference/data/r_competing_entry.json, with the
# software, its version, the exact call and the settings of each entry, as
# reference_r.R does. Generated with R 4.6.2 and survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_competing_entry.R
#
# Run make_fixtures.py first. Needs survival and jsonlite (Ubuntu:
# r-cran-survival r-cran-jsonlite). Re-running reproduces the file byte
# for byte.
#
# The data are two stored fixtures with entry times made from each row's
# number (from 1), so that the entries fall between the event times:
#
# * pbc (cause 1 transplant, 2 death, 0 censored): tl = round(years *
#   (row %% 4) / 5, 2), covariates age, log_bili and female. Rounded, as
#   coxph's timefix would otherwise merge the entries that are within
#   rounding error of an event time with it (years * 0.2 is another
#   patient's years to the last digit or two); the entries are stored,
#   as the entry "pbc_entry", for the tests to read;
#
# * competing (seeded synthetic, integer times with ties; cause 0
#   censored): tl = x * (row %% 3) / 4, covariates z and group. Many
#   entries equal an event time exactly (x = 4, tl = 1), which checks the
#   convention: an item is at risk at t when tl < t <= x.
#
# Each cause k, each tie method: coxph(Surv(tl, x, cause == k) ~ ...), the
# other causes censored, its coefficients, standard errors, maximised
# partial log-likelihood, and the baseline cumulative hazard at Z = 0
# (basehaz(fit, centered = FALSE)) at its times.
#
# The cumulative incidences: the multi-state coxph(Surv(tl, x,
# factor(cause)) ~ ..., id = id), whose coefficients for each transition
# are the cause-specific fits' (checked), and survfit(fit, newdata = nd),
# its pstate (the Aalen-Johansen estimate from the causes' hazards at the
# newdata row) at the given times, for each cause. Breslow ties only:
# with Efron ties survfit's multi-state curve does not use the Efron
# baseline increments that SurPyval's cif does (with or without delayed
# entry), so the two differ by about 1e-2 on tied data.

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

pbc_d <- fixture("pbc")
pbc_d$row <- seq_len(nrow(pbc_d))
pbc_d$id <- pbc_d$row
pbc_d$x <- pbc_d$years
pbc_d$tl <- round(pbc_d$years * (pbc_d$row %% 4) / 5, 2)

competing_d <- fixture("competing")
competing_d$row <- seq_len(nrow(competing_d))
competing_d$id <- competing_d$row
competing_d$tl <- competing_d$x * (competing_d$row %% 3) / 4

env <- environment()
refs <- list()

entry <- function(fixture, call, settings, values) list(
    fixture = fixture,
    software = "R survival",
    version = as.character(packageVersion("survival")),
    call = call,
    settings = settings,
    values = values
)

cause_specific <- function(id, fixture, data_name, covs, ties, entry_rule) {
    for (k in 1:2) {
        call <- sprintf(
            "coxph(Surv(tl, x, cause == %d) ~ %s, data = %s, ties = '%s')",
            k, covs, data_name, ties
        )
        fit <- eval(parse(text = call), envir = env)
        bh <- basehaz(fit, centered = FALSE)
        refs[[sprintf("%s_cause%d_%s", id, k, ties)]] <<- entry(
            fixture, call,
            list(cause = k, ties = ties, entry = entry_rule),
            list(
                coef = unname(coef(fit)),
                se = unname(sqrt(diag(vcov(fit)))),
                loglik = unname(fit$loglik[2]),
                time = unname(bh$time),
                cumhaz = unname(bh$hazard)
            )
        )
    }
}

cif <- function(id, fixture, data_name, covs, ties, entry_rule, nd, times) {
    call <- sprintf(
        paste("coxph(Surv(tl, x, factor(cause)) ~ %s, data = %s,",
              "id = id, ties = '%s')"),
        covs, data_name, ties
    )
    fit <- eval(parse(text = call), envir = env)
    # The multi-state fit's coefficients are the cause-specific fits'
    for (k in 1:2) {
        one <- eval(parse(text = sprintf(
            "coxph(Surv(tl, x, cause == %d) ~ %s, data = %s, ties = '%s')",
            k, covs, data_name, ties
        )), envir = env)
        stopifnot(max(abs(coef(fit)[grepl(sprintf("_1:%d$", k + 1),
                                           names(coef(fit)))] -
                          coef(one))) < 1e-8)
    }
    values <- list(times = times)
    for (j in seq_len(nrow(nd))) {
        s <- summary(survfit(fit, newdata = nd[j, , drop = FALSE]),
                     times = times, extend = TRUE)
        for (k in 1:2) {
            values[[sprintf("cif%d_row%d", k, j)]] <-
                unname(as.vector(s$pstate[, 1, k + 1]))
        }
    }
    refs[[sprintf("%s_cif_%s", id, ties)]] <<- entry(
        fixture, call,
        list(ties = ties, entry = entry_rule, newdata = as.list(nd),
             prediction = "summary(survfit(fit, newdata = nd[j, ]), times = times, extend = TRUE)$pstate"),
        values
    )
}

pbc_rule <- "tl = round(years * (row %% 4) / 5, 2), row from 1"
refs[["pbc_entry"]] <- entry("pbc", "round(years * (row %% 4) / 5, 2)",
                             list(entry = pbc_rule),
                             list(tl = pbc_d$tl))
competing_rule <- "tl = x * (row %% 3) / 4, row from 1"
for (ties in c("efron", "breslow")) {
    cause_specific("pbc", "pbc", "pbc_d", "age + log_bili + female", ties,
                   pbc_rule)
    cause_specific("competing", "competing", "competing_d", "z + group",
                   ties, competing_rule)
}
cif("pbc", "pbc", "pbc_d", "age + log_bili + female", "breslow", pbc_rule,
    data.frame(age = c(45, 60), log_bili = c(0, 1.5), female = c(1, 0)),
    c(1, 2, 4, 6, 8, 10))
cif("competing", "competing", "competing_d", "z + group", "breslow",
    competing_rule, data.frame(z = c(0, 1), group = c(0, 1)),
    c(1, 2, 3, 5, 8))

out <- list(
    generator = "scripts/reference/reference_r_competing_entry.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_competing_entry.json"))
cat("wrote", file.path(data_dir, "r_competing_entry.json"), "\n")
