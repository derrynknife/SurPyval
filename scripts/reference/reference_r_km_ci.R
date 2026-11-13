# Reference results from R for the pointwise confidence intervals of the
# Kaplan-Meier and Nelson-Aalen survival estimates, by interval type:
# NonParametric.cb(bound_type=...).
#
# Writes surpyval/tests/reference/data/r_km_ci.json, with the software, its
# version, the exact call and the settings of each entry, as the other
# generators do. Every fit is to a fixture already in fixtures.json (aml
# and lung, survival's data sets), so make_fixtures.py needs no change.
# Generated with R 4.6.2, survival 3.8.12 and km.ci 0.5.6.
#
#     Rscript scripts/reference/reference_r_km_ci.R
#
# Needs survival, km.ci and jsonlite. Re-running reproduces the file byte
# for byte.
#
# SurPyval's bound_type and R's names:
#
#   exp      survfit(conf.type = "log-log")
#   normal   survfit(conf.type = "plain"), which R cuts to [0, 1]
#   log      survfit(conf.type = "log"), upper bound cut at 1
#   logit    survfit(conf.type = "logit")
#   arcsine  survfit(conf.type = "arcsin")
#   peto     km.ci(method = "peto"): S +- z S sqrt((1 - S) / n.risk)
#   rothman  km.ci(method = "rothman"): Rothman's Wilson-type interval
#
# The Nelson-Aalen entries are survfit(stype = 2, ctype = 1), the survival
# exp(-H) with H the Nelson-Aalen estimate and the standard error of H
# Aalen's, sqrt(sum d / r^2), on which the same transformations act.

suppressPackageStartupMessages({
    library(survival)
    library(km.ci)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

aml_d <- fixture("aml")
lung_d <- fixture("lung")
lung_d$status <- 1 - lung_d$c
stopifnot(
    isTRUE(all.equal(aml_d$time, survival::aml$time)),
    isTRUE(all.equal(lung_d$time, survival::lung$time))
)

r_types <- c(exp = "log-log", normal = "plain", log = "log",
             logit = "logit", arcsine = "arcsin")
kmci_types <- c(peto = "peto", rothman = "rothman")

refs <- list()
record <- function(id, fixture_name, data_call, d, level, stype) {
    vals <- list()
    calls <- character(0)
    base <- survfit(Surv(time, status) ~ 1, data = d, conf.int = level,
                    stype = stype, ctype = 1)
    vals$time <- base$time
    vals$surv <- base$surv
    vals$n_risk <- base$n.risk
    for (k in names(r_types)) {
        fit <- survfit(Surv(time, status) ~ 1, data = d, conf.int = level,
                       conf.type = r_types[[k]], stype = stype, ctype = 1)
        vals[[paste0("lower_", k)]] <- fit$lower
        vals[[paste0("upper_", k)]] <- fit$upper
        calls <- c(calls, sprintf(
            paste0("survfit(Surv(time, status) ~ 1, %s, conf.int = %s, ",
                   "conf.type = \"%s\", stype = %d, ctype = 1)"),
            data_call, format(level), r_types[[k]], stype))
    }
    if (stype == 1) {
        for (k in names(kmci_types)) {
            fit <- km.ci(base, conf.level = level, method = kmci_types[[k]])
            vals[[paste0("lower_", k)]] <- fit$lower
            vals[[paste0("upper_", k)]] <- fit$upper
            calls <- c(calls, sprintf(
                "km.ci(survfit(...), conf.level = %s, method = \"%s\")",
                format(level), kmci_types[[k]]))
        }
    }
    refs[[id]] <<- list(
        fixture = fixture_name,
        software = if (stype == 1) "R survival + km.ci" else "R survival",
        version = if (stype == 1) {
            paste0("survival ", as.character(packageVersion("survival")),
                   ", km.ci ", as.character(packageVersion("km.ci")))
        } else {
            as.character(packageVersion("survival"))
        },
        call = paste(calls, collapse = "; "),
        settings = list(
            estimator = if (stype == 1) "Kaplan-Meier" else
                "Nelson-Aalen (exp(-H))",
            level = level,
            types = "lower_<type> and upper_<type>, SurPyval's bound_type"
        ),
        values = vals
    )
}

record("aml_km_95", "aml", "data = aml_d", aml_d, 0.95, 1)
record("aml_km_90", "aml", "data = aml_d", aml_d, 0.90, 1)
record("lung_km_95", "lung", "data = lung_d (status = 1 - c)", lung_d,
       0.95, 1)
record("aml_na_95", "aml", "data = aml_d", aml_d, 0.95, 2)

out <- list(
    generator = "scripts/reference/reference_r_km_ci.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_km_ci.json"))
cat("wrote", file.path(data_dir, "r_km_ci.json"), "\n")
