# Reference results from R for the diagnostics of Aalen's nonparametric
# additive hazards model (diagnose() of AalenAdditive).
#
# Writes surpyval/tests/diagnostics/data/r_aalen_additive.json in the
# layout of the files reference_r.R writes (the software, its version, the
# call, the settings and the values of each entry), from the fixtures in
# surpyval/tests/reference/data/fixtures.json.
#
#     R_LIBS_USER=~/Library/R-lib Rscript scripts/reference/reference_r_aalen_diagnostics.R
#
# Run from the repository root, after make_fixtures.py. Needs survival,
# timereg and jsonlite. Re-running reproduces the file byte for byte (the
# resampling is seeded with set.seed).
#
# timereg::aalen(..., robust = 1, n.sim = N) tests each term for a
# time-invariant effect ("Test for time invariant effects"): the
# Kolmogorov-Smirnov statistic sup_t |B(t) - B(tau) t / tau| (obs.testBeqC)
# and the Cramer-von Mises statistic, its square integrated over time
# (obs.testBeqC.is), with p-values from N resampled processes
# (pval.testBeqC, pval.testBeqC.is). Its tau is the last event time, with
# B held at its last estimated value over any event times past the point
# where the risk set can no longer separate the covariates; the times
# (cum[, 1]) and the held values are stored so that the Python side can
# rebuild the anchor. timereg's resampled processes are built from each
# unit's martingale residuals (B.iid), SurPyval's from the events (see
# AalenAdditiveModel.constant_effects_test), so the p-values are kept
# by the tests only as a record: timereg's over-reject constant effects at
# these sizes (scripts/studies/aalen_timereg_size.R). On the lung data
# SurPyval estimates the last event time, where four units are at risk
# for four terms, and timereg holds B there; the tests check the two
# agree before it.
#
# timereg takes one event per time: tied times are broken by adding
# 0.001 * (rank within the tie), after sorting by time, then events first,
# then row, which the Python side repeats.

suppressPackageStartupMessages({
    library(survival)
    library(timereg)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
out_file <- file.path("surpyval", "tests", "diagnostics", "data",
                      "r_aalen_additive.json")
if (!dir.exists(data_dir)) stop("Run from the repository root")
dir.create(dirname(out_file), showWarnings = FALSE)

fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

refs <- list()
record <- function(id, fixture, package, call, settings, values) {
    refs[[id]] <<- list(
        fixture = fixture,
        software = paste("R", package),
        version = as.character(packageVersion(package)),
        call = call,
        settings = settings,
        values = values
    )
}

untie <- function(d, time, status) {
    d <- d[order(d[[time]], -d[[status]], seq_len(nrow(d))), ]
    d[[time]] <- d[[time]] +
        0.001 * (ave(d[[time]], d[[time]], FUN = seq_along) - 1)
    d
}

N_SIM <- 10000

test_values <- function(d, time, status, covariates, seed) {
    f <- as.formula(paste0("Surv(", time, ", ", status, ") ~ ",
                           paste(covariates, collapse = " + ")))
    set.seed(seed)
    fit <- aalen(f, data = d, robust = 1, n.sim = N_SIM)
    list(
        terms = c("baseline", covariates),
        KS = unname(fit$obs.testBeqC),
        CvM = unname(fit$obs.testBeqC.is),
        KS_p = unname(fit$pval.testBeqC),
        CvM_p = unname(fit$pval.testBeqC.is),
        time = unname(fit$cum[-1, 1]),
        cum = unname(fit$cum[-1, -1])
    )
}

settings <- list(
    n_sim = N_SIM,
    statistics = "obs.testBeqC (sup |B(t) - B(tau) t / tau|) and obs.testBeqC.is (its square integrated)",
    p_values = "pval.testBeqC and pval.testBeqC.is, from n.sim processes resampled on the martingale residuals",
    ties = "broken by 0.001 * rank within the tie, after sorting by time, then events first, then row"
)
call_of <- function(rhs, data) {
    paste0("aalen(Surv(time, status) ~ ", rhs, ", data = ", data,
           ", robust = 1, n.sim = ", N_SIM, ")")
}

# --- veteran: karno's effect fades -----------------------------------------
vet <- fixture("veteran")
vet$cell <- vet$celltype
vet <- untie(vet, "time", "status")
record("constant_effects_veteran", "veteran", "timereg",
       call_of("trt + karno + age + cell", "vet"), c(settings, list(seed = 1)),
       test_values(vet, "time", "status",
                   c("trt", "karno", "age", "cell"), 1))

# --- lung: age, sex, ph_ecog (the row missing ph_ecog dropped) -------------
lung <- fixture("lung")
lung <- lung[!is.na(lung$ph_ecog), ]
lung$status <- 1 - lung$c
lung <- untie(lung, "time", "status")
record("constant_effects_lung", "lung", "timereg",
       call_of("age + sex + ph_ecog", "lung"), c(settings, list(seed = 2)),
       test_values(lung, "time", "status", c("age", "sex", "ph_ecog"), 2))

# --- the additive fixture (untied, constant effects by construction) -------
add <- fixture("additive")
add$status <- 1 - add$c
record("constant_effects_additive", "additive", "timereg",
       call_of("z1 + z2", "add"), c(settings, list(seed = 3)),
       test_values(add, "x", "status", c("z1", "z2"), 3))

out <- list(
    generator = "scripts/reference/reference_r_aalen_diagnostics.R",
    R = R.version.string,
    references = refs
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           out_file)
