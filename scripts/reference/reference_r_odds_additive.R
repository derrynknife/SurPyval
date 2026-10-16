# Reference results from R for the diagnostics of the proportional odds
# and additive hazards models (#870).
#
# Writes surpyval/tests/diagnostics/data/r_odds_additive.json in the
# layout of the files reference_r.R writes (the software, its version, the
# call, the settings and the values of each entry), from the fixtures in
# surpyval/tests/reference/data/fixtures.json.
#
#     Rscript scripts/reference/reference_r_odds_additive.R
#
# Run from the repository root, after make_fixtures.py. Needs survival,
# timereg and jsonlite. Re-running reproduces the file byte for byte.
#
# * The trend test of Lin and Ying's additive hazards model. Follow-up is
#   split into four periods at the quartiles of the event times (each cut
#   the first event time at which that share of the events has happened,
#   without repeats or a cut at or after the last event time), and the
#   model is refitted with the added term theta' u(Z) g(t), g the period
#   (0 to 3) and u(Z) the covariate less its mean (for a covariate with
#   three to five values, the indicator of each level after the first,
#   less its mean). The extended model has a covariate that changes at the
#   cuts: survSplit splits each row there, and aalen() with every term in
#   const() is Lin and Ying's estimator on the split rows, var.gamma its
#   sandwich variance. The Wald statistic of theta is what diagnose()
#   reports.
# * Aalen's nonparametric additive model, aalen() with no const() terms:
#   the cumulative regression coefficients (cum) and their optional
#   variation variance (var.cum, robust = 0), which diagnose() draws.
# * The Kaplan-Meier estimate of each group of a binary covariate, whose
#   log-odds of failure log(F / S) the proportional-odds plots draw.

suppressPackageStartupMessages({
    library(survival)
    library(timereg)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
out_file <- file.path("surpyval", "tests", "diagnostics", "data",
                      "r_odds_additive.json")
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

# The cut times: the k / 4 quantiles of the event times, as diagnose()
# takes them.
cuts_of <- function(time, status, periods = 4) {
    ev <- sort(unique(time[status == 1]))
    w <- as.numeric(table(factor(time[status == 1], levels = ev)))
    share <- cumsum(w) / sum(w)
    q <- seq_len(periods - 1) / periods
    cuts <- unique(sapply(q, function(p) ev[which(share >= p - 1e-12)[1]]))
    cuts[cuts < max(ev)]
}

# u(Z) of one covariate: the covariate, or its level indicators after the
# first, less the mean.
base_of <- function(z) {
    lv <- sort(unique(z))
    if (length(lv) >= 3 && length(lv) <= 5) {
        u <- sapply(lv[-1], function(l) as.numeric(z == l))
    } else {
        u <- matrix(z, ncol = 1)
    }
    u <- sweep(u, 2, colMeans(u))
    u
}

trend_values <- function(d, time, status, covariates, tested) {
    d$id <- seq_len(nrow(d))
    cuts <- cuts_of(d[[time]], d[[status]])
    base <- do.call(cbind, lapply(tested, function(v) base_of(d[[v]])))
    colnames(base) <- paste0("u", seq_len(ncol(base)))
    d <- cbind(d, base)
    s <- survSplit(as.formula(paste0("Surv(", time, ", ", status, ") ~ .")),
                   data = d, cut = cuts, episode = "period")
    for (k in colnames(base)) s[[paste0("w_", k)]] <- s[[k]] * (s$period - 1)
    terms <- c(covariates, paste0("w_", colnames(base)))
    f <- as.formula(paste0("Surv(tstart, ", time, ", ", status, ") ~ ",
                           paste0("const(", terms, ")", collapse = " + ")))
    fit <- aalen(f, data = s, robust = 0)
    p <- length(covariates)
    theta <- fit$gamma[-seq_len(p), 1]
    V <- fit$var.gamma[-seq_len(p), -seq_len(p), drop = FALSE]
    list(cuts = cuts, gamma = unname(fit$gamma[, 1]),
         var_gamma = unname(fit$var.gamma),
         statistic = as.numeric(t(theta) %*% solve(V) %*% theta),
         df = length(theta))
}

aalen_values <- function(d, time, status, covariates) {
    f <- as.formula(paste0("Surv(", time, ", ", status, ") ~ ",
                           paste(covariates, collapse = " + ")))
    fit <- aalen(f, data = d, robust = 0)
    list(time = unname(fit$cum[-1, 1]), cum = unname(fit$cum[-1, -1]),
         var_cum = unname(fit$var.cum[-1, -1]))
}

trend_settings <- list(
    periods = "4, split at the event-time quartiles",
    term = "u(Z) * (period - 1), u(Z) the covariate (or its level indicators) less its mean",
    variance = "var.gamma, Lin and Ying's sandwich (robust = 0)"
)
trend_call <- paste(
    "aalen(Surv(tstart, stop, status) ~ const(z...) + const(u(z) * g),",
    "data = survSplit(..., cut = cuts), robust = 0)")

# --- the additive fixture (untied, censored at random) ---------------------
add <- fixture("additive")
add$status <- 1 - add$c
for (v in c("z1", "z2")) {
    record(paste0("trend_additive_", v), "additive", "timereg", trend_call,
           c(trend_settings, list(tested = v)),
           trend_values(add, "x", "status", c("z1", "z2"), v))
}
record("trend_additive_global", "additive", "timereg", trend_call,
       c(trend_settings, list(tested = "z1, z2 (GLOBAL)")),
       trend_values(add, "x", "status", c("z1", "z2"), c("z1", "z2")))
record("aalen_additive", "additive", "timereg",
       "aalen(Surv(x, status) ~ z1 + z2, data = add, robust = 0)",
       list(variance = "var.cum, the optional variation"),
       aalen_values(add, "x", "status", c("z1", "z2")))

# --- the veteran fixture: karno's effect fades -----------------------------
# Untied for timereg: aalen() takes one event per time, so the tied times
# are broken by adding 0.001 * (rank within the tie), which the Python
# side repeats.
vet <- fixture("veteran")
vet <- vet[order(vet$time, -vet$status, seq_len(nrow(vet))), ]
vet$time <- vet$time + 0.001 * (ave(vet$time, vet$time, FUN = seq_along) - 1)
vet$cell <- vet$celltype
vet_cov <- c("trt", "karno", "age", "cell")
for (v in vet_cov) {
    record(paste0("trend_veteran_", v), "veteran", "timereg", trend_call,
           c(trend_settings, list(tested = v, ties = "broken by 0.001 * rank within the tie, after sorting by time, then events first, then row")),
           trend_values(vet, "time", "status", vet_cov, v))
}
record("trend_veteran_global", "veteran", "timereg", trend_call,
       c(trend_settings, list(tested = "trt, karno, age, cell (GLOBAL)")),
       trend_values(vet, "time", "status", vet_cov, vet_cov))
record("aalen_veteran", "veteran", "timereg",
       "aalen(Surv(time, status) ~ trt + karno + age + cell, data = vet, robust = 0)",
       list(variance = "var.cum, the optional variation"),
       aalen_values(vet, "time", "status", vet_cov))

# --- Kaplan-Meier log-odds by group (lung, by sex) -------------------------
lung <- fixture("lung")
lung$status <- 1 - lung$c
km <- survfit(Surv(time, status) ~ sex, data = lung)
s <- summary(km)
record("km_lung_sex", "lung", "survival",
       "summary(survfit(Surv(time, status) ~ sex, data = lung))",
       list(note = "the survival at each event time of each group"),
       list(sex = as.numeric(sub("sex=", "", as.character(s$strata))),
            time = s$time, surv = s$surv))

out <- list(
    generator = "scripts/reference/reference_r_odds_additive.R",
    R = R.version.string,
    references = refs
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           out_file)
