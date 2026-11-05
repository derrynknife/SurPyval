# Reference results from R for the standard errors, confidence intervals
# and model-comparison tests of the time-dependent Brier score, integrated
# Brier score and time-dependent AUC (surpyval.metrics.score_predictions).
#
# Writes surpyval/tests/reference/data/r_metric_inference.json, with the
# software, its version, the exact call (the call string is what is
# evaluated) and the settings of each entry, as reference_r.R does.
# Generated with R 4.6.2, riskRegression 2026.3.11 and timeROC 0.4:
#
#     Rscript scripts/reference/reference_r_metric_inference.R
#
# Run make_fixtures.py first. Needs survival, riskRegression, timeROC and
# jsonlite. Re-running reproduces the file byte for byte.
#
# Three data sets, two models each, so every entry has a comparison:
#
# * prediction_ties and prediction_continuous (make_fixtures.prediction):
#   model "m" is the fixture's stored survival matrix; model "m2" is
#   S(t | z) = exp(-(exp(0.5 z2) t / 10)^1.3) with z2 = round(2 z) / 2, the
#   true effect on a coarsened covariate (its risks tie), rounded to six
#   decimals like the fixture's. ties has whole-number times, so events
#   tie with censorings and with each other.
# * veteran (R survival::veteran): Cox models fitted in R,
#   coxph(Surv(time, status) ~ karno + factor(celltype)) ("cox") and
#   coxph(Surv(time, status) ~ karno) ("karno"), their predicted survival
#   at 30, 90, 180 and 360 days stored here at full precision, so both
#   sides score identical predictions.
#
# riskRegression::Score(se.fit = TRUE, cens.model = "km") gives the Brier
# score and AUC with their standard errors (the IPCW influence functions,
# the censoring Kaplan-Meier's own influence included), the intervals
# (clipped to [0, 1]) and the differences between the models (model minus
# reference) with their standard errors, intervals and p-values. Its
# influence functions (keep = "iid") give the integrated Brier score's: the
# trapezoid weights over the horizons divided by the span, applied to the
# influence functions of the Brier scores (riskRegression has no IBS
# standard error itself; this is the same linear combination SurPyval
# takes). timeROC(iid = TRUE) and its compare() give Blanche et al.'s AUC
# standard errors and the two-marker test; on the untied data they agree
# with Score exactly. With tied event times timeROC's AUC differs from
# Score's (it orders tied times arbitrarily), so it is recorded only on
# prediction_continuous, with markers z and z2 (the risks 1 - S of both
# models are increasing in them at every horizon).

suppressPackageStartupMessages({
    library(survival)
    library(riskRegression)
    library(timeROC)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))

prediction_set <- function(name) {
    p <- fx[[name]]
    d <- as.data.frame(p$columns)
    d$status <- 1 - d$c
    d$z2 <- round(2 * d$z) / 2
    times <- p$times
    S2 <- round(exp(-((exp(0.5 * d$z2) %o% times) / 10)^1.3), 6)
    list(d = d, times = times, S = list(m = p$survival, m2 = S2))
}
sets <- list(
    prediction_ties = prediction_set("prediction_ties"),
    prediction_continuous = prediction_set("prediction_continuous")
)

vet <- as.data.frame(fx$veteran$columns)
vet_times <- c(30, 90, 180, 360)
cox_full <- coxph(Surv(time, status) ~ karno + factor(celltype), data = vet,
                  x = TRUE)
cox_karno <- coxph(Surv(time, status) ~ karno, data = vet, x = TRUE)
sets$veteran <- list(
    d = data.frame(x = vet$time, status = vet$status),
    times = vet_times,
    S = list(cox = 1 - predictRisk(cox_full, newdata = vet, times = vet_times),
             karno = 1 - predictRisk(cox_karno, newdata = vet,
                                     times = vet_times))
)

# ---------------------------------------------------------------------------
# Recording.
# ---------------------------------------------------------------------------
refs <- list()
env <- environment()
finite_or_na <- function(v) {
    if (is.list(v)) return(lapply(v, finite_or_na))
    if (is.numeric(v)) {
        v <- unname(v)
        v[!is.finite(v)] <- NA
        if (is.matrix(v)) return(lapply(seq_len(nrow(v)), function(k) v[k, ]))
    }
    v
}
record <- function(id, fixture, package, call, settings, extract) {
    result <- eval(parse(text = call), envir = env)
    refs[[id]] <<- list(
        fixture = fixture,
        software = paste("R", package),
        version = as.character(packageVersion(package)),
        call = call,
        settings = settings,
        values = finite_or_na(extract(result))
    )
    invisible(result)
}

# The integrated Brier score and its influence function: the trapezoid
# rule over the horizons divided by their span.
trapezoid_weights <- function(times) {
    k <- length(times)
    a <- numeric(k)
    h <- diff(times)
    a[-k] <- a[-k] + h / 2
    a[-1] <- a[-1] + h / 2
    a / (times[k] - times[1])
}

score_values <- function(set, models) function(sc) {
    out <- list(times = set$times)
    N <- nrow(set$d)
    a <- trapezoid_weights(set$times)
    iid <- sc$Brier$iid.decomp
    ibs_if <- list()
    for (m in models) {
        b <- sc$Brier$score[model == m]
        u <- sc$AUC$score[model == m]
        out[[m]] <- list(
            survival = set$S[[m]],
            brier = b$Brier, brier_se = b$se,
            brier_lower = b$lower, brier_upper = b$upper,
            auc = u$AUC, auc_se = u$se,
            auc_lower = u$lower, auc_upper = u$upper
        )
        # The influence functions, one row per subject in data order.
        f <- sapply(set$times, function(t) {
            r <- iid[model == m & times == t]
            r$IF.Brier[order(r$riskRegression_ID)]
        })
        ibs_if[[m]] <- as.numeric(f %*% a)
        out[[m]]$ibs <- sum(a * b$Brier)
        out[[m]]$ibs_se <- sd(ibs_if[[m]]) / sqrt(N)
    }
    bc <- sc$Brier$contrasts
    ac <- sc$AUC$contrasts
    out$contrast <- list(
        model = as.character(bc$model[1]),
        reference = as.character(bc$reference[1]),
        brier = bc$delta.Brier, brier_se = bc$se, brier_lower = bc$lower,
        brier_upper = bc$upper, brier_p = bc$p,
        auc = ac$delta.AUC, auc_se = ac$se, auc_lower = ac$lower,
        auc_upper = ac$upper, auc_p = ac$p,
        ibs = out[[models[2]]]$ibs - out[[models[1]]]$ibs,
        ibs_se = sd(ibs_if[[models[2]]] - ibs_if[[models[1]]]) / sqrt(N)
    )
    out
}

for (nm in names(sets)) {
    set <- sets[[nm]]
    models <- names(set$S)
    assign(paste0("d_", nm), set$d, envir = env)
    for (m in models) {
        assign(paste0("risk_", nm, "_", m), 1 - set$S[[m]], envir = env)
    }
    risks <- paste(sprintf("%s = risk_%s_%s", models, nm, models),
                   collapse = ", ")
    record(paste0("score_", nm), nm, "riskRegression",
           sprintf(paste("Score(list(%s), formula = Surv(x, status) ~ 1,",
                         "data = d_%s, times = %s,",
                         "metrics = c('auc', 'brier'), cens.model = 'km',",
                         "null.model = FALSE, se.fit = TRUE,",
                         "conf.int = 0.95, keep = 'iid')"),
                   risks, nm, paste(deparse(set$times), collapse = "")),
           list(risk = "1 - predicted survival", cens.model = "km",
                conservative = "FALSE (the censoring model's variability included)",
                ibs = "trapezoid over 'times' / span, standard error from Score's iid.decomp",
                se = "sd(influence function) / sqrt(N)"),
           score_values(set, models))
}

d_tr <- sets$prediction_continuous$d
tr_times <- sets$prediction_continuous$times
timeroc_values <- function(m1, m2) function(tr) {
    list(times = tr_times,
         auc = unname(tr$AUC), auc_se = unname(tr$inference$vect_sd_1),
         auc_m2 = unname(m2$AUC),
         auc_se_m2 = unname(m2$inference$vect_sd_1),
         compare_p = unname(compare(m1, m2)$p_values_AUC))
}
tr_z <- timeROC(T = d_tr$x, delta = d_tr$status, marker = d_tr$z, cause = 1,
                weighting = "marginal", times = tr_times, iid = TRUE)
tr_z2 <- timeROC(T = d_tr$x, delta = d_tr$status, marker = d_tr$z2,
                 cause = 1, weighting = "marginal", times = tr_times,
                 iid = TRUE)
record("timeroc_prediction_continuous", "prediction_continuous", "timeROC",
       paste("timeROC(T = d_tr$x, delta = d_tr$status, marker = d_tr$z,",
             "cause = 1, weighting = 'marginal', times = tr_times,",
             "iid = TRUE)"),
       list(markers = "z for model m, z2 = round(2 z) / 2 for model m2",
            weighting = "marginal (Kaplan-Meier censoring)",
            compare = "compare(tr_z, tr_z2)$p_values_AUC"),
       timeroc_values(tr_z, tr_z2))

out <- list(
    generator = "scripts/reference/reference_r_metric_inference.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_metric_inference.json"))
cat("wrote", file.path(data_dir, "r_metric_inference.json"), "\n")
