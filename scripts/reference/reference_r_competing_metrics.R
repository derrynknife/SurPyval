# Reference results from R for the Brier score, integrated Brier score and
# time-dependent AUC of one cause with competing risks, with their standard
# errors, intervals and model comparisons (surpyval.metrics.brier_score,
# auc_td, integrated_brier_score and score_predictions with e= and cause=).
#
# Writes surpyval/tests/reference/data/r_competing_metrics.json, with the
# software, its version, the exact call (the call string is what is
# evaluated) and the settings of each entry, as reference_r.R does.
# Generated with R 4.6.2, riskRegression 2026.3.11 and timeROC 0.4:
#
#     Rscript scripts/reference/reference_r_competing_metrics.R
#
# Run make_fixtures.py first. Needs survival, riskRegression, prodlim,
# timeROC and jsonlite. Re-running reproduces the file byte for byte.
#
# Two data sets, two models each, so every entry has a comparison:
#
# * competing (make_fixtures.competing, seeded synthetic): two causes and
#   censoring on whole-number times, so events of both causes tie with
#   each other and with censorings. Cause-specific Cox models
#   CSC(Hist(x, cause) ~ group + z) ("csc") and a Fine-Gray model
#   FGR(Hist(x, cause) ~ group + z, cause = k) ("fgr"), scored for cause 1
#   and for cause 2.
# * pbc (the Mayo Clinic PBC trial, R survival::pbc; the fixture's first
#   row per patient of surpyval.datasets.load_pbc2): cause 1 transplant,
#   2 death. CSC(Hist(years, cause) ~ age + log_bili + drug) ("full") and
#   CSC(Hist(years, cause) ~ age) ("age"), scored for death.
#
# Each model's predicted cumulative incidence of the scored cause at the
# horizons (predictRisk) is stored at full precision, so both sides score
# identical predictions and only the metric is compared.
#
# riskRegression::Score(cause = k, se.fit = TRUE, cens.model = "km") gives
# the Brier score and AUC of cause k, their standard errors (the IPCW
# influence functions, the censoring Kaplan-Meier's own influence
# included), the intervals (clipped to [0, 1]) and the differences between
# the models (model minus reference) with their standard errors, intervals
# and p-values. Its influence functions (keep = "iid") give the integrated
# Brier score's, as in reference_r_metric_inference.R.
#
# In riskRegression 2026.3.11 the AUC's standard errors (and those of the
# AUC differences) depend on how the causes are labelled: scoring cause 2
# gives other standard errors than the same data with that cause relabelled
# 1 (and 1 relabelled 2), though the estimates and the Brier score's
# standard errors are the same. On pbc it gives 0.308 for an AUC of 0.83
# at 2 years from 312 patients, where the relabelled call and timeROC both
# give about 0.04. The relabelled call is the one consistent with timeROC,
# so for a cause other than 1 the AUC's standard errors are recorded from
# it too ("score_<set>_as_cause1", the scored cause coded 1).
#
# timeROC(cause = k, weighting = "marginal", iid = TRUE) gives AUC_1
# (controls free of every event) and AUC_2 (controls free of cause k: no
# event, or a competing one) with their standard errors, for a fixed marker
# (the "full" model's incidence of death at 5 years) on pbc. timeROC orders
# tied times arbitrarily, and pbc has seven, so they are broken first: the
# j-th repeat of a time gets 1e-6 * (j - 1) added (the times used are
# stored). AUC_1 and its standard error then agree with SurPyval's to
# rounding; AUC_2's standard errors differ from Score's (and SurPyval's)
# by up to about 2e-4 relative, a difference in timeROC's linearisation.

suppressPackageStartupMessages({
    library(survival)
    library(prodlim)
    library(riskRegression)
    library(timeROC)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))

cr <- as.data.frame(fx$competing$columns)
cr_times <- c(5, 10, 15, 20)
csc_cr <- CSC(Hist(x, cause) ~ group + z, data = cr)
pbc <- as.data.frame(fx$pbc$columns)
pbc_times <- c(2, 4, 6, 8)
csc_full <- CSC(Hist(years, cause) ~ age + log_bili + drug, data = pbc)
csc_age <- CSC(Hist(years, cause) ~ age, data = pbc)

sets <- list()
for (k in 1:2) {
    fgr <- FGR(Hist(x, cause) ~ group + z, data = cr, cause = k)
    sets[[paste0("competing_cause", k)]] <- list(
        fixture = "competing", d = data.frame(x = cr$x, cause = cr$cause),
        times = cr_times, cause = k,
        F = list(
            csc = predictRisk(csc_cr, newdata = cr, times = cr_times,
                              cause = k),
            fgr = predictRisk(fgr, newdata = cr, times = cr_times)
        )
    )
}
sets$pbc_death <- list(
    fixture = "pbc", d = data.frame(x = pbc$years, cause = pbc$cause),
    times = pbc_times, cause = 2,
    F = list(
        full = predictRisk(csc_full, newdata = pbc, times = pbc_times,
                           cause = 2),
        age = predictRisk(csc_age, newdata = pbc, times = pbc_times,
                          cause = 2)
    )
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

trapezoid_weights <- function(times) {
    k <- length(times)
    a <- numeric(k)
    h <- diff(times)
    a[-k] <- a[-k] + h / 2
    a[-1] <- a[-1] + h / 2
    a / (times[k] - times[1])
}

score_values <- function(set, models) function(sc) {
    out <- list(times = set$times, cause = set$cause)
    N <- nrow(set$d)
    a <- trapezoid_weights(set$times)
    iid <- sc$Brier$iid.decomp
    ibs_if <- list()
    for (m in models) {
        b <- sc$Brier$score[model == m]
        u <- sc$AUC$score[model == m]
        out[[m]] <- list(
            cif = set$F[[m]],
            brier = b$Brier, brier_se = b$se,
            brier_lower = b$lower, brier_upper = b$upper,
            auc = u$AUC, auc_se = u$se,
            auc_lower = u$lower, auc_upper = u$upper
        )
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
    models <- names(set$F)
    assign(paste0("d_", nm), set$d, envir = env)
    # The same data with the scored cause coded 1 and cause 1 coded with
    # the scored cause's code.
    relabel <- set$d
    relabel$cause[set$d$cause == set$cause] <- 1
    relabel$cause[set$d$cause == 1] <- set$cause
    assign(paste0("d1_", nm), relabel, envir = env)
    for (m in models) {
        assign(paste0("F_", nm, "_", m), set$F[[m]], envir = env)
    }
    risks <- paste(sprintf("%s = F_%s_%s", models, nm, models),
                   collapse = ", ")
    record(paste0("score_", nm), set$fixture, "riskRegression",
           sprintf(paste("Score(list(%s), formula = Hist(x, cause) ~ 1,",
                         "data = d_%s, times = %s, cause = %d,",
                         "metrics = c('auc', 'brier'), cens.model = 'km',",
                         "null.model = FALSE, se.fit = TRUE,",
                         "conf.int = 0.95, keep = 'iid')"),
                   risks, nm, paste(deparse(set$times), collapse = ""),
                   set$cause),
           list(risk = "predicted cumulative incidence of the cause",
                cause = set$cause, cens.model = "km",
                censoring = "Kaplan-Meier of the censoring of every cause",
                conservative = "FALSE (the censoring model's variability included)",
                ibs = "trapezoid over 'times' / span, standard error from Score's iid.decomp",
                se = "sd(influence function) / sqrt(N)"),
           score_values(set, models))
    if (set$cause != 1) {
        record(paste0("score_", nm, "_as_cause1"), set$fixture,
               "riskRegression",
               sprintf(paste("Score(list(%s), formula = Hist(x, cause) ~ 1,",
                             "data = d1_%s, times = %s, cause = 1,",
                             "metrics = c('auc', 'brier'),",
                             "cens.model = 'km', null.model = FALSE,",
                             "se.fit = TRUE, conf.int = 0.95,",
                             "keep = 'iid')"),
                       risks, nm, paste(deparse(set$times), collapse = "")),
               list(risk = "predicted cumulative incidence of the cause",
                    cause = set$cause,
                    relabelled = sprintf("cause %d coded 1 and 1 coded %d",
                                         set$cause, set$cause),
                    cens.model = "km",
                    conservative = "FALSE (the censoring model's variability included)",
                    se = "sd(influence function) / sqrt(N)"),
               score_values(set, models))
    }
}

# timeROC on pbc: a fixed marker, the "full" model's incidence of death at
# five years.
d_tr <- sets$pbc_death$d
d_tr$x <- d_tr$x + 1e-6 * (ave(d_tr$x, d_tr$x, FUN = seq_along) - 1)
tr_marker <- as.numeric(predictRisk(csc_full, newdata = pbc, times = 5,
                                    cause = 2))
tr_times <- c(3, 5, 7, 9)
record("timeroc_pbc_death", "pbc", "timeROC",
       paste("timeROC(T = d_tr$x, delta = d_tr$cause, marker = tr_marker,",
             "cause = 2, weighting = 'marginal', times = tr_times,",
             "iid = TRUE)"),
       list(marker = "the 'full' model's predicted incidence of death at 5 years",
            weighting = "marginal (Kaplan-Meier censoring)",
            ties = "the j-th repeat of a time + 1e-6 * (j - 1)",
            AUC_1 = "controls free of every event at t",
            AUC_2 = "controls free of cause 2 at t (no event, or a competing one)"),
       function(tr) list(
           times = tr_times, cause = 2, marker = tr_marker, x = d_tr$x,
           auc_1 = unname(tr$AUC_1), auc_1_se = unname(tr$inference$vect_sd_1),
           auc_2 = unname(tr$AUC_2), auc_2_se = unname(tr$inference$vect_sd_2)
       ))

out <- list(
    generator = "scripts/reference/reference_r_competing_metrics.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_competing_metrics.json"))
cat("wrote", file.path(data_dir, "r_competing_metrics.json"), "\n")
