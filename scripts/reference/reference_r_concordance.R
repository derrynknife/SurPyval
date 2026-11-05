# Reference results from R for the concordance indices: Harrell's C and
# Uno's IPCW C, truncated at a horizon or not, with their
# infinitesimal-jackknife variances and the covariance of two scores'
# indices on the same subjects.
#
# Writes surpyval/tests/reference/data/r_concordance.json, with the
# software, its version, the call and the values of each entry, as
# reference_r.R does for the other files. Generated with R 4.6.2,
# survival 3.8.12 and survC1 1.0-3.
#
#     Rscript scripts/reference/reference_r_concordance.R
#
# Run make_fixtures.py first. Needs survival, survC1 and jsonlite.
# Re-running reproduces the file byte for byte.
#
# The scores are fixed linear predictors, higher meaning an earlier death,
# stored in the file so the tests use exactly R's numbers:
#
# * lung (the rows with ph.ecog missing dropped): lp_a = 0.0173 age +
#   0.4721 ph.ecog - 0.5487 sex (few ties) and lp_b = ph.ecog (four
#   values, many ties);
# * veteran: lp_a = -0.0328 karno + 0.0071 age + 0.0113 diagtime and
#   lp_b = -karno (many ties).
#
# For each, survival::concordance(Surv(time, status) ~ lp_a + lp_b,
# reverse = TRUE, timewt = "n" or "n/G2", ymax = NULL or tau) gives both
# indices, their counts and the 2 x 2 jackknife covariance ($var); a
# one-score call gives the single variance. reverse = TRUE because a
# higher score predicts a shorter time.
#
# survC1::Est.Cval (Uno et al.'s own code) is a second reference for Uno's
# C, on the prediction_continuous and additive fixtures, which have no
# tied times (Est.Cval truncates times * 1000 and scores * 1e5 to integers
# and its sums are not exact to double precision, so it agrees with
# survival's "n/G2" to about 1e-8 there). It counts only the events
# before tau (survival: at or before), so tau is not a time in the data.

suppressPackageStartupMessages({
    library(survival)
    library(survC1)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
lung_d <- as.data.frame(fx$lung$columns)
lung_d$status <- 1 - lung_d$c
lung_cc <- lung_d[!is.na(lung_d$ph_ecog), ]
vet_d <- as.data.frame(fx$veteran$columns)
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(vet_d$time, survival::veteran$time)),
    isTRUE(all.equal(vet_d$status, survival::veteran$status)),
    isTRUE(all.equal(vet_d$karno, survival::veteran$karno))
)
lung_cc$lp_a <- 0.0173 * lung_cc$age + 0.4721 * lung_cc$ph_ecog -
    0.5487 * lung_cc$sex
lung_cc$lp_b <- lung_cc$ph_ecog
vet_d$lp_a <- -0.0328 * vet_d$karno + 0.0071 * vet_d$age +
    0.0113 * vet_d$diagtime
vet_d$lp_b <- -vet_d$karno

# Distinct scores are far apart, so a tie is a tie in both programs.
for (lp in list(lung_cc$lp_a, lung_cc$lp_b, vet_d$lp_a, vet_d$lp_b)) {
    stopifnot(min(diff(sort(unique(lp)))) > 1e-6)
}

env <- environment()
refs <- list()

entry <- function(fixture, software, call, settings, values) list(
    fixture = fixture,
    software = software,
    version = as.character(packageVersion(
        if (software == "R survC1") "survC1" else "survival"
    )),
    call = call,
    settings = settings,
    values = values
)

cases <- list(
    lung = list(data = "lung_cc", taus = list(NULL, 500)),
    veteran = list(data = "vet_d", taus = list(NULL, 100))
)
for (name in names(cases)) {
    d <- get(cases[[name]]$data)
    for (timewt in c("n", "n/G2")) {
        for (tau in cases[[name]]$taus) {
            ymax <- if (is.null(tau)) "NULL" else format(tau)
            call <- sprintf(paste(
                "concordance(Surv(time, status) ~ lp_a + lp_b, data = %s,",
                "reverse = TRUE, timewt = '%s', ymax = %s)"),
                cases[[name]]$data, timewt, ymax)
            fit <- eval(parse(text = call), envir = env)
            single <- lapply(c("lp_a", "lp_b"), function(v) {
                one <- sub("lp_a \\+ lp_b", v, call)
                eval(parse(text = one), envir = env)$var
            })
            id <- sprintf("%s_%s_%s", name,
                          if (timewt == "n") "harrell" else "uno",
                          if (is.null(tau)) "all" else paste0("tau", tau))
            refs[[id]] <- entry(
                name, "R survival", call,
                list(timewt = timewt, ymax = tau,
                     scores = "lp_a and lp_b (stored under values)",
                     rows = if (name == "lung")
                         "the rows with ph.ecog missing dropped" else "all"),
                list(
                    lp_a = d$lp_a,
                    lp_b = d$lp_b,
                    concordance = unname(fit$concordance),
                    count = lapply(1:2, function(k)
                        unname(fit$count[k, c("concordant", "discordant",
                                              "tied.x")])),
                    var = lapply(1:2, function(k) unname(fit$var[k, ])),
                    var_single = unlist(single)
                )
            )
        }
    }
}

# Uno's C from survC1 and survival on two fixtures without tied times.
# Est.Cval truncates times * 1000 and scores * 1e5 to integers, so it is
# given the times in thousandths and the scores in hundredths, rounded
# (whole numbers, the same ranking), and tau between two times.
pc_d <- as.data.frame(fx$prediction_continuous$columns)
add_d <- as.data.frame(fx$additive$columns)
uno_cases <- list(
    list(fixture = "prediction_continuous", data = "pc_d", score = "z",
         taus = c(4.5, 10.5)),
    list(fixture = "additive", data = "add_d", score = "z2",
         taus = c(5.5, 15.5))
)
for (case in uno_cases) {
    d <- get(case$data)
    stopifnot(!anyDuplicated(d$x))
    d$status <- 1 - d$c
    d$ms <- round(1000 * d$x)
    d$hs <- round(100 * d[[case$score]])
    assign("uno_d", d, envir = env)
    for (tau in case$taus) {
        call <- sprintf(paste("Est.Cval(cbind(uno_d$ms, uno_d$status,",
                              "uno_d$hs), tau = %s, nofit = TRUE)$Dhat"),
                        format(1000 * tau))
        dhat <- eval(parse(text = call), envir = env)
        s_call <- sprintf(paste("concordance(Surv(x, status) ~ %s, data =",
                                "uno_d, reverse = TRUE, timewt = 'n/G2',",
                                "ymax = %s)"), case$score, format(tau))
        fit <- eval(parse(text = s_call), envir = env)
        refs[[sprintf("%s_uno_survc1_tau%s", case$fixture, format(tau))]] <-
            entry(
                case$fixture, "R survC1", call,
                list(tau = tau, score = case$score,
                     scaling = paste("times in thousandths and scores in",
                                     "hundredths, rounded; tau in",
                                     "thousandths"),
                     survival_call = s_call),
                list(Dhat = dhat,
                     survival_concordance = unname(fit$concordance),
                     survival_var = fit$var)
            )
    }
}

out <- list(
    generator = "scripts/reference/reference_r_concordance.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_concordance.json"))
cat("wrote", file.path(data_dir, "r_concordance.json"), "\n")
