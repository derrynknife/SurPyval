# Reference results from R for the residuals, the proportional-hazards
# test, the robust standard errors and the concordance of a stratified
# Cox model.
#
# Writes surpyval/tests/reference/data/r_stratified_cox.json, with the
# software, its version, the call and the values of each entry, as
# reference_r.R does for the other files. Generated with R 4.6.2 and
# survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_stratified_cox.R
#
# Run make_fixtures.py first. Needs survival and jsonlite (Ubuntu:
# r-cran-survival r-cran-jsonlite). Re-running reproduces the file byte
# for byte.
#
# The fits:
#
# * lung: coxph(Surv(time, status) ~ age + ph_ecog + strata(sex)), the one
#   row with ph.ecog missing dropped, under Efron and Breslow ties;
# * heart (start-stop): coxph(Surv(start, stop, event) ~ age + year +
#   transplant + strata(surgery)), Efron ties, the robust variance
#   clustered by patient (cluster(id)).
#
# What is recorded: the coefficients; the martingale, deviance, score and
# dfbeta residuals (one per row, in the data's order); the Schoenfeld and
# scaled Schoenfeld residuals (one per event, in R's order: by stratum,
# then time, ties in the data's order) with each one's time and stratum;
# cox.zph's table on the km, rank and identity scales; the robust
# standard errors (robust = TRUE, or cluster(id)); concordance(fit), the
# pairs pooled over the strata, with its counts; and the score test of
# adding age^2 at the fitted coefficients (update(fit, . ~ . + I(age^2),
# init = c(coef(fit), 0), iter.max = 0)$score).

suppressPackageStartupMessages({
    library(survival)
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
heart_d <- as.data.frame(fx$heart$columns)
heart_d$event <- 1 - heart_d$c
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(lung_d$age, survival::lung$age)),
    isTRUE(all.equal(lung_d$sex, survival::lung$sex)),
    isTRUE(all.equal(lung_d$ph_ecog, survival::lung$ph.ecog)),
    isTRUE(all.equal(heart_d$start, survival::heart$start)),
    isTRUE(all.equal(heart_d$stop, survival::heart$stop)),
    isTRUE(all.equal(heart_d$event, survival::heart$event)),
    isTRUE(all.equal(heart_d$id, survival::heart$id)),
    isTRUE(all.equal(heart_d$surgery, survival::heart$surgery))
)

env <- environment()
refs <- list()

matrix_rows <- function(m) {
    m <- unname(as.matrix(m))
    lapply(seq_len(nrow(m)), function(k) m[k, ])
}

values_of <- function(fit, data, strata_col, time_col, robust_fit) {
    out <- list(coef = unname(coef(fit)))
    for (type in c("martingale", "deviance")) {
        out[[type]] <- unname(residuals(fit, type = type))
    }
    for (type in c("score", "dfbeta")) {
        out[[type]] <- matrix_rows(residuals(fit, type = type))
    }
    sch <- residuals(fit, type = "schoenfeld")
    out$schoenfeld <- matrix_rows(sch)
    out$scaled_schoenfeld <- matrix_rows(residuals(fit, type = "scaledsch"))
    # R's order of the events: by stratum, then time (order() is stable,
    # so tied times keep the data's order).
    events <- which(data$status_ == 1)
    ord <- events[order(data[[strata_col]][events], data[[time_col]][events])]
    out$schoenfeld_time <- data[[time_col]][ord]
    out$schoenfeld_stratum <- data[[strata_col]][ord]
    for (tr in c("km", "rank", "identity")) {
        z <- cox.zph(fit, transform = tr)
        out[[paste0("zph_", tr, "_chisq")]] <- unname(z$table[, "chisq"])
        out[[paste0("zph_", tr, "_p")]] <- unname(z$table[, "p"])
    }
    out$robust_se <- unname(sqrt(diag(robust_fit$var)))
    conc <- concordance(fit)
    out$concordance <- unname(conc$concordance)
    out$concordance_count <- matrix_rows(conc$count)
    out
}

entry <- function(fixture, call, settings, values) list(
    fixture = fixture,
    software = "R survival",
    version = as.character(packageVersion("survival")),
    call = call,
    settings = settings,
    values = values
)

for (ties in c("efron", "breslow")) {
    call <- sprintf(paste("coxph(Surv(time, status) ~ age + ph_ecog +",
                          "strata(sex), data = lung_cc, ties = '%s')"), ties)
    fit <- eval(parse(text = call), envir = env)
    robust <- update(fit, robust = TRUE)
    d <- lung_cc
    d$status_ <- d$status
    v <- values_of(fit, d, "sex", "time", robust)
    square <- update(fit, . ~ . + I(age^2), init = c(coef(fit), 0),
                     control = coxph.control(iter.max = 0))
    v$square_age <- unname(square$score)
    refs[[paste0("lung_strata_sex_", ties)]] <- entry(
        "lung", call,
        list(ties = ties, strata = "sex",
             missing = "the one row with ph.ecog missing is dropped",
             robust = "update(fit, robust = TRUE)",
             schoenfeld_order = "by stratum, then time",
             square_age = paste("update(fit, . ~ . + I(age^2), init =",
                                "c(coef(fit), 0), iter.max = 0)$score")),
        v
    )
}

call <- paste("coxph(Surv(start, stop, event) ~ age + year + transplant +",
              "strata(surgery), data = heart_d, ties = 'efron')")
fit <- eval(parse(text = call), envir = env)
robust <- update(fit, . ~ . + cluster(id))
d <- heart_d
d$status_ <- d$event
refs[["heart_strata_surgery_efron"]] <- entry(
    "heart", call,
    list(ties = "efron", strata = "surgery",
         form = "start-stop (counting process)",
         robust = "update(fit, . ~ . + cluster(id))",
         schoenfeld_order = "by stratum, then stop time"),
    values_of(fit, d, "surgery", "stop", robust)
)

out <- list(
    generator = "scripts/reference/reference_r_stratified_cox.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_stratified_cox.json"))
cat("wrote", file.path(data_dir, "r_stratified_cox.json"), "\n")
