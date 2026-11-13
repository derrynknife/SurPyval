# Reference results from R for the confidence bounds on a Cox model's
# predicted survival curve at given covariates.
#
# Writes surpyval/tests/reference/data/r_cox_curve.json, with the
# software, its version, the call and the values of each entry, as
# reference_r.R does for the other files. Generated with R 4.6.2 and
# survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_cox_curve.R
#
# Run make_fixtures.py first. Needs survival and jsonlite (Ubuntu:
# r-cran-survival r-cran-jsonlite). Re-running reproduces the file byte
# for byte.
#
# Each entry is survfit(fit, newdata = nd) for one coxph fit, with R's
# defaults: the survival exp(-H) (stype = 2), the cumulative hazard
# Breslow's under Breslow ties and Efron's (Fleming-Harrington) under
# Efron ties (ctype from the fit), and the standard error (std.err) of
# the cumulative hazard, the Tsiatis (1981) variance of the baseline plus
# the coefficients' covariance (fit$var; the robust one for a weighted
# fit). Recorded at the curve's times (censor = TRUE; 30 of them, evenly
# spaced in the list of times, the first and last included, where there
# are more): the survival, the cumulative hazard, std.err, and the lower
# and upper 95% bounds for conf.type "log" (R's default), "log-log" and
# "plain", and the 90% "log" bounds. One list per row of newdata.
#
# The fits:
#
# * lung: Surv(time, status) ~ age + sex, Efron and Breslow ties;
# * lung with case weights w = 0.5 + (row %% 5) / 4 (robust = TRUE),
#   Efron and Breslow ties;
# * lung: ~ age + ph_ecog + strata(sex), Efron ties, the one row with
#   ph.ecog missing dropped (a curve per row of newdata, each in its
#   row's stratum);
# * ovarian: Surv(futime, fustat) ~ age + ecog_ps, Efron ties;
# * heart (start-stop): Surv(start, stop, event) ~ age + year +
#   transplant, Efron ties.

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
lung_d$w <- 0.5 + (seq_len(nrow(lung_d)) %% 5) / 4
lung_cc <- lung_d[!is.na(lung_d$ph_ecog), ]
ovarian_d <- as.data.frame(fx$ovarian$columns)
heart_d <- as.data.frame(fx$heart$columns)
heart_d$event <- 1 - heart_d$c
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(lung_d$age, survival::lung$age)),
    isTRUE(all.equal(lung_d$sex, survival::lung$sex)),
    isTRUE(all.equal(lung_d$ph_ecog, survival::lung$ph.ecog)),
    isTRUE(all.equal(ovarian_d$futime, survival::ovarian$futime)),
    isTRUE(all.equal(ovarian_d$fustat, survival::ovarian$fustat)),
    isTRUE(all.equal(ovarian_d$age, survival::ovarian$age)),
    isTRUE(all.equal(ovarian_d$ecog_ps, survival::ovarian$ecog.ps)),
    isTRUE(all.equal(heart_d$start, survival::heart$start)),
    isTRUE(all.equal(heart_d$stop, survival::heart$stop)),
    isTRUE(all.equal(heart_d$event, survival::heart$event)),
    isTRUE(all.equal(heart_d$age, survival::heart$age))
)

env <- environment()
refs <- list()

thin <- function(n) {
    # The positions recorded: all, or 30 evenly spaced
    if (n <= 30) seq_len(n) else unique(round(seq(1, n, length.out = 30)))
}

columns <- function(m, k) {
    # One list per row of newdata (a column of R's matrices), at the
    # recorded times
    m <- unname(as.matrix(m))
    keep <- thin(nrow(m))
    lapply(seq_len(k), function(j) m[keep, j])
}

curve_values <- function(fit, nd, stratified) {
    k <- nrow(nd)
    one <- function(conf.type, conf.int = 0.95) {
        survfit(fit, newdata = nd, conf.type = conf.type, conf.int = conf.int)
    }
    s <- one("log")
    if (stratified) {
        # survfit puts each row's curve, in its own stratum, one after
        # the other: split them
        idx <- split(seq_along(s$time), rep(seq_len(k), s$strata))
        idx <- lapply(idx, function(i) i[thin(length(i))])
        pick <- function(v) unname(lapply(idx, function(i) unname(v[i])))
        out <- list(time = pick(s$time), surv = pick(s$surv),
                    cumhaz = pick(s$cumhaz), std_err = pick(s$std.err))
        for (ct in c("log", "log-log", "plain")) {
            f <- one(ct)
            key <- gsub("-", "_", ct)
            out[[paste0("lower_", key)]] <- pick(f$lower)
            out[[paste0("upper_", key)]] <- pick(f$upper)
        }
        f <- one("log", 0.9)
        out$lower_log_90 <- pick(f$lower)
        out$upper_log_90 <- pick(f$upper)
        return(out)
    }
    out <- list(time = unname(s$time)[thin(length(s$time))], surv = columns(s$surv, k),
                cumhaz = columns(s$cumhaz, k),
                std_err = columns(s$std.err, k))
    for (ct in c("log", "log-log", "plain")) {
        f <- one(ct)
        key <- gsub("-", "_", ct)
        out[[paste0("lower_", key)]] <- columns(f$lower, k)
        out[[paste0("upper_", key)]] <- columns(f$upper, k)
    }
    f <- one("log", 0.9)
    out$lower_log_90 <- columns(f$lower, k)
    out$upper_log_90 <- columns(f$upper, k)
    out
}

entry <- function(fixture, call, newdata, settings, values) list(
    fixture = fixture,
    software = "R survival",
    version = as.character(packageVersion("survival")),
    call = call,
    newdata = newdata,
    settings = settings,
    values = values
)

add <- function(id, fixture, call, nd, settings, stratified = FALSE) {
    fit <- eval(parse(text = call), envir = env)
    v <- curve_values(fit, nd, stratified)
    v$coef <- unname(coef(fit))
    refs[[id]] <<- entry(fixture, call, as.list(nd), settings, v)
}

lung_nd <- data.frame(age = c(60, 72, 45), sex = c(1, 2, 2))
for (ties in c("efron", "breslow")) {
    add(paste0("lung_", ties), "lung",
        sprintf("coxph(Surv(time, status) ~ age + sex, data = lung_d, ties = '%s')",
                ties),
        lung_nd, list(ties = ties))
    add(paste0("lung_weighted_", ties), "lung",
        sprintf(paste("coxph(Surv(time, status) ~ age + sex, data = lung_d,",
                      "weights = w, robust = TRUE, ties = '%s')"), ties),
        lung_nd,
        list(ties = ties, weights = "w = 0.5 + (row %% 5) / 4",
             variance = "robust (fit$var with robust = TRUE)"))
}
add("lung_strata_sex_efron", "lung",
    paste("coxph(Surv(time, status) ~ age + ph_ecog + strata(sex),",
          "data = lung_cc, ties = 'efron')"),
    data.frame(age = c(60, 70), ph_ecog = c(1, 2), sex = c(1, 2)),
    list(ties = "efron", strata = "sex",
         missing = "the one row with ph.ecog missing is dropped"),
    stratified = TRUE)
add("ovarian_efron", "ovarian",
    paste("coxph(Surv(futime, fustat) ~ age + ecog_ps, data = ovarian_d,",
          "ties = 'efron')"),
    data.frame(age = c(50, 65), ecog_ps = c(1, 2)),
    list(ties = "efron"))
add("heart_efron", "heart",
    paste("coxph(Surv(start, stop, event) ~ age + year + transplant,",
          "data = heart_d, ties = 'efron')"),
    data.frame(age = c(-5, 5), year = c(2, 4), transplant = c(0, 1)),
    list(ties = "efron", form = "start-stop (counting process)"))

out <- list(
    generator = "scripts/reference/reference_r_cox_curve.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_cox_curve.json"))
cat("wrote", file.path(data_dir, "r_cox_curve.json"), "\n")
