# Reference results from R for Cox models with time-varying coefficients,
# beta(t) = beta + gamma g(t): coxph(..., tt = function(x, t, ...) x * g(t)).
#
# Writes surpyval/tests/reference/data/r_cox_tt.json, with the software,
# its version, the call and the values of each entry, as reference_r.R
# does for the other files. Generated with R 4.6.2 and survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_cox_tt.R
#
# Run make_fixtures.py first. Needs survival and jsonlite (Ubuntu:
# r-cran-survival r-cran-jsonlite). Re-running reproduces the file byte
# for byte.
#
# The fits (CoxPH.fit(..., time_interaction=) in SurPyval):
#
# * veteran: karno + tt(karno) with g(t) = t and g(t) = log(t), under
#   Efron and Breslow ties, and trt + prior + karno + tt(karno) with
#   g(t) = log(t + 20), the example of the survival package's vignette on
#   time-dependent covariates;
# * lung (the row with ph.ecog missing dropped): age + ph.ecog +
#   tt(ph.ecog) with g(t) = log(t), Efron and Breslow, and the same with
#   strata(sex);
# * veteran with weights: the survSplit form with weights w and the
#   robust variance clustered by patient (coxph's tt() does not take
#   weights with a cluster), the weights 1, 1.5, 2 in turn by row.
#
# Every tt() fit is checked here against the survSplit form, the rows cut
# at every event time and the interaction an ordinary covariate,
# coxph(Surv(tstart, time, status) ~ z + z_g), to 1e-8 in the
# coefficients and the covariance. What is recorded: the coefficients,
# their covariance, the log partial likelihood at the fit, and beta(t)
# with its standard error at a few times (from coef and vcov by hand:
# sqrt(a' V a), a = (1, g(t)) at the covariate and its interaction). For
# veteran karno * t (Efron) the cumulative hazard of a patient with karno
# = 60 at every event time, from survfit() of the survSplit fit along
# that patient's split rows (survfit does not take a tt() model).

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
vet <- as.data.frame(fx$veteran$columns)
lung_d <- as.data.frame(fx$lung$columns)
lung_d$status <- 1 - lung_d$c
lung_cc <- lung_d[!is.na(lung_d$ph_ecog), ]
stopifnot(
    isTRUE(all.equal(vet$time, survival::veteran$time)),
    isTRUE(all.equal(vet$status, survival::veteran$status)),
    isTRUE(all.equal(vet$karno, survival::veteran$karno)),
    isTRUE(all.equal(vet$trt, survival::veteran$trt)),
    isTRUE(all.equal(vet$prior, survival::veteran$prior)),
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(lung_d$ph_ecog, survival::lung$ph.ecog))
)

g_funs <- list(
    t = function(t) t,
    log = function(t) log(t),
    log_t_plus_20 = function(t) log(t + 20)
)

entry <- function(fixture, call, settings, values) list(
    fixture = fixture,
    software = "R survival",
    version = as.character(packageVersion("survival")),
    call = call,
    settings = settings,
    values = values
)

matrix_rows <- function(m) {
    m <- unname(as.matrix(m))
    lapply(seq_len(nrow(m)), function(k) m[k, ])
}

# beta(t) of the covariate at position j of the coefficients and its
# interaction at position k, with its standard error.
beta_t <- function(fit, j, k, g, times) {
    b <- coef(fit)
    v <- vcov(fit)
    gt <- g(times)
    est <- b[j] + b[k] * gt
    se <- sqrt(v[j, j] + gt^2 * v[k, k] + 2 * gt * v[j, k])
    list(times = times, beta = unname(est), se = unname(se))
}

# The survSplit form of a tt() fit: the rows cut at every event time and
# the interaction column zg = z * g(stop).
split_rows <- function(data, z, g) {
    cut <- sort(unique(data$time[data$status == 1]))
    sp <- survSplit(Surv(time, status) ~ ., data = data, cut = cut,
                    episode = "episode_")
    sp$zg <- sp[[z]] * g(sp$time)
    sp
}

split_fit <- function(data, rhs, z, g, ties, strata = NULL) {
    sp <- split_rows(data, z, g)
    form <- paste("Surv(tstart, time, status) ~", rhs, "+ zg")
    if (!is.null(strata)) form <- paste0(form, " + strata(", strata, ")")
    list(fit = coxph(as.formula(form), data = sp, ties = ties), data = sp)
}

same <- function(a, b) {
    stopifnot(isTRUE(all.equal(unname(coef(a)), unname(coef(b)),
                               tolerance = 1e-8)))
    stopifnot(isTRUE(all.equal(unname(vcov(a)), unname(vcov(b)),
                               tolerance = 1e-8)))
}

refs <- list()
vet_times <- c(10, 50, 100, 200, 400)

for (g_name in c("t", "log")) {
    for (ties in c("efron", "breslow")) {
        if (g_name == "log" && ties == "breslow") next
        g <- g_funs[[g_name]]
        tt_fun <- function(x, t, ...) x * g(t)
        fit <- coxph(Surv(time, status) ~ karno + tt(karno), data = vet,
                     tt = tt_fun, ties = ties)
        split <- split_fit(vet, "karno", "karno", g, ties)
        same(fit, split$fit)
        bt <- beta_t(fit, 1, 2, g, vet_times)
        v <- list(coef = unname(coef(fit)), vcov = matrix_rows(vcov(fit)),
                  loglik = unname(fit$loglik[2]),
                  beta_t_times = bt$times, beta_t = bt$beta,
                  beta_t_se = bt$se)
        if (g_name == "t" && ties == "efron") {
            # The cumulative hazard of a patient with karno = 60 along the
            # split rows, at every event time.
            cut <- sort(unique(vet$time[vet$status == 1]))
            nd <- data.frame(tstart = c(0, head(cut, -1)), time = cut,
                             status = 0, karno = 60, zg = 60 * cut, id = 1)
            sf <- survfit(split$fit, newdata = nd, id = id)
            v$Hf_karno60_times <- sf$time
            v$Hf_karno60 <- sf$cumhaz
        }
        call <- sprintf(paste("coxph(Surv(time, status) ~ karno +",
                              "tt(karno), data = veteran, tt = function(x,",
                              "t, ...) x * %s, ties = '%s')"),
                        if (g_name == "t") "t" else "log(t)", ties)
        refs[[sprintf("veteran_karno_%s_%s", g_name, ties)]] <- entry(
            "veteran", call,
            list(ties = ties, g = g_name,
                 split_form = paste("survSplit at every event time, then",
                                    "coxph(Surv(tstart, time, status) ~",
                                    "karno + karno:g(time)): equal to",
                                    "1e-8"),
                 beta_t = "coef[1] + coef[2] g(t), se sqrt(a' V a)",
                 Hf_karno60 = paste("survfit(split fit, newdata = the",
                                    "split rows of karno = 60, id = id)",
                                    "$cumhaz")),
            v
        )
    }
}

# The vignette's example: g(t) = log(t + 20), with other covariates.
g <- g_funs$log_t_plus_20
fit <- coxph(Surv(time, status) ~ trt + prior + karno + tt(karno),
             data = vet, tt = function(x, t, ...) x * log(t + 20))
split <- split_fit(vet, "trt + prior + karno", "karno", g, "efron")
same(fit, split$fit)
bt <- beta_t(fit, 3, 4, g, vet_times)
refs[["veteran_trt_prior_karno_log_t20_efron"]] <- entry(
    "veteran",
    paste("coxph(Surv(time, status) ~ trt + prior + karno + tt(karno),",
          "data = veteran, tt = function(x, t, ...) x * log(t + 20))"),
    list(ties = "efron", g = "log(t + 20)"),
    list(coef = unname(coef(fit)), vcov = matrix_rows(vcov(fit)),
         loglik = unname(fit$loglik[2]), beta_t_times = bt$times,
         beta_t = bt$beta, beta_t_se = bt$se)
)

lung_times <- c(30, 100, 300, 600)
for (ties in c("efron", "breslow")) {
    for (stratified in c(FALSE, TRUE)) {
        if (stratified && ties == "breslow") next
        g <- g_funs$log
        form <- Surv(time, status) ~ age + ph_ecog + tt(ph_ecog)
        if (stratified) {
            form <- Surv(time, status) ~ age + ph_ecog + tt(ph_ecog) +
                strata(sex)
        }
        fit <- coxph(form, data = lung_cc, ties = ties,
                     tt = function(x, t, ...) x * log(t))
        split <- split_fit(lung_cc, "age + ph_ecog", "ph_ecog", g, ties,
                           strata = if (stratified) "sex" else NULL)
        same(fit, split$fit)
        bt <- beta_t(fit, 2, 3, g, lung_times)
        name <- sprintf("lung_ph_ecog_log_%s%s", ties,
                        if (stratified) "_strata_sex" else "")
        refs[[name]] <- entry(
            "lung",
            sprintf(paste("coxph(Surv(time, status) ~ age + ph_ecog +",
                          "tt(ph_ecog)%s, data = lung_cc, tt = function(x,",
                          "t, ...) x * log(t), ties = '%s')"),
                    if (stratified) " + strata(sex)" else "", ties),
            list(ties = ties, g = "log",
                 strata = if (stratified) "sex" else "none",
                 missing = "the one row with ph.ecog missing is dropped"),
            list(coef = unname(coef(fit)), vcov = matrix_rows(vcov(fit)),
                 loglik = unname(fit$loglik[2]), beta_t_times = bt$times,
                 beta_t = bt$beta, beta_t_se = bt$se)
        )
    }
}

# Weights: the survSplit form, the robust variance clustered by patient.
vet_w <- vet
vet_w$id <- seq_len(nrow(vet_w))
vet_w$w <- 1 + ((vet_w$id - 1) %% 3) / 2
sp_w <- split_rows(vet_w, "karno", g_funs$t)
fit <- coxph(Surv(tstart, time, status) ~ karno + zg, data = sp_w,
             weights = w, cluster = id, ties = "efron")
refs[["veteran_karno_t_efron_weighted"]] <- entry(
    "veteran",
    paste("coxph(Surv(tstart, time, status) ~ karno + zg, data =",
          "survSplit(veteran, cut = event times), weights = w, cluster =",
          "id, ties = 'efron'), zg = karno * time"),
    list(ties = "efron", g = "t",
         weights = "1 + ((row - 1) %% 3) / 2, row = 1, 2, ... in data order",
         variance = "robust, clustered by patient (row)"),
    list(coef = unname(coef(fit)), vcov = matrix_rows(vcov(fit)),
         w = vet_w$w)
)

out <- list(
    generator = "scripts/reference/reference_r_cox_tt.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_cox_tt.json"))
cat("wrote", file.path(data_dir, "r_cox_tt.json"), "\n")
