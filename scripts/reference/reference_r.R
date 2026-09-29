# Reference results from R for surpyval/tests/reference (#379).
#
# Reads the shared fixtures (surpyval/tests/reference/data/fixtures.json,
# written by make_fixtures.py) and writes one strict-JSON file per R
# package to the same folder: r_survival.json, r_cmprsk.json,
# r_timereg.json, r_pec.json, r_riskregression.json, r_npsurv.json and
# r_fitdistrplus.json. Each entry records the software, its version, the
# exact call that produced it (the call string is what is evaluated, so the
# two cannot drift apart), the settings that matter and the values.
#
# Run from the repository root:
#
#     Rscript scripts/reference/reference_r.R
#
# Needs R >= 4.3 and the packages survival, cmprsk, timereg, pec,
# riskRegression, npsurv, fitdistrplus and jsonlite. On Ubuntu 24.04 all
# come from the distribution (apt-get install r-base-core r-cran-survival
# r-cran-cmprsk r-cran-timereg r-cran-pec r-cran-npsurv
# r-cran-fitdistrplus r-cran-jsonlite; pec pulls in riskRegression).
# Re-running reproduces the files byte for byte.

suppressPackageStartupMessages({
    library(survival)
    library(cmprsk)
    library(timereg)
    library(pec)
    library(riskRegression)
    library(npsurv)
    library(fitdistrplus)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

# ---------------------------------------------------------------------------
# The fixtures, as data frames with the columns R's calls use.
# ---------------------------------------------------------------------------
aml_d <- fixture("aml")
ovarian_d <- fixture("ovarian")
lung_d <- fixture("lung")
lung_d$status <- 1 - lung_d$c
lung_cc <- lung_d[!is.na(lung_d$ph_ecog), ]
lung_cc$ecog_group <- pmin(lung_cc$ph_ecog, 2)
heart_d <- fixture("heart")
heart_d$event <- 1 - heart_d$c
pbc_d <- fixture("pbc")
mz_d <- fixture("mettas_zhao")
mz_d$start <- ave(mz_d$x, mz_d$i, FUN = function(v) c(0, head(v, -1)))
mz_d$event <- 1 - mz_d$c
ties_d <- fixture("ties")
ties_d$status <- 1 - ties_d$c
lt_d <- fixture("left_truncation")
lt_d$status <- 1 - lt_d$c
iv_d <- fixture("interval")
# Surv(type = "interval2"): NA on the left is left censored, NA on the
# right is right censored, equal ends are exact.
iv_d$l2 <- ifelse(iv_d$left == 0, NA, iv_d$left)
iv_d$r2 <- iv_d$right
iv_d$L <- iv_d$left
iv_d$R <- ifelse(is.na(iv_d$right), Inf, iv_d$right)
cr_d <- fixture("competing")
add_d <- fixture("additive")
pt <- fx$prediction_ties
pt_d <- as.data.frame(pt$columns)
pt_d$status <- 1 - pt_d$c
pt_S <- pt$survival
pc <- fx$prediction_continuous
pc_d <- as.data.frame(pc$columns)
pc_d$status <- 1 - pc_d$c
pc_S <- pc$survival

# The classic data sets must be R's own copies.
stopifnot(
    isTRUE(all.equal(aml_d$time, survival::aml$time)),
    isTRUE(all.equal(aml_d$status, survival::aml$status)),
    isTRUE(all.equal(aml_d$maintained,
                     as.numeric(survival::aml$x == "Maintained"))),
    isTRUE(all.equal(ovarian_d$futime, survival::ovarian$futime)),
    isTRUE(all.equal(ovarian_d$fustat, survival::ovarian$fustat)),
    isTRUE(all.equal(ovarian_d$age, survival::ovarian$age)),
    isTRUE(all.equal(ovarian_d$rx, survival::ovarian$rx)),
    isTRUE(all.equal(ovarian_d$ecog_ps, survival::ovarian$ecog.ps)),
    isTRUE(all.equal(ovarian_d$resid_ds, survival::ovarian$resid.ds)),
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(lung_d$age, survival::lung$age)),
    isTRUE(all.equal(lung_d$sex, survival::lung$sex)),
    isTRUE(all.equal(lung_d$ph_ecog, survival::lung$ph.ecog)),
    isTRUE(all.equal(heart_d$start, survival::heart$start)),
    isTRUE(all.equal(heart_d$stop, survival::heart$stop)),
    isTRUE(all.equal(heart_d$event, survival::heart$event)),
    isTRUE(all.equal(heart_d$age, survival::heart$age)),
    isTRUE(all.equal(heart_d$year, survival::heart$year)),
    isTRUE(all.equal(heart_d$surgery, survival::heart$surgery)),
    isTRUE(all.equal(heart_d$transplant,
                     as.numeric(as.character(survival::heart$transplant))))
)

# ---------------------------------------------------------------------------
# Recording.
# ---------------------------------------------------------------------------
store <- new.env()
env <- environment()

finite_or_na <- function(v) {
    if (is.list(v)) return(lapply(v, finite_or_na))
    if (is.numeric(v)) {
        v <- unname(v)
        v[!is.finite(v)] <- NA
        # A matrix goes out row by row.
        if (is.matrix(v)) return(lapply(seq_len(nrow(v)), function(k) v[k, ]))
    }
    v
}

# Evaluate `call` (a string) and record `extract(result)` as the values.
record <- function(file, id, fixture, package, call, settings, extract,
                   note = NULL) {
    result <- eval(parse(text = call), envir = env)
    entry <- list(
        fixture = fixture,
        software = paste("R", package),
        version = as.character(packageVersion(package)),
        call = call,
        settings = settings,
        values = finite_or_na(extract(result))
    )
    if (!is.null(note)) entry$note <- note
    if (is.null(store[[file]])) store[[file]] <- list()
    current <- store[[file]]
    current[[id]] <- entry
    store[[file]] <- current
    invisible(result)
}

# ---------------------------------------------------------------------------
# survival: Kaplan-Meier with Greenwood, Nelson-Aalen, RMST, median.
# ---------------------------------------------------------------------------
km_values <- function(f) {
    tab <- summary(f)$table
    list(
        time = f$time, n_risk = f$n.risk, n_event = f$n.event,
        n_censor = f$n.censor, surv = f$surv,
        std_err = f$std.err, lower = f$lower, upper = f$upper,
        cumhaz = f$cumhaz, std_chaz = f$std.chaz,
        median = tab[["median"]], median_lower = tab[["0.95LCL"]],
        median_upper = tab[["0.95UCL"]]
    )
}
km_settings <- list(
    conf.type = "log-log", conf.int = 0.95,
    std_err = "standard error of -log S (Greenwood), not of S"
)
record("survival", "km_lung", "lung", "survival",
       "survfit(Surv(time, status) ~ 1, data = lung_d, conf.type = 'log-log')",
       km_settings, km_values)
record("survival", "km_aml_maintained", "aml", "survival",
       paste("survfit(Surv(time, status) ~ 1, data = aml_d,",
             "subset = maintained == 1, conf.type = 'log-log')"),
       km_settings, km_values)
record("survival", "km_ties", "ties", "survival",
       "survfit(Surv(x, status) ~ 1, data = ties_d, conf.type = 'log-log')",
       km_settings, km_values,
       note = "events tie with censorings; R counts a censoring at t as at risk at t")
record("survival", "km_left_truncation", "left_truncation", "survival",
       paste("survfit(Surv(tl, x, status) ~ 1, data = lt_d,",
             "conf.type = 'log-log')"),
       km_settings, km_values)

na_settings <- list(ctype = 1, note = "Nelson-Aalen cumulative hazard with the Aalen (Poisson) variance")
record("survival", "na_lung", "lung", "survival",
       "survfit(Surv(time, status) ~ 1, data = lung_d, ctype = 1)",
       na_settings, km_values)
record("survival", "na_ties", "ties", "survival",
       "survfit(Surv(x, status) ~ 1, data = ties_d, ctype = 1)",
       na_settings, km_values)
record("survival", "na_left_truncation", "left_truncation", "survival",
       "survfit(Surv(tl, x, status) ~ 1, data = lt_d, ctype = 1)",
       na_settings, km_values)
record("survival", "fh_ties", "ties", "survival",
       "survfit(Surv(x, status) ~ 1, data = ties_d, ctype = 2)",
       list(ctype = 2, note = "Fleming-Harrington tie correction: d tied events add 1/r + 1/(r-1) + ... + 1/(r-d+1)"),
       km_values)

rmst_values <- function(tau) function(f) {
    tab <- summary(f, rmean = tau)$table
    list(tau = tau, rmean = tab[["rmean"]], se_rmean = tab[["se(rmean)"]],
         median = tab[["median"]])
}
record("survival", "rmst_lung_500", "lung", "survival",
       "survfit(Surv(time, status) ~ 1, data = lung_d)",
       list(summary = "summary(fit, rmean = 500)$table"), rmst_values(500))
record("survival", "rmst_lung_1000", "lung", "survival",
       "survfit(Surv(time, status) ~ 1, data = lung_d)",
       list(summary = "summary(fit, rmean = 1000)$table"), rmst_values(1000))
record("survival", "rmst_ties_15", "ties", "survival",
       "survfit(Surv(x, status) ~ 1, data = ties_d)",
       list(summary = "summary(fit, rmean = 15)$table"), rmst_values(15))

# Turnbull (EM) through survfit; npsurv below is the converged NPMLE.
record("survival", "turnbull_interval", "interval", "survival",
       "survfit(Surv(l2, r2, type = 'interval2') ~ 1, data = iv_d)",
       list(type = "interval2", note = "survfit's Turnbull EM, reported at the midpoints of its intervals"),
       function(f) list(time = f$time, surv = f$surv))

# ---------------------------------------------------------------------------
# survival: log-rank tests.
# ---------------------------------------------------------------------------
survdiff_values <- function(s) list(
    chisq = s$chisq, df = length(s$n) - 1,
    p = pchisq(s$chisq, length(s$n) - 1, lower.tail = FALSE),
    observed = if (is.matrix(s$obs)) rowSums(s$obs) else s$obs,
    expected = if (is.matrix(s$exp)) rowSums(s$exp) else s$exp
)
record("survival", "logrank_aml", "aml", "survival",
       "survdiff(Surv(time, status) ~ maintained, data = aml_d)",
       list(rho = 0), survdiff_values)
record("survival", "logrank_lung_sex", "lung", "survival",
       "survdiff(Surv(time, status) ~ sex, data = lung_d)",
       list(rho = 0), survdiff_values)
record("survival", "logrank_lung_sex_rho1", "lung", "survival",
       "survdiff(Surv(time, status) ~ sex, data = lung_d, rho = 1)",
       list(rho = 1, note = "Peto-Peto weights S(t-)^rho from the pooled Kaplan-Meier"),
       survdiff_values)
record("survival", "logrank_lung_ecog", "lung", "survival",
       "survdiff(Surv(time, status) ~ ph_ecog, data = lung_cc)",
       list(rho = 0, note = "four groups; the one patient with ph.ecog missing is dropped"),
       survdiff_values)
record("survival", "logrank_lung_sex_strata", "lung", "survival",
       "survdiff(Surv(time, status) ~ sex + strata(ecog_group), data = lung_cc)",
       list(rho = 0, strata = "pmin(ph.ecog, 2); the missing ph.ecog dropped"),
       survdiff_values)
record("survival", "logrank_ties", "ties", "survival",
       "survdiff(Surv(x, status) ~ z1, data = ties_d)",
       list(rho = 0), survdiff_values)

# ---------------------------------------------------------------------------
# survival: Cox proportional hazards.
# ---------------------------------------------------------------------------
# The baseline estimator that goes with the tie method, as survfit.coxph
# chooses by default: Efron's (ctype = 2) for an Efron fit, Breslow's
# (ctype = 1) otherwise. It is passed explicitly so the choice is recorded.
baseline_ctype <- function(fit) if (fit$method == "efron") 2 else 1
cox_values <- function(zero) function(fit) {
    base <- survfit(fit, newdata = zero, ctype = baseline_ctype(fit),
                    se.fit = FALSE)
    list(
        coef = coef(fit), se = sqrt(diag(vcov(fit))),
        loglik = fit$loglik, n = fit$n, nevent = fit$nevent,
        baseline_time = base$time, baseline_cumhaz = base$cumhaz
    )
}
cox_settings <- function(ties, extra = list()) c(
    list(ties = ties,
         baseline = paste("survfit(fit, newdata = <all covariates 0>,",
                          if (ties == "efron") "ctype = 2): the Efron" else
                              "ctype = 1): the Breslow",
                          "estimator of the uncentred baseline at the",
                          "fitted coefficients")),
    extra
)
lung_zero <- data.frame(age = 0, sex = 0, ph_ecog = 0)
for (ties in c("breslow", "efron")) {
    record("survival", paste0("cox_lung_", ties), "lung", "survival",
           sprintf(paste("coxph(Surv(time, status) ~ age + sex + ph_ecog,",
                         "data = lung_d, ties = '%s')"), ties),
           cox_settings(ties, list(missing = "the one row with ph.ecog missing is dropped (na.omit)")),
           cox_values(lung_zero))
}
ties_zero <- data.frame(z1 = 0, z2 = 0)
for (ties in c("breslow", "efron", "exact")) {
    record("survival", paste0("cox_ties_", ties), "ties", "survival",
           sprintf("coxph(Surv(x, status) ~ z1 + z2, data = ties_d, ties = '%s')",
                   ties),
           cox_settings(ties), cox_values(ties_zero),
           note = if (ties == "exact") "R's exact is the discrete (conditional logistic) likelihood" else NULL)
}
lt_zero <- data.frame(z = 0)
for (ties in c("breslow", "efron")) {
    record("survival", paste0("cox_left_truncation_", ties),
           "left_truncation", "survival",
           sprintf("coxph(Surv(tl, x, status) ~ z, data = lt_d, ties = '%s')",
                   ties),
           cox_settings(ties), cox_values(lt_zero))
}
heart_zero <- data.frame(age = 0, year = 0, surgery = 0, transplant = 0)
for (ties in c("breslow", "efron")) {
    record("survival", paste0("cox_heart_", ties), "heart", "survival",
           sprintf(paste("coxph(Surv(start, stop, event) ~ age + year +",
                         "surgery + transplant, data = heart_d,",
                         "ties = '%s')"), ties),
           cox_settings(ties, list(form = "start-stop (counting process)")),
           cox_values(heart_zero))
}
strata_values <- function(fit) {
    out <- list(coef = coef(fit), se = sqrt(diag(vcov(fit))),
                loglik = fit$loglik)
    base <- survfit(fit, newdata = data.frame(age = 0, ph_ecog = 0),
                    ctype = baseline_ctype(fit), se.fit = FALSE)
    s <- summary(base, censored = TRUE)
    for (k in c(1, 2)) {
        keep <- s$strata == paste0("sex=", k)
        out[[paste0("baseline_time_sex", k)]] <- s$time[keep]
        out[[paste0("baseline_cumhaz_sex", k)]] <- s$cumhaz[keep]
    }
    out
}
for (ties in c("breslow", "efron")) {
    record("survival", paste0("cox_lung_strata_", ties), "lung", "survival",
           sprintf(paste("coxph(Surv(time, status) ~ age + ph_ecog +",
                         "strata(sex), data = lung_d, ties = '%s')"), ties),
           cox_settings(ties, list(strata = "sex")), strata_values)
}

# ---------------------------------------------------------------------------
# survival: parametric AFT regression and intercept-only fits (survreg).
# survreg writes log T = X b + scale * W, so for SurPyval's Weibull
# alpha = exp(b0), beta = 1 / scale and an AFT coefficient is -b.
# ---------------------------------------------------------------------------
survreg_values <- function(fit) list(
    coef = coef(fit), scale = fit$scale, loglik = fit$loglik[2],
    var = fit$var
)
survreg_settings <- function(dist) list(
    dist = dist,
    parameterisation = "log T = X coef + scale * W; var is for (coef, log(scale))"
)
for (dist in c("weibull", "lognormal", "loglogistic", "exponential")) {
    record("survival", paste0("survreg_ovarian_", dist), "ovarian",
           "survival",
           sprintf(paste("survreg(Surv(futime, fustat) ~ ecog_ps + rx,",
                         "data = ovarian_d, dist = '%s')"), dist),
           survreg_settings(dist), survreg_values)
    record("survival", paste0("survreg_lung_", dist), "lung", "survival",
           sprintf(paste("survreg(Surv(time, status) ~ age + sex,",
                         "data = lung_d, dist = '%s')"), dist),
           survreg_settings(dist), survreg_values)
    record("survival", paste0("survreg_lung_null_", dist), "lung",
           "survival",
           sprintf("survreg(Surv(time, status) ~ 1, data = lung_d, dist = '%s')",
                   dist),
           survreg_settings(dist), survreg_values)
    record("survival", paste0("survreg_interval_", dist), "interval",
           "survival",
           sprintf(paste("survreg(Surv(l2, r2, type = 'interval2') ~ z,",
                         "data = iv_d, dist = '%s')"), dist),
           survreg_settings(dist), survreg_values)
    record("survival", paste0("survreg_interval_null_", dist), "interval",
           "survival",
           sprintf(paste("survreg(Surv(l2, r2, type = 'interval2') ~ 1,",
                         "data = iv_d, dist = '%s')"), dist),
           survreg_settings(dist), survreg_values)
}

# ---------------------------------------------------------------------------
# survival: mean cumulative function of recurrent events.
# ---------------------------------------------------------------------------
record("survival", "mcf_mettas_zhao", "mettas_zhao", "survival",
       "survfit(Surv(start, x, event) ~ 1, data = mz_d, id = i)",
       list(note = "Nelson-Aalen on the gap intervals; with id the std.chaz is the robust (infinitesimal jackknife) error, which is Lawless-Nadeau's"),
       function(f) list(time = f$time, n_risk = f$n.risk,
                        n_event = f$n.event, cumhaz = f$cumhaz,
                        std_chaz = f$std.chaz))

# ---------------------------------------------------------------------------
# cmprsk: Aalen-Johansen cumulative incidence, Gray's test, Fine-Gray.
# ---------------------------------------------------------------------------
cuminc_values <- function(times) function(ci) {
    out <- list()
    tests <- ci$Tests
    if (!is.null(tests)) {
        out$gray_stat <- tests[, "stat"]
        out$gray_p <- tests[, "pv"]
        out$gray_df <- tests[, "df"]
        out$gray_cause <- as.numeric(rownames(tests))
    }
    tp <- timepoints(ci[names(ci) != "Tests"], times)
    out$times <- times
    out$curves <- rownames(tp$est)
    out$est <- tp$est
    out$var <- tp$var
    out
}
record("cmprsk", "cuminc_competing", "competing", "cmprsk",
       "cuminc(cr_d$x, cr_d$cause, cr_d$group, cencode = 0)",
       list(cencode = 0, rho = 0, curves = "'<group> <cause>'"),
       cuminc_values(c(2, 5, 10, 15, 20)))
record("cmprsk", "cuminc_competing_rho1", "competing", "cmprsk",
       "cuminc(cr_d$x, cr_d$cause, cr_d$group, cencode = 0, rho = 1)",
       list(cencode = 0, rho = 1), cuminc_values(c(5, 10)))
record("cmprsk", "cuminc_competing_pooled", "competing", "cmprsk",
       "cuminc(cr_d$x, cr_d$cause, cencode = 0)",
       list(cencode = 0, curves = "'1 <cause>' (one group)"),
       cuminc_values(c(2, 5, 10, 15, 20)))
record("cmprsk", "cuminc_pbc", "pbc", "cmprsk",
       "cuminc(pbc_d$years, pbc_d$cause, pbc_d$drug, cencode = 0)",
       list(cencode = 0, rho = 0, groups = "drug (1 D-penicillamine)"),
       cuminc_values(c(1, 3, 5, 10)))

crr_values <- function(z_new, times) function(fit) {
    pred <- predict(fit, cov1 = z_new)
    keep <- findInterval(times, pred[, 1])
    list(
        coef = fit$coef, var_sandwich = fit$var, var_naive = fit$invinf,
        loglik = fit$loglik, times = times, z_new = z_new,
        cif = t(sapply(keep, function(k) if (k == 0) rep(0, nrow(z_new)) else pred[k, -1]))
    )
}
cr_new <- rbind(c(0, 0), c(1, 0.5))
for (cause in c(1, 2)) {
    record("cmprsk", paste0("crr_competing_cause", cause), "competing",
           "cmprsk",
           sprintf(paste("crr(cr_d$x, cr_d$cause, cbind(cr_d$group, cr_d$z),",
                         "failcode = %d, cencode = 0)"), cause),
           list(covariates = c("group", "z"),
                var_naive = "fit$invinf, the inverse information",
                var_sandwich = "fit$var, Fine and Gray's robust variance"),
           crr_values(cr_new, c(2, 5, 10, 20)))
}
pbc_new <- rbind(c(0, 50, 0, 1), c(1, 40, 1, 0))
record("cmprsk", "crr_pbc_death", "pbc", "cmprsk",
       paste("crr(pbc_d$years, pbc_d$cause,",
             "cbind(pbc_d$drug, pbc_d$age, pbc_d$log_bili, pbc_d$female),",
             "failcode = 2, cencode = 0)"),
       list(covariates = c("drug", "age", "log_bili", "female")),
       crr_values(pbc_new, c(2, 5, 10)))

# ---------------------------------------------------------------------------
# timereg: Lin-Ying additive hazards (all covariates constant).
# ---------------------------------------------------------------------------
aalen_values <- function(fit) list(
    gamma = fit$gamma[, 1], var_gamma = fit$var.gamma,
    robvar_gamma = fit$robvar.gamma,
    cum_time = fit$cum[, 1], cum_baseline = fit$cum[, 2]
)
record("timereg", "aalen_additive", "additive", "timereg",
       paste("aalen(Surv(x, c == 0) ~ const(z1) + const(z2),",
             "data = add_d, robust = 1)"),
       list(note = "semiparametric additive model with only the intercept time-varying: Lin and Ying's estimator; var.gamma is the Lin-Ying sandwich"),
       aalen_values)

# ---------------------------------------------------------------------------
# pec and riskRegression: Brier score and time-dependent AUC.
# ---------------------------------------------------------------------------
pec_values <- function(p) list(times = p$time, brier = p$AppErr$m)
for (nm in c("ties", "continuous")) {
    d_name <- if (nm == "ties") "pt_d" else "pc_d"
    s_name <- if (nm == "ties") "pt_S" else "pc_S"
    fx_name <- paste0("prediction_", nm)
    record("pec", paste0("brier_", nm), fx_name, "pec",
           sprintf(paste("pec(list(m = %s), formula = Surv(x, status) ~ 1,",
                         "data = %s, times = %s, start = NULL,",
                         "exact = FALSE, cens.model = 'marginal',",
                         "reference = FALSE,",
                         "verbose = FALSE)"),
                   s_name, d_name, paste(deparse(fx[[fx_name]]$times), collapse = "")),
           list(cens.model = "marginal (reverse Kaplan-Meier)",
                weights = "events 1/G(x_i-), survivors 1/G(t)"),
           pec_values)
    record("riskregression", paste0("score_", nm), fx_name,
           "riskRegression",
           sprintf(paste("Score(list(m = 1 - %s), formula = Surv(x, status) ~ 1,",
                         "data = %s, times = %s, metrics = c('auc', 'brier'),",
                         "cens.model = 'km', null.model = FALSE,",
                         "conf.int = FALSE)"),
                   s_name, d_name, paste(deparse(fx[[fx_name]]$times), collapse = "")),
           list(risk = "1 - predicted survival", cens.model = "km"),
           function(sc) list(times = sc$AUC$score$times,
                             auc = sc$AUC$score$AUC,
                             brier = sc$Brier$score$Brier))
}

# ---------------------------------------------------------------------------
# npsurv: the Turnbull NPMLE by the constrained Newton method.
# ---------------------------------------------------------------------------
record("npsurv", "npmle_interval", "interval", "npsurv",
       "npsurv(data.frame(L = iv_d$L, R = iv_d$R))",
       list(method = "cnm", intervals = "(L, R], L == R exact, R = Inf right censored"),
       function(f) list(left = f$f$left, right = f$f$right, p = f$f$p,
                        loglik = f$ll, maxgrad = f$maxgrad))

# ---------------------------------------------------------------------------
# fitdistrplus: censored maximum likelihood for other families.
# ---------------------------------------------------------------------------
iv_cens <- data.frame(left = iv_d$l2, right = iv_d$r2)
lung_cens <- data.frame(left = lung_d$time,
                        right = ifelse(lung_d$status == 1, lung_d$time, NA))
fd_values <- function(f) list(estimate = f$estimate, loglik = f$loglik)
for (spec in list(list("gamma", "iv_cens", "interval"),
                  list("gamma", "lung_cens", "lung"),
                  list("weibull", "iv_cens", "interval"),
                  list("lnorm", "iv_cens", "interval"),
                  list("norm", "iv_cens", "interval"),
                  list("logis", "iv_cens", "interval"))) {
    start <- if (spec[[1]] == "gamma" && spec[[3]] == "lung")
        ", start = list(shape = 1.5, rate = 0.005)" else ""
    record("fitdistrplus",
           paste0("fitdistcens_", spec[[3]], "_", spec[[1]]), spec[[3]],
           "fitdistrplus",
           sprintf("fitdistcens(%s, '%s'%s)", spec[[2]], spec[[1]], start),
           list(method = "mle", censdata = "left NA left censored, right NA right censored"),
           fd_values)
}

# ---------------------------------------------------------------------------
# Write.
# ---------------------------------------------------------------------------
for (file in ls(store)) {
    out <- list(
        generator = "scripts/reference/reference_r.R",
        R = R.version.string,
        references = store[[file]]
    )
    text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
                   pretty = TRUE, null = "null")
    writeLines(text, file.path(data_dir, paste0("r_", file, ".json")))
    cat("wrote", file.path(data_dir, paste0("r_", file, ".json")), "\n")
}
