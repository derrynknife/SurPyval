# Reference results from R for the diagnostics of the shared-frailty
# models (#872).
#
# Writes surpyval/tests/reference/data/r_frailty_diag.json, with the
# software, its version, the call and the values of each entry, as
# reference_r.R does for the other files. Every fit is to the kidney
# catheter data (the kidney fixture, R's survival::kidney), grouped by
# patient, with the covariates age and female = (sex == 2).
#
#     Rscript scripts/reference/reference_r_frailty_diag.R
#
# Run make_fixtures.py first. Needs survival, lme4 and jsonlite (Ubuntu:
# r-cran-survival r-cran-lme4 r-cran-jsonlite); frailtypack and parfm are
# not used (not installed here). Re-running reproduces the file byte for
# byte.
#
# What each check is compared with:
#
# * The test of no frailty (frailty_variance): twice the frailty fit's
#   log-likelihood less that of the model without frailty, with the
#   p-value of the 50:50 mixture of chi2(0) and chi2(1),
#   pchisq(T, 1, lower.tail = FALSE) / 2. For the Cox model the frailty
#   fit is coxph(... + frailty(id, dist = 'gamma')), with theta maximising
#   R's I-likelihood (history[[1]]$c.loglik) to 1e-10 as in
#   reference_r_frailty.R, and the null coxph(...)$loglik[2]. For the
#   Weibull and exponential baselines the null is survreg(...)$loglik[2].
# * Log-normal frailty, parametric baseline: lme4's glmer, as in
#   reference_r_frailty.R. A Weibull PH model of shape k with a normal
#   random intercept on the log hazard is a Poisson GLMM of the events
#   with offset log(stop^k - start^k) (start 0 without a split), whose
#   log-likelihood differs from the Weibull's by sum over the events of
#   log k + (k - 1) log t - log(t^k - start^k); glmer's logLik with
#   nAGQ > 1 is measured from the saturated model, -sum(status). The
#   full log-likelihood is maximised over k with optimize().
# * The step test of proportional hazards given the frailty
#   (proportional_hazards): the follow-up split at the cut tau, the lower
#   median of the failure times (survSplit), with a step `late` in the
#   log baseline after it under both models and `late:z` the step in z's
#   coefficient; for the log-normal frailty a Poisson GLMM of the pieces
#   with the patient's random intercept (the pieces of a patient share
#   its frailty).
# * The quadratic term (functional_form): the model with (age / 10)^2 added
#   (the model with age^2, scaled so that glmer converges to 1e-6).
# * Gamma frailty, Weibull baseline: no R package fits it here, so its
#   marginal likelihood is written out in closed form below (the
#   frailty integrated out of each patient's likelihood) and maximised
#   with optim() (BFGS, Nelder-Mead and BFGS again) and nlminb(); the
#   same larger models as for the log-normal.
# * The family and baseline screens (frailty_family,
#   baseline_distribution): the AIC, 2 k - 2 loglik, of each fit.

suppressPackageStartupMessages({
    library(survival)
    library(lme4)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!dir.exists(data_dir)) stop("Run from the repository root")

fx <- fromJSON(file.path(data_dir, "fixtures.json"))
k <- as.data.frame(fx$kidney$columns)
k$status <- 1 - k$c
stopifnot(
    isTRUE(all.equal(k$time, survival::kidney$time)),
    isTRUE(all.equal(k$status, survival::kidney$status)),
    isTRUE(all.equal(k$id, survival::kidney$id)),
    isTRUE(all.equal(k$age, survival::kidney$age)),
    isTRUE(all.equal(k$female, as.numeric(survival::kidney$sex == 2)))
)

refs <- list()
entry <- function(package, call, settings, values) list(
    fixture = "kidney",
    software = paste("R", package),
    version = as.character(packageVersion(package)),
    call = call,
    settings = settings,
    values = values
)
boundary_p <- function(stat) {
    if (stat <= 1e-7) 1 else pchisq(stat, 1, lower.tail = FALSE) / 2
}

# The cut: the lower median of the failure times.
events <- sort(k$time[k$status == 1])
tau <- events[ceiling(length(events) / 2)]
split <- survSplit(Surv(time, status) ~ ., data = k, cut = tau,
                   start = "tstart", end = "tstop",
                   episode = "piece")
split$late <- as.numeric(split$piece == 2)

# --- gamma frailty, Cox baseline (survival::coxph) -------------------------
ctl <- coxph.control(eps = 1e-10, toler.chol = 1e-13, iter.max = 500,
                     outer.max = 50)
fixed_theta <- function(theta, ties) {
    coxph(Surv(time, status) ~ age + female +
              frailty(id, dist = "gamma", theta = theta, sparse = FALSE),
          data = k, ties = ties, control = ctl)
}
for (ties in c("efron", "breslow")) {
    best <- optimize(
        function(lt) fixed_theta(exp(lt), ties)$history[[1]]$c.loglik,
        c(log(0.05), log(3)), maximum = TRUE, tol = 1e-10)
    fit <- suppressWarnings(fixed_theta(exp(best$maximum), ties))
    cox <- coxph(Surv(time, status) ~ age + female, data = k, ties = ties)
    ll <- fit$history[[1]]$c.loglik
    stat <- 2 * (ll - cox$loglik[2])
    refs[[paste0("kidney_cox_gamma_", ties)]] <- entry(
        "survival",
        paste0("coxph(Surv(time, status) ~ age + female + frailty(id, ",
               "dist = 'gamma', theta = theta, sparse = FALSE), ties = '",
               ties, "') against coxph(Surv(time, status) ~ age + female)"),
        list(ties = ties,
             loglik = "R's I-likelihood, history[[1]]$c.loglik, at the theta maximising it",
             statistic = "2 (I-likelihood - partial likelihood without frailty)",
             p = "pchisq(statistic, 1, lower.tail = FALSE) / 2"),
        list(theta = exp(best$maximum), loglik = ll,
             loglik_null = cox$loglik[2], statistic = stat,
             p = boundary_p(stat))
    )
}
pkd <- coxph(Surv(time, status) ~ age + female + I(disease == "PKD") +
                 frailty(id, dist = "gamma"),
             data = cbind(k, disease = survival::kidney$disease))
pkd_cox <- coxph(Surv(time, status) ~ age + female + I(disease == "PKD"),
                 data = cbind(k, disease = survival::kidney$disease))
stat <- max(0, 2 * (pkd$history[[1]]$c.loglik - pkd_cox$loglik[2]))
refs[["kidney_cox_gamma_no_frailty"]] <- entry(
    "survival",
    paste("coxph(Surv(time, status) ~ age + female + I(disease == 'PKD')",
          "+ frailty(id, dist = 'gamma')) against the same without frailty"),
    list(ties = "efron", note = "R's theta is 5e-9: no frailty"),
    list(theta = pkd$history[[1]]$theta, statistic = stat,
         p = boundary_p(stat))
)

# --- log-normal frailty, parametric baseline (lme4) -----------------------
gctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
glmm_ll <- function(rhs, d, shape) {
    d$off <- log(d$tstop^shape - d$tstart^shape)
    fit <- glmer(as.formula(paste("status ~", rhs, "+ (1 | id) + offset(off)")),
                 family = poisson, data = d, nAGQ = 25, control = gctrl)
    as.numeric(logLik(fit)) - sum(d$status) +
        sum(d$status * (log(shape) + (shape - 1) * log(d$tstop) - d$off))
}
whole <- k
whole$tstart <- 0
whole$tstop <- k$time
profiled <- function(rhs, d) {
    optimize(function(s) glmm_ll(rhs, d, s), c(0.5, 3), maximum = TRUE,
             tol = 1e-10)$objective
}
weibull_null <- survreg(Surv(time, status) ~ age + female, data = k,
                        dist = "weibull")$loglik[2]
exponential_null <- survreg(Surv(time, status) ~ age + female, data = k,
                            dist = "exponential")$loglik[2]
ln <- list(
    base = profiled("age + female", whole),
    exponential = glmm_ll("age + female", whole, 1),
    step = profiled("age + female + late", split),
    step_age = profiled("age + female + late + late:age", split),
    step_female = profiled("age + female + late + late:female", split),
    step_global = profiled("age + female + late + late:age + late:female",
                           split),
    square_age = profiled("age + female + I((age / 10)^2)", whole)
)
lrt <- function(fits, null_ll) {
    stat <- 2 * (fits$base - null_ll)
    list(
        tau = tau,
        loglik = fits$base,
        loglik_null = null_ll,
        statistic = stat,
        p = boundary_p(stat),
        step_age = 2 * (fits$step_age - fits$step),
        step_female = 2 * (fits$step_female - fits$step),
        step_global = 2 * (fits$step_global - fits$step),
        square_age = 2 * (fits$square_age - fits$base),
        aic = 2 * 5 - 2 * fits$base
    )
}
refs[["kidney_lognormal_weibull"]] <- entry(
    "lme4",
    paste("glmer(status ~ <terms> + (1 | id) + offset(log(stop^shape -",
          "start^shape)), family = poisson, nAGQ = 25), profiled over shape;",
          "null survreg(Surv(time, status) ~ age + female, dist = 'weibull')"),
    list(model = "Weibull PH baseline, log-normal frailty u = exp(w), w ~ N(0, theta)",
         step = paste("the split at tau with `late` under both models;",
                      "step_z = 2 (loglik with late:z - loglik with late)"),
         square_age = paste("2 (loglik with I((age / 10)^2) - loglik",
                            "without): the model with I(age^2), scaled",
                            "(unscaled, glmer stops 0.004 short)"),
         aic = "2 * 5 - 2 loglik (alpha, beta, two coefficients, theta)"),
    lrt(ln, weibull_null)
)
stat <- 2 * (ln$exponential - exponential_null)
refs[["kidney_lognormal_exponential"]] <- entry(
    "lme4",
    paste("glmer(status ~ age + female + (1 | id) + offset(log(time)),",
          "family = poisson, nAGQ = 25); null survreg(..., dist =",
          "'exponential')"),
    list(model = "exponential baseline, log-normal frailty",
         aic = "2 * 4 - 2 loglik"),
    list(loglik = ln$exponential, loglik_null = exponential_null,
         statistic = stat, p = boundary_p(stat),
         aic = 2 * 4 - 2 * ln$exponential)
)

# --- gamma frailty, Weibull baseline (closed form, optim) -----------------
# The Weibull PH model given the frailty u: hazard u h0(t) exp(eta), with
# h0(t) = beta t^(beta - 1) / alpha^beta (SurPyval's Weibull) and, with a
# split, the pieces' cumulative hazards. A patient with D events and
# cumulative hazard H (summed over its pieces) contributes
# log E[u^D exp(-u H)] = lgamma(D + 1/theta) - lgamma(1/theta)
#     - D log(1/theta) - (D + 1/theta) log(1 + theta H).
gamma_nll <- function(par, d, X) {
    a <- exp(par[1]); b <- exp(par[2]); theta <- exp(par[length(par)])
    eta <- drop(X %*% par[3:(length(par) - 1)])
    H <- exp(eta) * ((d$tstop / a)^b - (d$tstart / a)^b)
    log_h <- log(b) + (b - 1) * log(d$tstop) - b * log(a) + eta
    D <- tapply(d$status, d$id, sum)
    Hg <- tapply(H, d$id, sum)
    nu <- 1 / theta
    group <- lgamma(D + nu) - lgamma(nu) - D * log(nu) -
        (D + nu) * log1p(theta * Hg)
    -(sum(d$status * log_h) + sum(group))
}
gamma_fit <- function(rhs, d) {
    X <- model.matrix(as.formula(paste("~", rhs)), d)[, -1, drop = FALSE]
    start <- c(log(100), 0, rep(0, ncol(X)), log(0.5))
    f <- function(p) gamma_nll(p, d, X)
    ctrl <- list(maxit = 10000, reltol = 1e-14)
    res <- optim(start, f, method = "BFGS", control = ctrl)
    res <- optim(res$par, f, method = "Nelder-Mead", control = ctrl)
    res <- optim(res$par, f, method = "BFGS", control = ctrl)
    # PORT's quasi-Newton search to finish, which gets the last digits
    polish <- nlminb(res$par, f, control = list(
        rel.tol = 1e-15, x.tol = 1e-12, eval.max = 1e4, iter.max = 1e4))
    if (polish$objective < res$value) {
        res$value <- polish$objective
        res$par <- polish$par
    }
    list(loglik = -res$value, theta = exp(res$par[length(res$par)]),
         k = length(res$par))
}
gm <- list(
    base = gamma_fit("age + female", whole),
    step = gamma_fit("age + female + late", split),
    step_age = gamma_fit("age + female + late + late:age", split),
    step_female = gamma_fit("age + female + late + late:female", split),
    step_global = gamma_fit("age + female + late + late:age + late:female",
                            split),
    square_age = gamma_fit("age + female + I((age / 10)^2)", whole)
)
gm_ll <- lapply(gm, function(f) f$loglik)
values <- lrt(gm_ll, weibull_null)
values$theta <- gm$base$theta
refs[["kidney_gamma_weibull"]] <- entry(
    "stats",
    paste("optim() of the closed-form gamma-frailty Weibull PH marginal",
          "likelihood (see the comment above gamma_nll); null",
          "survreg(Surv(time, status) ~ age + female, dist = 'weibull')"),
    list(model = "Weibull PH baseline, gamma frailty of mean 1, variance theta",
         step = paste("the split at tau with `late` under both models;",
                      "step_z = 2 (loglik with late:z - loglik with late)"),
         square_age = "2 (loglik with I((age / 10)^2) - loglik without)",
         aic = "2 * 5 - 2 loglik"),
    values
)

out <- list(
    generator = "scripts/reference/reference_r_frailty_diag.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_frailty_diag.json"))
cat("wrote", file.path(data_dir, "r_frailty_diag.json"), "\n")
