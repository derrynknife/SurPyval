# Reference results from R for the diagnostics of the parametric
# proportional hazards and accelerated failure time models (#869).
#
# Writes surpyval/tests/reference/data/r_ph_aft.json, with the software,
# its version, the call and the values of each entry, as reference_r.R
# does for the other files. Every fit is to a fixture already in
# fixtures.json (lung, veteran, left_truncation and interval), so
# make_fixtures.py needs no change.
#
#     Rscript scripts/reference/reference_r_ph_aft.R
#
# Needs survival and jsonlite (Ubuntu: r-cran-survival r-cran-jsonlite).
# Re-running reproduces the file byte for byte.
#
# What each check is compared with:
#
# * The likelihood-ratio tests of a quadratic term (functional_form) and
#   of a shape per level (common_shape): survreg fits with and without the
#   term, and with and without strata(<factor>), which gives each level
#   its own scale (1 / shape for the Weibull, sigma for the log-normal);
#   the statistic is twice the difference of the log-likelihoods. A
#   Weibull AFT model is a Weibull PH model, so the Weibull values serve
#   both families.
# * The likelihood-ratio test of a step in a coefficient (and in the
#   baseline) at the median failure time (proportional_hazards): survival
#   has no parametric fit
#   with a time-varying covariate, so it is computed independently of
#   survreg. A Weibull PH model of shape k is a Poisson model of the
#   events of the follow-up split at the cut (survSplit), with offset
#   log(stop^k - start^k), whose log-likelihood differs from the Weibull's
#   by sum over the events of log k + (k - 1) log t - log(t^k - start^k)
#   (the piece's offset), so the Weibull PH likelihood is that sum plus
#   glm's, maximised over k with optimize(); the exponential is k = 1.
#   Delayed entry is the first piece's start. Both models have a step in
#   the baseline at the cut (`late`, the indicator of the second piece),
#   as SurPyval's test does, so that the test does not depend on where
#   each covariate's origin is.
# * The influence screen: residuals(survreg(...), type = "dfbeta"), the
#   one-step change of (intercept, coefficients, log scale); the test maps
#   it to the PH coefficients -coef / scale.
# * The standardised and Cox-Snell residuals: w =
#   (log(time) - linear predictor) / scale, and -log S(time) from it.
# * The AIC comparison: AIC(survreg(...)) of the Weibull, exponential and
#   log-normal fits.

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

lung_d <- fixture("lung")
lung_d$status <- 1 - lung_d$c
lung_cc <- lung_d[!is.na(lung_d$ph_ecog), ]
vet_d <- fixture("veteran")
lt_d <- fixture("left_truncation")
lt_d$status <- 1 - lt_d$c
iv_d <- fixture("interval")
iv_d$l2 <- ifelse(iv_d$left == 0, NA, iv_d$left)
iv_d$r2 <- iv_d$right
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(vet_d$time, survival::veteran$time)),
    isTRUE(all.equal(vet_d$celltype, as.integer(survival::veteran$celltype)))
)

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
# Evaluate `call` (a string) and record `extract(result)`.
record <- function(id, fixture, call, settings, extract) {
    result <- eval(parse(text = call), envir = env)
    refs[[id]] <<- list(
        fixture = fixture,
        software = "R survival",
        version = as.character(packageVersion("survival")),
        call = call,
        settings = settings,
        values = finite_or_na(extract(result))
    )
    invisible(result)
}

lr <- function(small, big) 2 * (as.numeric(logLik(big)) -
                                as.numeric(logLik(small)))

# --- survreg: quadratic terms, a shape per level, AIC, residuals ----------
survreg_values <- function(squares, strata, residuals = FALSE) function(fit) {
    out <- list(
        loglik = as.numeric(logLik(fit)),
        aic = AIC(fit),
        coef = unname(coef(fit)),
        scale = unname(fit$scale)
    )
    for (v in squares) {
        g <- update(fit, as.formula(sprintf(". ~ . + I(%s^2)", v)))
        out[[paste0("square_", v)]] <- lr(fit, g)
    }
    for (v in strata) {
        g <- update(fit, as.formula(sprintf(". ~ . + strata(%s)", v)))
        out[[paste0("shape_", v)]] <- lr(fit, g)
        out[[paste0("shape_", v, "_scales")]] <- unname(g$scale)
    }
    if (residuals) {
        # fit$y holds the times as given (not their logs)
        lp <- fit$linear.predictors
        w <- (log(fit$y[, 1]) - lp) / fit$scale
        # -log S(time): the extreme-value survival exp(-exp(w)) of the
        # Weibull, the normal one of the log-normal
        out$cox_snell <- if (fit$dist == "weibull") exp(w) else
            -pnorm(w, lower.tail = FALSE, log.p = TRUE)
        out$standardised <- w
        out$dfbeta <- residuals(fit, type = "dfbeta")
    }
    out
}
settings <- function(dist, extra = list()) c(
    list(dist = dist,
         squares = "2 * (logLik(update(fit, . ~ . + I(v^2))) - logLik(fit))",
         shapes = "2 * (logLik(update(fit, . ~ . + strata(v))) - logLik(fit))"),
    extra
)
lung_missing <- list(missing = "the one row with ph.ecog missing is dropped")
# survreg's default tolerance (1e-9 on the log-likelihood) leaves the
# coefficients, and so the dfbeta, good to about 1e-5; every fit here
# (and its update()s, which keep the control) converges further.
tight <- survreg.control(rel.tolerance = 1e-13, maxiter = 200)
for (dist in c("weibull", "lognormal")) {
    record(paste0("lung_", dist), "lung",
           sprintf(paste("survreg(Surv(time, status) ~ age + sex + ph_ecog,",
                         "data = lung_cc, dist = '%s', control = tight)"),
                   dist),
           settings(dist, lung_missing),
           survreg_values(c("age", "ph_ecog"), "sex", residuals = TRUE))
    record(paste0("veteran_", dist), "veteran",
           sprintf(paste("survreg(Surv(time, status) ~ factor(celltype) +",
                         "karno, data = vet_d, dist = '%s',",
                         "control = tight)"), dist),
           settings(dist), survreg_values("karno", "celltype"))
    record(paste0("interval_", dist), "interval",
           sprintf(paste("survreg(Surv(l2, r2, type = 'interval2') ~ z,",
                         "data = iv_d, dist = '%s', control = tight)"),
                   dist),
           settings(dist, list(censoring = "interval2: NA left is left censored, NA right is right censored")),
           survreg_values(character(0), "z"))
}
record("lung_exponential", "lung",
       paste("survreg(Surv(time, status) ~ age + sex + ph_ecog,",
             "data = lung_cc, dist = 'exponential', control = tight)"),
       settings("exponential", lung_missing),
       survreg_values(character(0), character(0)))

# --- the step in a coefficient at the median failure time -----------------
# The lower median of the failure times: the first at which half of them
# are at or before it.
lower_median <- function(t) {
    t <- sort(t)
    t[which(seq_along(t) / length(t) >= 0.5 - 1e-12)[1]]
}
# The maximised Weibull (or, with shape fixed at 1, exponential) PH
# log-likelihood of the split data `s` with covariates `terms`.
weibull_ph_loglik <- function(s, terms, shape = NULL) {
    ll <- function(k) {
        s$off <- log(s$stop^k - s$start^k)
        f <- as.formula(paste("status ~", paste(terms, collapse = " + "),
                              "+ offset(off)"))
        g <- glm(f, family = poisson, data = s,
                 control = glm.control(epsilon = 1e-14, maxit = 100))
        ev <- s$status == 1
        as.numeric(logLik(g)) + sum(log(k) + (k - 1) * log(s$stop[ev]) -
                                    s$off[ev])
    }
    if (!is.null(shape)) return(list(loglik = ll(shape), shape = shape))
    best <- optimize(ll, c(0.2, 10), maximum = TRUE, tol = 1e-12)
    list(loglik = best$objective, shape = best$maximum)
}
step_values <- function(d, covariates, shape = NULL) {
    tau <- lower_median(d$stop[d$status == 1])
    s <- survSplit(Surv(start, stop, status) ~ ., data = d, cut = tau,
                   episode = "period")
    s$late <- as.numeric(s$period == 2)
    late <- character(0)
    for (v in covariates) {
        s[[paste0(v, "_late")]] <- s[[v]] * (s$period == 2)
        late <- c(late, paste0(v, "_late"))
    }
    # The fitted model (which the split leaves as it was), then the null
    # of the tests: it with the baseline's step.
    fitted <- weibull_ph_loglik(s, covariates, shape)
    null <- weibull_ph_loglik(s, c(covariates, "late"), shape)
    out <- list(tau = tau, loglik_fitted = fitted$loglik,
                shape_fitted = fitted$shape, loglik = null$loglik,
                shape = null$shape)
    for (k in seq_along(covariates)) {
        alt <- weibull_ph_loglik(s, c(covariates, "late", late[k]), shape)
        out[[paste0("step_", covariates[k])]] <- 2 * (alt$loglik - null$loglik)
    }
    alt <- weibull_ph_loglik(s, c(covariates, "late", late), shape)
    out$step_global <- 2 * (alt$loglik - null$loglik)
    out
}
lung_split <- data.frame(start = 0, stop = lung_cc$time,
                         status = lung_cc$status, age = lung_cc$age,
                         sex = lung_cc$sex, ph_ecog = lung_cc$ph_ecog)
lt_split <- data.frame(start = lt_d$tl, stop = lt_d$x, status = lt_d$status,
                       z = lt_d$z)
step_settings <- function(extra = list()) c(
    list(cut = "the lower median of the failure times",
         null = "the fitted model with a step in the baseline at the cut",
         method = paste("survSplit at the cut; glm(status ~ terms +",
                        "offset(log(stop^k - start^k)), poisson) plus",
                        "sum over events of log k + (k - 1) log t - offset,",
                        "maximised over the Weibull shape k by optimize",
                        "(k = 1 for the exponential)")),
    extra
)
record("step_lung_weibull", "lung",
       "step_values(lung_split, c('age', 'sex', 'ph_ecog'))",
       step_settings(lung_missing), identity)
record("step_lung_exponential", "lung",
       "step_values(lung_split, c('age', 'sex', 'ph_ecog'), shape = 1)",
       step_settings(lung_missing), identity)
record("step_left_truncation_weibull", "left_truncation",
       "step_values(lt_split, 'z')",
       step_settings(list(form = "delayed entry: each unit's first piece starts at tl")),
       identity)

out <- list(
    generator = "scripts/reference/reference_r_ph_aft.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_ph_aft.json"))
cat("wrote", file.path(data_dir, "r_ph_aft.json"), "\n")
