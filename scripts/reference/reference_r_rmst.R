# Reference results from R for the restricted mean survival time (RMST)
# of the parametric models: Parametric.rmst / rmst_cb / mean(tau=) and
# the parametric regression models' rmst / rmst_cb / rmst_diff at Z.
#
# Writes surpyval/tests/reference/data/r_rmst.json, with the software,
# its version, the call and the values of each entry, as the other
# generators do. Every fit is to a fixture already in fixtures.json (lung
# and ovarian, survival's data sets), so make_fixtures.py needs no change.
#
#     Rscript scripts/reference/reference_r_rmst.R
#
# Needs survival and jsonlite; flexsurv, where installed, adds its own
# fit and RMST (summary(fit, type = "rmst", ci = FALSE)) as a cross-check
# of the point values (its intervals are simulated, so they are not
# stored). Run with R 4.6.2, survival 3.8.12 and flexsurv 2.3.2.
# Re-running reproduces the file byte for byte.
#
# What each value is:
#
# * The fit: survreg (Weibull, exponential, log-normal, log-logistic) at
#   tight tolerances; the Gamma, which survreg does not have, by optim on
#   its censored log-likelihood (shape and rate on the log scale, to a
#   relative tolerance of 1e-14), with optimHess for the information.
# * The RMST: integrate() of the fitted survival function from 0 to tau
#   (rel.tol 1e-12), at the R estimate.
# * Its standard error: the delta method, sqrt(g' V g), with V the
#   inverse observed information (vcov(survreg)) and g the gradient of the
#   integral in the same parameters, by central differences (step 1e-5,
#   relative). The delta-method standard error does not depend on the
#   parameterisation, so it is the same quantity SurPyval computes in its
#   own parameters.
# * The interval: the Wald interval on logit(rmst / tau), the one
#   SurPyval's rmst() gives: tau * expit(logit(mu / tau) +- z se (1 / mu +
#   1 / (tau - mu))).
# * The regression entries: the same at each covariate value, and the
#   difference of two covariate values with the delta method on the
#   difference (the two RMSTs share the parameters).

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
})
has_flexsurv <- requireNamespace("flexsurv", quietly = TRUE)

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

lung_d <- fixture("lung")
lung_d$status <- 1 - lung_d$c
ov_d <- fixture("ovarian")
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(ov_d$futime, survival::ovarian$futime))
)

tight <- survreg.control(rel.tolerance = 1e-13, maxiter = 200)
z <- qnorm(0.975)
expit <- function(u) 1 / (1 + exp(-u))
taus <- c(180, 365, 730)

rmst_of <- function(sf, tau) {
    integrate(sf, 0, tau, rel.tol = 1e-12, abs.tol = 0,
              subdivisions = 1000L)$value
}

gradient <- function(f, p) {
    sapply(seq_along(p), function(j) {
        h <- 1e-5 * max(1, abs(p[j]))
        up <- p; dn <- p
        up[j] <- up[j] + h; dn[j] <- dn[j] - h
        (f(up) - f(dn)) / (2 * h)
    })
}

summarise <- function(sf_at, p, V, tau) {
    mu <- rmst_of(sf_at(p), tau)
    g <- gradient(function(q) rmst_of(sf_at(q), tau), p)
    se <- sqrt(drop(t(g) %*% V %*% g))
    psi <- log(mu) - log(tau - mu)
    se_psi <- se * (1 / mu + 1 / (tau - mu))
    c(rmst = mu, se = se, lower = tau * expit(psi - z * se_psi),
      upper = tau * expit(psi + z * se_psi))
}

# The survival function of a survreg location-scale fit, at the linear
# predictor lp, as a function of the parameters (coefficients, then
# log(scale) unless the scale is fixed at 1).
survreg_sf <- function(dist, x_row, fixed_scale = FALSE) {
    function(p) {
        k <- length(x_row)
        lp <- sum(p[seq_len(k)] * x_row)
        sc <- if (fixed_scale) 1 else exp(p[k + 1])
        function(t) {
            w <- (log(t) - lp) / sc
            switch(dist,
                weibull = exp(-exp(w)),
                exponential = exp(-exp(w)),
                lognormal = pnorm(-w),
                loglogistic = 1 / (1 + exp(w)))
        }
    }
}

refs <- list()
add <- function(id, entry) refs[[id]] <<- entry

# Univariate fits to lung -------------------------------------------------
for (dist in c("weibull", "exponential", "lognormal", "loglogistic")) {
    fit <- survreg(Surv(time, status) ~ 1, data = lung_d, dist = dist,
                   control = tight)
    fixed <- dist == "exponential"
    p <- if (fixed) coef(fit) else c(coef(fit), log(fit$scale))
    V <- vcov(fit)
    sf_at <- survreg_sf(dist, 1, fixed)
    vals <- t(sapply(taus, function(tau) summarise(sf_at, p, V, tau)))
    entry <- list(
        fixture = "lung", software = "R survival",
        version = as.character(packageVersion("survival")),
        call = sprintf(paste0("survreg(Surv(time, status) ~ 1, data = lung, ",
                              "dist = '%s'); integrate(sf, 0, tau)"), dist),
        settings = list(dist = dist, tau = taus,
                        se = "delta method, central differences",
                        interval = "Wald on logit(rmst / tau), 95%"),
        values = list(
            lp = unname(coef(fit)), scale = fit$scale,
            loglik = fit$loglik[1],
            rmst = unname(vals[, "rmst"]), se = unname(vals[, "se"]),
            lower = unname(vals[, "lower"]), upper = unname(vals[, "upper"])
        )
    )
    if (has_flexsurv) {
        fdist <- if (dist == "loglogistic") "llogis" else
            if (dist == "lognormal") "lnorm" else
            if (dist == "exponential") "exp" else dist
        ffit <- flexsurv::flexsurvreg(Surv(time, status) ~ 1, data = lung_d,
                                      dist = fdist)
        fs <- summary(ffit, type = "rmst", t = taus, ci = FALSE)[[1]]
        entry$values$flexsurv_rmst <- unname(fs$est)
        entry$values$flexsurv_loglik <- ffit$loglik
        entry$values$flexsurv_params <- as.list(ffit$res[, "est"])
        entry$flexsurv_version <- as.character(packageVersion("flexsurv"))
    }
    add(paste0("lung_", dist), entry)
}

# The Gamma, by hand: shape a and rate b on the log scale.
gamma_nll <- function(q) {
    a <- exp(q[1]); b <- exp(q[2])
    ev <- lung_d$status == 1
    -(sum(dgamma(lung_d$time[ev], a, b, log = TRUE)) +
      sum(pgamma(lung_d$time[!ev], a, b, lower.tail = FALSE, log.p = TRUE)))
}
opt <- optim(c(log(1.5), log(0.005)), gamma_nll, method = "BFGS",
             control = list(reltol = 1e-14, maxit = 1000))
opt <- optim(opt$par, gamma_nll, method = "Nelder-Mead",
             control = list(reltol = 1e-15, maxit = 5000))
opt <- optim(opt$par, gamma_nll, method = "BFGS",
             control = list(reltol = 1e-15, maxit = 1000))
Vg <- solve(optimHess(opt$par, gamma_nll))
gamma_sf <- function(q) function(t) pgamma(t, exp(q[1]), exp(q[2]),
                                            lower.tail = FALSE)
vals <- t(sapply(taus, function(tau) summarise(gamma_sf, opt$par, Vg, tau)))
entry <- list(
    fixture = "lung", software = "R stats",
    version = R.version.string,
    call = paste0("optim on the censored Gamma log-likelihood (log shape, ",
                  "log rate); optimHess; integrate(sf, 0, tau)"),
    settings = list(dist = "gamma", tau = taus,
                    se = "delta method, central differences",
                    interval = "Wald on logit(rmst / tau), 95%"),
    values = list(
        shape = exp(opt$par[1]), rate = exp(opt$par[2]),
        loglik = -opt$value,
        rmst = unname(vals[, "rmst"]), se = unname(vals[, "se"]),
        lower = unname(vals[, "lower"]), upper = unname(vals[, "upper"])
    )
)
if (has_flexsurv) {
    ffit <- flexsurv::flexsurvreg(Surv(time, status) ~ 1, data = lung_d,
                                  dist = "gamma")
    fs <- summary(ffit, type = "rmst", t = taus, ci = FALSE)[[1]]
    entry$values$flexsurv_rmst <- unname(fs$est)
    entry$values$flexsurv_loglik <- ffit$loglik
    entry$values$flexsurv_params <- as.list(ffit$res[, "est"])
    entry$flexsurv_version <- as.character(packageVersion("flexsurv"))
}
add("lung_gamma", entry)

# Regression: an AFT model at two covariate values, and their difference --
regression <- function(id, data, formula, time, dist, zs, tau, fixture) {
    fit <- survreg(formula, data = data, dist = dist, control = tight)
    p <- c(coef(fit), log(fit$scale))
    V <- vcov(fit)
    rows <- lapply(zs, function(zv) c(1, zv))
    each <- t(sapply(rows, function(r) summarise(survreg_sf(dist, r), p, V,
                                                  tau)))
    diff_of <- function(q) {
        rmst_of(survreg_sf(dist, rows[[1]])(q), tau) -
            rmst_of(survreg_sf(dist, rows[[2]])(q), tau)
    }
    d <- diff_of(p)
    g <- gradient(diff_of, p)
    se_d <- sqrt(drop(t(g) %*% V %*% g))
    entry <- list(
        fixture = fixture, software = "R survival",
        version = as.character(packageVersion("survival")),
        call = sprintf("survreg(%s, dist = '%s'); integrate(sf(t | z), 0, tau)",
                       deparse(formula), dist),
        settings = list(dist = dist, tau = tau, z = zs,
                        se = "delta method, central differences",
                        interval = "Wald on logit(rmst / tau), 95%",
                        difference = "rmst(z[1]) - rmst(z[2]), delta method"),
        values = list(
            coef = unname(coef(fit)), scale = fit$scale,
            loglik = fit$loglik[2],
            rmst = unname(each[, "rmst"]), se = unname(each[, "se"]),
            lower = unname(each[, "lower"]), upper = unname(each[, "upper"]),
            difference = d, difference_se = se_d
        )
    )
    if (has_flexsurv) {
        fdist <- if (dist == "lognormal") "lnorm" else dist
        ffit <- flexsurv::flexsurvreg(formula, data = data, dist = fdist)
        nd <- data.frame(zs)
        names(nd) <- all.vars(formula)[3]
        fs <- summary(ffit, type = "rmst", t = tau, newdata = nd, ci = FALSE)
        entry$values$flexsurv_rmst <- unname(sapply(fs, function(s) s$est))
        entry$values$flexsurv_loglik <- ffit$loglik
        entry$flexsurv_version <- as.character(packageVersion("flexsurv"))
    }
    add(id, entry)
}
regression("lung_weibull_sex", lung_d, Surv(time, status) ~ sex, "time",
           "weibull", c(1, 2), 365, "lung")
regression("ovarian_lognormal_rx", ov_d, Surv(futime, fustat) ~ rx,
           "futime", "lognormal", c(1, 2), 730, "ovarian")

out <- list(
    generator = "scripts/reference/reference_r_rmst.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_rmst.json"))
cat("wrote", file.path(data_dir, "r_rmst.json"), "\n")
