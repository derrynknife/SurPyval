# Reference results from R for SurPyval's built-in GeneralizedGamma
# (flexsurv's "gengamma", Prentice's (mu, sigma, Q)) and Gompertz
# (flexsurv's "gompertz", (shape, rate)) and InverseGaussian (statmod's
# and SuppDists' (mean, shape)) and GeneralizedF (flexsurv's "genf",
# Prentice's (mu, sigma, Q, P)).
#
# Writes surpyval/tests/reference/data/r_distributions.json, with the
# software, its version, the call and the values of each entry, as the
# other generators do. The fits are to survival's lung data, the lung
# fixture already in fixtures.json, so make_fixtures.py needs no change.
#
#     Rscript scripts/reference/reference_r_distributions.R
#
# Needs survival, flexsurv, statmod, SuppDists, numDeriv and jsonlite. Run with R 4.6.2, survival 3.8.12
# and flexsurv 2.3.2. Re-running reproduces the file byte for byte.
#
# What each value is:
#
# * The density, distribution function, log survival function and log
#   density (dgengamma, pgengamma) at fixed times, and the quantiles
#   (qgengamma) at fixed probabilities, for parameter sets with both
#   signs of Q, Q near 0 and Q = 0 (the lognormal).
# * The fits: flexsurvreg(Surv(time, status) ~ 1) on lung, right
#   censored, and on lung left truncated at entry = floor(time / 4)
#   (Surv(entry, time, status)), with optim's relative tolerance
#   tightened to 1e-14: the maximised log-likelihood, the estimates and
#   their standard errors. For the Gompertz also ~ sex, which in
#   flexsurv acts on the rate: the proportional hazards model.
# * The inverse Gaussian: statmod's dinvgauss / pinvgauss / qinvgauss, and
#   SuppDists' dinvGauss / pinvGauss at the same points (a second
#   implementation). Neither package fits censored data, so the fits are
#   the censored (and left-truncated) log-likelihood written with
#   statmod's pinvgauss(log.p = TRUE), maximised by optim (Nelder-Mead,
#   then BFGS, reltol = 1e-14), standard errors from numDeriv's Richardson
#   Hessian.

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(statmod)
    library(SuppDists)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
lung_d <- as.data.frame(fx[["lung"]]$columns)
lung_d$status <- 1 - lung_d$c
lung_d$entry <- floor(lung_d$time / 4)
stopifnot(isTRUE(all.equal(lung_d$time, survival::lung$time)))

versions <- list(
    survival = as.character(packageVersion("survival")),
    flexsurv = as.character(packageVersion("flexsurv")),
    statmod = as.character(packageVersion("statmod")),
    SuppDists = as.character(packageVersion("SuppDists"))
)

refs <- list()
add <- function(id, entry) refs[[id]] <<- entry
tight <- list(reltol = 1e-14, maxit = 10000)

functions_entry <- function(dist, call, pars, times, probs, d, p, q) {
    list(
        fixture = "lung", software = "R flexsurv",
        version = versions$flexsurv, call = call,
        settings = list(t = times, p = probs, params = pars,
                        note = "no data: the fixture field is nominal"),
        values = list(
            pdf = lapply(pars, function(a) do.call(d, c(list(times), a))),
            log_pdf = lapply(pars, function(a)
                do.call(d, c(list(times), a, list(log = TRUE)))),
            cdf = lapply(pars, function(a) do.call(p, c(list(times), a))),
            log_sf = lapply(pars, function(a)
                do.call(p, c(list(times), a,
                             list(lower.tail = FALSE, log.p = TRUE)))),
            quantile = lapply(pars, function(a)
                do.call(q, c(list(probs), a)))
        )
    )
}

fit_entry <- function(dist, formula, label, inits = NULL) {
    fit <- if (is.null(inits)) {
        flexsurvreg(formula, data = lung_d, dist = dist, control = tight)
    } else {
        flexsurvreg(formula, data = lung_d, dist = dist, control = tight,
                    inits = inits)
    }
    list(
        fixture = "lung", software = "R flexsurv",
        version = versions$flexsurv,
        call = sprintf(
            "flexsurvreg(%s, data = lung, dist = '%s', control = list(reltol = 1e-14))",
            label, dist),
        settings = list(dist = dist, formula = label,
                        entry = "floor(time / 4) where truncated"),
        values = list(
            loglik = fit$loglik,
            names = rownames(fit$res),
            est = unname(fit$res[, "est"]),
            se = unname(fit$res[, "se"])
        )
    )
}

times <- c(0.05, 0.3, 1, 2, 5, 10, 30, 100)
probs <- c(1e-6, 0.001, 0.1, 0.5, 0.9, 0.999)

# ---- generalised gamma -------------------------------------------------
gg_pars <- list(c(1.5, 0.6, 0.5), c(1.5, 0.6, -0.5), c(0.2, 1.2, 2),
                c(0.2, 1.2, -2), c(2, 0.4, 0.05), c(1, 0.8, 1),
                c(1, 0.8, 0), c(3, 1.5, 0.8))
add("functions_gengamma", functions_entry(
    "gengamma", "dgengamma / pgengamma / qgengamma(., mu, sigma, Q)",
    gg_pars, times, probs, dgengamma, pgengamma, qgengamma))
add("lung_gengamma", fit_entry(
    "gengamma", Surv(time, status) ~ 1, "Surv(time, status) ~ 1"))
add("lung_gengamma_truncated", fit_entry(
    "gengamma", Surv(entry, time, status) ~ 1,
    "Surv(entry, time, status) ~ 1"))

# ---- Gompertz -----------------------------------------------------------
gz_pars <- list(c(0.1, 0.01), c(-0.1, 0.05), c(0, 0.2), c(1e-6, 0.2),
                c(2, 1e-3), c(-2, 0.5))
gz_probs <- c(1e-6, 0.001, 0.1, 0.5, 0.9)
add("functions_gompertz", functions_entry(
    "gompertz", "dgompertz / pgompertz / qgompertz(., shape, rate)",
    gz_pars, times, gz_probs, dgompertz, pgompertz, qgompertz))
# flexsurv's time is in days on lung; in years the shape is not tiny
lung_d$time_y <- lung_d$time / 365.25
lung_d$entry_y <- lung_d$entry / 365.25
add("lung_gompertz", fit_entry(
    "gompertz", Surv(time_y, status) ~ 1, "Surv(time / 365.25, status) ~ 1"))
add("lung_gompertz_truncated", fit_entry(
    "gompertz", Surv(entry_y, time_y, status) ~ 1,
    "Surv(entry / 365.25, time / 365.25, status) ~ 1"))
add("lung_gompertz_sex", fit_entry(
    "gompertz", Surv(time_y, status) ~ sex,
    "Surv(time / 365.25, status) ~ sex"))

# ---- inverse Gaussian ---------------------------------------------------
ig_pars <- list(c(10, 20), c(2, 0.3), c(1, 500), c(5, 1))
ig_times <- c(0.05, 0.3, 1, 2, 5, 10, 30, 100)
ig_probs <- c(1e-6, 0.001, 0.1, 0.5, 0.9, 0.999)
add("functions_invgauss", list(
    fixture = "lung", software = "R statmod",
    version = versions$statmod,
    call = "dinvgauss / pinvgauss / qinvgauss(., mean, shape)",
    settings = list(t = ig_times, p = ig_probs, params = ig_pars,
                    note = "no data: the fixture field is nominal"),
    values = list(
        pdf = lapply(ig_pars, function(a) dinvgauss(ig_times, a[1], a[2])),
        log_pdf = lapply(ig_pars, function(a)
            dinvgauss(ig_times, a[1], a[2], log = TRUE)),
        cdf = lapply(ig_pars, function(a) pinvgauss(ig_times, a[1], a[2])),
        log_sf = lapply(ig_pars, function(a)
            pinvgauss(ig_times, a[1], a[2], lower.tail = FALSE,
                      log.p = TRUE)),
        quantile = lapply(ig_pars, function(a)
            sapply(ig_probs, function(p) qinvgauss(p, a[1], a[2])))
    )
))
add("functions_invgauss_suppdists", list(
    fixture = "lung", software = "R SuppDists",
    version = versions$SuppDists,
    call = "dinvGauss / pinvGauss(., nu, lambda)",
    settings = list(t = ig_times, params = ig_pars,
                    note = "no data: the fixture field is nominal"),
    values = list(
        pdf = lapply(ig_pars, function(a) dinvGauss(ig_times, a[1], a[2])),
        cdf = lapply(ig_pars, function(a) pinvGauss(ig_times, a[1], a[2]))
    )
))
ig_fit <- function(time, status, entry, label) {
    nll <- function(lp) {
        m <- exp(lp[1]); s <- exp(lp[2])
        ll <- sum(ifelse(status == 1,
                         dinvgauss(time, m, s, log = TRUE),
                         pinvgauss(time, m, s, lower.tail = FALSE,
                                   log.p = TRUE)))
        if (!is.null(entry)) {
            ll <- ll - sum(pinvgauss(entry[entry > 0], m, s,
                                     lower.tail = FALSE, log.p = TRUE))
        }
        -ll
    }
    start <- log(c(mean(time), mean(time)))
    f <- optim(start, nll, method = "Nelder-Mead",
               control = list(reltol = 1e-14, maxit = 20000))
    f <- optim(f$par, nll, method = "BFGS",
               control = list(reltol = 1e-14, maxit = 10000))
    est <- exp(f$par)
    nat <- function(p) nll(log(p))
    H <- numDeriv::hessian(nat, est)
    list(
        fixture = "lung", software = "R statmod + optim",
        version = versions$statmod,
        call = sprintf(
            "optim of the censored log-likelihood (%s) with pinvgauss(log.p = TRUE)",
            label),
        settings = list(formula = label,
                        entry = "floor(time / 4) where truncated"),
        values = list(
            loglik = -f$value,
            names = c("mu", "lam"),
            est = est,
            se = sqrt(diag(solve(H)))
        )
    )
}
add("lung_invgauss", ig_fit(lung_d$time, lung_d$status, NULL,
                            "Surv(time, status)"))
add("lung_invgauss_truncated", ig_fit(lung_d$time, lung_d$status,
                                      lung_d$entry,
                                      "Surv(entry, time, status)"))

# ---- generalised F --------------------------------------------------------
gf_pars <- list(c(1.5, 0.6, 0.5, 1), c(1.5, 0.6, -0.5, 0.3),
                c(0.2, 1.2, 2, 0.5), c(1, 0.8, 0, 1), c(2, 0.4, 0.3, 0.01),
                c(1, 0.7, -1.5, 2), c(1, 0.5, 0.5, 1e-4))
add("functions_genf", functions_entry(
    "genf", "dgenf / pgenf / qgenf(., mu, sigma, Q, P)",
    gf_pars, times, probs, dgenf, pgenf, qgenf))
add("lung_genf", fit_entry(
    "genf", Surv(time, status) ~ 1, "Surv(time, status) ~ 1"))
# flexsurvreg cannot fit the left-truncated genf on lung (optim stops on a
# non-finite difference from any start tried), so the truncated
# likelihood is checked at fixed parameters instead: the log-likelihood
# of Surv(entry, time, status) at the untruncated estimates, from dgenf
# and pgenf directly.
gf_est <- refs[["lung_genf"]]$values$est
gf_ll <- with(lung_d, sum(ifelse(
    status == 1,
    dgenf(time, gf_est[1], gf_est[2], gf_est[3], gf_est[4], log = TRUE),
    pgenf(time, gf_est[1], gf_est[2], gf_est[3], gf_est[4],
          lower.tail = FALSE, log.p = TRUE))) -
    sum(pgenf(entry[entry > 0], gf_est[1], gf_est[2], gf_est[3], gf_est[4],
              lower.tail = FALSE, log.p = TRUE)))
add("lung_genf_truncated_loglik", list(
    fixture = "lung", software = "R flexsurv",
    version = versions$flexsurv,
    call = "sum of dgenf / pgenf(log) less pgenf(entry, log), at the lung_genf estimates",
    settings = list(params = gf_est,
                    entry = "floor(time / 4)"),
    values = list(loglik = gf_ll)
))

out <- list(
    generator = "scripts/reference/reference_r_distributions.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_distributions.json"))
cat("wrote", file.path(data_dir, "r_distributions.json"), "\n")
