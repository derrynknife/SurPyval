# Reference results from R for diagnose() of the distributional
# regressions (WeibullDistributional and the others): flexsurv's
# flexsurvreg with covariates on every parameter (anc=) and gamlss.
#
# Writes surpyval/tests/reference/data/r_distributional_diagnostics.json,
# with the software, its version, the call and the values of each entry,
# as reference_r_ph_aft.R does. Every fit is to a fixture already in
# fixtures.json (lung and left_truncation), so make_fixtures.py needs no
# change.
#
#     R_LIBS_USER=~/Library/R-lib Rscript \
#         scripts/reference/reference_r_distributional_diagnostics.R
#
# Run with R 4.6.2, flexsurv 2.3.2, gamlss 5.5.5 and jsonlite.
# Re-running reproduces the file byte for byte.
#
# What each check is compared with:
#
# * The Cox-Snell residuals (the distribution check's plot): flexsurv's
#   coxsnell_flexsurvreg, each row's fitted cumulative hazard at its own
#   parameters (with delayed entry, also at its entry time).
# * The likelihood-ratio tests of a quadratic term on one parameter
#   (functional_form): flexsurvreg fits with and without I(v^2) in that
#   parameter's formula (the scale's own formula, or anc= for the
#   shape); twice the difference of the log-likelihoods.
# * The AIC comparison: AIC of the Weibull, log-logistic, log-normal,
#   gamma and exponential flexsurvreg fits with the same covariates on
#   every parameter (on the rate alone for the exponential).
# * gamlss's normalised quantile residuals, qnorm(F(x | z)) (what
#   plot(gamlss) draws), of a log-normal fit to the failed rows of lung:
#   a monotone function of the Cox-Snell residual H, qnorm(1 - exp(-H)).

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(gamlss)
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
lt_d <- fixture("left_truncation")
lt_d$status <- 1 - lt_d$c

refs <- list()
env <- environment()
finite_or_na <- function(v) {
    if (is.list(v)) return(lapply(v, finite_or_na))
    if (is.numeric(v)) {
        v <- unname(v)
        v[!is.finite(v)] <- NA
    }
    v
}
record <- function(id, fixture, software, call, settings, extract) {
    result <- eval(parse(text = call), envir = env)
    pkg <- strsplit(software, " ")[[1]][2]
    refs[[id]] <<- list(
        fixture = fixture,
        software = software,
        version = as.character(packageVersion(pkg)),
        call = call,
        settings = settings,
        values = finite_or_na(extract(result))
    )
    invisible(result)
}

ctl <- list(reltol = 1e-14, maxit = 20000)
lr <- function(small, big) 2 * (as.numeric(logLik(big)) -
                                as.numeric(logLik(small)))

flex <- function(data, surv, rhs, anc_rhs, dist) {
    f <- as.formula(paste(surv, "~", rhs))
    anc <- NULL
    if (!is.null(anc_rhs)) {
        anc <- list(as.formula(paste("~", anc_rhs)))
        names(anc) <- if (dist == "lnorm") "sdlog" else "shape"
    }
    flexsurvreg(f, anc = anc, data = data, dist = dist, control = ctl)
}

# Each column of `squares` squared on the scale (the location) and on the
# shape, one at a time
distributional_values <- function(data, surv, covs, squares) {
    rhs <- paste(covs, collapse = " + ")
    fit <- flex(data, surv, rhs, rhs, "weibull")
    cs <- coxsnell_flexsurvreg(fit)
    out <- list(
        loglik = as.numeric(logLik(fit)),
        aic = AIC(fit),
        cox_snell = cs$est
    )
    for (v in squares) {
        sq <- paste0(rhs, " + I(", v, "^2)")
        out[[paste0("square_alpha_", v)]] <- lr(
            fit, flex(data, surv, sq, rhs, "weibull"))
        out[[paste0("square_beta_", v)]] <- lr(
            fit, flex(data, surv, rhs, sq, "weibull"))
    }
    out
}

record(
    "lung_weibull", "lung", "R flexsurv",
    paste0("distributional_values(lung_cc, 'Surv(time, status)', ",
           "c('age', 'sex', 'ph_ecog'), c('age', 'ph_ecog'))"),
    list(rows = "lung without missing ph_ecog",
         model = "weibull, age + sex + ph_ecog on the scale and the shape",
         control = "reltol = 1e-14"),
    identity
)

aic_values <- function(data) {
    rhs <- "age + sex + ph_ecog"
    surv <- "Surv(time, status)"
    list(
        weibull = AIC(flex(data, surv, rhs, rhs, "weibull")),
        llogis = AIC(flex(data, surv, rhs, rhs, "llogis")),
        lnorm = AIC(flex(data, surv, rhs, rhs, "lnorm")),
        gamma = AIC(flex(data, surv, rhs, rhs, "gamma")),
        exp = AIC(flex(data, surv, rhs, NULL, "exp"))
    )
}
record(
    "lung_aic", "lung", "R flexsurv", "aic_values(lung_cc)",
    list(rows = "lung without missing ph_ecog",
         model = "age + sex + ph_ecog on every parameter",
         control = "reltol = 1e-14"),
    identity
)

record(
    "left_truncation_weibull", "left_truncation", "R flexsurv",
    "distributional_values(lt_d, 'Surv(tl, x, status)', 'z', character(0))",
    list(model = "weibull, z on the scale and the shape, entry at tl",
         control = "reltol = 1e-14"),
    function(out) {
        fit <- flex(lt_d, "Surv(tl, x, status)", "z", "z", "weibull")
        p <- summary(fit, type = "cumhaz", t = lt_d$tl, newdata = lt_d,
                     ci = FALSE, tidy = TRUE)
        # the cumulative hazard of each row at its own entry time
        n <- nrow(lt_d)
        out$entry <- p$est[(seq_len(n) - 1) * n + seq_len(n)]
        out
    }
)

lung_failed <- lung_cc[lung_cc$status == 1, c("time", "age", "sex", "ph_ecog")]
gamlss_values <- function(fit) list(
    loglik = as.numeric(logLik(fit)),
    quantile_residuals = as.numeric(residuals(fit))
)
record(
    "lung_failed_lognormal", "lung", "R gamlss",
    paste0("gamlss(time ~ age + sex + ph_ecog, sigma.formula = ~ age + sex, ",
           "family = LOGNO(), data = lung_failed, control = ",
           "gamlss.control(c.crit = 1e-10, n.cyc = 2000, trace = FALSE))"),
    list(rows = "the failed rows of lung without missing ph_ecog",
         residuals = "normalised quantile residuals qnorm(F(time | z))"),
    gamlss_values
)

out <- list(
    generator = "scripts/reference/reference_r_distributional_diagnostics.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
path <- file.path(data_dir, "r_distributional_diagnostics.json")
writeLines(text, path)
cat("wrote", path, "\n")
