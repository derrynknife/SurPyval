# Reference results from R for Firth's penalised Cox fit,
# CoxPH.fit_firth (Heinze and Schemper 2001).
#
# Writes surpyval/tests/reference/data/r_firth_cox.json, with the
# software, its version, the call and the values of each entry, as
# reference_r.R does for the other files. Generated with R 4.6.2,
# survival 3.8.12 and coxphf 1.13.4.
#
#     Rscript scripts/reference/reference_r_firth_cox.R
#
# Run make_fixtures.py first. Needs survival, coxphf and jsonlite.
# Re-running reproduces the file byte for byte.
#
# coxphf maximises the partial log-likelihood (Breslow ties) plus half
# the log determinant of its information, and gives the coefficients,
# their covariance (the inverse of the information of the partial
# likelihood at the estimate), profile penalised-likelihood intervals
# and p-values (pl = TRUE) and the penalised likelihood-ratio test.
# Its convergence criteria are tightened (epsilon = gconv = 1e-12,
# maxit = 200) so that the reference is coxphf's answer at
# convergence; at its defaults (1e-6, 1e-4) the coefficients move by up
# to 2e-9 on these data. coxphf divides each covariate by its standard
# deviation before fitting, which leaves the estimates and every test
# unchanged but adds sum(log(sd)) to its log-likelihoods; that sum is
# recorded as "log_sd_sum".
#
# The fits:
#
# * separation (seeded synthetic): every event has z1 = 1, so the
#   ordinary Cox coefficient of z1 is infinite; z1 + z2, at the 95% and
#   the 90% level;
# * lung: age + sex, where Firth's fit is close to the ordinary one;
# * ties (seeded synthetic, many tied times): z1 + z2;
# * left_truncation (seeded synthetic): Surv(tl, x, status) ~ z;
# * heart (start-stop): Surv(start, stop, event) ~ age + year +
#   transplant.

suppressPackageStartupMessages({
    library(survival)
    library(coxphf)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
columns <- function(name) {
    d <- as.data.frame(fx[[name]]$columns)
    d$status <- 1 - d$c
    d
}
separation <- columns("separation")
lung_d <- columns("lung")
ties_d <- columns("ties")
lt_d <- columns("left_truncation")
heart_d <- columns("heart")
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(lung_d$age, survival::lung$age)),
    isTRUE(all.equal(lung_d$sex, survival::lung$sex)),
    isTRUE(all.equal(heart_d$start, survival::heart$start)),
    isTRUE(all.equal(heart_d$stop, survival::heart$stop)),
    isTRUE(all.equal(heart_d$status, survival::heart$event)),
    isTRUE(all.equal(heart_d$age, survival::heart$age)),
    isTRUE(all.equal(heart_d$year, survival::heart$year)),
    isTRUE(all.equal(heart_d$transplant,
                     as.numeric(survival::heart$transplant) - 1)),
    # The separation: every event has z1 = 1
    all(separation$z1[separation$status == 1] == 1)
)

env <- environment()
refs <- list()
tight <- "epsilon = 1e-12, gconv = 1e-12, maxit = 200"

matrix_rows <- function(m) {
    m <- unname(as.matrix(m))
    lapply(seq_len(nrow(m)), function(k) m[k, ])
}

add <- function(id, fixture, formula, data_name, covariates, alpha) {
    call <- sprintf("coxphf(%s, data = %s, pl = TRUE, alpha = %s, %s)",
                    formula, data_name, alpha, tight)
    fit <- eval(parse(text = call), envir = env)
    wald_call <- sub("pl = TRUE", "pl = FALSE", call)
    wald <- eval(parse(text = wald_call), envir = env)
    data <- get(data_name, envir = env)
    sds <- apply(as.matrix(data[, covariates, drop = FALSE]), 2, sd)
    stopifnot(all(fit$iter.ci < 200), fit$iter < 200)
    refs[[id]] <<- list(
        fixture = fixture,
        software = "R coxphf",
        version = as.character(packageVersion("coxphf")),
        call = call,
        settings = list(
            ties = "breslow (coxphf's only)",
            survival = as.character(packageVersion("survival")),
            covariates = covariates,
            wald_call = wald_call,
            ci = "log(ci.lower), log(ci.upper): the coefficient's bounds",
            p = "prob: the penalised likelihood-ratio test of each coefficient",
            loglik = "c(null, fitted), penalised, of the standardised covariates"
        ),
        values = list(
            coef = unname(fit$coefficients),
            var = matrix_rows(fit$var),
            ci_lower = unname(log(fit$ci.lower)),
            ci_upper = unname(log(fit$ci.upper)),
            p = unname(fit$prob),
            wald_ci_lower = unname(log(wald$ci.lower)),
            wald_ci_upper = unname(log(wald$ci.upper)),
            wald_p = unname(wald$prob),
            loglik_null = fit$loglik[1],
            loglik = fit$loglik[2],
            lr_statistic = 2 * (fit$loglik[2] - fit$loglik[1]),
            df = fit$df,
            alpha = alpha,
            log_sd_sum = sum(log(sds))
        )
    )
}

add("separation", "separation", "Surv(x, status) ~ z1 + z2", "separation",
    c("z1", "z2"), 0.05)
add("separation_90", "separation", "Surv(x, status) ~ z1 + z2",
    "separation", c("z1", "z2"), 0.1)
add("lung_age_sex", "lung", "Surv(time, status) ~ age + sex", "lung_d",
    c("age", "sex"), 0.05)
add("ties", "ties", "Surv(x, status) ~ z1 + z2", "ties_d", c("z1", "z2"),
    0.05)
add("left_truncation", "left_truncation", "Surv(tl, x, status) ~ z",
    "lt_d", "z", 0.05)
add("heart_start_stop", "heart",
    "Surv(start, stop, status) ~ age + year + transplant", "heart_d",
    c("age", "year", "transplant"), 0.05)

out <- list(
    generator = "scripts/reference/reference_r_firth_cox.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_firth_cox.json"))
cat("wrote", file.path(data_dir, "r_firth_cox.json"), "\n")
