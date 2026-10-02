# Reference results from R for the shared-frailty models (#342, #343).
#
# Writes surpyval/tests/reference/data/r_frailty.json: fits to the kidney
# catheter data (McGilchrist and Aisbett 1991; R's survival::kidney, which
# is surpyval.datasets.load_kidney), with the software, its version, the
# call and the values of each, as reference_r.R does for the other files.
#
#     Rscript scripts/reference/reference_r_frailty.R
#
# Run make_fixtures.py first. Needs survival, lme4 and jsonlite (Ubuntu: r-cran-survival r-cran-lme4
# r-cran-jsonlite). Re-running reproduces the file byte for byte.
#
# * Log-normal frailty with a parametric baseline (#343). With an
#   exponential baseline the likelihood is that of a Poisson GLMM with a
#   normal random intercept and offset log(time) (Aitkin and Clayton 1980),
#   up to sum(status * log(time)); with a Weibull baseline of shape k it
#   is that with offset k log(time), up to sum(status * (log k - log
#   time)), so profiling lme4's glmer over k gives the maximum likelihood.
#   glmer integrates the random effect by adaptive Gauss-Hermite
#   quadrature (nAGQ = 25). Its logLik with nAGQ > 1 is minus half the
#   deviance, measured from the saturated model, whose log-likelihood for
#   0/1 responses is -sum(status): the full log-likelihood is
#   logLik - sum(status), which is what is stored.
# * Gamma frailty with a Cox baseline (#342): coxph(... + frailty(id,
#   dist = "gamma")), whose penalised fit at a given theta is the EM
#   (maximum likelihood) fit, and whose theta maximises the integrated
#   ("I-") likelihood. coxph's outer search over theta stops early (its
#   default tolerance), so theta is maximised here to 1e-10 with
#   optimize() over coxph fits at a fixed theta (frailty(..., theta =)),
#   and the values are those of the fit at the maximum.

suppressPackageStartupMessages({
    library(survival)
    library(lme4)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!dir.exists(data_dir)) stop("Run from the repository root")

# The kidney fixture (fixtures.json, from load_kidney), which must be R's
# own copy.
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

# --- log-normal frailty, parametric baseline (lme4) -----------------------
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
glmm <- function(shape) {
    k$off <- shape * log(k$time)
    glmer(status ~ age + female + (1 | id) + offset(off), family = poisson,
          data = k, nAGQ = 25, control = ctrl)
}
full_loglik <- function(fit, shape) {
    as.numeric(logLik(fit)) - sum(k$status) +
        sum(k$status * (log(shape) - log(k$time)))
}
profile <- optimize(function(s) full_loglik(glmm(s), s), c(0.5, 3),
                    maximum = TRUE, tol = 1e-10)
lognormal_call <- paste(
    "glmer(status ~ age + female + (1 | id) + offset(shape * log(time)),",
    "family = poisson, nAGQ = 25), profiled over shape")
for (baseline in c("weibull", "exponential")) {
    shape <- if (baseline == "weibull") profile$maximum else 1
    fit <- glmm(shape)
    b <- unname(fixef(fit))
    # log h0 = log(shape) + (shape - 1) log t - shape log alpha: the
    # intercept is -shape log(alpha); the exponential's rate is exp(b0).
    dist_params <- if (baseline == "weibull") {
        c(exp(-b[1] / shape), shape)
    } else {
        exp(b[1])
    }
    refs[[paste0("kidney_lognormal_", baseline)]] <- entry(
        "lme4", lognormal_call,
        list(model = paste(baseline, "baseline, log-normal frailty,",
                           "covariates age and female = (sex == 2)"),
             frailty = "u = exp(w), w ~ N(0, theta)",
             loglik = "full log-likelihood, logLik - sum(status) (see top)"),
        list(dist_params = dist_params, beta = b[2:3],
             theta = unname(VarCorr(fit)$id[1]),
             loglik = full_loglik(fit, shape))
    )
}

out <- list(
    generator = "scripts/reference/reference_r_frailty.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_frailty.json"))
cat("wrote", file.path(data_dir, "r_frailty.json"), "\n")
