# Reference results from R for the mixed-effects Cox model and the
# log-normal shared frailty with a Cox baseline.
#
# Writes surpyval/tests/reference/data/r_coxme.json: fits of R's coxme
# (Therneau) and of survival::coxph(... + frailty(dist = "gaussian")) to
# the kidney catheter data, the lung cancer data by institution and the
# seeded clustered fixture (nested, crossed and random-slope terms, with
# left truncation and strata), with the software, its version, the call
# and the values of each.
#
#     Rscript scripts/reference/reference_r_coxme.R
#
# Run make_fixtures.py first. Needs survival, coxme and jsonlite.
#
# * coxme maximises the Laplace-approximated integrated partial likelihood
#   over the variances with optim(); its default tolerance (reltol 1e-5)
#   stops early (on eortc's correlated random slope by 0.014 in the
#   log-likelihood, with a correlation of 0.785 where the maximum is at
#   0.992). Here optim runs with reltol 1e-14, by Nelder-Mead and by BFGS,
#   and the fit with the higher integrated log-likelihood is kept.
# * sparse = c(1000, 0.02) keeps coxme's information matrix dense (it is
#   sparse by default only for a grouping factor with more than 50
#   levels, and none here has that many).
# * coxph's gaussian frailty chooses its variance by an approximate REML
#   (McGilchrist 1993), not by maximum likelihood; its penalised fit at a
#   given variance (frailty(..., theta = )) is the same penalised partial
#   likelihood as coxme's, so it is compared at a fixed variance.

suppressPackageStartupMessages({
    library(survival)
    library(coxme)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!dir.exists(data_dir)) stop("Run from the repository root")

fx <- fromJSON(file.path(data_dir, "fixtures.json"))
k <- as.data.frame(fx$kidney$columns)
k$status <- 1 - k$c
stopifnot(isTRUE(all.equal(k$time, survival::kidney$time)))
lung <- as.data.frame(fx$lung$columns)
lung$status <- 2 - lung$c
cl <- as.data.frame(fx$clustered$columns)
cl$status <- 1 - cl$c

refs <- list()
entry <- function(fixture, package, call, settings, values) list(
    fixture = fixture,
    software = paste("R", package),
    version = as.character(packageVersion(package)),
    call = call,
    settings = settings,
    values = values
)

tight <- function(formula, data, ties) {
    fits <- lapply(c("Nelder-Mead", "BFGS"), function(method) {
        ctl <- suppressWarnings(coxme.control(
            optpar = list(method = method,
                          control = list(reltol = 1e-14, maxit = 5000)),
            eps = 1e-12, toler.chol = .Machine$double.eps^0.75,
            iter.max = 100))
        tryCatch(suppressWarnings(coxme(formula, data = data, ties = ties,
                                        control = ctl,
                                        sparse = c(1000, 0.02))),
                 error = function(e) NULL)
    })
    fits <- Filter(Negate(is.null), fits)
    best <- fits[[which.max(sapply(fits, function(f) f$loglik[2]))]]
    best
}

values_of <- function(fit) {
    vc <- VarCorr(fit)
    re <- ranef(fit)
    list(
        beta = unname(fixef(fit)),
        se = unname(sqrt(diag(as.matrix(vcov(fit))))),
        loglik_null = unname(fit$loglik[1]),
        loglik_integrated = unname(fit$loglik[2]),
        loglik_penalised = unname(fit$loglik[3]),
        loglik_fitted = unname(fit$loglik[3] + fit$penalty),
        df_penalised = unname(fit$df[2]),
        variance = lapply(vc, function(v) unname(as.matrix(v))),
        ranef = lapply(re, function(r) {
            r <- as.matrix(r)
            list(labels = rownames(r) %||% names(re[[1]]),
                 values = unname(r))
        })
    )
}
`%||%` <- function(a, b) if (is.null(a)) b else a

settings <- function(ties) list(
    ties = ties,
    optim = "Nelder-Mead and BFGS with reltol 1e-14; the higher kept",
    variance = paste("VarCorr: a variance per term, or the effects'",
                     "covariance with correlations off the diagonal"),
    loglik = paste("coxme's loglik: NULL, Integrated, Penalized; fitted",
                   "is Penalized + penalty, the partial likelihood at the",
                   "fit (coxme prints it as Fitted)"),
    ranef = "ranef(fit): each group's predicted random effects"
)

for (ties in c("efron", "breslow")) {
    f <- Surv(time, status) ~ age + female + (1 | id)
    fit <- tight(f, k, ties)
    refs[[paste0("kidney_coxme_", ties)]] <- entry(
        "kidney", "coxme",
        paste0("coxme(Surv(time, status) ~ age + female + (1 | id), ",
               "ties = '", ties, "')"),
        settings(ties), values_of(fit))
}

fit <- tight(Surv(time, status) ~ age + sex + (1 | inst), lung, "efron")
refs[["lung_coxme_inst"]] <- entry(
    "lung", "coxme",
    "coxme(Surv(time, status) ~ age + sex + (1 | inst)), rows without inst dropped",
    settings("efron"), values_of(fit))

fit <- tight(Surv(tl, x, status) ~ trt + age + strata(sex) +
                 (1 | hospital/ward) + (1 | batch), cl, "efron")
refs[["clustered_nested_crossed"]] <- entry(
    "clustered", "coxme",
    paste("coxme(Surv(tl, x, status) ~ trt + age + strata(sex) +",
          "(1 | hospital/ward) + (1 | batch))"),
    settings("efron"), values_of(fit))

for (ties in c("efron", "breslow")) {
    fit <- tight(Surv(tl, x, status) ~ trt + age + (1 + trt | hospital),
                 cl, ties)
    refs[[paste0("clustered_random_slope_", ties)]] <- entry(
        "clustered", "coxme",
        paste0("coxme(Surv(tl, x, status) ~ trt + age + ",
               "(1 + trt | hospital), ties = '", ties, "')"),
        settings(ties), values_of(fit))
}

# --- coxph's gaussian frailty at a fixed variance --------------------------
ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 500,
                     outer.max = 50)
fit <- coxph(Surv(time, status) ~ age + female +
                 frailty(id, dist = "gaussian", theta = 0.5, sparse = FALSE),
             data = k, control = ctl)
default <- coxph(Surv(time, status) ~ age + female +
                     frailty(id, dist = "gaussian"), data = k)
refs[["kidney_coxph_gaussian_fixed"]] <- entry(
    "kidney", "survival",
    paste("coxph(Surv(time, status) ~ age + female + frailty(id, dist =",
          "'gaussian', theta = 0.5, sparse = FALSE))"),
    list(ties = "efron",
         loglik = paste("loglik[2] is the partial likelihood at the",
                        "penalised fit, penalty its penalty"),
         se = "sqrt(diag(var)) with sparse = FALSE (the full information)",
         default_call = paste(
             "coxph's own choice of theta (frailty(id, dist = 'gaussian'),",
             "method 'reml') is", signif(default$history[[1]]$theta, 7))),
    list(theta = 0.5, beta = unname(coef(fit)[1:2]),
         se = unname(sqrt(diag(fit$var))[1:2]),
         frail = unname(coef(fit)[-(1:2)]),
         loglik_partial = unname(fit$loglik[2]),
         penalty = unname(fit$penalty[2]),
         reml_theta = default$history[[1]]$theta)
)

out <- list(
    generator = "scripts/reference/reference_r_coxme.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_coxme.json"))
cat("wrote", file.path(data_dir, "r_coxme.json"), "\n")
