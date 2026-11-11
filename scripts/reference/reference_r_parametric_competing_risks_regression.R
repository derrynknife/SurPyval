# Reference results from R flexsurv for ParametricCompetingRisksRegression:
# each cause's parametric regression, fitted with the other causes
# censored, and the cumulative incidences of the competing-risks model
# built from them at given covariate rows.
#
# Writes surpyval/tests/reference/data/r_parametric_competing_regression.json,
# with the software, its version, the exact call and the settings of each
# entry, as reference_r.R does. Generated with R 4.6.2, flexsurv 2.3.2 and
# survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_parametric_competing_risks_regression.R
#
# Run make_fixtures.py first. Needs flexsurv and jsonlite. Re-running
# reproduces the file byte for byte.
#
# The data are two stored fixtures:
#
# * pbc (cause 1 transplant, 2 death, 0 censored; time in years),
#   covariates age, log_bili and female;
# * competing (seeded synthetic, integer times with ties; cause 0
#   censored), covariates group and z.
#
# Each cause k, flexsurvreg(Surv(time, cause == k) ~ covariates, dist =
# "weibullPH") with optim's relative tolerance tightened to 1e-14: the
# maximised log-likelihood, the shape, the scale and the coefficients,
# and their covariance on flexsurv's scale (log shape, log scale, the
# coefficients). flexsurv's weibullPH has hazard scale * shape *
# t^(shape - 1) * exp(coefficients' Z); SurPyval's WeibullPH has
# (beta / alpha) (t / alpha)^(beta - 1) exp(coefficients' Z), so shape is
# beta and scale is alpha^(-beta), and the coefficients are the same.
# On pbc, cause 2 as a lognormal accelerated failure time model too
# (dist = "lnorm": meanlog = mu + coefficients' Z, sdlog): SurPyval's
# LogNormalAFT with the coefficients negated (in SurPyval a positive
# coefficient shortens the life).
#
# The cumulative incidences: the two fits as the transitions of a
# competing-risks multi-state model (fmsm with the transition matrix 1 ->
# 2, 1 -> 3), pmatrix.fs(newdata = row, t = times), solved by deSolve's
# ODE solver with its tolerances tightened to 1e-12, for each covariate
# row: the probabilities of states 2 and 3 are the causes' cumulative
# incidences, and of state 1 the all-cause survival.

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

refs <- list()
tight <- list(reltol = 1e-14, maxit = 10000)

fit_cause <- function(data, time, covs, k, dist) {
    f <- as.formula(sprintf("Surv(%s, cause == %d) ~ %s", time, k,
                            paste(covs, collapse = " + ")))
    flexsurvreg(f, data = data, dist = dist, control = tight)
}

cause_values <- function(fit) list(
    loglik = fit$loglik,
    names = rownames(fit$res),
    est = unname(fit$res[, "est"]),
    cov_names = colnames(vcov(fit)),
    cov = unname(vcov(fit))
)

record_fit <- function(id, fixture_name, time, covs, k, dist, fit) {
    refs[[id]] <<- list(
        fixture = fixture_name,
        software = "R flexsurv",
        version = as.character(packageVersion("flexsurv")),
        call = sprintf(paste("flexsurvreg(Surv(%s, cause == %d) ~ %s,",
                             "dist = \"%s\", control = list(reltol =",
                             "1e-14, maxit = 10000))"),
                       time, k, paste(covs, collapse = " + "), dist),
        settings = list(covariates = covs, dist = dist,
                        cov = "vcov(fit): log of the positive parameters"),
        values = cause_values(fit)
    )
}

record_cif <- function(id, fixture_name, fits, covs, rows, times, dists) {
    tmat <- rbind(c(NA, 1, 2), c(NA, NA, NA), c(NA, NA, NA))
    model <- fmsm(fits[[1]], fits[[2]], trans = tmat)
    cif <- list()
    for (r in seq_len(nrow(rows))) {
        nd <- as.data.frame(rows[r, , drop = FALSE])
        names(nd) <- covs
        p <- pmatrix.fs(model, t = times, newdata = nd,
                        rtol = 1e-12, atol = 1e-12)
        if (length(times) == 1) p <- list(p)
        cif[[r]] <- list(
            row = unname(as.numeric(rows[r, ])),
            survival = unname(sapply(p, function(m) m[1, 1])),
            cause1 = unname(sapply(p, function(m) m[1, 2])),
            cause2 = unname(sapply(p, function(m) m[1, 3]))
        )
    }
    refs[[id]] <<- list(
        fixture = fixture_name,
        software = "R flexsurv",
        version = as.character(packageVersion("flexsurv")),
        call = paste("pmatrix.fs(fmsm(fit1, fit2, trans = rbind(c(NA, 1,",
                     "2), NA, NA)), t = times, newdata = row, rtol =",
                     "1e-12, atol = 1e-12)"),
        settings = list(covariates = covs, dists = dists,
                        states = "1 no failure, 2 cause 1, 3 cause 2"),
        values = list(times = times, rows = cif)
    )
}

pbc_d <- fixture("pbc")
pbc_covs <- c("age", "log_bili", "female")
pbc_fits <- list()
for (k in 1:2) {
    pbc_fits[[k]] <- fit_cause(pbc_d, "years", pbc_covs, k, "weibullPH")
    record_fit(paste0("pbc_weibullPH_cause", k), "pbc", "years", pbc_covs,
               k, "weibullPH", pbc_fits[[k]])
}
record_cif("pbc_weibullPH_cif", "pbc", pbc_fits, pbc_covs,
           rbind(c(45, 0, 1), c(60, 1.5, 0)), c(1, 3, 6, 10, 20, 40),
           c("weibullPH", "weibullPH"))

pbc_ln <- fit_cause(pbc_d, "years", pbc_covs, 2, "lnorm")
record_fit("pbc_lnorm_cause2", "pbc", "years", pbc_covs, 2, "lnorm", pbc_ln)
record_cif("pbc_mixed_cif", "pbc", list(pbc_fits[[1]], pbc_ln), pbc_covs,
           rbind(c(45, 0, 1), c(60, 1.5, 0)), c(1, 3, 6, 10, 20, 40),
           c("weibullPH", "lnorm"))

cr_d <- fixture("competing")
cr_covs <- c("group", "z")
cr_fits <- list()
for (k in 1:2) {
    cr_fits[[k]] <- fit_cause(cr_d, "x", cr_covs, k, "weibullPH")
    record_fit(paste0("competing_weibullPH_cause", k), "competing", "x",
               cr_covs, k, "weibullPH", cr_fits[[k]])
}
record_cif("competing_weibullPH_cif", "competing", cr_fits, cr_covs,
           rbind(c(0, 0), c(1, 0.5)), c(2, 5, 10, 20, 60),
           c("weibullPH", "weibullPH"))

out <- list(
    generator = paste0("scripts/reference/",
                       "reference_r_parametric_competing_risks_regression.R"),
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_parametric_competing_regression.json"))
cat("wrote", file.path(data_dir, "r_parametric_competing_regression.json"),
    "\n")
