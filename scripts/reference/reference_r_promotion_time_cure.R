# Reference results from R for the promotion-time (non-mixture, bounded
# cumulative hazard) cure regression, surpyval.PromotionTimeCure.
#
# Writes surpyval/tests/reference/data/promotion_time_cure_r.json, which
# surpyval/tests/univariate/regression/test_promotion_time_cure.py reads
# when it is present (the test is skipped until then). For each fit: the
# software, its version, the call, the estimates on the link scale with
# their standard errors, the log-likelihood and the survival at fixed
# times for two covariate rows; and the data, stored whole.
#
#     Rscript scripts/reference/reference_r_promotion_time_cure.R
#
# Run from the repository root. Needs flexsurv, flexsurvcure and
# jsonlite. NOT RUN YET: written against the flexsurvcure 1.3 documentation
# (mixture = FALSE, link = "loglog"); check the call on the first run.
#
# The model, AS ASSUMED HERE AND TO BE CONFIRMED ON THE FIRST RUN:
# flexsurvcure's non-mixture model is S(t) = theta^F0(t), with theta the
# cured fraction; if its link = "loglog" means theta = exp(-exp(eta)) (and
# not the complementary log-log, theta = 1 - exp(-exp(eta))), then
# S(t) = exp(-exp(eta) F0(t)): SurPyval's promotion-time model with
# log(theta_surpyval) = eta, the same coefficients and the same sign.
# Check that on the first run (e.g. the fitted cure fraction against
# exp(-theta) from SurPyval) before trusting the test.
#
# The data: 500 units with a binary z1 and a normal z2 (seed 20261127).
# Each carries a Poisson number of latent causes, mean
# 0.6 exp(0.7 z1 - 0.4 z2), each with a Weibull(shape 1.6, scale 12)
# promotion time; the unit fails at the first, and never if it has
# none. Follow-up ends at a uniform(30, 60) time, far past the lives, so
# the cured fraction shows as a plateau.
#
# * weibull: dist = "weibull", theta ~ z1 + z2.
# * lnorm: dist = "lnorm", theta ~ z1 + z2.
# * llogis: dist = "llogis", theta ~ z1 + z2.
# * weibull_scale: dist = "weibull", theta ~ z1 + z2 and the scale on
#   z1 (anc = list(scale = ~ z1)).

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(flexsurvcure)
    library(jsonlite)
})

out_path <- file.path("surpyval", "tests", "reference", "data",
                      "promotion_time_cure_r.json")
if (!dir.exists(dirname(out_path))) stop("Run from the repository root")

set.seed(20261127)
n <- 500
z1 <- rbinom(n, 1, 0.5)
z2 <- round(rnorm(n), 4)
causes <- rpois(n, 0.6 * exp(0.7 * z1 - 0.4 * z2))
life <- vapply(causes, function(k) {
    if (k == 0) Inf else min(rweibull(k, shape = 1.6, scale = 12))
}, 0.0)
end <- round(runif(n, 30, 60), 3)
time <- round(pmin(life, end), 4)
status <- as.integer(life <= end)
d <- data.frame(time, status, z1, z2)

rows <- data.frame(z1 = c(0, 1), z2 = c(0, 0.5))
times <- c(2, 5, 10, 20, 40)

fit_one <- function(dist, anc = NULL) {
    args <- list(Surv(time, status) ~ z1 + z2, data = d, dist = dist,
                 link = "loglog", mixture = FALSE)
    if (!is.null(anc)) args$anc <- anc
    fit <- do.call(flexsurvcure, args)
    sm <- summary(fit, newdata = rows, t = times, type = "survival",
                  ci = FALSE, tidy = TRUE)
    list(
        software = "R flexsurvcure",
        version = as.character(packageVersion("flexsurvcure")),
        call = paste0("flexsurvcure(Surv(time, status) ~ z1 + z2, dist = '",
                      dist, "', link = 'loglog', mixture = FALSE",
                      if (!is.null(anc)) ", anc = list(scale = ~ z1)" else "",
                      ")"),
        names = rownames(fit$res.t),
        estimate = unname(fit$res.t[, "est"]),
        se = unname(fit$res.t[, "se"]),
        loglik = fit$loglik,
        times = times,
        rows = rows,
        sf = sm$est
    )
}

refs <- list(
    weibull = fit_one("weibull"),
    lnorm = fit_one("lnorm"),
    llogis = fit_one("llogis"),
    weibull_scale = fit_one("weibull", anc = list(scale = ~ z1))
)

out <- list(
    generator = "scripts/reference/reference_r_promotion_time_cure.R",
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    data = as.list(d),
    references = refs
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE),
           out_path)
cat("wrote", out_path, "\n")
