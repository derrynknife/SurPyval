# Reference values for a regression's life at fixed covariates
# (``model.at(Z)``) and for ``with_params``: on survival's lung data
# (fixtures.json), Weibull AFT and PH regressions on sex, fitted by
# flexsurv and survreg, then evaluated at sex = 2 at R's own estimate.
#
# Prints the values that surpyval/tests/univariate/regression/
# test_life_at.py holds (typed in, with this script named beside them):
#
#     Rscript scripts/reference/reference_r_regression_at.R
#
# Run with R 4.6.2, survival 3.8.12 and flexsurv 2.3.2. The test builds the
# SurPyval model at R's estimate with ``with_params`` (the parameters
# mapped as below) and compares ``at([2])``'s functions with R's at the
# same parameters, so the comparison is of the same function at the same
# parameters, not of two fits.
#
# Parameter maps (SurPyval's sign: a positive coefficient, a shorter life):
#
# * flexsurvreg(dist = "weibull"), the AFT form, and survreg(dist =
#   "weibull"): log T = mu + b sex + sigma W, so WeibullAFT's
#   [alpha, beta, coef] = [exp(mu), 1 / sigma, -b].
# * flexsurvreg(dist = "weibullPH"): S(t) = exp(-m e^{b sex} t^a), so
#   WeibullPH's [alpha, beta, coef] = [m^(-1 / a), a, b].

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
    library(flexsurv)
})

fx <- fromJSON(file.path("surpyval", "tests", "reference", "data",
                         "fixtures.json"))
lung_d <- as.data.frame(fx$lung$columns)
lung_d$status <- 1 - lung_d$c
stopifnot(isTRUE(all.equal(lung_d$time, survival::lung$time)))

times <- c(100, 365, 730)
probs <- c(0.1, 0.5, 0.9)
nd <- data.frame(sex = 2)
show <- function(label, x) cat(label, "=", sprintf("%.15g", x), "\n")

# flexsurv, Weibull AFT
aft <- flexsurvreg(Surv(time, status) ~ sex, data = lung_d, dist = "weibull")
cf <- aft$res[, "est"]
show("aft params [exp(mu), 1/sigma, -b]",
     c(cf["scale"], cf["shape"], -coef(aft)["sex"]))
show("aft sf", summary(aft, newdata = nd, t = times, type = "survival",
                       ci = FALSE)[[1]]$est)
show("aft hf", summary(aft, newdata = nd, t = times, type = "hazard",
                       ci = FALSE)[[1]]$est)
show("aft Hf", summary(aft, newdata = nd, t = times, type = "cumhaz",
                       ci = FALSE)[[1]]$est)
show("aft qf", summary(aft, newdata = nd, quantiles = probs,
                       type = "quantile", ci = FALSE)[[1]]$est)
show("aft mean", summary(aft, newdata = nd, type = "mean",
                         ci = FALSE)[[1]]$est)

# flexsurv, Weibull PH
ph <- flexsurvreg(Surv(time, status) ~ sex, data = lung_d,
                  dist = "weibullPH")
a <- ph$res["shape", "est"]
m <- ph$res["scale", "est"]
show("ph params [m^(-1/a), a, b]", c(m^(-1 / a), a, coef(ph)["sex"]))
show("ph sf", summary(ph, newdata = nd, t = times, type = "survival",
                      ci = FALSE)[[1]]$est)
show("ph hf", summary(ph, newdata = nd, t = times, type = "hazard",
                      ci = FALSE)[[1]]$est)
show("ph qf", summary(ph, newdata = nd, quantiles = probs,
                      type = "quantile", ci = FALSE)[[1]]$est)

# survreg, Weibull AFT: the quantiles at sex = 2 (predict type "quantile")
tight <- survreg.control(rel.tolerance = 1e-13, maxiter = 200)
sr <- survreg(Surv(time, status) ~ sex, data = lung_d, dist = "weibull",
              control = tight)
show("survreg params [exp(mu), 1/sigma, -b]",
     c(exp(coef(sr)[1]), 1 / sr$scale, -coef(sr)[2]))
show("survreg qf", predict(sr, newdata = nd, type = "quantile", p = probs))
