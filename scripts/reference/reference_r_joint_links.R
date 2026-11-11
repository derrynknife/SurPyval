# Reference values for the joint model's links (the hazard following the
# marker's current value, its slope, both, or the area under it), printed
# for surpyval/tests/reference/test_joint_links.py.
#
#     R_LIBS_USER=<library with JM> Rscript scripts/reference/reference_r_joint_links.R
#
# Written with R 4.6.2, JM 1.5.2, nlme and survival (the versions are
# printed).
#
# The Mayo PBC trial (surpyval/datasets/pbc2.csv; the pbc2 of JM): the
# marker is log serum bilirubin, a straight line in years since enrolment
# with a level and a rate of each patient's own; the event is death
# (transplanted or alive: censored); the treatment (D-penicillamine)
# enters the hazard only. jointModel(method = "weibull-PH-aGH") has the
# Weibull proportional hazards
#
#     h(t) = sigma.t t^(sigma.t - 1) exp(g0 + g1 drug + alpha m(t) + Dalpha f(t)),
#
# m the patient's line and f what derivForm describes: the line's slope
# (parameterization "slope" or "both"), or, given the line's integral in
# place of its derivative, the area under it from 0 (the cumulative
# parameterization, Rizopoulos 2012, section 5.1). SurPyval's Weibull is
# (beta / alpha) (t / alpha)^(beta - 1): beta = sigma.t, and its alpha at a
# marker of 0 is exp(-g0 / sigma.t).
#
# The likelihood is by pseudo-adaptive Gauss-Hermite quadrature (15 points
# a dimension) and 15-point Gauss-Kronrod for the cumulative hazard, the
# search run to tight tolerances.
suppressPackageStartupMessages({
  library(JM)
})
cat("R", R.version$major, ".", R.version$minor,
    " JM ", as.character(packageVersion("JM")),
    " nlme ", as.character(packageVersion("nlme")),
    " survival ", as.character(packageVersion("survival")), "\n", sep = "")

pbc <- read.csv("surpyval/datasets/pbc2.csv", check.names = FALSE)
pbc$treated <- as.numeric(pbc$drug == "D-penicil")
pbc$y <- log(pbc$serBilir)
pbc.id <- pbc[!duplicated(pbc$id), ]

lmeFit <- lme(y ~ year, random = ~ year | id, data = pbc,
              control = lmeControl(opt = "optim"))
coxFit <- coxph(Surv(years, status2) ~ treated, data = pbc.id, x = TRUE)

slope <- list(fixed = ~ 1, indFixed = 2, random = ~ 1, indRandom = 2)
area <- list(fixed = ~ -1 + year + I(year^2 / 2), indFixed = 1:2,
             random = ~ -1 + year + I(year^2 / 2), indRandom = 1:2)
control <- list(GHk = 15, GKk = 15, iter.EM = 200, iter.qN = 1000,
                tol1 = 1e-9, tol2 = 1e-9, tol3 = 1e-12,
                numeriDeriv = "cd", eps.Hes = 1e-5)

show <- function(label, ...) {
  fit <- jointModel(lmeFit, coxFit, timeVar = "year",
                    method = "weibull-PH-aGH", control = control, ...)
  co <- fit$coefficients
  D <- co$D
  cat(sprintf("%s:\n", label))
  out <- c(a = co$betas[[1]], b = co$betas[[2]],
           sd_a = sqrt(D[1, 1]), sd_b = sqrt(D[2, 2]),
           corr_a_b = D[1, 2] / sqrt(D[1, 1] * D[2, 2]),
           sd_noise = co$sigma,
           alpha = exp(-co$gammas[[1]] / co$sigma.t), beta = co$sigma.t,
           treated = co$gammas[[2]])
  if (!is.null(co$alpha)) out <- c(out, alpha_value = co$alpha)
  if (!is.null(co$Dalpha)) out <- c(out, alpha_deriv = co$Dalpha)
  for (n in names(out)) cat(sprintf("  %s = %.10g\n", n, out[[n]]))
  cat(sprintf("  logLik = %.8f\n", as.numeric(logLik(fit))))
  se <- sqrt(diag(vcov(fit)))
  keep <- grep("^T\\.(treated|alpha|Dalpha)|^Y\\.", names(se))
  for (k in keep) cat(sprintf("  se[%s] = %.6g\n", names(se)[k], se[k]))
  cat(sprintf("  convergence = %d\n", fit$convergence))
}

show("value")
show("slope", parameterization = "slope", derivForm = slope)
show("value and slope", parameterization = "both", derivForm = slope)
show("cumulative", parameterization = "slope", derivForm = area)
