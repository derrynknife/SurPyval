# Reference values for the joint model with a lag (the hazard at age t
# following the marker's line at max(t - lag, 0)), printed for
# surpyval/tests/reference/test_joint_lag.py.
#
#     R_LIBS_USER=<library with JM> Rscript scripts/reference/reference_r_joint_lag.R
#
# Written with R 4.6.2, JM 1.5.2, nlme and survival (the versions are
# printed). The data, the models and the controls are those of
# reference_r_joint_links.R (the Mayo PBC trial, log serum bilirubin a
# line in years, death the event, the treatment in the hazard only), with
# jointModel's lag = 1 (year). JM evaluates the line at max(t - lag, 0),
# as SurPyval does, for the value and for the area under the line (0
# before the lag). Its slope before the lag is the line's rate at age 0,
# where SurPyval's is 0 (the path held at its age-0 value), so the slope
# link is not compared.
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

area <- list(fixed = ~ -1 + year + I(year^2 / 2), indFixed = 1:2,
             random = ~ -1 + year + I(year^2 / 2), indRandom = 1:2)
control <- list(GHk = 15, GKk = 15, iter.EM = 200, iter.qN = 1000,
                tol1 = 1e-9, tol2 = 1e-9, tol3 = 1e-12,
                numeriDeriv = "cd", eps.Hes = 1e-5)

show <- function(label, ...) {
  fit <- jointModel(lmeFit, coxFit, timeVar = "year",
                    method = "weibull-PH-aGH", control = control,
                    lag = 1, ...)
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

show("value, lag 1")
show("cumulative, lag 1", parameterization = "slope", derivForm = area)
