# The joint model of two markers on the Mayo PBC trial in R, for
# comparison with markers_pbc.py: log serum bilirubin and albumin, each a
# line in years with a level and a rate of each patient's own (the four
# correlated), the hazard in both markers' current values, no covariates.
#
#     R_LIBS=<library with JMbayes2 and joineRML> \
#         Rscript scripts/studies/joint_recovery/markers_pbc.R
#
# JMbayes2::jm is Bayesian (MCMC; a B-spline baseline log hazard, 2
# chains of 6,000 iterations, 1,000 burn-in, seed 2026): its posterior
# means are compared loosely, within their posterior spread.
# joineRML::mjoint is maximum likelihood by Monte Carlo EM with a Cox
# (unspecified) baseline. Neither has SurPyval's Weibull baseline, so the
# associations agree to within their standard errors, not to digits.
# Run 2026-11-25 with R 4.6.2, JMbayes2 0.6.6 and joineRML 0.4.9; the
# numbers are quoted in docs/Joint Models.rst ("Several markers").

suppressPackageStartupMessages({library(nlme); library(survival); library(JMbayes2); library(joineRML)})
pbc <- read.csv("surpyval/datasets/pbc2.csv")
pbc$lbil <- log(pbc$serBilir)
pbc <- pbc[!is.na(pbc$albumin) & !is.na(pbc$lbil), ]
pbc.id <- pbc[!duplicated(pbc$id), ]
cat("JMbayes2", as.character(packageVersion("JMbayes2")), "joineRML", as.character(packageVersion("joineRML")), "\n")
t0 <- Sys.time()
m1 <- lme(lbil ~ year, random = ~ year | id, data = pbc)
m2 <- lme(albumin ~ year, random = ~ year | id, data = pbc)
cox <- coxph(Surv(years, status2) ~ 1, data = pbc.id)
set.seed(2026)
fit <- jm(cox, list(m1, m2), time_var = "year", n_iter = 6000L, n_burnin = 1000L, n_chains = 2L, cores = 2L)
s <- summary(fit)
print(s$Survival); print(s$Outcome1); print(s$Outcome2); print(s$D)
cat("JMbayes2 time", format(Sys.time() - t0), "\n")
t0 <- Sys.time()
set.seed(2026)
jr <- mjoint(formLongFixed = list("lbil" = lbil ~ year, "alb" = albumin ~ year),
             formLongRandom = list("lbil" = ~ year | id, "alb" = ~ year | id),
             formSurv = Surv(years, status2) ~ 1, data = pbc, survData = pbc.id,
             timeVar = "year", verbose = FALSE)
print(summary(jr))
cat("joineRML time", format(Sys.time() - t0), "\n")
