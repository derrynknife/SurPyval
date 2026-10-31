# The reference values of surpyval/tests/reference/
# test_distributional_references.py: R's gamlss with gamlss.cens (every
# censoring at once) and flexsurv (left truncation), for distributional
# regression (covariates on every parameter). It prints the values the
# test holds; the data are the test's L, R, Z1, Z2 and TL (NA for an
# infinite end).
#
#     Rscript scripts/reference/reference_r_distributional.R
#
# Run with R 4.4.3, gamlss 5.5.5, gamlss.cens 5.0.7 and flexsurv 2.3.2.
suppressMessages({library(gamlss); library(gamlss.cens); library(flexsurv); library(survival)})
L <- c(
  8.0, 2.39, 4.32, 1.37, NA, 3.02, 10.2, 1.15, 7.2, 2.0, 4.0, 5.2, 6.25,
  1.59, 2.0, 7.96, 12.97, 4.72, 3.95, 1.43, NA, 4.0, 4.0, 1.45, 11.76,
  4.11, 1.26, 6.54, 0.39, NA, 5.27, 0.09, 2.0, 1.86, 7.16, 5.71, 5.07,
  1.68, 2.24, 5.26, 23.59, 2.0, 5.45, 14.88, 13.38, 11.08, 5.3, 7.36,
  16.42, NA, 4.15, 0.39, 9.41, 2.23, 3.86, 4.56, 7.79, 4.0, 2.07, 3.87,
  4.81, 4.0, 0.94, 1.38, 14.0, NA, 3.37, 3.73, NA, 16.1, 3.73, 4.31, NA,
  0.79, 2.45, 5.64, 13.26, 3.7, 14.0, 6.83
)
R <- c(
  10.0, 2.39, 4.32, NA, 2.0, 3.02, NA, 1.15, NA, 4.0, 6.0, NA, 6.25, NA,
  4.0, 7.96, 12.97, 4.72, NA, 1.43, 26.41, 6.0, 6.0, 1.45, 11.76, 4.11,
  NA, NA, NA, 4.53, 5.27, NA, 4.0, NA, 7.16, NA, NA, NA, 2.24, 5.26, NA,
  4.0, 5.45, 14.88, NA, 11.08, 5.3, 7.36, NA, 2.0, 4.15, 0.39, NA, NA,
  3.86, 4.56, NA, 6.0, NA, 3.87, 4.81, 6.0, 0.94, NA, 16.0, 2.0, 3.37,
  3.73, 5.3, NA, NA, NA, 3.5, 0.79, NA, 5.64, 13.26, NA, 16.0, NA
)
z1 <- c(
  0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
  1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
  1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
  1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0,
  1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
)
z2 <- c(
  -2.51, 0.02, 0.09, -0.68, -0.44, -0.47, -0.58, -0.4, 0.12, -0.6, 0.02,
  0.19, -1.02, 1.44, 0.45, 0.43, 0.32, 0.65, 1.56, -0.43, -0.97, 1.13,
  1.4, -0.8, 1.68, -0.96, -0.06, -0.15, 0.64, -0.14, -0.08, 2.69, -0.86,
  -1.0, 1.04, -1.41, 0.04, 2.29, -0.42, 1.13, 1.56, -1.53, 0.65, 1.38,
  -0.93, -1.11, 0.81, 0.94, 1.27, -0.7, -0.15, -1.28, 2.27, 1.24, -0.74,
  0.31, 0.91, -0.53, 0.7, -0.96, -1.8, -0.3, -1.38, -1.76, 0.18, 0.54,
  1.01, -2.58, -0.87, 0.76, 0.68, -1.01, 0.39, 1.25, 2.64, -0.57, 0.1,
  0.67, 0.42, 0.61
)
tl <- c(
  0.0, 0.0, 0.0, 0.0, 0.0, 0.91, 3.06, 0.0, 0.0, 0.0, 0.0, 1.56, 0.0, 0.0,
  0.0, 0.0, 3.89, 0.0, 1.18, 0.43, 0.0, 1.2, 1.2, 0.44, 0.0, 1.23, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, 4.01, 0.0, 0.0, 0.0, 4.93, 0.0, 0.0, 0.0, 2.82, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, 1.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.01, 0.0, 0.0,
  4.83, 0.0, 1.29, 0.0, 0.24, 0.0, 0.0, 0.0, 1.11, 0.0, 0.0
)
d <- data.frame(L = L, R = R, z1 = z1, z2 = z2, tl = tl)
d$L2 <- d$L; d$R2 <- d$R   # interval2: NA left = left censored, NA right = right censored
y <- Surv(d$L2, d$R2, type = "interval2")
dz <- data.frame(z1 = d$z1, z2 = d$z2)
dz$y <- y
out <- function(tag, m) {
  cat(tag, "ll", sprintf("%.10f", logLik(m)), "\n")
  cat(tag, "mu", sprintf("%.10f", coef(m, "mu")), "\n")
  cat(tag, "sigma", sprintf("%.10f", coef(m, "sigma")), "\n")
}
ctl <- gamlss.control(c.crit = 1e-10, n.cyc = 2000, trace = FALSE)
for (fam in c("WEI", "LOGNO", "NO", "GU", "LO", "RG")) {
  gen.cens(fam, type = "interval")
  fit <- tryCatch(gamlss(y ~ z1 + z2, sigma.formula = ~ z1, data = dz,
                family = get(paste0(fam, "ic"))(), control = ctl, method = mixed(20, 200)),
                error = function(e) {cat(fam, "FAILED", conditionMessage(e), "\n"); NULL})
  if (!is.null(fit)) out(fam, fit)
}
# Left truncation, flexsurv (Weibull, Gamma): the rows with tl > 0
d$event <- ifelse(is.na(d$R), 0, 1)
obs <- !is.na(d$L) & !is.na(d$R) & d$L == d$R
rc <- is.na(d$R)
dd <- d[obs | rc, ]
dd$time <- ifelse(is.na(dd$R), dd$L, dd$R)
fw <- flexsurvreg(Surv(tl, time, event) ~ z1 + z2, anc = list(shape = ~ z1), data = dd,
                  dist = "weibull", control = list(reltol = 1e-14, maxit = 10000))
cat("FSW ll", sprintf("%.10f", fw$loglik), "\n"); cat("FSW est", sprintf("%.10f", fw$res.t[, "est"]), "\n")
cat("FSW se", sprintf("%.10f", sqrt(diag(fw$cov))), "\n")
fg <- flexsurvreg(Surv(tl, time, event) ~ z1 + z2, anc = list(shape = ~ z1), data = dd,
                  dist = "gamma", control = list(reltol = 1e-14, maxit = 10000))
cat("FSG ll", sprintf("%.10f", fg$loglik), "\n"); cat("FSG est", sprintf("%.10f", fg$res.t[, "est"]), "\n")
cat("rows", nrow(dd), "\n")
cat(R.version.string, "gamlss", as.character(packageVersion("gamlss")), "gamlss.cens", as.character(packageVersion("gamlss.cens")), "flexsurv", as.character(packageVersion("flexsurv")), "\n")
