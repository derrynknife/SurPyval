# Reference values for the competing-risks regressions with sampling
# weights (CompetingRisksProportionalHazards.fit(..., w=) and
# FineGray.fit(..., w=)), printed for
# surpyval/tests/univariate/competing_risks/test_weighted_competing_regression.py.
#
#     Rscript scripts/reference/reference_r_weighted_competing_regression.R
#
# Written with R 4.6.2, survival 3.8.12 and cmprsk 2.2.12 (the versions
# are printed).
#
# The data are surpyval.tests._helpers.weighted_reference_data(), typed
# out below (without its late entries), with two causes: a failure is
# cause 2 where its row number (from 1) is 1, 4, 7, ... and cause 1
# otherwise.
#
# Cause-specific Cox: coxph(Surv(x, cause == k) ~ z1 + z2, weights = w,
# robust = TRUE), each tie method; the coefficients and the robust
# standard errors.
#
# Fine-Gray: cmprsk::crr has no weights argument, and
# survival::finegray(..., weights = w) multiplies the user's weights into
# its IPCW weights but takes the censoring distribution G from the
# *unweighted* Kaplan-Meier (its survfit call has no weights). SurPyval
# takes G from the weighted Kaplan-Meier of the censoring times (so that
# whole-number weights give the fit of the rows counted that many times),
# so the counting-process data are built here directly: every unit is at
# risk on (0, x] with weight w; a unit that failed from the other cause
# is also at risk at each later event time t of the cause, on (t_prev, t],
# with weight w G(t-) / G(x-), G the weighted Kaplan-Meier of the
# censoring (survfit(Surv(x, censored) ~ 1, weights = w)), taken just
# before each time as cmprsk::crr does. The fit is
# coxph(Surv(start, stop, status) ~ z1 + z2, weights = wt, cluster = id,
# ties = "breslow"): its coefficients and the robust standard errors
# clustered by unit, which hold G fixed. The same construction with w = 1
# is first checked against crr (it stops if they differ), and finegray's
# own weighted fit (the unweighted G) is printed for comparison.
suppressPackageStartupMessages({
  library(survival)
  library(cmprsk)
})
cat("R", R.version$major, ".", R.version$minor, " survival ",
    as.character(packageVersion("survival")), " cmprsk ",
    as.character(packageVersion("cmprsk")), "\n", sep = "")

x <- c(2.9, 5.8, 4.3, 3.7, 8.3, 2.7, 4.3, 2.2, 1.0, 3.8, 8.8, 8.7, 6.9,
       2.0, 3.2, 3.2, 1.7, 6.3, 6.6, 14.5, 2.1, 2.3, 4.3, 8.3, 0.8, 7.8,
       2.9, 6.1, 7.3, 9.6, 0.1, 3.6, 3.7, 6.1, 1.0, 1.6, 3.9, 3.2, 1.1,
       8.3)
d <- c(1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
       1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0)
w <- c(1.82, 1.39, 2.27, 1.96, 0.34, 2.47, 1.59, 0.58, 1.79, 0.45, 1.18,
       1.56, 2.73, 2.24, 2.29, 1.96, 1.34, 2.88, 0.77, 1.23, 1.89, 2.03,
       1.67, 2.51, 0.99, 1.73, 2.96, 0.36, 1.33, 0.87, 1.32, 1.61, 1.57,
       1.36, 0.81, 2.89, 2.25, 1.37, 1.01, 0.53)
z1 <- c(0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0)
z2 <- c(1.3, 0.36, -1.35, -1.95, -0.22, 0.76, -0.02, 0.05, 0.02, 0.74,
        -0.47, -0.32, 0.54, 1.19, 0.19, 0.07, 1.69, -0.24, -1.33, -1.1,
        0.2, 0.89, 0.76, -1.8, 0.38, -0.98, -0.31, -1.37, -1.09, -1.58,
        -0.21, 0.09, 1.69, 0.56, -0.05, 0.71, 0.38, -0.55, 1.21, -0.69)
row <- seq_along(x)
cause <- ifelse(d == 1, ifelse((row - 1) %% 3 == 0, 2, 1), 0)
dat <- data.frame(id = row, x = x, cause = cause, w = w, z1 = z1, z2 = z2)

show <- function(label, values) {
  cat(label, sprintf("%.15g", values), "\n")
}

# --- Cause-specific Cox -----------------------------------------------
for (k in 1:2) {
  for (ties in c("efron", "breslow")) {
    fit <- coxph(Surv(x, cause == k) ~ z1 + z2, data = dat, weights = w,
                 robust = TRUE, ties = ties)
    show(sprintf("cox cause %d %s coef", k, ties), coef(fit))
    show(sprintf("cox cause %d %s se  ", k, ties), sqrt(diag(vcov(fit))))
  }
}

# --- Fine-Gray --------------------------------------------------------
# The counting-process data of cause k with unit weights wu, the
# censoring distribution the Kaplan-Meier weighted by wu.
fg_data <- function(k, wu) {
  sf <- survfit(Surv(x, cause == 0) ~ 1, data = dat, weights = wu)
  G_before <- function(t) {
    i <- findInterval(t, sf$time, left.open = TRUE)
    c(1, sf$surv)[i + 1]
  }
  times <- sort(unique(dat$x[dat$cause == k]))
  rows <- list()
  for (i in seq_len(nrow(dat))) {
    rows[[length(rows) + 1]] <- data.frame(
      id = i, start = 0, stop = dat$x[i],
      status = as.numeric(dat$cause[i] == k), wt = wu[i],
      z1 = dat$z1[i], z2 = dat$z2[i])
    if (dat$cause[i] != 0 && dat$cause[i] != k) {
      later <- times[times > dat$x[i]]
      if (length(later)) {
        rows[[length(rows) + 1]] <- data.frame(
          id = i, start = c(dat$x[i], head(later, -1)), stop = later,
          status = 0, wt = wu[i] * G_before(later) / G_before(dat$x[i]),
          z1 = dat$z1[i], z2 = dat$z2[i])
      }
    }
  }
  do.call(rbind, rows)
}

for (k in 1:2) {
  # The construction against crr, unweighted
  fg1 <- fg_data(k, rep(1, nrow(dat)))
  fit1 <- coxph(Surv(start, stop, status) ~ z1 + z2, data = fg1,
                weights = wt, ties = "breslow")
  ref <- crr(dat$x, dat$cause, cbind(dat$z1, dat$z2), failcode = k,
             cencode = 0)
  stopifnot(max(abs(coef(fit1) - ref$coef)) < 1e-5)
  show(sprintf("fg cause %d unweighted coef", k), coef(fit1))

  fgw <- fg_data(k, w)
  fit <- coxph(Surv(start, stop, status) ~ z1 + z2, data = fgw,
               weights = wt, cluster = id, ties = "breslow")
  show(sprintf("fg cause %d coef", k), coef(fit))
  show(sprintf("fg cause %d se  ", k), sqrt(diag(vcov(fit))))

  # survival::finegray's own weighted fit, with the unweighted G
  fgd <- finegray(Surv(x, factor(cause)) ~ ., data = dat[, -1],
                  etype = k, weights = w)
  fgd$id <- rep(NA, nrow(fgd))
  fitfg <- coxph(Surv(fgstart, fgstop, fgstatus) ~ z1 + z2, data = fgd,
                 weights = fgwt, ties = "breslow")
  show(sprintf("finegray cause %d coef (unweighted G)", k), coef(fitfg))
}
