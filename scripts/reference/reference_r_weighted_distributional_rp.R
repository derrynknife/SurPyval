# Reference values for distributional and Royston-Parmar regression with
# sampling weights (WeibullDistributional.fit(..., w=) and
# RoystonParmarRegression.fit(..., w=)), printed for
# surpyval/tests/reference/test_weighted_distributional_rp.py.
#
#     Rscript scripts/reference/reference_r_weighted_distributional_rp.R
#
# Written with R 4.6.2, flexsurv 2.3.2 and numDeriv (the versions are
# printed). The data are surpyval.tests._helpers.weighted_reference_data(),
# typed out below: right censored, a late entry on some rows.
#
# The point estimates are flexsurv's, fitted with `weights = w` (the
# weighted log-likelihood sum w_i l_i). flexsurv's standard errors with
# weights are model based (the inverse of the weighted Hessian), so the
# robust (sandwich) covariance is computed here by hand, HC0 with no
# small-sample factor: each row's score s_i by numDeriv on its own
# log-likelihood, the bread B the inverse of the numDeriv Hessian of the
# weighted negative log-likelihood, and V = B (sum_i w_i^2 s_i s_i') B.
#
# Royston-Parmar knots: flexsurvspline places its default knots at
# weighted quantiles of the log event times by an approximate expansion
# (each weight rounded to a multiple of 1 / (100 N)), which can move them
# from the unweighted fit's even at w = 1 (on these data the three
# coincide; the last lines print them). SurPyval places them on the event times
# counted by n alone (not weighted); the fits here are given those knots
# (type 7 quantiles of the log event times, checked against SurPyval's).
suppressPackageStartupMessages({library(survival); library(flexsurv); library(numDeriv)})
cat("R", R.version$major, ".", R.version$minor, " flexsurv ",
    as.character(packageVersion("flexsurv")), " numDeriv ",
    as.character(packageVersion("numDeriv")), "\n", sep = "")

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
tl <- c(0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.9, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.5, 0.6, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1,
        0.4)
df <- data.frame(x, d, w, z1, z2, tl)
ctl <- list(reltol = 1e-14, maxit = 10000)

show <- function(label, v) {
  cat(sprintf("%-22s %s\n", label, paste(sprintf("%.10g", v), collapse = ", ")))
}

sandwich_se <- function(ll_rows, est) {
  S <- jacobian(ll_rows, est)
  H <- hessian(function(p) -sum(w * ll_rows(p)), est)
  B <- solve(H)
  M <- t(S) %*% (S * w^2)
  list(robust = sqrt(diag(B %*% M %*% B)), model = sqrt(diag(B)))
}

# -- Weibull distributional: log scale ~ z1 + z2, log shape ~ z1 ----------
fw <- flexsurvreg(Surv(tl, x, d) ~ z1 + z2, anc = list(shape = ~ z1),
                  data = df, dist = "weibull", weights = w, control = ctl)
est <- fw$coefficients  # shape, scale, z1, z2, shape(z1); logs of the first two
ll_weibull <- function(p) {
  shape <- exp(p[1] + p[5] * z1)
  scale <- exp(p[2] + p[3] * z1 + p[4] * z2)
  f <- dweibull(x, shape, scale, log = TRUE)
  s <- pweibull(x, shape, scale, lower.tail = FALSE, log.p = TRUE)
  s0 <- pweibull(tl, shape, scale, lower.tail = FALSE, log.p = TRUE)
  d * f + (1 - d) * s - s0
}
se <- sandwich_se(ll_weibull, est)
show("WD names", seq_along(est))
cat("WD order: log shape, log scale, scale z1, scale z2, shape z1\n")
show("WD est", est)
show("WD loglik", fw$loglik)
show("WD loglik check", sum(w * ll_weibull(est)))
show("WD robust se", se$robust)
show("WD model se", se$model)
show("WD flexsurv se", sqrt(diag(fw$cov)))

# -- Royston-Parmar, hazard and odds scales, one internal knot (df = 2) ---
ev <- log(x[d == 1])
knots <- quantile(ev, c(0, 0.5, 1), type = 7, names = FALSE)
show("RP knots", knots)
for (sc in c("hazard", "odds")) {
  fr <- flexsurvspline(Surv(tl, x, d) ~ z1 + z2, data = df, weights = w,
                       knots = knots[2], bknots = knots[c(1, 3)],
                       scale = sc, control = ctl)
  est <- fr$coefficients  # gamma0, gamma1, gamma2, z1, z2
  ll_rp <- function(p) {
    g <- cbind(p[1] + p[4] * z1 + p[5] * z2, p[2], p[3])
    f <- dsurvspline(x, gamma = g, knots = fr$knots, scale = sc, log = TRUE)
    s <- psurvspline(x, gamma = g, knots = fr$knots, scale = sc,
                     lower.tail = FALSE, log.p = TRUE)
    s0 <- ifelse(tl > 0,
                 psurvspline(pmax(tl, 1e-300), gamma = g, knots = fr$knots,
                             scale = sc, lower.tail = FALSE, log.p = TRUE),
                 0)
    d * f + (1 - d) * s - s0
  }
  se <- sandwich_se(ll_rp, est)
  tag <- if (sc == "hazard") "RPH" else "RPO"
  show(paste(tag, "est"), est)
  show(paste(tag, "loglik"), fr$loglik)
  show(paste(tag, "loglik check"), sum(w * ll_rp(est)))
  show(paste(tag, "robust se"), se$robust)
  show(paste(tag, "model se"), se$model)
  show(paste(tag, "flexsurv se"), sqrt(diag(fr$cov)))
}
# flexsurvspline's own default knots with w (for the record): not the
# unweighted ones even at w = 1.
fd <- flexsurvspline(Surv(tl, x, d) ~ z1 + z2, data = df, weights = w, k = 1)
show("RP default knots w", fd$knots)
fd1 <- flexsurvspline(Surv(tl, x, d) ~ z1 + z2, data = df,
                      weights = rep(1, 40), k = 1)
show("RP default knots w=1", fd1$knots)
f0 <- flexsurvspline(Surv(tl, x, d) ~ z1 + z2, data = df, k = 1)
show("RP default knots", f0$knots)
