# Reference results from R for the GeneralLinear life model,
# surpyval/tests/univariate/regression/test_general_linear.py, which
# holds the values this prints.
#
# AcceleratedLife(Normal, GeneralLinear) is mu = a + b'Z with the time
# itself (not its log) normal: survival::survreg(Surv(y, status) ~ z1 +
# z2, dist = "gaussian"). "logistic" and "extreme" (the smallest extreme
# value, SurPyval's Gumbel) likewise. survreg's intercept and
# coefficients are a and b with the same sign, and its scale is sigma.
#
# The data are deterministic (no random numbers), so the test builds the
# same 60 rows in Python: two covariates, normal scores for the noise,
# times on both sides of zero and right censoring at a varying time.
#
#     Rscript scripts/reference/reference_r_general_linear.R
#
# Needs R >= 4.3 with survival.

suppressPackageStartupMessages(library(survival))

i <- 1:60
z1 <- (i %% 7) - 3
z2 <- round(sin(i), 3)
eps <- qnorm(((i * 0.6180339887) %% 1) * 0.98 + 0.01)
y <- 2 + 1.5 * z1 - 2.5 * z2 + 3 * eps
cens <- 4 + 2 * cos(i)
status <- as.integer(y <= cens)
time <- pmin(y, cens)
d <- data.frame(time, status, z1, z2)

for (dist in c("gaussian", "logistic", "extreme")) {
    fit <- survreg(Surv(time, status) ~ z1 + z2, data = d, dist = dist,
                   control = survreg.control(rel.tolerance = 1e-12,
                                             maxiter = 200))
    cat(sprintf(
        "%s: a = %.12g, b1 = %.12g, b2 = %.12g, sigma = %.12g, loglik = %.12g\n",
        dist, coef(fit)[1], coef(fit)[2], coef(fit)[3], fit$scale,
        fit$loglik[2]
    ))
}
cat(sprintf("events = %d of %d\n", sum(status), length(status)))
