# Reference values for the weighted penalised Cox fit,
# surpyval.CoxPH.fit_penalised(..., w=), printed for
# surpyval/tests/univariate/regression/test_penalised_weighted.py.
#
#     Rscript scripts/reference/reference_r_weighted_penalised.R \
#         surpyval/datasets
#
# Written with R 4.6.2, glmnet 5.2 and survival 3.8.12 (the versions are
# printed).
#
# glmnet's Cox objective with weights is
#     -(1 / sum(w)) * l_w(beta) + lambda * penalty(beta),
# l_w the weighted partial log-likelihood (each row's death and risk-set
# terms multiplied by its weight, ties by Breslow's or Efron's rule as
# survival::coxph takes them with weights), and with standardize = TRUE
# the columns are divided by their w-weighted standard deviations with
# divisor sum(w) (glmnet scales the weights to sum to 1 first, which
# changes neither). That is fit_penalised's objective with N = sum(n w).
#
# 1. The Rossi data (one row per subject) with the deterministic weights
#    w = 0.5 + ((row * 37) %% 11) / 5 (0.5 to 2.5), both tie rules, a
#    ridge, an elastic-net and a lasso penalty: glmnet on a path of 60
#    penalties from 1 down to the target (its warm starts), threshold
#    1e-30, as reference_r_penalised_more.R.
# 2. The forty weighted rows of surpyval.tests._helpers
#    .weighted_reference_data() (typed out below) with their delayed
#    entries, Surv(tl, x, d): glmnet counts rows, as fit_penalised with
#    tl does.
# 3. A check that glmnet's weighted Efron rule is survival's: the ridge
#    fit of 1. by coxph(..., weights = w, ties = "efron") with a ridge()
#    term on the standardised columns, theta = lambda * sum(w) (coxph's
#    penalty is theta / 2 * sum(b^2) on the unscaled log-likelihood).
suppressMessages({library(glmnet); library(survival)})
cat("R ", R.version$major, ".", R.version$minor, " glmnet ",
    as.character(packageVersion("glmnet")), " survival ",
    as.character(packageVersion("survival")), "\n", sep = "")
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "surpyval/datasets"
path_to <- function(lambda) exp(seq(log(1), log(lambda), length.out = 60))
ctrl <- list(thresh = 1e-30, maxit = 1e8)
show <- function(label, b) {
  cat(label, ": [", paste(sprintf("%.12g", b), collapse = ", "), "]\n",
      sep = "")
}
settings <- list(c(0.05, 0), c(0.05, 0.5), c(0.02, 1))

rossi <- read.csv(file.path(dir, "rossi.csv"))
cols <- c("fin", "age", "race", "wexp", "mar", "paro", "prio")
X <- as.matrix(rossi[, cols])
w <- 0.5 + ((seq_len(nrow(rossi)) * 37) %% 11) / 5
y <- Surv(rossi$week, rossi$arrest)
for (ties in c("breslow", "efron")) {
  for (setting in settings) {
    lambda <- setting[1]; alpha <- setting[2]
    f <- glmnet(X, y, family = "cox", weights = w, alpha = alpha,
                lambda = path_to(lambda), cox.ties = ties, control = ctrl)
    show(sprintf("rossi %s lambda=%g alpha=%g", ties, lambda, alpha),
         f$beta[, ncol(f$beta)])
  }
}

# 2. Delayed entry
x <- c(2.9, 5.8, 4.3, 3.7, 8.3, 2.7, 4.3, 2.2, 1.0, 3.8, 8.8, 8.7, 6.9,
       2.0, 3.2, 3.2, 1.7, 6.3, 6.6, 14.5, 2.1, 2.3, 4.3, 8.3, 0.8, 7.8,
       2.9, 6.1, 7.3, 9.6, 0.1, 3.6, 3.7, 6.1, 1.0, 1.6, 3.9, 3.2, 1.1,
       8.3)
d <- c(1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1,
       1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0)
wr <- c(1.82, 1.39, 2.27, 1.96, 0.34, 2.47, 1.59, 0.58, 1.79, 0.45, 1.18,
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
Xr <- cbind(z1 = z1, z2 = z2)
yr <- Surv(tl, x, d)
for (ties in c("breslow", "efron")) {
  for (setting in list(c(0.05, 0), c(0.1, 1))) {
    lambda <- setting[1]; alpha <- setting[2]
    f <- glmnet(Xr, yr, family = "cox", weights = wr, alpha = alpha,
                lambda = path_to(lambda), cox.ties = ties, control = ctrl)
    show(sprintf("entry %s lambda=%g alpha=%g", ties, lambda, alpha),
         f$beta[, ncol(f$beta)])
  }
}

# 3. glmnet's weighted Efron against survival's
m <- colSums(w * X) / sum(w)
s <- sqrt(colSums(w * sweep(X, 2, m)^2) / sum(w))
Xs <- sweep(X, 2, s, "/")
fit <- coxph(y ~ ridge(Xs, theta = 0.05 * sum(w), scale = FALSE),
             weights = w, ties = "efron", robust = FALSE,
             control = coxph.control(eps = 1e-12, toler.chol = 1e-15,
                                     iter.max = 100))
show("coxph ridge efron lambda=0.05", coef(fit) / s)
