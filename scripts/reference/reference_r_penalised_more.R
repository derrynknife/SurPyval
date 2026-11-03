# The reference values of the penalised fits in
# surpyval/tests/univariate/regression/test_penalised_more.py: R's glmnet
# for the cause-specific Cox model (one penalised Cox fit per cause, the
# other causes censored) on the PBC data at enrolment, and for the
# proportional intensity HPP (a Poisson regression of each subject's
# events with the log of its follow-up as offset) on the Rossi data.
#
#     Rscript scripts/reference/reference_r_penalised_more.R \
#         surpyval/datasets
#
# Run with R 4.6.2, glmnet 5.2 and survival 3.8.12. Each fit follows a
# path of 60 penalties from 1 down to the target (glmnet's warm starts)
# with a threshold of 1e-30 (at 1e-20 the transplant cause's lasso
# stops 1e-7 short on the albumin coefficient), and prints the last
# coefficients.
suppressMessages({library(glmnet); library(survival)})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "surpyval/datasets"
path_to <- function(lambda) exp(seq(log(1), log(lambda), length.out = 60))
ctrl <- list(thresh = 1e-30, maxit = 1e8)
show <- function(label, b) {
  cat(label, ": [", paste(sprintf("%.10g", b), collapse = ", "), "]\n", sep = "")
}

# Cause-specific Cox on the PBC data at enrolment (one row per patient).
pbc <- read.csv(file.path(dir, "pbc2.csv"))
pbc <- pbc[!duplicated(pbc$id), ]
X <- cbind(
  age = pbc$age,
  female = as.numeric(pbc$sex == "female"),
  drug = as.numeric(pbc$drug == "D-penicil"),
  logbili = log(pbc$serBilir),
  albumin = pbc$albumin
)
for (ties in c("breslow", "efron")) {
  for (setting in list(c(0.05, 0), c(0.05, 0.5), c(0.02, 1))) {
    lambda <- setting[1]; alpha <- setting[2]
    for (cause in c("dead", "transplanted")) {
      y <- Surv(pbc$years, as.numeric(pbc$status == cause))
      f <- glmnet(X, y, family = "cox", alpha = alpha,
                  lambda = path_to(lambda), cox.ties = ties, control = ctrl)
      show(sprintf("cox %s lambda=%g alpha=%g cause=%s", ties, lambda,
                   alpha, cause), f$beta[, ncol(f$beta)])
    }
  }
}

# Proportional intensity HPP on the Rossi data: one subject per row, its
# arrests (0 or 1) by the end of its follow-up, offset log(week).
rossi <- read.csv(file.path(dir, "rossi.csv"))
cols <- c("fin", "age", "race", "wexp", "mar", "paro", "prio")
X <- as.matrix(rossi[, cols])
for (setting in list(c(0.05, 0), c(0.02, 0.5), c(0.01, 1))) {
  lambda <- setting[1]; alpha <- setting[2]
  f <- glmnet(X, rossi$arrest, family = "poisson", alpha = alpha,
              offset = log(rossi$week), lambda = path_to(lambda),
              control = ctrl)
  show(sprintf("hpp lambda=%g alpha=%g intercept", lambda, alpha),
       f$a0[length(f$a0)])
  show(sprintf("hpp lambda=%g alpha=%g", lambda, alpha),
       f$beta[, ncol(f$beta)])
}
