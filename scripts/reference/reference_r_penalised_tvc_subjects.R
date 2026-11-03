# The reference values of the subject-counting penalised start-stop Cox
# fit (CoxPH.fit_tvc_penalised) in
# surpyval/tests/univariate/regression/test_penalised_more.py: R's glmnet
# on the Rossi person-week rows, with the objective rescaled to count
# subjects. fit_tvc_penalised minimises
#
#     nll / N_subjects + lambda * P(b / s),
#
# with s each covariate's standard deviation over the subjects (divisor
# N_subjects) of the subject's covariate averaged over its follow-up
# (rows weighted by stop - start). glmnet minimises nll / N_rows +
# lambda' * P(b'), so the same fit is glmnet with standardize = FALSE on
# the columns X / s, lambda' = lambda * N_subjects / N_rows, and
# b = b' / s.
#
#     Rscript scripts/reference/reference_r_penalised_tvc_subjects.R \
#         surpyval/datasets
#
# Run with R 4.6.2, glmnet 5.2 and survival 3.8.12, threshold 1e-30.
suppressMessages({library(glmnet); library(survival)})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "surpyval/datasets"
tv <- read.csv(file.path(dir, "rossi_tv.csv"))
X <- cbind(fin = as.numeric(tv$fin == "yes"), age = tv$age, prio = tv$prio,
           employed = as.numeric(tv$employed == "yes"))
len <- tv$stop - tv$start
avg <- apply(X, 2, function(z) tapply(len * z, tv$id, sum) / tapply(len, tv$id, sum))
Ns <- nrow(avg); Nr <- nrow(X)
s <- apply(avg, 2, function(z) sqrt(mean((z - mean(z))^2)))
s[s == 0] <- 1
y <- Surv(tv$start, tv$stop, tv$event)
Xs <- sweep(X, 2, s, "/")
# (The lasso at 0.0005 is left out: with this threshold glmnet ran for
# more than 13 minutes on it without finishing.)
for (setting in list(c("breslow", 0.001, 0), c("efron", 0.001, 0.5))) {
  ties <- setting[1]; lambda <- as.numeric(setting[2])
  alpha <- as.numeric(setting[3])
  target <- lambda * Ns / Nr
  path <- exp(seq(log(1), log(target), length.out = 60))
  f <- glmnet(Xs, y, family = "cox", alpha = alpha, lambda = path,
              standardize = FALSE, cox.ties = ties,
              control = list(thresh = 1e-30, maxit = 1e8))
  b <- f$beta[, ncol(f$beta)] / s
  cat(sprintf("%s lambda=%g alpha=%g: [%s]\n", ties, lambda, alpha,
              paste(sprintf("%.10g", b), collapse = ", ")))
}
