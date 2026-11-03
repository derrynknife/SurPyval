# The reference values of
# surpyval/tests/univariate/regression/test_penalty_path.py: R's glmnet
# path for the Cox model on the Rossi data (Breslow ties): the largest
# penalty (glmnet's lambda_max, the smallest that sets every coefficient
# to 0; for alpha = 0 glmnet's is the one for alpha = 0.001) and the
# coefficients along a path of 10 penalties down to 0.001 times it.
#
#     Rscript scripts/reference/reference_r_penalty_path.R surpyval/datasets
#
# Run with R 4.6.2, glmnet 5.2 and survival 3.8.12, threshold 1e-30.
suppressMessages({library(glmnet); library(survival)})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "surpyval/datasets"
rossi <- read.csv(file.path(dir, "rossi.csv"))
cols <- c("fin", "age", "race", "wexp", "mar", "paro", "prio")
X <- as.matrix(rossi[, cols])
y <- Surv(rossi$week, rossi$arrest)
for (alpha in c(1, 0.5, 0)) {
  f <- glmnet(X, y, family = "cox", alpha = alpha, nlambda = 10,
              lambda.min.ratio = 1e-3, cox.ties = "breslow",
              control = list(thresh = 1e-30, maxit = 1e8))
  cat(sprintf("alpha=%g lambda: [%s]\n", alpha,
              paste(sprintf("%.10g", f$lambda), collapse = ", ")))
  for (k in seq_along(f$lambda)) {
    cat(sprintf("alpha=%g k=%d: [%s]\n", alpha, k,
                paste(sprintf("%.10g", f$beta[, k]), collapse = ", ")))
  }
}
