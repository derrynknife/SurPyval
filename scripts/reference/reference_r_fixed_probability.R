# Reference values for the bounds on the probability p of a fitted
# Bernoulli, FixedEventProbability or Binomial (param_cb("p", method=...)),
# printed for
# surpyval/tests/univariate/parametric/test_probability_uncertainty.py.
#
#     Rscript scripts/reference/reference_r_fixed_probability.R
#
# Written with R 4.6.2 and binom 1.1.2 (the versions are printed).
#
# * "exact" (Clopper-Pearson): stats::binom.test(k, n, conf.level = 1 - a)
#   for the two-sided interval, alternative = "less" for the upper bound
#   and "greater" for the lower one.
# * "jeffreys": the equal-tailed quantiles of Beta(k + 1/2, n - k + 1/2),
#   qbeta(a / 2, ...) and qbeta(1 - a / 2, ...), with 0 below at k = 0 and
#   1 above at k = n (Brown, Cai and DasGupta, 2001). Inside (0 < k < n)
#   these are binom::binom.confint(k, n, methods = "bayes", type =
#   "central", prior.shape1 = 0.5, prior.shape2 = 0.5), printed as a
#   cross-check. At k = 0 and k = n binom puts the whole of a in the one
#   open tail; SurPyval keeps a / 2 there, as Clopper-Pearson does, so
#   those rows are from qbeta alone.
# * The variance of p-hat, p (1 - p) / n, the observed information's
#   inverse, is printed for the interior cases.
suppressPackageStartupMessages(library(binom))
cat("R", R.version$major, ".", R.version$minor, " binom ",
    as.character(packageVersion("binom")), "\n", sep = "")

cases <- list(c(0, 10), c(1, 10), c(8, 10), c(10, 10), c(3, 1200),
              c(0, 1200), c(2, 50), c(49, 50))
f <- function(v) paste(format(v, digits = 17), collapse = ", ")

for (case in cases) {
  k <- case[1]
  n <- case[2]
  cat(sprintf("k = %d, n = %d\n", k, n))
  for (a in c(0.05, 0.1)) {
    two <- binom.test(k, n, conf.level = 1 - a)$conf.int
    up <- binom.test(k, n, conf.level = 1 - a, alternative = "less")$conf.int
    lo <- binom.test(k, n, conf.level = 1 - a,
                     alternative = "greater")$conf.int
    cat(sprintf("  alpha %.2f exact two-sided: [%s]\n", a, f(two)))
    cat(sprintf("  alpha %.2f exact upper: %s lower: %s\n", a,
                f(up[2]), f(lo[1])))
    jl <- if (k == 0) 0 else qbeta(a / 2, k + 0.5, n - k + 0.5)
    ju <- if (k == n) 1 else qbeta(1 - a / 2, k + 0.5, n - k + 0.5)
    j1u <- if (k == n) 1 else qbeta(1 - a, k + 0.5, n - k + 0.5)
    j1l <- if (k == 0) 0 else qbeta(a, k + 0.5, n - k + 0.5)
    cat(sprintf("  alpha %.2f jeffreys two-sided: [%s]\n", a, f(c(jl, ju))))
    cat(sprintf("  alpha %.2f jeffreys upper: %s lower: %s\n", a,
                f(j1u), f(j1l)))
    if (k > 0 && k < n) {
      b <- binom.confint(k, n, conf.level = 1 - a, methods = "bayes",
                         type = "central", prior.shape1 = 0.5,
                         prior.shape2 = 0.5)
      cat(sprintf("  alpha %.2f binom.confint bayes: [%s]\n", a,
                  f(c(b$lower, b$upper))))
    }
  }
  if (k > 0 && k < n) {
    p <- k / n
    cat(sprintf("  variance: %s\n", f(p * (1 - p) / n)))
  }
}
