# Reference values for the log-rank and Wilcoxon-type tests of
# interval-censored data (surpyval.interval_logrank), printed for
# surpyval/tests/reference/test_interval_logrank.py.
#
#     Rscript scripts/reference/reference_r_interval_logrank.R
#
# Needs the interval package (CRAN, GPL >= 2) and Icens (Bioconductor:
# BiocManager::install("Icens")). Written with R 4.6.2, interval 1.1.1.0,
# Icens 1.84.0 and perm 1.0.0.4 (the versions are printed).
#
# Data:
# - bcos, the breast cosmesis data of Finkelstein and Wolfe (1985), as
#   shipped with the interval package (published study data; 46 units
#   on radiotherapy, 48 on radiotherapy and chemotherapy).
# - sim3, 45 units in three groups simulated below with a fixed seed:
#   Weibull times, a mix of exact, right, left and interval-censored rows
#   (inspections every 3 time units). The values are typed out so the
#   reference does not depend on R's random number generator.
#
# ictest's NPMLE (icfit) stops when its Karush-Kuhn-Tucker gap is below
# epsilon (default 1e-6). The "tight" references use epsilon = 1e-12, the
# same stopping rule as SurPyval's default tol; the "default" ones show
# how far R's default stop moves the numbers.
#
# Simulation of sim3 (run once; its output is typed into L and R below):
#   set.seed(20261105)
#   n <- 45; g <- rep(c("a", "b", "c"), each = 15)
#   scale <- c(a = 10, b = 14, c = 8)[g]
#   t <- round(rweibull(n, 1.5, scale), 1)
#   kind <- sample(c("exact", "right", "left", "interval"), n, TRUE,
#                  c(.2, .2, .15, .45))
#   L <- R <- t; insp <- 3
#   L[kind == "interval"] <- floor(t[kind == "interval"] / insp) * insp
#   R[kind == "interval"] <- L[kind == "interval"] + insp
#   cen <- round(runif(n, 2, 20), 1)
#   L[kind == "right"] <- pmin(cen[kind == "right"], t[kind == "right"])
#   R[kind == "right"] <- Inf
#   L[kind == "left"] <- 0; R[kind == "left"] <- pmax(t[kind == "left"], 1)
suppressPackageStartupMessages(library(interval))
cat("R", R.version$major, ".", R.version$minor, " interval ",
    as.character(packageVersion("interval")), " Icens ",
    as.character(packageVersion("Icens")), " perm ",
    as.character(packageVersion("perm")), "\n", sep = "")
data(bcos)
L <- c( 11, 3, 3, 3, 6, 3.4, 12, 4.7, 5.3, 7.5, 0, 0, 6, 0, 0, 0, 5.6, 7.7, 24, 6.7, 0, 12, 0, 6.3, 0, 12, 4.6, 6, 7.3, 2, 5.9, 2.2, 0, 0, 3.9, 7.9, 6, 6, 0, 0, 0, 5.3, 15, 3, 6 )
R <- c( Inf, 6, 6, 6, 9, 3.4, 15, Inf, 5.3, 7.5, 7.7, 7.8, 9, 10.5, 3, 16.6, 5.6, Inf, 27, 6.7, 3, 15, 14.8, 6.3, 6.2, 15, Inf, 6, Inf, Inf, Inf, 2.2, 3, 17.9, Inf, 7.9, 9, 9, 3, 3, 4.7, Inf, 18, 6, 9 )
g <- rep(c("a", "b", "c"), each = 15)
tight <- icfitControl(epsilon = 1e-12, maxit = 1e6)
for (eps in c("default", "tight")) {
  ctl <- if (eps == "tight") tight else icfitControl()
  for (sc in c("logrank1", "logrank2", "wmw")) {
    t1 <- ictest(bcos$left, bcos$right, bcos$treatment, scores = sc,
                 method = "pclt", icontrol = ctl)
    cat(sprintf("bcos %-8s %-7s Z=%.15g chisq=%.15g p=%.15g U=%.15g\n", sc, eps,
                t1$statistic, t1$statistic^2, t1$p.value, t1$U[1]))
    t3 <- ictest(L, R, g, scores = sc, method = "pclt", icontrol = ctl)
    cat(sprintf("sim3 %-8s %-7s chisq=%.15g p=%.15g U=%s\n", sc, eps,
                t3$statistic, t3$p.value, paste(sprintf("%.15g", t3$U), collapse=",")))
    t2 <- ictest(L[g != "c"], R[g != "c"], g[g != "c"], scores = sc,
                 method = "pclt", icontrol = ctl)
    cat(sprintf("sim2 %-8s %-7s Z=%.15g chisq=%.15g p=%.15g\n", sc, eps,
                t2$statistic, t2$statistic^2, t2$p.value))
  }
}
# scores themselves (tight), for a unit test of the scores
s <- ictest(bcos$left, bcos$right, bcos$treatment, scores = "logrank1", method = "pclt", icontrol = tight)$scores
cat("bcos logrank1 scores[1:6]:", sprintf("%.15g", s[1:6]), "\n")
cat("bcos fit numit (tight):", icfit(bcos$left, bcos$right, control = tight)$numit, "\n")
# Monte Carlo permutation p-values (perm's exact.mc), many draws
for (sc in c("logrank1", "wmw")) {
  m1 <- ictest(bcos$left, bcos$right, bcos$treatment, scores = sc,
               method = "exact.mc", icontrol = tight,
               mcontrol = mControl(nmc = 99999, tsmethod = "abs"))
  cat(sprintf("bcos MC %s p=%.6f\n", sc, m1$p.value))
  m3 <- ictest(L, R, g, scores = sc, method = "exact.mc", icontrol = tight,
               mcontrol = mControl(nmc = 99999))
  cat(sprintf("sim3 MC %s p=%.6f\n", sc, m3$p.value))
}
