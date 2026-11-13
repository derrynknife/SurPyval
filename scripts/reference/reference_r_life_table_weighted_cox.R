# The reference values of surpyval/tests/reference/
# test_life_table_weighted_cox.py:
#
# - the actuarial life table of KMsurv::lifetab's own example (the
#   weaning data of Klein and Moeschberger 2003, section 5.4), every
#   column;
# - the weighted Cox model's survival at chosen covariates and times,
#   survfit(coxph(..., weights = w), newdata = ...), with Efron's and
#   Breslow's ties, with and without late entry, on the forty weighted
#   reference rows of surpyval/tests/_helpers.py
#   (weighted_reference_data).
#
#     Rscript scripts/reference/reference_r_life_table_weighted_cox.R
#
# Run with R 4.6.2, KMsurv 0.1.6 and survival 3.8.12.
suppressMessages({library(KMsurv); library(survival)})
options(digits = 15)
p <- function(name, v) cat(name, "=", paste(sprintf("%.15g", v), collapse = ", "), "\n")

# -- the life table ---------------------------------------------------------
tis <- c(0, 2, 3, 5, 7, 11, 17, 25, 37, 53, NA)
nsubs <- c(927, 848, 774, 649, 565, 449, 296, 186, 112, 27)
nlost <- c(2, 3, 6, 9, 7, 5, 3, rep(0, 3))
nevent <- c(77, 71, 119, 75, 109, 148, 107, 74, 85, 27)
lt <- lifetab(tis, nsubs[1], nlost, nevent)
for (col in names(lt)) p(col, lt[[col]])

# -- the weighted Cox baseline ------------------------------------------------
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
times <- c(0.5, 1.0, 2.0, 2.9, 3.2, 4.3, 6.0, 8.3, 12.0)
newdata <- data.frame(z1 = c(0, 1, 0), z2 = c(0, 0.5, -1))
for (ties in c("efron", "breslow")) {
  for (entry in c("none", "late")) {
    f <- if (entry == "none") {
      coxph(Surv(x, d) ~ z1 + z2, data = df, weights = w, ties = ties)
    } else {
      coxph(Surv(tl, x, d) ~ z1 + z2, data = df, weights = w, ties = ties)
    }
    s <- summary(survfit(f, newdata = newdata), times = times, extend = TRUE)
    p(paste(ties, entry, "coef"), coef(f))
    for (k in seq_len(nrow(newdata))) p(paste(ties, entry, "sf", k), s$surv[, k])
  }
}
