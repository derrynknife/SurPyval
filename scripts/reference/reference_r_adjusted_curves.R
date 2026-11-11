# Reference results from R for directly adjusted (standardised) survival
# curves: a regression model's predicted curves averaged over the sample,
# with the group set to each level.
#
# Writes surpyval/tests/reference/data/r_adjusted_curves.json, with the
# software, its version, the call and the values of each entry, as the
# other reference scripts do. Generated with R 4.6.2, survival 3.8.12
# and riskRegression 2026.3.11.
#
#     Rscript scripts/reference/reference_r_adjusted_curves.R
#
# Run make_fixtures.py first. Needs survival, riskRegression and
# jsonlite. Re-running reproduces the file byte for byte.
#
# The data are lung (survival::lung) without the one row whose ph.ecog is
# missing; the group is sex (1, 2) and the other covariates age and
# ph.ecog. Three entries:
#
# * cox_by_hand: coxph (Efron ties), survfit(fit, newdata = d with sex
#   set to g) and the row means of its survival, at fixed times; and the
#   area under each averaged step curve to tau = 700 (the restricted
#   mean).
# * weibull_by_hand: survreg(dist = "weibull"); the averaged survival of
#   each level, their difference, and the restricted means to tau = 700
#   (integrate, rel.tol 1e-10) with delta-method standard errors: a
#   central-difference gradient in (coefficients, log scale) and
#   vcov(fit). These hold the covariates fixed; the "sampled" standard
#   errors add sum((v_i - mean)^2) / n^2, v_i the unit's own prediction.
# * cox_ate: riskRegression::ate(fit, treatment = "sex", ...), the
#   G-formula with influence-function standard errors (which count the
#   covariates as sampled, and use the Cox model's robust influence
#   function rather than its model-based covariance): the average risks
#   (1 - survival), their difference and the standard errors.

suppressPackageStartupMessages({
    library(survival)
    library(riskRegression)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
d <- as.data.frame(fx$lung$columns)
d$status <- 1 - d$c
d <- d[!is.na(d$ph_ecog), c("time", "status", "sex", "age", "ph_ecog")]
stopifnot(nrow(d) == 227)
n <- nrow(d)
times <- c(100, 200, 365, 500, 700)
tau <- 700
levels <- c(1, 2)

refs <- list()

# -- Cox, by hand -----------------------------------------------------------
cox <- coxph(Surv(time, status) ~ sex + age + ph_ecog, data = d,
             ties = "efron", x = TRUE)
cox_curve <- function(g) {
    nd <- d
    nd$sex <- g
    s <- survfit(cox, newdata = nd)
    list(time = s$time, surv = rowMeans(s$surv))
}
cox_values <- list(times = times, tau = tau, coef = unname(coef(cox)))
for (g in levels) {
    cv <- cox_curve(g)
    at <- sapply(times, function(t) {
        i <- sum(cv$time <= t)
        if (i == 0) 1 else cv$surv[i]
    })
    # area under the step curve to tau
    keep <- cv$time < tau
    edges <- c(0, cv$time[keep], tau)
    heights <- c(1, cv$surv[keep])
    cox_values[[paste0("sf_", g)]] <- at
    cox_values[[paste0("rmst_", g)]] <- sum(diff(edges) * heights)
}
refs$cox_by_hand <- list(
    fixture = "lung",
    software = "R survival",
    version = as.character(packageVersion("survival")),
    call = paste("coxph(Surv(time, status) ~ sex + age + ph_ecog, data = d,",
                 "ties = 'efron'); rowMeans(survfit(fit, newdata =",
                 "transform(d, sex = g))$surv)"),
    settings = list(ties = "efron",
                    missing = "the one row with ph.ecog missing is dropped"),
    values = cox_values
)

# -- Weibull AFT, by hand ---------------------------------------------------
wb <- survreg(Surv(time, status) ~ sex + age + ph_ecog, data = d,
              dist = "weibull")
theta <- c(coef(wb), log(wb$scale))
V <- vcov(wb)
X <- function(g) cbind(1, g, d$age, d$ph_ecog)
row_sf <- function(th, g, t) {
    lp <- as.vector(X(g) %*% th[1:4])
    exp(-(t / exp(lp))^(1 / exp(th[5])))
}
row_rmst <- function(th, g) {
    lp <- as.vector(X(g) %*% th[1:4])
    sapply(lp, function(l) integrate(
        function(t) exp(-(t / exp(l))^(1 / exp(th[5]))), 0, tau,
        rel.tol = 1e-10, subdivisions = 500)$value)
}
grad <- function(f) {
    sapply(seq_along(theta), function(i) {
        h <- 1e-5 * max(abs(theta[i]), 1)
        e <- replace(numeric(length(theta)), i, h)
        (f(theta + e) - f(theta - e)) / (2 * h)
    })
}
delta_se <- function(f) {
    g <- grad(f)
    sqrt(as.numeric(t(g) %*% V %*% g))
}
spread <- function(v) sum((v - mean(v))^2) / n^2

wb_values <- list(times = times, tau = tau, coef = unname(theta))
for (g in levels) {
    key <- function(s) paste0(s, "_", g)
    wb_values[[key("sf")]] <- sapply(times, function(t) mean(row_sf(theta, g, t)))
    fixed <- sapply(times, function(t) delta_se(function(th) mean(row_sf(th, g, t))))
    wb_values[[key("sf_se_fixed")]] <- fixed
    wb_values[[key("sf_se_sampled")]] <- sqrt(
        fixed^2 + sapply(times, function(t) spread(row_sf(theta, g, t))))
    r <- row_rmst(theta, g)
    wb_values[[key("rmst")]] <- mean(r)
    r_fixed <- delta_se(function(th) mean(row_rmst(th, g)))
    wb_values[[key("rmst_se_fixed")]] <- r_fixed
    wb_values[[key("rmst_se_sampled")]] <- sqrt(r_fixed^2 + spread(r))
}
diff_sf <- function(th, t) row_sf(th, 2, t) - row_sf(th, 1, t)
wb_values$sf_diff <- sapply(times, function(t) mean(diff_sf(theta, t)))
d_fixed <- sapply(times, function(t) delta_se(function(th) mean(diff_sf(th, t))))
wb_values$sf_diff_se_fixed <- d_fixed
wb_values$sf_diff_se_sampled <- sqrt(
    d_fixed^2 + sapply(times, function(t) spread(diff_sf(theta, t))))
diff_rmst <- function(th) row_rmst(th, 2) - row_rmst(th, 1)
wb_values$rmst_diff <- mean(diff_rmst(theta))
dr_fixed <- delta_se(function(th) mean(diff_rmst(th)))
wb_values$rmst_diff_se_fixed <- dr_fixed
wb_values$rmst_diff_se_sampled <- sqrt(dr_fixed^2 + spread(diff_rmst(theta)))
refs$weibull_by_hand <- list(
    fixture = "lung",
    software = "R survival",
    version = as.character(packageVersion("survival")),
    call = paste("survreg(Surv(time, status) ~ sex + age + ph_ecog, data = d,",
                 "dist = 'weibull'); mean over rows of the predicted",
                 "survival with sex set to g; delta method with vcov(fit)"),
    settings = list(
        gradient = "central differences, step 1e-5 * max(|theta|, 1)",
        differences = "level 2 minus level 1",
        missing = "the one row with ph.ecog missing is dropped"),
    values = wb_values
)

# -- riskRegression::ate ----------------------------------------------------
da <- d
da$sex <- factor(da$sex)
cox_f <- coxph(Surv(time, status) ~ sex + age + ph_ecog, data = da,
               ties = "efron", x = TRUE, y = TRUE)
a <- ate(cox_f, treatment = "sex", data = da, times = times, se = TRUE,
         verbose = FALSE)
mr <- as.data.frame(a$meanRisk)
dr <- as.data.frame(a$diffRisk)
pick <- function(df, cond, col) {
    sub <- df[cond, ]
    sub[order(sub$time), col]
}
refs$cox_ate <- list(
    fixture = "lung",
    software = "R riskRegression",
    version = as.character(packageVersion("riskRegression")),
    call = paste("ate(coxph(Surv(time, status) ~ sex + age + ph_ecog,",
                 "data = d, ties = 'efron', x = TRUE, y = TRUE),",
                 "treatment = 'sex', data = d, times = times, se = TRUE)"),
    settings = list(ties = "efron", estimator = "G-formula",
                    se = "influence function",
                    differences = "level 2 minus level 1 (of the risk)"),
    values = list(
        times = times,
        risk_1 = pick(mr, mr$treatment == "1", "estimate"),
        risk_2 = pick(mr, mr$treatment == "2", "estimate"),
        risk_se_1 = pick(mr, mr$treatment == "1", "se"),
        risk_se_2 = pick(mr, mr$treatment == "2", "se"),
        risk_diff = pick(dr, dr$A == "1" & dr$B == "2", "estimate"),
        risk_diff_se = pick(dr, dr$A == "1" & dr$B == "2", "se")
    )
)

out <- list(
    generator = "scripts/reference/reference_r_adjusted_curves.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_adjusted_curves.json"))
cat("wrote", file.path(data_dir, "r_adjusted_curves.json"), "\n")
