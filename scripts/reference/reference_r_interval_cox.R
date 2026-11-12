# Reference results from R for the semi-parametric proportional hazards
# (Cox) model for interval-censored data: icenReg::ic_sp(model = "ph"),
# whose baseline is a non-parametric maximum-likelihood estimate on the
# Turnbull intervals, fitted jointly with the coefficients.
#
# Writes surpyval/tests/reference/data/r_interval_cox.json in the layout
# of the other reference files: for each fit the fixture, the software,
# its version, the call, the settings and the values. Generated with
# R 4.6.2 and icenReg 2.0.16.
#
#     python scripts/reference/make_fixtures.py
#     Rscript scripts/reference/reference_r_interval_cox.R
#
# Run from the repository root, after make_fixtures.py. Needs icenReg and
# jsonlite. The fits are deterministic; the bootstrap standard errors are
# seeded (set.seed(20261113) before each fit with bs_samples = 1000), so
# re-running reproduces the file.
#
# The data (fixtures):
#
# * ic_mice: icenReg's miceData (Hoel and Walburg 1972), 144 mice
#   inspected once each (current status): l = 0 is "had the tumour by u"
#   (left censored), u = Inf "had none at l" (right censored); grp is 1
#   for the germ-free environment ("ge"), 0 for conventional ("ce").
# * ic_diabetes: icenReg's IR_diabetes (Borch-Johnsen et al. 1985), 731
#   patients' times from diabetes to diabetic nephropathy in years,
#   interval censored between visits or exact (left == right); male is 1
#   for a man. Endpoints are whole years and are shared between rows.
# * ic_mixed: simulated here (seed 20261113) and kept as the shared
#   fixture; the script stops if its draws differ. 300 units with a
#   binary z1 and a continuous z2, life Weibull PH, S(t) = exp(-(t / 10)
#   ^ 1.5 exp(0.7 z1 - 0.4 z2)); each unit is inspected from a
#   uniform(0, 2) time every uniform(1, 3) until 25: a failure before the
#   first visit is left censored, one after the last right censored, the
#   rest interval censored between two visits, except a fifth of those,
#   whose time is seen exactly. `exact_right` holds the same lives seen
#   exactly up to the unit's last visit and right censored there (no
#   interval rows), for the comparison with coxph.
#
# icenReg reads [l, u] as a closed interval by default; B = c(0, 1) makes
# every interval (l, u], SurPyval's convention (a row with l == u is an
# exact time either way).
#
# Each entry records the coefficients, the log-likelihood, the Turnbull
# intervals with the survival after each at two covariate rows
# (getSCurves; the first value is 1, the survival before the first
# interval), and the bootstrap standard errors of 1000 case resamples.

suppressPackageStartupMessages({
    library(icenReg)
    library(survival)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))

check <- function(name, d) {
    stored <- as.data.frame(fx[[name]]$columns)
    for (col in names(d)) {
        a <- stored[[col]]
        b <- as.numeric(d[[col]])
        a[is.na(a)] <- Inf
        stopifnot(isTRUE(all.equal(a, b, tolerance = 0)))
    }
}

data(miceData)
mice <- data.frame(l = miceData$l, u = miceData$u,
                   grp = as.numeric(miceData$grp == "ge"))
check("ic_mice", mice)
data(IR_diabetes)
diabetes <- data.frame(left = IR_diabetes$left, right = IR_diabetes$right,
                       male = as.numeric(IR_diabetes$gender == "male"))
check("ic_diabetes", diabetes)

set.seed(20261113)
n <- 300
z1 <- rbinom(n, 1, 0.5)
z2 <- round(rnorm(n), 4)
life <- round(10 * (rexp(n) / exp(0.7 * z1 - 0.4 * z2))^(1 / 1.5), 4)
l <- u <- numeric(n)
last <- numeric(n)
for (i in seq_len(n)) {
    visits <- cumsum(c(runif(1, 0, 2), runif(30, 1, 3)))
    visits <- round(visits[visits <= 25], 4)
    last[i] <- max(visits)
    if (life[i] <= visits[1]) {
        l[i] <- 0; u[i] <- visits[1]
    } else if (life[i] > last[i]) {
        l[i] <- last[i]; u[i] <- Inf
    } else {
        k <- max(which(visits < life[i]))
        l[i] <- visits[k]; u[i] <- visits[k + 1]
    }
}
seen <- runif(n) < 0.2 & l > 0 & is.finite(u)
l[seen] <- life[seen]; u[seen] <- life[seen]
er_time <- pmin(life, last)
er_event <- as.numeric(life <= last)
mixed <- data.frame(l = l, u = u, z1 = z1, z2 = z2, er_time = er_time,
                    er_event = er_event)
check("ic_mixed", mixed)

refs <- list()
add <- function(id, fixture, formula, data, newdata, rows) {
    call <- sprintf("ic_sp(%s, data = data, model = 'ph', B = c(0, 1))",
                    formula)
    fit <- ic_sp(as.formula(formula), data = data, model = "ph",
                 B = c(0, 1))
    set.seed(20261113)
    boot <- ic_sp(as.formula(formula), data = data, model = "ph",
                  B = c(0, 1), bs_samples = 1000)
    s <- getSCurves(fit, newdata = newdata)
    refs[[id]] <<- list(
        fixture = fixture,
        software = "R icenReg",
        version = as.character(packageVersion("icenReg")),
        call = call,
        settings = list(
            rows = rows,
            bootstrap = paste(
                "bs_samples = 1000 after set.seed(20261113): the standard",
                "errors are sqrt(diag(var)) of the refits"
            ),
            curves = paste(
                "getSCurves(fit, newdata): the survival after each",
                "Turnbull interval, after a leading 1"
            ),
            intervals = "Turnbull intervals as [lower, upper]"
        ),
        values = list(
            names = names(coef(fit)),
            coef = unname(coef(fit)),
            loglik = as.numeric(fit$llk),
            bootstrap_se = unname(sqrt(diag(boot$var))),
            lower = unname(s$Tbull_ints[, "lower"]),
            upper = unname(s$Tbull_ints[, "upper"]),
            newdata = as.list(newdata),
            sf_row1 = unname(s$S_curves[[1]]),
            sf_row2 = unname(s$S_curves[[2]])
        )
    )
}

add("mice", "ic_mice", "cbind(l, u) ~ grp", mice,
    data.frame(grp = c(0, 1)), "all")
add("diabetes", "ic_diabetes", "cbind(left, right) ~ male", diabetes,
    data.frame(male = c(0, 1)), "all")
add("mixed", "ic_mixed", "cbind(l, u) ~ z1 + z2", mixed,
    data.frame(z1 = c(0, 1), z2 = c(0, 0.5)), "all")
er <- data.frame(l = mixed$er_time,
                 u = ifelse(mixed$er_event == 1, mixed$er_time, Inf),
                 z1 = z1, z2 = z2)
add("exact_right", "ic_mixed", "cbind(l, u) ~ z1 + z2", er,
    data.frame(z1 = c(0, 1), z2 = c(0, 0.5)),
    "er_time exact where er_event = 1, else right censored there")
# coxph on the same exact and right-censored data, for comparison
for (ties in c("breslow", "efron")) {
    cfit <- coxph(Surv(er_time, er_event) ~ z1 + z2, data = mixed,
                  ties = ties)
    refs[[paste0("exact_right_coxph_", ties)]] <- list(
        fixture = "ic_mixed",
        software = "R survival",
        version = as.character(packageVersion("survival")),
        call = sprintf(
            "coxph(Surv(er_time, er_event) ~ z1 + z2, data = data, ties = '%s')",
            ties
        ),
        settings = list(rows = "all", ties = ties),
        values = list(names = names(coef(cfit)), coef = unname(coef(cfit)),
                      se = unname(sqrt(diag(vcov(cfit)))))
    )
}

out <- list(
    generator = "scripts/reference/reference_r_interval_cox.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_interval_cox.json"))
cat("wrote", file.path(data_dir, "r_interval_cox.json"), "\n")
