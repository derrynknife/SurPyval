# Reference results from R for the calibration check of diagnose() (#881).
#
# Writes surpyval/tests/reference/data/r_calibration.json, with the
# software, its version, the call and the values of each entry, as
# reference_r.R does for the other files. Every fit is to a fixture already
# in fixtures.json (lung, ph_violation and left_truncation), so make_fixtures.py needs no
# change.
#
#     Rscript scripts/reference/reference_r_calibration.R
#
# Needs survival, rms, polspline and jsonlite. Re-running reproduces the
# file byte for byte.
#
# The method is Austin, Harrell and van Klaveren (2020): at t0, each unit's
# predicted risk p = 1 - S(t0 | Z) from the model, and a Cox model of the
# observed times on rcs(log(-log(1 - p)), knots); the observed risk is its
# 1 - S(t0), and ICI, E50 and E90 are the mean, median and 90th percentile
# of |observed - p|. The model and its predictions are R's (coxph and
# survfit for the Cox model, survreg and psurvreg for the parametric
# ones), so the comparison covers the whole computation. The calibration
# model is fitted with coxph (Efron's ties) and its curve and pointwise
# band come from survfit (conf.type "log", the default), as SurPyval's
# CoxPH.fit and its delta-method band. The knots are the 10th, 50th and
# 90th percentiles of the complementary log-log (R's default quantile, the
# 5th smallest and largest values below 100 units), which is what
# Hmisc::rcspline.eval chooses for three knots on these data; both are
# recorded. The times t0 are the failure times by which a quarter, a half
# and three quarters of the failures had happened.
#
# Two further estimates of the same curve are recorded for comparison, not
# for equality: Austin et al.'s own code (rms::cph with rcs(cll, 3) and
# survest), and their other estimator, hazard regression
# (polspline::hare and phare).

suppressPackageStartupMessages({
    library(survival)
    library(rms)
    library(polspline)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

lung_d <- fixture("lung")
lung_d$status <- 1 - lung_d$c
lung_cc <- lung_d[!is.na(lung_d$ph_ecog), ]

refs <- list()
env <- environment()
finite_or_na <- function(v) {
    if (is.list(v)) return(lapply(v, finite_or_na))
    if (is.numeric(v)) {
        v <- unname(v)
        v[!is.finite(v)] <- NA
    }
    v
}
record <- function(id, fixture, call, settings, extract) {
    result <- eval(parse(text = call), envir = env)
    refs[[id]] <<- list(
        fixture = fixture,
        software = "R survival, rms, polspline",
        version = paste(
            as.character(packageVersion("survival")),
            as.character(packageVersion("rms")),
            as.character(packageVersion("polspline")),
            sep = ", "
        ),
        call = call,
        settings = settings,
        values = finite_or_na(extract(result))
    )
    invisible(result)
}

# The failure times by which a share q of the failures had happened.
quartiles <- function(time, status) {
    t <- sort(time[status == 1])
    sapply(c(0.25, 0.5, 0.75), function(q) t[ceiling(q * length(t))])
}

knots_of <- function(cll) {
    # rcspline.eval's rule for five or fewer distinct values: the interior
    # ones
    u <- sort(unique(cll))
    if (length(u) <= 5) return(u[-c(1, length(u))])
    k <- unname(quantile(cll, c(0.1, 0.5, 0.9)))
    if (length(cll) < 100) {
        s <- sort(cll)
        k[1] <- s[5]
        k[3] <- s[length(s) - 4]
    }
    k
}

# The calibration at each t0 of the predicted survival surv(t0) (a vector
# per unit), with the outcome `y` (a Surv object, of the data frame `d`).
calibrate <- function(d, y, surv, t0s, grid = c(0.2, 0.4, 0.6)) {
    out <- list(t0 = t0s)
    for (k in seq_along(t0s)) {
        t0 <- t0s[k]
        p <- 1 - surv(t0)
        p <- pmin(pmax(p, 1e-10), 1 - 1e-10)
        cll <- log(-log(1 - p))
        kn <- unique(knots_of(cll))
        cd <- data.frame(cll = cll)
        # with fewer than three distinct knots the predictions enter
        # linearly (the calibration-slope model), as in SurPyval
        linear <- length(kn) < 3
        cal <- if (linear) coxph(y ~ cll, data = cd) else
            coxph(y ~ rcs(cll, kn), data = cd)
        sf <- summary(survfit(cal, newdata = cd), times = t0, extend = TRUE)
        obs <- 1 - as.numeric(sf$surv)
        diff <- abs(obs - p)
        gd <- data.frame(cll = log(-log(1 - grid)))
        sg <- summary(survfit(cal, newdata = gd), times = t0, extend = TRUE)
        # Austin et al.'s own code: cph and survest
        dd <- datadist(cd)
        options(datadist = dd)
        cph_fit <- if (linear)
            cph(y ~ cll, data = cd, x = TRUE, y = TRUE, surv = TRUE) else
            cph(y ~ rcs(cll, 3), data = cd, x = TRUE, y = TRUE, surv = TRUE)
        obs_rms <- 1 - survest(cph_fit, newdata = cd, times = t0)$surv
        options(datadist = NULL)
        # hazard regression
        h <- hare(data = y[, ncol(y) - 1], delta = y[, ncol(y)],
                  cov = as.matrix(cll))
        obs_hare <- phare(t0, cll, h)
        diff_rms <- abs(obs_rms - p)
        diff_hare <- abs(obs_hare - p)
        tag <- function(name) paste0(name, "_", k)
        out[[tag("knots")]] <- if (linear) median(cll) else kn
        out[[tag("linear")]] <- linear
        if (!linear) {
            out[[tag("knots_rcspline_eval")]] <- unname(
                attr(rcspline.eval(cll, nk = 3), "knots")
            )
        }
        out[[tag("coef")]] <- unname(coef(cal))
        out[[tag("ici")]] <- mean(diff)
        out[[tag("e50")]] <- median(diff)
        out[[tag("e90")]] <- unname(quantile(diff, 0.9))
        out[[tag("predicted_range")]] <- range(p)
        out[[tag("grid")]] <- grid
        out[[tag("grid_observed")]] <- 1 - as.numeric(sg$surv)
        out[[tag("grid_lower")]] <- 1 - as.numeric(sg$upper)
        out[[tag("grid_upper")]] <- 1 - as.numeric(sg$lower)
        out[[tag("ici_rms")]] <- mean(diff_rms)
        out[[tag("e50_rms")]] <- median(diff_rms)
        out[[tag("e90_rms")]] <- unname(quantile(diff_rms, 0.9))
        out[[tag("ici_hare")]] <- mean(diff_hare)
        out[[tag("e50_hare")]] <- median(diff_hare)
        out[[tag("e90_hare")]] <- unname(quantile(diff_hare, 0.9))
    }
    out
}

cox_surv <- function(fit, d) function(t0) {
    as.numeric(summary(survfit(fit, newdata = d), times = t0,
                       extend = TRUE)$surv)
}
survreg_surv <- function(fit) function(t0) {
    1 - psurvreg(t0, mean = fit$linear.predictors, scale = fit$scale,
                 distribution = fit$dist)
}
tight <- survreg.control(rel.tolerance = 1e-12, iter.max = 200)
settings <- function(model, extra = list()) c(
    list(
        model = model,
        t0 = "the failure times by which 25%, 50% and 75% of the failures had happened",
        calibration = "coxph(y ~ rcs(log(-log(1 - p)), knots)); survfit at t0, conf.type 'log'",
        knots = "quantile(cll, c(0.1, 0.5, 0.9)); below 100 units the 5th smallest and largest",
        rms = "cph(y ~ rcs(cll, 3)) and survest (Austin et al.'s code)",
        hare = "hare(data = time, delta = status, cov = cll) and phare"
    ),
    extra
)
lung_missing <- list(missing = "the one row with ph.ecog missing is dropped")

lung_y <- with(lung_cc, Surv(time, status))
lung_t0 <- quartiles(lung_cc$time, lung_cc$status)

lung_cox <- coxph(Surv(time, status) ~ age + sex + ph_ecog, data = lung_cc)
record("lung_cox", "lung",
       "calibrate(lung_cc, lung_y, cox_surv(lung_cox, lung_cc), lung_t0)",
       settings("coxph(Surv(time, status) ~ age + sex + ph_ecog)",
                lung_missing),
       identity)

lung_weibull <- survreg(Surv(time, status) ~ age + sex + ph_ecog,
                        data = lung_cc, dist = "weibull", control = tight)
record("lung_weibull", "lung",
       "calibrate(lung_cc, lung_y, survreg_surv(lung_weibull), lung_t0)",
       settings("survreg(Surv(time, status) ~ age + sex + ph_ecog, dist = 'weibull')",
                lung_missing),
       identity)

lung_lognormal <- survreg(Surv(time, status) ~ age + sex + ph_ecog,
                          data = lung_cc, dist = "lognormal", control = tight)
record("lung_lognormal", "lung",
       "calibrate(lung_cc, lung_y, survreg_surv(lung_lognormal), lung_t0)",
       settings("survreg(Surv(time, status) ~ age + sex + ph_ecog, dist = 'lognormal')",
                lung_missing),
       identity)

# hare takes no delayed entry: its values are left out
without_hare <- function(out) {
    out[grepl("hare", names(out))] <- NULL
    out
}
entry <- list(entry = "delayed entry: Surv(tl, x, status) in both models; no hare values (it takes no entry times)")

pv_d <- fixture("ph_violation")
pv_d$status <- 1 - pv_d$c
pv_y <- with(pv_d, Surv(tl, x, status))
pv_t0 <- quartiles(pv_d$x, pv_d$status)
pv_cox <- coxph(Surv(tl, x, status) ~ z1 + z2 + z3, data = pv_d)
record("ph_violation_cox", "ph_violation",
       "calibrate(pv_d, pv_y, cox_surv(pv_cox, pv_d), pv_t0)",
       settings("coxph(Surv(tl, x, status) ~ z1 + z2 + z3)", entry),
       without_hare)

lt_d <- fixture("left_truncation")
lt_d$status <- 1 - lt_d$c
lt_y <- with(lt_d, Surv(tl, x, status))
lt_t0 <- quartiles(lt_d$x, lt_d$status)
lt_cox <- coxph(Surv(tl, x, status) ~ z, data = lt_d)
record("left_truncation_cox", "left_truncation",
       "calibrate(lt_d, lt_y, cox_surv(lt_cox, lt_d), lt_t0)",
       settings("coxph(Surv(tl, x, status) ~ z)",
                c(entry, list(linear = "z is binary: two distinct predictions, so the predictions enter linearly"))),
       without_hare)

out <- list(
    generator = "scripts/reference/reference_r_calibration.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_calibration.json"))
cat("wrote", file.path(data_dir, "r_calibration.json"), "\n")
