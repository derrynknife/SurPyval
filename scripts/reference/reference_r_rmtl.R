# Reference results from R for the restricted mean time lost (RMTL) to a
# cause of failure: CompetingRisks.rmtl and surpyval.rmtl_diff.
#
# Writes surpyval/tests/reference/data/r_rmtl.json, with the software, its
# version, the exact call and the settings of each entry, as the other
# generators do. Every fit is to a fixture already in fixtures.json (the
# `competing` fixture, with its two groups, and pbc), so make_fixtures.py
# needs no change. Generated with R 4.6.2, survival 3.8.12, cmprsk 2.2.12,
# survRM2 1.0.4 and GFDrmtl 0.1.0.
#
#     Rscript scripts/reference/reference_r_rmtl.R
#
# Needs survival, cmprsk, survRM2, GFDrmtl and jsonlite. Re-running reproduces the
# file byte for byte.
#
# What each value is:
#
# * rmtl_survfit: the area under each cause's state probability (pstate of
#   survfit(Surv(x, factor(cause)) ~ 1), the Aalen-Johansen estimate),
#   integrated exactly as the step function it is, from 0 to tau.
# * rmtl_cuminc: the same area under cmprsk::cuminc's est, a second
#   implementation of the estimate.
# * rmean_survival / se_rmean_survival: survival's own restricted mean time
#   in each cause's state, summary(fit, rmean = tau)$table. Its standard
#   error is the infinitesimal-jackknife one (each unit's influence),
#   another estimator of the same variance than SurPyval's delta method;
#   it is stored to compare loosely, not to match.
# * se_delta: the delta-method standard error SurPyval documents
#   (aalen_johansen_rmtl), written out here in R from the numbers at risk
#   and of events, as an independent implementation of the formula.
# * rmtl_gfdrmtl / se_gfdrmtl: GFDrmtl's (Munko, Dobler and Ditzhaus)
#   estimate of the time lost and its asymptotic variance, from its
#   internal GFDrmtl:::RMTL(time, status, tau, M), the function its
#   RMTL.test uses for each group (the variance it returns is already
#   that of the estimate, not scaled by n).
# * one_cause: every failure as one cause, where the time lost is tau less
#   the restricted mean survival time and its standard error is the RMST's:
#   survRM2's (Greenwood-based) on the same data, from its one-arm
#   function rmst1 (which reports the time lost too; rmst2 needs two
#   arms).

suppressPackageStartupMessages({
    library(survival)
    library(cmprsk)
    library(survRM2)
    library(GFDrmtl)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

versions <- paste0(
    "survival ", as.character(packageVersion("survival")),
    ", cmprsk ", as.character(packageVersion("cmprsk")),
    ", survRM2 ", as.character(packageVersion("survRM2")),
    ", GFDrmtl ", as.character(packageVersion("GFDrmtl"))
)

# Exact area under a right-continuous step function (value v[i] from
# time t[i] on, 0 before t[1]) from 0 to tau.
step_area <- function(t, v, tau) {
    keep <- t <= tau
    t <- t[keep]
    v <- v[keep]
    if (!length(t)) return(0)
    sum(v * (c(t[-1], tau) - t))
}

# SurPyval's delta-method variance of the time lost to cause k, written
# from its formula.
delta_se <- function(x, cause, k, tau) {
    tt <- sort(unique(x[x <= tau]))
    r <- sapply(tt, function(u) sum(x >= u))
    d <- sapply(tt, function(u) sum(x == u & cause != 0))
    dk <- sapply(tt, function(u) sum(x == u & cause == k))
    p <- d / r
    pk <- dk / r
    po <- p - pk
    S <- cumprod(1 - p)
    Sp <- c(1, head(S, -1))
    F <- cumsum(Sp * pk)
    up <- c(tt[-1], tau)
    A <- rev(cumsum(rev(F * (up - tt))))
    L <- tau - tt
    ratio <- ifelse(S > 0, Sp / S, 0)
    Go <- -ratio * (A - F * L)
    Gk <- Sp * L + Go
    v <- (Gk^2 * pk * (1 - pk) + Go^2 * po * (1 - po) -
          2 * Gk * Go * pk * po) / r
    sqrt(sum(v))
}

refs <- list()
record <- function(id, fixture_name, data_call, x, cause, taus) {
    fit <- survfit(Surv(x, factor(cause, 0:2)) ~ 1)
    ci <- cuminc(x, cause, cencode = 0)
    vals <- list(taus = taus)
    for (k in 1:2) {
        col <- which(fit$states == as.character(k))
        cik <- ci[[paste("1", k)]]
        vals[[paste0("rmtl_survfit_", k)]] <- sapply(
            taus, function(tau) step_area(fit$time, fit$pstate[, col], tau))
        vals[[paste0("rmtl_cuminc_", k)]] <- sapply(
            taus, function(tau) step_area(cik$time, cik$est, tau))
        tab <- lapply(taus, function(tau)
            summary(fit, rmean = tau)$table[as.character(k), ])
        vals[[paste0("rmean_survival_", k)]] <- sapply(
            tab, function(row) row[["rmean"]])
        vals[[paste0("se_rmean_survival_", k)]] <- sapply(
            tab, function(row) row[["se(rmean)"]])
        vals[[paste0("se_delta_", k)]] <- sapply(
            taus, function(tau) delta_se(x, cause, k, tau))
    }
    gfd <- lapply(taus, function(tau) GFDrmtl:::RMTL(x, cause, tau, 2))
    for (k in 1:2) {
        vals[[paste0("rmtl_gfdrmtl_", k)]] <- sapply(
            gfd, function(g) g$rmtl[k])
        vals[[paste0("se_gfdrmtl_", k)]] <- sapply(
            gfd, function(g) sqrt(g$var_rmtl[k, k]))
    }
    one <- lapply(taus, function(tau)
        survRM2:::rmst1(x, as.integer(cause != 0), tau = tau)$rmtl)
    vals$one_cause_rmtl <- sapply(one, function(m) m[["Est."]])
    vals$one_cause_se <- sapply(one, function(m) m[["se"]])
    refs[[id]] <<- list(
        fixture = fixture_name,
        software = "R survival, cmprsk, survRM2, GFDrmtl",
        version = versions,
        call = paste0(
            "fit <- survfit(Surv(x, factor(cause, 0:2)) ~ 1) with ",
            data_call, "; step_area(fit$time, fit$pstate[, k], tau); ",
            "cuminc(x, cause, cencode = 0); summary(fit, rmean = tau); ",
            "survRM2:::rmst1(x, cause != 0, tau = tau)$rmtl; ",
            "GFDrmtl:::RMTL(x, cause, tau, 2)"
        ),
        settings = list(
            causes = c(1, 2),
            censored = "cause 0",
            rmtl = "area under the step cumulative incidence from 0 to tau",
            se_delta = "SurPyval's delta-method formula, written in R",
            se_rmean_survival = "survival's infinitesimal-jackknife se",
            se_gfdrmtl = "sqrt of GFDrmtl's var_rmtl diagonal",
            one_cause = "every failure one cause: survRM2 rmst1's rmtl and se"
        ),
        values = vals
    )
}

cr_d <- fixture("competing")
record("competing_all", "competing", "x, cause = competing$x, $cause",
       cr_d$x, cr_d$cause, c(5, 10, 20))
for (g in c(0, 1)) {
    sub <- cr_d[cr_d$group == g, ]
    record(paste0("competing_group", g), "competing",
           sprintf("the rows of competing with group == %d", g),
           sub$x, sub$cause, c(5, 10, 15))
}
pbc_d <- fixture("pbc")
record("pbc", "pbc", "x, cause = pbc$years, pbc$cause", pbc_d$years,
       pbc_d$cause, c(2, 5, 10))

out <- list(
    generator = "scripts/reference/reference_r_rmtl.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_rmtl.json"))
cat("wrote", file.path(data_dir, "r_rmtl.json"), "\n")
