# Reference results from R for the IDI and NRI of a new risk model over an
# old one at a horizon (surpyval.metrics.idi_nri), for
# surpyval/tests/reference/test_reclassification.py.
#
# Writes surpyval/tests/reference/data/r_reclassification.json, with the
# software, its version, the call and the settings of each entry. Run from
# the repository root, after python scripts/reference/make_fixtures.py,
# with survIDINRI, nricens and jsonlite in the library path:
#
#     R_LIBS_USER=<library> Rscript scripts/reference/reference_r_reclassification.R
#
# Written with R 4.6.2, survival 3.8.12, survIDINRI 1.1-2, survC1 1.0-3,
# nricens 1.6 and jsonlite. Re-running reproduces the file byte for byte
# (every resampling is seeded).
#
# The data: the "pbc" fixture (the 312 Mayo PBC trial patients at
# baseline, years of follow-up, death the event, transplant and alive
# censored). The old model is a Cox model of age and sex, the new one adds
# log bilirubin; the horizon is 5 years (no time is exactly 5). The
# predicted risks of death by 5 years are survIDINRI's own (its
# Est.HR.FUN, the risk 1 - exp(-Lambda0(t0) exp(b'z)) of each fit), stored
# so that the tests score exactly R's risks.
#
# What each entry is:
#
# * "idi_inf": survIDINRI::IDI.INF(indata, covs0, covs1, t0, npert =
#   1000, seed1 = 1): M1 (the IDI) and M2 (P(up | event) - P(up |
#   non-event)) with its event and non-event parts, and its perturbation
#   bounds. Its censoring weights are the Fleming-Harrington estimate
#   (survfit type = "fl"), not the Kaplan-Meier, its cases are x < t0, and
#   its means are trapezoid integrals of the empirical survival of the risk
#   differences on a grid of about 2000 points on [-1, 1]; its bounds also
#   perturb the two Cox fits. (Its PTB.IDI.FUN increments its counter
#   before storing each draw, so the first of its npert + 1 stored draws
#   is a 0 that was never drawn; the bounds are read from those.)
# * "idi_fun_km": survIDINRI's own estimator, IDI.FUN, given the
#   Kaplan-Meier censoring weights 1 / G(x_i-) (and the same grid): the
#   estimator idi_nri computes, up to the grid.
# * "idi_fixed_risk_perturbation": the perturbation bounds with the risks
#   held fixed: IDI.FUN on each of 4000 Exp(1) perturbations (set.seed(2)),
#   the censoring weights moved by survC1::kmcens's linear expansion, as
#   PTB.IDI.FUN does, without refitting the Cox models. Quantiles 0.025
#   and 0.975 (type 6, the (n + 1) p rule).
# * "nricens_<updown>_<method>": nricens::nricens(time, event, p.std,
#   p.new, t0, updown, cut, point.method, niter = 1000) after
#   set.seed(3): the NRI, its event and non-event parts and the four
#   shares, with the bootstrap bounds (quantile type 2, 2.5% and 97.5%).
#   updown = "diff" with cut = 0 is the continuous NRI; "category" with
#   cut = c(0.2, 0.4) the categorical NRI.

suppressPackageStartupMessages({
    library(survival)
    library(survIDINRI)
    library(nricens)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_reclassification.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
pbc_d <- as.data.frame(fx$pbc$columns)
time <- pbc_d$years
status <- as.integer(pbc_d$cause == 2)
t0 <- 5
stopifnot(!any(time == t0))
indata <- cbind(time, status)
covs0 <- as.matrix(pbc_d[, c("age", "female")])
covs1 <- as.matrix(pbc_d[, c("age", "female", "log_bili")])

versions <- function(pkgs) {
    paste(sapply(pkgs, function(p) paste(p, packageVersion(p))),
          collapse = ", ")
}
references <- list()
add <- function(id, software, call, settings, values) {
    references[[id]] <<- list(
        fixture = "pbc",
        software = software,
        version = paste0(R.version.string, "; ", versions(software)),
        call = call,
        settings = settings,
        values = values
    )
}

fit0 <- survIDINRI:::Est.HR.FUN(cbind(indata, covs0), t0)
fit1 <- survIDINRI:::Est.HR.FUN(cbind(indata, covs1), t0)
p_old <- c(fit0$risk)
p_new <- c(fit1$risk)

# -- survIDINRI::IDI.INF as it stands ----------------------------------------
res <- IDI.INF(indata, covs0, covs1, t0, npert = 1000, seed1 = 1)
add("idi_inf", c("survIDINRI", "survC1", "survival"),
    "IDI.INF(cbind(years, death), cbind(age, female), cbind(age, female, log_bili), t0 = 5, npert = 1000, seed1 = 1)",
    list(t0 = t0, npert = 1000, seed = 1, alpha = 0.05,
         censoring = "Fleming-Harrington (survfit type = 'fl')",
         cases = "x < t0 and an event", controls = "x >= t0"),
    list(p_old = p_old, p_new = p_new,
         m1 = unname(res$m1), m2 = unname(res$m2),
         m1_est = unname(res$m1.est), m2_est = unname(res$m2.est)))

# -- IDI.FUN with the Kaplan-Meier censoring weights -------------------------
# G(x_i-): the reverse Kaplan-Meier just before x_i, with an event taken
# to come before a censoring at its time (a tie is then not yet a
# censoring at x_i, so the left limit is the same either way).
km_c <- survfit(Surv(time, 1 - status) ~ 1)
g_left <- stepfun(km_c$time, c(1, km_c$surv), right = TRUE)(time)
w_km <- ifelse(time <= t0, status / g_left, 0)
w_km[time > t0] <- 1
pt <- survIDINRI:::IDI.FUN(indata, t0, p_new, p_old, Wi = w_km)
add("idi_fun_km", c("survIDINRI", "survival"),
    "IDI.FUN(cbind(years, death), t0 = 5, pnew, pold, Wi = 1 / G(x-) for events by t0, 1 after)",
    list(t0 = t0, censoring = "Kaplan-Meier, G(x-)",
         cases = "x < t0 and an event", controls = "x >= t0",
         grid = "sort(unique(c(-1, 0, 1, seq(-1, 1, length = 2000))))"),
    list(idi = c(pt$IDI), idi_d1 = c(pt$IDI_D1), idi_d0 = c(pt$IDI_D0),
         txi = c(pt$TXI), txi_d1 = c(pt$TXI_D1), txi_d0 = c(pt$TXI_D0)))

# -- perturbation bounds with the risks held fixed ---------------------------
kmc <- survC1:::kmcens(time, status)
distinct <- kmc$distinct
n <- length(time)
B <- 4000
set.seed(2)
draws <- matrix(NA, B, 2)
for (b in 1:B) {
    rn <- rexp(n)
    ghat <- c(kmc$surv - (t(kmc$psii) %*% (rn - 1) / n))
    ghat_t0 <- ghat[max(which(distinct <= t0))]
    ghat_x <- ghat[match(time, distinct)]
    pit <- rep(0, n)
    pit[status == 1] <- ghat_x[status == 1]
    pit[time >= t0] <- ghat_t0
    vit <- as.numeric(time <= t0) * status + as.numeric(time > t0)
    wgt <- rep(0, n)
    wgt[vit == 1] <- 1 / pit[vit == 1]
    s <- survIDINRI:::IDI.FUN(indata, t0, p_new, p_old, Wi = wgt,
                              PTB.Vi = rn)
    draws[b, ] <- c(s$IDI, s$TXI)
}
q <- apply(draws, 2, quantile, probs = c(0.025, 0.975), type = 6)
add("idi_fixed_risk_perturbation", c("survIDINRI", "survC1", "survival"),
    "set.seed(2); 4000 x IDI.FUN(..., Wi = perturbed weights, PTB.Vi = rexp(n)) with the risks fixed",
    list(t0 = t0, resamples = B, seed = 2, quantile_type = 6,
         censoring = "kmcens linear expansion, as PTB.IDI.FUN"),
    list(idi_bounds = unname(q[, 1]), m2_bounds = unname(q[, 2]),
         idi_sd = sd(draws[, 1]), m2_sd = sd(draws[, 2])))

# -- nricens ------------------------------------------------------------------
for (updown in c("diff", "category")) {
    cut <- if (updown == "diff") 0 else c(0.2, 0.4)
    for (method in c("ipw", "km")) {
        set.seed(3)
        out <- suppressMessages(capture.output(r <- nricens(
            time = time, event = status, p.std = p_old, p.new = p_new,
            t0 = t0, updown = updown, cut = cut, point.method = method,
            niter = 1000, msg = FALSE)))
        tab <- r$nri
        add(paste("nricens", updown, method, sep = "_"),
            c("nricens", "survival"),
            sprintf("set.seed(3); nricens(time, event, p.std = p_old, p.new = p_new, t0 = 5, updown = '%s', cut = %s, point.method = '%s', niter = 1000)",
                    updown, deparse(cut), method),
            list(t0 = t0, updown = updown, cut = cut, method = method,
                 niter = 1000, seed = 3, quantile_type = 2),
            list(rows = rownames(tab), estimate = tab$Estimate,
                 lower = tab$Lower, upper = tab$Upper))
    }
}

out <- list(
    generator = "scripts/reference/reference_r_reclassification.R",
    R = R.version.string,
    references = references
)
writeLines(
    toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
    out_file
)
