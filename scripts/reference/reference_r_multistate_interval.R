# Reference results from R for the parametric multi-state models with
# interval-censored moves (surpyval.multistate.ParametricMultiState with
# 'x' as [xl, xr]), for
# surpyval/tests/reference/test_multistate_interval.py.
#
# Writes surpyval/tests/reference/data/r_multistate_interval.json, with the
# software, its version, the call and the settings of each entry. Run from
# the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with msm and flexsurv in the library path:
#
#     R_LIBS_USER=<library with msm and flexsurv> \
#         Rscript scripts/reference/reference_r_multistate_interval.R
#
# Written with R 4.6.2, msm 1.9 and flexsurv 2.3.2. Re-running reproduces
# the file.
#
# The fixture (fixtures.json, "multistate_inspected") is 300 simulated
# units of a progressive illness-death model (1 healthy, 2 ill, 3 dead;
# the moves 1-2, 1-3 and 2-3) seen at visits, with deaths timed exactly.
#
# What each reference is:
#
# * "exponential": msm(state ~ time, subject = id, qmatrix = Q, obstype =
#   ifelse(exact == 1, 2, 1)): a death is an exactly timed move from the
#   state seen at the visit before it (msm's obstype 2), every other row a
#   visit. With no move back to healthy, a unit seen healthy and then ill
#   moved once, directly, between the two visits, so this is the
#   likelihood of SurPyval's long form with the onset of illness interval
#   censored. The log-likelihood and the intensities (qmatrix.msm, with
#   delta-method standard errors), with optim's reltol = 1e-14.
# * "two_state": flexsurvreg(Surv(l, r, type = "interval2") ~ 1, dist =
#   "weibull") on the time to the onset of illness alone (the onset
#   interval of each unit seen ill, right censored at the last visit seen
#   healthy otherwise, leaving out units seen healthy only at time 0,
#   which say nothing about it): a two-state model is the univariate fit. Its
#   parameters (shape, scale) and log-likelihood.
suppressPackageStartupMessages({
  library(msm)
  library(survival)
  library(flexsurv)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate_interval.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_msm <- as.character(packageVersion("msm"))
v_flexsurv <- as.character(packageVersion("flexsurv"))

d <- as.data.frame(fx$multistate_inspected$columns)
d$obstype <- ifelse(d$exact == 1, 2, 1)
Q <- rbind(c(0, 0.1, 0.1), c(0, 0, 0.1), c(0, 0, 0))
ctl <- list(reltol = 1e-14, maxit = 10000)

references <- list()
add <- function(id, software, version, call, settings, values) {
  references[[id]] <<- list(
    fixture = "multistate_inspected",
    software = software,
    version = version,
    call = call,
    settings = settings,
    values = values
  )
}

fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           obstype = obstype, control = ctl)
q <- qmatrix.msm(fit, ci = "delta")
pick <- function(m) c(m[1, 2], m[1, 3], m[2, 3])
add("exponential", "R msm", v_msm,
    paste("msm(state ~ time, subject = id, data = d, qmatrix = Q,",
          "obstype = ifelse(exact == 1, 2, 1), control = list(reltol =",
          "1e-14, maxit = 10000))"),
    list(transitions = "1-2, 1-3, 2-3"),
    list(loglik = as.numeric(logLik(fit)), q = pick(q$estimates),
         q_se = pick(q$SE)))

# The onset of illness: (last healthy visit, first ill visit] for a unit
# seen ill, right censored at its last healthy visit otherwise.
onset <- do.call(rbind, lapply(split(d, d$id), function(u) {
  u <- u[order(u$time), ]
  healthy <- u$time[u$state == 1]
  ill <- u$time[u$state == 2]
  if (length(ill) > 0) {
    data.frame(l = max(healthy[healthy < min(ill)]), r = min(ill))
  } else {
    data.frame(l = max(healthy), r = NA)
  }
}))
# (a unit seen healthy only at time 0 says nothing about the onset)
onset <- onset[!(onset$l == 0 & is.na(onset$r)), ]
fs <- flexsurvreg(Surv(l, r, type = "interval2") ~ 1, data = onset,
                  dist = "weibull",
                  control = list(reltol = 1e-14, maxit = 10000))
add("two_state", "R flexsurv", v_flexsurv,
    paste("flexsurvreg(Surv(l, r, type = 'interval2') ~ 1, dist =",
          "'weibull') on the onset of illness"),
    list(data = paste("each unit's (last healthy visit, first ill visit],",
                      "or right censored at its last healthy visit")),
    list(loglik = as.numeric(fs$loglik),
         shape = unname(fs$res["shape", "est"]),
         scale = unname(fs$res["scale", "est"]),
         n_interval = sum(!is.na(onset$r))))

out <- list(
  generator = "scripts/reference/reference_r_multistate_interval.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
