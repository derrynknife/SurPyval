# Reference results from R msm for the panel-data multi-state models
# (surpyval.multistate.PanelMultiState and PanelMultiStateRegression), for
# surpyval/tests/reference/test_multistate_panel.py.
#
# Writes surpyval/tests/reference/data/r_multistate_panel.json, with the
# software, its version, the call and the settings of each entry. Run from
# the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with msm in the library path:
#
#     R_LIBS_USER=<library with msm> \
#         Rscript scripts/reference/reference_r_multistate_panel.R
#
# Written with R 4.6.2 and msm 1.9. Re-running reproduces the file.
#
# The fixture (fixtures.json, "multistate_panel") is 300 simulated units
# of a four-state Markov model seen at visits, with the transitions of
# msm's heart-transplant example (1-2, 1-4, 2-1, 2-3, 2-4, 3-2, 3-4),
# exactly timed deaths (obstype 3) and last visits that saw only "alive"
# (state 99, one of 1, 2, 3).
#
# What each reference is (every fit with optim's reltol = 1e-14, so that
# its maximum is reached to the digits compared):
#
# * "panel": msm(state ~ time, subject = id, qmatrix = Q, obstype =
#   obstype, deathexact = 4, censor = 99, censor.states = c(1, 2, 3)):
#   the log-likelihood (logLik), qmatrix.msm (the intensities, their
#   delta-method standard errors and 95% bounds), pmatrix.msm at t = 1
#   and 5, and sojourn.msm (the mean stays, standard errors and bounds).
# * "panel_snapshot": the same data with the rows of state 99 left out
#   and every row a snapshot (obstype 1, no exact deaths).
# * "covariates": as "panel" with covariates = ~ male + w (w changes at
#   each visit): the log-likelihood, qmatrix.msm(covariates = 0) and
#   hazard.msm (the intensity ratios of each covariate, with bounds).
# * "shared": covariates = ~ male, constraint = list(male = rep(1, 7)):
#   one effect of male on every transition.
# * "pci": as "panel" with pci = c(3, 6): the log-likelihood,
#   qmatrix.msm(covariates = 0) (the first period's intensities) and
#   hazard.msm (each later period's intensity ratios).
suppressPackageStartupMessages({
  library(msm)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate_panel.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_msm <- as.character(packageVersion("msm"))

d <- as.data.frame(fx$multistate_panel$columns)
Q <- rbind(
  c(0, 0.25, 0, 0.25),
  c(0.166, 0, 0.166, 0.166),
  c(0, 0.25, 0, 0.25),
  c(0, 0, 0, 0)
)
ctl <- list(reltol = 1e-14, maxit = 10000)

mat <- function(m) unname(lapply(seq_len(nrow(m)), function(i) m[i, ]))
trans_vec <- function(m) {
  # the allowed transitions' entries, row by row (msm's order)
  out <- c()
  for (i in 1:4) for (j in 1:4) if (Q[i, j] > 0) out <- c(out, m[i, j])
  out
}
qvals <- function(fit, cov = NULL) {
  q <- if (is.null(cov)) qmatrix.msm(fit, ci = "delta")
       else qmatrix.msm(fit, covariates = cov, ci = "delta")
  list(q = trans_vec(q$estimates), q_se = trans_vec(q$SE),
       q_lower = trans_vec(q$L), q_upper = trans_vec(q$U))
}
hvals <- function(fit) {
  h <- hazard.msm(fit)
  out <- list()
  for (nm in names(h)) {
    out[[paste0("hr_", nm)]] <- unname(h[[nm]][, "HR"])
    out[[paste0("hr_lower_", nm)]] <- unname(h[[nm]][, "L"])
    out[[paste0("hr_upper_", nm)]] <- unname(h[[nm]][, "U"])
  }
  out
}

references <- list()
add <- function(id, call, settings, values) {
  references[[id]] <<- list(
    fixture = "multistate_panel",
    software = "R msm",
    version = v_msm,
    call = call,
    settings = settings,
    values = values
  )
}
base_call <- paste("msm(state ~ time, subject = id, data = d, qmatrix = Q,",
                   "obstype = obstype, deathexact = 4, censor = 99,",
                   "censor.states = c(1, 2, 3)%s, control = list(reltol =",
                   "1e-14, maxit = 10000))")
settings <- list(transitions = "1-2, 1-4, 2-1, 2-3, 2-4, 3-2, 3-4 (msm order)",
                 bounds = "95%, delta method, log scale")

fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           obstype = obstype, deathexact = 4, censor = 99,
           censor.states = c(1, 2, 3), control = ctl)
soj <- sojourn.msm(fit)
vals <- c(list(loglik = as.numeric(logLik(fit))), qvals(fit),
          list(times = c(1, 5),
               P1 = mat(pmatrix.msm(fit, t = 1)),
               P5 = mat(pmatrix.msm(fit, t = 5)),
               sojourn = unname(soj$estimates), sojourn_se = unname(soj$SE),
               sojourn_lower = unname(soj$L), sojourn_upper = unname(soj$U)))
add("panel", sprintf(base_call, ""), settings, vals)

snap <- d[d$state != 99, ]
fit <- msm(state ~ time, subject = id, data = snap, qmatrix = Q,
           control = ctl)
vals <- c(list(loglik = as.numeric(logLik(fit))), qvals(fit),
          list(P5 = mat(pmatrix.msm(fit, t = 5))))
add("panel_snapshot",
    paste("msm(state ~ time, subject = id, data = d[d$state != 99, ],",
          "qmatrix = Q, control = list(reltol = 1e-14, maxit = 10000))"),
    c(settings, list(rows = "state 99 left out; every row a snapshot")),
    vals)

fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           obstype = obstype, deathexact = 4, censor = 99,
           censor.states = c(1, 2, 3), covariates = ~ male + w,
           control = ctl)
vals <- c(list(loglik = as.numeric(logLik(fit))),
          qvals(fit, cov = list(male = 0, w = 0)), hvals(fit),
          list(P5_male = mat(pmatrix.msm(fit, t = 5,
                                         covariates = list(male = 1,
                                                           w = 0.5)))))
add("covariates", sprintf(base_call, ", covariates = ~ male + w"),
    c(settings, list(baseline = "qmatrix.msm(covariates = 0)",
                     P5_male = "pmatrix.msm(t = 5, covariates = list(male = 1, w = 0.5))")),
    vals)

fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           obstype = obstype, deathexact = 4, censor = 99,
           censor.states = c(1, 2, 3), covariates = ~ male,
           constraint = list(male = rep(1, 7)), control = ctl)
vals <- c(list(loglik = as.numeric(logLik(fit))),
          qvals(fit, cov = list(male = 0)), hvals(fit))
add("shared", sprintf(base_call, paste(", covariates = ~ male, constraint",
                                       "= list(male = rep(1, 7))")),
    settings, vals)

fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           obstype = obstype, deathexact = 4, censor = 99,
           censor.states = c(1, 2, 3), pci = c(3, 6), control = ctl)
vals <- c(list(loglik = as.numeric(logLik(fit))),
          qvals(fit, cov = 0), hvals(fit))
add("pci", sprintf(base_call, ", pci = c(3, 6)"),
    c(settings, list(baseline = "qmatrix.msm(covariates = 0), the first period")),
    vals)

out <- list(
  generator = "scripts/reference/reference_r_multistate_panel.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
