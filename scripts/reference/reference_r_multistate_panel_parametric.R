# Reference results from R for the parametric panel multi-state model
# (surpyval.multistate.ParametricPanelMultiState), for
# surpyval/tests/reference/test_multistate_panel_parametric.py.
#
# Writes surpyval/tests/reference/data/r_multistate_panel_parametric.json,
# with the software, its version, the call and the settings of each
# entry. Run from the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with msm and nhm in the library path:
#
#     R_LIBS_USER=<library with msm and nhm> \
#         Rscript scripts/reference/reference_r_multistate_panel_parametric.R
#
# Written with R 4.6.2, msm 1.9 and nhm 0.1.2. Re-running reproduces the
# file. The nhm fit solves the forward equations at a tight tolerance and
# takes tens of minutes.
#
# The fixture (fixtures.json, "multistate_inspected") is 300 simulated
# units of a progressive illness-death model (1 healthy, 2 ill, 3 dead;
# Weibull hazards on the time since the start) seen at visits, with deaths
# timed exactly. Here a death is an exactly timed entry from a state not
# seen (msm's deathexact, nhm's death = TRUE): a unit seen healthy and then
# dead may have been ill in between.
#
# What each reference is:
#
# * "exponential": msm(state ~ time, subject = id, qmatrix = Q,
#   deathexact = 3): a time-homogeneous Markov model. The log-likelihood
#   and the intensities (qmatrix.msm, with delta-method standard errors),
#   with optim's reltol = 1e-14.
# * "weibull": nhm(model.nhm(state ~ time, subject = id, type = "weibull",
#   trans, nonh, death = TRUE, death.states = 3)): each intensity
#   q(t) = shape / scale (t / scale)^(shape - 1), nhm's parameters the log
#   of 1 / scale and the log of the shape. The scales and shapes, the
#   minimised negative log-likelihood (nhm's "value") and the standard
#   errors of nhm's parameters (from its Hessian), with lsoda's tolerances
#   at 1e-11 and the ODEs solved from each start time (safe = TRUE).
suppressPackageStartupMessages({
  library(msm)
  library(nhm)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate_panel_parametric.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_msm <- as.character(packageVersion("msm"))
v_nhm <- as.character(packageVersion("nhm"))

d <- as.data.frame(fx$multistate_inspected$columns)

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

Q <- rbind(c(0, 0.1, 0.1), c(0, 0, 0.1), c(0, 0, 0))
fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           deathexact = 3, control = list(reltol = 1e-14, maxit = 10000))
q <- qmatrix.msm(fit, ci = "delta")
pick <- function(m) c(m[1, 2], m[1, 3], m[2, 3])
add("exponential", "R msm", v_msm,
    paste("msm(state ~ time, subject = id, data = d, qmatrix = Q,",
          "deathexact = 3, control = list(reltol = 1e-14, maxit = 10000))"),
    list(transitions = "1-2, 1-3, 2-3"),
    list(loglik = as.numeric(logLik(fit)), q = pick(q$estimates),
         q_se = pick(q$SE)))

trans <- rbind(c(0, 1, 2), c(0, 0, 3), c(0, 0, 0))
m <- model.nhm(state ~ time, data = d, subject = id, type = "weibull",
               trans = trans, nonh = trans, death = TRUE,
               death.states = 3)
nf <- nhm(m, initial = c(log(1 / 6), log(1 / 25), log(1 / 8), 0, 0, 0),
          control = nhm.control(rtol = 1e-11, atol = 1e-11, safe = TRUE,
                                print.level = 0))
add("weibull", "R nhm", v_nhm,
    paste("nhm(model.nhm(state ~ time, data = d, subject = id, type =",
          "'weibull', trans = trans, nonh = trans, death = TRUE,",
          "death.states = 3), control = nhm.control(rtol = 1e-11, atol =",
          "1e-11, safe = TRUE))"),
    list(transitions = "1-2, 1-3, 2-3",
         parameters = "log(1 / scale) then log(shape), by transition"),
    list(value = nf$value,
         scale = exp(-nf$par[1:3]),
         shape = exp(nf$par[4:6]),
         par = nf$par,
         par_se = sqrt(diag(solve(nf$hess)))))

out <- list(
  generator = "scripts/reference/reference_r_multistate_panel_parametric.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
