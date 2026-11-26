# Reference results from R for the parametric panel multi-state
# regression (surpyval.multistate.ParametricPanelMultiStateRegression),
# for surpyval/tests/reference/test_multistate_panel_regression.py.
#
# Writes surpyval/tests/reference/data/r_multistate_panel_regression.json,
# with the software, its version, the call and the settings of each
# entry. Run from the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with msm and nhm in the library path:
#
#     R_LIBS_USER=<library with msm and nhm> \
#         Rscript scripts/reference/reference_r_multistate_panel_regression.R
#
# Written with R 4.6.2, msm 1.9 and nhm 0.1.2. Re-running reproduces the
# file. The nhm fit solves the forward equations at a tight tolerance and
# takes tens of minutes.
#
# The fixture (fixtures.json, "multistate_inspected_covariates") is 300
# simulated units of a progressive illness-death model (1 healthy, 2 ill,
# 3 dead) with Weibull proportional hazards on the time since the start,
# seen at visits, with deaths timed exactly; "male" is fixed for a unit,
# "w" is drawn at each visit and holds until the next. A death is an
# exactly timed entry from a state not seen (msm's deathexact, nhm's
# death = TRUE).
#
# What each reference is:
#
# * "exponential": msm(state ~ time, subject = id, qmatrix = Q,
#   deathexact = 3, covariates = ~ male + w): a time-homogeneous Markov
#   model with q_rs(z) = q_rs exp(b_rs' z), the covariates of each visit
#   holding over the gap that starts at it. The log-likelihood, the
#   baseline intensities (qmatrix.msm at the covariates 0) and the
#   coefficients (msm's estimates, with standard errors from its
#   Hessian), at optim's reltol = 1e-14.
# * "weibull": nhm(model.nhm(state ~ time, subject = id, type =
#   "weibull", trans, nonh, covariates = "male", covm, death = TRUE,
#   death.states = 3)): each intensity q(t) = shape / scale (t /
#   scale)^(shape - 1) exp(b male), nhm's parameters the log of 1 /
#   scale, the log of the shape and b. The scales, shapes and
#   coefficients, the minimised negative log-likelihood (nhm's "value")
#   and the standard errors of nhm's parameters (from its Hessian), with
#   lsoda's tolerances at 1e-11, the ODEs solved from each start time
#   (safe = TRUE) and maxLik's search to reltol 1e-15, gradtol 1e-10 (at
#   its default tolerances the search stopped 1.2e-6 above the minimum,
#   the rare 1->3 scale 1.2e-4 short).
suppressPackageStartupMessages({
  library(msm)
  library(nhm)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate_panel_regression.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_msm <- as.character(packageVersion("msm"))
v_nhm <- as.character(packageVersion("nhm"))

d <- as.data.frame(fx$multistate_inspected_covariates$columns)

references <- list()
add <- function(id, software, version, call, settings, values) {
  references[[id]] <<- list(
    fixture = "multistate_inspected_covariates",
    software = software,
    version = version,
    call = call,
    settings = settings,
    values = values
  )
}

Q <- rbind(c(0, 0.1, 0.1), c(0, 0, 0.1), c(0, 0, 0))
fit <- msm(state ~ time, subject = id, data = d, qmatrix = Q,
           deathexact = 3, covariates = ~ male + w,
           control = list(reltol = 1e-14, maxit = 10000))
q <- qmatrix.msm(fit, covariates = 0, ci = "delta")
pick <- function(m) c(m[1, 2], m[1, 3], m[2, 3])
est <- fit$estimates
se <- sqrt(diag(fit$covmat))
nq <- 3
add("exponential", "R msm", v_msm,
    paste("msm(state ~ time, subject = id, data = d, qmatrix = Q,",
          "deathexact = 3, covariates = ~ male + w, control =",
          "list(reltol = 1e-14, maxit = 10000))"),
    list(transitions = "1-2, 1-3, 2-3",
         coefficients = "male then w, each by transition"),
    list(loglik = as.numeric(logLik(fit)),
         q = pick(q$estimates),
         q_se = pick(q$SE),
         b_male = unname(est[(nq + 1):(2 * nq)]),
         b_w = unname(est[(2 * nq + 1):(3 * nq)]),
         b_male_se = unname(se[(nq + 1):(2 * nq)]),
         b_w_se = unname(se[(2 * nq + 1):(3 * nq)])))

trans <- rbind(c(0, 1, 2), c(0, 0, 3), c(0, 0, 0))
m <- model.nhm(state ~ time, data = d, subject = id, type = "weibull",
               trans = trans, nonh = trans, covariates = "male",
               covm = list(male = trans), death = TRUE,
               death.states = 3)
nf <- nhm(m, initial = c(log(1 / 6), log(1 / 25), log(1 / 8), 0, 0, 0,
                         0, 0, 0),
          control = nhm.control(rtol = 1e-11, atol = 1e-11, safe = TRUE,
                                print.level = 0,
                                maxLikcontrol = list(reltol = 1e-15,
                                                     gradtol = 1e-10,
                                                     iterlim = 1000)))
add("weibull", "R nhm", v_nhm,
    paste("nhm(model.nhm(state ~ time, data = d, subject = id, type =",
          "'weibull', trans = trans, nonh = trans, covariates = 'male',",
          "covm = list(male = trans), death = TRUE, death.states = 3),",
          "control = nhm.control(rtol = 1e-11, atol = 1e-11, safe =",
          "TRUE, maxLikcontrol = list(reltol = 1e-15, gradtol = 1e-10,",
          "iterlim = 1000)))"),
    list(transitions = "1-2, 1-3, 2-3",
         parameters = paste("log(1 / scale), then log(shape), then the",
                            "male coefficient, by transition")),
    list(value = nf$value,
         names = nf$parnames,
         scale = exp(-nf$par[1:3]),
         shape = exp(nf$par[4:6]),
         b_male = nf$par[7:9],
         par = nf$par,
         par_se = sqrt(diag(solve(nf$hess)))))

out <- list(
  generator = "scripts/reference/reference_r_multistate_panel_regression.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
