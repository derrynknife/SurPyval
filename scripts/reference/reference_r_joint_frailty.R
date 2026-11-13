# Reference results from R for the joint frailty model of recurrent events
# and a terminal event (surpyval.recurrent.JointFrailty), for
# surpyval/tests/reference/test_joint_frailty.py.
#
# Writes surpyval/tests/reference/data/r_frailtypack.json, with the
# software, its version, the call and the settings of each entry. Run from
# the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with frailtypack in the library path:
#
#     R_LIBS_USER=<library with frailtypack> \
#         Rscript scripts/reference/reference_r_joint_frailty.R
#
# Written with R 4.6.2 and frailtypack 3.8.2. Re-running reproduces the
# file byte for byte.
#
# The fixture "joint_frailty" (fixtures.json) is 250 units of a joint
# gamma-frailty model: recurrent events (c 0) and one closing row per
# unit (c 1, terminal 1 where the terminal event closed it), covariates
# z1 (0 or 1) and z2.
#
# What each reference is: frailtyPenal(..., joint = TRUE through
# terminal(), hazard = "Weibull") on the calendar time scale (counting
# process rows, recurrentAG = TRUE) and the gap time scale (gaps,
# recurrentAG = FALSE), with the gamma frailty (RandDist = "Gamma", the
# default; Gauss-Laguerre quadrature with nb.gl = 32 nodes, its default)
# and the log-normal one (RandDist = "LogN"; Gauss-Hermite quadrature,
# nb.gh = 32): the coefficients of each part, theta (sigma2 for the
# log-normal), alpha, the Weibull shapes and scales and the reported
# log-likelihood. frailtypack's Weibull baseline is SurPyval's,
# R0(t) = (t / scale) ^ shape (checked: SurPyval's likelihood at
# frailtypack's estimates is close to its reported log-likelihood, and far
# from it in the other parameterisation).
#
# R's readmission data (frailtypack's, so not copied into the repository)
# were compared locally (not part of the test suite) with the covariates sex, dukes and chemo in both
# parts (charlson changes within a patient):
#
# * calendar, gamma: frailtypack logLik -4227.026, theta 1.0419, alpha
#   1.0322; SurPyval -4225.725, theta 1.2755, alpha 0.9554, every
#   coefficient within 0.05 of frailtypack's; SurPyval's log-likelihood at
#   frailtypack's estimate is -4226.758.
# * calendar, log-normal: frailtypack -4218.881, sigma2 1.3125, alpha
#   0.9992; SurPyval -4218.869, 1.3121, 0.9988.
#
# The gap time scale with the gamma frailty is left out: frailtypack 3.8.2
# returned no fit for it on this fixture (istop 4). On R's readmission
# data it gives logLik -4159.127, theta 0.741, alpha 1.531; SurPyval
# -4159.730, theta 0.703, alpha 1.577, and -4159.777 at frailtypack's
# estimate. readmission is not stored here, as it is frailtypack's data.
suppressPackageStartupMessages({
  library(survival)
  library(frailtypack)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_frailtypack.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_frailtypack <- as.character(packageVersion("frailtypack"))

d <- as.data.frame(fx[["joint_frailty"]]$columns)
d <- d[order(d$id, d$x, d$c), ]
# Counting-process rows: each row runs from the unit's previous row to its
# own age; an event row ends in a recurrent event, the closing row in
# censoring of the recurrent process (and the terminal flag).
d$start <- ave(d$x, d$id, FUN = function(v) c(0, head(v, -1)))
d$gap <- d$x - d$start
d$event <- 1 - d$c

references <- list()
nums <- function(v) unname(as.numeric(v))
add <- function(id, call, settings, fit) {
  sigma <- if (is.null(fit$sigma2)) fit$theta else fit$sigma2
  references[[id]] <<- list(
    fixture = "joint_frailty",
    software = "R frailtypack",
    version = v_frailtypack,
    call = call,
    settings = settings,
    values = list(
      recurrent_coef = nums(fit$coef[1:2]),
      terminal_coef = nums(fit$coef[3:4]),
      theta = nums(sigma),
      alpha = nums(fit$alpha),
      recurrent_shape = nums(fit$shape.weib[1]),
      recurrent_scale = nums(fit$scale.weib[1]),
      terminal_shape = nums(fit$shape.weib[2]),
      terminal_scale = nums(fit$scale.weib[2]),
      loglik = nums(fit$logLik),
      istop = nums(fit$istop)
    )
  )
}

calendar <- Surv(start, x, event) ~ cluster(id) + z1 + z2 + terminal(terminal)
gap <- Surv(gap, event) ~ cluster(id) + z1 + z2 + terminal(terminal)
cases <- list(
  list("calendar_gamma", calendar, TRUE, "Gamma"),
  list("calendar_lognormal", calendar, TRUE, "LogN"),
  list("gap_lognormal", gap, FALSE, "LogN")
)
for (case in cases) {
  fit <- frailtyPenal(case[[2]], formula.terminalEvent = ~ z1 + z2,
                      data = d, recurrentAG = case[[3]], hazard = "Weibull",
                      RandDist = case[[4]], print.times = FALSE)
  add(paste0("frailtypack_", case[[1]]),
      paste(deparse(case[[2]]), collapse = " "),
      list(recurrentAG = case[[3]], hazard = "Weibull",
           RandDist = case[[4]], formula.terminalEvent = "~ z1 + z2",
           quadrature = if (case[[4]] == "Gamma") "nb.gl = 32" else
             "nb.gh = 32"),
      fit)
}

out <- list(
  generator = "scripts/reference/reference_r_joint_frailty.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
