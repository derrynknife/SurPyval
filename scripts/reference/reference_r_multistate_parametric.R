# Reference results from R for the parametric multi-state models
# (surpyval.multistate.ParametricMultiState and
# ParametricMultiStateRegression), for
# surpyval/tests/reference/test_multistate_parametric.py.
#
# Writes surpyval/tests/reference/data/r_multistate_parametric.json, with
# the software, its version, the call and the settings of each entry. Run
# from the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with flexsurv in the library path:
#
#     R_LIBS_USER=<library with flexsurv> \
#         Rscript scripts/reference/reference_r_multistate_parametric.R
#
# Written with R 4.6.2, survival 3.8.12, flexsurv 2.3.2 and deSolve 1.42.
# Re-running reproduces the file byte for byte (the simulated intervals
# are seeded).
#
# The fixtures (fixtures.json) are those of reference_r_multistate.R:
# "mgus2_multistate" (survival::mgus2 as illness-death spells, months) and
# "multistate" (300 simulated units of an illness-death model with
# recovery, delayed entry and tied times).
#
# What each reference is:
#
# * flexsurvreg(Surv(Tstart, Tstop, status) ~ 1, subset = trans == h,
#   dist = d) for each transition h of the stacked spells: the "clock
#   forward" model, each spell left truncated at its start. Its
#   parameters (flexsurv's: shape and scale of "weibull", meanlog and
#   sdlog of "lnorm", rate of "exp") and maximised log-likelihood.
# * pmatrix.fs(fmsm(...), t = times) with deSolve's tolerances set to
#   rtol = atol = 1e-12: P(0, t), the forward equations solved by
#   deSolve's lsoda (flexsurv has no start time, so P(s, t) is
#   P(0, s)^-1 P(0, t), by the Chapman-Kolmogorov equations).
# * With ci = TRUE, B = 1000, after set.seed(1): flexsurv's intervals,
#   the 2.5% and 97.5% quantiles of P(0, t) over parameters drawn from
#   their asymptotic normal distribution (a simulation, not the delta
#   method).
# * dist = "weibullPH" with the covariate male, and pmatrix.fs(newdata =
#   data.frame(male = 1)): P(0, t | male = 1), and each transition's
#   coefficient of male (a log hazard ratio).
suppressPackageStartupMessages({
  library(survival)
  library(flexsurv)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate_parametric.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_flexsurv <- as.character(packageVersion("flexsurv"))
v_desolve <- as.character(packageVersion("deSolve"))

spells <- function(name, states) {
  d <- as.data.frame(lapply(fx[[name]]$columns, function(v) {
    if (is.list(v)) sapply(v, function(u) if (is.null(u)) NA else u) else v
  }))
  d$state <- states[d$state]
  d$e <- ifelse(is.na(d$e), NA, states[d$e])
  d
}

mgus_states <- c("mgus", "pcm", "death")
sim_states <- c("healthy", "ill", "dead")
mgus_long <- spells("mgus2_multistate", mgus_states)
sim_long <- spells("multistate", sim_states)

# The transitions, numbered as SurPyval orders them (by 'from', then 'to',
# in the order of the states given): mgus -> pcm, mgus -> death, pcm ->
# death; healthy -> ill, healthy -> dead, ill -> healthy, ill -> dead.
mgus_tmat <- rbind(c(NA, 1, 2), c(NA, NA, 3), c(NA, NA, NA))
sim_tmat <- rbind(c(NA, 1, 2), c(3, NA, 4), c(NA, NA, NA))
dimnames(mgus_tmat) <- list(from = mgus_states, to = mgus_states)
dimnames(sim_tmat) <- list(from = sim_states, to = sim_states)

stack <- function(long, tmat, covs = character()) {
  rows <- list()
  states <- colnames(tmat)
  for (k in seq_len(nrow(long))) {
    from <- match(long$state[k], states)
    for (to in which(!is.na(tmat[from, ]))) {
      row <- data.frame(id = long$id[k], from = from, to = to,
                        trans = tmat[from, to],
                        Tstart = long$tl[k], Tstop = long$x[k],
                        status = as.numeric(!is.na(long$e[k]) &&
                                            long$e[k] == states[to]))
      for (cv in covs) row[[cv]] <- long[[cv]][k]
      rows[[length(rows) + 1]] <- row
    }
  }
  do.call(rbind, rows)
}

nums <- function(v) unname(as.numeric(v))

fit_all <- function(st, tmat, dist, formula) {
  lapply(seq_len(max(tmat, na.rm = TRUE)), function(h) {
    flexsurvreg(formula, data = st[st$trans == h, ], dist = dist)
  })
}

pmat <- function(fits, tmat, t, newdata = NULL) {
  fm <- do.call(fmsm, c(fits, list(trans = tmat)))
  # one (K, K) matrix per time
  lapply(t, function(u) {
    p <- pmatrix.fs(fm, t = u, newdata = newdata, rtol = 1e-12,
                    atol = 1e-12)
    unname(as.matrix(p))
  })
}

matrices <- function(ps, states, prefix = "P") {
  res <- list()
  for (a in seq_along(states)) {
    for (b in seq_along(states)) {
      res[[paste0(prefix, "_", states[a], "_", states[b])]] <-
        sapply(ps, function(p) p[a, b])
    }
  }
  res
}

fit_values <- function(fits) {
  list(
    params = lapply(fits, function(f) nums(f$res[, "est"])),
    param_names = lapply(fits, function(f) rownames(f$res)),
    se = lapply(fits, function(f) nums(sqrt(diag(vcov(f))))),
    loglik = sapply(fits, function(f) f$loglik)
  )
}

references <- list()
add <- function(id, fixture, call, settings, values) {
  references[[id]] <<- list(
    fixture = fixture,
    software = "R flexsurv",
    version = v_flexsurv,
    deSolve = v_desolve,
    call = call,
    settings = settings,
    values = values
  )
}

mgus_times <- c(12, 60, 120, 240, 360, 420, 600)
sim_times <- c(0.5, 1, 1.5, 2, 3, 4, 5, 6.5, 7.5, 12)
mgus_st <- stack(mgus_long, mgus_tmat, c("male"))
sim_st <- stack(sim_long, sim_tmat)
call0 <- paste("flexsurvreg(Surv(Tstart, Tstop, status) ~ 1, subset =",
               "trans == h, dist = '%s'); pmatrix.fs(fmsm(...), t)")

cases <- list(
  list("mgus2", "mgus2_multistate", mgus_st, mgus_tmat, mgus_states,
       mgus_times, 60),
  list("sim", "multistate", sim_st, sim_tmat, sim_states, sim_times, 2)
)
for (case in cases) {
  for (dist in c("weibull", "lnorm", "exp")) {
    fits <- fit_all(case[[3]], case[[4]], dist,
                    Surv(Tstart, Tstop, status) ~ 1)
    times <- case[[6]]
    ps <- pmat(fits, case[[4]], times)
    vals <- c(list(times = times), fit_values(fits),
              matrices(ps, case[[5]]))
    # P(s, t) = P(0, s)^-1 P(0, t)
    s <- case[[7]]
    later <- times[times >= s]
    ps0 <- pmat(fits, case[[4]], s)[[1]]
    pst <- lapply(pmat(fits, case[[4]], later), function(p) solve(ps0, p))
    vals$s <- s
    vals$times_s <- later
    vals <- c(vals, matrices(pst, case[[5]], "Ps"))
    add(paste0("flexsurv_", case[[1]], "_", dist), case[[2]],
        sprintf(call0, dist),
        list(clock = "forward (Tstart, Tstop]", ode = "rtol = atol = 1e-12",
             from = "each state at 0, and at s by P(0, s)^-1 P(0, t)"),
        vals)
  }
}

# ---------------------------------------------------------------------------
# flexsurv's simulated intervals (Weibull, mgus2)
# ---------------------------------------------------------------------------
fits <- fit_all(mgus_st, mgus_tmat, "weibull", Surv(Tstart, Tstop, status) ~ 1)
fm <- do.call(fmsm, c(fits, list(trans = mgus_tmat)))
ci_times <- c(120, 240)
lower <- list()
upper <- list()
for (k in seq_along(ci_times)) {
  set.seed(1)
  p <- pmatrix.fs(fm, t = ci_times[k], ci = TRUE, B = 1000)
  lower[[k]] <- unname(attr(p, "lower"))
  upper[[k]] <- unname(attr(p, "upper"))
}
vals <- c(list(times = ci_times), matrices(lower, mgus_states, "L"),
          matrices(upper, mgus_states, "U"))
add("flexsurv_mgus2_weibull_ci", "mgus2_multistate",
    "set.seed(1); pmatrix.fs(fmsm(...), t, ci = TRUE, B = 1000)",
    list(dist = "weibull", cl = 0.95,
         method = "simulation from the parameters' normal distribution"),
    vals)

# ---------------------------------------------------------------------------
# A Weibull proportional-hazards regression of each transition
# ---------------------------------------------------------------------------
fits <- fit_all(mgus_st, mgus_tmat, "weibullPH",
                Surv(Tstart, Tstop, status) ~ male)
ps <- pmat(fits, mgus_tmat, mgus_times, newdata = data.frame(male = 1))
vals <- c(list(times = mgus_times, male = 1), fit_values(fits),
          matrices(ps, mgus_states))
add("flexsurv_mgus2_weibullPH_male", "mgus2_multistate",
    paste("flexsurvreg(Surv(Tstart, Tstop, status) ~ male, subset =",
          "trans == h, dist = 'weibullPH'); pmatrix.fs(fmsm(...), t,",
          "newdata = data.frame(male = 1))"),
    list(clock = "forward (Tstart, Tstop]", ode = "rtol = atol = 1e-12"),
    vals)

out <- list(
  generator = "scripts/reference/reference_r_multistate_parametric.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
