# Reference results from R for the semi-Markov ("clock reset") parametric
# multi-state models (surpyval.multistate.ParametricMultiState and
# ParametricMultiStateRegression with clock="reset"), and the expected
# length of stay of both clocks, for
# surpyval/tests/reference/test_multistate_semimarkov.py.
#
# Writes surpyval/tests/reference/data/r_multistate_semimarkov.json, with
# the software, its version, the call and the settings of each entry. Run
# from the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with flexsurv in the library path:
#
#     R_LIBS_USER=<library with flexsurv> \
#         Rscript scripts/reference/reference_r_multistate_semimarkov.R
#
# Written with R 4.6.2, survival 3.8.12, flexsurv 2.3.2 and deSolve 1.42.
# Re-running reproduces the file (the simulations are seeded).
#
# The fixtures are those of reference_r_multistate_parametric.R:
# "mgus2_multistate" (survival::mgus2 as illness-death spells, months) and
# "multistate" (300 simulated units of an illness-death model with
# recovery, delayed entry and tied times).
#
# What each reference is:
#
# * flexsurvreg(Surv(Tstop - Tstart, status) ~ 1, subset = trans == h,
#   dist = d) for each transition h of the stacked spells: the "clock
#   reset" model, each transition fitted on the time spent in its 'from'
#   state. Its parameters and maximised log-likelihood.
# * pmatrix.simfs(fmsm(...), trans, t, M = 1e6) after set.seed(1):
#   P(0, t) of the clock-reset Weibull models (each row from entering
#   that state at 0), estimated from 10^6 simulated paths, and
#   totlos.simfs(..., start = 1, M = 1e6), the expected time in each
#   state by t from the first state. Simulations: their Monte Carlo
#   standard errors are about sqrt(p (1 - p) / M), under 5e-4.
# * totlos.fs(fmsm(...), t, rtol = atol = 1e-12) on the clock-forward
#   (Markov) Weibull models: the expected time in each state by t, from
#   the forward equations extended by the integral of P (deSolve).
# * dist = "weibullPH" with the covariate male on the clock reset, and
#   pmatrix.simfs(newdata = data.frame(male = 1), M = 1e6).
suppressPackageStartupMessages({
  library(survival)
  library(flexsurv)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate_semimarkov.json")
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

M <- 1e6
mgus_st <- stack(mgus_long, mgus_tmat, c("male"))
sim_st <- stack(sim_long, sim_tmat)
mgus_st$years <- mgus_st$Tstop - mgus_st$Tstart
sim_st$years <- sim_st$Tstop - sim_st$Tstart

sim_matrices <- function(fm, tmat, times, newdata = NULL) {
  lapply(times, function(u) {
    set.seed(1)
    p <- pmatrix.simfs(fm, trans = tmat, t = u, M = M, newdata = newdata)
    unname(as.matrix(p))
  })
}

sim_los <- function(fm, tmat, u) {
  set.seed(1)
  nums(totlos.simfs(fm, trans = tmat, t = u, start = 1, M = M))
}

cases <- list(
  list("mgus2", "mgus2_multistate", mgus_st, mgus_tmat, mgus_states,
       c(60, 120, 240), 240),
  list("sim", "multistate", sim_st, sim_tmat, sim_states,
       c(1, 3, 6), 6)
)
call_fit <- paste("flexsurvreg(Surv(Tstop - Tstart, status) ~ 1, subset =",
                  "trans == h, dist = '%s')")
for (case in cases) {
  for (dist in c("weibull", "lnorm", "exp")) {
    fits <- fit_all(case[[3]], case[[4]], dist, Surv(years, status) ~ 1)
    vals <- fit_values(fits)
    call <- sprintf(call_fit, dist)
    settings <- list(clock = "reset (time in the state)")
    if (dist == "weibull") {
      fm <- do.call(fmsm, c(fits, list(trans = case[[4]])))
      times <- case[[6]]
      ps <- sim_matrices(fm, case[[4]], times)
      vals <- c(vals, list(times = times), matrices(ps, case[[5]]))
      vals$los_t <- case[[7]]
      vals$los <- sim_los(fm, case[[4]], case[[7]])
      call <- paste0(call, "; set.seed(1); pmatrix.simfs(fmsm(...), trans,",
                     " t, M = 1e6); set.seed(1); totlos.simfs(fmsm(...),",
                     " trans, t, start = 1, M = 1e6)")
      settings$M <- M
    }
    add(paste0("flexsurv_reset_", case[[1]], "_", dist), case[[2]], call,
        settings, vals)
  }
}

# ---------------------------------------------------------------------------
# The expected length of stay of the clock-forward (Markov) models
# ---------------------------------------------------------------------------
fwd <- list(
  list("mgus2", "mgus2_multistate", mgus_st, mgus_tmat, c(120, 240, 480)),
  list("sim", "multistate", sim_st, sim_tmat, c(2, 6, 12))
)
for (case in fwd) {
  fits <- fit_all(case[[3]], case[[4]], "weibull",
                  Surv(Tstart, Tstop, status) ~ 1)
  fm <- do.call(fmsm, c(fits, list(trans = case[[4]])))
  vals <- fit_values(fits)
  vals$times <- case[[5]]
  for (k in seq_along(case[[5]])) {
    tl <- totlos.fs(fm, trans = case[[4]], t = case[[5]][k], rtol = 1e-12,
                    atol = 1e-12)
    vals[[paste0("los_", k)]] <- lapply(seq_len(nrow(tl)),
                                        function(a) nums(tl[a, ]))
  }
  add(paste0("flexsurv_forward_los_", case[[1]]), case[[2]],
      paste("flexsurvreg(Surv(Tstart, Tstop, status) ~ 1, subset = trans ==",
            "h, dist = 'weibull'); totlos.fs(fmsm(...), trans, t,",
            "rtol = 1e-12, atol = 1e-12)"),
      list(clock = "forward (Tstart, Tstop]",
           ode = "rtol = atol = 1e-12",
           los = "row a: from state a at 0"),
      vals)
}

# ---------------------------------------------------------------------------
# A Weibull proportional-hazards regression of each transition, clock reset
# ---------------------------------------------------------------------------
fits <- fit_all(mgus_st, mgus_tmat, "weibullPH", Surv(years, status) ~ male)
fm <- do.call(fmsm, c(fits, list(trans = mgus_tmat)))
times <- c(120, 240)
ps <- sim_matrices(fm, mgus_tmat, times, newdata = data.frame(male = 1))
vals <- c(fit_values(fits), list(times = times, male = 1),
          matrices(ps, mgus_states))
add("flexsurv_reset_mgus2_weibullPH_male", "mgus2_multistate",
    paste("flexsurvreg(Surv(Tstop - Tstart, status) ~ male, subset =",
          "trans == h, dist = 'weibullPH'); set.seed(1);",
          "pmatrix.simfs(fmsm(...), trans, t, M = 1e6,",
          "newdata = data.frame(male = 1))"),
    list(clock = "reset (time in the state)", M = M),
    vals)

out <- list(
  generator = "scripts/reference/reference_r_multistate_semimarkov.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
