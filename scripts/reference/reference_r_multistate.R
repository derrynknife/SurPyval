# Reference results from R for the multi-state models
# (surpyval.multistate), for surpyval/tests/reference/test_multistate.py.
#
# Writes surpyval/tests/reference/data/r_multistate.json, with the
# software, its version, the call and the settings of each entry. Run from
# the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with mstate in the library path:
#
#     R_LIBS_USER=<library with mstate> \
#         Rscript scripts/reference/reference_r_multistate.R
#
# Written with R 4.6.2, survival 3.8.12 and mstate 0.3.3. Re-running
# reproduces the file byte for byte.
#
# The fixtures (fixtures.json):
#
# * "mgus2_multistate": survival::mgus2 as an illness-death model (MGUS ->
#   plasma-cell malignancy (pcm) -> death, MGUS -> death; months), as
#   load_mgus2(spells=True) gives it: as in the survival package's
#   vignette on multi-state models, a progression in the month of death is
#   put 0.1 months earlier. Checked here against R's own mgus2.
# * "multistate": 300 units simulated by make_fixtures.py from a Markov
#   illness-death model with recovery, with delayed entry and tied times.
#
# What each reference is:
#
# * survfit(Surv(tl, x, event) ~ 1, id = id, istate = state, timefix =
#   FALSE): the Aalen-Johansen state occupation probabilities and their
#   infinitesimal-jackknife std.err, from the start (p0 the states of
#   those at risk then), and from s = 2 with that p0 or with p0 given.
# * mstate: coxph(Surv(Tstart, Tstop, status) ~ strata(trans), method =
#   "breslow") on the stacked data, msfit(vartype = v) and
#   probtrans(predt = s, method = v) for v in "greenwood", "aalen": the
#   transition probabilities P(s, t) from each state with their standard
#   errors. (Breslow's ties make the increments d / r, the Aalen-Johansen
#   estimate; coxph's default, Efron's, changes them.)
# * mstate with covariates: coxph with the covariates expanded per
#   transition (expand.covs), or "male" shared by every transition, and
#   strata(trans), ties "breslow" or "efron"; msfit(newdata) for one
#   covariate profile and probtrans(predt = s) (method "aalen", the
#   default): P(s, t | z) and its standard errors, and the coefficients
#   with their standard errors.
# * mstate's general expanded design: a covariate's effect shared by some
#   transitions only, and transitions with proportional baselines (one
#   stratum, with an indicator), on the MGUS data and on the simulated
#   data with two covariates made from the unit number; the coefficients,
#   their standard errors, the log partial likelihood and P(s, t | z) with
#   its standard errors.
suppressPackageStartupMessages({
  library(survival)
  library(mstate)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_multistate.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
v_survival <- as.character(packageVersion("survival"))
v_mstate <- as.character(packageVersion("mstate"))

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

# The fixture is R's mgus2, as the vignette turns it into spells.
m <- mgus2
ptemp <- ifelse(m$pstat == 1 & m$ptime == m$futime, m$ptime - 0.1, m$ptime)
n_spells <- 1 + (m$pstat == 1)
stopifnot(nrow(mgus_long) == sum(n_spells))
first <- mgus_long[!duplicated(mgus_long$id), ]
stopifnot(all(first$id == m$id))
stopifnot(isTRUE(all.equal(first$x, as.numeric(ifelse(m$pstat == 1, ptemp, m$futime)))))
stopifnot(isTRUE(all.equal(first$z1, (as.numeric(m$age) - 70) / 10)))
stopifnot(all(first$male == as.numeric(m$sex == "M")))

mgus_tmat <- transMat(list(c(2, 3), c(3), c()), names = mgus_states)
sim_tmat <- transMat(list(c(2, 3), c(1, 3), c()), names = sim_states)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
stack <- function(long, tmat, covs = character()) {
  tr <- to.trans2(tmat)
  rows <- list()
  for (k in seq_len(nrow(long))) {
    from <- match(long$state[k], colnames(tmat))
    for (h in tr$transno[tr$from == from]) {
      to <- tr$to[tr$transno == h]
      row <- data.frame(id = long$id[k], from = from, to = to, trans = h,
                        Tstart = long$tl[k], Tstop = long$x[k],
                        status = as.numeric(!is.na(long$e[k]) &&
                                            long$e[k] == colnames(tmat)[to]))
      for (cv in covs) row[[cv]] <- long[[cv]][k]
      rows[[length(rows) + 1]] <- row
    }
  }
  st <- do.call(rbind, rows)
  attr(st, "trans") <- tmat
  class(st) <- c("msdata", "data.frame")
  st
}

at <- function(times, grid, values) {
  # A right-continuous step function of the grid, its first value before
  # the grid starts.
  idx <- findInterval(times, grid)
  values[pmax(idx, 1)]
}

nums <- function(v) unname(as.numeric(v))

survfit_values <- function(long, states, times, start = NULL, p0 = NULL) {
  long$ev <- factor(ifelse(is.na(long$e), "censor", long$e),
                    levels = c("censor", states))
  long$ist <- factor(long$state, levels = states)
  if (is.null(start)) {
    fit <- survfit(Surv(tl, x, ev) ~ 1, data = long, id = id,
                   istate = ist, timefix = FALSE)
  } else if (is.null(p0)) {
    fit <- survfit(Surv(tl, x, ev) ~ 1, data = long, id = id,
                   istate = ist, timefix = FALSE, start.time = start)
  } else {
    fit <- survfit(Surv(tl, x, ev) ~ 1, data = long, id = id,
                   istate = ist, timefix = FALSE, start.time = start,
                   p0 = p0)
  }
  s <- summary(fit, times = times, extend = TRUE)
  res <- list(states = fit$states, times = times)
  for (k in seq_along(fit$states)) {
    res[[paste0("p_", fit$states[k])]] <- nums(s$pstate[, k])
    res[[paste0("se_", fit$states[k])]] <- nums(s$std.err[, k])
  }
  res
}

probtrans_values <- function(pt, states, times) {
  res <- list(times = times)
  for (a in seq_along(states)) {
    tab <- pt[[a]]
    for (b in seq_along(states)) {
      res[[paste0("P_", states[a], "_", states[b])]] <-
        nums(at(times, tab$time, tab[[paste0("pstate", b)]]))
      res[[paste0("se_", states[a], "_", states[b])]] <-
        nums(at(times, tab$time, tab[[paste0("se", b)]]))
    }
  }
  res
}

references <- list()
add <- function(id, fixture, software, call, settings, values) {
  references[[id]] <<- list(
    fixture = fixture,
    software = paste("R", software),
    version = if (software == "survival") v_survival else v_mstate,
    call = call,
    settings = settings,
    values = values
  )
}

# ---------------------------------------------------------------------------
# survfit: occupation probabilities and the robust std.err
# ---------------------------------------------------------------------------
mgus_times <- c(12, 60, 120, 240, 360, 420)
sim_times <- c(0.5, 1, 1.5, 2, 3, 4, 5, 6.5, 7.5)
survfit_call <- paste("survfit(Surv(tl, x, ev) ~ 1, id = id,",
                      "istate = state, timefix = FALSE")

add("survfit_mgus2", "mgus2_multistate", "survival",
    paste0(survfit_call, ")"),
    list(start = "the earliest entry", p0 = "the states at risk then"),
    survfit_values(mgus_long, mgus_states, mgus_times))
add("survfit_sim", "multistate", "survival",
    paste0(survfit_call, ")"),
    list(start = "the earliest entry", p0 = "the states at risk then"),
    survfit_values(sim_long, sim_states, sim_times))
add("survfit_sim_s2", "multistate", "survival",
    paste0(survfit_call, ", start.time = 2)"),
    list(start = "2", p0 = "the states at risk then (healthy and ill)"),
    survfit_values(sim_long, sim_states, sim_times[sim_times >= 2],
                   start = 2))
add("survfit_sim_s2_ill", "multistate", "survival",
    paste0(survfit_call, ", start.time = 2, p0 = c(0, 1, 0))"),
    list(start = "2", p0 = "ill"),
    survfit_values(sim_long, sim_states, sim_times[sim_times >= 2],
                   start = 2, p0 = c(0, 1, 0)))

# ---------------------------------------------------------------------------
# mstate, no covariates: P(s, t) with the Greenwood and Aalen variances
# ---------------------------------------------------------------------------
mgus_st <- stack(mgus_long, mgus_tmat, c("z1", "male"))
sim_st <- stack(sim_long, sim_tmat)
cases <- list(
  list("mgus2", "mgus2_multistate", mgus_st, mgus_tmat, mgus_states,
       mgus_times, 0),
  list("mgus2_s60", "mgus2_multistate", mgus_st, mgus_tmat, mgus_states,
       mgus_times[mgus_times >= 60], 60),
  list("sim", "multistate", sim_st, sim_tmat, sim_states, sim_times, 0),
  list("sim_s2", "multistate", sim_st, sim_tmat, sim_states,
       sim_times[sim_times >= 2], 2)
)
for (v in c("greenwood", "aalen")) {
  for (case in cases) {
    c0 <- coxph(Surv(Tstart, Tstop, status) ~ strata(trans),
                data = case[[3]], method = "breslow")
    msf <- msfit(c0, trans = case[[4]], vartype = v)
    pt <- suppressWarnings(probtrans(msf, predt = case[[7]], method = v))
    add(paste0("mstate_", case[[1]], "_", v), case[[2]], "mstate",
        sprintf(paste0("probtrans(msfit(coxph(Surv(Tstart, Tstop, status)",
                       " ~ strata(trans), method = 'breslow'), vartype =",
                       " '%s'), predt = %g, method = '%s')"),
                v, case[[7]], v),
        list(data = "the stacked spells (one row per possible transition)",
             predt = case[[7]], from = "each state"),
        probtrans_values(pt, case[[5]], case[[6]]))
  }
}

# ---------------------------------------------------------------------------
# mstate with covariates
# ---------------------------------------------------------------------------
ex <- expand.covs(mgus_st, c("z1", "male"), longnames = FALSE)
profile <- c(z1 = 0.5, male = 1)
newdata <- data.frame(trans = 1:3, strata = 1:3)
for (cv in c("z1", "male")) {
  for (h in 1:3) {
    newdata[[paste0(cv, ".", h)]] <- ifelse(newdata$trans == h,
                                            profile[[cv]], 0)
  }
  newdata[[cv]] <- profile[[cv]]
}
fits <- list(
  specific = Surv(Tstart, Tstop, status) ~ z1.1 + z1.2 + z1.3 + male.1 +
    male.2 + male.3 + strata(trans),
  shared = Surv(Tstart, Tstop, status) ~ z1.1 + z1.2 + z1.3 + male +
    strata(trans)
)
for (name in names(fits)) {
  for (ties in c("breslow", "efron")) {
    cx <- coxph(fits[[name]], data = ex, method = ties)
    msf <- msfit(cx, newdata = newdata, trans = mgus_tmat)
    for (s in c(0, 60)) {
      # (probtrans warns of a negative diagonal of I + dA late on, where
      # few are at risk; SurPyval warns there too)
      pt <- suppressWarnings(probtrans(msf, predt = s))
      vals <- probtrans_values(pt, mgus_states, mgus_times[mgus_times >= s])
      vals$coef <- nums(coef(cx))
      vals$coef_names <- names(coef(cx))
      vals$se_coef <- nums(sqrt(diag(vcov(cx))))
      vals$profile <- as.list(profile)
      add(paste0("mstate_cox_mgus2_", name, "_", ties, "_s", s),
          "mgus2_multistate", "mstate",
          sprintf("probtrans(msfit(coxph(%s, method = '%s'), newdata), predt = %g)",
                  deparse(fits[[name]], width.cutoff = 500L), ties, s),
          list(effects = name, ties = ties, profile = "z1 = 0.5, male = 1",
               method = "aalen"),
          vals)
    }
  }
}

# ---------------------------------------------------------------------------
# mstate with effects shared by some transitions, and proportional
# baselines
# ---------------------------------------------------------------------------
# The general expanded design of mstate (Putter, Fiocco and Geskus, 2007,
# section 4.3): each column of the design applies a covariate (or, for a
# proportional baseline, the constant 1) to a set of transitions and is 0
# on the others; the strata are the transitions, with transitions whose
# baselines are proportional put in one stratum. Each design is a list of
# columns list(name, covariate (NA for the indicator), transitions).
design_data <- function(st, design) {
  for (col in design) {
    on <- st$trans %in% col$trans
    v <- if (is.na(col$cov)) 1 else st[[col$cov]]
    st[[col$name]] <- ifelse(on, v, 0)
  }
  st
}
design_newdata <- function(design, profile, stratum) {
  K <- length(stratum)
  nd <- data.frame(trans = 1:K, strata = stratum)
  for (col in design) {
    v <- if (is.na(col$cov)) 1 else profile[[col$cov]]
    nd[[col$name]] <- ifelse(nd$trans %in% col$trans, v, 0)
  }
  nd
}
design_case <- function(id, fixture, st, tmat, states, times, design,
                        stratum, profile, ties, starts, describe) {
  ex <- design_data(st, design)
  ex$stratum <- stratum[ex$trans]
  f <- as.formula(paste("Surv(Tstart, Tstop, status) ~",
                        paste(sapply(design, `[[`, "name"), collapse = " + "),
                        "+ strata(stratum)"))
  cx <- coxph(f, data = ex, method = ties)
  msf <- msfit(cx, newdata = design_newdata(design, profile, stratum),
               trans = tmat)
  for (s in starts) {
    pt <- suppressWarnings(probtrans(msf, predt = s))
    vals <- probtrans_values(pt, states, times[times >= s])
    vals$coef <- nums(coef(cx))
    vals$coef_names <- names(coef(cx))
    vals$se_coef <- nums(sqrt(diag(vcov(cx))))
    vals$loglik <- nums(cx$loglik[2])
    vals$profile <- as.list(profile)
    add(paste0(id, "_", ties, "_s", s), fixture, "mstate",
        sprintf("probtrans(msfit(coxph(%s, method = '%s'), newdata), predt = %g)",
                deparse(f, width.cutoff = 500L), ties, s),
        c(describe, list(ties = ties, method = "aalen",
                         strata = paste(stratum, collapse = ", "))),
        vals)
  }
}
col <- function(name, cov, trans) list(name = name, cov = cov, trans = trans)

# MGUS (transitions 1 mgus -> pcm, 2 mgus -> death, 3 pcm -> death).
# (a) The effect of age shared by the two transitions into death, sex per
# transition.
mgus_groups <- list(
  col("z1.1", "z1", 1), col("z1.23", "z1", c(2, 3)),
  col("male.1", "male", 1), col("male.2", "male", 2),
  col("male.3", "male", 3)
)
# (b) The baselines of the two transitions into death proportional (one
# stratum, pr the indicator of pcm -> death), age per transition, sex
# shared by all.
mgus_prop <- list(
  col("z1.1", "z1", 1), col("z1.2", "z1", 2), col("z1.3", "z1", 3),
  col("male", "male", 1:3), col("pr", NA, 3)
)
for (ties in c("breslow", "efron")) {
  design_case("mstate_cox_mgus2_groups", "mgus2_multistate", mgus_st,
              mgus_tmat, mgus_states, mgus_times, mgus_groups, 1:3,
              profile, ties, c(0, 60),
              list(effects = "z1 shared by mgus -> death and pcm -> death",
                   profile = "z1 = 0.5, male = 1"))
  design_case("mstate_cox_mgus2_proportional", "mgus2_multistate", mgus_st,
              mgus_tmat, mgus_states, mgus_times, mgus_prop, c(1, 2, 2),
              profile, ties, c(0, 60),
              list(effects = paste("baselines of mgus -> death and",
                                   "pcm -> death proportional; male shared"),
                   profile = "z1 = 0.5, male = 1"))
}

# The simulated model with recovery (transitions 1 healthy -> ill, 2
# healthy -> dead, 3 ill -> healthy, 4 ill -> dead), with two covariates
# made from the unit number: zb = 1 for every third unit, zc = (id mod 7)
# / 7. zb is shared by the transitions into death, zc by every
# transition, and the baselines into death are proportional.
sim_long$zb <- as.numeric(sim_long$id %% 3 == 0)
sim_long$zc <- (sim_long$id %% 7) / 7
sim_cov_st <- stack(sim_long, sim_tmat, c("zb", "zc"))
sim_design <- list(
  col("zb.1", "zb", 1), col("zb.24", "zb", c(2, 4)), col("zb.3", "zb", 3),
  col("zc", "zc", 1:4), col("pr", NA, 4)
)
design_case("mstate_cox_sim_groups", "multistate", sim_cov_st, sim_tmat,
            sim_states, sim_times, sim_design, c(1, 2, 3, 2),
            c(zb = 1, zc = 0.5), "efron", c(0, 2),
            list(effects = paste("zb shared by the transitions into dead,",
                                 "zc by all; baselines into dead",
                                 "proportional"),
                 covariates = "zb = id %% 3 == 0, zc = (id %% 7) / 7",
                 profile = "zb = 1, zc = 0.5"))

out <- list(
  generator = "scripts/reference/reference_r_multistate.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
