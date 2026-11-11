# Reference values for the multi-state models (surpyval.multistate),
# written to surpyval/tests/multistate/data/r_multistate.json for
# surpyval/tests/multistate/test_reference_multistate.py.
#
#     R_LIBS_USER=<library with mstate> \
#         Rscript scripts/reference/reference_r_multistate.R
#
# Written with R 4.6.2, survival 3.8.12, mstate 0.3.3 and jsonlite (the
# versions are stored in the file). Re-running reproduces the file byte
# for byte.
#
# The data are stored in the file with the results, so the tests need
# neither R nor the data's packages:
#
# * "mgus2": survival::mgus2 (Kyle et al., 2002) as an illness-death
#   model: MGUS -> plasma-cell malignancy (pcm) -> death, and MGUS ->
#   death; months. As in the survival package's vignette on multi-state
#   models, a progression in the month of death is moved 0.1 months
#   earlier, so that the spell in pcm has a length. Covariates: z1 =
#   (age - 70) / 10 and male (1 for a man).
# * "sim": 300 units simulated here from a Markov illness-death model
#   with recovery (healthy -> ill 0.3, healthy -> dead 0.1, ill ->
#   healthy 0.4, ill -> dead 0.3, constant hazards), each first seen at
#   a uniform time in (0, 1) (delayed entry: units dead by then are
#   never seen) and censored at a uniform time in (2, 8); times rounded
#   to 0.01, which gives tied times.
#
# What each reference is:
#
# * survfit(Surv(tl, x, event) ~ 1, id = id, istate = state, timefix =
#   FALSE): the Aalen-Johansen state occupation probabilities and their
#   infinitesimal-jackknife std.err, from the start (p0 the states of
#   those at risk then) and from s with p0 given.
# * mstate: coxph(Surv(tl, x, status) ~ strata(trans), method =
#   "breslow") on the stacked data, msfit(vartype = v) and
#   probtrans(predt = s, method = v) for v in "greenwood", "aalen": the
#   transition probabilities P(s, t) from each state, with their
#   standard errors. (Breslow's ties make the increments d / r, the
#   Aalen-Johansen estimate; coxph's default, Efron's, changes them.)
# * mstate with covariates: coxph with the covariates expanded per
#   transition (expand.covs), or "male" shared by every transition, and
#   strata(trans), ties "breslow" or "efron"; msfit(newdata) for one
#   covariate profile and probtrans(predt = s) (method "aalen", the
#   default): P(s, t | z) and its standard errors, and the coefficients
#   with their standard errors.
suppressPackageStartupMessages({
  library(survival)
  library(mstate)
  library(jsonlite)
})

out <- list(
  generator = "scripts/reference/reference_r_multistate.R",
  software = list(
    R = paste0(R.version$major, ".", R.version$minor),
    survival = as.character(packageVersion("survival")),
    mstate = as.character(packageVersion("mstate"))
  ),
  data = list(),
  references = list()
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
m <- mgus2
ptemp <- ifelse(m$pstat == 1 & m$ptime == m$futime, m$ptime - 0.1, m$ptime)
rows <- list()
for (i in seq_len(nrow(m))) {
  z1 <- (m$age[i] - 70) / 10
  male <- as.numeric(m$sex[i] == "M")
  if (m$pstat[i] == 1) {
    rows[[length(rows) + 1]] <- data.frame(
      id = m$id[i], tl = 0, x = ptemp[i], state = "mgus", e = "pcm",
      z1 = z1, male = male)
    rows[[length(rows) + 1]] <- data.frame(
      id = m$id[i], tl = ptemp[i], x = m$futime[i], state = "pcm",
      e = ifelse(m$death[i] == 1, "death", NA), z1 = z1, male = male)
  } else {
    rows[[length(rows) + 1]] <- data.frame(
      id = m$id[i], tl = 0, x = m$futime[i], state = "mgus",
      e = ifelse(m$death[i] == 1, "death", NA), z1 = z1, male = male)
  }
}
mgus_long <- do.call(rbind, rows)
# Spells of length zero (follow-up of 0 months) cannot be at risk.
mgus_long <- mgus_long[mgus_long$x > mgus_long$tl, ]
mgus_states <- c("mgus", "pcm", "death")
mgus_tmat <- transMat(list(c(2, 3), c(3), c()), names = mgus_states)

set.seed(20261112)
rates <- matrix(0, 3, 3)
rates[1, 2] <- 0.3; rates[1, 3] <- 0.1; rates[2, 1] <- 0.4; rates[2, 3] <- 0.3
sim_states <- c("healthy", "ill", "dead")
rows <- list()
for (i in 1:300) {
  entry <- round(runif(1, 0, 1), 2)
  cens <- round(runif(1, 2, 8), 2)
  t <- 0; s <- 1; spells <- list()
  repeat {
    total <- sum(rates[s, ])
    nt <- t + rexp(1, total)
    if (nt >= cens) {
      spells[[length(spells) + 1]] <- c(t, cens, s, NA)
      break
    }
    to <- sample(1:3, 1, prob = rates[s, ])
    spells[[length(spells) + 1]] <- c(t, nt, s, to)
    t <- nt; s <- to
    if (s == 3) break
  }
  sp <- do.call(rbind, spells)
  sp <- sp[sp[, 2] > entry, , drop = FALSE]
  if (nrow(sp) == 0) next
  sp[1, 1] <- max(sp[1, 1], entry)
  sp[, 1:2] <- round(sp[, 1:2], 2)
  for (k in seq_len(nrow(sp))) {
    if (k > 1) sp[k, 1] <- sp[k - 1, 2]
    if (sp[k, 2] <= sp[k, 1]) sp[k, 2] <- sp[k, 1] + 0.01
  }
  rows[[length(rows) + 1]] <- data.frame(
    id = i, tl = sp[, 1], x = sp[, 2], state = sim_states[sp[, 3]],
    e = ifelse(is.na(sp[, 4]), NA, sim_states[sp[, 4]]))
}
sim_long <- do.call(rbind, rows)
sim_tmat <- transMat(list(c(2, 3), c(1, 3), c()), names = sim_states)

store_data <- function(d) {
  lst <- as.list(d)
  lst$e <- ifelse(is.na(d$e), NA, d$e)
  lst
}
out$data$mgus2 <- store_data(mgus_long)
out$data$sim <- store_data(sim_long)

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
  # A step function (right-continuous) of the grid, its first value
  # before the grid starts.
  idx <- findInterval(times, grid)
  values[pmax(idx, 1)]
}

nums <- function(v) unname(as.numeric(v))

survfit_ref <- function(long, states, times, start = NULL, p0 = NULL) {
  ev <- factor(ifelse(is.na(long$e), "censor", long$e),
               levels = c("censor", states))
  ist <- factor(long$state, levels = states)
  long$ev <- ev; long$ist <- ist
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

probtrans_ref <- function(pt, states, times) {
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

add <- function(id, call, settings, values) {
  out$references[[id]] <<- list(call = call, settings = settings,
                                values = values)
}

# ---------------------------------------------------------------------------
# survfit: occupation probabilities and the robust std.err
# ---------------------------------------------------------------------------
mgus_times <- c(12, 60, 120, 240, 360, 420)
sim_times <- c(0.5, 1, 1.5, 2, 3, 4, 5, 6.5, 7.5)

add("survfit_mgus2", "survfit(Surv(tl, x, ev) ~ 1, id = id, istate = state, timefix = FALSE)",
    "from the start, p0 the states at risk then",
    survfit_ref(mgus_long, mgus_states, mgus_times))
add("survfit_sim", "survfit(Surv(tl, x, ev) ~ 1, id = id, istate = state, timefix = FALSE)",
    "from the start (the earliest entry), p0 the states at risk then",
    survfit_ref(sim_long, sim_states, sim_times))
add("survfit_sim_s2", "survfit(..., start.time = 2)",
    "from s = 2, p0 the states at risk then (healthy and ill)",
    survfit_ref(sim_long, sim_states, sim_times[sim_times >= 2], start = 2))
add("survfit_sim_s2_ill", "survfit(..., start.time = 2, p0 = c(0, 1, 0))",
    "from s = 2 in 'ill'",
    survfit_ref(sim_long, sim_states, sim_times[sim_times >= 2], start = 2,
                p0 = c(0, 1, 0)))

# ---------------------------------------------------------------------------
# mstate, no covariates: P(s, t) with the Greenwood and Aalen variances
# ---------------------------------------------------------------------------
mgus_st <- stack(mgus_long, mgus_tmat, c("z1", "male"))
sim_st <- stack(sim_long, sim_tmat)
for (v in c("greenwood", "aalen")) {
  for (case in list(list("mgus2", mgus_st, mgus_tmat, mgus_states,
                         mgus_times, 0),
                    list("mgus2_s60", mgus_st, mgus_tmat, mgus_states,
                         mgus_times[mgus_times >= 60], 60),
                    list("sim", sim_st, sim_tmat, sim_states, sim_times, 0),
                    list("sim_s2", sim_st, sim_tmat, sim_states,
                         sim_times[sim_times >= 2], 2))) {
    st <- case[[2]]
    c0 <- coxph(Surv(Tstart, Tstop, status) ~ strata(trans), data = st,
                method = "breslow")
    msf <- msfit(c0, trans = case[[3]], vartype = v)
    pt <- probtrans(msf, predt = case[[6]], method = v)
    add(paste0("mstate_", case[[1]], "_", v),
        sprintf("probtrans(msfit(coxph(Surv(Tstart, Tstop, status) ~ strata(trans), method = 'breslow'), vartype = '%s'), predt = %g, method = '%s')",
                v, case[[6]], v),
        paste("P(s, t) from each state, s =", case[[6]]),
        probtrans_ref(pt, case[[4]], case[[5]]))
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
      pt <- probtrans(msf, predt = s)
      vals <- probtrans_ref(pt, mgus_states, mgus_times[mgus_times >= s])
      vals$coef <- nums(coef(cx))
      vals$coef_names <- names(coef(cx))
      vals$se_coef <- nums(sqrt(diag(vcov(cx))))
      vals$profile <- as.list(profile)
      add(paste0("mstate_cox_mgus2_", name, "_", ties, "_s", s),
          sprintf("probtrans(msfit(coxph(%s, method = '%s'), newdata), predt = %g)",
                  deparse(fits[[name]], width.cutoff = 500L), ties, s),
          paste("covariates", name, "per transition; profile z1 = 0.5,",
                "male = 1; method 'aalen'"),
          vals)
    }
  }
}

text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE)
dir.create("surpyval/tests/multistate/data", showWarnings = FALSE,
           recursive = TRUE)
writeLines(text, "surpyval/tests/multistate/data/r_multistate.json")
