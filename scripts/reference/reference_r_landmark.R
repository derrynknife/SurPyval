# Reference results from R for landmark analysis and dynamic prediction
# (surpyval.landmark, surpyval.LandmarkCox and
# surpyval.landmark_kaplan_meier), for
# surpyval/tests/reference/test_landmark.py.
#
# Writes surpyval/tests/reference/data/r_landmark.json, with the
# software, its version, the call and the settings of each entry. Run
# from the repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with dynpred in the library path (it is archived on CRAN; install
# dynpred_0.1.2.tar.gz from the CRAN archive with R CMD INSTALL):
#
#     R_LIBS_USER=<library with dynpred> \
#         Rscript scripts/reference/reference_r_landmark.R
#
# Written with R 4.6.2, survival 3.8.12 and dynpred 0.1.2. Re-running
# reproduces the file byte for byte.
#
# The fixture "landmark" (fixtures.json): 300 units with a fixed
# covariate z1 and a covariate z2 that switches from 0 to 1 at v.
#
# What each reference is:
#
# * dynpred::cutLM(..., format = "wide", covs = list(fixed = "z1",
#   varying = "v")) at each landmark: the landmark data set, whose
#   time-varying covariate is 1 where v <= LM (the value at or before
#   the landmark). Its rows (unit, time, status, z2) are stored. (cutLM
#   has no delayed entry: unit 2, which enters at 0.5, is dropped from
#   the landmark 0 by hand, as it is not yet at risk there.)
# * coxph(Surv(time, status) ~ z1 + z2) on the landmark data at 2: a
#   Cox model at one landmark.
# * The supermodels of van Houwelingen and Putter (2011, chapter 7) on
#   the stacked landmark data at 0, 1, 2, 3, 4 with the window 3, the
#   effects quadratic in u = (LM - 0) / (4 - 0):
#   coxph(Surv(start, time, status) ~ z1 + z2 + z1u + z1u2 + z2u + z2u2
#   + strata(LM) + cluster(id)) (z1u the product z1 u, and so on) (stratified), and the same with u + u2
#   in place of strata(LM) (pooled, the "proportional baselines" model).
#   The coefficients, their robust (clustered) and model-based standard
#   errors, and the window probabilities 1 - S(LM + 3) / S(LM) from
#   survfit(fit, newdata) for the four profiles (z1, z2) in {0, 1}^2, at
#   each landmark (and at 1.5 for the pooled model).
suppressPackageStartupMessages({
  library(survival)
  library(dynpred)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_landmark.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
cols <- fx$landmark$columns
d <- data.frame(
  id = cols$id, entry = cols$entry, time = cols$time,
  status = cols$status, z1 = cols$z1,
  v = ifelse(is.na(cols$v), Inf, cols$v)
)
v_dynpred <- as.character(packageVersion("dynpred"))
v_survival <- as.character(packageVersion("survival"))

nums <- function(x) unname(as.numeric(x))
references <- list()
add <- function(id, software, version, call, settings, values) {
  references[[id]] <<- list(
    fixture = "landmark", software = software, version = version,
    call = call, settings = settings, values = values
  )
}

at <- c(0, 1, 2, 3, 4)
w <- 3
lm_data <- function(LM) {
  out <- cutLM(d, outcome = list(time = "time", status = "status"),
               LM = LM, horizon = LM + w,
               covs = list(fixed = c("id", "z1", "entry"), varying = "v"))
  out <- out[out$entry <= LM, ]
  out$z2 <- out$v
  out
}

for (LM in at) {
  ld <- lm_data(LM)
  ld <- ld[order(ld$id), ]
  add(sprintf("cutLM_%g", LM), "dynpred", v_dynpred,
      sprintf("cutLM(d, LM = %g, horizon = %g, covs = list(fixed = 'z1', varying = 'v'))", LM, LM + w),
      list(landmark = LM, window = w, note = "units entering after LM dropped"),
      list(id = nums(ld$id), time = nums(ld$time), status = nums(ld$status),
           z2 = nums(ld$z2)))
}

single <- lm_data(2)
f1 <- coxph(Surv(time, status) ~ z1 + z2, data = single, ties = "efron")
newd <- expand.grid(z1 = c(0, 1), z2 = c(0, 1))
sf1 <- summary(survfit(f1, newdata = newd), times = 2 + w, extend = TRUE)
add("cox_landmark_2", "survival", v_survival,
    "coxph(Surv(time, status) ~ z1 + z2, data = cutLM(d, LM = 2, horizon = 5))",
    list(landmark = 2, window = w, ties = "efron"),
    list(coef = nums(coef(f1)), se = nums(sqrt(diag(vcov(f1)))),
         profiles = list(z1 = newd$z1, z2 = newd$z2),
         probability = nums(1 - sf1$surv)))

stacked <- do.call(rbind, lapply(at, lm_data))
stacked$LM <- rep(at, sapply(at, function(L) nrow(lm_data(L))))
stacked$start <- stacked$LM
stacked$u <- (stacked$LM - min(at)) / (max(at) - min(at))
stacked$u2 <- stacked$u^2
# The products as columns of their own: survfit.coxph refuses an
# interaction term without its main effect (u alone, in the stratified
# model), though the model is the same
add_products <- function(df) {
  df$z1u <- df$z1 * df$u
  df$z1u2 <- df$z1 * df$u2
  df$z2u <- df$z2 * df$u
  df$z2u2 <- df$z2 * df$u2
  df
}
stacked <- add_products(stacked)

window_prob <- function(fit, s, strat) {
  u <- (s - min(at)) / (max(at) - min(at))
  nd <- add_products(data.frame(z1 = newd$z1, z2 = newd$z2, u = u,
                                u2 = u^2, LM = s, start = s))
  out <- c()
  for (k in seq_len(nrow(nd))) {
    sf <- survfit(fit, newdata = nd[k, ])
    S <- stepfun(sf$time, c(1, sf$surv))
    out <- c(out, 1 - S(s + w) / S(s))
  }
  out
}

forms <- list(
  stratified = Surv(start, time, status) ~ z1 + z2 + z1u + z1u2 + z2u +
    z2u2 + strata(LM) + cluster(id),
  pooled = Surv(start, time, status) ~ z1 + z2 + z1u + z1u2 + z2u +
    z2u2 + u + u2 + cluster(id)
)
for (name in names(forms)) {
  for (ties in c("efron", "breslow")) {
    fit <- coxph(forms[[name]], data = stacked, ties = ties)
    s_pred <- if (name == "pooled") c(at, 1.5) else at
    probs <- lapply(s_pred, function(s) nums(window_prob(fit, s)))
    add(sprintf("supermodel_%s_%s", name, ties), "survival", v_survival,
        sprintf("coxph(%s, data = stacked, ties = '%s')",
                paste(deparse(forms[[name]], width.cutoff = 500L), collapse = ""),
                ties),
        list(at = at, window = w, degree = 2, baseline = name, ties = ties,
             u = "(LM - 0) / (4 - 0)"),
        list(coef = nums(coef(fit)), coef_names = names(coef(fit)),
             robust_se = nums(sqrt(diag(vcov(fit)))),
             naive_se = nums(sqrt(diag(fit$naive.var))),
             n_rows = nrow(stacked),
             landmarks = s_pred,
             profiles = list(z1 = newd$z1, z2 = newd$z2),
             probability = probs))
  }
}

out <- list(
  generator = "scripts/reference/reference_r_landmark.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
