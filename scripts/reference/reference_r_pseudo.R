# Reference results from R for the pseudo-observations and the regressions
# fitted to them.
#
# Writes surpyval/tests/reference/data/r_pseudo.json, with the software,
# its version, the exact call and the settings of each entry, as
# reference_r.R does:
#
#     Rscript scripts/reference/reference_r_pseudo.R
#
# Run from the repository root, after make_fixtures.py. Needs survival,
# pseudo, geepack, survRM2 and jsonlite. Re-running reproduces the file
# byte for byte.
#
# * pseudo::pseudosurv, pseudo::pseudomean and pseudo::pseudoci: the
#   jackknife (leave-one-out) pseudo-values of the Kaplan-Meier survival
#   at some times, of the restricted mean survival time to tau, and of the
#   Aalen-Johansen cumulative incidence of each cause at some times.
# * geepack::geeglm on those pseudo-values: an independence working
#   correlation, one cluster per unit (id), the gaussian variance (each
#   pseudo-value counts the same) and the robust (sandwich) standard
#   errors. With several times the model has one intercept per time
#   (factor(time) - 1) and the covariates' coefficients common to them.
#   SurPyval models the event probability, 1 - S(t), so the survival
#   regressions are fitted to 1 - pseudosurv; its RMST coefficients have
#   the opposite sign of these (a positive coefficient means a shorter
#   life), which the tests allow for.
# * survRM2::rmst2 with covariates: an IPCW regression, not pseudo-values,
#   stored only as a loose cross-check of the RMST regression.

suppressPackageStartupMessages({
    library(survival)
    library(pseudo)
    library(geepack)
    library(survRM2)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

# lung: SurPyval's flag c (0 died, 1 censored); ovarian: fustat (1 died);
# competing: integer times with ties, cause 0 censored; pbc: cause 1
# transplant, 2 death, 0 censored.
lung <- fixture("lung")
lung$event <- 1 - lung$c
ovarian <- fixture("ovarian")
cr <- fixture("competing")
pbc <- fixture("pbc")

refs <- list()
env <- environment()
finite_or_na <- function(v) {
    if (is.list(v)) return(lapply(v, finite_or_na))
    if (is.numeric(v)) {
        v <- unname(v)
        v[!is.finite(v)] <- NA
        if (is.matrix(v)) return(lapply(seq_len(nrow(v)), function(k) v[k, ]))
    }
    v
}
record <- function(id, fixture, package, call, settings, extract) {
    result <- eval(parse(text = call), envir = env)
    refs[[id]] <<- list(
        fixture = fixture,
        software = paste("R", package),
        version = as.character(packageVersion(package)),
        call = call,
        settings = settings,
        values = finite_or_na(extract(result))
    )
    invisible(result)
}

# ---------------------------------------------------------------------------
# The pseudo-values.
# ---------------------------------------------------------------------------
pv_surv <- function(r) list(time = r$time, pseudo = r$pseudo)
record("pseudosurv_lung", "lung", "pseudo",
       "pseudosurv(lung$time, lung$event, tmax = c(180, 365, 540))",
       list(time = "time", event = "1 - c"), pv_surv)
record("pseudosurv_ovarian", "ovarian", "pseudo",
       "pseudosurv(ovarian$futime, ovarian$fustat, tmax = c(365, 730))",
       list(time = "futime", event = "fustat"), pv_surv)
record("pseudomean_lung", "lung", "pseudo",
       "pseudomean(lung$time, lung$event, tmax = 365)",
       list(time = "time", event = "1 - c", tau = 365),
       function(r) list(pseudo = r))
record("pseudomean_ovarian", "ovarian", "pseudo",
       "pseudomean(ovarian$futime, ovarian$fustat, tmax = 1000)",
       list(time = "futime", event = "fustat", tau = 1000),
       function(r) list(pseudo = r))
pv_ci <- function(r) list(time = r$time, cause1 = r$pseudo$cause1,
                          cause2 = r$pseudo$cause2)
record("pseudoci_competing", "competing", "pseudo",
       "pseudoci(cr$x, cr$cause, tmax = c(2, 4, 8))",
       list(time = "x", cause = "cause (0 censored)",
            data = "integer times: causes tie with each other and censorings"),
       pv_ci)
record("pseudoci_pbc", "pbc", "pseudo",
       "pseudoci(pbc$years, pbc$cause, tmax = c(2, 5, 8))",
       list(time = "years", cause = "cause (0 censored)"), pv_ci)

# ---------------------------------------------------------------------------
# The regressions on them.
# ---------------------------------------------------------------------------
# Long data: one row per unit and time, sorted by unit (geeglm's clusters
# must be contiguous).
long <- function(d, pv, times, covs) {
    k <- length(times)
    out <- d[rep(seq_len(nrow(d)), each = k), covs, drop = FALSE]
    out$id <- rep(seq_len(nrow(d)), each = k)
    out$tt <- factor(rep(times, nrow(d)))
    out$y <- as.vector(t(pv))
    out
}
# glm's default start (mu = y) fails where a pseudo-value is outside the
# link's range, so its search starts from the values clipped into it
# (geeglm passes `start` on to model.frame, so `mustart` is used, as a
# column); the gaussian estimating equations have one root, which glm and
# then geese find.
gee <- function(formula, data, link) {
    lo <- if (link == "identity") -Inf else 1e-3
    hi <- if (link %in% c("cloglog", "logit")) 1 - 1e-3 else Inf
    data$mu0 <- pmin(pmax(data$y, lo), hi)
    geeglm(formula, data = data, id = id, corstr = "independence",
           family = gaussian(link = link), mustart = mu0,
           control = geese.control(epsilon = 1e-12, maxit = 100))
}
gee_values <- function(fit) {
    s <- summary(fit)$coefficients
    list(names = rownames(s), coef = unname(s[, "Estimate"]),
         se = unname(s[, "Std.err"]), p = unname(s[, "Pr(>|W|)"]))
}
gee_settings <- function(extra = list()) {
    c(list(corstr = "independence", family = "gaussian",
           std.err = "san.se (robust sandwich)",
           geepack = as.character(packageVersion("geepack")),
           pseudo = as.character(packageVersion("pseudo"))), extra)
}

lung_times <- c(180, 365, 540)
lung_F <- long(lung, 1 - pseudosurv(lung$time, lung$event,
                                    tmax = lung_times)$pseudo,
               lung_times, c("sex", "age"))
for (link in c("cloglog", "logit")) {
    record(paste0("gee_survival_lung_", link), "lung", "geepack",
           sprintf("gee(y ~ tt + sex + age - 1, lung_F, '%s')", link),
           gee_settings(list(response = "1 - pseudosurv at 180, 365, 540",
                             link = link)),
           gee_values)
}
ov_times <- c(365, 730)
ov_F <- long(ovarian, 1 - pseudosurv(ovarian$futime, ovarian$fustat,
                                     tmax = ov_times)$pseudo,
             ov_times, c("rx", "age"))
record("gee_survival_ovarian_cloglog", "ovarian", "geepack",
       "gee(y ~ tt + rx + age - 1, ov_F, 'cloglog')",
       gee_settings(list(response = "1 - pseudosurv at 365, 730",
                         link = "cloglog")),
       gee_values)
# One time: a single intercept.
lung_F1 <- long(lung, 1 - pseudosurv(lung$time, lung$event,
                                     tmax = 365)$pseudo,
                365, c("sex", "age"))
record("gee_survival_lung_365_cloglog", "lung", "geepack",
       "gee(y ~ sex + age, lung_F1, 'cloglog')",
       gee_settings(list(response = "1 - pseudosurv at 365",
                         link = "cloglog")),
       gee_values)

lung_rm <- long(lung, matrix(pseudomean(lung$time, lung$event,
                                        tmax = 365)),
                365, c("sex", "age"))
ov_rm <- long(ovarian, matrix(pseudomean(ovarian$futime, ovarian$fustat,
                                         tmax = 1000)),
              1000, c("rx", "age"))
for (link in c("identity", "log")) {
    record(paste0("gee_rmst_lung_", link), "lung", "geepack",
           sprintf("gee(y ~ sex + age, lung_rm, '%s')", link),
           gee_settings(list(response = "pseudomean to 365", link = link)),
           gee_values)
    record(paste0("gee_rmst_ovarian_", link), "ovarian", "geepack",
           sprintf("gee(y ~ rx + age, ov_rm, '%s')", link),
           gee_settings(list(response = "pseudomean to 1000", link = link)),
           gee_values)
}

cr_times <- c(2, 4, 8)
cr_ci <- pseudoci(cr$x, cr$cause, tmax = cr_times)$pseudo
pbc_times <- c(2, 5, 8)
pbc_ci <- pseudoci(pbc$years, pbc$cause, tmax = pbc_times)$pseudo
for (link in c("cloglog", "logit")) {
    for (k in 1:2) {
        cr_long <- long(cr, cr_ci[[k]], cr_times, c("group", "z"))
        assign(sprintf("cr_long%d", k), cr_long)
        record(sprintf("gee_cif_competing_cause%d_%s", k, link),
               "competing", "geepack",
               sprintf("gee(y ~ tt + group + z - 1, cr_long%d, '%s')",
                       k, link),
               gee_settings(list(response = sprintf(
                   "pseudoci cause %d at 2, 4, 8", k), link = link)),
               gee_values)
    }
}
pbc_long2 <- long(pbc, pbc_ci$cause2, pbc_times,
                  c("drug", "age", "log_bili"))
record("gee_cif_pbc_cause2_cloglog", "pbc", "geepack",
       "gee(y ~ tt + drug + age + log_bili - 1, pbc_long2, 'cloglog')",
       gee_settings(list(response = "pseudoci cause 2 (death) at 2, 5, 8",
                         link = "cloglog")),
       gee_values)

# ---------------------------------------------------------------------------
# survRM2's IPCW regression: a loose cross-check only.
# ---------------------------------------------------------------------------
lung_arm <- lung$sex - 1
record("rmst2_lung", "lung", "survRM2",
       paste("rmst2(lung$time, lung$event, lung_arm, tau = 365,",
             "covariates = matrix(lung$age))"),
       list(arm = "sex - 1 (1 female)", covariates = "age", tau = 365,
            note = "IPCW regression, not pseudo-values: a loose check"),
       function(r) {
           m <- r$RMST.difference.adjusted
           list(names = rownames(m), coef = unname(m[, "coef"]),
                se = unname(m[, "se(coef)"]))
       })

out <- list(
    generator = "scripts/reference/reference_r_pseudo.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_pseudo.json"))
cat("wrote", file.path(data_dir, "r_pseudo.json"), "\n")
