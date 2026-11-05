# Reference results from R flexsurv for the documented CustomDistribution
# recipes of the generalised gamma and the generalised F in Prentice's
# parameterisations (flexsurv's "gengamma", (mu, sigma, Q), and "genf",
# (mu, sigma, Q, P)); see "Recipes: more distributions from their
# cumulative hazard" in docs/Parametric SurPyval Modelling.rst.
#
# Writes surpyval/tests/reference/data/r_custom_recipes.json, with the
# software, its version, the call and the values of each entry, as the
# other generators do. The fits are to the lung fixture already in
# fixtures.json (survival's lung data), so make_fixtures.py needs no
# change.
#
#     Rscript scripts/reference/reference_r_custom_recipes.R
#
# Needs survival, flexsurv and jsonlite. Run with R 4.6.2, survival 3.8.12
# and flexsurv 2.3.2. Re-running reproduces the file byte for byte.
#
# What each value is:
#
# * The survival function, pgengamma / pgenf(lower.tail = FALSE), at
#   fixed times and parameters, including a negative Q (the branch where
#   the survival is the lower incomplete gamma) and both signs of Q for
#   the generalised F.
# * The fits: flexsurvreg(Surv(time, status) ~ 1) and ~ sex on lung,
#   with optim's relative tolerance tightened to 1e-14: the maximised
#   log-likelihood, the estimates and their standard errors (on the
#   natural scale, from the inverse observed information). A covariate in
#   flexsurvreg acts on mu, the location of log(T), which is the
#   accelerated failure time model; flexsurv's coefficient is the shift in
#   mu, so SurPyval's AFT coefficient (positive = shorter life) is its
#   negative.

suppressPackageStartupMessages({
    library(survival)
    library(flexsurv)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
lung_d <- as.data.frame(fx[["lung"]]$columns)
lung_d$status <- 1 - lung_d$c
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(lung_d$sex, survival::lung$sex))
)

versions <- list(
    survival = as.character(packageVersion("survival")),
    flexsurv = as.character(packageVersion("flexsurv"))
)

refs <- list()
add <- function(id, entry) refs[[id]] <<- entry

# The survival functions at fixed parameters
times <- c(0.5, 2, 5, 12)
gg_pars <- list(c(1, 0.7, 0.5), c(1, 0.7, -0.8), c(1, 0.7, 2))
gf_pars <- list(c(1, 0.7, 0.5, 1.3), c(1, 0.7, -0.5, 0.6))
add("sf_gengamma", list(
    fixture = "lung", software = "R flexsurv",
    version = versions$flexsurv,
    call = "pgengamma(t, mu, sigma, Q, lower.tail = FALSE)",
    settings = list(t = times, params = gg_pars,
                    note = "no data: the fixture field is nominal"),
    values = list(sf = lapply(gg_pars, function(p)
        pgengamma(times, p[1], p[2], p[3], lower.tail = FALSE)))
))
add("sf_genf", list(
    fixture = "lung", software = "R flexsurv",
    version = versions$flexsurv,
    call = "pgenf(t, mu, sigma, Q, P, lower.tail = FALSE)",
    settings = list(t = times, params = gf_pars,
                    note = "no data: the fixture field is nominal"),
    values = list(sf = lapply(gf_pars, function(p)
        pgenf(times, p[1], p[2], p[3], p[4], lower.tail = FALSE)))
))

# The fits
tight <- list(reltol = 1e-14, maxit = 10000)
fit_entry <- function(dist, formula, label) {
    fit <- flexsurvreg(formula, data = lung_d, dist = dist,
                       control = tight)
    list(
        fixture = "lung", software = "R flexsurv",
        version = versions$flexsurv,
        call = sprintf(
            "flexsurvreg(%s, data = lung, dist = '%s', control = list(reltol = 1e-14))",
            label, dist),
        settings = list(dist = dist, formula = label,
                        covariate = "sex (1 male, 2 female), on mu: AFT"),
        values = list(
            loglik = fit$loglik,
            names = rownames(fit$res),
            est = unname(fit$res[, "est"]),
            se = unname(fit$res[, "se"])
        )
    )
}
for (dist in c("gengamma", "genf")) {
    add(paste0("lung_", dist),
        fit_entry(dist, Surv(time, status) ~ 1, "Surv(time, status) ~ 1"))
    add(paste0("lung_", dist, "_sex"),
        fit_entry(dist, Surv(time, status) ~ sex,
                  "Surv(time, status) ~ sex"))
}

out <- list(
    generator = "scripts/reference/reference_r_custom_recipes.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_custom_recipes.json"))
cat("wrote", file.path(data_dir, "r_custom_recipes.json"), "\n")
