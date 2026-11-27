# Reference results from R's frailtyEM for the inverse Gaussian and
# positive stable shared frailties with a Cox baseline,
# CoxFrailty.fit(..., family="inverse_gaussian" / "positive_stable").
#
# Writes surpyval/tests/reference/data/r_frailtyem.json, which
# surpyval/tests/univariate/regression/test_frailty_laplace_families.py
# reads when it is present (the test is skipped until then): fits to the
# kidney catheter data (R's survival::kidney, which is
# surpyval.datasets.load_kidney), with the software, its version, the call
# and the values of each.
#
#     Rscript scripts/reference/reference_r_frailtyem.R
#
# Needs survival, frailtyEM and jsonlite. NOT RUN YET: written against the
# frailtyEM documentation (emfrail with emfrail_dist(dist = "stable") and
# emfrail_dist(dist = "pvf", pvfm = -1/2), the inverse Gaussian), whose
# frailty parameter theta is not SurPyval's: the test compares Kendall's
# tau, which every parameterisation shares, as SurPyval computes it from
# the theta this script converts. The conversions below are as the
# documentation is read here (positive stable: index theta / (1 + theta);
# inverse Gaussian: variance 1 / theta), and are to be checked on the
# first run against the tau frailtyEM's summary prints. frailtyEM fits
# with Breslow's ties, the ties SurPyval's fit of these families takes.

suppressPackageStartupMessages({
    library(survival)
    library(frailtyEM)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!dir.exists(data_dir)) stop("Run from the repository root")

data(kidney, package = "survival")
kidney$female <- as.integer(kidney$sex == 2)

fit_one <- function(dist) {
    m <- emfrail(Surv(time, status) ~ age + female + cluster(id),
                 data = kidney, distribution = dist)
    theta_em <- exp(m$logtheta)
    list(
        coef = unname(coef(m)),
        se = unname(sqrt(diag(m$var))),
        theta_frailtyem = theta_em,
        loglik = m$loglik[2],
        loglik_no_frailty = m$loglik[1]
    )
}

stable <- fit_one(emfrail_dist(dist = "stable"))
stable$theta <- 1 - stable$theta_frailtyem / (1 + stable$theta_frailtyem)
ig <- fit_one(emfrail_dist(dist = "pvf", pvfm = -0.5))
ig$theta <- 1 / ig$theta_frailtyem

out <- list(
    software = paste("frailtyEM", packageVersion("frailtyEM")),
    r = R.version.string,
    call = "emfrail(Surv(time, status) ~ age + female + cluster(id), kidney)",
    positive_stable = stable,
    inverse_gaussian = ig
)
write(toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE),
      file.path(data_dir, "r_frailtyem.json"))
