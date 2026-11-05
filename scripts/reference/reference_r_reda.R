# Reference results from R's reda for the two-sample comparison of mean
# cumulative functions (surpyval.recurrent.mcf_diff).
#
# Writes surpyval/tests/reference/data/r_reda.json, with the software, its
# version, the exact call and the settings of each entry, as reference_r.R
# does:
#
#     Rscript scripts/reference/reference_r_reda.R
#
# Run make_fixtures.py first. Needs reda and jsonlite. Re-running
# reproduces the file byte for byte.
#
# The data are the seeded synthetic recurrent-event fixtures of
# make_fixtures.recurrent_groups (two groups of systems, a gamma frailty
# per system, group 1 followed for less time), one with two-decimal times
# and one with whole-number event times that tie across systems and
# groups. For each:
#
# * mcf(Recur(time, id, event) ~ group): each group's Nelson-Aalen MCF and
#   its Lawless-Nadeau (robust) standard error, at every time of the group.
# * mcfDiff(mcf_fit): the difference of the MCFs (group 0 minus group 1)
#   and its standard error, sqrt(se_0^2 + se_1^2), at every time of either
#   group. reda carries each MCF at its last value past the group's last
#   time; SurPyval leaves the difference undefined there, so the tests
#   compare the times up to the shorter follow-up.
# * mcfDiff.test(mcf_fit, testVariance = "robust" | "Poisson"): the
#   pseudo-score statistics of Cook, Lawless and Nadeau (1996) with the
#   constant and linear weights, their variances, chi-squared statistics
#   and p-values.

suppressPackageStartupMessages({
    library(reda)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
version <- as.character(packageVersion("reda"))

entry <- function(name) {
    d <- as.data.frame(fx[[name]]$columns)
    fit <- mcf(Recur(time, id, event) ~ group, data = d)
    tab <- fit@MCF
    g0 <- tab[tab$group == 0, ]
    g1 <- tab[tab$group == 1, ]
    diff <- mcfDiff(fit)
    robust <- mcfDiff.test(fit, testVariance = "robust")
    poisson <- mcfDiff.test(fit, testVariance = "Poisson")
    test_values <- function(t) {
        list(
            statistic = unname(t[, 1]),
            variance = unname(t[, 2]),
            chisq = unname(t[, 3]),
            p_value = unname(t[, 5])
        )
    }
    list(
        fixture = name,
        software = "R reda",
        version = version,
        call = paste(
            "fit <- mcf(Recur(time, id, event) ~ group, data = d);",
            "mcfDiff(fit); mcfDiff.test(fit, testVariance = 'robust');",
            "mcfDiff.test(fit, testVariance = 'Poisson')"
        ),
        settings = list(
            variance = "mcf: Lawless-Nadeau (reda's default)",
            difference = "group 0 minus group 1",
            tests = "rows constant, linear weights; 1 degree of freedom"
        ),
        values = list(
            time_0 = g0$time, mcf_0 = g0$MCF, se_0 = g0$se,
            time_1 = g1$time, mcf_1 = g1$MCF, se_1 = g1$se,
            time = diff@MCF$time, diff = diff@MCF$MCF, diff_se = diff@MCF$se,
            robust = test_values(robust),
            poisson = test_values(poisson)
        )
    )
}

out <- list(
    generator = "scripts/reference/reference_r_reda.R",
    R = R.version.string,
    references = list(
        recurrent_groups = entry("recurrent_groups"),
        recurrent_groups_ties = entry("recurrent_groups_ties")
    )
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_reda.json"))
cat("wrote", file.path(data_dir, "r_reda.json"), "\n")
