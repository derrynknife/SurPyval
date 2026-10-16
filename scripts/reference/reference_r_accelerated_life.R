# Reference results from R for the accelerated life diagnostics (#871),
# surpyval/tests/diagnostics/test_accelerated_life.py.
#
# The data are the insulating-fluid breakdown times of Nelson (1990,
# Accelerated Testing, chapter 3): 76 times to breakdown, in minutes, at
# seven voltages from 26 to 38 kV. The test fits them as they are
# (complete) and censored at 500 minutes (Type I censoring, which leaves
# the 26 kV level one failure), with the Weibull, the LogNormal and (with
# no shape to test) the Exponential.
#
# Each case records, from survival::survreg:
#   - the log-likelihood of each level fitted alone (separate shape and
#     life) and their sum,
#   - the log-likelihood of the common-shape, separate-life model
#     (~ factor(V)), the saturated model of the life-stress relationship,
#   - the log-likelihood of each life model that survreg can write as a
#     log-linear model of the life: Power (~ log V), Exponential/Arrhenius
#     (~ I(1 / V)), Eyring (~ I(1 / V) + offset(-log V)), InverseEyring
#     (~ I(1 / V) + offset(log V)) and GeneralLogLinear (~ V),
#   - the likelihood-ratio statistics and p-values of the two tests the
#     diagnostics compute: common shape (separate vs common shape, each
#     level its own life) and the life-stress relationship (the Power
#     model vs the separate lives).
# A level enters the tests only if it has at least two failures (the
# diagnostics' rule): the censored case leaves out 26 kV.
#
# Run from the repository root; prints JSON to paste into the test:
#
#     Rscript scripts/reference/reference_r_accelerated_life.R
#
# Needs R >= 4.3 with survival and jsonlite.

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
})

fluid <- list(
    "26" = c(5.79, 1579.52, 2323.70),
    "28" = c(68.85, 108.29, 110.29, 426.07, 1067.60),
    "30" = c(7.74, 17.05, 20.46, 21.02, 22.66, 43.40, 47.30, 139.07,
             144.12, 175.88, 194.90),
    "32" = c(0.27, 0.40, 0.69, 0.79, 2.75, 3.91, 9.88, 13.95, 15.93, 27.80,
             53.24, 82.85, 89.29, 100.59, 215.10),
    "34" = c(0.19, 0.78, 0.96, 1.31, 2.78, 3.16, 4.15, 4.67, 4.85, 6.50,
             7.35, 8.01, 8.27, 12.06, 31.75, 32.52, 33.91, 36.71, 72.89),
    "36" = c(0.35, 0.59, 0.96, 0.99, 1.69, 1.97, 2.07, 2.58, 2.71, 2.90,
             3.67, 3.99, 5.35, 13.77, 25.50),
    "38" = c(0.09, 0.39, 0.47, 0.73, 0.74, 1.13, 1.40, 2.38)
)
base <- data.frame(
    t = unlist(fluid, use.names = FALSE),
    V = rep(as.numeric(names(fluid)), lengths(fluid))
)

ll <- function(formula, data, dist) {
    as.numeric(logLik(survreg(formula, data = data, dist = dist)))
}

one_case <- function(data, dist) {
    shaped <- dist != "exponential"
    failures <- tapply(data$status, data$V, sum)
    tested <- as.numeric(names(failures)[failures >= 2])
    sub <- data[data$V %in% tested, ]
    sub$level <- factor(sub$V)
    separate <- sapply(tested, function(v) {
        ll(Surv(t, status) ~ 1, sub[sub$V == v, ], dist)
    })
    common <- ll(Surv(t, status) ~ level, sub, dist)
    life <- list(
        Power = ll(Surv(t, status) ~ log(V), sub, dist),
        Exponential = ll(Surv(t, status) ~ I(1 / V), sub, dist),
        Eyring = ll(Surv(t, status) ~ I(1 / V) + offset(-log(V)), sub, dist),
        InverseEyring = ll(
            Surv(t, status) ~ I(1 / V) + offset(log(V)), sub, dist
        ),
        GeneralLogLinear = ll(Surv(t, status) ~ V, sub, dist)
    )
    k <- length(tested)
    shape_stat <- 2 * (sum(separate) - common)
    lof_stat <- 2 * (common - life$Power)
    out <- list(
        dist = dist,
        tested_levels = tested,
        separate_loglik = unname(separate),
        common_shape_loglik = common,
        life_model_loglik = life,
        power_lof_lr = lof_stat,
        power_lof_df = k - 2,
        power_lof_p = pchisq(lof_stat, k - 2, lower.tail = FALSE)
    )
    if (shaped) {
        # The Exponential has no shape: each level alone is the model
        # with a life at each level, and there is no shape test.
        out$shape_lr <- shape_stat
        out$shape_df <- k - 1
        out$shape_p <- pchisq(shape_stat, k - 1, lower.tail = FALSE)
    }
    out
}

complete <- transform(base, status = 1)
censored <- transform(base, status = as.numeric(t <= 500), t = pmin(t, 500))

out <- list(
    software = paste("R", getRversion(), "survival",
                     as.character(packageVersion("survival"))),
    weibull_complete = one_case(complete, "weibull"),
    weibull_censored = one_case(censored, "weibull"),
    lognormal_complete = one_case(complete, "lognormal"),
    exponential_complete = one_case(complete, "exponential")
)
cat(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), "\n")
