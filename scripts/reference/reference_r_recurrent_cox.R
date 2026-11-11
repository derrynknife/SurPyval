# Reference results from R for the semi-parametric recurrent-event
# regressions: Andersen-Gill, Prentice-Williams-Peterson (total and gap
# time) and Wei-Lin-Weissfeld, with the robust variance clustered by
# subject.
#
# Writes surpyval/tests/reference/data/r_recurrent_cox.json: the data,
# and for each fit the software, its version, the call and the values.
# Generated with R 4.6.2 and survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_recurrent_cox.R
#
# Needs survival and jsonlite. Re-running reproduces the file byte for
# byte.
#
# The data are survival's bladder cancer recurrences (Wei, Lin and
# Weissfeld 1989), each subject's first four recurrences, in its two
# layouts: bladder2, start-stop rows (stored here, with the fits on it),
# and bladder, one row per subject and recurrence number (the
# Wei-Lin-Weissfeld layout, which the test builds from bladder2's rows).
#
# The fits, each with rx + size + number and cluster(id):
#
# * ag:        coxph(Surv(start, stop, event) ~ ...), bladder2
# * ag_breslow: the same with ties = "breslow"
# * pwp_total: ... + strata(enum), bladder2
# * pwp_gap:   coxph(Surv(stop - start, event) ~ ... + strata(enum))
# * wlw:       coxph(Surv(stop, event) ~ ... + strata(enum)), bladder
#
# What is recorded: the coefficients, the model-based (naive.var) and
# robust (var) covariances, the partial log-likelihood at 0 and at the
# fit, and the cumulative baseline at Z = 0 (basehaz(fit, centered =
# FALSE)) at a few times, by stratum where there are strata.

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
})

out_file <- file.path(
    "surpyval", "tests", "reference", "data", "r_recurrent_cox.json"
)
software <- list(
    name = "survival",
    version = as.character(packageVersion("survival")),
    r = paste(R.version$major, R.version$minor, sep = ".")
)
times <- c(5, 10, 20, 40)

baseline <- function(fit) {
    bh <- basehaz(fit, centered = FALSE)
    at <- function(b) {
        sapply(times, function(t) {
            before <- b$hazard[b$time <= t]
            if (length(before)) max(before) else 0
        })
    }
    if (is.null(bh$strata)) {
        return(list(all = at(bh)))
    }
    lapply(split(bh, bh$strata), at)
}

entry <- function(call, fit) {
    list(
        software = software,
        call = call,
        values = list(
            coef = unname(coef(fit)),
            naive_covariance = unname(fit$naive.var),
            robust_covariance = unname(fit$var),
            loglik = fit$loglik,
            times = times,
            baseline = baseline(fit)
        )
    )
}

rhs <- "rx + size + number + cluster(id)"
calls <- list(
    ag = list(
        "coxph(Surv(start, stop, event) ~ %s, bladder2)",
        function() coxph(
            Surv(start, stop, event) ~ rx + size + number + cluster(id),
            bladder2
        )
    ),
    ag_breslow = list(
        "coxph(Surv(start, stop, event) ~ %s, bladder2, ties = 'breslow')",
        function() coxph(
            Surv(start, stop, event) ~ rx + size + number + cluster(id),
            bladder2, ties = "breslow"
        )
    ),
    pwp_total = list(
        "coxph(Surv(start, stop, event) ~ %s + strata(enum), bladder2)",
        function() coxph(
            Surv(start, stop, event) ~ rx + size + number + strata(enum) +
                cluster(id),
            bladder2
        )
    ),
    pwp_gap = list(
        "coxph(Surv(stop - start, event) ~ %s + strata(enum), bladder2)",
        function() coxph(
            Surv(stop - start, event) ~ rx + size + number + strata(enum) +
                cluster(id),
            bladder2
        )
    ),
    wlw = list(
        "coxph(Surv(stop, event) ~ %s + strata(enum), bladder)",
        function() coxph(
            Surv(stop, event) ~ rx + size + number + strata(enum) +
                cluster(id),
            bladder
        )
    )
)

references <- lapply(calls, function(item) {
    entry(sprintf(item[[1]], rhs), item[[2]]())
})

out <- list(
    data = list(
        name = "bladder2",
        source = paste(
            "survival::bladder2 (Wei, Lin and Weissfeld 1989): start-stop",
            "rows of each subject's first four recurrences"
        ),
        columns = as.list(
            bladder2[c("id", "rx", "number", "size", "start", "stop", "event")]
        )
    ),
    references = references
)
writeLines(
    toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), out_file
)
