# Reference results from R for the semi-parametric recurrent-event
# regressions: Andersen-Gill, Prentice-Williams-Peterson (total and gap
# time) and Wei-Lin-Weissfeld, with the robust variance clustered by
# subject.
#
# Writes surpyval/tests/reference/data/r_recurrent_cox.json in the layout
# of the other reference files: for each fit the fixture, the software,
# its version, the call, the settings and the values. Generated with
# R 4.6.2 and survival 3.8.12.
#
#     python scripts/reference/make_fixtures.py
#     Rscript scripts/reference/reference_r_recurrent_cox.R
#
# Run from the repository root, after make_fixtures.py. Needs survival
# and jsonlite. Re-running reproduces the file byte for byte.
#
# The data are survival's bladder cancer recurrences (Wei, Lin and
# Weissfeld 1989), each subject's first four recurrences, in its two
# layouts: bladder2, start-stop rows (the shared fixture "bladder2",
# checked here against R's copy), and bladder, one row per subject and
# recurrence number (the Wei-Lin-Weissfeld layout, which the test builds
# from bladder2's rows).
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

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_recurrent_cox.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
b2 <- as.data.frame(fx$bladder2$columns)
for (col in c("id", "rx", "number", "size", "start", "stop", "event")) {
    stopifnot(isTRUE(all.equal(b2[[col]], as.numeric(bladder2[[col]]))))
}
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

entry <- function(call, fit, data) {
    list(
        fixture = "bladder2",
        software = "R survival",
        version = as.character(packageVersion("survival")),
        call = call,
        settings = list(
            data = data,
            variance = "robust (var), clustered by id; naive.var beside it",
            baseline = "basehaz(fit, centered = FALSE) at times"
        ),
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

layouts <- list(
    ag = "bladder2, start-stop rows",
    ag_breslow = "bladder2, start-stop rows",
    pwp_total = "bladder2, start-stop rows",
    pwp_gap = "bladder2, each row's gap (stop - start)",
    wlw = paste(
        "bladder, a row per subject and recurrence number 1 to 4",
        "(the test builds it from bladder2's rows)"
    )
)
references <- lapply(names(calls), function(name) {
    item <- calls[[name]]
    entry(sprintf(item[[1]], rhs), item[[2]](), layouts[[name]])
})
names(references) <- names(calls)

out <- list(
    generator = "scripts/reference/reference_r_recurrent_cox.R",
    R = R.version.string,
    references = references
)
writeLines(
    toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), out_file
)
