# Reference results from R for the competing-risks diagnostics (#874).
#
# Writes surpyval/tests/reference/data/r_competing_diagnostics.json, with
# the software, its version, the exact call (the call string is what is
# evaluated) and the settings of each entry, as reference_r.R does:
#
#     Rscript scripts/reference/reference_r_competing_diagnostics.R
#
# Run make_fixtures.py first. Needs survival, cmprsk and jsonlite (Ubuntu:
# r-cran-survival r-cran-cmprsk r-cran-jsonlite). Re-running reproduces
# the file byte for byte.
#
# * Cause-specific Cox: cox.zph (survival 3.0 and later) on each cause's
#   own coxph(Surv(time, cause == k) ~ ...), the other causes censored, on
#   the km and rank scales, and the score test of each continuous
#   covariate's square (coxph started at the fit with the square's
#   coefficient 0 and no iterations reports it as $score).
# * Fine-Gray: the Fine-Gray fit is the weighted Cox fit of the
#   counting-process data in which a unit that failed from a competing
#   cause stays at risk after its failure with the weight G(t-)/G(x_i-),
#   G the Kaplan-Meier estimate of the censoring distribution (Fine and
#   Gray 1999; survival::finegray builds the same data). fg_data() builds
#   them with cmprsk::crr's convention for ties (G just before each time:
#   at a time shared by events and censorings the events come first),
#   which is SurPyval's, and the script checks that coxph on them gives
#   crr's coefficients (and, on data without tied censorings, that
#   survival::finegray's data give the same). cox.zph on that fit is the
#   Grambsch-Therneau test of the weighted Schoenfeld residuals. Its
#   scales are passed as functions of time: "cif" is 1 - S(t-), S the
#   weighted Kaplan-Meier of those data, which is the Aalen-Johansen
#   cumulative incidence of the cause (checked against cmprsk::cuminc),
#   and "rank" the average rank of t among the original rows (cox.zph's
#   own rank would rank the split rows).

suppressPackageStartupMessages({
    library(survival)
    library(cmprsk)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

# pbc: the first row of each patient of surpyval.datasets.load_pbc2;
# cause 1 transplant, 2 death, 0 censored. competing: the seeded
# synthetic set of make_fixtures.competing, integer times with ties.
pbc_d <- fixture("pbc")
pbc_d$x <- pbc_d$years
cr_d <- fixture("competing")

# ---------------------------------------------------------------------------
# The Fine-Gray data.
# ---------------------------------------------------------------------------
fg_data <- function(d, k) {
    cens <- survfit(Surv(x, cause == 0) ~ 1, data = d)
    G_at <- function(t) c(1, cens$surv)[findInterval(t, cens$time) + 1]
    G_before <- function(t) {
        c(1, cens$surv)[findInterval(t, cens$time, left.open = TRUE) + 1]
    }
    last <- max(d$x[d$cause == k])
    out <- list()
    for (i in seq_len(nrow(d))) {
        row <- d[i, , drop = FALSE]
        first <- cbind(row, start = 0, stop = row$x,
                       status = as.numeric(row$cause == k), w = 1)
        out[[length(out) + 1]] <- first
        if (row$cause != 0 && row$cause != k && row$x < last) {
            # Constant weight between the distinct times after x_i.
            cuts <- sort(unique(d$x[d$x > row$x & d$x < last]))
            starts <- c(row$x, cuts)
            stops <- c(cuts, last)
            w <- G_at(starts) / G_before(row$x)
            rest <- row[rep(1, length(starts)), , drop = FALSE]
            out[[length(out) + 1]] <- cbind(rest, start = starts,
                                            stop = stops, status = 0, w = w)
        }
    }
    do.call(rbind, out)
}

# cox.zph's g on the cif scale: 1 - S(t-), S the weighted Kaplan-Meier of
# the Fine-Gray data; and on the rank scale: t's average rank among the
# original rows.
cif_before <- function(fg) {
    s <- survfit(Surv(start, stop, status) ~ 1, data = fg, weights = w)
    function(t) {
        1 - c(1, s$surv)[findInterval(t, s$time, left.open = TRUE) + 1]
    }
}
rank_among <- function(x) function(t) {
    sapply(t, function(s) sum(x < s) + (sum(x == s) + 1) / 2)
}

# ---------------------------------------------------------------------------
# Recording.
# ---------------------------------------------------------------------------
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

squares_of <- function(fit, squares) {
    out <- list()
    for (v in squares) {
        g <- update(fit, as.formula(sprintf(". ~ . + I(%s^2)", v)),
                    init = c(coef(fit), 0),
                    control = coxph.control(iter.max = 0))
        out[[paste0("square_", v)]] <- unname(g$score)
    }
    out
}

zph_table <- function(z, prefix) {
    out <- list()
    out[[paste0(prefix, "_chisq")]] <- unname(z$table[, "chisq"])
    out[[paste0(prefix, "_df")]] <- unname(z$table[, "df"])
    out[[paste0(prefix, "_p")]] <- unname(z$table[, "p"])
    out
}

ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 100)

# ---------------------------------------------------------------------------
# Cause-specific Cox.
# ---------------------------------------------------------------------------
cs_values <- function(squares) function(fit) {
    out <- list(coef = unname(coef(fit)))
    for (tr in c("km", "rank")) {
        out <- c(out, zph_table(cox.zph(fit, transform = tr), tr))
    }
    out$rows <- rownames(cox.zph(fit)$table)
    c(out, squares_of(fit, squares))
}
cs_settings <- list(
    ties = "efron",
    model = "cause-specific: the other causes and censoring are censored",
    transform = paste("km: 1 - S(t-), the left-continuous Kaplan-Meier",
                      "of all the rows; rank: rank(time) of all the rows"),
    squares = paste("update(fit, . ~ . + I(v^2), init = c(coef(fit), 0),",
                    "control = coxph.control(iter.max = 0))$score")
)
for (k in c(1, 2)) {
    record(sprintf("cs_pbc_cause%d", k), "pbc", "survival",
           sprintf(paste("coxph(Surv(years, cause == %d) ~ drug + age +",
                         "log_bili + female, data = pbc_d, ties = 'efron',",
                         "control = ctl)"), k),
           cs_settings, cs_values(c("age", "log_bili")))
    record(sprintf("cs_competing_cause%d", k), "competing", "survival",
           sprintf(paste("coxph(Surv(x, cause == %d) ~ group + z,",
                         "data = cr_d, ties = 'efron', control = ctl)"), k),
           cs_settings, cs_values("z"))
}

# ---------------------------------------------------------------------------
# Fine-Gray.
# ---------------------------------------------------------------------------
fg_values <- function(d, k, covs, squares, residuals = FALSE) function(fit) {
    fg <- fit$fg_rows
    cr <- crr(d$x, d$cause, as.matrix(d[, covs]), failcode = k,
              cencode = 0)
    stopifnot(max(abs(coef(fit) - cr$coef)) < 1e-5)
    g <- cif_before(fg)
    # The weighted Kaplan-Meier is the Aalen-Johansen estimate (cuminc),
    # but for the order of a competing failure and a censoring at one
    # time (pbc has one such tie, which moves it by 4e-7).
    ci <- cuminc(d$x, d$cause, cencode = 0)[[paste("1", k)]]
    times <- sort(unique(d$x[d$cause == k]))
    aj <- sapply(times, function(s) {
        j <- which(ci$time < s)
        if (length(j)) ci$est[max(j)] else 0
    })
    stopifnot(max(abs(g(times) - aj)) < 1e-6 || any(d$cause == 0 &
        d$x %in% d$x[d$cause != 0 & d$cause != k]))
    z_cif <- cox.zph(fit, transform = g)
    z_rank <- cox.zph(fit, transform = rank_among(d$x))
    out <- c(list(coef = unname(coef(fit)), crr_coef = unname(cr$coef),
                  rows = rownames(z_cif$table)),
             zph_table(z_cif, "cif"), zph_table(z_rank, "rank"),
             squares_of(fit, squares))
    if (residuals) {
        out$cif_time <- z_cif$time
        out$cif_x <- z_cif$x
        out$cif_y <- unname(z_cif$y)
    }
    out
}
fg_settings <- function(extra = list()) c(
    list(ties = "breslow, as crr",
         weights = paste("G(t-)/G(x_i-) after a competing failure, G the",
                         "censoring Kaplan-Meier (fg_data, crr's ties)"),
         transform = paste("cif: 1 - S(t-), S the weighted Kaplan-Meier of",
                           "the Fine-Gray data (the Aalen-Johansen",
                           "incidence); rank: the rank of t among the",
                           "original rows"),
         squares = paste("update(fit, . ~ . + I(v^2), init = c(coef(fit),",
                         "0), control = coxph.control(iter.max = 0))$score")),
    extra
)
fit_fg <- function(d, k, rhs) {
    # The data are kept by name, so that update() (squares_of) finds them.
    name <- sprintf("fg_rows_%s_%d", deparse(substitute(d)), k)
    assign(name, fg_data(d, k), envir = env)
    fit <- eval(parse(text = sprintf(paste(
        "coxph(Surv(start, stop, status) ~ %s, data = %s, weights = w,",
        "ties = 'breslow', control = ctl)"), rhs, name)), envir = env)
    fit$fg_rows <- get(name, envir = env)
    fit
}
pbc_covs <- c("drug", "age", "log_bili", "female")
for (k in c(1, 2)) {
    record(sprintf("fg_pbc_cause%d", k), "pbc", "survival",
           sprintf(paste("fit_fg(pbc_d, %d, 'drug + age + log_bili +",
                         "female')"), k),
           fg_settings(),
           fg_values(pbc_d, k, pbc_covs, c("age", "log_bili"),
                     residuals = k == 2))
    record(sprintf("fg_competing_cause%d", k), "competing", "survival",
           sprintf("fit_fg(cr_d, %d, 'group + z')", k),
           fg_settings(list(data = "integer times: events tie with censorings")),
           fg_values(cr_d, k, c("group", "z"), "z"))
}
# survival::finegray builds the same data where no censoring ties an event
# (it breaks such ties the other way): on pbc its fit agrees with the
# fg_data one to the convergence tolerance.
pbc_d$ev <- factor(pbc_d$cause, 0:2, c("censored", "transplant", "death"))
check <- coxph(Surv(fgstart, fgstop, fgstatus) ~ drug + age + log_bili +
                   female, weights = fgwt, ties = "breslow", control = ctl,
               data = finegray(Surv(years, ev) ~ ., data = pbc_d,
                               etype = "death"))
stopifnot(max(abs(coef(check) - refs$fg_pbc_cause2$values$coef)) < 1e-5)

out <- list(
    generator = "scripts/reference/reference_r_competing_diagnostics.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_competing_diagnostics.json"))
cat("wrote", file.path(data_dir, "r_competing_diagnostics.json"), "\n")
