# Reference results from R for Cox fits to case-cohort and nested
# case-control samples: CoxPH.fit_case_cohort and
# CoxPH.fit_nested_case_control in SurPyval.
#
# Writes surpyval/tests/reference/data/case_cohort_r.json, read by
# surpyval/tests/univariate/regression/test_cox_case_cohort.py. Generated
# with R 4.6.2 and survival 3.8.12 (both recorded in the file).
#
#     Rscript scripts/reference/reference_r_case_cohort.R
#
# Run from the repository root. Needs only survival (the JSON is written
# by hand, so jsonlite is not needed). Re-running reproduces the file
# byte for byte.
#
# The data: the National Wilms Tumor Study cohort, survival::nwtco (4028
# children; rel = 1 a relapse at edrel days; in.subcohort the random
# subcohort of 668 drawn by Breslow and Chatterjee), stored whole in the
# file. The covariates are those of the example of ?cch: stage (I-IV, as
# three indicators for II, III and IV), histology (an indicator for
# unfavourable) and age in years.
#
# * cch_*: the case-cohort sample of ?cch (the subcohort plus every
#   relapse), fitted by cch() with each method: Prentice, SelfPrentice,
#   LinYing (robust FALSE and TRUE), and I.Borgan and II.Borgan
#   (stratified by instit, with the cohort's sizes per institution).
# * cch_*_entry: the same with a delayed entry, tenter = floor(edrel / 4)
#   days (made up, to check the entry times).
# * barlow: Barlow's (1994) weighting, which cch() does not offer, fitted
#   as Therneau and Li (1999) describe: every subcohort member at weight
#   N / n while not failing; a case enters at weight 1 an instant before
#   its relapse (just under half the smallest gap between relapse days);
#   coxph() with these weights and the robust variance clustered on the
#   child.
# * full_cohort_*: cch() on the whole cohort taken as the subcohort, and
#   coxph() on the whole cohort (Efron and Breslow ties): the case-cohort
#   fits reduce to the ordinary one.
# * ncc_*: a nested case-control sample from the cohort, drawn here
#   (seed 20261127): at each relapse day the children relapsing then (the
#   cases, so a set can have several) and two controls drawn from those
#   still under follow-up after that day; fitted by clogit() with the
#   exact conditional likelihood and with Breslow's approximation.
# * infert: clogit()'s own example, the matched infertility study
#   (datasets::infert), case ~ spontaneous + induced + strata(stratum).

suppressPackageStartupMessages(library(survival))

out_path <- file.path("surpyval", "tests", "reference", "data",
                      "case_cohort_r.json")
if (!dir.exists(dirname(out_path))) {
    stop("Run from the repository root")
}

# --- a small JSON writer (base R) -----------------------------------------
num <- function(v) {
    if (any(!is.finite(v))) stop("non-finite value")
    s <- sprintf("%.17g", v)
    paste0("[", paste(s, collapse = ", "), "]")
}
int <- function(v) paste0("[", paste(as.integer(v), collapse = ", "), "]")
str1 <- function(s) { stopifnot(length(s) == 1); paste0("\"", gsub("\"", "\\\\\"", s), "\"") }
mat <- function(m) {
    rows <- vapply(seq_len(nrow(m)), function(i) num(m[i, ]), "")
    paste0("[", paste(rows, collapse = ", "), "]")
}
obj <- function(fields, indent = "  ") {
    keys <- names(fields)
    body <- paste0(indent, "  ", vapply(keys, str1, ""), ": ",
                   unlist(fields), collapse = ",\n")
    paste0("{\n", body, "\n", indent, "}")
}

meta <- function(call) {
    list(software = str1("R survival"),
         version = str1(as.character(packageVersion("survival"))),
         call = str1(call))
}
fit_entry <- function(fit, call, var = vcov(fit)) {
    obj(c(meta(call),
          list(coef = num(unname(coef(fit))), var = mat(unname(var)))),
        indent = "    ")
}

# --- the cohort and the case-cohort sample (as in ?cch) -------------------
cohort <- nwtco
cohort$age_years <- cohort$age / 12
design <- function(d) {
    cbind(stageII = as.numeric(d$stage == 2),
          stageIII = as.numeric(d$stage == 3),
          stageIV = as.numeric(d$stage == 4),
          histolUH = as.numeric(d$histol == 2),
          age = d$age_years)
}
X_cohort <- design(cohort)
cohort[, colnames(X_cohort)] <- X_cohort
cohort$entry <- floor(cohort$edrel / 4)
sel <- cohort$rel == 1 | cohort$in.subcohort
ccoh <- cohort[sel, ]
ccoh$subcohort <- as.numeric(ccoh$in.subcohort)
N <- nrow(cohort)
stratsizes <- table(cohort$instit)

rhs <- "stageII + stageIII + stageIV + histolUH + age"
f_right <- as.formula(paste("Surv(edrel, rel) ~", rhs))
f_entry <- as.formula(paste("Surv(entry, edrel, rel) ~", rhs))

refs <- list()
add <- function(name, entry) refs[[name]] <<- entry

methods <- c("Prentice", "SelfPrentice", "LinYing")
for (m in methods) {
    for (entry in c(FALSE, TRUE)) {
        f <- if (entry) f_entry else f_right
        fit <- cch(f, data = ccoh, subcoh = ~subcohort, id = ~seqno,
                   cohort.size = N, method = m)
        key <- paste0("cch_", m, if (entry) "_entry" else "")
        add(key, fit_entry(fit, sprintf(
            "cch(%s, subcoh = ~subcohort, id = ~seqno, cohort.size = 4028, method = '%s')",
            paste(deparse(f, width.cutoff = 500L), collapse = " "), m)))
    }
}
fit <- cch(f_right, data = ccoh, subcoh = ~subcohort, id = ~seqno,
           cohort.size = N, method = "LinYing", robust = TRUE)
add("cch_LinYing_robust", fit_entry(fit,
    "cch(Surv(edrel, rel) ~ ..., cohort.size = 4028, method = 'LinYing', robust = TRUE)"))
for (m in c("I.Borgan", "II.Borgan")) {
    for (entry in c(FALSE, TRUE)) {
        f <- if (entry) f_entry else f_right
        fit <- cch(f, data = ccoh, subcoh = ~subcohort, id = ~seqno,
                   stratum = ~instit, cohort.size = stratsizes, method = m)
        key <- paste0("cch_", sub(".", "_", m, fixed = TRUE),
                      if (entry) "_entry" else "")
        add(key, fit_entry(fit, sprintf(
            "cch(%s, subcoh = ~subcohort, id = ~seqno, stratum = ~instit, cohort.size = table(nwtco$instit), method = '%s')",
            paste(deparse(f, width.cutoff = 500L), collapse = " "), m)))
    }
}

# --- Barlow's weighting, as Therneau and Li (1999) ------------------------
barlow_rows <- function(d, n_sub, N, entry) {
    dtime <- sort(unique(d$edrel[d$rel == 1]))
    eps <- min(diff(dtime)) / 2 * 0.999
    alpha <- n_sub / N
    start0 <- if (entry) d$entry else rep(0, nrow(d))
    rows <- list()
    # Subcohort members: weight 1 / alpha up to their exit (a case: up to
    # an instant before its relapse)
    s <- d[d$subcohort == 1, ]
    s_start <- start0[d$subcohort == 1]
    stop_s <- ifelse(s$rel == 1, s$edrel - eps, s$edrel)
    keep <- stop_s > s_start
    rows[[1]] <- data.frame(id = s$seqno[keep], start = s_start[keep],
                            stop = stop_s[keep], event = 0,
                            w = 1 / alpha, s[keep, colnames(X_cohort)])
    # Every case: weight 1 from an instant before its relapse
    k <- d[d$rel == 1, ]
    k_start <- pmax(start0[d$rel == 1], k$edrel - eps)
    rows[[2]] <- data.frame(id = k$seqno, start = k_start, stop = k$edrel,
                            event = 1, w = 1, k[, colnames(X_cohort)])
    do.call(rbind, rows)
}
for (entry in c(FALSE, TRUE)) {
    b <- barlow_rows(ccoh, sum(ccoh$subcohort), N, entry)
    fit <- coxph(as.formula(paste("Surv(start, stop, event) ~", rhs)),
                 data = b, weights = w, cluster = id, timefix = FALSE)
    add(paste0("barlow", if (entry) "_entry" else ""), fit_entry(fit,
        "coxph(Surv(start, stop, event) ~ ..., weights = w, cluster = id): Barlow's rows (Therneau and Li 1999)"))
}

# --- the whole cohort as the subcohort ------------------------------------
full <- cohort
full$subcohort <- 1
for (m in methods) {
    fit <- cch(f_right, data = full, subcoh = ~subcohort, id = ~seqno,
               cohort.size = N, method = m)
    add(paste0("full_cohort_", m), fit_entry(fit, sprintf(
        "cch(Surv(edrel, rel) ~ ..., data = nwtco, subcoh = all, cohort.size = 4028, method = '%s')",
        m)))
}
for (ties in c("efron", "breslow")) {
    fit <- coxph(f_right, data = full, ties = ties)
    add(paste0("full_cohort_coxph_", ties), fit_entry(fit, sprintf(
        "coxph(Surv(edrel, rel) ~ ..., data = nwtco, ties = '%s')", ties)))
}

# --- a nested case-control sample -----------------------------------------
set.seed(20261127)
times <- sort(unique(cohort$edrel[cohort$rel == 1]))
ncc <- list()
for (k in seq_along(times)) {
    t <- times[k]
    cases <- which(cohort$edrel == t & cohort$rel == 1)
    pool <- which(cohort$edrel > t)
    controls <- pool[sample.int(length(pool), 2)]
    ncc[[k]] <- data.frame(set = k, case = c(rep(1, length(cases)), 0, 0),
                           row = c(cases, controls))
}
ncc <- do.call(rbind, ncc)
ncc <- cbind(ncc, X_cohort[ncc$row, ])
f_ncc <- as.formula(paste("case ~", rhs, "+ strata(set)"))
for (method in c("exact", "breslow")) {
    fit <- clogit(f_ncc, data = ncc, method = method)
    add(paste0("ncc_", method), fit_entry(fit, sprintf(
        "clogit(case ~ ... + strata(set), method = '%s')", method)))
}
# The same as a stratified Cox model at one time (exact ties)
chk <- coxph(as.formula(paste("Surv(rep(1, nrow(ncc)), case) ~", rhs,
                              "+ strata(set)")),
             data = ncc, ties = "exact")
stopifnot(max(abs(coef(chk) - coef(clogit(f_ncc, data = ncc)))) < 1e-10)

fit <- clogit(case ~ spontaneous + induced + strata(stratum), data = infert)
add("infert", fit_entry(fit,
    "clogit(case ~ spontaneous + induced + strata(stratum), data = infert)"))

# --- write ----------------------------------------------------------------
data_block <- obj(list(
    seqno = int(cohort$seqno), instit = int(cohort$instit),
    histol = int(cohort$histol), stage = int(cohort$stage),
    rel = int(cohort$rel), edrel = int(cohort$edrel),
    age_months = int(nwtco$age),
    in_subcohort = int(cohort$in.subcohort)), indent = "    ")
ncc_block <- obj(list(set = int(ncc$set), case = int(ncc$case),
                      row = int(ncc$row)), indent = "    ")
infert_block <- obj(list(case = int(infert$case),
                         spontaneous = int(infert$spontaneous),
                         induced = int(infert$induced),
                         stratum = int(infert$stratum)), indent = "    ")
ref_block <- paste0("{\n", paste0("    ", vapply(names(refs), str1, ""),
                                  ": ", unlist(refs), collapse = ",\n"),
                    "\n  }")
json <- paste0(
    "{\n",
    "  \"generator\": \"scripts/reference/reference_r_case_cohort.R\",\n",
    "  \"r_version\": ", str1(paste(R.version$major, R.version$minor,
                                    sep = ".")), ",\n",
    "  \"columns\": [\"stageII\", \"stageIII\", \"stageIV\", \"histolUH\", \"age\"],\n",
    "  \"nwtco\": ", data_block, ",\n",
    "  \"ncc_sample\": ", ncc_block, ",\n",
    "  \"infert\": ", infert_block, ",\n",
    "  \"references\": ", ref_block, "\n",
    "}\n")
writeLines(json, out_path, sep = "")
cat("wrote", out_path, "with", length(refs), "references\n")
