# Reference values for the Aalen-Johansen cumulative incidence with delayed
# entry (left truncation; surpyval.CompetingRisks.fit(..., tl=)), printed
# for surpyval/tests/reference/test_aalen_johansen_entry.py.
#
#     Rscript scripts/reference/reference_r_aalen_johansen_entry.R
#
# Written with R 4.6.2 and survival 3.8.12 (the versions are printed).
#
# survfit(Surv(entry, exit, factor(e)) ~ 1, id = ) fits the multi-state
# (competing risks) Aalen-Johansen estimate of counting-process data: a
# unit is at risk in (entry, exit]. Its std.err is the
# infinitesimal-jackknife (robust) one, which SurPyval gives for a fit
# with weights (w = 1 for the unweighted data). timefix = FALSE keeps the
# times as they are: by default survfit merges times that differ only by
# round-off, and age + follow-up has such near-ties (SurPyval takes the
# numbers as given).
#
# "pbc_age": the Mayo PBC trial (surpyval/datasets/pbc2.csv, one row per
# patient) on the age time scale: each patient enters at their age at
# enrolment and leaves at that age plus their follow-up; the causes are
# death and transplant (alive is censored). "pbc_age_weighted": the same
# with the weights 1 + (id mod 3) / 2. "ties": a small set with tied
# failures, a failure tied with a censoring, and entries exactly at
# failure times (not at risk for them, under the (entry, exit] rule).
suppressPackageStartupMessages(library(survival))
cat("R", R.version$major, ".", R.version$minor, " survival ",
    as.character(packageVersion("survival")), "\n", sep = "")

pbc <- read.csv("surpyval/datasets/pbc2.csv")
pbc <- pbc[!duplicated(pbc$id), ]
entry <- pbc$age
exit <- pbc$age + pbc$years
e <- match(pbc$status, c("alive", "dead", "transplanted")) - 1

show <- function(label, entry, exit, e, w, times) {
  # One row per unit: id = each row (survfit needs it for (start, stop]
  # multi-state data).
  fit <- survfit(Surv(entry, exit, factor(e, 0:2)) ~ 1, weights = w,
                 id = seq_along(e), timefix = FALSE)
  s <- summary(fit, times = times, extend = TRUE)
  for (k in 1:2) {
    cat(sprintf("%s cif%d = [%s]\n", label, k,
                paste(sprintf("%.15g", s$pstate[, k + 1]), collapse = ", ")))
    cat(sprintf("%s se%d = [%s]\n", label, k,
                paste(sprintf("%.15g", s$std.err[, k + 1]), collapse = ", ")))
  }
}
pbc_times <- c(45, 50, 55, 60, 65, 70, 75, 80)
show("pbc_age", entry, exit, e, rep(1, length(e)), pbc_times)
show("pbc_age_weighted", entry, exit, e, 1 + (pbc$id %% 3) / 2, pbc_times)

t_entry <- c(0, 0, 0, 0, 0, 2, 2, 3, 3, 3, 5, 5, 6, 1, 4)
t_exit <- c(2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 3, 9)
t_e <- c(1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 1, 0)
show("ties", t_entry, t_exit, t_e, rep(1, length(t_e)), c(2, 3, 5, 6, 8, 10))
