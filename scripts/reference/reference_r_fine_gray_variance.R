# Reference results from R for the Fine-Gray model's two covariances:
# Fine and Gray's robust (sandwich) variance, cmprsk::crr's `var`, which
# FineGray's covariance() reports, and the model-based one, crr's
# `invinf`, covariance(robust=False).
#
# Writes surpyval/tests/reference/data/r_fine_gray_variance.json, with the
# software, its version, the exact call and the settings of each entry, as
# reference_r.R does. Generated with R 4.6.2, cmprsk 2.2.12 and survival
# 3.8.12.
#
#     Rscript scripts/reference/reference_r_fine_gray_variance.R
#
# Run make_fixtures.py first. Needs cmprsk and jsonlite. Re-running
# reproduces the file byte for byte.
#
# reference_r.R already stores crr's two variances on the `competing`
# fixture (both causes) and on pbc (death). This adds the cases that
# stress the censoring term of the sandwich:
#
# * pbc with the times rounded up to half years, so that events of both
#   causes and censorings share most times (29 distinct times for 312
#   patients), for each cause (1 transplant, 2 death);
#
# * the `competing` fixture with counts, n = 1 + (row %% 3), row from 1:
#   crr takes no counts, so it is given each row n times, which is what a
#   count means; for each cause;
#
# * the `competing` fixture with every censored row made a competing
#   failure of cause 2 (no censoring): the censoring term is then zero and
#   the sandwich is the score residuals' alone.
#
# crr has no weights argument and SurPyval's Fine-Gray model takes no
# strata or censoring groups, so neither has a reference here.

suppressPackageStartupMessages({
    library(cmprsk)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

refs <- list()
record <- function(id, fixture, call, settings, fit) {
    refs[[id]] <<- list(
        fixture = fixture,
        software = "R cmprsk",
        version = as.character(packageVersion("cmprsk")),
        call = call,
        settings = settings,
        values = list(
            coef = as.numeric(fit$coef),
            var_sandwich = fit$var,
            var_naive = fit$invinf,
            loglik = fit$loglik,
            converged = fit$converged
        )
    )
}

pbc_d <- fixture("pbc")
pbc_d$half <- ceiling(pbc_d$years * 2) / 2
pbc_cov <- cbind(pbc_d$drug, pbc_d$age, pbc_d$log_bili, pbc_d$female)
for (cause in c(1, 2)) {
    record(
        paste0("pbc_half_years_cause", cause), "pbc",
        sprintf(paste("crr(ceiling(pbc_d$years * 2) / 2, pbc_d$cause,",
                      "cbind(pbc_d$drug, pbc_d$age, pbc_d$log_bili,",
                      "pbc_d$female), failcode = %d, cencode = 0)"), cause),
        list(covariates = c("drug", "age", "log_bili", "female"),
             time = "ceiling(years * 2) / 2",
             var_naive = "fit$invinf, the inverse information",
             var_sandwich = "fit$var, Fine and Gray's robust variance"),
        crr(pbc_d$half, pbc_d$cause, pbc_cov, failcode = cause, cencode = 0)
    )
}

cr_d <- fixture("competing")
cr_d$n <- 1 + (seq_len(nrow(cr_d)) %% 3)
rep_d <- cr_d[rep(seq_len(nrow(cr_d)), cr_d$n), ]
for (cause in c(1, 2)) {
    record(
        paste0("competing_counts_cause", cause), "competing",
        sprintf(paste("rep_d <- cr_d[rep(seq_len(nrow(cr_d)), cr_d$n), ];",
                      "crr(rep_d$x, rep_d$cause, cbind(rep_d$group,",
                      "rep_d$z), failcode = %d, cencode = 0)"), cause),
        list(covariates = c("group", "z"),
             counts = "n = 1 + (row %% 3), row from 1; each row n times",
             var_naive = "fit$invinf, the inverse information",
             var_sandwich = "fit$var, Fine and Gray's robust variance"),
        crr(rep_d$x, rep_d$cause, cbind(rep_d$group, rep_d$z),
            failcode = cause, cencode = 0)
    )
}

all_d <- cr_d
all_d$cause[all_d$cause == 0] <- 2
record(
    "competing_no_censoring_cause1", "competing",
    paste("all_d$cause[all_d$cause == 0] <- 2;",
          "crr(all_d$x, all_d$cause, cbind(all_d$group, all_d$z),",
          "failcode = 1, cencode = 0)"),
    list(covariates = c("group", "z"),
         causes = "every censored row (cause 0) made cause 2",
         var_naive = "fit$invinf, the inverse information",
         var_sandwich = "fit$var, Fine and Gray's robust variance"),
    crr(all_d$x, all_d$cause, cbind(all_d$group, all_d$z), failcode = 1,
        cencode = 0)
)

out <- list(
    generator = "scripts/reference/reference_r_fine_gray_variance.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_fine_gray_variance.json"))
cat("wrote", file.path(data_dir, "r_fine_gray_variance.json"), "\n")
