# Reference results from R for the extensions of the log-rank test:
# the max-combo test (maxcombo_logrank), the test for trend
# (trend_logrank), the permutation test (permutation_logrank) and the
# maximally selected log-rank statistic (maxstat_logrank).
#
# Writes surpyval/tests/reference/data/r_logrank_extensions.json, with the
# software, its version, the call and the values of each entry, as the
# other generators do. Every entry is computed on a fixture already in
# fixtures.json (survival's lung, veteran and aml), so make_fixtures.py
# needs no change.
#
#     Rscript scripts/reference/reference_r_logrank_extensions.R
#
# (with the packages in a private library, set R_LIBS to it first).
#
# Needs survival, jsonlite, nph, coin and maxstat (with mvtnorm). Run with
# R 4.6.2, survival 3.8.12, nph 2.1, coin 1.4.6, maxstat 0.7.26 and
# mvtnorm 1.4.2. Re-running reproduces the file byte for byte (the Monte
# Carlo entries are seeded).
#
# What each value is:
#
# * Max-combo: nph::logrank.maxtest, the components' z, their correlation
#   and the multivariate normal p-value (mvtnorm's GenzBretz at
#   abseps = 1e-8, maxpts = 1e7, seeded: its error is that of the
#   integrator, about 1e-5 absolute).
# * Trend: survdiff's obs - exp and var, with the trend statistic
#   (s'(O - E))^2 / (s' V s) computed by hand, as survminer's
#   surv_pvalue(test.for.trend = TRUE) does; and the score test of
#   coxph(ties = "breslow") with the score as a numeric covariate.
# * Permutation: coin::logrank_test, its statistic (the standardised z
#   for two groups, the quadratic form for more), the asymptotic p-value
#   and the exact (two groups) or approximate (nresample = 1e5, seeded)
#   p-value. coin's scores are minus SurPyval's, so a two-group z has the
#   opposite sign. The stratified entry forms the scores within each
#   stratum (see the comment there).
# * Maximally selected statistic: maxstat.test on the scores of
#   coin::logrank_trafo (smethod = "Data"), whose ties are SurPyval's, with
#   each pmethod; and maxstat's own smethod = "LogRank", whose scores
#   (exactRankTests::cscores) treat tied times as ordered, recorded to
#   show the difference.

suppressPackageStartupMessages({
    library(survival)
    library(jsonlite)
    library(nph)
    library(coin)
    library(maxstat)
    library(mvtnorm)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

lung_d <- fixture("lung")
lung_d$status <- 1 - lung_d$c
vet <- fixture("veteran")
aml_d <- fixture("aml")
stopifnot(
    isTRUE(all.equal(lung_d$time, survival::lung$time)),
    isTRUE(all.equal(lung_d$status, survival::lung$status - 1)),
    isTRUE(all.equal(vet$time, survival::veteran$time)),
    isTRUE(all.equal(aml_d$time, survival::aml$time))
)
versions <- list(
    survival = as.character(packageVersion("survival")),
    nph = as.character(packageVersion("nph")),
    coin = as.character(packageVersion("coin")),
    maxstat = as.character(packageVersion("maxstat")),
    mvtnorm = as.character(packageVersion("mvtnorm"))
)

refs <- list()
add <- function(id, entry) refs[[id]] <<- entry

# --- max-combo ------------------------------------------------------------
maxcombo <- function(id, fixture_name, time, event, group, alternative,
                     rho, gamma) {
    set.seed(1)
    res <- logrank.maxtest(time, event, group, alternative = alternative,
                           rho = rho, gamma = gamma,
                           algorithm = GenzBretz(maxpts = 1e7,
                                                 abseps = 1e-8, releps = 0))
    add(id, list(
        fixture = fixture_name,
        software = "R nph",
        version = versions$nph,
        call = sprintf(paste0(
            "logrank.maxtest(time, event, group, alternative = '%s', ",
            "rho = c(%s), gamma = c(%s))"), alternative,
            paste(rho, collapse = ", "), paste(gamma, collapse = ", ")),
        settings = list(alternative = alternative, rho = rho, gamma = gamma,
                        integrator = "GenzBretz(maxpts = 1e7, abseps = 1e-8)",
                        mvtnorm = versions$mvtnorm),
        values = list(z = res$tests$z, p_each = res$tests$p,
                      correlation = unname(res$korr), p_value = res$pmult,
                      p_bonferroni = res$p.Bonf)
    ))
}
for (alt in c("two.sided", "greater", "less")) {
    maxcombo(paste0("maxcombo_lung_sex_", sub("\\.", "_", alt)), "lung",
             lung_d$time, lung_d$status, lung_d$sex, alt,
             c(0, 0, 1, 1), c(0, 1, 0, 1))
}
maxcombo("maxcombo_veteran_trt_nph_default", "veteran", vet$time,
         vet$status, vet$trt, "two.sided", c(0, 0, 1), c(0, 1, 0))

# --- trend ----------------------------------------------------------------
trend <- function(id, fixture_name, data, formula, scores, rho = 0,
                  strata_note = NULL) {
    sd <- survdiff(formula, data = data, rho = rho)
    oe <- if (is.matrix(sd$obs)) rowSums(sd$obs) - rowSums(sd$exp) else
        sd$obs - sd$exp
    U <- sum(scores * oe)
    V <- as.numeric(t(scores) %*% sd$var %*% scores)
    z <- U / sqrt(V)
    add(id, list(
        fixture = fixture_name,
        software = "R survival",
        version = versions$survival,
        call = paste0("survdiff(", deparse(formula), ", rho = ", rho,
                      "); s'(obs - exp) / sqrt(s' var s)"),
        settings = list(scores = scores, rho = rho,
                        strata = if (is.null(strata_note)) "none" else
                            strata_note),
        values = list(obs_minus_exp = unname(oe), z = z, chisq = z^2,
                      p_value = pchisq(z^2, 1, lower.tail = FALSE))
    ))
}
lung_e <- lung_d[!is.na(lung_d$ph_ecog) & lung_d$ph_ecog < 3, ]
trend("trend_lung_ecog", "lung", lung_e,
      Surv(time, status) ~ ph_ecog, c(0, 1, 2))
trend("trend_lung_ecog_rho1", "lung", lung_e,
      Surv(time, status) ~ ph_ecog, c(0, 1, 2), rho = 1)
trend("trend_lung_ecog_strata_sex", "lung", lung_e,
      Surv(time, status) ~ ph_ecog + strata(sex), c(0, 1, 2),
      strata_note = "sex")
trend("trend_veteran_celltype_unequal", "veteran", vet,
      Surv(time, status) ~ celltype, c(1, 2, 4, 8))
cox <- coxph(Surv(time, status) ~ ph_ecog, data = lung_e, ties = "breslow",
             iter.max = 0)
add("trend_lung_ecog_cox_score", list(
    fixture = "lung",
    software = "R survival",
    version = versions$survival,
    call = paste0("coxph(Surv(time, status) ~ ph_ecog, ties = 'breslow', ",
                  "iter.max = 0); summary(fit)$sctest"),
    settings = list(scores = c(0, 1, 2), ties = "breslow"),
    values = list(score_test = unname(summary(cox)$sctest["test"]))
))

# --- permutation ----------------------------------------------------------
aml_d$group <- factor(aml_d$maintained)
perm2 <- function(id, type = "logrank", rho = NULL, gamma = NULL) {
    lt <- if (type == "logrank") {
        logrank_test(Surv(time, status) ~ group, data = aml_d,
                     distribution = "exact")
    } else {
        logrank_test(Surv(time, status) ~ group, data = aml_d,
                     distribution = "exact", type = type, rho = rho,
                     gamma = gamma)
    }
    asym <- logrank_test(Surv(time, status) ~ group, data = aml_d,
                         type = type, rho = rho, gamma = gamma)
    add(id, list(
        fixture = "aml",
        software = "R coin",
        version = versions$coin,
        call = sprintf(paste0("logrank_test(Surv(time, status) ~ group, ",
                              "distribution = 'exact', type = '%s'%s)"),
                       type, if (is.null(rho)) "" else
                           sprintf(", rho = %s, gamma = %s", rho, gamma)),
        settings = list(type = type, rho = if (is.null(rho)) 0 else rho,
                        gamma = if (is.null(gamma)) 0 else gamma,
                        group_order = "maintained = 0 first (coin's level 1)",
                        ties = "mid-ranks"),
        values = list(z = unname(statistic(lt)),
                      p_value = unname(pvalue(lt)),
                      p_value_asymptotic = unname(pvalue(asym)))
    ))
}
perm2("permutation_aml_exact")
perm2("permutation_aml_exact_fh10", "Fleming-Harrington", 1, 0)

lung_e$ecog <- factor(lung_e$ph_ecog)
set.seed(20261113)
lk <- logrank_test(Surv(time, status) ~ ecog, data = lung_e,
                   distribution = approximate(nresample = 1e5))
lk_asym <- logrank_test(Surv(time, status) ~ ecog, data = lung_e)
add("permutation_lung_ecog_mc", list(
    fixture = "lung",
    software = "R coin",
    version = versions$coin,
    call = paste0("logrank_test(Surv(time, status) ~ factor(ph_ecog), ",
                  "distribution = approximate(nresample = 1e5))"),
    settings = list(nresample = 1e5, seed = 20261113,
                    subset = "ph_ecog < 3"),
    values = list(statistic = unname(statistic(lk)),
                  p_value = unname(pvalue(lk)),
                  p_value_asymptotic = unname(pvalue(lk_asym)))
))
vet$trtf <- factor(vet$trt)
vet$cellf <- factor(vet$celltype)
# The stratified test: the log-rank scores formed within each stratum and
# the labels permuted within it. coin's logrank_test(y ~ x | block) forms
# the scores from the pooled data (only the permutation is within the
# blocks), which is not the stratified log-rank test, so the scores are
# given per block through trafo(block = ).
block_scores <- function(data) {
    trafo(data, surv_trafo = function(y) logrank_trafo(y), block = vet$cellf)
}
set.seed(20261114)
vs <- independence_test(Surv(time, status) ~ trtf | cellf, data = vet,
                        ytrafo = block_scores,
                        distribution = approximate(nresample = 1e5))
vs_asym <- independence_test(Surv(time, status) ~ trtf | cellf, data = vet,
                             ytrafo = block_scores)
pooled <- logrank_test(Surv(time, status) ~ trtf | cellf, data = vet)
add("permutation_veteran_trt_strata_mc", list(
    fixture = "veteran",
    software = "R coin",
    version = versions$coin,
    call = paste0("independence_test(Surv(time, status) ~ factor(trt) | ",
                  "factor(celltype), ytrafo = function(d) trafo(d, ",
                  "surv_trafo = logrank_trafo, block = celltype), ",
                  "distribution = approximate(nresample = 1e5))"),
    settings = list(nresample = 1e5, seed = 20261114,
                    scores = "within each stratum"),
    values = list(z = unname(statistic(vs)),
                  p_value = unname(pvalue(vs)),
                  p_value_asymptotic = unname(pvalue(vs_asym)),
                  survdiff_strata_chisq = survdiff(
                      Surv(time, status) ~ trtf + strata(cellf),
                      data = vet)$chisq,
                  coin_pooled_scores_z = unname(statistic(pooled)))
))

# --- maximally selected statistic -----------------------------------------
lung_d$scores <- logrank_trafo(Surv(lung_d$time, lung_d$status))
vet$scores <- logrank_trafo(Surv(vet$time, vet$status))
mstat <- function(id, fixture_name, data, covariate, pmethod) {
    f <- as.formula(paste("scores ~", covariate))
    set.seed(20261115)
    extra <- switch(pmethod,
                    exactGauss = list(abseps = 1e-8, maxpts = 1e7),
                    condMC = list(B = 1e5), list())
    res <- do.call(maxstat.test, c(list(f, data = data, smethod = "Data",
                                        pmethod = pmethod, minprop = 0.1,
                                        maxprop = 0.9), extra))
    own <- maxstat.test(as.formula(paste("Surv(time, status) ~", covariate)),
                        data = data, smethod = "LogRank", pmethod = "none")
    add(id, list(
        fixture = fixture_name,
        software = "R maxstat",
        version = versions$maxstat,
        call = sprintf(paste0("maxstat.test(logrank_trafo(Surv(time, ",
                              "status)) ~ %s, smethod = 'Data', pmethod = ",
                              "'%s', minprop = 0.1, maxprop = 0.9)"),
                       covariate, pmethod),
        settings = list(pmethod = pmethod, minprop = 0.1, maxprop = 0.9,
                        scores = "coin::logrank_trafo (mid-ranks)",
                        coin = versions$coin,
                        B = if (pmethod == "condMC") 1e5 else 0),
        values = list(statistic = unname(res$statistic),
                      cutpoint = unname(res$estimate),
                      p_value = unname(res$p.value),
                      cuts = unname(res$cuts),
                      stats = unname(res$stats),
                      cscores_statistic = unname(own$statistic),
                      cscores_cutpoint = unname(own$estimate))
    ))
}
for (pm in c("exactGauss", "Lau94", "Lau92", "condMC")) {
    mstat(paste0("maxstat_lung_age_", pm), "lung", lung_d, "age", pm)
}
mstat("maxstat_veteran_karno_exactGauss", "veteran", vet, "karno",
      "exactGauss")

out <- list(
    generator = "scripts/reference/reference_r_logrank_extensions.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_logrank_extensions.json"))
cat("wrote", file.path(data_dir, "r_logrank_extensions.json"), "\n")
