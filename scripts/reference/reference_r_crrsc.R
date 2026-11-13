# Reference results from R for the stratified and the clustered Fine-Gray
# model: FineGray.fit(..., strata=, strata_censoring=, cluster=).
#
# Writes surpyval/tests/reference/data/r_crrsc.json, with the software, its
# version, the exact call and the settings of each entry, as the other
# generators do. The fits are to two seeded synthetic fixtures of
# fixtures.json (make_fixtures.fine_gray_strata and fine_gray_cluster).
# Generated with R 4.6.2, crrSC 1.1.2 and cmprsk 2.2.12.
#
#     Rscript scripts/reference/reference_r_crrsc.R
#
# Run make_fixtures.py first. Needs crrSC, cmprsk and jsonlite. Re-running
# reproduces the file byte for byte.
#
# What each value is:
#
# * crrs(..., ctype = 1): the stratified model with the censoring
#   distribution estimated within each stratum (Zhou, Latouche, Rocha and
#   Fine, 2011): coef and the square roots of diag(var), at gtol = 1e-12
#   (the default 1e-6 stops about 1e-7 short of the maximum).
# * crrs(..., ctype = 2): one censoring distribution for every stratum.
#   Its variance (crrSC's C routine crrvvh) sums each stratum's score
#   residuals before squaring them, so the strata are also its clusters.
#   With ctype = 2 the strata must be coded 1, 2, ... for crrs to return a
#   variance, so the script recodes them with as.integer(factor(.)).
# * crrc(...): the clustered model (Zhou, Fine, Latouche and Labopin,
#   2012): coef and the square roots of diag(var).
# * cmprsk::crr on the same rows, the unstratified, unclustered fit.
# * cif_<stratum>: each stratum's cumulative incidence at the covariate
#   row `z` and the times `times`, from crr fitted to the stratum's rows
#   with the stratified coefficients held (init = coef, maxiter = 0):
#   predict.crr gives the Breslow baseline of those rows, with their own
#   censoring distribution, which is the stratified model's baseline in
#   that stratum when the censoring is estimated within it (ctype = 1).
# * The cluster fixture's 100 pairs are also fitted as 100 small strata
#   (ctype 1 and 2), the setting crrs's ctype = 2 is meant for, on z2
#   alone (z1, shared by a pair, is constant within each stratum).

suppressPackageStartupMessages({
    library(crrSC)
    library(cmprsk)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
fixture <- function(name) as.data.frame(fx[[name]]$columns)

versions <- paste0(
    "crrSC ", as.character(packageVersion("crrSC")),
    ", cmprsk ", as.character(packageVersion("cmprsk"))
)
se <- function(fit) sqrt(diag(fit$var))
tight <- 1e-12

refs <- list()

# Three strata with their own baselines and censoring ---------------------
st <- fixture("fine_gray_strata")
cov_s <- cbind(st$z1, st$z2)
code <- as.integer(factor(st$stratum))
within <- crrs(st$x, st$cause, cov1 = cov_s, strata = code, ctype = 1,
               gtol = tight, maxiter = 50)
common <- crrs(st$x, st$cause, cov1 = cov_s, strata = code, ctype = 2,
               gtol = tight, maxiter = 50)
plain <- crr(st$x, st$cause, cov_s, gtol = tight, maxiter = 50)
z <- c(1, 0.5)
times <- c(0.25, 0.5, 1, 2)
cif_at <- function(rows, coef) {
    fit <- crr(st$x[rows], st$cause[rows], cov_s[rows, ], init = coef,
               maxiter = 0)
    stopifnot(isTRUE(all.equal(as.numeric(fit$coef), coef)))
    p <- predict(fit, cov1 = matrix(z, nrow = 1))
    sapply(times, function(t) {
        k <- which(p[, 1] <= t)
        if (length(k)) p[max(k), 2] else 0
    })
}
levels_s <- sort(unique(st$stratum))
cifs <- lapply(levels_s, function(s)
    cif_at(which(st$stratum == s), as.numeric(within$coef)))
names(cifs) <- paste0("cif_", levels_s)
refs$strata <- list(
    fixture = "fine_gray_strata",
    software = "R crrSC, cmprsk",
    version = versions,
    call = paste(
        "crrs(x, cause, cov1 = cbind(z1, z2), strata =",
        "as.integer(factor(stratum)), ctype = 1 or 2, gtol = 1e-12,",
        "maxiter = 50); crr(x, cause, cbind(z1, z2)); per stratum",
        "predict(crr(rows of the stratum, init = coef, maxiter = 0),",
        "cov1 = z)"),
    settings = list(
        failcode = 1, cencode = 0, covariates = c("z1", "z2"),
        strata = "stratum", z = z, times = times
    ),
    values = c(list(
        coef_within = as.numeric(within$coef),
        se_within = se(within),
        coef_common = as.numeric(common$coef),
        se_common = se(common),
        coef_crr = as.numeric(plain$coef),
        se_crr = se(plain)
    ), cifs)
)

# Pairs of dependent units -------------------------------------------------
cl <- fixture("fine_gray_cluster")
cov_c <- cbind(cl$z1, cl$z2)
clustered <- crrc(cl$x, cl$cause, cov1 = cov_c, cluster = cl$cluster,
                  gtol = tight, maxiter = 50)
plain <- crr(cl$x, cl$cause, cov_c, gtol = tight, maxiter = 50)
pair <- as.integer(factor(cl$cluster))
small_within <- crrs(cl$x, cl$cause, cov1 = cl$z2, strata = pair,
                     ctype = 1, gtol = tight, maxiter = 50)
small_common <- crrs(cl$x, cl$cause, cov1 = cl$z2, strata = pair,
                     ctype = 2, gtol = tight, maxiter = 50)
refs$cluster <- list(
    fixture = "fine_gray_cluster",
    software = "R crrSC, cmprsk",
    version = versions,
    call = paste(
        "crrc(x, cause, cov1 = cbind(z1, z2), cluster = cluster,",
        "gtol = 1e-12, maxiter = 50); crr(x, cause, cbind(z1, z2));",
        "crrs(x, cause, cov1 = z2, strata = as.integer(factor(cluster)),",
        "ctype = 1 or 2)"),
    settings = list(
        failcode = 1, cencode = 0, covariates = c("z1", "z2"),
        cluster = "cluster",
        small_strata = "each cluster a stratum, z2 only (z1 is constant within one)"
    ),
    values = list(
        coef_cluster = as.numeric(clustered$coef),
        se_cluster = se(clustered),
        coef_crr = as.numeric(plain$coef),
        se_crr = se(plain),
        coef_small_within = as.numeric(small_within$coef),
        se_small_within = se(small_within),
        coef_small_common = as.numeric(small_common$coef),
        se_small_common = se(small_common)
    )
)

out <- list(
    generator = "scripts/reference/reference_r_crrsc.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(data_dir, "r_crrsc.json"))
cat("wrote", file.path(data_dir, "r_crrsc.json"), "\n")
