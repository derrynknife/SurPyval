# Reference results from R for the stratified and the clustered Fine-Gray
# model: FineGray.fit(..., strata=, strata_censoring=, cluster=).
#
# Writes surpyval/tests/reference/data/crrsc_fine_gray.json: the three
# example data sets of R's crrSC package (bce, cdata and center, the
# rows with a missing value dropped, as crrs does) together with the
# values, each entry with the software, its version and the exact call.
# The data travel with the values (crrSC's own data sets, not fixtures
# of fixtures.json). Generated with R 4.6.2, crrSC 1.1.2, cmprsk 2.2.12
# and survival 3.8.12.
#
#     Rscript scripts/reference/reference_r_crrsc.R
#
# Needs crrSC, cmprsk and jsonlite. Re-running reproduces the file byte
# for byte.
#
# What each value is:
#
# * crrs(..., ctype = 1): the stratified model with the censoring
#   distribution estimated within each stratum (Zhou, Latouche, Rocha and
#   Fine, 2011): coef and the square roots of diag(var), at gtol = 1e-12
#   (the default 1e-6 stops about 1e-7 short of the maximum).
# * crrs(..., ctype = 2): one censoring distribution for every stratum.
#   Its variance (crrSC's C routine crrvvh) sums each stratum's score
#   residuals before squaring them: the strata are its clusters. It
#   indexes the strata by their codes 1, 2, ..., so they are passed as
#   as.integer(factor(.)); with other codes (center$id) it returns a
#   zero variance.
# * crrc(...): the clustered model (Zhou, Fine, Latouche and Labopin,
#   2012): coef and the square roots of diag(var).
# * cmprsk::crr on the same rows, for the unstratified, unclustered fit
#   the others reduce to.
# * cif_<stratum>: each stratum's cumulative incidence at the covariate
#   row `z` and the times `times`, from crr fitted to the stratum's rows
#   with the stratified coefficients held (init = coef, maxiter = 0):
#   predict.crr gives the Breslow baseline of those rows, with their own
#   censoring distribution, which is the stratified model's baseline in
#   that stratum when the censoring is estimated within it (ctype = 1).

suppressPackageStartupMessages({
    library(crrSC)
    library(cmprsk)
    library(jsonlite)
})

out_dir <- file.path("surpyval", "tests", "reference", "data")
if (!dir.exists(out_dir)) {
    stop("Run from the repository root (", out_dir, " not found)")
}

versions <- paste0(
    "crrSC ", as.character(packageVersion("crrSC")),
    ", cmprsk ", as.character(packageVersion("cmprsk"))
)
se <- function(fit) sqrt(diag(fit$var))
tight <- 1e-12

data(bce, package = "crrSC")
data(cdata, package = "crrSC")
data(center, package = "crrSC")
center <- na.omit(center)
bce$stratum <- as.character(cut(bce$nnodes, c(-Inf, 3, 9, Inf),
                                 labels = c("1-3", "4-9", "10+")))

refs <- list()

# bce: three strata of the number of positive nodes ------------------------
bce_cov <- cbind(bce$trt, bce$age, bce$tsize)
bce_code <- as.integer(factor(bce$stratum, levels = c("1-3", "4-9", "10+")))
within <- crrs(bce$time, bce$type, cov1 = bce_cov, strata = bce_code,
               ctype = 1, gtol = tight, maxiter = 50)
common <- crrs(bce$time, bce$type, cov1 = bce_cov, strata = bce_code,
               ctype = 2, gtol = tight, maxiter = 50)
plain <- crr(bce$time, bce$type, bce_cov, gtol = tight, maxiter = 50)
z <- c(1, 70, 30)
times <- c(1, 2, 5, 10)
cif_at <- function(rows, coef) {
    fit <- crr(bce$time[rows], bce$type[rows], bce_cov[rows, ],
               init = coef, maxiter = 0)
    stopifnot(isTRUE(all.equal(as.numeric(fit$coef), coef)))
    p <- predict(fit, cov1 = matrix(z, nrow = 1))
    sapply(times, function(t) {
        k <- which(p[, 1] <= t)
        if (length(k)) p[max(k), 2] else 0
    })
}
cifs <- lapply(c("1-3", "4-9", "10+"), function(s)
    cif_at(which(bce$stratum == s), as.numeric(within$coef)))
names(cifs) <- paste0("cif_", c("1-3", "4-9", "10+"))
refs$bce <- list(
    software = "R crrSC, cmprsk",
    version = versions,
    call = paste(
        "crrs(bce$time, bce$type, cov1 = cbind(trt, age, tsize), strata =",
        "as.integer(factor(stratum)), ctype = 1 or 2, gtol = 1e-12,",
        "maxiter = 50); crr(...) the same without strata; per stratum",
        "predict(crr(rows of the stratum, init = coef, maxiter = 0),",
        "cov1 = z)"),
    settings = list(
        failcode = 1, cencode = 0,
        covariates = c("trt", "age", "tsize"),
        stratum = "nnodes in 1-3, 4-9, 10+",
        z = z, times = times
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

# center: one stratum per centre (many small strata) -------------------------
center_cov <- cbind(center$fm, center$cells)
center_code <- as.integer(factor(center$id))
within <- crrs(center$ftime, center$fstatus, cov1 = center_cov,
               strata = center_code, ctype = 1, gtol = tight, maxiter = 50)
common <- crrs(center$ftime, center$fstatus, cov1 = center_cov,
               strata = center_code, ctype = 2, gtol = tight, maxiter = 50)
refs$center <- list(
    software = "R crrSC",
    version = versions,
    call = paste(
        "crrs(center$ftime, center$fstatus, cov1 = cbind(fm, cells),",
        "strata = as.integer(factor(id)), ctype = 1 or 2, gtol = 1e-12,",
        "maxiter = 50), center = na.omit(center)"),
    settings = list(
        failcode = 1, cencode = 0, covariates = c("fm", "cells"),
        stratum = "id (the centre)"
    ),
    values = list(
        coef_within = as.numeric(within$coef),
        se_within = se(within),
        coef_common = as.numeric(common$coef),
        se_common = se(common)
    )
)

# cdata: pairs of units ------------------------------------------------------
clustered <- crrc(cdata$ftime, cdata$fstatus, cov1 = cdata$z,
                  cluster = cdata$ID, gtol = tight, maxiter = 50)
singletons <- crrc(cdata$ftime, cdata$fstatus, cov1 = cdata$z,
                   gtol = tight, maxiter = 50)
plain <- crr(cdata$ftime, cdata$fstatus, cdata$z, gtol = tight,
             maxiter = 50)
refs$cdata <- list(
    software = "R crrSC, cmprsk",
    version = versions,
    call = paste(
        "crrc(cdata$ftime, cdata$fstatus, cov1 = cdata$z, cluster =",
        "cdata$ID, gtol = 1e-12, maxiter = 50); the same without cluster;",
        "crr(cdata$ftime, cdata$fstatus, cdata$z)"),
    settings = list(failcode = 1, cencode = 0, covariates = "z",
                    cluster = "ID"),
    values = list(
        coef_cluster = as.numeric(clustered$coef),
        se_cluster = se(clustered),
        coef_unclustered = as.numeric(singletons$coef),
        se_unclustered = se(singletons),
        coef_crr = as.numeric(plain$coef),
        se_crr = se(plain)
    )
)

out <- list(
    generator = "scripts/reference/reference_r_crrsc.R",
    R = R.version.string,
    data = list(
        bce = list(time = bce$time, type = bce$type, trt = bce$trt,
                   age = bce$age, tsize = bce$tsize,
                   stratum = bce$stratum),
        cdata = list(ftime = cdata$ftime, fstatus = cdata$fstatus,
                     z = cdata$z, ID = cdata$ID),
        center = list(ftime = center$ftime, fstatus = center$fstatus,
                      id = center$id, fm = center$fm,
                      cells = center$cells)
    ),
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE, null = "null")
writeLines(text, file.path(out_dir, "crrsc_fine_gray.json"))
cat("wrote", file.path(out_dir, "crrsc_fine_gray.json"), "\n")
