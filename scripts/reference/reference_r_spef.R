# Reference results from R's spef for panel-count data
# (surpyval.recurrent.PanelCountMCF and PanelCountRegression).
#
# Writes surpyval/tests/reference/data/r_spef.json in the layout of the
# other reference files: for each entry the fixture, the software, its
# version, the call, the settings and the values. Generated with R 4.6.2
# and spef 1.0.9 (archived on CRAN; installed from the CRAN archive
# tarball):
#
#     python scripts/reference/make_fixtures.py
#     Rscript scripts/reference/reference_r_spef.R
#
# Run from the repository root, after make_fixtures.py. Needs spef and
# jsonlite. Re-running reproduces the file byte for byte.
#
# The data are the shared fixture "panel_counts": 85 simulated units,
# each seen at its own visits, with the number of new events at each and
# a treatment indicator and two baseline covariates (see
# make_fixtures.panel_counts).
#
# The entries:
#
# * mpl: panelReg(PanelSurv(id, time, count) ~ treatment + size + num,
#   method = "MPL"), the maximum pseudo-likelihood estimator of Wellner
#   and Zhang (2007), run with absTol = relTol = 1e-12: at the default
#   tolerances (1e-6) spef's iteration and SurPyval's search stop at
#   slightly different points, so the two are compared at the tighter
#   setting. The coefficients and the baseline mean function at every
#   visit time.
# * mls: the same with method = "MLs", the sieve maximum likelihood
#   estimator of Lu, Zhang and Huang (2007): the likelihood SurPyval's
#   how="likelihood" maximises, with the baseline a monotone cubic
#   spline instead of a step function, so close to but not equal to the
#   step-function maximum. The coefficients.
# * mpl_nonparametric: the maximum pseudo-likelihood mean function with
#   no covariates (Sun and Kalbfleisch 1995), from spef's own MPL code
#   (its max-min formula for the isotonic regression, iso() in
#   doPanelFit.MPL) at beta = 0, at every visit time.

suppressPackageStartupMessages({
    library(spef)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_spef.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
bp <- as.data.frame(fx$panel_counts$columns)
version <- as.character(packageVersion("spef"))
formula <- PanelSurv(id, time, count) ~ treatment + size + num
grid <- sort(unique(bp$time))

entry <- function(call, settings, values) {
    list(
        fixture = "panel_counts",
        software = "R spef",
        version = version,
        call = call,
        settings = settings,
        values = values
    )
}

tight <- list(absTol = 1e-12, relTol = 1e-12, maxIter = 10000)
mpl <- panelReg(formula, data = bp, method = "MPL", se = "NULL",
                control = tight)
stopifnot(mpl$convergence == 0)
mls <- panelReg(formula, data = bp, method = "MLs", se = "NULL")
stopifnot(mls$convergence == 0)

# spef's own isotonic regression (doPanelFit.MPL's iso) at beta = 0: the
# cumulative counts of each patient on the grid of visit times
ids <- sort(unique(bp$id))
K <- length(grid)
N <- matrix(0, length(ids), K)
obs <- matrix(FALSE, length(ids), K)
for (k in seq_along(ids)) {
    rows <- bp[bp$id == ids[k], ]
    rows <- rows[order(rows$time), ]
    cols <- match(rows$time, grid)
    N[k, cols] <- cumsum(rows$count)
    obs[k, cols] <- TRUE
}
cumSumMatrix <- function(x) {
    m <- outer(rep(1, length(x)), x)
    m[row(m) > col(m)] <- 0
    t(apply(m, 1, cumsum))
}
m <- cumSumMatrix(colSums(N)) / cumSumMatrix(colSums(obs))
m[row(m) > col(m)] <- 0
for (i in 1:(K - 1)) m[, K - i] <- pmin(m[, K - i], m[, K - i + 1])
iso0 <- apply(m, 2, max)

references <- list(
    mpl = entry(
        paste(
            "panelReg(PanelSurv(id, time, count) ~ treatment + size + num,",
            "panel_counts, method = 'MPL', se = 'NULL', control = list(absTol =",
            "1e-12, relTol = 1e-12, maxIter = 10000))"
        ),
        list(
            estimator = "maximum pseudo-likelihood (Wellner and Zhang 2007)",
            baseline = "mpl$baseline at every visit time (step function)"
        ),
        list(
            coef = unname(mpl$beta),
            times = grid,
            baseline = mpl$baseline(grid)
        )
    ),
    mls = entry(
        paste(
            "panelReg(PanelSurv(id, time, count) ~ treatment + size + num,",
            "panel_counts, method = 'MLs', se = 'NULL')"
        ),
        list(
            estimator = paste(
                "sieve maximum likelihood, monotone cubic spline baseline",
                "(Lu, Zhang and Huang 2007); default control"
            )
        ),
        list(coef = unname(mls$beta))
    ),
    mpl_nonparametric = entry(
        "spef doPanelFit.MPL iso() at beta = 0 on panel_counts",
        list(
            estimator = paste(
                "maximum pseudo-likelihood mean function with no",
                "covariates (Sun and Kalbfleisch 1995), spef's max-min",
                "formula"
            )
        ),
        list(times = grid, mcf = iso0)
    )
)

out <- list(
    generator = "scripts/reference/reference_r_spef.R",
    R = R.version.string,
    references = references
)
writeLines(
    toJSON(out, digits = NA, auto_unbox = TRUE, pretty = TRUE), out_file
)
