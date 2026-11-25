# Reference values for the joint model of several markers: a marker the
# hazard does not follow, its random effects uncorrelated with the other
# marker's, is a linear mixed model of its own. Printed for
# surpyval/tests/univariate/joint/test_joint_markers.py (LME).
#
#     python scripts/reference/joint_markers_readings.py readings.csv
#     Rscript scripts/reference/reference_r_joint_markers.R readings.csv
#
# The readings are the temperature readings of the test fleet
# (surpyval/tests/univariate/joint/_multi_fleet.simulate(n=80, seed=7)):
# a straight line in age with a level of each item's own (a random
# intercept) and a common rate, fitted by maximum likelihood (not REML,
# which the joint likelihood is not). Written with R 4.6.2 and nlme 3.1.
suppressPackageStartupMessages(library(nlme))
args <- commandArgs(trailingOnly = TRUE)
d <- read.csv(args[1])
fit <- lme(y ~ t, random = ~ 1 | id, data = d, method = "ML",
           control = lmeControl(tolerance = 1e-12, msTol = 1e-12,
                                niterEM = 200))
cat("R", R.version$major, ".", R.version$minor, " nlme ",
    as.character(packageVersion("nlme")), "\n", sep = "")
print(fixef(fit), digits = 15)
print(c(sd_a = sqrt(getVarCov(fit)[1, 1]), sigma = fit$sigma), digits = 15)
print(logLik(fit), digits = 12)
