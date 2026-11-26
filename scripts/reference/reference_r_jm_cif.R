# Reference results from R for the chance of each failure mode of a
# running item under a joint model, surpyval's
# JointModel.predict(...).cif(x, mode).
#
# Writes surpyval/tests/reference/data/r_jm_cif.json, which
# surpyval/tests/univariate/joint/test_joint_cif.py::test_against_r_jm
# reads when it is present (the test is skipped until then): the PBC data
# in long form, and for a few patients still alive at an age, the
# cumulative incidence of death and of transplant at later ages.
#
#     Rscript scripts/reference/reference_r_jm_cif.R
#
# Needs JM and jsonlite. NOT RUN YET: written against the JM 1.5
# documentation (jointModel(..., CompRisk = TRUE) on the stacked data of
# crLong, and survfitJM); check the calls on the first run. JM's baseline
# hazard here is piecewise constant and SurPyval's Weibull, so the
# comparison is loose (0.03 in the incidence): it checks the meaning of
# the incidence (conditioning on the readings and on survival to the age
# now), not the digits.

suppressPackageStartupMessages({
    library(JM)
    library(jsonlite)
})

# the marker: log serum bilirubin, a straight line in years per patient
lme_fit <- lme(log(serBilir) ~ year, random = ~ year | id, data = pbc2)

# the two modes, stacked (one row per patient and mode)
pbc2.idCR <- crLong(pbc2.id, statusVar = "status",
                    censLevel = "alive", nameStrata = "CR")
cox_fit <- coxph(Surv(years, status2) ~ strata(CR),
                 data = pbc2.idCR, x = TRUE)
joint_fit <- jointModel(lme_fit, cox_fit, timeVar = "year",
                        method = "spline-PH-aGH", CompRisk = TRUE,
                        interFact = list(value = ~ CR, data = pbc2.idCR))

# patients still alive at an age, their readings to it
cases <- list(list(id = 2, given = 5), list(id = 25, given = 3),
              list(id = 93, given = 4))
predictions <- lapply(cases, function(case) {
    rows <- pbc2[pbc2$id == case$id & pbc2$year <= case$given, ]
    x <- case$given + c(1, 2, 4)
    fit <- survfitJM(joint_fit, newdata = rows, idVar = "id",
                     last.time = case$given, survTimes = x, M = 1)
    # JM reports each mode's cumulative incidence per stratum
    cif <- lapply(fit$summaries[[1]], function(s) s[, "Mean"])
    names(cif) <- c("dead", "transplanted")
    list(id = case$id, given = case$given, x = x, cif = cif)
})

long <- pbc2[, c("id", "years", "status", "year", "serBilir")]
long$status <- as.character(long$status)
out <- list(
    software = paste("R", getRversion(), "JM", packageVersion("JM")),
    data = long,
    predictions = predictions
)
write_json(out, "surpyval/tests/reference/data/r_jm_cif.json",
           auto_unbox = TRUE, digits = NA, pretty = TRUE)
