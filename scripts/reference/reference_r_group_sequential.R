# Reference values for the group-sequential designs
# (surpyval.test_planning.group_sequential_design and
# logrank_group_sequential), stored in
# surpyval/tests/reference/data/group_sequential_r.json for
# surpyval/tests/reference/test_group_sequential.py.
#
#     Rscript scripts/reference/reference_r_group_sequential.R
#
# Needs gsDesign (3.11.0 was used), rpact (4.4.0) and jsonlite, with
# R 4.6.2. Run from the repository root. Re-running reproduces the file.
# Neither package is a SurPyval dependency; install them into a private
# library and put it first in .libPaths() (R_LIBS_USER).
#
# What each entry holds:
#
# * gsDesign::gsDesign(k, test.type, alpha, beta, timing, sfu, sfupar,
#   sfl, sflpar) with n.fix = 1: the efficacy (upper) and futility
#   (lower) boundaries, the inflation factor (n.I at the last look, the
#   maximum information relative to one analysis), the crossing
#   probabilities by look under the null (theta = 0) and the alternative
#   (theta = delta), and the expected information (en) under each.
#   gsDesign's alpha is one-sided (test.type = 2 doubles it: alpha =
#   0.025 is the two-sided 5% design). test.type 1 is one-sided without
#   futility, 2 two-sided symmetric, 3 one-sided with binding beta-spending
#   futility, 4 the same non-binding.
# * gsDesign with sfu = "OF", "Pocock" or "WT" (sfupar = Delta): the
#   classical Wang-Tsiatis boundaries.
# * gsDesign::nEvents (Schoenfeld) and gsDesign(n.fix = <events>): the
#   events of a log-rank design.
# * gsDesign(..., n.I = <events observed>, maxn.IPlan = <planned>): the
#   boundaries re-computed at other information than planned.
# * rpact::getDesignGroupSequential and getDesignCharacteristics: the
#   same designs from the other package, Haybittle-Peto ("HP"), and
#   getSampleSizeSurvival's events (Schoenfeld) for a log-rank design.
# * rpact::getAnalysisResults: the stage-wise p-value, median-unbiased
#   estimate and confidence interval, and repeated confidence intervals,
#   of a log-rank study stopped at an interim look.

suppressPackageStartupMessages({
    library(gsDesign)
    library(rpact)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!dir.exists(data_dir)) stop("Run from the repository root")

versions <- list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    gsDesign = as.character(packageVersion("gsDesign")),
    rpact = as.character(packageVersion("rpact"))
)

gs_entry <- function(call, g) {
    list(
        software = "gsDesign",
        call = call,
        values = list(
            timing = g$timing,
            upper = g$upper$bound,
            lower = if (g$test.type %in% c(3, 4)) g$lower$bound else NULL,
            inflation_factor = g$n.I[g$k],
            information = g$n.I,
            delta = g$delta,
            upper_prob_null = g$upper$prob[, 1],
            upper_prob_alt = g$upper$prob[, 2],
            lower_prob_null = g$lower$prob[, 1],
            lower_prob_alt = g$lower$prob[, 2],
            expected_null = g$en[1],
            expected_alt = g$en[2]
        )
    )
}

refs <- list()

refs$ldof_one_sided <- gs_entry(
    "gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.1, sfu = sfLDOF)",
    gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.1, sfu = sfLDOF)
)
refs$ldpocock_two_sided <- gs_entry(
    "gsDesign(k = 4, test.type = 2, alpha = 0.025, beta = 0.2, sfu = sfLDPocock)",
    gsDesign(k = 4, test.type = 2, alpha = 0.025, beta = 0.2,
             sfu = sfLDPocock)
)
refs$ldof_two_sided <- gs_entry(
    "gsDesign(k = 3, test.type = 2, alpha = 0.025, beta = 0.2, sfu = sfLDOF)",
    gsDesign(k = 3, test.type = 2, alpha = 0.025, beta = 0.2, sfu = sfLDOF)
)
refs$hsd_unequal <- gs_entry(
    paste0("gsDesign(k = 5, test.type = 1, alpha = 0.025, beta = 0.1, ",
           "timing = c(0.2, 0.45, 0.7, 0.85, 1), sfu = sfHSD, sfupar = -4)"),
    gsDesign(k = 5, test.type = 1, alpha = 0.025, beta = 0.1,
             timing = c(0.2, 0.45, 0.7, 0.85, 1), sfu = sfHSD, sfupar = -4)
)
refs$nonbinding_hsd <- gs_entry(
    paste0("gsDesign(k = 3, test.type = 4, alpha = 0.025, beta = 0.1, ",
           "sfu = sfLDOF, sfl = sfHSD, sflpar = -2)"),
    gsDesign(k = 3, test.type = 4, alpha = 0.025, beta = 0.1,
             sfu = sfLDOF, sfl = sfHSD, sflpar = -2)
)
refs$binding_hsd <- gs_entry(
    paste0("gsDesign(k = 3, test.type = 3, alpha = 0.025, beta = 0.1, ",
           "sfu = sfLDOF, sfl = sfHSD, sflpar = -2)"),
    gsDesign(k = 3, test.type = 3, alpha = 0.025, beta = 0.1,
             sfu = sfLDOF, sfl = sfHSD, sflpar = -2)
)
refs$power_nonbinding <- gs_entry(
    paste0("gsDesign(k = 4, test.type = 4, alpha = 0.05, beta = 0.2, ",
           "timing = c(0.3, 0.5, 0.8, 1), sfu = sfPower, sfupar = 3, ",
           "sfl = sfPower, sflpar = 2)"),
    gsDesign(k = 4, test.type = 4, alpha = 0.05, beta = 0.2,
             timing = c(0.3, 0.5, 0.8, 1), sfu = sfPower, sfupar = 3,
             sfl = sfPower, sflpar = 2)
)
refs$ldpocock_binding <- gs_entry(
    paste0("gsDesign(k = 4, test.type = 3, alpha = 0.025, beta = 0.2, ",
           "sfu = sfLDPocock, sfl = sfLDOF)"),
    gsDesign(k = 4, test.type = 3, alpha = 0.025, beta = 0.2,
             sfu = sfLDPocock, sfl = sfLDOF)
)
refs$classical_of <- gs_entry(
    "gsDesign(k = 4, test.type = 2, alpha = 0.025, beta = 0.2, sfu = \"OF\")",
    gsDesign(k = 4, test.type = 2, alpha = 0.025, beta = 0.2, sfu = "OF")
)
refs$classical_pocock <- gs_entry(
    "gsDesign(k = 5, test.type = 2, alpha = 0.025, beta = 0.1, sfu = \"Pocock\")",
    gsDesign(k = 5, test.type = 2, alpha = 0.025, beta = 0.1, sfu = "Pocock")
)
refs$classical_wt <- gs_entry(
    paste0("gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.1, ",
           "sfu = \"WT\", sfupar = 0.25)"),
    gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.1, sfu = "WT",
             sfupar = 0.25)
)

# A log-rank design: hazard ratio 0.7, equal groups, one-sided 2.5%, 90%
# power, three looks with O'Brien-Fleming-type spending.
fixed <- nEvents(hr = 0.7, alpha = 0.025, beta = 0.1, ratio = 1, sided = 1)
lr <- gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.1,
               sfu = sfLDOF, n.fix = fixed)
refs$logrank_events <- list(
    software = "gsDesign",
    call = paste0("gsDesign(k = 3, test.type = 1, alpha = 0.025, ",
                  "beta = 0.1, sfu = sfLDOF, n.fix = nEvents(hr = 0.7, ",
                  "alpha = 0.025, beta = 0.1, ratio = 1, sided = 1))"),
    values = list(
        fixed_events = fixed,
        events = lr$n.I,
        expected_null = lr$en[1],
        expected_alt = lr$en[2]
    )
)
# The same design re-computed at the events the looks came at.
upd <- gsDesign(k = 3, test.type = 1, alpha = 0.025, beta = 0.1,
                sfu = sfLDOF, n.I = c(130, 240, 337),
                maxn.IPlan = lr$n.I[3], delta = lr$delta)
refs$logrank_update <- list(
    software = "gsDesign",
    call = paste0("gsDesign(k = 3, test.type = 1, alpha = 0.025, ",
                  "beta = 0.1, sfu = sfLDOF, n.I = c(130, 240, 337), ",
                  "maxn.IPlan = <the design's n.I[3]>, delta = <its delta>)"),
    values = list(
        max_events = lr$n.I[3],
        upper = upd$upper$bound,
        power = sum(upd$upper$prob[, 2])
    )
)
# The same with a non-binding futility boundary re-computed. The last
# look is past the planned events: gsDesign spends at a last look only
# what its fraction of the plan allows (all of it from 1 on), SurPyval
# all that is left wherever it falls, so the two agree from 1 on. Its
# last futility boundary is its beta spending's; SurPyval's is the
# efficacy boundary (below it, the last look does not reject), so only
# the interim futility boundaries compare.
lrf <- gsDesign(k = 3, test.type = 4, alpha = 0.025, beta = 0.1,
                sfu = sfLDOF, sfl = sfHSD, sflpar = -2, n.fix = fixed)
updf <- gsDesign(k = 3, test.type = 4, alpha = 0.025, beta = 0.1,
                 sfu = sfLDOF, sfl = sfHSD, sflpar = -2,
                 n.I = c(120, 250, 360), maxn.IPlan = lrf$n.I[3],
                 delta = lrf$delta)
refs$logrank_update_futility <- list(
    software = "gsDesign",
    call = paste0("gsDesign(k = 3, test.type = 4, alpha = 0.025, ",
                  "beta = 0.1, sfu = sfLDOF, sfl = sfHSD, sflpar = -2, ",
                  "n.I = c(120, 250, 360), maxn.IPlan = <planned>, ",
                  "delta = <planned>)"),
    values = list(
        max_events = lrf$n.I[3],
        upper = updf$upper$bound,
        lower = updf$lower$bound,
        power = sum(updf$upper$prob[, 2])
    )
)

rp_entry <- function(call, d) {
    ch <- getDesignCharacteristics(d)
    list(
        software = "rpact",
        call = call,
        values = list(
            timing = d$informationRates,
            critical = d$criticalValues,
            alpha_spent = d$alphaSpent,
            # rpact writes "no futility boundary" as -6 at every look
            futility = if (all(is.na(d$futilityBounds)) ||
                           all(d$futilityBounds <= -6)) NULL else
                d$futilityBounds,
            inflation_factor = ch$inflationFactor,
            expected_null = ch$averageSampleNumber0,
            expected_alt = ch$averageSampleNumber1
        )
    )
}
refs$rpact_asof_two_sided <- rp_entry(
    paste0("getDesignGroupSequential(kMax = 3, alpha = 0.05, sided = 2, ",
           "typeOfDesign = \"asOF\", beta = 0.2)"),
    getDesignGroupSequential(kMax = 3, alpha = 0.05, sided = 2,
                             typeOfDesign = "asOF", beta = 0.2)
)
refs$rpact_hp_two_sided <- rp_entry(
    paste0("getDesignGroupSequential(kMax = 4, alpha = 0.05, sided = 2, ",
           "typeOfDesign = \"HP\", constantBoundsHP = 3, beta = 0.2)"),
    getDesignGroupSequential(kMax = 4, alpha = 0.05, sided = 2,
                             typeOfDesign = "HP", constantBoundsHP = 3,
                             beta = 0.2)
)
refs$rpact_wt_one_sided <- rp_entry(
    paste0("getDesignGroupSequential(kMax = 4, alpha = 0.025, sided = 1, ",
           "typeOfDesign = \"WT\", deltaWT = 0.25, beta = 0.1, ",
           "informationRates = c(0.25, 0.5, 0.8, 1))"),
    getDesignGroupSequential(kMax = 4, alpha = 0.025, sided = 1,
                             typeOfDesign = "WT", deltaWT = 0.25,
                             beta = 0.1,
                             informationRates = c(0.25, 0.5, 0.8, 1))
)
refs$rpact_ksd_nonbinding <- rp_entry(
    paste0("getDesignGroupSequential(kMax = 3, alpha = 0.025, sided = 1, ",
           "typeOfDesign = \"asKD\", gammaA = 3, typeBetaSpending = ",
           "\"bsKD\", gammaB = 2, bindingFutility = FALSE, beta = 0.1)"),
    getDesignGroupSequential(kMax = 3, alpha = 0.025, sided = 1,
                             typeOfDesign = "asKD", gammaA = 3,
                             typeBetaSpending = "bsKD", gammaB = 2,
                             bindingFutility = FALSE, beta = 0.1)
)
dsurv <- getDesignGroupSequential(kMax = 3, alpha = 0.025, sided = 1,
                                  typeOfDesign = "asOF", beta = 0.1)
ss <- getSampleSizeSurvival(dsurv, hazardRatio = 0.7, pi2 = 0.3,
                            eventTime = 12, accrualTime = 12,
                            followUpTime = 12)
refs$rpact_logrank_events <- list(
    software = "rpact",
    call = paste0("getSampleSizeSurvival(getDesignGroupSequential(kMax = 3, ",
                  "alpha = 0.025, sided = 1, typeOfDesign = \"asOF\", ",
                  "beta = 0.1), hazardRatio = 0.7, ...)"),
    values = list(
        max_events = ss$maxNumberOfEvents,
        events = as.numeric(ss$cumulativeEventsPerStage),
        expected_null = ss$expectedEventsH0,
        expected_alt = ss$expectedEventsH1
    )
)

# The analysis of a study that stopped at its second look: rpact's
# stage-wise ordering p-value, median-unbiased estimate and confidence
# interval, and the repeated confidence intervals, of the hazard ratio
# (log-rank statistics oriented so that large values favour a hazard
# ratio above 1; 336 events planned, looks at 112 and 224).
an_design <- getDesignGroupSequential(kMax = 3, alpha = 0.025, sided = 1,
                                      typeOfDesign = "asOF")
an_data <- getDataset(overallEvents = c(112, 224),
                      overallLogRanks = c(1.5, 2.6),
                      overallAllocationRatios = c(1, 1))
an <- getAnalysisResults(an_design, an_data, directionUpper = TRUE)
refs$rpact_analysis <- list(
    software = "rpact",
    call = paste0("getAnalysisResults(getDesignGroupSequential(kMax = 3, ",
                  "alpha = 0.025, sided = 1, typeOfDesign = \"asOF\"), ",
                  "getDataset(overallEvents = c(112, 224), overallLogRanks ",
                  "= c(1.5, 2.6), overallAllocationRatios = c(1, 1)), ",
                  "directionUpper = TRUE)"),
    values = list(
        events = c(112, 224),
        z = c(1.5, 2.6),
        max_events = 336,
        p_value = an$finalPValues[2],
        estimate = an$medianUnbiasedEstimates[2],
        lower = an$finalConfidenceIntervalLowerBounds[2],
        upper = an$finalConfidenceIntervalUpperBounds[2],
        repeated_lower = an$repeatedConfidenceIntervalLowerBounds[1:2],
        repeated_upper = an$repeatedConfidenceIntervalUpperBounds[1:2]
    )
)

out <- list(
    generator = "scripts/reference/reference_r_group_sequential.R",
    versions = versions,
    references = refs
)
writeLines(
    toJSON(out, digits = I(15), auto_unbox = TRUE, pretty = TRUE,
           null = "null"),
    file.path(data_dir, "group_sequential_r.json")
)
