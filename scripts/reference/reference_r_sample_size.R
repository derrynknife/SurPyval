# Reference results from R for the sample-size and power calculations of
# surpyval.test_planning.sample_size: a Cox model's coefficient
# (powerSurvEpi), non-inferiority of a hazard ratio (gsDesign, rpact), a
# difference in restricted mean survival time and a log-rank test with
# cured units (npsurvSS).
#
# Writes surpyval/tests/reference/data/r_sample_size.json in the layout of
# the other reference files: for each calculation the fixture ("no_data":
# a design calculation uses no data, only the arguments in its call), the
# software, its version, the call, the settings and the values. Generated
# with R 4.6.2, powerSurvEpi 0.1.5, gsDesign 3.11.0, rpact 4.4.0 and
# npsurvSS 1.1.0 (the versions are recorded in each entry).
#
#     python scripts/reference/make_fixtures.py
#     Rscript scripts/reference/reference_r_sample_size.R
#
# Run from the repository root. Needs powerSurvEpi, gsDesign, rpact,
# npsurvSS and jsonlite. Re-running reproduces the file byte for byte.
#
# Conventions. npsurvSS is called with sides = 1 and alpha = 0.025: its
# two-sided sample size (sides = 2) steps the units down one at a time
# while its power, which counts both tails, stays above the target, so it
# is not the formula; its one-sided value at 0.025 is SurPyval's
# two-sided value at 0.05 unrounded. npsurvSS's Weibull is
# S(t) = exp(-(scale t)^shape); a hazard ratio h on the latency is a
# scale multiplied by h^(1 / shape). Its loss to follow-up needs a scale;
# 1e-12 stands for none. arm1 is SurPyval's first group, arm0 its second.
#
# npsurvSS's expected events `d` use prob_event(), which for an arm of
# its subclass "lachin" (exponential survival, uniform accrual) is a
# closed form that leaves out surv_cure: with a cured fraction and an
# exponential latency it counts the cured as failing. Its sample size
# and power do not use it. Each entry therefore also stores
# `d_integral`, the same events from npsurvSS's general integral
# (prob_event.arm, the density of an observed event integrated over the
# study), which is the number compared.

suppressPackageStartupMessages({
    library(powerSurvEpi)
    library(gsDesign)
    library(rpact)
    library(npsurvSS)
    library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
    stop("Run from the repository root (fixtures.json not found)")
}

refs <- list()
add <- function(id, package, call, settings, values) {
    refs[[id]] <<- list(
        fixture = "no_data",
        software = paste("R", package),
        version = as.character(packageVersion(package)),
        call = call,
        settings = settings,
        values = values
    )
}
run <- function(call) eval(parse(text = call))

# --- Cox model: powerSurvEpi ------------------------------------------------

cont <- list(
    # Hsieh and Lavori (2000), the example on page 557 (one-sided 5%, so
    # alpha = 0.1 two-sided here, as powerSurvEpi's help says).
    hsieh_lavori = list(power = 0.806, theta = exp(1), sigma2 = 0.3126^2,
                        psi = 0.738, rho2 = 0.1837, alpha = 0.1),
    continuous_a = list(power = 0.8, theta = 1.5, sigma2 = 1, psi = 0.4,
                        rho2 = 0, alpha = 0.05),
    continuous_b = list(power = 0.9, theta = 0.7, sigma2 = 2.5, psi = 0.25,
                        rho2 = 0.35, alpha = 0.01)
)
for (id in names(cont)) {
    a <- cont[[id]]
    call <- sprintf(paste0(
        "ssizeEpiCont.default(power = %.17g, theta = %.17g, sigma2 = ",
        "%.17g, psi = %.17g, rho2 = %.17g, alpha = %.17g)"),
        a$power, a$theta, a$sigma2, a$psi, a$rho2, a$alpha)
    add(paste0("ssize_cont_", id), "powerSurvEpi", call,
        list(test = "two-sided Wald test of one coefficient"),
        c(a, list(n = run(call))))
}

powers <- list(
    a = list(n = 300, theta = 1.4, sigma2 = 0.8, psi = 0.3, rho2 = 0.2,
             alpha = 0.05),
    b = list(n = 120, theta = exp(1), sigma2 = 0.3126^2, psi = 0.738,
             rho2 = 0.1837, alpha = 0.05)
)
for (id in names(powers)) {
    a <- powers[[id]]
    call <- sprintf(paste0(
        "powerEpiCont.default(n = %.17g, theta = %.17g, sigma2 = %.17g, ",
        "psi = %.17g, rho2 = %.17g, alpha = %.17g)"),
        a$n, a$theta, a$sigma2, a$psi, a$rho2, a$alpha)
    add(paste0("power_cont_", id), "powerSurvEpi", call,
        list(test = "two-sided Wald test of one coefficient"),
        c(a, list(power = run(call))))
}

binary <- list(
    # Latouche, Porcher and Chevret (2004), the example at the end of
    # section 5.2, as in powerSurvEpi's help.
    latouche = list(power = 0.8, theta = 2, p = 0.39, rho2 = 0.132^2,
                    alpha = 0.05),
    binary_b = list(power = 0.9, theta = 0.6, p = 0.25, rho2 = 0.4,
                    alpha = 0.05)
)
for (id in names(binary)) {
    a <- binary[[id]]
    call <- sprintf(paste0(
        "numDEpi.default(power = %.17g, theta = %.17g, p = %.17g, ",
        "rho2 = %.17g, alpha = %.17g)"),
        a$power, a$theta, a$p, a$rho2, a$alpha)
    add(paste0("events_binary_", id), "powerSurvEpi", call,
        list(test = "two-sided Wald test of a two-level covariate"),
        c(a, list(events = run(call))))
}

# --- Non-inferiority: gsDesign and rpact ------------------------------------

noninf <- list(
    a = list(hr = 1, hr0 = 1.3, ratio = 1, beta = 0.2),
    b = list(hr = 0.9, hr0 = 1.25, ratio = 2, beta = 0.1),
    c = list(hr = 1.05, hr0 = 1.5, ratio = 0.5, beta = 0.2)
)
for (id in names(noninf)) {
    a <- noninf[[id]]
    call <- sprintf(paste0(
        "nEvents(hr = %.17g, hr0 = %.17g, ratio = %.17g, alpha = 0.025, ",
        "beta = %.17g, sided = 1)"), a$hr, a$hr0, a$ratio, a$beta)
    add(paste0("noninferiority_gsdesign_", id), "gsDesign", call,
        list(test = "one-sided 2.5%: the upper 95% limit below hr0",
             ratio = "experimental per control"),
        c(a, list(events = run(call))))
    call <- sprintf(paste0(
        "getSampleSizeSurvival(hazardRatio = %.17g, thetaH0 = %.17g, ",
        "allocationRatioPlanned = %.17g, alpha = 0.025, beta = %.17g, ",
        "sided = 1, lambda2 = 0.1, accrualTime = c(0, 12), ",
        "followUpTime = 12)$maxNumberOfEvents"),
        a$hr, a$hr0, a$ratio, a$beta)
    add(paste0("noninferiority_rpact_", id), "rpact", call,
        list(test = "one-sided 2.5%: the upper 95% limit below thetaH0",
             note = "the events only; the accrual is a placeholder"),
        c(a, list(events = run(call))))
}

# --- RMST and the log-rank test with cured units: npsurvSS ------------------

arm_call <- function(cure, interval, shape, scale, loss, accrual, follow,
                     size) {
    sprintf(paste0(
        "create_arm(size = %.17g, accr_time = %.17g, surv_cure = %.17g, ",
        "surv_interval = c(%s), surv_shape = %.17g, surv_scale = c(%s), ",
        "loss_scale = %.17g, follow_time = %.17g)"),
        size, accrual, cure, paste(interval, collapse = ", "), shape,
        paste(sprintf("%.17g", scale), collapse = ", "), loss, follow)
}

designs <- list(
    rmst_exponential = list(
        test = "list(test = 'rmst difference', milestone = 5)",
        arm1 = list(cure = 0, interval = c(0, "Inf"), shape = 1, scale = 0.1),
        arm0 = list(cure = 0, interval = c(0, "Inf"), shape = 1,
                    scale = 1 / 6),
        loss = 1e-12, accrual = 2, follow = 4, ratio = 1, n = c(152, 152)),
    rmst_weibull_piecewise = list(
        test = "list(test = 'rmst difference', milestone = 5)",
        arm1 = list(cure = 0, interval = c(0, "Inf"), shape = 1.3,
                    scale = 1 / 8),
        arm0 = list(cure = 0, interval = c(0, 2, "Inf"), shape = 1,
                    scale = c(0.15, 0.25)),
        loss = 0.05, accrual = 2, follow = 4, ratio = 1, n = c(84, 84)),
    rmst_unequal = list(
        test = "list(test = 'rmst difference', milestone = 3)",
        arm1 = list(cure = 0.2, interval = c(0, "Inf"), shape = 0.8,
                    scale = 0.3),
        arm0 = list(cure = 0, interval = c(0, "Inf"), shape = 0.8,
                    scale = 0.4),
        loss = 0.02, accrual = 1.5, follow = 2, ratio = 2, n = c(300, 150)),
    cure_fraction = list(
        test = "list(test = 'weighted logrank')",
        arm1 = list(cure = 0.45, interval = c(0, "Inf"), shape = 1.2,
                    scale = 0.5),
        arm0 = list(cure = 0.30, interval = c(0, "Inf"), shape = 1.2,
                    scale = 0.5),
        loss = 1e-12, accrual = 1, follow = 3, ratio = 1, n = c(247, 247)),
    cure_latency = list(
        test = "list(test = 'weighted logrank')",
        arm1 = list(cure = 0.3, interval = c(0, "Inf"), shape = 1.2,
                    scale = 0.5 * 0.7^(1 / 1.2)),
        arm0 = list(cure = 0.3, interval = c(0, "Inf"), shape = 1.2,
                    scale = 0.5),
        loss = 1e-12, accrual = 1, follow = 3, ratio = 1, n = c(474, 474)),
    cure_both_dropout_unequal = list(
        test = "list(test = 'weighted logrank')",
        arm1 = list(cure = 0.35, interval = c(0, "Inf"), shape = 1,
                    scale = 0.4 * 0.8),
        arm0 = list(cure = 0.25, interval = c(0, "Inf"), shape = 1,
                    scale = 0.4),
        loss = 0.05, accrual = 2, follow = 2, ratio = 2, n = c(400, 200)),
    cure_piecewise_latency = list(
        test = "list(test = 'weighted logrank')",
        arm1 = list(cure = 0.4, interval = c(0, 1, "Inf"), shape = 1,
                    scale = c(0.6, 0.3)),
        arm0 = list(cure = 0.2, interval = c(0, 1, "Inf"), shape = 1,
                    scale = c(0.6, 0.3)),
        loss = 1e-12, accrual = 0.5, follow = 2.5, ratio = 1,
        n = c(150, 150))
)
for (id in names(designs)) {
    d <- designs[[id]]
    a1 <- arm_call(d$arm1$cure, d$arm1$interval, d$arm1$shape,
                   d$arm1$scale, d$loss, d$accrual, d$follow, d$ratio)
    a0 <- arm_call(d$arm0$cure, d$arm0$interval, d$arm0$shape,
                   d$arm0$scale, d$loss, d$accrual, d$follow, 1)
    size <- sprintf(paste0(
        "size_two_arm(%s, %s, %s, power = 0.8, alpha = 0.025, sides = 1)"),
        a0, a1, d$test)
    power <- sprintf(paste0(
        "power_two_arm(%s, %s, %s, alpha = 0.025, sides = 1)"),
        sub("size = [^,]*,", sprintf("size = %d,", d$n[2]), a0),
        sub("size = [^,]*,", sprintf("size = %d,", d$n[1]), a1), d$test)
    out <- run(size)
    arm0 <- run(a0)
    arm1 <- run(a1)
    ns <- asNamespace("npsurvSS")
    d_integral <- unname(out["n0"]) * ns$prob_event.arm(arm0) +
        unname(out["n1"]) * ns$prob_event.arm(arm1)
    add(id, "npsurvSS", paste(size, power, sep = "; "),
        list(test = d$test, sides = "one-sided 2.5% (two-sided 5%)",
             arms = "arm1 is SurPyval's first group, arm0 its second"),
        list(
            arm1 = d$arm1, arm0 = d$arm0, loss_rate = d$loss,
            accrual_time = d$accrual, follow_up = d$follow,
            ratio = d$ratio,
            n1 = unname(out["n1"]), n0 = unname(out["n0"]),
            d = unname(out["d"]), d_integral = d_integral,
            lachin = inherits(arm1, "lachin") || inherits(arm0, "lachin"),
            power_at = d$n, power = run(power)
        ))
}

out <- list(
    generator = "scripts/reference/reference_r_sample_size.R",
    R = R.version.string,
    references = refs
)
text <- toJSON(out, auto_unbox = TRUE, digits = NA, na = "null",
               pretty = TRUE)
writeLines(text, file.path(data_dir, "r_sample_size.json"))
