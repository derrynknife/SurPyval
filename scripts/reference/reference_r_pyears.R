# Reference results from R for surpyval.person_years (the exposure, or
# person-years, table), for surpyval/tests/reference/test_person_years.py.
#
# Writes surpyval/tests/reference/data/r_pyears.json, with the software,
# its version, the call and the settings of each entry. Run from the
# repository root, after
#
#     python scripts/reference/make_fixtures.py
#
# with jsonlite in the library path:
#
#     R_LIBS_USER=<library with jsonlite> \
#         Rscript scripts/reference/reference_r_pyears.R
#
# Written with R 4.6.2, survival 3.8.12 and jsonlite. Re-running reproduces
# the file byte for byte (nothing is random).
#
# The fixture is "mgus2_pyears" (survival::mgus2, one row per patient,
# with a late entry tl for every fourth id). Every time is in months, so
# the sums are of whole numbers: the ages and calendar years at diagnosis
# are multiplied by 12.
#
# What each reference is:
#
# * pyears(Surv(futime, death) ~ tcut(12 age, ...) + tcut(12 dxyr, ...)
#   + sex, ratetable = rt, scale = 1, data.frame = TRUE): the months,
#   patients and deaths in each cell of attained age by calendar period
#   by sex, and the expected deaths under a rate table "rt" (deaths per
#   month) whose cut points are the bands', so the expected count of a
#   cell is its rate times its months.
# * pyears(Surv(tl, futime, death) ~ tcut(12 age, c(50, 65, 80, 120) 12)
#   + tcut(-tl, c(0, 24, 60, 500)) + sex, scale = 1): late entry, with
#   time since entry as the second time scale (its value at time 0 is
#   -tl), and ages below 50 off the table (R refuses an infinite edge).
# * poisson.test: the exact (Garwood) interval of a rate, of the
#   observed to expected ratio, and the exact conditional interval and
#   test of the ratio of two rates (the men's death rate over the
#   women's).
suppressPackageStartupMessages({
  library(survival)
  library(jsonlite)
})

data_dir <- file.path("surpyval", "tests", "reference", "data")
if (!file.exists(file.path(data_dir, "fixtures.json"))) {
  stop("Run from the repository root (fixtures.json not found)")
}
out_file <- file.path(data_dir, "r_pyears.json")
fx <- fromJSON(file.path(data_dir, "fixtures.json"))
d <- as.data.frame(fx[["mgus2_pyears"]]$columns)
d$sex <- factor(ifelse(d$male == 1, "M", "F"), levels = c("F", "M"))
v_survival <- as.character(packageVersion("survival"))

references <- list()
add <- function(id, call, settings, values) {
  references[[id]] <<- list(
    fixture = "mgus2_pyears", software = "R survival",
    version = v_survival, call = call, settings = settings,
    values = values
  )
}

# -- attained age by calendar period by sex, with a rate table ------------
ages <- c(20, 60, 70, 80, 110)
years <- c(1960, 1980, 1990, 2010)
rt <- array(0, dim = c(4, 3, 2), dimnames = list(
  age = as.character(ages[1:4]), year = as.character(years[1:3]),
  sex = c("F", "M")))
for (i in 1:4) for (j in 1:3) for (k in 1:2)
  rt[i, j, k] <- 0.001 * i * i + 0.0003 * j + 0.002 * (k - 1)
attr(rt, "dimid") <- c("age", "year", "sex")
attr(rt, "cutpoints") <- list(12 * ages[1:4], 12 * years[1:3], NULL)
attr(rt, "type") <- c(2, 2, 1)
class(rt) <- "ratetable"
d$age_m <- 12 * d$age
d$year_m <- 12 * d$dxyr
p <- pyears(Surv(futime, death) ~ tcut(age_m, 12 * ages) +
              tcut(year_m, 12 * years) + sex,
            data = d, ratetable = rt, scale = 1, data.frame = TRUE,
            rmap = list(age = age_m, year = year_m, sex = sex))
df <- p$data
rate <- rt[cbind(as.integer(df[[1]]), as.integer(df[[2]]),
                 as.integer(df$sex))]
add("pyears_age_period_sex",
    paste("pyears(Surv(futime, death) ~ tcut(12 * age, 12 * ages) +",
          "tcut(12 * dxyr, 12 * years) + sex, ratetable = rt, rmap =",
          "list(age = 12 * age, year = 12 * dxyr, sex = sex), scale = 1,",
          "data.frame = TRUE)"),
    list(ages = ages, years = years, unit = "months",
         rate = "0.001 i^2 + 0.0003 j + 0.002 (sex == M), per month"),
    list(age = 12 * ages[as.integer(df[[1]])],
         year = 12 * years[as.integer(df[[2]])],
         sex = as.character(df$sex), rate = rate,
         pyears = df$pyears, n = df$n, event = df$event,
         expected = df$expected, offtable = p$offtable))

# -- late entry, time since entry, off-table time --------------------------
ages2 <- c(50, 65, 80, 120)
since <- c(0, 24, 60, 500)
d$since0 <- -d$tl
p2 <- pyears(Surv(tl, futime, death) ~ tcut(age_m, 12 * ages2) +
               tcut(since0, since) + sex,
             data = d, scale = 1, data.frame = TRUE)
df2 <- p2$data
add("pyears_late_entry",
    paste("pyears(Surv(tl, futime, death) ~ tcut(12 * age, 12 * ages) +",
          "tcut(-tl, since) + sex, scale = 1, data.frame = TRUE)"),
    list(ages = ages2, since = since,
         unit = "months"),
    list(age = (12 * ages2)[as.integer(df2[[1]])],
         since = since[as.integer(df2[[2]])],
         sex = as.character(df2$sex),
         pyears = df2$pyears, n = df2$n, event = df2$event,
         offtable = p2$offtable))

# -- exact Poisson intervals ------------------------------------------------
cells <- list(c(0, 1000), c(1, 250.5), c(7, 3120), c(52, 81234))
lo <- c(); hi <- c(); dd <- c(); tt <- c()
for (cl in cells) {
  ci <- poisson.test(cl[1], cl[2], conf.level = 0.95)$conf.int
  dd <- c(dd, cl[1]); tt <- c(tt, cl[2]); lo <- c(lo, ci[1]); hi <- c(hi, ci[2])
}
ci90 <- poisson.test(7, 3120, conf.level = 0.9)$conf.int
add("garwood",
    "poisson.test(d, T, conf.level = 0.95)$conf.int (and 0.9 for d = 7)",
    list(conf_level = 0.95),
    list(events = dd, exposure = tt, lower = lo, upper = hi,
         lower_90 = ci90[1], upper_90 = ci90[2]))

O <- sum(df$event); E <- sum(df$expected)
smr <- poisson.test(O, E)
men <- df$sex == "M"
d1 <- sum(df$event[men]); T1 <- sum(df$pyears[men])
d0 <- sum(df$event[!men]); T0 <- sum(df$pyears[!men])
rr <- poisson.test(c(d1, d0), c(T1, T0))
add("ratios",
    paste("poisson.test(O, E) on the first table's totals;",
          "poisson.test(c(d_M, d_F), c(T_M, T_F)) for the men's rate over",
          "the women's"),
    list(conf_level = 0.95, alternative = "two.sided"),
    list(observed = O, expected = E, oe = O / E,
         oe_lower = smr$conf.int[1], oe_upper = smr$conf.int[2],
         ratio = unname(rr$estimate), ratio_lower = rr$conf.int[1],
         ratio_upper = rr$conf.int[2], p_value = rr$p.value))

out <- list(
  generator = "scripts/reference/reference_r_pyears.R",
  R = R.version.string,
  references = references
)
writeLines(
  toJSON(out, digits = NA, auto_unbox = TRUE, na = "null", pretty = TRUE),
  out_file
)
