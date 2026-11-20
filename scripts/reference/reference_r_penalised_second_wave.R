# The reference values of the second wave of penalised fits in
# surpyval/tests/univariate/regression/test_penalised_second_wave.py:
#
# - Fine-Gray (FineGray.fit_penalised) on the PBC data at enrolment
#   (death the cause of interest, transplant the competing cause). The
#   subdistribution partial likelihood is a Cox partial likelihood in
#   which a unit that failed from the competing cause at x stays in the
#   risk set of each later event time t_j of the cause of interest with
#   weight w_j = G(t_j-) / G(x-) (G the Kaplan-Meier of the censoring,
#   survfit's, as cmprsk::crr). The weights fall with t_j, so they are
#   written as right-censored rows: one row ending at each t_j with
#   weight w_j - w_{j+1} (w after the last 0), and the unit's own row
#   ending at x with weight 1 - w_1; at each event time the unit's rows
#   at risk then weigh w_j, and 1 before x. Every unit's rows weigh 1 in
#   all, so glmnet's family = "cox" with these weights (Breslow's ties),
#   standardize = FALSE on the covariates divided by their standard
#   deviations over the units (divisor N), at the same lambda, is the
#   penalised fit. cmprsk::crr is printed beside a glmnet fit at lambda
#   1e-12 as a check of the rows (they agree to 1e-10).
# - Lin and Ying's additive hazards model (AdditiveHazards.fit_penalised)
#   on the Rossi data: the penalised fit minimises a quadratic, so it is
#   a least-squares lasso on its Cholesky factor, which glmnet's
#   family = "gaussian" (no intercept, no standardisation) solves. A and
#   b (the estimate solves A beta = b) are computed here, risk sets at
#   each distinct exit time and covariate means at each event time (ties
#   share one mean), the convention of AdditiveHazards.fit; on untied
#   data they are timereg's intZHZ and intZHdN. The response is scaled
#   to unit root mean square, which glmnet's gaussian fit otherwise does
#   itself and which would change its ridge penalty.
#
#     Rscript scripts/reference/reference_r_penalised_second_wave.R \
#         surpyval/datasets
#
# Run with R 4.3.3, glmnet 4.1.8, survival 3.5.8 and cmprsk 2.2.11. Each
# fit follows a path of 60 penalties down to the target (glmnet's
# warm starts) with a threshold of 1e-30 (thresh and maxit are glmnet's
# own arguments: glmnet 4.1 takes no control list) and prints the last
# coefficients, on the covariates' own scale.
suppressMessages({library(glmnet); library(survival); library(cmprsk)})
args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "surpyval/datasets"
# (from 1, or from ten times the target where that is above 1: glmnet
# takes the penalties in decreasing order)
path_to <- function(lambda) {
  exp(seq(log(max(1, 10 * lambda)), log(lambda), length.out = 60))
}
glmnet.control(fdev = 0)
show <- function(label, b) {
  cat(label, ": [", paste(sprintf("%.10g", b), collapse = ", "), "]\n", sep = "")
}
sd_n <- function(X) sqrt(colMeans(sweep(X, 2, colMeans(X))^2))

# -- Fine-Gray on the PBC data at enrolment ---------------------------------
pbc <- read.csv(file.path(dir, "pbc2.csv"))
pbc <- pbc[!duplicated(pbc$id), ]
X <- cbind(
  age = pbc$age,
  female = as.numeric(pbc$sex == "female"),
  drug = as.numeric(pbc$drug == "D-penicil"),
  logbili = log(pbc$serBilir),
  albumin = pbc$albumin
)
time <- pbc$years
status <- ifelse(pbc$status == "dead", 1, ifelse(pbc$status == "transplanted", 2, 0))
N <- nrow(X)
s <- sd_n(X)
# G(t-), the censoring Kaplan-Meier just before t
km <- survfit(Surv(time, status == 0) ~ 1)
G_before <- function(t) sapply(t, function(u) {
  k <- which(km$time < u)
  if (length(k)) km$surv[max(k)] else 1
})
events <- sort(unique(time[status == 1]))
rows <- data.frame(stop = time, event = as.numeric(status == 1), w = 1,
                   unit = seq_len(N))
for (i in which(status == 2)) {
  later <- events[events > time[i]]
  if (!length(later)) next
  wt <- G_before(later) / G_before(time[i])
  rows$w[i] <- 1 - wt[1]
  rows <- rbind(rows, data.frame(stop = later, event = 0,
                                 w = wt - c(wt[-1], 0), unit = i))
}
rows <- rows[rows$w > 0, ]
cat(sprintf("rows %d, total weight %.12g, units %d\n", nrow(rows),
            sum(rows$w), N))
Xs <- sweep(X, 2, s, "/")[rows$unit, ]
y <- Surv(rows$stop, rows$event)
fg <- function(lambda, alpha) {
  f <- glmnet(Xs, y, family = "cox", weights = rows$w, alpha = alpha,
              standardize = FALSE, lambda = path_to(lambda),
              thresh = 1e-30, maxit = 1e8)
  f$beta[, ncol(f$beta)] / s
}
crr_fit <- crr(time, status, X, failcode = 1, cencode = 0)
show("crr", crr_fit$coef)
show("fine-gray lambda=1e-12 alpha=1", fg(1e-12, 1))
for (setting in list(c(0.05, 0), c(0.05, 0.5), c(0.02, 1), c(0.002, 1))) {
  show(sprintf("fine-gray lambda=%g alpha=%g", setting[1], setting[2]),
       fg(setting[1], setting[2]))
}

# -- Lin and Ying's additive hazards on the Rossi data -----------------------
rossi <- read.csv(file.path(dir, "rossi.csv"))
cols <- c("fin", "age", "race", "wexp", "mar", "paro", "prio")
X <- as.matrix(rossi[, cols])
x <- rossi$week
d <- rossi$arrest
N <- nrow(X)
p <- ncol(X)
u <- sort(unique(x))
widths <- diff(c(0, u))
A <- matrix(0, p, p)
b <- rep(0, p)
for (j in seq_along(u)) {
  at_risk <- x >= u[j]
  Zr <- X[at_risk, , drop = FALSE]
  m <- colMeans(Zr)
  A <- A + widths[j] * crossprod(sweep(Zr, 2, m))
  ev <- at_risk & x == u[j] & d == 1
  if (any(ev)) b <- b + colSums(sweep(X[ev, , drop = FALSE], 2, m))
}
s <- sd_n(X)
tau <- mean(x)
Q <- A / (tau * outer(s, s)) / N
q <- b / s / N
R <- chol(Q)
yy <- backsolve(R, q, transpose = TRUE)
k <- sqrt(mean(yy^2))
lin_ying <- function(lambda, alpha) {
  f <- glmnet(R / k, yy / k, family = "gaussian", intercept = FALSE,
              standardize = FALSE, alpha = alpha,
              lambda = path_to(lambda / (p * k^2)), thresh = 1e-30,
              maxit = 1e8)
  v <- f$beta[, ncol(f$beta)]
  # The optimality conditions of the penalised quadratic, as a check
  g <- Q %*% v - q + lambda * (1 - alpha) * v
  kkt <- ifelse(v != 0, abs(g + lambda * alpha * sign(v)),
                pmax(abs(g) - lambda * alpha, 0))
  cat(sprintf("  (largest breach of optimality %.2e)\n", max(kkt)))
  v / (s * tau)
}
show("lin-ying unpenalised", solve(A, b))
show("lin-ying ridge closed form lambda=0.05",
     solve(Q + 0.05 * diag(p), q) / (s * tau))
for (setting in list(c(0.05, 0), c(0.05, 0.5), c(0.02, 1), c(0.005, 1))) {
  show(sprintf("lin-ying lambda=%g alpha=%g", setting[1], setting[2]),
       lin_ying(setting[1], setting[2]))
}
