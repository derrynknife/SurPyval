"""Fine-Gray censoring weights at tied censoring and event times.

A subject that failed from a competing cause at ``x_i`` stays in the
subdistribution risk set at a later event time ``t`` with the weight

    w_i(t) = G(t-) / G(x_i-),

``G`` the reverse Kaplan-Meier estimate of the censoring survival. This is
the weight of R's ``cmprsk::crr`` (the R wrapper computes ``uuu``, the
censoring Kaplan-Meier at ``ftime-``, and the Fortran ``crrfsv`` weights
competing failures by ``uuu(t)/uuu(i)``): an event and a censoring at the
same time are ordered event first. SurPyval used ``G(t)/G(x_i)`` before,
which counted the censorings at ``t`` against the events they tie with.
"""

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
    FineGray,
)
from surpyval.utils.ipcw import step_at, step_left_limit

# A small data set with censorings tied to events at t = 2, 3 and 5, and a
# last censored row that takes the censoring survival to zero.
#   status: "a" the cause of interest, "b" the competing cause, None censored
X = np.array([1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8], dtype=float)
E = np.array(["b", "a", None, "b", None, "a", "a", None, "b", "a", None])
Z = np.array([0.2, 1.0, 0.0, -1.0, 0.5, 0.8, -0.4, 1.2, 0.3, 0.0, 0.7])

# Hand computation. The censoring Kaplan-Meier (censorings are the "events";
# the at-risk count at t is everyone with x >= t, as survfit counts it):
#   t = 2: 10 at risk, 1 censored -> G(2) = 9/10
#   t = 3:  8 at risk, 1 censored -> G(3) = 9/10 * 7/8 = 63/80
#   t = 5:  5 at risk, 1 censored -> G(5) = 63/80 * 4/5 = 63/100
#   t = 8:  1 at risk, 1 censored -> G(8) = 0
# so G(1-) = G(2-) = 1, G(3-) = 9/10, G(4-) = G(5-) = 63/80 and
# G(6-) = G(7-) = 63/100. The competing failures are rows 0 (x = 1),
# 3 (x = 3) and 8 (x = 6); at the cause-"a" event times 2, 4, 5, 7:
#   row 0: G(2-)/G(1-) = 1, G(4-)/G(1-) = 63/80, G(5-)/G(1-) = 63/80,
#          G(7-)/G(1-) = 63/100
#   row 3: G(4-)/G(3-) = 7/8, G(5-)/G(3-) = 7/8, G(7-)/G(3-) = 7/10
#   row 8: G(7-)/G(6-) = 1
# Everyone with x >= t has weight 1; censored rows and earlier "a" events 0.
# (With G(t)/G(x_i) instead, row 0 would weigh 9/10 at t = 2, and so on.)
EVENT_TIMES = np.array([2.0, 4.0, 5.0, 7.0])
W_HAND = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [63 / 80, 0, 0, 7 / 8, 0, 1, 1, 1, 1, 1, 1],
        [63 / 80, 0, 0, 7 / 8, 0, 0, 1, 1, 1, 1, 1],
        [63 / 100, 0, 0, 7 / 10, 0, 0, 0, 0, 1, 1, 1],
    ]
)
EVENT_ROWS = np.array([1, 5, 6, 9])


def _hand_neg_ll(beta: float) -> float:
    """Minus the Breslow weighted partial log-likelihood with W_HAND."""
    denom = W_HAND @ np.exp(Z * beta)
    return -float(np.sum(Z[EVENT_ROWS] * beta - np.log(denom)))


def test_step_left_limit_is_the_value_before_each_step():
    times = np.array([1.0, 2.0, 4.0])
    values = np.array([0.9, 0.6, 0.3])
    q = np.array([0.5, 1.0, 1.5, 2.0, 4.0, 5.0])
    np.testing.assert_array_equal(
        step_left_limit(times, values, q, before=1.0),
        [1.0, 1.0, 0.9, 0.9, 0.6, 0.3],
    )
    # Off the grid the left limit and the right-continuous value agree.
    off = np.array([0.5, 1.5, 3.0, 5.0])
    np.testing.assert_array_equal(
        step_left_limit(times, values, off, before=1.0),
        step_at(times, values, off, before=1.0),
    )


def test_tied_fit_matches_hand_weighted_partial_likelihood():
    model = FineGray.fit(X, Z[:, None], E, cause="a")
    d = model.to_dict()

    # The fitted objective is the hand likelihood at the fitted beta.
    assert d["neg_ll"] == pytest.approx(_hand_neg_ll(model.beta[0]), rel=1e-12)
    # And the fitted beta is the hand likelihood's maximiser.
    hand = minimize_scalar(
        _hand_neg_ll, bracket=(-1.0, 1.0), method="brent", tol=1e-12
    )
    assert model.beta[0] == pytest.approx(hand.x, abs=1e-5)

    # Breslow baseline: d(t) / sum_i w_i(t) exp(beta z_i) at each event time.
    dL = 1.0 / (W_HAND @ np.exp(Z * model.beta[0]))
    np.testing.assert_allclose(d["baseline_times"], EVENT_TIMES)
    np.testing.assert_allclose(d["baseline_cumhaz"], np.cumsum(dL), rtol=1e-12)


def test_tied_fit_differs_from_right_continuous_weights():
    # The old weights G(t)/G(x_i): G(1) = 1, G(2) = 9/10, G(3) = 63/80,
    # G(4) = 63/80, G(5) = 63/100, G(6) = G(7) = 63/100.
    W_old = np.array(
        [
            [9 / 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [63 / 80, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            [63 / 100, 0, 0, 63 / 100 / (63 / 80), 0, 0, 1, 1, 1, 1, 1],
            [63 / 100, 0, 0, 63 / 100 / (63 / 80), 0, 0, 0, 0, 1, 1, 1],
        ]
    )

    def old_neg_ll(beta: float) -> float:
        denom = W_old @ np.exp(Z * beta)
        return -float(np.sum(Z[EVENT_ROWS] * beta - np.log(denom)))

    model = FineGray.fit(X, Z[:, None], E, cause="a")
    old = minimize_scalar(
        old_neg_ll, bracket=(-1.0, 1.0), method="brent", tol=1e-12
    )
    assert abs(model.beta[0] - old.x) > 1e-3


def _crr_neg_ll(ftime, fstatus, Zm):
    """
    A line-by-line transcription of cmprsk::crr's objective: the R wrapper's
    ``uuu`` (the censoring Kaplan-Meier, fitted as survfit does, read at
    ``ftime * (1 - 100 eps)``) and the Fortran ``crrfsv`` loop over distinct
    failure times. ``fstatus`` is 0 censored, 1 the cause, 2 a competitor.
    """
    order = np.argsort(ftime, kind="mergesort")
    ftime, fstatus, Zm = ftime[order], fstatus[order], Zm[order]
    ut = np.unique(ftime)
    surv, s = [], 1.0
    for u in ut:
        s *= 1.0 - np.sum((ftime == u) & (fstatus == 0)) / np.sum(ftime >= u)
        surv.append(s)
    eps = np.finfo(float).eps
    knots = np.concatenate(
        [[min(0.0, ut[0]) - 10 * eps], ut, [ut[-1] * (1 + 10 * eps)]]
    )
    kval = np.concatenate([[1.0], surv, [0.0]])
    at = np.searchsorted(knots, ftime * (1 - 100 * eps), side="right") - 1
    uuu = kval[at]
    n = ftime.size

    def neg_ll(b):
        wk = Zm @ b
        lik = 0.0
        iuc = n - 1
        while iuc >= 0:
            failures = np.flatnonzero(fstatus[: iuc + 1] == 1)
            if failures.size == 0:
                break
            cft = ftime[failures[-1]]
            tied = np.flatnonzero(ftime == cft)
            iuc = tied[0]  # first row at this time
            twf = np.sum(fstatus[tied] == 1)
            lik -= np.sum(wk[tied][fstatus[tied] == 1])
            xb1 = 0.0
            for i in range(n):
                if ftime[i] < cft:
                    if fstatus[i] <= 1:
                        continue
                    xb1 += np.exp(wk[i]) * uuu[iuc] / uuu[i]
                else:
                    xb1 += np.exp(wk[i])
            lik += twf * np.log(xb1)
            iuc -= 1
        return lik

    return neg_ll


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tied_fit_matches_crr_transcription(seed):
    rng = np.random.default_rng(seed)
    N = 50
    Zm = rng.normal(size=(N, 2))
    t1 = rng.exponential(1 / (0.3 * np.exp(Zm @ [0.6, -0.3])))
    t2 = rng.exponential(3.0, N)
    tc = rng.exponential(4.0, N)
    x = np.ceil(np.minimum(np.minimum(t1, t2), tc) * 2)  # heavy ties
    status = np.where(tc < np.minimum(t1, t2), 0, np.where(t1 < t2, 1, 2))
    e = np.array([None, "a", "b"], dtype=object)[status]
    # The data really does tie censorings with events.
    assert np.intersect1d(x[status == 0], x[status > 0]).size > 0

    model = FineGray.fit(x, Zm, e, cause="a")
    crr = _crr_neg_ll(x, status, Zm)
    assert model.to_dict()["neg_ll"] == pytest.approx(
        crr(model.beta), rel=1e-12
    )
    ref = minimize(crr, np.zeros(2), method="BFGS", options={"gtol": 1e-9})
    np.testing.assert_allclose(model.beta, ref.x, atol=1e-5)


def test_untied_fit_equals_right_continuous_weights():
    # With no censoring time equal to an event time G(t-) = G(t) at every
    # time the weights use, so the fit is the right-continuous one.
    rng = np.random.default_rng(7)
    N = 60
    Zm = rng.normal(size=(N, 1))
    t1 = rng.exponential(1 / (0.3 * np.exp(0.5 * Zm[:, 0])))
    t2 = rng.exponential(3.0, N)
    tc = rng.exponential(4.0, N)
    x = np.minimum(np.minimum(t1, t2), tc)
    status = np.where(tc < np.minimum(t1, t2), 0, np.where(t1 < t2, 1, 2))
    e = np.array([None, "a", "b"], dtype=object)[status]
    assert np.intersect1d(x[status == 0], x[status > 0]).size == 0

    ev = status == 1
    ut = np.unique(x)
    G = np.cumprod(
        [1 - np.sum((x == u) & (status == 0)) / np.sum(x >= u) for u in ut]
    )
    # (Only the competing failures' G(x_i) is used; a censored last row can
    # have G(x_i) = 0.)
    G_x = np.where(status == 2, step_at(ut, G, x, before=1.0), 1.0)
    G_t = step_at(ut, G, x[ev], before=1.0)
    W = (x[None, :] >= x[ev][:, None]) + (
        (status == 2)[None, :] & (x[None, :] < x[ev][:, None])
    ) * (G_t[:, None] / G_x[None, :])

    def neg_ll(b):
        return -float(np.sum(Zm[ev] @ b - np.log(W @ np.exp(Zm @ b))))

    model = FineGray.fit(x, Zm, e, cause="a")
    assert model.to_dict()["neg_ll"] == pytest.approx(
        neg_ll(model.beta), rel=1e-12
    )


def test_tied_counts_equal_repeated_rows():
    # The weights are count-weighted: n copies of a row is one row with n.
    n = np.array([2, 1, 3, 1, 2, 1, 1, 2, 1, 1, 1])
    expanded = FineGray.fit(
        np.repeat(X, n), np.repeat(Z, n)[:, None], np.repeat(E, n), cause="a"
    )
    counted = FineGray.fit(X, Z[:, None], E, n=n, cause="a")
    np.testing.assert_allclose(counted.beta, expanded.beta, atol=1e-6)
    np.testing.assert_allclose(
        counted.to_dict()["baseline_cumhaz"],
        expanded.to_dict()["baseline_cumhaz"],
        rtol=1e-5,
    )


def test_censoring_survival_reaching_zero_needs_no_guard():
    # The last row is censored alone, so G(8) = 0; the weights only ever use
    # G just before an observed time, which stays positive, and the fit is
    # finite without warnings.
    with np.errstate(divide="raise", invalid="raise"):
        model = FineGray.fit(X, Z[:, None], E, cause="a")
    assert np.all(np.isfinite(model.beta))
    assert np.all(np.isfinite(model.to_dict()["baseline_cumhaz"]))


def test_crph_fine_gray_uses_the_same_tied_weights():
    crph = CompetingRisksProportionalHazards.fit(
        X, Z[:, None], E, how="Fine-Gray"
    )
    standalone = FineGray.fit(X, Z[:, None], E, cause="a")
    np.testing.assert_allclose(
        crph.betas[crph.event_idx_map["a"]], standalone.beta
    )
