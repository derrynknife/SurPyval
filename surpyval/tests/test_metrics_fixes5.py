"""Prediction metrics: censoring convention at ties and the #290 review.

* #365: the censoring survival behind the IPCW weights counted an event as
  still at risk of being censored at its own time. The metrics now use the
  events-first reverse Kaplan-Meier and weight an event by ``1 / G(x_i-)``
  (Gerds and Schumacher 2006, R's ``pec``). On a *population* data set --
  every combination of covariate, event time and censoring time once, so
  that the Kaplan-Meier estimates are exact -- the IPCW Brier score and AUC
  must then equal their true values exactly, ties and all.
* Without ties between event and censoring times the metrics agree with
  scikit-survival to rounding.
* The integrated Brier score integrates in time order whatever the order of
  the grid; the flags must be right-censoring flags of the right length;
  a score that needs the censoring survival where the training estimate has
  reached 0 is ``nan`` rather than silently biased.
"""

import itertools

import numpy as np
import pytest

from surpyval.metrics import auc_td, brier_score, integrated_brier_score
from surpyval.utils.ipcw import censoring_survival

# Event time given a binary covariate z, and an independent censoring time,
# each uniform on a few integers so that events and censorings tie.
T_GIVEN_Z = {0: [2.0, 3.0, 4.0], 1: [1.0, 2.0, 3.0]}
C_VALUES = [1.0, 2.0, 3.0, 4.0]
PRED = {0: 0.7, 1: 0.4}  # predicted survival, constant in t


def _population():
    z, x, c = [], [], []
    for zi, ts in T_GIVEN_Z.items():
        for t, cens in itertools.product(ts, C_VALUES):
            z.append(zi)
            x.append(min(t, cens))
            c.append(0 if t <= cens else 1)  # a tie is recorded as an event
    return np.array(z), np.array(x), np.array(c)


def _true_brier(t):
    pairs = [(z, T) for z, ts in T_GIVEN_Z.items() for T in ts]
    return np.mean([(float(T > t) - PRED[z]) ** 2 for z, T in pairs])


def _true_auc(t):
    pairs = [(z, T) for z, ts in T_GIVEN_Z.items() for T in ts]
    cases = [z for z, T in pairs if T <= t]
    controls = [z for z, T in pairs if T > t]
    return np.mean(
        [float(i > j) + 0.5 * float(i == j) for i in cases for j in controls]
    )


def test_censoring_survival_tie_conventions():
    x = np.array([1.0, 2.0, 2.0, 3.0])
    cens = np.array([False, False, True, False])
    # Default (Fine-Gray / cmprsk::crr): the event at 2 is at risk of
    # censoring at 2, so the step there is 1 - 1/3. Unchanged.
    _, g = censoring_survival(x, cens)
    np.testing.assert_allclose(g, [1.0, 2 / 3, 2 / 3])
    # Events first (prodlim reverse = TRUE, scikit-survival): 1 - 1/2.
    _, g = censoring_survival(x, cens, ties="events_first")
    np.testing.assert_allclose(g, [1.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="ties"):
        censoring_survival(x, cens, ties="censored_first")


@pytest.mark.parametrize("t", [1.0, 2.0, 2.5, 3.0])
def test_brier_exact_on_population_with_ties(t):
    # At t = 2 the truth is 0.2250. The old code (events at risk of
    # censoring, 1/G(x_i)) gave 0.2330; the events-first estimate with
    # 1/G(x_i) (scikit-survival's weights) gives 0.2881.
    z, x, c = _population()
    S = np.array([PRED[zi] for zi in z]).reshape(-1, 1)
    _, bs = brier_score(x, c, S, [t])
    assert bs[0] == pytest.approx(_true_brier(t), abs=1e-12)


@pytest.mark.parametrize("t", [1.0, 2.0, 2.5, 3.0])
def test_auc_exact_on_population_with_ties(t):
    # Binary risk score: tied scores count one half. At t = 2 the truth is
    # 2/3; the old code gave 0.6703, scikit-survival's weights 0.6603.
    z, x, c = _population()
    _, auc = auc_td(x, c, z.astype(float), [t])
    assert auc[0] == pytest.approx(_true_auc(t), abs=1e-12)


def test_tied_event_and_censoring_hand_example():
    # Weights at the tie at 3: G(3-) = 1 for the event, G(3.5) = 1/2 for
    # the survivor (at risk of censoring at 3: the censored row and x = 4).
    x = [1.0, 2.0, 3.0, 3.0, 4.0]
    c = [0, 0, 0, 1, 0]
    S = [[0.8], [0.6], [0.5], [0.5], [0.3]]
    _, bs = brier_score(x, c, S, [3.5])
    assert bs[0] == pytest.approx((0.64 + 0.36 + 0.25 + 2 * 0.49) / 5)


def _sksurv_data(seed, n):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = rng.exponential(np.exp(-0.8 * z))
    cens = rng.exponential(1.5, n)
    return np.minimum(t, cens), (cens < t).astype(int), z


def test_agrees_with_sksurv_without_event_censoring_ties():
    sm = pytest.importorskip("sksurv.metrics")
    from sksurv.util import Surv

    xtr, ctr, _ = _sksurv_data(1, 200)
    xte, cte, zte = _sksurv_data(2, 150)
    # Round the test times so events tie with each other (not with
    # censorings of the training data, which stay continuous).
    ev = cte == 0
    xte[ev] = np.round(xte[ev], 1) + 1e-9
    times = np.quantile(xte[ev], [0.2, 0.4, 0.6, 0.8])
    S = np.exp(-np.outer(np.exp(0.8 * zte), times))
    ytr = Surv.from_arrays(ctr == 0, xtr)
    yte = Surv.from_arrays(cte == 0, xte)

    _, bs = brier_score(xte, cte, S, times, x_train=xtr, c_train=ctr)
    np.testing.assert_allclose(bs, sm.brier_score(ytr, yte, S, times)[1])
    ibs = integrated_brier_score(xte, cte, S, times, x_train=xtr, c_train=ctr)
    assert ibs == pytest.approx(sm.integrated_brier_score(ytr, yte, S, times))
    # Tied risk scores (rounded) count one half in both.
    risk = np.round(1.0 - S, 1)
    _, auc = auc_td(xte, cte, risk, times, x_train=xtr, c_train=ctr)
    ref, _ = sm.cumulative_dynamic_auc(ytr, yte, risk, times)
    np.testing.assert_allclose(auc, ref)


def test_ibs_does_not_depend_on_grid_order():
    # Old code: 0.1908 for the shuffled grid against 0.1949 sorted.
    rng = np.random.default_rng(3)
    x = rng.exponential(5, 50)
    c = (rng.uniform(size=50) < 0.3).astype(int)
    times = np.array([1.0, 2.0, 3.0, 4.0])
    S = np.exp(-np.outer(np.ones(50), times) / 5)
    p = [2, 0, 3, 1]
    assert integrated_brier_score(x, c, S[:, p], times[p]) == pytest.approx(
        integrated_brier_score(x, c, S, times)
    )


def test_flags_must_be_right_censoring_of_matching_length():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    S = np.full((4, 1), 0.5)
    # A left-censored row was scored as a known survivor past its time.
    with pytest.raises(ValueError, match="right censored"):
        brier_score(x, [0, -1, 0, 1], S, [1.5])
    with pytest.raises(ValueError, match="right censored"):
        auc_td(x, [0, 2, 0, 1], S, [1.5])
    # A length-one c broadcast silently to 'every row an event'.
    with pytest.raises(ValueError, match="same length"):
        brier_score(x, [0], S, [1.5])
    with pytest.raises(ValueError, match="both"):
        brier_score(x, [0, 0, 0, 1], S, [1.5], x_train=x)


def test_nan_where_training_censoring_survival_is_zero():
    # The training censoring estimate reaches 0 at 5 (its last row is
    # censored). A survivor at the horizon 6, or an event after 5, needs
    # 1/G = 1/0: the score is not identified. A zero weight used to drop
    # those rows and bias the score towards 0.
    xtr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ctr = np.array([0, 1, 0, 0, 1])
    xte = np.array([1.5, 2.5, 5.5, 7.0, 8.0])
    cte = np.array([0, 1, 0, 0, 1])
    S = np.full((5, 2), 0.5)
    times = [2.0, 6.0]
    _, bs = brier_score(xte, cte, S, times, x_train=xtr, c_train=ctr)
    assert np.isfinite(bs[0]) and np.isnan(bs[1])
    _, auc = auc_td(xte, cte, xte[::-1], times, x_train=xtr, c_train=ctr)
    assert np.isfinite(auc[0]) and np.isnan(auc[1])
