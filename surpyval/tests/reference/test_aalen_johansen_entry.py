"""The Aalen-Johansen cumulative incidence with delayed entry (left
truncation, ``CompetingRisks.fit(..., tl=)``) against R ``survival``.

R 4.6.2 and ``survival`` 3.8.12: ``summary(survfit(Surv(tl, x, factor(e,
0:2)) ~ 1, weights = w, id = , timefix = FALSE), times = TIMES, extend =
TRUE)``, its ``pstate`` and ``std.err`` (the infinitesimal-jackknife,
robust, standard error), printed by
``scripts/reference/reference_r_aalen_johansen_entry.R``.

- "pbc_age": the Mayo PBC trial (``load_pbc2``, one row per patient) on
  the age time scale: each patient enters at their age at enrolment and
  leaves at that age plus their follow-up; cause 1 is death and cause 2
  transplant (alive is censored). Entries fall between the events
  throughout, and the risk set is small at the youngest ages.
- "pbc_age_weighted": the same with the weights ``1 + (id mod 3) / 2``.
- "ties": fifteen units with tied failures, a failure tied with a
  censoring, and entries exactly at failure times (not at risk for them,
  under the ``(entry, exit]`` rule).

The cumulative incidences of the unweighted fit and the robust standard
errors of the weighted fit (``w = 1`` for the unweighted data) are closed
forms of the same risk sets as R's: equal to rounding (``rtol=1e-10``).

The unweighted fit's own standard error is Aalen's (that of
``cmprsk::cuminc``), which no R function computes with delayed entry
(``cuminc`` has no entry times, and ``mstate``'s Aalen-type estimate is a
different one). It is a function of the numbers at risk and of events
only, so it carries over to delayed entry unchanged (Andersen, Borgan,
Gill and Keiding, 1993, IV.4); here it is checked against R's robust one,
which estimates the same variance: within 6% on these data (they differ
most where few are at risk), and exactly by the identities in
``test_nonparametric_competing_risks_entry.py``.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.datasets import load_pbc2
from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_variance,
)

PBC_TIMES = np.array([45, 50, 55, 60, 65, 70, 75, 80.0])
TIES_TIMES = np.array([2, 3, 5, 6, 8, 10.0])

# fmt: off
R = {
    "pbc_age": {
        "cif1": [0.252376636748268, 0.355276302655414, 0.412409585563236, 0.479567911945595, 0.522626416002136, 0.55160395614863, 0.583300102654567, 0.595444255925782],  # noqa: E501
        "se1": [0.0667380913238604, 0.0727406939256003, 0.0767627814821729, 0.0823670370728601, 0.0866927575065928, 0.0900023968490053, 0.0937815512232695, 0.0952724936479307],  # noqa: E501
        "cif2": [0.345276661764494, 0.377876818800848, 0.394390930754025, 0.396496535131951, 0.396496535131951, 0.398624878523159, 0.398624878523159, 0.398624878523159],  # noqa: E501
        "se2": [0.102502969615308, 0.0986031137653277, 0.0965711845041413, 0.0963037725696075, 0.0963037725696075, 0.0960327532268399, 0.0960327532268399, 0.0960327532268399],  # noqa: E501
    },
    "pbc_age_weighted": {
        "cif1": [0.268766817513031, 0.381496080447651, 0.44252568719014, 0.515437339681008, 0.559741785527674, 0.590867888482176, 0.623336571833526, 0.63810558473683],  # noqa: E501
        "se1": [0.0649923862255392, 0.0667310102073675, 0.0679207517438776, 0.0700715563392636, 0.0722309934436158, 0.0742631377325678, 0.0765596169558572, 0.0776409647083084],  # noqa: E501
        "cif2": [0.30943597553323, 0.33752539760014, 0.353868356177455, 0.355358085124004, 0.355358085124004, 0.356821880457596, 0.356821880457596, 0.356821880457596],  # noqa: E501
        "se2": [0.0816866071245552, 0.0795794349952619, 0.0783252101836667, 0.0781969489000996, 0.0781969489000996, 0.078069869164323, 0.078069869164323, 0.078069869164323],  # noqa: E501
    },
    "ties": {
        "cif1": [0.166666666666667, 0.388888888888889, 0.452380952380952, 0.497732426303855, 0.58843537414966, 0.58843537414966],  # noqa: E501
        "se1": [0.152145154862546, 0.173224879017312, 0.162442417929066, 0.154780900809535, 0.157494078874727, 0.157494078874727],  # noqa: E501
        "cif2": [0.166666666666667, 0.166666666666667, 0.23015873015873, 0.23015873015873, 0.275510204081633, 0.41156462585034],  # noqa: E501
        "se2": [0.152145154862546, 0.152145154862546, 0.158163829715135, 0.158163829715135, 0.157963446210676, 0.157494078874727],  # noqa: E501
    },
}
# fmt: on


def _data(case):
    if case == "ties":
        tl = np.array([0, 0, 0, 0, 0, 2, 2, 3, 3, 3, 5, 5, 6, 1, 4.0])
        x = np.array([2, 2, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 3, 9.0])
        e = np.array([1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 1, 0])
        return x, tl, np.where(e == 0, None, e), None, TIES_TIMES
    df = load_pbc2()
    patients = df[~df["id"].duplicated()]
    tl = patients["age"].to_numpy()
    x = tl + patients["years"].to_numpy()
    status = patients["status"].to_numpy()
    e = np.where(
        status == "alive", None, np.where(status == "dead", 1, 2)
    ).astype(object)
    w = None
    if case == "pbc_age_weighted":
        w = 1 + (patients["id"].to_numpy() % 3) / 2
    return x, tl, e, w, PBC_TIMES


def _se(var, model, times):
    idx = np.searchsorted(model.x, times, side="right") - 1
    return np.where(idx >= 0, np.sqrt(var[np.maximum(idx, 0)]), 0.0)


@pytest.mark.parametrize("case", sorted(R))
def test_cumulative_incidence_matches_survfit(case):
    x, tl, e, w, times = _data(case)
    model = sp.CompetingRisks.fit(x, e, tl=tl, w=w)
    for cause in (1, 2):
        assert_allclose(
            model.cif(times, cause), R[case][f"cif{cause}"], rtol=1e-10
        )


@pytest.mark.parametrize("case", sorted(R))
def test_robust_standard_error_matches_survfit(case):
    x, tl, e, w, times = _data(case)
    w = np.ones(len(x)) if w is None else w
    with warnings.catch_warnings():
        # Fifteen units are few for a robust variance; the fit says so.
        warnings.simplefilter("ignore")
        model = sp.CompetingRisks.fit(x, e, tl=tl, w=w)
    for cause in (1, 2):
        var = model.cif_var[model.event_idx_map[cause]]
        assert_allclose(
            _se(var, model, times),
            R[case][f"se{cause}"],
            rtol=1e-10,
            atol=1e-14,
        )


def test_aalen_standard_error_is_close_to_the_robust_one():
    x, tl, e, _, times = _data("pbc_age")
    model = sp.CompetingRisks.fit(x, e, tl=tl)
    for cause in (1, 2):
        var = aalen_johansen_variance(
            model.r, model.d, model.d_e[model.event_idx_map[cause]]
        )
        assert_allclose(
            _se(var, model, times), R["pbc_age"][f"se{cause}"], rtol=0.06
        )
