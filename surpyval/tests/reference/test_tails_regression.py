"""Accuracy of the proportional-odds regression functions in the tails
(#528), against 50-digit values.

A proportional-odds model's survival is
:math:`S = \\phi S_0 / (F_0 + \\phi S_0)`, so its cumulative hazard is
:math:`-\\ln S = \\ln(1 + F_0 / (\\phi S_0))`. The 50-digit baseline values
:math:`S_0` and :math:`F_0` of every PO baseline (Exponential, Normal,
Weibull, Gumbel, Logistic, LogNormal, Gamma) at extreme parameters and times
are those ``reference/test_tails.py`` checks the distributions against
(``data/tails_mpmath.json``, mpmath at 50 digits); the PO values are formed
from them here at 60 digits with ``decimal``, for an odds multiplier
:math:`\\phi = e^{\\beta z}` with :math:`\\beta z` in {-3, 0.5, 25}. ``Hf``,
``log_sf`` and ``Hf_tvc`` along a constant ``StepSchedule`` are checked by
the rules of ``test_tails.py`` (relative error at most 1e-8, or the
rounding of the inputs where the function is ill-conditioned). Before
#528, ``Hf`` was :math:`H_0 - \\ln\\phi + \\ln(F_0 + \\phi S_0)`, whose terms
cancel where the cumulative hazard is small: 20 % wrong at 4e-16.
"""

import decimal
import math
import warnings
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
import pytest

import surpyval
from surpyval.univariate.regression.tvc_schedule import StepSchedule

from ._data import _load
from .test_tails import _problem

BASELINES = (
    "Exponential",
    "Normal",
    "Weibull",
    "Gumbel",
    "Logistic",
    "LogNormal",
    "Gamma",
)
LOG_PHI = (-3.0, 0.5, 25.0)
FUNCTIONS = ("Hf", "log_sf", "Hf_tvc")

_CTX = decimal.Context(prec=60, Emin=-999999, Emax=999999)


def _content():
    return _load("tails_mpmath")


def _log1p(r: decimal.Decimal) -> decimal.Decimal:
    """``log(1 + r)`` to 60 digits, also where ``r`` is far below 1e-60."""
    if r < decimal.Decimal("1e-15"):
        # The fourth term is below 1e-60 of the first.
        return _CTX.subtract(
            _CTX.add(r, _CTX.divide(_CTX.power(r, 3), 3)),
            _CTX.divide(_CTX.power(r, 2), 2),
        )
    return _CTX.ln(_CTX.add(1, r))


def _truth(S0_text: str, F0_text: str, log_phi: float) -> "tuple[str, float]":
    """The true PO cumulative hazard (as text) and its sensitivity to the
    rounding of the baseline values and of ``phi``, relative to the
    baseline sensitivities (filled in by the caller)."""
    S0 = decimal.Decimal(S0_text)
    F0 = decimal.Decimal(F0_text)
    phi = _CTX.exp(decimal.Decimal(log_phi))
    if F0 == 0:
        return "0.0", 0.0
    if S0 == 0:
        return "inf", 0.0
    r = _CTX.divide(F0, _CTX.multiply(phi, S0))
    return str(_log1p(r)), float(r / (1 + r))


@lru_cache(maxsize=None)
def _model(dist_name: str) -> Any:
    """A PO model on ``dist_name`` with one covariate, whose parameters the
    test sets."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 1))
    dist = getattr(surpyval, dist_name)
    if dist_name in ("Normal", "Logistic", "Gumbel"):
        x = rng.normal(10.0, 2.0, 60)
    else:
        x = rng.weibull(1.5, 60) * 10.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return surpyval.PO(dist).fit(x, Z)


@lru_cache(maxsize=None)
def _package(c: int, log_phi: float) -> dict:
    """The package's ``Hf``, ``log_sf`` and ``Hf_tvc`` at every point of
    case ``c``, for the coefficient ``log_phi`` at ``z = 1``."""
    case = _content()["cases"][c]
    model = _model(case["dist"])
    model.params = np.array(list(case["params"]) + [log_phi])
    x = np.asarray(case["x"], dtype=float)
    Z = np.ones((x.size, 1))
    out = {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out["Hf"] = np.asarray(model.Hf(x, Z), dtype=float)
        out["log_sf"] = np.asarray(
            model.model.log_sf(x, Z, *model.params), dtype=float
        ).ravel()
        schedule = StepSchedule.constant([1.0])
        # Hf_tvc's query times must be in the schedule's span; times below
        # the support's bottom are the same evaluation.
        out["Hf_tvc"] = np.asarray(model.Hf_tvc(x, schedule), dtype=float)
    return out


def _groups() -> dict:
    groups = defaultdict(list)
    for c, case in enumerate(_content()["cases"]):
        if case["dist"] not in BASELINES:
            continue
        for i, regime in enumerate(case["regime"]):
            for fn in FUNCTIONS:
                groups[(case["dist"], fn, regime)].append((c, i))
    return groups


GROUPS = _groups()


def check_group(key: tuple) -> list:
    """Every failing point of a group, worst first."""
    dist, fn, regime = key
    functions = _content()["settings"]["functions"]
    j_sf, j_ff = functions.index("sf"), functions.index("ff")
    failures = []
    for c, i in GROUPS[key]:
        case = _content()["cases"][c]
        S0_text, F0_text = case["values"][i][j_sf], case["values"][i][j_ff]
        sens_S = float(case["sens"][i][j_sf])
        sens_F = float(case["sens"][i][j_ff])
        for log_phi in LOG_PHI:
            true_text, weight = _truth(S0_text, F0_text, log_phi)
            S0, F0 = float(S0_text), float(F0_text)
            # d H / d log(theta) = r / (1 + r) (d log F0 - d log S0 + d
            # log phi): the rounding of the inputs alone costs this.
            sens = weight * (
                (sens_F / F0 if F0 > 0 else 0.0)
                + (sens_S / S0 if S0 > 0 else 0.0)
                + 1.0
            )
            got = float(_package(c, log_phi)[fn][i])
            if fn == "log_sf":
                got = -got if got != 0 else 0.0
            # Hf is checked as Hf; log_sf as -log_sf against the same truth.
            problem = _problem("Hf", got, true_text, repr(sens), False)
            if problem is None:
                continue
            true = float(true_text)
            err = (
                math.inf
                if true == 0 or math.isinf(true) or math.isnan(got)
                else abs(got - true) / abs(true)
            )
            failures.append(
                (
                    err,
                    "{} at params={} log(phi)={} x={!r}: got {!r}, true {!r} "
                    "(rel err {:.2g})".format(
                        problem,
                        tuple(case["params"]),
                        log_phi,
                        case["x"][i],
                        got,
                        float(true_text),
                        err,
                    ),
                )
            )
    failures.sort(key=lambda f: -f[0])
    return failures


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(key, id="PO-{}.{}-{}".format(*key))
        for key in sorted(GROUPS)
    ],
)
def test_po_accuracy_in_the_tails(key):
    failures = check_group(key)
    assert not failures, "PO {}.{} in {}: {} wrong; {}".format(
        *key, len(failures), failures[0][1]
    )


def test_every_po_baseline_is_covered():
    covered = {key[0] for key in GROUPS}
    assert covered == set(BASELINES)


def test_hf_agrees_with_ff_on_the_issue_example():
    # The reproduction of #528: Hf and -log1p(-ff) disagreed by 1e-3 at
    # x = 1e-7.
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(400, 1))
    x = rng.weibull(1.8, size=400) * 80.0 * np.exp(-0.4 * Z[:, 0]) + 1e-3
    model = surpyval.WeibullPO.fit(x, Z)
    t = np.array([1e-7, 1e-5, 1e-3, 1.0])
    for z in (0.5, -3.0):
        Zt = np.full((4, 1), z)
        ratio = model.Hf(t, Zt) / -np.log1p(-model.ff(t, Zt))
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-14, atol=0)
