"""The semi-parametric recurrent-event regressions against R's
``coxph(... + cluster(id))`` on the bladder cancer recurrences, in the
layout each model takes (``scripts/reference/reference_r_recurrent_cox.R``
wrote the data and the values).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surpyval.recurrent import (
    AndersenGill,
    PrenticeWilliamsPeterson,
    WeiLinWeissfeld,
)

STORED = json.loads(
    (Path(__file__).parent / "data" / "r_recurrent_cox.json").read_text()
)
Z_COLS = ["rx", "size", "number"]


def _rows():
    df = pd.DataFrame(STORED["data"]["columns"])
    df["c"] = 1 - df["event"]
    return df


def _start_stop(fitter, **options):
    return fitter.fit_tvc_from_df(
        _rows(),
        i_col="id",
        xl_col="start",
        xr_col="stop",
        c_col="c",
        Z_cols=Z_COLS,
        **options,
    )


def _event_log(fitter, **options):
    # Every subject is followed from 0 without a break, so the stop
    # times are the event log.
    return fitter.fit_from_df(
        _rows(), x_col="stop", i_col="id", c_col="c", Z_cols=Z_COLS, **options
    )


FITS = {
    "ag": lambda fit: fit(AndersenGill),
    "ag_breslow": lambda fit: fit(AndersenGill, tie_method="breslow"),
    "pwp_total": lambda fit: fit(PrenticeWilliamsPeterson),
    "pwp_gap": lambda fit: fit(PrenticeWilliamsPeterson, time_scale="gap"),
    "wlw": lambda fit: fit(WeiLinWeissfeld),
}
CASES = [
    (ref, layout)
    for ref in FITS
    for layout in (_start_stop, _event_log)
    # The marginal model has no start-stop form.
    if not (ref == "wlw" and layout is _start_stop)
]


@pytest.mark.parametrize(
    "ref, layout", CASES, ids=[f"{r}-{f.__name__}" for r, f in CASES]
)
def test_fit_matches_coxph_with_cluster(ref, layout):
    model = FITS[ref](layout)
    values = STORED["references"][ref]["values"]
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.params, values["coef"], rtol=1e-7)
    np.testing.assert_allclose(
        model.naive_covariance(), values["naive_covariance"], rtol=1e-6
    )
    np.testing.assert_allclose(
        model.covariance(), values["robust_covariance"], rtol=1e-6
    )
    np.testing.assert_allclose(
        model.standard_errors(),
        np.sqrt(np.diag(values["robust_covariance"])),
        rtol=1e-6,
    )
    assert model.log_likelihood == pytest.approx(values["loglik"][1], rel=1e-9)


@pytest.mark.parametrize("ref", list(FITS))
def test_cumulative_baseline_matches_basehaz(ref):
    model = FITS[ref](_event_log)
    values = STORED["references"][ref]["values"]
    times = values["times"]
    for stratum, expected in values["baseline"].items():
        label = None if stratum == "all" else int(stratum.split("=")[-1])
        got = model.cif(times, np.zeros(3), stratum=label)
        np.testing.assert_allclose(got, expected, rtol=1e-7)


def test_the_robust_errors_differ_from_the_naive_ones_as_in_r():
    # The point of the sandwich: R reports 0.2656 for rx where the
    # model-based error is 0.1997.
    model = FITS["ag"](_start_stop)
    table = model.summary()
    assert table.loc["rx", "se(coef)"] == pytest.approx(0.26556, abs=5e-6)
    assert table.loc["rx", "naive se(coef)"] == pytest.approx(
        0.19973, abs=5e-6
    )
