"""Card: life from a condition-monitoring signal.

Persona: a condition-monitoring engineer with monthly vibration RMS
readings on 20 main bearings, from a healthy level of 1.0 mm/s, and an
alarm (the failure definition) at 7 mm/s. Questions (Meeker and Escobar
ch. 13; Lawless and Crowder, 2004):

1. Gamma or Wiener process: what is the mean life to the alarm, and the
   B10, for a bearing installed today?
2. For a bearing reading 5.5 mm/s now, how long until the alarm?

Truth: Gamma-process increments, shape 4.0 per year, scale 0.5, so
2.0 mm/s per year on average and about 3.0 years from 1.0 to 7.0 mm/s.
"""

import numpy as np
import pandas as pd
import pytest

import surpyval as sp

BASELINE, ALARM = 1.0, 7.0
RATE_SHAPE, SCALE = 4.0, 0.5
MEAN_LIFE = (ALARM - BASELINE) / (RATE_SHAPE * SCALE)


def _readings():
    rng = np.random.default_rng(5)
    rows = []
    for unit in range(20):
        months = rng.integers(12, 37)
        t = np.arange(months + 1) / 12.0
        y = (
            BASELINE
            + np.r_[0, np.cumsum(rng.gamma(RATE_SHAPE * np.diff(t), SCALE))]
        )
        rows += [dict(i=f"B{unit}", x=a, y=b) for a, b in zip(t, y)]
    return pd.DataFrame(rows)


DF = _readings()


def test_increments_recover_the_process():
    # The increments do not depend on the starting level: the fit with
    # the baseline removed and with it kept is the same.
    kept = sp.GammaProcess.fit_from_df(DF, threshold=ALARM)
    removed = sp.GammaProcess.fit_from_df(
        DF.assign(y=DF.y - BASELINE), threshold=ALARM - BASELINE
    )
    np.testing.assert_allclose(kept.params, removed.params)
    assert removed.mean() == pytest.approx(MEAN_LIFE, rel=0.1)


def test_remaining_life_from_the_current_level():
    model = sp.GammaProcess.fit_from_df(DF, threshold=ALARM)
    rul = model.predict_rul(5.5)
    assert 0 < rul.rul < (ALARM - 5.5) / (RATE_SHAPE * SCALE) * 2


def test_life_to_the_alarm_from_the_healthy_level():
    # #574: the life is the first passage from the level a new unit
    # starts at (y0, estimated from the readings: every unit is read at
    # installation here, so it is their healthy level), not from 0. So the
    # readings as they are and with the baseline removed give the same
    # life to the alarm, and the life is the truth's.
    model = sp.GammaProcess.fit_from_df(DF, threshold=ALARM)
    removed = sp.GammaProcess.fit_from_df(
        DF.assign(y=DF.y - BASELINE), threshold=ALARM - BASELINE
    )
    assert model.y0 == pytest.approx(BASELINE)
    assert model.mean() == pytest.approx(removed.mean(), rel=1e-8)
    assert model.qf(0.1) == pytest.approx(removed.qf(0.1), rel=1e-8)
    assert model.mean() == pytest.approx(MEAN_LIFE, rel=0.05)
