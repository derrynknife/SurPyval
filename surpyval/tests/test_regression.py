"""
This code was created for and sponsored by Cartiga (www.cartiga.com). 
Cartiga makes no representations or warranties in connection with the code 
and waives any and all liability in connection therewith. Your use of the 
code constitutes acceptance of these terms.

Copyright 2022 Cartiga LLC
"""

import pytest
import numpy as np
from ..datasets import (
    Lung,
    Rossi,
    Heart
)
from surpyval.regression import CoxPH

def test_coxph_against_ll_rossi_static():
    ll_answer = np.array([-0.37942216, -0.05743772,  0.31389978,
                          -0.14979572, -0.43370385, -0.08487107, 
                           0.09149708])

    rossi = Rossi.data
    model = CoxPH.fit_from_df(rossi, 
            x_col='week',
            c_col='arrest',
            Z_cols=['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio'],
            method="efron")

    assert np.allclose(model.beta, ll_answer)

# Examples taken from:
# http://www.sthda.com/english/wiki/cox-proportional-hazards-model

def test_coxph_against_r_lung_1():
    r_answer = np.array([-0.5310235])

    lung = Lung.data
    x = lung['time'].values
    c = lung['status'].values
    Z = lung[['sex']].values

    model = CoxPH.fit(x=x, Z=Z, c=c, method="efron")

    assert np.allclose(model.beta, r_answer)

def test_coxph_against_r_lung_2():
    r_answer = np.array([0.01106676, -0.55261240, 0.46372848])

    lung = Lung.data
    model = CoxPH.fit_from_df(
        lung, 
        x_col='time',
        c_col='status',
        Z_cols=['age', 'sex', 'ph.ecog'],
        method="efron",
    )

    assert np.allclose(model.beta, r_answer)
