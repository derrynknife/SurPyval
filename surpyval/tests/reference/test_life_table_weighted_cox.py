"""The actuarial life table and the weighted Cox model's survival, checked
against R.

- ``life_table`` against ``KMsurv::lifetab`` on that function's own
  example, the weaning data of Klein and Moeschberger (2003, section
  5.4), every column of the table;
- the survival of a Cox model with sampling weights at chosen covariates
  and times against ``survfit(coxph(..., weights = w), newdata = ...)``,
  with Efron's and Breslow's ties, with and without late entry, on the
  forty weighted reference rows (``weighted_reference_data``). The
  coefficients are checked in ``test_weights.py``; this checks the
  baseline hazard built under the weights.

The values were printed by
``scripts/reference/reference_r_life_table_weighted_cox.R`` (R 4.6.2,
KMsurv 0.1.6, survival 3.8.12):

.. code-block:: r

    lifetab(tis, nsubs[1], nlost, nevent)
    f <- coxph(Surv(x, d) ~ z1 + z2, weights = w, ties = "efron")
    # Surv(tl, x, d) for late entry; ties = "breslow" too
    summary(survfit(f, newdata = newdata), times = times, extend = TRUE)
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data

NA = np.nan

# -- the life table -----------------------------------------------------------

EDGES = [0, 2, 3, 5, 7, 11, 17, 25, 37, 53]
NLOST = [2, 3, 6, 9, 7, 5, 3, 0, 0, 0]
NEVENT = [77, 71, 119, 75, 109, 148, 107, 74, 85, 27]

# lifetab's columns, in SurPyval's names
LIFETAB = {
    "at_risk_start": [927, 848, 774, 649, 565, 449, 296, 186, 112, 27],
    "censored": NLOST,
    "at_risk": [926, 846.5, 771, 644.5, 561.5, 446.5, 294.5, 186, 112, 27],
    "events": NEVENT,
    "surv": [
        1,
        0.916846652267819,
        0.83994634239066,
        0.71030481872725,
        0.627647159449448,
        0.505806482014025,
        0.338148342399074,
        0.215289691680225,
        0.129636803592394,
        0.0312517294374521,
    ],
    "pdf": [
        0.0415766738660907,
        0.076900309877159,
        0.0648207618317046,
        0.0413288296389013,
        0.0304601693588556,
        0.0279430232691585,
        0.0153573313398561,
        0.00713774067398597,
        0.00614906713468386,
        NA,
    ],
    "hazard": [
        0.0433802816901408,
        0.0875462392108508,
        0.0836261419536191,
        0.0617792421746293,
        0.0537475345167653,
        0.0662192393736018,
        0.0554979253112033,
        0.0413870246085011,
        0.0764388489208633,
        NA,
    ],
    "se_surv": [
        0,
        0.00907366472056602,
        0.0120583779713434,
        0.0149472153236314,
        0.0159669265132624,
        0.0165929246283761,
        0.0158121965929014,
        0.013826141880798,
        0.0113582686928699,
        0.0059118693680673,
    ],
    "se_pdf": [
        0.00453683236028301,
        0.00876836029616168,
        0.00554300634205374,
        0.00456951226220066,
        0.00273133262794079,
        0.00208980857532889,
        0.00138529695221306,
        0.000790374492944734,
        0.000630473185564179,
        NA,
    ],
    "se_hazard": [
        0.00493898812453533,
        0.0103798684201042,
        0.00763914340908689,
        0.00712002599004606,
        0.00511825018785585,
        0.00533469939625213,
        0.00523131777937741,
        0.00466044686077105,
        0.00656010460357464,
        NA,
    ],
}


def _weaning_table():
    # The grouped data, one row per interval and outcome at the interval's
    # start (the form the life_table docstring gives for lifetab's input)
    x, c, n = [], [], []
    for start, events, lost in zip(EDGES, NEVENT, NLOST):
        for flag, count in ((0, events), (1, lost)):
            if count:
                x.append(start)
                c.append(flag)
                n.append(count)
    return sp.life_table(x, c, n, intervals=EDGES)


def test_life_table_matches_lifetab():
    table = _weaning_table()
    assert len(table) == len(EDGES)
    np.testing.assert_array_equal(table["start"], EDGES)
    np.testing.assert_array_equal(table["end"], EDGES[1:] + [np.inf])
    assert (table["entered"] == 0).all()
    for column, expected in LIFETAB.items():
        np.testing.assert_allclose(
            table[column].to_numpy(float),
            expected,
            rtol=1e-8,
            atol=1e-15,
            err_msg=column,
        )


# -- the weighted Cox model's survival ---------------------------------------

TIMES = [0.5, 1.0, 2.0, 2.9, 3.2, 4.3, 6.0, 8.3, 12.0]
NEWDATA = [[0.0, 0.0], [1.0, 0.5], [0.0, -1.0]]

# survfit(...)$surv, one list per row of NEWDATA, at TIMES
SURVFIT = {
    ("efron", "none"): [
        [
            0.992019302201987,
            0.969612277641704,
            0.915170002429074,
            0.802027511078684,
            0.774095616675715,
            0.638914107775833,
            0.638914107775833,
            0.368543529703948,
            0.368543529703948,
        ],
        [
            0.962062749878009,
            0.861612550371335,
            0.651894883255982,
            0.344780531142813,
            0.290560196968034,
            0.115057196185799,
            0.115057196185799,
            0.00808233944046188,
            0.00808233944046188,
        ],
        [
            0.997093195186753,
            0.988851463300213,
            0.96830796702382,
            0.922978821212667,
            0.911168758743073,
            0.849800136303502,
            0.849800136303502,
            0.695831943799784,
            0.695831943799784,
        ],
    ],
    ("efron", "late"): [
        [
            0.987132035646012,
            0.964313899527865,
            0.910768719304054,
            0.799219258783128,
            0.771641717160132,
            0.637991933260817,
            0.637991933260817,
            0.369448859244701,
            0.369448859244701,
        ],
        [
            0.938361232236925,
            0.836522594188804,
            0.631835893172623,
            0.332563927107257,
            0.279874174260198,
            0.109954289190083,
            0.109954289190083,
            0.00751169919416603,
            0.00751169919416603,
        ],
        [
            0.995325360032771,
            0.986939481461654,
            0.966750964992707,
            0.922117710051396,
            0.910477277484629,
            0.849935161417237,
            0.849935161417237,
            0.697508121072856,
            0.697508121072856,
        ],
    ],
    ("breslow", "none"): [
        [
            0.992063669745312,
            0.969971337966732,
            0.915763704329286,
            0.804714203289092,
            0.776819468551774,
            0.64486225425636,
            0.64486225425636,
            0.37337902122343,
            0.37337902122343,
        ],
        [
            0.961924169036838,
            0.861965768610271,
            0.651344477220331,
            0.346969135699525,
            0.292176263887842,
            0.117959355842589,
            0.117959355842589,
            0.00823257001075412,
            0.00823257001075412,
        ],
        [
            0.99710432879422,
            0.988965251505212,
            0.968481778417636,
            0.9239728558928,
            0.912185322289053,
            0.852427345891924,
            0.852427345891924,
            0.698695859892313,
            0.698695859892313,
        ],
    ],
    ("breslow", "late"): [
        [
            0.987192567845334,
            0.96469503472045,
            0.911384986600014,
            0.801899954193545,
            0.774361131731947,
            0.643897884731466,
            0.643897884731466,
            0.374265755702555,
            0.374265755702555,
        ],
        [
            0.938083302931014,
            0.836752827288153,
            0.63121825777001,
            0.334637641638589,
            0.28139773026123,
            0.112722026629441,
            0.112722026629441,
            0.00764875052687634,
            0.00764875052687634,
        ],
        [
            0.995339743691992,
            0.987059302099566,
            0.966933744594855,
            0.923113202366395,
            0.911496977323895,
            0.852548081022669,
            0.852548081022669,
            0.700371563690362,
            0.700371563690362,
        ],
    ],
}
# coef(f) for the late-entry fits (the others are in test_weights.py)
LATE_COEF = {
    "efron": [1.0833651610119, 1.01671555418292],
    "breslow": [1.09358479898583, 1.0150586797622],
}


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("entry", ["none", "late"])
def test_weighted_cox_survival_matches_survfit(ties, entry):
    data = weighted_reference_data()
    late = {"tl": data["tl"]} if entry == "late" else {}
    model = sp.CoxPH.fit(
        data["x"], data["Z"], c=data["c"], w=data["w"], tie_method=ties, **late
    )
    if late:
        np.testing.assert_allclose(model.params, LATE_COEF[ties], rtol=1e-7)
    for Z, expected in zip(NEWDATA, SURVFIT[ties, entry]):
        np.testing.assert_allclose(
            np.ravel(model.sf(np.asarray(TIMES), np.asarray(Z))),
            expected,
            rtol=1e-6,
            err_msg=str(Z),
        )
